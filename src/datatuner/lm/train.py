import json
import logging
import math
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from shutil import copyfile, rmtree

import mlflow
import torch
from datatuner.lm.data_loader import MODEL_INPUTS, get_data_loaders
from datatuner.lm.model_loader import get_model_directory, load_pretrained, load_training_args, read_special_tokens
from datatuner.lm.novograd import Novograd
from datatuner.lm.utils import average_distributed_scalar, load_task_config
from datatuner.utils import get_curr_time, is_empty_or_absent_dir
from fairseq.optim.adafactor import Adafactor
from ignite.contrib.handlers import LinearCyclicalScheduler, PiecewiseLinear, ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CONFIG_NAME, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__file__)


def train():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3."
    )
    parser.add_argument(
        "--logdir", type=str, default=None, help="If provided, the model will be output to this folder."
    )
    parser.add_argument("--dataset_cache", type=str, default="./dataset_cache", help="Path or url of the dataset cache")
    parser.add_argument("--use_mlflow", action="store_true", help="If true we enable mlflow")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")

    parser.add_argument(
        "--tracking_uri", type=str, default="http://localhost:5000", help="url for mlflow tracking server"
    )
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates for training")

    parser.add_argument("--experiment", type=str, help="experiment name for mlflow")

    parser.add_argument("--task_config", type=str, help="Path to the tokenization config file")
    parser.add_argument("--special_tokens_file", type=str, default=None, help="Path to the special tokens file")
    parser.add_argument(
        "--model_checkpoint", type=str, default="distilgpt2", help="Path, url or short name of the model"
    )
    parser.add_argument("--model_type", type=str, default=None, help="gpt or gpt2")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps"
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_epsilon", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--patience", type=int, default=1, help="patience parameter for early stopping")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_data", type=int, default=0, help="Number of data items (0 includes everything)")
    parser.add_argument(
        "--val_max_data", type=int, default=0, help="Number of validation data items (0 includes everything)"
    )
    parser.add_argument(
        "--eval_before_start", action="store_true", help="If true start with a first evaluation before training"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="If true, and the logdir is explictly passed, it will be overwritten",
    )
    parser.add_argument("--ul", action="store_true", help="If true use unlikelihood sampling")
    parser.add_argument("--freeze", action="store_true", help="If true freeze layers")
    parser.add_argument("--smoothing", type=float, default=0.0, help="label smoothing epsilon")
    parser.add_argument("--ignore_cache", action="store_true", help="If true ignore the dataset cache")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)"
    )
    parser.add_argument("--warmup-steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # custom training
    parser.add_argument("--sequence-tune-rate", type=float, default=0.5)
    parser.add_argument("--sequence-ngram-n", type=int, default=4)
    parser.add_argument(
        "--multitask", action="store_true", help="If true use multitask training with multiple choice loss"
    )
    parser.add_argument(
        "--retrain_base",
        type=str,
        default=None,
        help="JSON file with training parameters or MLflow run_id from which to get the parameters for retraining",
    )
    parser.add_argument(
        "--training_args_file",
        type=str,
        default=None,
        help="File with the training arguments generated by a previous run to use as parameters",
    )
    parser.add_argument("--scheduler", type=str, default="piecewiselinear", help="scheduler choice")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer choice")
    parser.add_argument(
        "--max_block_size", type=int, default=None, help="If set, data is truncated to fit this max size"
    )

    args = parser.parse_args()

    if args.retrain_base:
        try:
            logger.info(f"reading the arguments from {args.retrain_base}")
            model_training_args = json.load(open(args.retrain_base))
        except:
            model_training_args = load_training_args(args.retrain_base)

        passed_args = [x[2:] for x in sys.argv if x.startswith("--")]
        # this is set by pytorch
        passed_args.extend(["ignore_cache", "local_rank"])

        for key, value in model_training_args.items():
            # we only update an argument if it's not passed explicitly
            if key not in passed_args:
                if value:
                    args.__setattr__(key, value)
        logger.info(vars(args))

    if args.logdir is None:
        args.logdir = Path(f"runs/{get_curr_time()}")
    else:
        args.logdir = Path(args.logdir)
        if not is_empty_or_absent_dir(args.logdir) and not args.overwrite_output_dir:
            logger.error(f"Error: {args.logdir} is not empty and you did not pass --overwrite_output_dir as True")
            exit()
        else:
            if args.local_rank in [-1, 0]:
                logger.info(f"deleting the existing folder {args.logdir}")
                try:
                    rmtree(args.logdir)
                except:
                    pass

    logger.info(f"outputting model to {args.logdir}")
    try:

        def finalize():

            if args.local_rank not in [-1, 0,]:
                # Make sure only the first process in distributed training will download model & vocab
                torch.distributed.barrier()

            if args.local_rank in [-1, 0] and args.n_epochs > 0:
                try:
                    # On the main process: rename the last checkpoint
                    # (for easy re-loading with from_pretrained method)
                    os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(args.logdir, WEIGHTS_NAME))

                    if args.use_mlflow:
                        mlflow.log_artifact(args.logdir / WEIGHTS_NAME, "training")
                        logger.info("ending mlflow run")
                        logger.info(f"run_id: {mlflow.active_run().info.run_id}")
                        mlflow.end_run()

                        rmtree(args.logdir)

                except:
                    logger.info("No checkpoint to finalize the model. Deleting run")
                    # TODO: fix issue in mlflow trying to delete the experiment multiple times
                    mlflow.delete_run(mlflow.active_run().info.run_id)
                    rmtree(args.logdir)

                if args.local_rank == 0:
                    torch.distributed.barrier()

        args.logdir.mkdir(parents=True, exist_ok=True)
        TRAINING_ARGS_FILE = args.logdir / "model_training_args.json"
        args_dict = deepcopy(vars(args))
        args_dict["logdir"] = str(args_dict["logdir"])
        json.dump(args_dict, open(TRAINING_ARGS_FILE, "w"), indent=2)

        if args.use_mlflow:
            if args.local_rank in [-1, 0]:
                assert args.tracking_uri
                assert args.experiment
                mlflow.set_tracking_uri(args.tracking_uri)
                mlflow.set_experiment(args.experiment)
                mlflow.start_run()

                # Log parameters
                mlflow.log_params(vars(args))
                # Log training arguments into a file
                mlflow.log_artifact(TRAINING_ARGS_FILE, "training")

        # The validation maximum number of items shouldn't be more than the training (used during debugging)
        if args.val_max_data == 0 and args.max_data > 0:
            args.val_max_data = args.max_data

        # Logging is set to INFO (resp. WARN) for main (resp. auxiliary)
        # process. logger.info => log main process only, logger.warning => log all processes
        logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        # This is a logger.warning: it will be printed by all distributed processes
        logger.warning("Running process %d", args.local_rank)

        # Initialize distributed training if needed
        args.distributed = args.local_rank != -1

        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

        logger.info(f"Reading the task configuration: {args.task_config}")
        copyfile(args.task_config, args.logdir / "task_config.json")
        task_config = load_task_config(args.task_config)

        logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")

        model_directory, is_local = get_model_directory(args.model_checkpoint)

        model, tokenizer = load_pretrained(
            model_directory,
            model_type=args.model_type,
            smoothing=args.smoothing,
            multitask=args.multitask,
            special_tokens_file=args.special_tokens_file,
            task_config=task_config,
            dataset_path=args.dataset_path,
        )

        special_tokens = read_special_tokens(
            task_config=task_config,
            special_tokens_file=args.special_tokens_file,
            dataset_path=args.dataset_path
        )
        logger.info(f"adding {len(special_tokens)}")
        tokenizer.add_tokens(special_tokens)

        model.resize_token_embeddings(len(tokenizer))

        model.to(args.device)

        if args.freeze:
            transformer = list(model.children())[0]
            i = 0
            for param in transformer.parameters():
                param.requires_grad = False
                i += 1
                if i >= len(list(transformer.parameters())) // 2:
                    break

        if args.optimizer.lower() == "rmsprop":
            optimizer = RMSprop(model.parameters(), lr=args.lr)
        elif args.optimizer.lower() == "adam":
            optimizer = Adam(model.parameters(), lr=args.lr)
        elif args.optimizer.lower() == "adafactor":
            optimizer = Adafactor(model.parameters(), lr=args.lr, warmup_init=False)
        elif args.optimizer.lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=args.lr)
        elif args.optimizer.lower() == "novograd":
            optimizer = Novograd(model.parameters(), lr=args.lr)
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr)

        # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
        if args.fp16:
            from apex import amp  # Apex is only required if we use fp16 training

            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

        if args.distributed:
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

        logger.info("Prepare datasets")
        train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, task_config, tokenizer)

        def named_batch(batch, with_labels=True):
            """Helper function so that we get a dictionary with key as the input name and the value as the input value. 
            This makes it easier to pass parameters to the model by their name, without caring about the order
            """
            named_batch = {}
            # The components in the batch are ordered as in MODEL_INPUTS
            i = 0
            for input_name in MODEL_INPUTS:

                if not with_labels and "labels" in input_name:
                    continue

                key = input_name
                if not args.multitask:
                    if "mc_" in input_name:
                        continue
                    # the field is called `lm_labels` in the DoubleHeads and `labels` in single head model
                    if input_name == "lm_labels":
                        key = "labels"

                named_batch[key] = batch[i]
                i += 1
            return named_batch

        # Training function and trainer
        def update(engine, batch):
            model.train()

            n_batch = named_batch(tuple(input_tensor.to(args.device) for input_tensor in batch))

            outputs = model(**n_batch)

            lm_loss = outputs[0]
            if args.multitask:
                mc_loss = outputs[1]
            else:
                mc_loss = 0

            loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            if engine.state.iteration % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            return loss.item()

        trainer = Engine(update)

        # Evaluation function and evaluator (evaluator output is the input of the metrics)
        def inference(engine, batch):
            model.eval()
            with torch.no_grad():
                n_batch = named_batch(tuple(input_tensor.to(args.device) for input_tensor in batch))
                outputs = model(**{key: n_batch[key] for key in n_batch if "labels" not in key})
                lm_logits = outputs[0]
                lm_labels = n_batch["lm_labels"] if args.multitask else n_batch["labels"]

                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

                if args.multitask:
                    mc_logits = outputs[1]
                    mc_labels = n_batch["mc_labels"]

                    return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
                else:
                    return lm_logits_flat_shifted, lm_labels_flat_shifted

        evaluator = Engine(inference)

        def checkpointing_score_function(engine):
            """"""
            val_metric = engine.state.metrics["average_ppl"]
            logger.info(val_metric)
            return -val_metric

        def score_function(engine):
            """"""
            val_ppl = engine.state.metrics["average_ppl"]
            return -val_ppl

        # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
        if args.n_epochs < 1:
            trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
        if args.eval_before_start:
            trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))
        # Attach mlflow logger
        # trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))

        # Make sure distributed data samplers split the dataset nicely between the distributed processes
        if args.distributed:
            trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
            evaluator.add_event_handler(
                Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch)
            )

        if args.scheduler.lower() == "piecewiselinear":
            # Linearly decrease the learning rate from lr to zero
            scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
        elif args.scheduler.lower() == "linearcyclical":
            scheduler = LinearCyclicalScheduler(optimizer, "lr", args.lr / 10, args.lr, len(train_loader))
        elif args.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, args.n_epochs * len(train_loader), 1e-4)
        elif args.warmup_steps > 0:
            t_total = len(train_loader) // args.gradient_accumulation_steps * args.n_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        # Prepare metrics - note how we compute distributed metrics
        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
        if args.multitask:
            metrics = {
                "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
                "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1])),
            }
            metrics.update(
                {
                    "average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args),
                }
            )
        else:
            metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean"))}
            metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})

        metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])

        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        # On the main process: add progress bar, tensorboard, checkpoints and save model,
        # configuration and tokenizer before we start to train

        if args.local_rank in [-1, 0]:
            pbar = ProgressBar(persist=True)
            pbar.attach(trainer, metric_names=["loss"])
            evaluator.add_event_handler(
                Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics))
            )

            checkpoint_handler = ModelCheckpoint(
                args.logdir,
                filename_prefix="checkpoint",
                score_function=checkpointing_score_function,
                create_dir=True,
                n_saved=2,
            )

            evaluator.add_event_handler(
                Events.COMPLETED, checkpoint_handler, {"mymodel": getattr(model, "module", model)}
            )  # "getattr" takes care of distributed encapsulation

            getattr(model, "module", model).config.to_json_file(os.path.join(args.logdir, CONFIG_NAME))
            tokenizer.save_pretrained(args.logdir)

            early_handler = EarlyStopping(patience=args.patience, score_function=score_function, trainer=trainer)
            evaluator.add_event_handler(Events.COMPLETED, early_handler)

        if args.use_mlflow and args.local_rank in [-1, 0]:

            class MLflowTracker:
                def __init__(self):
                    self.iteration = 1

                def eval_metric_logger(self, engine):
                    mlflow.log_metric("last_epoch", self.iteration)
                    for metric in engine.state.metrics:
                        mlflow.log_metric(f"eval_{metric}", engine.state.metrics[metric], step=self.iteration)
                    self.iteration += 1

                def train_metric_logger(self, engine):
                    for metric in engine.state.metrics:
                        mlflow.log_metric(f"train_{metric}", engine.state.metrics[metric], step=engine.state.epoch)

                def finish_experiment(self, engine):
                    mlflow.log_metric("finished", True)

                def start_experiment(self, engine):
                    # log the initial artifacts in the dir
                    mlflow.log_artifacts(args.logdir, "training")
                    mlflow.log_metric("finished", False)

            mlflow_tracker = MLflowTracker()
            trainer.add_event_handler(Events.STARTED, mlflow_tracker.start_experiment)
            # Log the train and validation metrics
            trainer.add_event_handler(Events.EPOCH_COMPLETED, mlflow_tracker.train_metric_logger)
            evaluator.add_event_handler(Events.COMPLETED, mlflow_tracker.eval_metric_logger)
            # Log the model
            trainer.add_event_handler(Events.COMPLETED, mlflow_tracker.finish_experiment)

        # Run the training
        trainer.run(train_loader, max_epochs=args.n_epochs)
    except KeyboardInterrupt:
        finalize()

    logger.info("training about to finish")
    finalize()
    logger.info("finalized training")


if __name__ == "__main__":
    train()
