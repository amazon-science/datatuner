import json
import logging
import os
from collections import defaultdict

import torch
from datatuner.lm.converters import converters
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__file__)

PAD_TOKEN = "<pad>"
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

MASKED_OUTPUT = -1


def build_input_from_segments(
        data_point,
        tokenizer,
        task_config,
        with_eos=True,
        mask_lm_labels=False,
        last_learnt_field=None,
        candidate_val=None,
        max_block_size=None,
):
    """Build the input from the data"""
    instance = {}
    sequence, token_types, lm_labels = [], [], []
    curr_span_type = 0
    # TODO: change this to be the max of the current tokenizer by name, not min of all maxes
    max_tokenizer_size = min(tokenizer.max_model_input_sizes.values())
    if max_block_size is not None:
        max_tokenizer_size = min(max_block_size, max_tokenizer_size)

    for item in task_config["data_shape"]:
        if item["type"] == "special":
            x = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item["id"]))
            curr_span_type = x[0]
            tokens = x
        elif item["type"] == "special_id":
            tokens = item["id"]
        elif item["type"] == "text":
            # if we are using the DoubleHeads setting, we might have the input as a list of texts
            # the candidate_val contains the tokens of the item which we consider now
            if data_point[item["id"]] and type(data_point[item["id"]][0]) == list:
                tokens = candidate_val
            else:
                tokens = data_point[item["id"]]
        else:
            raise Exception("Invalid item type in the data shape")

        sequence += tokens

        current_token_types = [curr_span_type] * len(tokens)
        if "token_typing" not in task_config or task_config["token_typing"] != "coarse_grained":
            # if we have special tokens within the tokens, we adjust the token_type_ids so that anything after
            # a special token has the token_type as the id of that special token.
            for t_i, token in enumerate(tokens):
                if token in tokenizer.added_tokens_decoder:
                    curr_span_type = token
                current_token_types[t_i] = curr_span_type

        token_types += current_token_types
        lm_labels += tokens if item["learn"] else ([MASKED_OUTPUT] * len(tokens))

    if with_eos:
        eos = tokenizer.convert_tokens_to_ids(["<eos>"])
        sequence += eos
        token_types += [curr_span_type]
        lm_labels += eos

    sequence = sequence[:max_tokenizer_size]
    token_types = token_types[:max_tokenizer_size]
    lm_labels = lm_labels[:max_tokenizer_size]

    assert len(sequence) == len(token_types)
    assert len(token_types) == len(lm_labels)

    instance["input_ids"] = sequence
    instance["token_type_ids"] = token_types

    if mask_lm_labels:
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
    else:
        instance["lm_labels"] = lm_labels

    return instance, sequence


def get_inputs(item, device, tokenizer, task_config):
    """Get the input_ids and the token_type_ids from the item dictionary"""
    instance, _ = build_input_from_segments(item, tokenizer, task_config, with_eos=False)
    input_ids = torch.tensor(instance["input_ids"], device=device).unsqueeze(0)
    token_type_ids = torch.tensor(instance["token_type_ids"], device=device).unsqueeze(0)
    return input_ids, token_type_ids


def pad_dataset(dataset, padding=0):
    """Pad the dataset. This could be optimized by defining a
    Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        if name in dataset:
            dataset[name] = [
                x + [padding if name != "lm_labels" else MASKED_OUTPUT] * (max_l - len(x)) for x in dataset[name]
            ]
    return dataset


def get_data_loaders(args, task_config, tokenizer):
    """ Prepare the dataset for training and evaluation """
    datasets_raw = {}
    logger.info("Loading training data")

    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
        args.ignore_cache = False

    for split in ["validation", "train"]:
        logger.info(f"Loading {split} data")
        datasets_raw[split] = get_dataset(
            tokenizer,
            args.dataset_cache,
            task_config,
            args.dataset_path,
            split,
            args.max_data if split == "train" else args.val_max_data,
            args.ignore_cache,
            args.max_block_size,
        )

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "validation": defaultdict(list)}

    for dataset_name, dataset in datasets_raw.items():
        # get the last learnt field
        last_learnt_field = [x["id"] for x in task_config["data_shape"][::-1] if x["learn"] and x["type"] == "text"][0]

        if args.multitask:
            assert type(dataset[0][last_learnt_field]) == list
            num_candidates = len(dataset[0][last_learnt_field])
        else:
            num_candidates = 1

        if args.num_candidates > 0 and dataset_name in ["train", "validation"]:
            num_candidates = min(args.num_candidates, num_candidates)

        for data_point in dataset:
            if type(data_point[last_learnt_field]) == str:
                data_point[last_learnt_field] = [data_point[last_learnt_field]]

            for j, candidate_val in enumerate(data_point[last_learnt_field][-num_candidates:]):
                # the last item in the array is the ground truth. For other distractor items, we mask the LM labels
                mask_lm_labels = bool(j != num_candidates - 1)
                instance, _ = build_input_from_segments(
                    data_point,
                    tokenizer,
                    task_config,
                    mask_lm_labels=mask_lm_labels,
                    last_learnt_field=last_learnt_field,
                    candidate_val=candidate_val,
                    max_block_size=args.max_block_size,
                )
                if args.multitask:
                    # this is an indicator for the last input token, used in the Double Head model
                    instance["mc_token_ids"] = len(instance["input_ids"]) - 1

                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)

            datasets[dataset_name]["n_candidates"] = num_candidates

            # the ground truth is the last item in the array; previous items are distractors
            if args.multitask:
                datasets[dataset_name]["mc_labels"].append(num_candidates - 1)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "validation": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(PAD_TOKEN))
        for input_name in MODEL_INPUTS:
            if input_name in dataset:
                tensor = torch.tensor(dataset[input_name])
                if input_name != "mc_labels":
                    # adjust the shape as we might have more than one candidate in the case of DoubleHeads
                    try:
                        tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
                    except:
                        import ipdb

                        ipdb.set_trace()
                tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = (
        TensorDataset(*tensor_datasets["train"]),
        TensorDataset(*tensor_datasets["validation"]),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed)
    )
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("validation dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    return train_loader, valid_loader, train_sampler, valid_sampler


def get_dataset_from_file(tokenizer, filename, task_config, max_data, max_block_size=None):
    """Read dataset from file"""

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    with open(filename, "r") as f:
        data = json.load(f)

    # get the max size supported by the tokenizer model
    # {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024, 'distilgpt2': 1024}
    max_tokenizer_size = min(tokenizer.max_model_input_sizes.values())
    if max_block_size is not None:
        max_tokenizer_size = min(max_block_size, max_tokenizer_size)

    if max_data > 0:
        data = data[:max_data]

    ignored_sequences = 0

    output_data = []
    logger.info(f"initial data: {len(data)}")

    text_fields = [x for x in task_config["data_shape"] if x["type"] == "text"]
    len_special_fields = 0
    for x in task_config["data_shape"]:
        if x["type"] == "special":
            len_special_fields += len(tokenizer.tokenize(x["id"]))
        elif x["type"] == "special_id":
            len_special_fields += len(x["id"])

    failed_conversions = 0
    for inst_i, inst in enumerate(tqdm(data)):

        # check the inclusion criteria
        if "include" in task_config:
            include = True
            for field, value in task_config["include"].items():
                if field in inst and inst[field] != value:
                    include = False
                    break
            if not include:
                continue

        item = {}

        total_seq_len = 0
        stop = False
        for field in text_fields:
            field_v = inst[field["id"]]

            if "converter" in field:
                try:
                    func = converters[field["converter"]]
                except:
                    logger.error(f"Unable to get the converter {field['converter']}")
                    raise

                field_v = func(field_v)
                if field_v is None:
                    stop = True
                    break

            item[field["id"]] = tokenize(field_v)

            total_seq_len += len(item[field["id"]])

        if stop:
            failed_conversions += 1
            continue

        if "extra_fields" in task_config:
            for field in task_config["extra_fields"]:
                item[field] = inst[field]

        # 1 is for eos token
        if total_seq_len + len_special_fields + 1 > max_tokenizer_size:
            for field in text_fields:
                item[field["id"]] = item[field["id"]][: max_tokenizer_size - 100]
            print(f"warning: this input is longer than the sequence length so we truncated: {inst_i}")
            ignored_sequences += 1
            # continue
        output_data.append(item)

    logger.info(
        "%d / %d sequences ignored due to positional embedding restriction or max block size restriction"
        % (ignored_sequences, len(data))
    )
    logger.info("%d / %d removed due to failed conversions" % (failed_conversions, len(data)))
    logger.info(f"preprocessed data: {len(output_data)}")
    return output_data


def get_dataset(tokenizer, dataset_cache, task_config, path, split, max_data, ignore_cache, max_block_size):
    """Load the dataset for the given split"""
    dataset_cache = (
        f"{dataset_cache}_{split}_{task_config['name']}_{max_data}_{type(tokenizer).__name__}"
    )  # Do avoid using GPT cache for GPT-2 and vice-versa

    if not ignore_cache and dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        data = torch.load(dataset_cache)
        return data

    dataset_path = f"{path}/{split}.json"
    data = get_dataset_from_file(tokenizer, dataset_path, task_config, max_data, max_block_size=max_block_size)

    if dataset_cache:
        torch.save(data, dataset_cache)

    logger.info("Dataset cached at %s", dataset_cache)

    return data
