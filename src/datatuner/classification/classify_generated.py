import json
import logging
import os
from collections import Counter
from pathlib import Path
from shutil import copyfile
from subprocess import run

import numpy as np
import pandas as pd
from datatuner.classification.consistency_classifier import dataset_fields
from datatuner.lm.metrics import bleu
from datatuner.lm.runs_data import runs_data
from fire import Fire
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)
THIS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


def generate(
        in_file,
        dataset=None,
        out_folder=None,
        model_folder=None,
        model_type="roberta",
        model_name="roberta-large",
        python_location="/home/ec2-user/miniconda3/envs/finetune/bin/python",
        classifier_script="/home/ec2-user/DataTuner/src/datatuner/classification/run_classifier.py",
        correct_label="accurate",
        text_key=None,
        data_key=None,
):
    """Classify data generated from the language model"""

    in_file = Path(in_file)
    data_folder = in_file.parent
    data = json.load(open(in_file))
    basic_texts = []
    if text_key is None:
        text_key = dataset_fields[dataset]["text"]
    else:
        text_key = text_key.strip() + (" " * len("original_"))

    if data_key is None:
        data_key = dataset_fields[dataset]["data"]

    # Prepare data for the classifier
    for item in data:
        if type(item[text_key]) == list:
            for x in item[text_key]:
                basic_texts.append(
                    {"text": x.replace("\n", " "), "data": item[data_key].replace("\n", ";"), "label": correct_label}
                )
        elif type(item[data_key] == list):
            for x in item[data_key]:
                basic_texts.append(
                    {"text": item[text_key].replace("\n", " "), "data": x.replace("\n", ";"), "label": correct_label}
                )

    df = pd.DataFrame(basic_texts)

    df.to_csv(data_folder / "test.tsv", sep="|", index=False, columns=["label", "data", "text"])

    if model_folder is None:
        model_folder = f"/home/ec2-user/{dataset}_consistency_roberta-large_lower"
    model_folder = Path(model_folder)

    # Run the classifier command
    command = (
        f"{python_location} {classifier_script} --task_name mnli --data_dir {data_folder} --stats_dir {data_folder} "
        f" --model_name {model_name} --output_dir {model_folder} --model_type {model_type}  --do_eval"
        f" --overwrite_cache  --per_gpu_eval_batch_size 32 --do_lower_case"
    )

    print(command)
    run(command, shell=True)

    rerank_and_eval(
        in_file,
        dataset,
        model_folder=model_folder,
        out_folder=out_folder,
        correct_label=correct_label,
        text_key=text_key,
        data_key=data_key,
    )


def get_stats(data, dataset):
    """Get stats about the dataset"""
    if dataset == "webnlg":
        return json.dumps(
            {
                "num_triples": Counter(x["num_triples"] for x in data),
                "category": Counter(x["category"] for x in data),
                "category_type": Counter(x["category_type"] for x in data),
            },
            indent=2,
        )
    else:
        return ""


def rerank_and_eval(
        in_file,
        dataset,
        model_folder=None,
        out_folder=None,
        nbest=100,
        correct_label="accurate",
        text_key=None,
        data_key=None,
):
    """Compute the metrics based on the generated and classified data before and after reranking"""

    in_file = Path(in_file)
    if model_folder is None:
        model_folder = f"/home/ec2-user/{dataset}_consistency_roberta-large_lower"

    model_folder = Path(model_folder)
    if text_key is None:
        text_key = dataset_fields[dataset]["text"]
    if data_key is None:
        data_key = dataset_fields[dataset]["data"]
    original_keys = [data_key]
    data = json.load(open(in_file))
    if out_folder is None:
        out_folder = in_file.parent
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # This file is produced by the `generate` function above. It will have a dictionary.
    # `preds` is mapped to the list of labels predicted
    # `preds_prob` is mapped to the list of probabilities corresponding to the labels
    results = json.load(open(out_folder / "results.json", "r"))

    # Labels in the order we want to prioritize (correct first, then less severe errors)
    sorted_labels = ["accurate", "value_error", "repetition", "omission", "hallucination", "pronoun_error"]

    k = 0
    assert len(data) > 0

    for item in tqdm(data):
        cand_len = len(item[text_key])
        indices = list(range(cand_len))[:nbest]
        item["pred_prob"] = results["preds_prob"][k: k + cand_len][:nbest]
        item["pred"] = results["preds"][k: k + cand_len][:nbest]

        current_labels = [sorted_labels.index(x) for x in item["pred"]]
        reranked = [
            x
            for x in sorted(
                list(zip(current_labels, item["pred_prob"], indices, item[text_key][:nbest])),
                key=lambda x: (x[0], x[2]),
            )
        ]
        item["reranked"] = [x[3] for x in reranked]
        item["reranked_pred_prob"] = [x[1] for x in reranked]
        item["reranked_pred"] = [sorted_labels[x[0]] for x in reranked]

        k += cand_len

    correct_data = [x for x in data if x["pred"] and x["pred"][0] == correct_label and x["pred_prob"][0]]
    wrong_data = [x for x in data if (not x["pred"]) or x["pred"][0] != correct_label]

    print("Evaluating")

    out_stats = ""

    try:
        out_stats += f"all data: {get_stats(data, dataset)}\n"

        original_key = f"original_{text_key.strip()}"

        # Get stats before reranking
        out_stats += f"correct data: {get_stats(correct_data, dataset)}\n"
        out_stats += f"wrong data: {get_stats(wrong_data, dataset)}\n"

        out_stats += f"data, text:             {bleu(original_key, text_key, data, True, case_insensitive=True, all_keys=original_keys)}\n"
        bleu_correct = bleu(original_key, text_key, correct_data, False, case_insensitive=True, all_keys=original_keys)
        out_stats += f"correct_data, text:     {bleu_correct}\n"
        bleu_wrong = bleu(original_key, text_key, wrong_data, False, case_insensitive=True, all_keys=original_keys)
        out_stats += f"wrong_data, text:       {bleu_wrong}\n"
        out_stats += f"percent correct: {len(correct_data) / len(data) * 100}\n"

        r, p = stats.pointbiserialr(
            [0] * bleu_correct["count"] + [1] * bleu_wrong["count"],
            [bleu_correct["value"]] * bleu_correct["count"] + [bleu_wrong["value"]] * bleu_wrong["count"]
        )
        out_stats += f"r: {r}, p-value: {p}\n"
    except:
        print("Not computing stats for before reranking")

    # Get stats after reranking
    correct_data = [x for x in data if x["reranked_pred"] and x["reranked_pred"][0] == correct_label]
    wrong_data = [x for x in data if (not x["reranked_pred"]) or x["reranked_pred"][0] != correct_label]

    try:
        out_stats += f"correct data: {get_stats(correct_data, dataset)}\n"
        out_stats += f"wrong data: {get_stats(wrong_data, dataset)}\n"

        out_stats += (
            f"data, reranked:         "
            f'{bleu(original_key, "reranked", data, True, case_insensitive=True, all_keys=original_keys)}\n'
        )
        bleu_correct = bleu(
            original_key, "reranked", correct_data, False, case_insensitive=True, all_keys=original_keys
        )
        out_stats += f"correct_data, reranked: {bleu_correct}\n"
        bleu_wrong = bleu(original_key, "reranked", wrong_data, False, case_insensitive=True, all_keys=original_keys)
        out_stats += f"wrong_data, reranked:   {bleu_wrong}\n"
        out_stats += f"percent correct: {len(correct_data) / len(data) * 100}\n"

        r, p = stats.pointbiserialr(
            [0] * bleu_correct["count"] + [1] * bleu_wrong["count"],
            [bleu_correct["value"]] * bleu_correct["count"] + [bleu_wrong["value"]] * bleu_wrong["count"],
        )

        out_stats += f"r: {r}, p-value: {p}\n"

    except:
        print("Not computing stats for after reranking")

    json.dump(data, open(out_folder / "classified.json", "w"), indent=2)
    json.dump(correct_data, open(out_folder / "classified_correct.json", "w"), indent=2)
    json.dump(wrong_data, open(out_folder / "classified_wrong.json", "w"), indent=2)
    print(out_stats)
    (out_folder / "stats.txt").write_text(out_stats)
    for item in data:
        item[text_key] = item["reranked"]
        item["pred_prob"] = item["reranked_pred_prob"]
        item["pred"] = item["reranked_pred"]
        del item["reranked"]
        del item["reranked_pred_prob"]
        del item["reranked_pred"]
    json.dump(data, open(out_folder / "reranked.json", "w"), indent=2)


def eval_all(
        classify=False,
        ignore_existing=True,
        systems=["systemNoFc", "systemNoFcNoFs", "systemFcPost"],
):
    for system in systems:
        for dataset in ["viggo", "ldc", "webnlg", "e2e"]:

            print(f"dataset: {dataset}")
            run = runs_data[dataset]
            print(f"run: {run}")
            if system not in run:
                continue
            directory = (
                f"/home/ec2-user/mlflow/{run['id']}/{run[system]['run_id']}"
                f"/artifacts/evaluation/{run[system]['eval_folder']}"
            )
            directory = Path(directory)
            classified_exists = (directory / "classified.json").exists()
            if classified_exists and ignore_existing:
                print(f"{dataset} exists")
                continue

            copyfile(
                f"/home/ec2-user/DataTuner/data/{dataset}_consistency/labels.txt",
                f"{directory}/labels.txt",
            )
            in_file = f"{directory}/generated.json"
            if classify or not classified_exists:
                generate(in_file, dataset)
            else:
                rerank_and_eval(in_file, dataset)


systems = ["systemFcPost", "systemNoFc", "systemNoFcNoFs"]
datasets = ["ldc", "webnlg", "e2e", "viggo"]


def get_semantic_stats(data, folder, system, dataset):
    sfc_correct = [item["sfc_correct"] for item in data]

    results = {"sfc_correct": np.mean(sfc_correct)}
    if dataset != "ldc":
        ser_correct = [item["ser_correct"] for item in data]
        ser = [item["ser"] for item in data]
        results.update(
            {
                "ser_correct": np.mean(ser_correct),
                "ser": np.mean(ser),
                "both_correct": np.mean(np.prod([sfc_correct, ser_correct], axis=0)),
                "at_least_one_correct": np.mean(np.logical_or(sfc_correct, ser_correct)),
                "sfc_correct_ser_wrong": np.mean(
                    [int(sfc_correct[i] == 1 and ser_correct[i] == 0) for i in range(len(data))]
                ),
                "sfc_wrong_ser_correct": np.mean(
                    [int(sfc_correct[i] == 0 and ser_correct[i] == 1) for i in range(len(data))]
                ),
            }
        )

    print(results)
    out_file = folder / (f"results_{system}.json")
    print(f"written to {out_file}")
    json.dump(results, open(out_file, "w"), indent=2)
    return data


if __name__ == "__main__":
    Fire()
