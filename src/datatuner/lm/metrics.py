import copy
import difflib
import json
import logging
import os
from collections import OrderedDict
from itertools import groupby
from pathlib import Path
from subprocess import PIPE, Popen
from tempfile import mkdtemp

import mlflow
import numpy as np
from datatuner.ops.mlflow import get_artifact
from datatuner.utils import flatten
from fire import Fire

logger = logging.getLogger(__file__)

THIS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

E2E_METRICS_FOLDER = THIS_DIR / "../../../paper/tmp/e2e-metrics"
PYTHON_BIN = "/home/ec2-user/miniconda3/envs/finetune/bin/python"


def get_str_diff(case_a, case_b):
    """Get the string difference between two strings"""
    return ("").join([li[-1] for li in difflib.ndiff(case_a, case_b) if li[0] != " " and li[-1] not in [" ", "'", ","]])


def almostmatch(original, current, all_outputs, final):
    """Computes match average while allowing a difference in articles. The metric is computed for the given
    keys across the list of dictionaries `all_outputs`
    """
    lst = [
        int(x[original] == x[current] or get_str_diff(x[original], x[current]) in ["the", "a", "an"])
        for x in all_outputs
    ]
    return {"value": np.mean(lst), "count": len(all_outputs)}


def match(original, current, all_outputs, final):
    """Computes exact match average across the values of the given keys in the list of dictionaries `all_outputs`"""

    def postprocess(x):
        return x[current][0] if type(x[current]) == list else x[current]

    lst = [int(str(x[original]).lower() == str(postprocess(x).lower())) for x in all_outputs]
    return {"value": np.mean(lst), "count": len(all_outputs)}


def bleu(original, current, all_outputs, final, case_insensitive=True, all_keys=None):
    """Computes bleu score for the values of the given keys in the list of dictionaries `all_outputs`"""
    if len(all_outputs) == 0:
        return {"value": 0, "count": 0}

    from sacrebleu import corpus_bleu

    def process(s):
        return s.lower() if case_insensitive else s

    # group by all the other keys
    all_outputs = copy.deepcopy(all_outputs)
    if all_keys is None:
        keys = all_outputs[0].keys()
    else:
        keys = all_keys
        print(keys)

    other_keys = list(set([key for key in keys if key not in [original, current]]))

    group = {}
    max_refs = 1
    for item in all_outputs:
        # other inputs concatenated
        search_key = str([item[x] for x in other_keys if x in item])
        if type(item[current]) == list:
            item[current] = item[current][0]

        current_val = process(item[current])
        original_val = process(item[original])

        if search_key in group:
            group[search_key]["references"].append(original_val)
            group[search_key]["prediction"] = current_val
            if len(group[search_key]["references"]) > max_refs:
                max_refs = len(group[search_key]["references"])
        else:
            group[search_key] = {"references": [original_val], "prediction": current_val}

    all_predictions = []
    all_references = [[] for i in range(max_refs)]

    for item in group.values():
        all_predictions.append(item["prediction"])
        for i in range(max_refs):
            try:
                all_references[i].append(item["references"][i])
            except:
                all_references[i].append("")

    e2e_metrics = {}
    if final:
        e2e_metrics = get_e2e_metrics(all_predictions, all_references)
    e2e_metrics.update({"value": corpus_bleu(all_predictions, all_references).score, "count": len(all_predictions)})
    return e2e_metrics


def get_e2e_metrics(all_predictions, all_references):
    tempdir = Path(mkdtemp())
    human = tempdir / "human_refs.txt"
    system = tempdir / "system.txt"
    with open(human, "w") as h:
        with open(system, "w") as s:
            for i, x in enumerate(all_predictions):
                s.write(x + "\n")
                for j in range(len(all_references)):
                    v = all_references[j][i]
                    if v.strip():
                        h.write(v + "\n")
                h.write("\n")
    print(E2E_METRICS_FOLDER / "measure_scores.py")
    p = Popen(
        [
            PYTHON_BIN,
            E2E_METRICS_FOLDER / "measure_scores.py",
            f"{human}",
            f"{system}",
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    output, err = p.communicate()
    stats = output.decode("utf-8").split("\n")
    stats = [x for x in stats if x not in ["", "==============", "SCORES:"]]
    stats_dict = {}
    for item in stats:
        key, value = item.split(": ")
        value = float(value)
        if key in ["BLEU", "METEOR", "ROUGE_L"]:
            value *= 100
        if key == "BLEU":
            key = "e2e_BLEU"
        stats_dict[key] = value

    return stats_dict


def round_dict(d):
    """Round values in a dictionary"""
    items = [(k, round(v * 100.0, 2)) for k, v in d.items()]
    return dict(sorted(items, key=lambda t: t[1]))


def group_by_field(all_outputs, field):
    """group a list of dictionaries by the given field value"""
    all_outputs.sort(key=lambda k: k[field])
    return groupby(all_outputs, key=lambda k: k[field])


def compute_metric(metric, original, current, all_outputs, final):
    """compute the result for the given metric"""
    try:
        # get the function name from the "metrics.py" file
        func = metrics[metric]
        return func(original, current, all_outputs, final)
    except:
        logger.info(f"Unable to compute the metric {metric}")
        raise


def aggregate_metrics(all_outputs, fields, metrics_fields, output_to_metrics, final=False):
    """Combine the stats array into a value for a given metric"""

    out_metrics = {}
    for field in fields:
        original = "original_" + field
        current = field + " " * len("original_")

        out_metrics[field] = {}
        for metric in output_to_metrics[field]:
            # first we compute the aggregated metric
            out_metrics[field][metric] = {}
            out_metrics[field][metric]["total"] = compute_metric(metric, original, current, all_outputs, final)
            logger.info(f"{field},{metric},{out_metrics[field][metric]['total']}")
            # We then split the metrics computation per metric field.
            # We do this by taking all the inputs so far. Although this involves repetition, this is more generalizable
            # to cases where the metric is corpus-wide (e.g. BLEU).
            for metric_field in metrics_fields:
                grouped_items = group_by_field(all_outputs, metric_field)
                out_metrics[field][metric][metric_field] = []
                for metric_field_value, field_outputs in grouped_items:
                    out_metrics[field][metric][metric_field].append(
                        (metric_field_value, compute_metric(metric, original, current, list(field_outputs), False))
                    )
                out_metrics[field][metric][metric_field].sort(key=lambda k: k[1]["value"])
                out_metrics[field][metric][metric_field] = OrderedDict(out_metrics[field][metric][metric_field])
    return out_metrics


def compute_metrics_from_run(field, filename=None, run_id=None, eval_folder=None, metrics=None):
    if run_id is not None:
        assert eval_folder is not None
        filename = get_artifact(run_id, f"evaluation/{eval_folder}/generated.json")

    filename = Path(filename)
    all_outputs = json.load(open(filename, "r"))
    output_to_metrics = {}
    if metrics is None:
        metrics = ["bleu"]
    output_to_metrics[field] = metrics
    stats = aggregate_metrics(all_outputs, [field], [], output_to_metrics, final=True)
    print(json.dumps(stats, indent=2))
    out_folder = filename.parent
    (out_folder / f"stats_{filename.stem}.json").write_text(json.dumps(stats, indent=2))

    if run_id is not None:
        mlflow.start_run(run_id)
        flattened_stats = flatten(stats)
        flattened_stats = {k: flattened_stats[k] for k in flattened_stats if k.count("-") <= 3}

        mlflow.log_metrics(flattened_stats)


metrics = {"match": match, "bleu": bleu, "almostmatch": almostmatch}

if __name__ == "__main__":
    Fire(compute_metrics_from_run)
