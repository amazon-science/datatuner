import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from fire import Fire
from sklearn.metrics import classification_report, cohen_kappa_score

from datatuner.classification.consistency_classifier import dataset_fields
from mlxtend.evaluate import mcnemar, mcnemar_table

random.seed(43)


def dedup_consecutive_data(our_data, key):
    dedup_our_data = []
    for i, item in enumerate(our_data):
        if i > 0 and item[key] == our_data[i - 1][key]:
            continue
        else:
            dedup_our_data.append(item)

    return dedup_our_data


datasets = ["ldc", "viggo", "webnlg", "e2e"]
systems = ["systemNoFc", "systemFcPost", "sota", "human", "systemNoFcNoFs"]


def sample(A, k):
    try:
        return random.sample(A, k)
    except:
        return []


def prepare(data_folder, out_folder, task, fluency_samples=150, fidelity_samples=24):
    '''Select the specified number of samples for annotation from each system, sampling based on the classification
    of the heuristic semantic error rate checkers (where available) and the semantic fidelity classifiers'''
    data_folder = Path(data_folder)
    out_folder = Path(out_folder)

    for dataset in datasets:

        print(f"processing {dataset}")
        systems_data = {}

        text_key = dataset_fields[dataset]["text"]
        original_text_key = f"original_{text_key.strip()}"
        orig_data_key = dataset_fields[dataset]["original_data"]

        for system in systems:
            systems_data[system] = json.load(open(data_folder / dataset / f"{system}.json"))

        sota_data = systems_data["sota"]
        print(task)
        if task == "fluency":
            print(fluency_samples)
            sample_indices = random.sample(list(range(len(sota_data))), fluency_samples)
            print(sample_indices)
        elif task == "fidelity_annotations":
            print(fidelity_samples)
            sample_per_system = 1

            sampling_systems = ["human", "sota", "systemFcPost"]
            sample_indices = []
            while len(sample_indices) < fidelity_samples:

                for system in sampling_systems:
                    print(system)
                    if dataset == "ldc":
                        # For LDC there is no heuristic semantic error rate, only the semantic fidelity classifier
                        sfc_correct = [
                            i
                            for i, item in enumerate(systems_data[system])
                            if item["sfc_correct"] == 1 and i not in sample_indices
                        ]
                        sampled = sample(sfc_correct, sample_per_system)
                        print(f"sfc_correct: {len(sampled)}")
                        sample_indices.extend(sampled)

                        sfc_wrong = [
                            i
                            for i, item in enumerate(systems_data[system])
                            if item["sfc_correct"] == 0 and i not in sample_indices
                        ]
                        sampled = sample(sfc_wrong, sample_per_system)
                        print(f"sfc_wrong: {len(sampled)}")
                        sample_indices.extend(sampled)

                    else:
                        sfc_correct_ser_wrong = [
                            i
                            for i, item in enumerate(systems_data[system])
                            if item["sfc_correct"] == 1 and item["ser_correct"] == 0 and i not in sample_indices
                        ]
                        sampled = sample(sfc_correct_ser_wrong, sample_per_system)
                        sample_indices.extend(sampled)
                        print(f"sfc_correct_ser_wrong: {len(sampled)}")

                        sfc_wrong_ser_correct = [
                            i
                            for i, item in enumerate(systems_data[system])
                            if item["sfc_correct"] == 0 and item["ser_correct"] == 1 and i not in sample_indices
                        ]
                        sampled = sample(sfc_wrong_ser_correct, sample_per_system)
                        sample_indices.extend(sampled)
                        print(f"sfc_wrong_ser_correct: {len(sampled)}")

                        both_wrong = [
                            i
                            for i, item in enumerate(systems_data[system])
                            if item["sfc_correct"] == 0 and item["ser_correct"] == 0 and i not in sample_indices
                        ]
                        sampled = sample(both_wrong, sample_per_system)
                        sample_indices.extend(sampled)
                        print(f"both_wrong: {len(sampled)}")

                        both_correct = [
                            i
                            for i, item in enumerate(systems_data[system])
                            if item["sfc_correct"] == 1 and item["ser_correct"] == 1 and i not in sample_indices
                        ]
                        sampled = sample(both_correct, sample_per_system)
                        sample_indices.extend(sampled)
                        print(f"both_correct: {len(sampled)}")

            sample_indices = random.sample(sample_indices, fidelity_samples)
            assert len(sample_indices) == len(set(sample_indices))
            assert len(sample_indices) == fidelity_samples
        mturk_data = []

        for i in sample_indices:
            texts = []
            for system in systems_data:
                system_text = systems_data[system][i][text_key][0]
                original_text = systems_data[system][i][original_text_key]
                texts.append((system_text, system))

            random.shuffle(texts)

            def preprocess_text(t):
                if dataset == "ldc":
                    # The SOTA results for LDC are lowercased, so we lowercase also for consistent annotations
                    return t.lower()
                else:
                    return t

            data_i = {f"text{j + 1}": preprocess_text(texts[j][0]) for j in range(len(texts))}
            data_i.update({f"system{j + 1}": texts[j][1] for j in range(len(texts))})
            data_i["humantext"] = original_text
            data_i["index"] = i
            data_i["data"] = sota_data[i][orig_data_key]

            data_i["data"] = data_i["data"].replace("<", "{").replace(">", "}").replace(";", "<br>")
            data_i["data"] = data_i["data"].replace("|||", "<br>")
            data_i["data"] = data_i["data"].strip()

            mturk_data.append(data_i)

        mturk_df = pd.DataFrame(mturk_data)

        mturk_out_folder = out_folder / task
        print(f"writing files to {mturk_out_folder}")
        mturk_out_folder.mkdir(parents=True, exist_ok=True)
        mturk_df.to_csv(mturk_out_folder / f"mturk_{dataset}.csv", sep=",", index=False)


def add_closest_score(score, true_scores, ref_score):
    if score < 1 and score > 0:
        true_scores.append(ref_score)
    else:
        true_scores.append(score)


def compute_stat_sig(systems_data, measure):
    significance = defaultdict(list)
    for system in ["systemNoFcNoFs", "systemNoFc", "systemFcPost", "sota"]:
        for other_system in ["sota", "human"]:
            if system == other_system:
                continue
            sys_data = [x[measure] for x in systems_data[system]]
            other_sys_data = [x[measure] for x in systems_data[other_system]]
            true_data = [1] * len(sys_data)

            tb_b = mcnemar_table(
                y_target=np.array(true_data), y_model1=np.array(sys_data), y_model2=np.array(other_sys_data)
            )

            chi2, p_value = mcnemar(ary=tb_b, corrected=True)
            print(tb_b)
            print(f"mcnemar {system},{other_system}: chi2: {chi2}, p-value {p_value}")
            if p_value <= 0.05 and p_value >= 0:
                significance[system].append(other_system[0])
        significance[system] = ",".join(significance[system])
    return significance


def score(data_folder, out_folder, task, score_folder):
    data_folder = Path(data_folder)
    out_folder = Path(out_folder)
    datasets = ["ldc", "viggo", "webnlg", "e2e"]
    systems = ["systemNoFcNoFs", "systemNoFc", "systemFcPost", "sota", "human"]
    stats = {}
    first = []
    second = []
    for dataset in datasets:

        print(f"processing {dataset}")
        systems_data = {}

        for system in systems:
            systems_data[system] = json.load(open(data_folder / dataset / f"{system}.json"))

        print(f"dataset: {dataset}")
        all_scored = defaultdict(list)
        score_folder = Path(score_folder)
        score_file = score_folder / task / (f"{dataset}.csv")
        total_texts = 5
        try:
            df = pd.read_csv(score_file)
        except:
            print(f"{score_file} not available.")
            continue
        scores = df.to_dict(orient="records")
        try:
            input_df = pd.read_csv(out_folder / task / (f"mturk_{dataset}.csv"))
        except:
            print(f"ignoring {dataset}")
            continue
        input_data = input_df.to_dict(orient="records")

        if task == "fidelity_annotations":
            for item in scores:
                for i in range(total_texts):
                    text = item[f"Input.text{i + 1}"]
                    index = item["Input.index"]
                    accurate = f"Answer.text{i + 1}_accurate.text{i + 1}_accurate"
                    key = f"{index}_{text}"
                    try:
                        all_scored[key].append({"accurate": item[accurate]})
                    except:
                        import ipdb
                        ipdb.set_trace()

            fidelity_scores = []

            all_ser_scores = []
            all_sfc_scores = []
            true_scores_sfc = []
            true_scores_ser = []
            sfc_data = defaultdict(list)
            ser_data = defaultdict(list)

            for x in all_scored:
                try:
                    one = all_scored[x][0]["accurate"]
                    two = all_scored[x][1]["accurate"]
                    first.append(one)
                    second.append(two)
                except:
                    pass

            for item in input_data:
                for i in range(total_texts):
                    text_i = item[f"text{i + 1}"]
                    system = item[f"system{i + 1}"]
                    index = item["index"]
                    key = f"{index}_{text_i}"

                    if key in all_scored:
                        obj = systems_data[system][index]
                        score = np.mean([int(x["accurate"]) for x in all_scored[key]])
                        # these have to be reconciled if disagreeing: take ceil or floor

                        sample_type = f'{"A_D" if obj["sfc_correct"] else "E_D"}'
                        if dataset != "ldc":
                            sample_type += f',{"A_H" if obj["ser_correct"] else "E_H"}'

                        fidelity_scores.append(
                            {
                                "ind": index,
                                "system": system,
                                "value": math.ceil(score),
                                "sample_type": sample_type,
                                "text": text_i,
                                "data": item["data"],
                                "original_text": obj["original_" + dataset_fields[dataset]["text"].strip()],
                                "sfc_correct": obj["sfc_correct"],
                                "ser_correct": obj["ser_correct"] if "ser_correct" in obj else "",
                            }
                        )
                        # Reconciled cases are those where the expert annotators disagreed. They discussed these and
                        # reached the following agreements
                        reconciled = {
                            "Example 1": 0,
                            "Example 2": 1,
                        }
                        if text_i in reconciled:
                            true_scores_sfc.append(reconciled[text_i])
                            true_scores_ser.append(reconciled[text_i])
                        else:
                            add_closest_score(score, true_scores_sfc, obj["sfc_correct"])
                            if dataset != "ldc":
                                add_closest_score(score, true_scores_ser, obj["ser_correct"])

                        all_sfc_scores.append(obj["sfc_correct"])

                        sfc_data[system].append(obj["sfc_correct"])

                        if dataset != "ldc":
                            all_ser_scores.append(obj["ser_correct"])
                            ser_data[system].append(obj["ser_correct"])

            if dataset != "ldc":
                c_report = classification_report(true_scores_ser, all_ser_scores)
                stats[f"{dataset}_ser_report"] = classification_report(
                    true_scores_ser, all_ser_scores, output_dict=True
                )
                print("SER")
                print(c_report)

            c_report = classification_report(true_scores_sfc, all_sfc_scores)
            stats[f"{dataset}_sfc_report"] = classification_report(true_scores_sfc, all_sfc_scores, output_dict=True)
            print("SFC")
            print(c_report)

            mturk_df = pd.DataFrame(fidelity_scores)

            agg_stats = mturk_df.groupby(["system"]).agg(["mean", "count"])
            print(agg_stats)
            stats[f"{dataset}_score"] = agg_stats.to_dict()[("value", "mean")]
            stats[f"{dataset}_count"] = agg_stats.to_dict()[("value", "count")]
            print(mturk_df.groupby(["system", "sample_type"]).agg(["mean", "count"]))

            if dataset != "ldc":
                tb_b = mcnemar_table(
                    y_target=np.array(true_scores_sfc),
                    y_model1=np.array(all_sfc_scores),
                    y_model2=np.array(all_ser_scores),
                )
                print(tb_b)
                chi2, p = mcnemar(ary=tb_b, corrected=True)
                print(f"mcnemar chi2: {chi2}, p-value {p}")

            for measure in ["sfc_correct", "ser_correct"]:
                if measure == "ser_correct" and dataset == "ldc":
                    continue
                stats[f"{dataset}_significance_{measure}"] = compute_stat_sig(systems_data, system, measure)

        elif task == "fluency":

            for item in scores:
                for i in range(total_texts):
                    field = f"Input.text{i + 1}"
                    answer_field = f"Answer.fluency{i + 1}"
                    all_scored[item[field]].append(item[answer_field])

            for x in all_scored:
                all_scored[x] = {"average": np.mean(all_scored[x]), "count": len(all_scored[x])}

            fluency_scores = defaultdict(list)

            for item in input_data:
                for i in range(total_texts):
                    if item[f"text{i + 1}"] in all_scored:
                        score = all_scored[item[f"text{i + 1}"]]["average"]
                        system = item[f"system{i + 1}"]
                        fluency_scores[system].append(score)

            fluency_df_values = []
            for system in fluency_scores:
                fluency_df_values.extend(
                    [{"system": system, "value": fluency_scores[system][i]} for i in range(len(fluency_scores[system]))]
                )

            mturk_df = pd.DataFrame(fluency_df_values)
            agg_stats = mturk_df.groupby(["system"]).agg(["mean", "count", "median"])
            print(agg_stats)
            stats[dataset] = agg_stats.to_dict()[("value", "mean")]

            test_stats = sp.posthoc_wilcoxon(
                mturk_df, val_col="value", group_col="system", sort=True, zero_method="zsplit"
            )
            print(test_stats)
            significance = defaultdict(list)
            for system in ["systemNoFcNoFs", "systemNoFc", "systemFcPost", "sota"]:
                for other_system in ["sota", "human"]:
                    p_value = test_stats.loc[system, other_system]
                    if p_value <= 0.05 and p_value >= 0:
                        significance[system].append(other_system[0])
                significance[system] = ",".join(significance[system])
            stats[f"{dataset}_significance"] = significance

    print(cohen_kappa_score(first, second))
    json.dump(stats, open(data_folder / f"{task}.json", "w"), indent=2)


if __name__ == "__main__":
    Fire()
