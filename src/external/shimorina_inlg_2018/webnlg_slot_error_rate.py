import os
import pandas as pd
import re
import sys

from external.shimorina_webnlg_baseline.benchmark_reader import Benchmark
from external.shimorina_webnlg_baseline.webnlg_baseline_input import select_files
from nltk.tokenize import wordpunct_tokenize
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def substring_match(x, values):
    return any(x in v for v in values)


verbose = False


def calculate_ser(mr, pred):
    values = clean_mr(mr)
    total_n_slots = len(values)
    missing = 0
    hallucinated = 0
    for value in values:
        if value not in pred.lower():
            if verbose:
                print("\n")
                print("Missing:", value)
                print(mr)
                print(value)
                print(pred)
            missing += 1
    # delete s and o that are present in MR
    # account for the case where the item is "texas" and the values have an entry: "abilene, texas". This is not hallucinated.
    all_subj_obj_not_pres = [
        item for item in ENTITIES if not substring_match(item, values)
    ]
    # all_subj_obj_not_pres = [item for item in ENTITIES if item not in values]

    for entity in all_subj_obj_not_pres:
        if entity in pred.lower().split():
            hallucinated += 1
            if  verbose:
                print("\n")
                print("Hallucination:")
                print(mr)
                print(entity)
                print(pred)
    # print('COUNTS: Missing', missing, 'Hallucinated', hallucinated, 'Denominator', total_n_slots)
    ser = (missing + hallucinated) / total_n_slots
    return ser


def clean_mr(mr):
    # (19255)_1994_VK8 | density | 2.0(gramPerCubicCentimetres) | | |
    # extract all subjects and objects and clean them
    subj_obj = []
    triples = mr.strip().split("|||")  # the last one is empty
    triples = [triple for triple in triples if triple]  # delete empty triples
    for triple in triples:
        s, p, o = triple.split(" | ")
        s = s.lower().replace("_", " ")
        o = o.lower().replace("_", " ")
        # separate punct signs from text
        s = " ".join(re.split(r"(\W)", s))
        o = " ".join(re.split(r"(\W)", o))
        # Drop quotes
        s = s.replace('"', "")
        o = o.replace('"', "")
        # delete white spaces
        subj_obj.append(" ".join(s.split()))
        subj_obj.append(" ".join(o.split()))
    return subj_obj


def get_all_subj_obj():
    # read all the webnlg corpus
    # extract all subjects and objects
    base_path = "/paper/tmp/webnlg/data/v1.4/en/"
    path_train = base_path + "train"
    path_dev = base_path + "dev"
    path_test = base_path + "test"
    b = Benchmark()
    files_train = select_files(path_train)
    files_dev = select_files(path_dev)
    files_test = select_files(path_test)
    b.fill_benchmark(files_train + files_dev + files_test)
    subjs, objs = b.subjects_objects()
    # clean subj and obj
    subjs_cleaned = []
    for subj in list(subjs):
        subjs_cleaned.append(clean(subj))
    objs_cleaned = []
    for obj in list(objs):
        objs_cleaned.append(clean(obj))
    return subjs_cleaned, objs_cleaned


def clean(entity):
    entity = entity.lower().replace("_", " ")
    # separate punct signs from text
    entity = " ".join(re.split(r"(\W)", entity))
    entity = " ".join(entity.split())  # delete whitespaces
    return entity


def get_all_entities_in_corpus():
    # get all cleaned s and o from the whole corpus
    all_subj_cleaned, all_obj_cleaned = get_all_subj_obj()
    entities = list(set(all_subj_cleaned + all_obj_cleaned))
    # delete all numbers from entities
    for i, entity in enumerate(entities):
        try:
            float(entity.replace(" ", ""))
            del entities[i]
        except ValueError:
            pass
    return entities


ENTITIES = get_all_entities_in_corpus()


def compute_ser(datafile, outfile, mr_field, text_field):
    df = pd.read_json(datafile, orient="records")

    df["ser"] = df.apply(
        lambda x: calculate_ser(
            x[mr_field], " ".join(wordpunct_tokenize(x[text_field][0]))
        ),
        axis=1,
    )

    df["ser_correct"] = df["ser"].apply(lambda x: 0 if x > 0 else 1)

    results = {}
    results["mean_ser"] = round(df["ser"].mean(), 4)
    results["percent_correct_ser"] = round(len(df[df["ser"] == 0]) / len(df) * 100, 4)
    print(json.dumps(results, indent=2))

    data_dict = df.to_dict(orient="records")
    json.dump(data_dict, open(outfile, "w"), indent=2)

    return results
