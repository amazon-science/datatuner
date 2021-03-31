import json
import random
from copy import deepcopy
from pathlib import Path

import pandas as pd
from datatuner.classification.distractors import (get_distractors,
                                                  write_classification_data)
from datatuner.lm.special_token_generator import generate_from_json
from datatuner.lm.utils import fix_text_in_dir
from fire import Fire

random.seed(42)


def parse_mr(mr):
    i = mr.index("(")
    intro, params_str = mr[:i], mr[i + 1: -1]
    j = 0
    scope = "key"
    current_key = ""
    current_val = ""
    keys = []
    values = []
    while j < len(params_str):
        next_char = params_str[j]
        if scope == "key":
            if next_char == "[":
                scope = "value"
                keys.append(current_key)
                current_key = ""
            else:
                current_key += next_char

        elif scope == "value":
            if next_char == "]":
                scope = "between"
                values.append(current_val)
                current_val = ""
            else:
                current_val += next_char

        elif scope == "between":
            if next_char == " ":
                scope = "key"

        j += 1
    assert len(keys) == len(values)
    return {"keys": keys, "values": values, "intro": intro}


def preprocess(in_folder, out_folder, classification_dir):
    in_folder = Path(in_folder)
    out_folder = Path(out_folder)

    out_folder.mkdir(parents=True, exist_ok=True)

    splits = {"viggo-test.csv": "test.json", "viggo-train.csv": "train.json", "viggo-valid.csv": "validation.json"}
    for split in splits:
        df = pd.read_csv(in_folder / split)
        data = df.to_dict(orient="records")
        original_data = deepcopy(data)
        classification_data = []

        for item in data:
            mr = item["mr"]
            parsed = parse_mr(mr)
            new_params = [
                f"<{key}> {key.replace('_', ' ')}: [ {value} ]" for key, value in zip(parsed["keys"], parsed["values"])
            ]
            new_mr = f"<{parsed['intro']}> {parsed['intro'].replace('_', ' ')} ( {', '.join(new_params)}> )"

            item["new_mr"] = new_mr

            valid_values = [x for x in parsed["values"] if x]
            swapping_candidates = [valid_values]
            cutting_candidates = [valid_values]

            rand_item = None
            while rand_item is None or rand_item == item:
                rand_item = random.choice(original_data)
            random_text = rand_item["ref"]

            distractors, classification_items = get_distractors(
                new_mr,
                item["ref"],
                swapping_candidates,
                cutting_candidates,
                random_text,
                num_candidates=10,
                max_per_operation=10,
            )
            classification_data.extend(classification_items)

            item["ref"] = distractors + [item["ref"]]

        json.dump(data, open(out_folder / (splits[split]), "w"), indent=2)
        write_classification_data(classification_data, classification_dir, splits[split].replace(".json", ""))

    generate_from_json(out_folder, out_folder / "special_tokens.txt", fields={"new_mr": "amr"})
    fix_text_in_dir(out_folder)


if __name__ == "__main__":
    Fire(preprocess)
