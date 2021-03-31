import json
import math
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import pandas as pd
from datatuner.classification.distractors import (get_distractors,
                                                  write_classification_data)
from datatuner.lm.special_token_generator import generate_from_json
from datatuner.lm.utils import fix_text_in_dir
from fire import Fire


def parse_mr(mr):
    params_str = mr
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
            scope = "key"
            j += 2
            continue

        j += 1

    assert len(keys) == len(values)
    return {"keys": keys, "values": values}


def preprocess(in_folder, out_folder, classification_dir):
    in_folder = Path(in_folder)
    splits = {"train-fixed.no-ol": "train", "devel-fixed.no-ol": "validation", "test-fixed": "test"}

    for split in splits:
        classification_data = []
        df = pd.read_csv(in_folder / (split + ".csv"))
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        data = df.to_dict(orient="records")
        original_data = deepcopy(data)

        new_data = OrderedDict()
        print(len(data))
        for item in data:
            key = item["mr"]
            if key in new_data:
                new_data[key].append(item)
            else:
                new_data[key] = [item]

        out_data = []
        for mr_key in new_data:

            for item in new_data[mr_key]:

                mr = item["mr"]
                parsed = parse_mr(mr)
                new_params = [f"<{key}> {key} = [ {value} ]" for key, value in zip(parsed["keys"], parsed["values"])]
                new_mr = " ; ".join(new_params)
                item["new_mr"] = new_mr
                out_data.append(item)

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
                    num_candidates=1,
                    max_per_operation=1,
                )

                classification_data.extend(classification_items)

        print(f"written for {split}")
        json.dump(out_data, open(out_folder / (splits[split] + ".json"), "w"), indent=2)
        classification_data = random.sample(classification_data, int(math.ceil(0.7 * len(classification_data))))
        write_classification_data(classification_data, classification_dir, splits[split].replace(".json", ""))

    generate_from_json(out_folder, out_folder / "special_tokens.txt", fields={"new_mr": "amr"})
    fix_text_in_dir(out_folder)


if __name__ == "__main__":
    Fire(preprocess)
