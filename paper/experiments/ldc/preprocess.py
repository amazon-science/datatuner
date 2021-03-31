import json
import random
from copy import deepcopy
from pathlib import Path

from datatuner.classification.distractors import (get_distractors,
                                                  write_classification_data)
from datatuner.lm.special_token_generator import generate_from_json
from datatuner.lm.utils import fix_text_in_dir
from datatuner.utils import bracket_contents
from fire import Fire

random.seed(42)


def get_entities(amr):
    options = bracket_contents(amr, opening="(", ending=")")
    options = [option.strip("() ") for option in options if option.count("(") == 1 and option.count("<") == 0]
    options = [option for option in options if option[0].isupper()]
    return options


def preprocess(in_folder, out_folder, classification_dir, num_candidates=10, max_per_operation=10):
    """Linearize the data already processed into surface texts and AMRs into our format"""

    splits = {"test": "test", "dev": "validation", "train": "train"}

    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    for split in splits:
        amrs = (in_folder / split / "nodes.pp.txt").read_text().split("\n")
        surfaces = (in_folder / split / "surface.pp.txt").read_text().split("\n")
        raw_amrs = (in_folder / ".." / "tmp_amr" / split / "graphs.txt").read_text().split("\n")
        items = [
            {"linearized_amr": amr, "answer_text": surface, "raw_amr": raw_amr}
            for amr, surface, raw_amr in zip(amrs, surfaces, raw_amrs)
            if amr and surface
        ]

        classification_data = []
        original_items = deepcopy(items)
        for item in items:

            entities = get_entities(item["linearized_amr"])

            swapping_candidates = [entities]
            cutting_candidates = [entities]

            rand_item = None
            while rand_item is None or rand_item == item:
                rand_item = random.choice(original_items)

            random_text = rand_item["answer_text"]

            distractors, classification_items = get_distractors(
                item["linearized_amr"],
                item["answer_text"],
                swapping_candidates,
                cutting_candidates,
                random_text,
                num_candidates=num_candidates,
                max_per_operation=max_per_operation,
            )
            classification_data.extend(classification_items)

            item["answer_text"] = distractors + [item["answer_text"]]

        json.dump(items, open(out_folder / (splits[split] + ".json"), "w"), indent=2)
        write_classification_data(classification_data, classification_dir, splits[split].replace(".json", ""))

    generate_from_json(out_folder, out_folder / "special_tokens.txt", fields={"linearized_amr": "amr"})
    fix_text_in_dir(out_folder)


if __name__ == "__main__":
    Fire(preprocess)
