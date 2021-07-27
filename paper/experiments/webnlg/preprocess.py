import json
import os
import random
from pathlib import Path
from xml.etree import ElementTree as ET

from fire import Fire

from datatuner.classification.distractors import (get_distractors,
                                                  write_classification_data)
from datatuner.lm.special_token_generator import generate_from_json
from datatuner.lm.utils import fix_text_in_dir
from webnlg_utils import camel_case_split, cleanup

random.seed(42)

seen_categories = [
    "Airport.xml",
    "Astronaut.xml",
    "Building.xml",
    "City.xml",
    "ComicsCharacter.xml",
    "Food.xml",
    "Monument.xml",
    "SportsTeam.xml",
    "University.xml",
    "WrittenWork.xml",
]

unseen_categories = ["Athlete.xml", "Artist.xml", "MeanOfTransportation.xml", "CelestialBody.xml", "Politician.xml"]


class Triple:

    def __init__(self, s, p, o):
        self.s = s #subject
        self.o = o #object
        self.p = p #predicate


def process_tripleset(s):
    """Format the triples set in our target format"""
    s = cleanup(s)
    key = "<mtriple>"
    s = s[len(key) : -len(key) - 1]
    subject, predicate, obj = s.split("|")
    subject, obj = cleanup(subject), cleanup(obj)
    predicate = camel_case_split(predicate)
    return {
        "text": f"<subject> {subject} <predicate> {predicate} <object> {obj}",
        "dict": {"subject": subject, "predicate": predicate, "object": obj},
    }


def get_nearby_text(entries, e):
    i = 1
    random_sentence = None
    while random_sentence is None:
        for j in range(2):
            try:
                if j == 0:
                    entry = entries[e + i]
                else:
                    entry = entries[e - i]

                random_sentence = entry.findall("lex")[0].find("text").text
                if random_sentence:
                    break

            except:
                pass
        i += 1
    return random_sentence


def parse(in_file, classification_data, num_candidates=5, max_per_operation=2):
    """Parse the given file and update `classification_data` with the parsed data"""

    tree = ET.parse(in_file)
    root = tree.getroot()

    entries = list(root.find("entries"))
    items = []
    for e, entry in enumerate(entries):

        tripletsets = list(entry.find("modifiedtripleset").findall("mtriple")) + list(
            entry.find("modifiedtripleset").findall("otriple")
        )
        tripletsets = [process_tripleset(x) for x in tripletsets]

        modifiedtripleset = [x["text"] for x in tripletsets]
        modifiedtripleset.sort()

        mtripleset = entry.find("modifiedtripleset")
        modtripleset = []
        raw_tripleset = ""
        for mtriple in mtripleset:
            e1, pred, e2 = mtriple.text.split(" | ")
            raw_tripleset += mtriple.text + " ||| "

            modtripleset.append(Triple(cleanup(e1), pred, cleanup(e2)))

        all_lex = entry.findall("lex")
        for lex in all_lex:

            sortedtripleset = ""
            for sent in lex.find("sortedtripleset").findall("sentence"):
                for x in sent.findall("striple"):
                    sortedtripleset += process_tripleset(x)["text"] + ", "

            references = cleanup(lex.find("references"))
            template = cleanup(lex.find("template"))

            try:
                text = lex.find("text").text
                if not text:
                    print("empty text")
                    text = ""
                    continue
            except:
                print("exception text")
                text = ""
                continue

            try:
                template = lex.find("template").text
                if not template:
                    print("empty template")
                    template = ""
                    continue
            except:
                print("exception template")
                template = ""
                continue

            # preprocess distractors
            subjects = [x["dict"]["subject"] for x in tripletsets]
            objects = [x["dict"]["object"] for x in tripletsets]
            predicates = [x["dict"]["predicate"] for x in tripletsets]

            swapping_candidates = [subjects + objects]
            cutting_candidates = [subjects + objects]

            random_text = get_nearby_text(entries, e)

            tripletset_str = " ; ".join(modifiedtripleset)

            distractors, classification_items = get_distractors(
                tripletset_str,
                text,
                swapping_candidates,
                cutting_candidates,
                random_text,
                num_candidates=num_candidates,
                max_per_operation=max_per_operation,
            )

            classification_data.extend(classification_items)

            item = {
                "raw_modifiedtripleset": raw_tripleset,
                "modifiedtripleset": " ; ".join(modifiedtripleset),
                "sortedtripleset": sortedtripleset,
                "references": references,
                "template": template,
                "text": distractors + [text],
                "num_triples": Path(in_file).parent.name,
                "category": Path(in_file).name,
                "category_type": "seen" if Path(in_file).name in seen_categories else "unseen",
            }
            items.append(item)

    return items


def run_parser(set_path, classification_data):
    """Get the entry set for the give path """
    entryset = []
    dirtriples = filter(lambda item: not str(item).startswith("."), os.listdir(set_path))
    dirtriples = sorted(list(dirtriples))
    for dirtriple in dirtriples:
        fcategories = filter(lambda item: not str(item).startswith("."), os.listdir(os.path.join(set_path, dirtriple)))
        fcategories = sorted(list(fcategories))
        for fcategory in fcategories:
            entryset.extend(list(parse(os.path.join(set_path, dirtriple, fcategory), classification_data)))

    return entryset


def run(
    in_folder="./tmp/webnlg/data/v1.4/en/",
    out_folder="datatuner/data/webnlg",
    classification_dir="datatuner/data/webnlg_consistency",
    output_classification_data=True,
):
    """Run the webnlg data formatting task"""
    out_folder = Path(out_folder)
    in_folder = Path(in_folder)
    classification_dir = Path(classification_dir)
    out_folder.mkdir(exist_ok=True, parents=True)
    classification_dir.mkdir(exist_ok=True, parents=True)
    splits = {"train": "train", "dev": "validation", "test": "test"}

    for split in splits:
        data_path = in_folder / split
        classification_data = []
        entryset = run_parser(data_path, classification_data)
        json.dump(entryset, open(out_folder / (splits[split] + ".json"), "w"), indent=2)
        if output_classification_data:
            write_classification_data(classification_data, classification_dir, splits[split])

    generate_from_json(out_folder, out_folder / "special_tokens.txt", fields={"modifiedtripleset": "amr"})
    fix_text_in_dir(out_folder)


if __name__ == "__main__":
    Fire(run)
