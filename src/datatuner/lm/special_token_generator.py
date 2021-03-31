import itertools
import json
from pathlib import Path

from datatuner.utils import bracket_contents
from fire import Fire
from tqdm import tqdm


def get_custom_tags(s):
    """Get tags starting with a token and ending with another in the string"""
    return bracket_contents(s, opening="<", ending=">")


fn_map = {
    "question_sig": [get_custom_tags],
    "amr": [get_custom_tags],
}


def generate_from_item(item, fields, all_tokens):
    for field_name in fields:
        if field_name in item:
            tokens = list(itertools.chain(*[fn(item[field_name]) for fn in fn_map[fields[field_name]]]))
            all_tokens.update(tokens)


def generate_from_json(data_folder, outfile, fields={"mrl": "mrl"}):
    """Generate the special tokens from the given folder with files train.json, validation.json, and test.json
    The used field is defined by the key in the `fields` dictionary and the method used is defined based
    on that field.
    """

    data_folder = Path(data_folder)
    all_tokens = set()

    for split in ["test", "train", "validation"]:
        try:
            data = json.load(open(data_folder / (split + ".json"), "r"))
            for item in data:
                generate_from_item(item, fields, all_tokens)
        except:
            print(f"file absent: {split}")

    Path(outfile).write_text("\n".join(all_tokens))


def generate_from_jsonl(data_file, outfile, fields={"mrl": "mrl"}, max_items=0):
    """Generate the special tokens from the given jsonl file.
    The used field is defined by the key in the `fields` dictionary and the method used is defined based
    on that field.
    """

    all_tokens = set()
    i = 0
    with open(data_file, "r") as f:
        for line in tqdm(f):
            item = json.loads(line.rstrip())
            generate_from_item(item, fields, all_tokens)
            i += 1
            if max_items > 0 and i >= max_items:
                break
    Path(outfile).write_text("\n".join(all_tokens))


if __name__ == "__main__":
    Fire()
