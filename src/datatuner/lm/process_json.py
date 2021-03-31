import json
from pathlib import Path

from fire import Fire


def split_list(data, n):
    vals_per_item = [0 for _ in range(n)]
    for ix, _ in enumerate(data):
        vals_per_item[ix % n] += 1
    ix = 0
    new_list = []
    subset = []
    for _, d in enumerate(data):
        if len(subset) < vals_per_item[ix]:
            subset.append(d)
        if len(subset) == vals_per_item[ix]:
            new_list.append(subset)
            ix += 1
            subset = []
    return new_list


def split(filename, out_folder, splits):
    j = json.load(open(filename))
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    chunks = split_list(j, splits)
    for i, chunk in enumerate(chunks):
        json.dump(chunk, open(out_folder / f"chunk_{i}.json", "w"), indent=2)


def combine(base_folder_name, splits):
    output_data = []
    for i in range(splits):
        folder = f"{base_folder_name}/chunks/chunk_{i}"
        folder = Path(folder)
        output_data.extend(json.load(open(folder / "generated.json")))

    base_folder_name = Path(base_folder_name)
    base_folder_name.mkdir(parents=True, exist_ok=True)
    json.dump(output_data, open(base_folder_name / "generated.json", "w"), indent=2)


if __name__ == "__main__":
    Fire()
