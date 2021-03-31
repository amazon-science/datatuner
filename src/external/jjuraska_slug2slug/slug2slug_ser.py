from external.jjuraska_slug2slug.slot_aligner.data_analysis import score_slot_realizations
from tempfile import mkdtemp
import pandas as pd
from pathlib import Path
from fire import Fire
import json
from datatuner.classification.consistency_classifier import dataset_fields
import numpy as np


def compute_ser(datafile, scored_file, mr_field, text_field):
    dataset = "viggo" if "viggo" in str(datafile) else "e2e"
    if mr_field is None:
        mr_field = dataset_fields[dataset]["original_data"]
    if text_field is None:
        text_field = dataset_fields[dataset]["text"]

    data = json.load(open(datafile))
    if dataset == "viggo":
        subfolder = "video_game"
    elif dataset == "e2e":
        subfolder = "rest_e2e"
    tempdir = Path(mkdtemp()) / subfolder
    tempdir.mkdir(parents=True, exist_ok=True)
    new_items = []
    for item in data:
        new_item = {}
        new_item[mr_field] = item[mr_field]
        text = item[text_field]
        if type(text) == list:
            text = text[-1]

        new_item[text_field] = text
        new_items.append(new_item)
    df = pd.DataFrame(new_items)

    out_file = tempdir / "test.csv"
    df.to_csv(out_file, index=False)

    score_slot_realizations(tempdir, "test.csv")
    err_df = pd.read_csv(tempdir / ("test [errors].csv"))

    assert len(err_df) == len(df)
    err_data = err_df.to_dict(orient="records")
    percent_correct_list = []
    for err_item, item in zip(err_data, data):

        item["errors"] = err_item["errors"]
        if (
            type(err_item["incorrect slots"]) == float
            and "nan" in str(err_item["incorrect slots"]).lower()
        ):
            err_item["incorrect slots"] = "?"

        else:
            item["incorrect_slots"] = (
                err_item["incorrect slots"] if err_item["errors"] > 0 else ""
            )

        item["ser_correct"] = int(item["errors"] == 0)

        item["ser"] = item["errors"] / err_item["mr"].count("[")
        

        percent_correct_list.append(item["ser_correct"])

    datafile = Path(datafile)
    print(f"written to {scored_file}")
    json.dump(data, open(scored_file, "w"), indent=2)


if __name__ == "__main__":
    # python ser.py data/e2e_dataset/test.json --dataset e2e
    Fire(compute_ser)
