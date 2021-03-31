import json
from collections import Counter
from pathlib import Path

import textstat
from fire import Fire

from datatuner.classification.consistency_classifier import dataset_fields


def count_words(text, casing="any"):
    """Counts word frequency using Counter from collections"""
    if casing == "any":
        text = text.lower()

    skips = [".", ", ", ":", ";", "'", '"']
    for ch in skips:
        text = text.replace(ch, "")
    words = text.split(" ")
    if casing == "lower":
        words = [x for x in words if x and x[0].islower()]
    elif casing == "upper":
        words = [x for x in words if x and x[0].isupper()]
    word_counts = Counter(words)
    return word_counts


def calculate_stats(data_folder):
    """Calculate stat of test.json file in a folder"""
    data_folder = Path(data_folder)
    for dataset in dataset_fields:
        print(f"loading {dataset}")
        field = dataset_fields[dataset]["text"].strip()
        sentences = []
        for item in json.load(open(data_folder / dataset / "test.json")):
            sentences.append(item[field][-1] if type(item[field]) == list else item[field])

        text = " ".join(sentences)
        lex_count = textstat.lexicon_count(text)
        print(lex_count)
        unique_words = count_words(text)
        print(f"all unique {len(unique_words)}")

        lower_unique_words = count_words(text, casing="lower")
        print(f"lowercase unique {len(lower_unique_words)}")

        upper_unique_words = count_words(text, casing="upper")
        print(f"uppercase unique {len(upper_unique_words)}")

        print(f"ratio {len(upper_unique_words) / len(unique_words)}")

        text_standard = textstat.text_standard(text, float_output=True)
        print(f"text_standard: {text_standard}")

        dale_chall_readability_score = textstat.dale_chall_readability_score(text)
        print(f"dale_chall_readability_score: {dale_chall_readability_score}")

        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        print(f"flesch_kincaid_grade: {flesch_kincaid_grade}")


if __name__ == "__main__":
    Fire(calculate_stats)
