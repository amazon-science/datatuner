import collections
import os
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import pandas as pd


def bracket_contents(string, level=None, opening="[", ending="]"):
    """Generate brackets' contents as strings"""
    stack = []
    result = []
    for i, c in enumerate(string):
        if c == opening:
            stack.append(i)
        elif c == ending and stack:
            start = stack.pop()
            result.append((len(stack), f"{opening}{string[start + 1: i]}{ending}"))

    if level is not None:
        result = [x for x in result if x[0] == level]

    return [x[1] for x in result]


def uniquify_in_order(seq):
    """Get unique sequence from given sequence while preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def str_part_matches_array(s, arr):
    return any(s in x for x in arr)


def str_start_matches_array(s, arr):
    return any(x.startswith(s) for x in arr)


def arr_part_matches_string(s, arr):
    """True if some item in the array arr is a substring of s"""
    return any(x in s for x in arr)


def ewm_mean(iterable, alpha=0.9):
    if len(iterable) > 0:
        df = pd.DataFrame({"B": iterable})
        av = df.ewm(alpha=alpha).mean().B.iloc[-1]

        return av


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


def newest_file(folder_path, pattern):
    folder_path = Path(folder_path)
    list_of_paths = folder_path.glob(pattern)
    latest_path = max(list_of_paths, key=lambda p: p.stat().st_ctime)
    return latest_path


def flatten(d, parent_key="", sep="-"):
    items = []
    for k, v in d.items():
        new_key = k + sep + parent_key if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_curr_time():
    return strftime("%Y-%m-%d_%H-%M-%S", gmtime())


def dedup_consecutive_data(our_data, key):
    dedup_our_data = []
    cache = {}
    for i, item in enumerate(our_data):
        if item[key].replace(" ", "") in cache:
            continue
        else:
            dedup_our_data.append(item)
            cache[item[key].replace(" ", "")] = True

    return dedup_our_data


def read_lines_from_file(file):
    file = Path(file)
    texts = file.read_text().split("\n")
    texts = [x for x in texts if x.strip()]
    return texts


def is_empty_or_absent_dir(dir_name):
    return not os.path.exists(dir_name) or not os.listdir(dir_name)
