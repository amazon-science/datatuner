#!/usr/bin/env python3
# -"- encoding: utf-8 -"-
# the script is Python2/3 compatible

from __future__ import print_function
from __future__ import unicode_literals

from argparse import ArgumentParser
import pandas as pd
import re
import codecs
import json
import sys
import os

from external.ufal_dsg_tgen.data import DA

REALIZATIONS = {
    "area": {
        "city centre": [
            r"(?:city|town) cent(?:re|er)",
            r"cent(?:re|er) of (?:the )?(?:city|town)",
            r"in the cent(?:re|er)",
        ],
        "riverside": [
            r"riverside",
            r"(?:near|by|at|close to|along|on|off|beside) the river",
        ],
    },
    "eat_type": {
        "coffee shop": [r"coffee[- ]+shop", r"caf[eé]", r"coffee",],
        "pub": [r"pub",],
        "restaurant": [r"restaurant",],
    },
    "family_friendly": {
        "no": [
            r"(?:isn't|not|non|no)[ -]+(?:\w+ ){0,2}(?:child|children|family|kids|kid)[ -]+(?:friendly|orien(?:ta)?ted)",
            r"(?:child|children|family|kids|kid)[ -]+unfriendly",
            r"adults?[ -]+only",
            r"only for adults",
            r"(?:no|not) (?:kids|children|famil(?:y|ies))",
            r"(?:not|no)(?: good| suitable| friendly| orien(?:ta)?ted| open(?:ed))? (?:at|for|to|with)(?: the)? (?:kids|children|family|families|all age)",
            r"(?:kids?|child(?:ren)?|famil(?:y|ies)) (?:are|is)(?:n't| not) (?:welcome|allowed|accepted)",
            r"(?:does not|doesn't) (?:welcome|allow|accept) (?:\w+ ){0,2}(?:kids?|child(?:ren)?|famil(?:y|ies)|all age)",
            r"adult (?:establishment|venue|place|establish)",
        ],
        "yes": [
            r"for (?:kids|children|family|families)",
            r"family place",
            r"place to bring the(?: whole)? family",
            r"(?:friendly|suitable|good|orien(?:ta)?ted|open(?:ed)) (?:at|with|to|for)(?: the)(?:kids?|child(?:ren)?|famil(?:y|ies)?|all age)",
            r"(?:child|children|family|kids|kid)[ -]+(?:friendly|orien(?:ta)?ted)",
            r"(?:kids?|child(?:ren)?|famil(?:y|ies)) (?:are|is) (?:welcome|allowed|accepted)",
            r"(?:welcomes?|allows?|accepts?) (?:\w+ ){0,2}(?:kids?|child(?:ren)?|famil(?:y|ies)|all age)",
        ],
    },
    "food": {
        "Chinese": [r"Chinese", r"Chines"],
        "English": [r"English", r"British"],
        "Fast food": [r"Fast food"],
        "French": [r"French"],
        "Indian": [r"Indian"],
        "Italian": [r"Italian"],
        "Japanese": [r"Japanese"],
    },
    "name": [
        r"Alimentum",
        r"Aromi",
        r"Bibimbap House",
        r"Blue Spice",
        r"Browns Cambridge",
        r"Clowns",
        r"Cocum",
        r"Cotto",
        r"Fitzbillies",
        r"Giraffe",
        r"Green Man",
        r"Loch Fyne",
        r"Midsummer House",
        r"Strada",
        r"Taste of Cambridge",
        r"The Cambridge Blue",
        r"The Cricketers",
        r"The Dumpling Tree",
        r"The Eagle",
        r"The Golden Curry",
        r"The Golden Palace",
        r"The Mill",
        r"The Olive Grove",
        r"The Phoenix",
        r"The Plough",
        r"The Punter",
        r"The Rice Boat",
        r"The Twenty Two",
        r"The Vaults",
        r"The Waterman",
        r"The Wrestlers",
        r"Travellers Rest Beefeater",
        r"Wildwood",
        r"Zizzi",
    ],
    "near": [
        r"All Bar One",
        r"Avalon",
        r"Burger King",
        r"Café Adriatic",
        r"Café Brazil",
        r"Café Rouge",
        r"Café Sicilia",
        r"Clare Hall",
        r"Crowne Plaza Hotel",
        r"Express by Holiday Inn",
        r"Rainbow Vegetarian Café",
        r"Raja Indian Cuisine",
        r"Ranch",
        r"The Bakers",
        r"The Portland Arms",
        r"The Rice Boat",
        r"The Six Bells",
        r"The Sorrento",
        r"Yippee Noodle Bar",
    ],
    "price_range": {
        "cheap": [
            r"(?:inexpensive|cheap)(?:ly)?",
            r"low[- ]+price[ds]?",
            r"affordabl[ey]",
            r"prices?(?: range)?(?: \w+){0,3} low",
        ],
        "less than £20": [
            r"(?:inexpensive|cheap)(?:ly)?",
            r"affordabl[ey]",
            r"(?:less than|under) £? *20",
            r"moderately priced",
            r"low[- ]+price[ds]?",
            r"prices?(?: range)?(?: \w+){0,3} low",
        ],
        "more than £30": [
            r"(?:more than|over) £? *30",
            r"high[- ]+price[ds]?",
            r"expensive",
            r"not cheap",
            r"prices?(?: range)?(?: \w+){0,3} high",
        ],
        "high": [
            r"high[- ]+price[ds]?",
            r"expensive",
            r"prices?(?: range)?(?: \w+){0,3} high",
        ],
        "moderate": [
            r"(?:moderate|reasonable|ok|average)(?:ly)?[- ]+price[ds]?",
            r"not cheap",
            r"affordable",
            r"mid[- ]+(?:range[- ]+)price[ds]?",
            r"prices?(?: range)?(?: \w+){0,3} (?:ok|average|moderate|reasonable)",
        ],
        "£20-25": [
            r"£? *20 *(?:[-–]*|to) *25",
            r"(?:moderate|reasonable|ok|average)(?:ly)?[- ]+price[ds]?",
            r"prices?(?: range)?(?: \w+){0,3} (?:ok|average|moderate|reasonable)",
            r"affordable",
        ],
    },
    "rating": {
        "1 out of 5": [
            r"(?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?(?:low|bad|poor)(?:(?: \w+){0,3} (?:1|one)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            r"(?:low|bad|poor|(?:not|doesn't|isn't)(?: \w+){0,2} (:?good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "3 out of 5": [
            r"(?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?average(?:(?: \w+){0,3} (?:3|three)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            r"(?:average|(?<!very )(?:good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "5 out of 5": [
            r"(?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?high(?:(?: \w+){0,3} (?:5|five)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            r"(?:high|excellent|very good|great)(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "high": [
            r"(?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?high",
            r"(?:high|excellent|very good|great|well)(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
        "average": [
            r"(?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?average",
            r"(?:average|(?<!very )(?:good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
        "low": [
            r"(?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?(?:low|bad|poor)",
            r"(?:low|bad|poor|(?:not|doesn't|isn't)(?: \w+){0,2} (?:well|good))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
    },
}


def compile_patterns(patterns):
    """Compile a list of patterns into one big option regex. Note that all of them will match whole words only."""
    # pad intent patterns with \b (word boundary), unless they contain '^'/'$' (start/end)
    return re.compile(
        "|".join(
            [
                (
                    (r"\b" if not pat.startswith("^") else "")
                    + pat
                    + (r"\b" if not pat.endswith("$") else "")
                )
                for pat in patterns
            ]
        ),
        re.I | re.UNICODE,
    )


# store "proper" capitalization of the values
CAPITALIZE = {}
# compile realization patterns
for slot in REALIZATIONS.keys():
    if isinstance(REALIZATIONS[slot], list):
        CAPITALIZE[slot] = {val.lower(): val for val in REALIZATIONS[slot]}
        REALIZATIONS[slot] = compile_patterns(REALIZATIONS[slot])
    else:
        CAPITALIZE[slot] = {val.lower(): val for val in REALIZATIONS[slot].keys()}
        for value in REALIZATIONS[slot].keys():
            REALIZATIONS[slot][value] = compile_patterns(REALIZATIONS[slot][value])


class Match(object):
    """Realization pattern match in the system output"""

    def __init__(self, slot, value, regex_match):
        self.slot = slot
        self.value = value
        self._start = regex_match.start()
        self._end = regex_match.end()

    def is_same_string(self, other):
        return self._start == other._start and self._end == other._end

    def is_substring(self, other):
        return (self._start > other._start and self._end <= other._end) or (
            self._start >= other._start and self._end < other._end
        )

    def __eq__(self, other):
        return (
            self.slot == other.slot
            and self.value == other.value
            and self.is_same_string(other)
        )

    def __str__(self):
        return "Match[%s=%s:%d-%d]" % (self.slot, self.value, self._start, self._end)

    def __repr__(self):
        return str(self)


def reclassify_mr(ref, gold_mr=DA()):
    """Classify the MR given a text. Can use a gold-standard MR to make the classification more
    precise (in case of ambiguity, goes with the gold-standard value). Returns a dict-based MR format
    for the system output MR and the gold-standard MR."""
    # convert MR to dict for comparing & checking against
    mr_dict = {}
    for dai in gold_mr.dais:
        mr_dict[dai.slot] = mr_dict.get(dai.slot, {})
        val = CAPITALIZE[dai.slot][dai.value.lower()]
        mr_dict[dai.slot][val] = mr_dict[dai.slot].get(val, 0) + 1

    # create MR dict representation of the output text
    # first, collect all value matches
    matches = []
    for slot in REALIZATIONS.keys():
        # verbatim slot
        if not isinstance(REALIZATIONS[slot], dict):
            matches.extend(
                [
                    Match(slot, CAPITALIZE[slot][match.group(0).lower()], match)
                    for match in REALIZATIONS[slot].finditer(ref)
                ]
            )
        # slot with variable realizations
        else:
            # collect all matches for all values
            for value in REALIZATIONS[slot].keys():
                matches.extend(
                    [
                        Match(slot, CAPITALIZE[slot][value.lower()], match)
                        for match in REALIZATIONS[slot][value].finditer(ref)
                    ]
                )

    # then filter out those that are substrings/duplicates (let only one value match,
    # preferrably the one indicated by the true MR -- check with the MR dict)
    filt_matches = []
    for match in matches:
        skip = False
        for other_match in matches:
            if match is other_match:
                continue
            if match.is_substring(other_match) or (
                match.is_same_string(other_match)
                and (
                    other_match.value in mr_dict.get(other_match.slot, {}).keys()
                    or other_match in filt_matches
                )
            ):
                skip = True
                break
        if not skip:
            filt_matches.append(match)

    # now put it all into a dict
    out_dict = {}
    for match in filt_matches:
        out_dict[match.slot] = out_dict.get(match.slot, {})
        out_dict[match.slot][match.value] = out_dict[match.slot].get(value, 0) + 1

    return out_dict, mr_dict


def check_output(gold_mr, out_mr, fix_type="all"):
    """Check conformity of the given system output (ref) with the input MR. Assumes dict MR representation."""
    # count the errors in the output, looking at the MR
    added, missing, valerr, repeated = 0, 0, 0, 0
    diff = {}
    for slot in set(gold_mr.keys()) | set(out_mr.keys()):
        if slot in gold_mr and slot not in out_mr:
            if (
                fix_type != "all" and "missing" not in fix_type
            ):  # ignore missing stuff -- adjust out_mr
                out_mr[slot] = dict(gold_mr[slot])
            else:
                missing += sum(gold_mr[slot].values())
                diff[slot] = {val: -count for val, count in gold_mr[slot].items()}
        elif slot not in gold_mr and slot in out_mr:
            if (
                fix_type != "all" and "added" not in fix_type
            ):  # ignore added stuff -- adjust out_mr
                del out_mr[slot]
            else:
                added += sum(out_mr[slot].values())
                diff[slot] = out_mr[slot]
        else:
            # remove repeated first (check if MR has same val less than out + same value more than 1x)
            for val in out_mr[slot].keys():
                if val in gold_mr[slot] and gold_mr[slot][val] < out_mr[slot][val]:
                    repeated += out_mr[slot][val] - gold_mr[slot][val]
                    out_mr[slot][val] = gold_mr[slot][val]
            # now compute the diff in the # of value occurrences
            slot_diff = {
                val: gold_mr[slot].get(val, 0) - out_mr[slot].get(val, 0)
                for val in set(gold_mr[slot].keys()) | set(out_mr[slot].keys())
            }
            if fix_type != "all":
                for val, val_diff in list(slot_diff.items()):
                    if (
                        "missing" not in fix_type and val_diff > 0
                    ):  # ignore missing stuff -- adjust out_mr
                        out_mr[slot][val] = out_mr[slot].get(val, 0) + val_diff
                        del slot_diff[val]
                    if (
                        "added" not in fix_type and val_diff < 0
                    ):  # ignore added stuff -- adjust out_mr
                        out_mr[slot][val] = out_mr[slot].get(val, 0) - val_diff
                        del slot_diff[val]
            diff[slot] = {val: -count for val, count in slot_diff.items() if count != 0}
            # diffs both ways
            mr_not_out = sum([count for count in slot_diff.values() if count > 0])
            out_not_mr = -sum([count for count in slot_diff.values() if count < 0])
            # value errors up to the same # of values
            valerr += min(mr_not_out, out_not_mr)
            # others fall under missing & added
            missing += max(mr_not_out - out_not_mr, 0)
            added += max(out_not_mr - mr_not_out, 0)

    diff = json.dumps({slot: vals for slot, vals in diff.items() if vals})
    return added, missing, valerr, repeated, diff, out_mr


def load_lines(filename):
    with codecs.open(filename, "r", "UTF-8") as fh:
        lines = [line.strip() for line in fh.readlines()]
    return lines


def process_file(
    filename, dump=None, fix=None, fix_type="all", out=sys.stdout, mrs=None
):
    """Analyze a single file, optionally dump per-instance stats to a TSV.
    Will print to the `out` file provided (defaults to stdout)."""
    if mrs:
        raw_mrs = [mr.to_diligent_da_string() for mr in mrs]
        refs = load_lines(filename)
        mr_col = "mr"
        ref_col = "ref"
        df = pd.DataFrame({mr_col: raw_mrs, ref_col: refs})
    else:
        # read input from CSV or TSV
        with codecs.open(filename, "r", "UTF-8") as fh:
            line = fh.readline()
            sep = "\t" if "\t" in line else ","
        df = pd.read_csv(filename, sep=sep, encoding="UTF-8")
        # accept column names used in the dataset itself and in system outputs
        mr_col = "MR" if "MR" in df.columns else "mr"
        ref_col = "output" if "output" in df.columns else "ref"
        raw_mrs = list(df[mr_col])
        mrs = [DA.parse_diligent_da(mr) for mr in raw_mrs]  # parse MRs
        refs = list(df[ref_col])

    # count the statistics
    added, missing, valerr, repeated, mr_len, diffs, fixed_mrs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    tot_ok, tot_m, tot_a, tot_ma = 0, 0, 0, 0
    for mr, ref in zip(mrs, refs):
        # check the text (classify MR)
        out_mr, gold_mr = reclassify_mr(ref, mr)
        # build a MR diff
        inst_a, inst_m, inst_v, inst_r, diff, out_mr = check_output(
            gold_mr, out_mr, fix_type
        )
        # just add the totals
        if (inst_a and inst_m) or inst_v:
            tot_ma += 1
        elif inst_a or inst_r:
            tot_a += 1
        elif inst_m:
            tot_m += 1
        else:
            tot_ok += 1
        added.append(inst_a)
        missing.append(inst_m)
        valerr.append(inst_v)
        repeated.append(inst_r)
        diffs.append(diff)
        mr_len.append(len(mr))
        fixed_mr = DA.parse_dict(out_mr)
        fixed_mr.dais.sort(
            key=lambda dai: [
                "name",
                "eat_type",
                "food",
                "price_range",
                "rating",
                "area",
                "family_friendly",
                "near",
            ].index(dai.slot)
        )  # same order as E2E NLG data
        fixed_mr = re.sub(
            r"rating\[", r"customer rating[", fixed_mr.to_diligent_da_string()
        )
        fixed_mr = re.sub(
            r"_([a-z])", lambda match: match.group(1).upper(), fixed_mr
        )  # price_range -> priceRange
        fixed_mrs.append(fixed_mr)

    # print the statistics
    print(filename, file=out)
    print(
        "A: %5d, M: %5d, V: %5d, R: %5d, L: %5d"
        % (sum(added), sum(missing), sum(valerr), sum(repeated), sum(mr_len)),
        file=out,
    )
    semerr = (sum(added) + sum(missing) + sum(valerr) + sum(repeated)) / float(
        sum(mr_len)
    )
    insrate = sum(added) / float(sum(mr_len))
    delrate = sum(missing) / float(sum(mr_len))
    wvlrate = sum(valerr) / float(sum(mr_len))
    print(
        "SemERR = %.4f [InsRate = %.4f, DelRate = %.4f, WVlRate = %.4f]"
        % (semerr, insrate, delrate, wvlrate),
        file=out,
    )
    print(
        "InstOK : %5d / %5d = %.4f" % (tot_ok, len(refs), tot_ok / float(len(refs))),
        file=out,
    )
    print(
        "InstAdd: %5d / %5d = %.4f" % (tot_a, len(refs), tot_a / float(len(refs))),
        file=out,
    )
    print(
        "InstMis: %5d / %5d = %.4f" % (tot_m, len(refs), tot_m / float(len(refs))),
        file=out,
    )
    print(
        "InstM+A: %5d / %5d = %.4f" % (tot_ma, len(refs), tot_ma / float(len(refs))),
        file=out,
    )

    print(
        "Fixed MR String identical to original in %d cases."
        % sum(orig_mr == fixed_mr for orig_mr, fixed_mr in zip(raw_mrs, fixed_mrs))
    )

    # dump per-instance stats to TSV if needed
    if dump:
        df["added"] = added
        df["missing"] = missing
        df["valerr"] = valerr
        df["repeated"] = repeated
        df["mr_len"] = mr_len
        df["diff"] = diffs
        df["fixed_mr"] = fixed_mrs
        df.to_csv(
            dump,
            sep=str("\t"),
            encoding="utf-8",
            index=False,
            columns=[
                mr_col,
                ref_col,
                "added",
                "missing",
                "valerr",
                "repeated",
                "mr_len",
                "diff",
                "fixed_mr",
            ],
        )
    if fix:
        df[mr_col] = fixed_mrs
        df["orig_mr"] = raw_mrs
        df["fixed"] = [
            1 if fixed_mr != orig_mr else 0
            for orig_mr, fixed_mr in zip(raw_mrs, fixed_mrs)
        ]
        df.to_csv(
            fix,
            encoding="utf-8",
            index=False,
            columns=[mr_col, ref_col, "fixed", "orig_mr"],
        )

    # return the stats for CSV stat output if multiple files are processed
    return {
        "filename": filename,
        "semerr": semerr,
        "added": sum(added),
        "missing": sum(missing),
        "valerr": sum(valerr),
        "repeated": sum(repeated),
        "total_attr": sum(mr_len),
        "total_insts": len(refs),
        "inst_ok": tot_ok / float(len(refs)),
        "inst_add": tot_a / float(len(refs)),
        "inst_mis": tot_m / float(len(refs)),
        "inst_m+a": tot_ma / float(len(refs)),
    }


def calculate_ser_for_mr_and_text(mr, text):
    out_mr, gold_mr = reclassify_mr(text, DA.parse_diligent_da(mr))
    added, missing, valerr, repeated, _, _ = check_output(gold_mr, out_mr)
    ser = (added + missing + valerr + repeated) / float(len(gold_mr))
    return ser


def test_calculate_ser_for_mr_and_text():
    assert 1 / 3 == calculate_ser_for_mr_and_text(
        "name[Blue Spice], eatType[coffee shop], area[city centre]",
        "A coffee shop in the city centre area",
    )
    assert 0 == calculate_ser_for_mr_and_text(
        "name[Blue Spice], eatType[coffee shop], area[city centre]",
        "Blue Spice is a coffee shop in the city centre area",
    )



def compute_ser(datafile, outfile, mr_field, text_field):
    df = pd.read_json(datafile, orient="records")
    df["ser"] = df.apply(
        lambda x: calculate_ser_for_mr_and_text(x[mr_field], x[text_field][0]), axis=1
    )
    df["ser_correct"] = df["ser"].apply(lambda x: 0 if x > 0 else 1)

    results = {}
    results["mean_ser"] = round(df["ser"].mean(), 4)
    results["percent_correct_ser"] = round(
        len(df[df["ser"] == 0]) / len(df) * 100, 4
    )
    print(json.dumps(results, indent=2))
    data_dict = df.to_dict(orient="records")
    json.dump(data_dict, open(outfile, "w"), indent=2)

    return results



