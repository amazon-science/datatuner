import math
import random
import re
from copy import deepcopy
from itertools import chain
from pathlib import Path

import nltk
import pandas as pd

random.seed(42)


def get_distractors(data, text, swapping_candidates, cutting_candidates, random_text, num_candidates=5,
                    max_per_operation=5):
    """Get the distractors for the given inputs"""
    distractors_dict = {}

    for cands in swapping_candidates:
        distractors_dict["value_error"] = swap_entities(cands, text, max_outputs=max_per_operation)

    for cands in cutting_candidates:
        distractors_dict["value_error"].extend(cut_entities(cands, text, max_outputs=max_per_operation))

    distractors_dict["value_error"].extend(add_negation_errors(text, max_outputs=int(math.ceil(max_per_operation / 2))))
    distractors_dict["omission"] = add_omission(text, max_outputs=max_per_operation)
    if "," in text:
        distractors_dict["omission"].extend(add_phrase_omission(text, max_outputs=1 + max_per_operation))

    distractors_dict["repetition"] = add_repetition(text, max_outputs=1 + max_per_operation)
    distractors_dict["hallucination"] = add_repetition(
        text, random_text=random_text, replace=True, max_outputs=max_per_operation
    ) + add_repetition(text, random_text=random_text, max_outputs=max_per_operation)

    distractors = set(chain(*distractors_dict.values()))

    # Remove text itself if present
    if text in distractors:
        distractors.remove(text)

    # Shuffle and cut
    distractors = list(distractors)
    random.shuffle(distractors)
    distractors = distractors[:num_candidates]

    # If no distractors found, add placeholders
    if len(distractors) == 0:
        distractors = ["placeholder"] * num_candidates
    # Pad to get to the right number of candidates
    if len(distractors) < num_candidates:
        ratio = int(math.ceil(num_candidates / len(distractors)))
        distractors = (distractors * ratio)[:num_candidates]

    classification_items = [
                               {"text": value, "data": data, "label": key} for key in distractors_dict for value in
                               distractors_dict[key]
                           ] + [{"text": text, "data": data, "label": "accurate"}]

    # Add negation
    replacements = {"[ no ]": "[ yes ]", "[ yes ]": "[ no ]"}
    for cand in replacements:
        if cand in data:
            negated_data = data.replace(cand, replacements[cand], 1)
            classification_items.extend([{"text": text, "data": negated_data, "label": "value_error"}])

    random.shuffle(classification_items)
    classification_items = classification_items[:num_candidates]
    return distractors, classification_items


def add_negation_errors(original_text, max_outputs=5):
    outputs = []
    current_text = original_text
    blacklisted = ["not", "n't"]
    for x in blacklisted:
        if "not " in current_text:
            new_text = current_text.replace("not", "", 1)
            new_text = new_text.replace("  ", " ")
            outputs.append(new_text)
            current_text = new_text

    return outputs[:max_outputs]


def cut_entities(entity_list, original_text, max_outputs=5):
    """Remove part of the entity"""
    output = []
    entity_list = deepcopy(entity_list)[:max_outputs]
    for entity in entity_list:
        rand_ind = random.randint(0, len(entity) - 1)
        cut_entity = entity[:rand_ind].strip()
        if entity in original_text:
            output.append(original_text.replace(entity, cut_entity))

    return output


def swap_entities(entity_list, original_text, max_outputs=5):
    """Swap an entity from the `entity_list` with another from the list if present in the text"""
    entity_set = set(entity_list)
    output = []

    entity_list = deepcopy(entity_list)[:max_outputs]

    random.shuffle(entity_list)
    for entity in entity_list:
        passed_entities = deepcopy(entity_set)
        passed_entities.remove(entity)
        passed_entities = list(passed_entities)
        if len(passed_entities) > 0:
            rand_entity = random.choice(passed_entities)
            if entity in original_text:
                text = original_text.replace(entity, rand_entity)
                output.append(text)

    return output


def swap_pronouns(original_text):
    """Swap pronoun with a different one"""
    lower = ["he", "she", "it", "they"]
    upper = ["He", "She", "It", "They"]

    # find if a pronoun is there
    tokens = set(re.findall(r"[\w']+", original_text.lower()))
    pronoun_i = -1
    for i, p in enumerate(lower):
        if p in tokens:
            pronoun_i = i
            break

    text = original_text
    # if a pronoun is found, we replace its occurrences with a random other pronoun
    if pronoun_i >= 0:

        # get a random other pronoun
        candidates = set(list(range(len(lower))))
        candidates.remove(pronoun_i)
        other_pronoun_i = random.choice(list(candidates))

        for pronouns in [lower, upper]:
            pronoun = pronouns[pronoun_i]
            other_pronoun = pronouns[other_pronoun_i]
            text = re.sub(r"\b{}\b".format(pronoun), other_pronoun, text)
        return [text]

    return []


def add_phrase_omission(text, max_outputs=5):
    indices = [i for i, x in enumerate(text) if x == ","]
    output = []
    random.shuffle(indices)
    end_strs = [".", ","]
    random.shuffle(end_strs)
    for end_str in end_strs:
        for i in indices:
            try:
                if len(output) >= max_outputs:
                    break
                # until the index before comma + index from next dot if any
                output.append(text[:i] + text[text.index(end_str, i + 1):])

            except:
                pass

    return output


def add_omission(text, max_outputs=5):
    """Remove the shortest sentence from the text"""
    sentences = nltk.sent_tokenize(text)
    output = []
    if len(sentences) > 1:
        for omit_ind in range(min(max_outputs, len(sentences))):
            # sort by increasing length; goal is to remove shortest for subtle omissions
            sentences = sorted(sentences, key=lambda x: len(x))
            removed_sentence = sentences[omit_ind]
            output.append(text.replace(removed_sentence, "").strip())
    return output


def add_repetition(text, random_text=None, replace=False, max_outputs=5):
    """Repeat the shortest sentence int the text"""
    sentences = nltk.sent_tokenize(text)

    assert not (random_text is None and replace)

    if random_text is None:
        sorted_sentences = sorted(sentences, key=lambda x: len(x))
        repeat_ind = 0
        random_sentence = sorted_sentences[repeat_ind]
    else:
        random_sentences = nltk.sent_tokenize(random_text)
        random_sentence = sorted(random_sentences, key=lambda x: len(x))[0]

    indices = list(range(min(max_outputs, len(sentences))))
    random.shuffle(indices)

    outputs = []
    for insert_at_ind in indices:
        if replace:
            sentences[insert_at_ind] = random_sentence
        else:
            sentences.insert(insert_at_ind, random_sentence)

        outputs.append(" ".join(sentences).strip())
    return outputs[:max_outputs]


def write_classification_data(classification_data, classification_dir, split):
    classification_dir = Path(classification_dir)
    classification_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(classification_data)
    max_data = 100000 if "train" in split else 10000
    classification_data = classification_data[:max_data]
    print(split)
    print(f"Length of classification_data: {len(classification_data)}")
    for x in classification_data:
        x["text"] = x["text"].replace("\n", " ").replace("\r", " ")

    df = pd.DataFrame(classification_data)

    print(f"Original classes distribution:")
    print(f"{df.label.value_counts()}")
    if "train" in split:
        max_size = df["label"].value_counts().max()

        lst = [df]
        for class_index, group in df.groupby("label"):
            sizes = [max_size, 3 * len(group), max_size - len(group)]
            size = min(sizes)
            lst.append(group.sample(size, replace=True))

        df_new = pd.concat(lst)
        df_new = df_new.sample(frac=min(1, max_data / len(df_new))).reset_index(drop=True)
        print(f"New classes distribution:")
        print(f"{df_new.label.value_counts()}")
    else:
        df_new = df.sample(frac=min(1, max_data / len(df))).reset_index(drop=True)

    labels = list(df_new.label.unique())
    df_new.to_csv(classification_dir / (split + ".tsv"), sep="|", index=False, columns=["label", "data", "text"])

    labels_file = classification_dir / "labels.txt"
    labels_file.write_text("\n".join(sorted(labels)))
    print("")
