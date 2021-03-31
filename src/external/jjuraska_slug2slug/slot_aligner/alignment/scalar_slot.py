import re

from external.jjuraska_slug2slug.slot_aligner.alignment.utils import find_first_in_list, find_all_in_list, get_slot_value_alternatives


DIST_IDX_THRESH = 10
DIST_POS_THRESH = 30


def align_scalar_slot(text, text_tok, slot, value, slot_mapping=None, value_mapping=None, slot_stem_only=False):
    slot_stem_indexes = []
    slot_stem_positions = []
    leftmost_pos = -1

    text = re.sub(r'\'', '', text)

    # Get the words that possibly realize the slot
    slot_stems = __get_scalar_slot_stems(slot)

    if slot_mapping is not None:
        slot = slot_mapping
    alternatives = get_slot_value_alternatives(slot)

    # Search for all possible slot realizations
    for slot_stem in slot_stems:
        if len(slot_stem) == 1 and not slot_stem.isalnum():
            # Exception for single-letter special-character slot stems
            slot_stem_pos = [m.start() for m in re.finditer(slot_stem, text)]
        elif len(slot_stem) > 4 or ' ' in slot_stem:
            slot_stem_pos = [m.start() for m in re.finditer(slot_stem, text)]
        else:
            slot_stem_idx, slot_stem_pos = find_all_in_list(slot_stem, text_tok)
            if len(slot_stem_idx) > 0:
                slot_stem_indexes.extend(slot_stem_idx)

        if len(slot_stem_pos) > 0:
            slot_stem_positions.extend(slot_stem_pos)

    slot_stem_positions.sort()
    slot_stem_indexes.sort()

    # If it's only required that the slot stem is matched, don't search for the value
    if slot_stem_only and len(slot_stem_positions) > 0:
        return slot_stem_positions[0]

    # Get the value's alternative realizations
    value_alternatives = [value]
    if value_mapping is not None:
        value = value_mapping[value]
        value_alternatives.append(value)
    if value in alternatives:
        value_alternatives += alternatives[value]

    # Search for all possible value equivalents
    for val in value_alternatives:
        if len(val) > 4 or ' ' in val:
            # Search for multi-word values in the string representation
            val_positions = [m.start() for m in re.finditer(val, text)]
            for pos in val_positions:
                # Remember the leftmost value position as a fallback in case there is no nearby slot stem mention
                if pos < leftmost_pos or leftmost_pos == -1:
                    leftmost_pos = pos

                # Find a slot stem mention within a certain distance from the value realization
                if len(slot_stem_positions) > 0:
                    for slot_stem_pos in slot_stem_positions:
                        if abs(pos - slot_stem_pos) < DIST_POS_THRESH:
                            return pos
        else:
            # Search for single-word values in the tokenized representation
            val_indexes, val_positions = find_all_in_list(val, text_tok)
            for i, idx in enumerate(val_indexes):
                # Remember the leftmost value position as a fallback in case there is no nearby slot stem mention
                if val_positions[i] < leftmost_pos or leftmost_pos == -1:
                    leftmost_pos = val_positions[i]

                # Find a slot stem mention within a certain distance from the value realization
                if len(slot_stem_indexes) > 0:
                    for slot_stem_idx in slot_stem_indexes:
                        if abs(idx - slot_stem_idx) < DIST_IDX_THRESH:
                            return val_positions[i]

    return leftmost_pos


def __get_scalar_slot_stems(slot):
    slot_stems = {
        'esrb': ['esrb'],
        'rating': ['rating', 'ratings', 'rated', 'rate', 'review', 'reviews'],
        'customerrating': ['customer', 'rating', 'ratings', 'rated', 'rate', 'review', 'reviews', 'star', 'stars'],
        'pricerange': ['price', 'pricing', 'cost', 'costs', 'dollars', 'pounds', 'euros', '\$', '£', '€']
    }

    return slot_stems.get(slot, [])
