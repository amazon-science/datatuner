import re

from external.jjuraska_slug2slug.slot_aligner.alignment.utils import find_first_in_list, find_all_in_list


NEG_IDX_FALSE_PRE_THRESH = 10
NEG_POS_FALSE_PRE_THRESH = 30
NEG_IDX_TRUE_PRE_THRESH = 5
NEG_POS_TRUE_PRE_THRESH = 15
NEG_IDX_POST_THRESH = 10
NEG_POS_POST_THRESH = 30

negation_cues_pre = [
    'no', 'not', 'non', 'none', 'neither', 'nor', 'never', 'n\'t', 'cannot',
    'excluded', 'lack', 'lacks', 'lacking', 'unavailable', 'without', 'zero',
    'everything but'
]
negation_cues_post = [
    'not', 'nor', 'never', 'n\'t', 'cannot',
    'excluded', 'unavailable'
]
contrast_cues = [
    'but', 'however', 'although', 'though', 'nevertheless'
]


def align_boolean_slot(text, text_tok, slot, value, true_val='yes', false_val='no'):
    pos = -1
    text = re.sub(r'\'', '', text)

    # Get the words that possibly realize the slot
    slot_stems = __get_boolean_slot_stems(slot)

    # Search for all possible slot realizations
    for slot_stem in slot_stems:
        idx, pos = find_first_in_list(slot_stem, text_tok)
        if pos >= 0:
            if value == true_val:
                # Match an instance of the slot stem without a preceding negation
                if not __find_negation(text, text_tok, idx, pos, expected_true=True, after=False):
                    return pos
            else:
                # Match an instance of the slot stem with a preceding or a following negation
                if __find_negation(text, text_tok, idx, pos, expected_true=False, after=True):
                    return pos

    # If no match found and the value ~ False, search for alternative expressions of the opposite
    if pos < 0 and value == false_val:
        slot_antonyms = __get_boolean_slot_antonyms(slot)
        for slot_antonym in slot_antonyms:
            if ' ' in slot_antonym:
                pos = text.find(slot_antonym)
            else:
                _, pos = find_first_in_list(slot_antonym, text_tok)

            if pos >= 0:
                return pos

    return -1


def __find_negation(text, text_tok, idx, pos, expected_true=False, after=False):
    # Set the thresholds depending on the expected boolean value of the slot
    if expected_true:
        idx_pre_thresh = NEG_IDX_TRUE_PRE_THRESH
        pos_pre_thresh = NEG_POS_TRUE_PRE_THRESH
    else:
        idx_pre_thresh = NEG_IDX_FALSE_PRE_THRESH
        pos_pre_thresh = NEG_POS_FALSE_PRE_THRESH

    for negation in negation_cues_pre:
        if ' ' in negation:
            neg_pos = text.find(negation)
            if neg_pos >= 0:
                if 0 < (pos - neg_pos - text[neg_pos:pos].count(',')) <= pos_pre_thresh:
                    # Look for a contrast cue between the negation and the slot realization
                    neg_text_segment = text[neg_pos + len(negation):pos]
                    if __has_contrast_after_negation(neg_text_segment):
                        return False
                    else:
                        return True
        else:
            neg_idxs, _ = find_all_in_list(negation, text_tok)
            for neg_idx in neg_idxs:
                if 0 < (idx - neg_idx - text_tok[neg_idx + 1:idx].count(',')) <= idx_pre_thresh:
                    # Look for a contrast cue between the negation and the slot realization
                    neg_text_segment = text_tok[neg_idx + 1:idx]
                    if __has_contrast_after_negation_tok(neg_text_segment):
                        return False
                    else:
                        return True

    if after:
        for negation in negation_cues_post:
            if ' ' in negation:
                neg_pos = text.find(negation)
                if neg_pos >= 0:
                    if 0 < (neg_pos - pos) < NEG_POS_POST_THRESH:
                        return True
            else:
                neg_idxs, _ = find_all_in_list(negation, text_tok)
                for neg_idx in neg_idxs:
                    if 0 < (neg_idx - idx) < NEG_IDX_POST_THRESH:
                        return True

    return False


def __has_contrast_after_negation(text):
    for contr_tok in contrast_cues:
        if text.find(contr_tok) >= 0:
            return True

    return False


def __has_contrast_after_negation_tok(text_tok):
    for contr_tok in contrast_cues:
        if contr_tok in text_tok:
            return True

    return False


def __get_boolean_slot_stems(slot):
    slot_stems = {
        'familyfriendly': ['family', 'families', 'kid', 'kids', 'child', 'children'],
        'hasusbport': ['usb'],
        'isforbusinesscomputing': ['business'],
        'hasmultiplayer': ['multiplayer', 'friends', 'others'],
        'availableonsteam': ['steam'],
        'haslinuxrelease': ['linux'],
        'hasmacrelease': ['mac']
    }

    return slot_stems.get(slot, [])


def __get_boolean_slot_antonyms(slot):
    slot_antonyms = {
        'familyfriendly': ['adult', 'adults'],
        'isforbusinesscomputing': ['personal', 'general', 'home', 'nonbusiness'],
        'hasmultiplayer': ['single player']
    }

    return slot_antonyms.get(slot, [])
