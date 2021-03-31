import os
import io
import string
import json
import pandas as pd
from collections import OrderedDict

from external.jjuraska_slug2slug import config
from external.jjuraska_slug2slug import data_loader
from external.jjuraska_slug2slug.slot_aligner.slot_alignment import split_content, find_alignment, get_scalar_slots


def augment_by_utterance_splitting(dataset, filename, denoise_only=False):
    """Performs utterance splitting and augments the dataset with new pseudo-samples whose utterances are
    one sentence long. The MR of each pseudo-sample contains only slots mentioned in the corresponding sentence.
    Assumes a CSV or JSON file as input.
    """

    if not filename.lower().endswith(('.csv', '.json')):
        raise ValueError('Unexpected file type. Please provide a CSV or JSON file as input.')

    mrs_dicts = []
    data_new = []

    print('Performing utterance splitting on ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, _, _, value_orig = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value_orig

        mrs_dicts.append(mr_dict)

    new_mrs, new_utterances = split_content(mrs_dicts, utterances, filename, permute=False, denoise_only=denoise_only)

    suffix = ' [' + ('denoised' if denoise_only else 'utt. split') + ']'
    filename_out = os.path.splitext(filename)[0] + suffix + os.path.splitext(filename)[1]

    if filename.lower().endswith('.csv'):
        for row, mr in enumerate(new_mrs):
            if len(mr) == 0:
                continue

            mr_str = ', '.join(['{0}[{1}]'.format(slot, value) for slot, value in mr.items()])

            data_new.append([mr_str, new_utterances[row]])

        # Write the augmented dataset to a new file
        pd.DataFrame(data_new).to_csv(os.path.join(config.DATA_DIR, dataset, filename_out),
                                      header=['mr', 'ref'],
                                      index=False,
                                      encoding='utf8')
    elif filename.lower().endswith('.json'):
        for row, mr in enumerate(new_mrs):
            if len(mr) == 0:
                continue

            mr_str = mr.pop('da')
            mr_str += '(' + slot_sep.join(
                ['{0}{1}{2}'.format(key.rstrip(string.digits), val_sep, value) for key, value in mr.items()]
            ) + ')'

            data_new.append([mr_str, new_utterances[row]])

        # Write the augmented dataset to a new file
        with io.open(os.path.join(config.DATA_DIR, dataset, filename_out), 'w', encoding='utf8') as f_data_new:
            json.dump(data_new, f_data_new, indent=4)


def augment_with_aux_indicators(dataset, filename, indicators, mode='all', alt_contrast_mode=False):
    """Augment MRs in a dataset with auxiliary tokens indicating desired discourse phenomena in the corresponding
    utterances. Depending on the mode, the augmented dataset will only contain samples which exhibit 1.) at most one
    of the desired indicators ('single'), 2.) the one selected indicator only ('only'), or 3.) all the desired
    indicators at once ('combo'). The default mode ('all') keeps all samples in the dataset.
    """

    if indicators is None or len(indicators) == 0:
        return

    mrs_augm = []
    mrs_single = []
    utterances_single = []
    mrs_emph_only = []
    utterances_emph_only = []
    mrs_contrast_only = []
    utterances_contrast_only = []
    mrs_combo = []
    utterances_combo = []

    emph_only_ctr = 0
    contrast_only_ctr = 0
    combo_ctr = 0

    print('Augmenting MRs with ' + ' + '.join(indicators) + ' in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    for mr, utt in zip(mrs, utterances):
        mr_dict = OrderedDict()
        mr_list_augm = []

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, slot_orig, value_orig = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value
            mr_list_augm.append((slot, value_orig))
            # mrs[i] = mrs[i].replace(slot_orig, slot)

        # Find the slot alignment
        alignment = find_alignment(utt, mr_dict)

        # Augment the MR with auxiliary tokens
        if 'emphasis' in indicators:
            __add_emphasis_tokens(mr_list_augm, alignment)
        if 'contrast' in indicators:
            __add_contrast_tokens(mr_list_augm, utt, alignment, alt_mode=alt_contrast_mode)

        # Convert augmented MR from list to string representation
        mr_augm = (slot_sep + ' ').join([s + val_sep + v + (val_sep_end if val_sep_end is not None else '') for s, v in mr_list_augm])

        mrs_augm.append(mr_augm)

        # Count and separate the different augmentation instances
        slots_augm = set([s for s, v in mr_list_augm])
        if config.EMPH_TOKEN in slots_augm and (config.CONTRAST_TOKEN in slots_augm or config.CONCESSION_TOKEN in slots_augm):
            mrs_combo.append(mr_augm)
            utterances_combo.append(utt)
            combo_ctr += 1
        else:
            mrs_single.append(mr_augm)
            utterances_single.append(utt)
            if config.EMPH_TOKEN in slots_augm:
                mrs_emph_only.append(mr_augm)
                utterances_emph_only.append(utt)
                emph_only_ctr += 1
            elif config.CONTRAST_TOKEN in slots_augm or config.CONCESSION_TOKEN in slots_augm:
                mrs_contrast_only.append(mr_augm)
                utterances_contrast_only.append(utt)
                contrast_only_ctr += 1

    print('# of MRs with emphasis only:', emph_only_ctr)
    print('# of MRs with contrast/concession only:', contrast_only_ctr)
    print('# of MRs with emphasis & contrast/concession:', combo_ctr)

    new_df = pd.DataFrame(columns=['mr', 'ref'])
    if mode == 'single':
        new_df['mr'] = mrs_single
        new_df['ref'] = utterances_single
    elif mode == 'only':
        if 'emphasis' in indicators:
            new_df['mr'] = mrs_emph_only
            new_df['ref'] = utterances_emph_only
        elif 'contrast' in indicators:
            new_df['mr'] = mrs_contrast_only
            new_df['ref'] = utterances_contrast_only
    elif mode == 'combo':
        new_df['mr'] = mrs_combo
        new_df['ref'] = utterances_combo
    else:
        new_df['mr'] = mrs_augm
        new_df['ref'] = utterances

    # Store augmented dataset to a new file
    filename_out = os.path.splitext(filename)[0] + '_augm_' + '_'.join(indicators) \
        + (('_' + mode) if mode != 'all' else '') + ('_alt' if alt_contrast_mode else '') + '.csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


def __add_emphasis_tokens(mr_list, slot_alignment):
    """Augments an MR with auxiliary tokens indicating that the following slot should be emphasized in the
    generated utterance, i.e. should be mentioned before the name.
    """

    for pos, slot, _ in slot_alignment:
        if slot == 'name':
            break

        idx = 0
        for slot_value_pair in mr_list:
            if slot_value_pair[0] == slot:
                break
            idx += 1
        mr_list.insert(idx, (config.EMPH_TOKEN, ''))


def __add_contrast_tokens(mr_list, utt, slot_alignment, alt_mode=False):
    """Augments an MR with an auxiliary token indicating a pair of slots that should be contrasted in the
    corresponding generated utterance.
    """

    contrast_connectors = ['but', 'however', 'yet']
    scalar_slots = get_scalar_slots()

    for contrast_conn in contrast_connectors:
        contrast_pos = utt.find(contrast_conn)
        if contrast_pos >= 0:
            slot_before = None
            value_before = None
            slot_after = None
            value_after = None

            for pos, slot, value in slot_alignment:
                if pos > contrast_pos:
                    if slot_before is None:
                        # DEBUG PRINT
                        print('Unrecognized type of contrast/concession:')
                        print(utt)
                        print()

                        break
                    if slot in scalar_slots:
                        slot_after = slot
                        value_after = value
                        break
                else:
                    if slot in scalar_slots:
                        slot_before = slot
                        value_before = value

            if slot_before is not None and slot_after is not None:
                if slot_before in scalar_slots and slot_after in scalar_slots:
                    if scalar_slots[slot_before][value_before] - scalar_slots[slot_after][value_after] == 0:
                        if alt_mode:
                            mr_list.insert([s for s, v in mr_list].index(slot_before), (config.CONCESSION_TOKEN, ''))
                            mr_list.insert([s for s, v in mr_list].index(slot_after), (config.CONCESSION_TOKEN, ''))
                        else:
                            mr_list.append((config.CONCESSION_TOKEN, slot_before + ' ' + slot_after))
                            # mr_list.append((config.CONCESSION_TOKEN, ''))
                    else:
                        if alt_mode:
                            mr_list.insert([s for s, v in mr_list].index(slot_before), (config.CONTRAST_TOKEN, ''))
                            mr_list.insert([s for s, v in mr_list].index(slot_after), (config.CONTRAST_TOKEN, ''))
                        else:
                            mr_list.append((config.CONTRAST_TOKEN, slot_before + ' ' + slot_after))
                            # mr_list.append((config.CONTRAST_TOKEN, ''))

            break


# DEPRECATED
def augment_with_contrast_tgen(dataset, filename):
    """Augments the MRs with auxiliary tokens indicating a pair of slots that should be contrasted in the
    corresponding generated utterance. The output is in the format accepted by TGen.
    """

    contrast_connectors = ['but', 'however', 'yet']
    scalar_slots = get_scalar_slots()

    alignments = []
    contrasts = []

    print('Augmenting MRs with contrast in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, slot_orig, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value
            mrs[i] = mrs[i].replace(slot_orig, slot)

        alignments.append(find_alignment(utterances[i], mr_dict))

    for i in range(len(utterances)):
        contrasts.append(['none', 'none', 'none', 'none'])
        for contrast_conn in contrast_connectors:
            contrast_pos = utterances[i].find(contrast_conn)
            if contrast_pos >= 0:
                slot_before = None
                value_before = None
                slot_after = None
                value_after = None

                for pos, slot, value in alignments[i]:
                    if pos > contrast_pos:
                        if not slot_before:
                            break
                        if slot in scalar_slots:
                            slot_after = slot
                            value_after = value
                            break
                    else:
                        if slot in scalar_slots:
                            slot_before = slot
                            value_before = value

                if slot_before and slot_after:
                    if scalar_slots[slot_before][value_before] - scalar_slots[slot_after][value_after] == 0:
                        contrasts[i][2] = slot_before
                        contrasts[i][3] = slot_after
                    else:
                        contrasts[i][0] = slot_before
                        contrasts[i][1] = slot_after

                break

    new_df = pd.DataFrame(columns=['mr', 'ref', 'contrast1', 'contrast2', 'concession1', 'concession2'])
    new_df['mr'] = mrs
    new_df['ref'] = utterances
    new_df['contrast1'] = [tup[0] for tup in contrasts]
    new_df['contrast2'] = [tup[1] for tup in contrasts]
    new_df['concession1'] = [tup[2] for tup in contrasts]
    new_df['concession2'] = [tup[3] for tup in contrasts]

    filename_out = ''.join(filename.split('.')[:-1]) + '_augm_contrast_tgen.csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


if __name__ == '__main__':
    augment_by_utterance_splitting('rest_e2e', 'devset_e2e.csv', denoise_only=True)
    # augment_by_utterance_splitting('laptop', 'valid.json')
    # augment_by_utterance_splitting('tv', 'train.json')
    # augment_by_utterance_splitting('video_game', 'train.csv')

    # augment_with_aux_indicators('rest_e2e', 'devset_e2e.csv', ['contrast'], 'all', alt_contrast_mode=False)

    # augment_with_contrast_tgen('rest_e2e', 'trainset_e2e.csv')
