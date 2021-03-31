import os
import pandas as pd
from collections import Counter, OrderedDict

from external.jjuraska_slug2slug import config
from external.jjuraska_slug2slug import data_loader
from external.jjuraska_slug2slug.slot_aligner.slot_alignment import find_alignment, count_errors, score_alignment


def align_slots(dataset, filename):
    """Aligns slots of the MRs with their mentions in the corresponding utterances."""

    alignments = []
    alignment_strings = []

    print('Aligning slots in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs and the utterances
    mrs = [data_loader.preprocess_mr(mr, data_cont['separators']) for mr in mrs_orig]
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value

        alignments.append(find_alignment(utterances[i], mr_dict))

    for i in range(len(utterances)):
        alignment_strings.append(' '.join(['({0}: {1})'.format(pos, slot) for pos, slot, _ in alignments[i]]))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'alignment'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['alignment'] = alignment_strings

    filename_out = os.path.splitext(filename)[0] + ' [aligned].csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


def score_slot_realizations(path, filename):
    """Analyzes unrealized and hallucinated slot mentions in the utterances."""

    errors = []
    incorrect_slots = []
    # slot_cnt = 0

    # print('Analyzing missing slot realizations and hallucinations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(path, filename))
    dataset_name = data_cont['dataset_name']
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs and the utterances
    mrs = [data_loader.preprocess_mr(mr, data_cont['separators']) for mr in mrs_orig]
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    # DEBUG PRINT
    # print('\n'.join([mr1 + '\n' + mr2 + '\n' for mr1, mr2 in zip(mrs_orig, mrs)]))

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value

            # Count auxiliary slots
            # if not re.match(r'<!.*?>', slot):
            #     slot_cnt += 1

        # TODO: get rid of this hack
        # Move the food-slot to the end of the dict (because of delexing)
        if 'food' in mr_dict:
            food_val = mr_dict['food']
            del(mr_dict['food'])
            mr_dict['food'] = food_val

        # Delexicalize the MR and the utterance
        utterances[i] = data_loader.delex_sample(mr_dict, utterances[i], dataset=dataset_name)

        # Count the missing and hallucinated slots in the utterance
        cur_errors, cur_incorrect_slots = count_errors(utterances[i], mr_dict)
        errors.append(cur_errors)
        incorrect_slots.append(', '.join(cur_incorrect_slots))

    # DEBUG PRINT
    # print(slot_cnt)

    new_df = pd.DataFrame(columns=['mr', 'ref', 'errors', 'incorrect slots'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['errors'] = errors
    new_df['incorrect slots'] = incorrect_slots

    filename_out = os.path.splitext(filename)[0] + ' [errors].csv'
    new_df.to_csv(os.path.join(path, filename_out), index=False, encoding='utf8')


def score_emphasis(dataset, filename):
    """Determines how many of the indicated emphasis instances are realized in the utterance."""

    emph_missed = []
    emph_total = []

    print('Analyzing emphasis realizations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.EVAL_DIR, dataset, filename))
    dataset_name = data_cont['dataset_name']
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs and the utterances
    mrs = [data_loader.preprocess_mr(mr, data_cont['separators']) for mr in mrs_orig]
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs):
        expect_emph = False
        emph_slots = set()
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)

            # Extract slots to be emphasized
            if slot == config.EMPH_TOKEN:
                expect_emph = True
            else:
                mr_dict[slot] = value
                if expect_emph:
                    emph_slots.add(slot)
                    expect_emph = False

        # Delexicalize the MR and the utterance
        utterances[i] = data_loader.delex_sample(mr_dict, utterances[i], dataset=dataset_name)

        # Determine the slot alignment in the utterance
        alignment = find_alignment(utterances[i], mr_dict)

        emph_total.append(len(emph_slots))

        # Check how many emphasized slots were not realized before the name-slot
        for pos, slot, _ in alignment:
            # DEBUG PRINT
            # print(alignment)
            # print(emph_slots)
            # print()

            if slot == 'name':
                break

            if slot in emph_slots:
                emph_slots.remove(slot)

        emph_missed.append(len(emph_slots))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'missed emphasis', 'total emphasis'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['missed emphasis'] = emph_missed
    new_df['total emphasis'] = emph_total

    filename_out = os.path.splitext(filename)[0] + ' [emphasis eval].csv'
    new_df.to_csv(os.path.join(config.EVAL_DIR, dataset, filename_out), index=False, encoding='utf8')


def score_contrast(dataset, filename):
    """Determines whether the indicated contrast relation is correctly realized in the utterance."""

    contrast_connectors = ['but', 'however', 'yet']
    contrast_missed = []
    contrast_incorrectness = []
    contrast_total = []

    print('Analyzing contrast realizations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.EVAL_DIR, dataset, filename))
    dataset_name = data_cont['dataset_name']
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs and the utterances
    mrs = [data_loader.preprocess_mr(mr, data_cont['separators']) for mr in mrs_orig]
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs):
        contrast_found = False
        contrast_correct = False
        contrast_slots = []
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)

            # Extract slots to be contrasted
            if slot in [config.CONTRAST_TOKEN, config.CONCESSION_TOKEN]:
                contrast_slots.extend(value.split())
            else:
                mr_dict[slot] = value

        # Delexicalize the MR and the utterance
        utterances[i] = data_loader.delex_sample(mr_dict, utterances[i], dataset=dataset_name)

        # Determine the slot alignment in the utterance
        alignment = find_alignment(utterances[i], mr_dict)

        contrast_total.append(1 if len(contrast_slots) > 0 else 0)

        if len(contrast_slots) > 0:
            for contrast_conn in contrast_connectors:
                contrast_pos = utterances[i].find(contrast_conn)
                if contrast_pos < 0:
                    continue

                slot_left_pos = -1
                slot_right_pos = -1
                dist = 0

                contrast_found = True

                # Check whether the correct pair of slots was contrasted
                for pos, slot, _ in alignment:
                    # DEBUG PRINT
                    # print(alignment)
                    # print(contrast_slots)
                    # print()

                    if slot_left_pos > -1:
                        dist += 1

                    if slot in contrast_slots:
                        if slot_left_pos == -1:
                            slot_left_pos = pos
                        else:
                            slot_right_pos = pos
                            break

                if slot_left_pos > -1 and slot_right_pos > -1:
                    if slot_left_pos < contrast_pos < slot_right_pos and dist <= 2:
                        contrast_correct = True
                        break
        else:
            contrast_found = True
            contrast_correct = True

        contrast_missed.append(0 if contrast_found else 1)
        contrast_incorrectness.append(0 if contrast_correct else 1)

    new_df = pd.DataFrame(columns=['mr', 'ref', 'missed contrast', 'incorrect contrast', 'total contrast'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['missed contrast'] = contrast_missed
    new_df['incorrect contrast'] = contrast_incorrectness
    new_df['total contrast'] = contrast_total

    filename_out = os.path.splitext(filename)[0] + ' [contrast eval].csv'
    new_df.to_csv(os.path.join(config.EVAL_DIR, dataset, filename_out), index=False, encoding='utf8')


def analyze_contrast_relations(dataset, filename):
    """Identifies the slots involved in a contrast relation."""

    contrast_connectors = ['but', 'however', 'yet']
    slots_before = []
    slots_after = []

    print('Analyzing contrast relations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs
    mrs = [data_loader.preprocess_mr(mr, data_cont['separators']) for mr in mrs_orig]

    for mr, utt in zip(mrs, utterances_orig):
        mr_dict = OrderedDict()
        mr_list_augm = []

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, slot_orig, value_orig = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value
            mr_list_augm.append((slot, value_orig))

        # Find the slot alignment
        alignment = find_alignment(utt, mr_dict)

        slot_before = None
        slot_after = None

        for contrast_conn in contrast_connectors:
            contrast_pos = utt.find(contrast_conn)
            if contrast_pos >= 0:
                slot_before = None
                slot_after = None

                for pos, slot, value in alignment:
                    slot_before = slot_after
                    slot_after = slot

                    if pos > contrast_pos:
                        break

                break

        slots_before.append(slot_before if slot_before is not None else '')
        slots_after.append(slot_after if slot_after is not None else '')

    # Calculate the frequency distribution of slots involved in a contrast relation
    contrast_slot_cnt = Counter()
    contrast_slot_cnt.update(slots_before + slots_after)
    del contrast_slot_cnt['']
    print('\n---- Slot distribution in contrast relations ----\n')
    print('\n'.join(slot + ': ' + str(freq) for slot, freq in contrast_slot_cnt.most_common()))

    # Calculate the frequency distribution of slot pairs involved in a contrast relation
    contrast_slot_cnt = Counter()
    slot_pairs = [tuple(sorted(slot_pair)) for slot_pair in zip(slots_before, slots_after) if slot_pair != ('', '')]
    contrast_slot_cnt.update(slot_pairs)
    print('\n---- Slot pair distribution in contrast relations ----\n')
    print('\n'.join(slot_pair[0] + ', ' + slot_pair[1] + ': ' + str(freq) for slot_pair, freq in contrast_slot_cnt.most_common()))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'slot before contrast', 'slot after contrast'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['slot before contrast'] = slots_before
    new_df['slot after contrast'] = slots_after

    filename_out = os.path.splitext(filename)[0] + ' [contrast relations].csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


if __name__ == '__main__':
    # align_slots('rest_e2e', 'devset_e2e.csv')
    # align_slots('video_game', 'valid.csv')

    # ----

    # score_slot_realizations(config.E2E_DATA_DIR, 'devset_e2e.csv')
    # score_slot_realizations(config.LAPTOP_DATA_DIR, 'test.json')
    score_slot_realizations(os.path.join(config.EVAL_DIR, 'predictions rest_e2e (new)'),
                            'predictions RNN utt. split (11.4k) beam 10, no reranking.csv')

    # ----

    # score_emphasis('predictions-rest_e2e_stylistic_selection/devset', 'predictions RNN (4+4) augm emph (reference).csv')

    # ----

    # predictions_dir = 'predictions rest_e2e (emphasis+contrast)'
    # predictions_file = 'predictions TRANS emphasis+contrast, train single, test combo extra (23.2k iter).csv'
    #
    # score_emphasis(predictions_dir, predictions_file)
    # score_contrast(predictions_dir, predictions_file)

    # ----

    # analyze_contrast_relations('rest_e2e', 'devset_e2e.csv')
