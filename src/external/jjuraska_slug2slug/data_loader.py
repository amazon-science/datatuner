import os
import io
import random
import string
import re
import json
import pandas as pd
import numpy as np
from collections import OrderedDict
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from external.jjuraska_slug2slug import config


EMPH_TOKEN = config.EMPH_TOKEN
CONTRAST_TOKEN = config.CONTRAST_TOKEN
CONCESSION_TOKEN = config.CONCESSION_TOKEN


# TODO: redesign the data loading so as to be object-oriented
def load_training_data(data_trainset, data_devset, input_concat=False, generate_vocab=False, skip_if_exist=True):
    """Generate source and target files in the required input format for the model training.
    """

    training_source_file = os.path.join(config.DATA_DIR, 'training_source.txt')
    training_target_file = os.path.join(config.DATA_DIR, 'training_target.txt')
    dev_source_file = os.path.join(config.DATA_DIR, 'dev_source.txt')
    dev_target_file = os.path.join(config.DATA_DIR, 'dev_target.txt')

    if skip_if_exist:
        # If there is an existing source and target file, skip their generation
        if os.path.isfile(training_source_file) and \
                os.path.isfile(training_target_file) and \
                os.path.isfile(dev_source_file) and \
                os.path.isfile(dev_target_file):
            print('Found existing input files. Skipping their generation.')
            return

    dataset = init_training_data(data_trainset, data_devset)
    dataset_name = dataset['dataset_name']
    x_train, y_train, x_dev, y_dev = dataset['data']
    _, _, slot_sep, val_sep, val_sep_end = dataset['separators']

    # Preprocess the MRs and the utterances
    x_train = [preprocess_mr(x, dataset['separators']) for x in x_train]
    x_dev = [preprocess_mr(x, dataset['separators']) for x in x_dev]
    y_train = [preprocess_utterance(y) for y in y_train]
    y_dev = [preprocess_utterance(y) for y in y_dev]

    # Produce sequences of extracted words from the meaning representations (MRs) in the trainset
    x_train_seq = []
    for i, mr in enumerate(x_train):
        slot_ctr = 0
        emph_idxs = set()
        # contrast_idxs = set()
        # concession_idxs = set()
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_end)

            if slot == EMPH_TOKEN:
                emph_idxs.add(slot_ctr)
            # elif slot == CONTRAST_TOKEN:
            #     contrast_idxs.add(slot_ctr)
            # elif slot == CONCESSION_TOKEN:
            #     concession_idxs.add(slot_ctr)
            else:
                mr_dict[slot] = value
                slot_ctr += 1

        # Delexicalize the MR and the utterance
        y_train[i] = delex_sample(mr_dict, y_train[i], dataset=dataset_name, input_concat=input_concat)

        slot_ctr = 0

        # Convert the dictionary to a list
        x_train_seq.append([])
        for key, val in mr_dict.items():
            # Insert the emphasis token where appropriate
            if slot_ctr in emph_idxs:
                x_train_seq[i].append(EMPH_TOKEN)
            # Insert the contrast token where appropriate
            # if slot_ctr in contrast_idxs:
            #     x_train_seq[i].append(CONTRAST_TOKEN)
            # # Insert the concession token where appropriate
            # if slot_ctr in concession_idxs:
            #     x_train_seq[i].append(CONCESSION_TOKEN)

            if len(val) > 0:
                x_train_seq[i].extend([key] + val.split())
            else:
                x_train_seq[i].append(key)

            slot_ctr += 1

        if input_concat:
            # Append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
            x_train_seq[i].append('<STOP>')

    # Produce sequences of extracted words from the meaning representations (MRs) in the devset
    x_dev_seq = []
    for i, mr in enumerate(x_dev):
        slot_ctr = 0
        emph_idxs = set()
        # contrast_idxs = set()
        # concession_idxs = set()
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_end)

            if slot == EMPH_TOKEN:
                emph_idxs.add(slot_ctr)
            # elif slot == CONTRAST_TOKEN:
            #     contrast_idxs.add(slot_ctr)
            # elif slot == CONCESSION_TOKEN:
            #     concession_idxs.add(slot_ctr)
            else:
                mr_dict[slot] = value
                slot_ctr += 1

        # Delexicalize the MR and the utterance
        y_dev[i] = delex_sample(mr_dict, y_dev[i], dataset=dataset_name, input_concat=input_concat)

        slot_ctr = 0

        # Convert the dictionary to a list
        x_dev_seq.append([])
        for key, val in mr_dict.items():
            # Insert the emphasis token where appropriate
            if slot_ctr in emph_idxs:
                x_dev_seq[i].append(EMPH_TOKEN)
            # Insert the contrast token where appropriate
            # if slot_ctr in contrast_idxs:
            #     x_dev_seq[i].append(CONTRAST_TOKEN)
            # # Insert the concession token where appropriate
            # if slot_ctr in concession_idxs:
            #     x_dev_seq[i].append(CONCESSION_TOKEN)

            if len(val) > 0:
                x_dev_seq[i].extend([key] + val.split())
            else:
                x_dev_seq[i].append(key)

            slot_ctr += 1

        if input_concat:
            # Append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
            x_dev_seq[i].append('<STOP>')

    y_train_seq = [word_tokenize(y) for y in y_train]
    y_dev_seq = [word_tokenize(y) for y in y_dev]

    # Generate a vocabulary file if necessary
    if generate_vocab:
        generate_vocab_file(np.concatenate(x_train_seq + x_dev_seq + y_train_seq + y_dev_seq),
                            vocab_filename='vocab.lang_gen.tokens')
        # generate_vocab_file(np.concatenate(x_train_seq + x_dev_seq),
        #                     vocab_filename='vocab.lang_gen_multi_vocab.source')
        # generate_vocab_file(np.concatenate(y_train_seq + y_dev_seq),
        #                     vocab_filename='vocab.lang_gen_multi_vocab.target')

    with io.open(training_source_file, 'w', encoding='utf8') as f_x_train:
        for line in x_train_seq:
            f_x_train.write('{}\n'.format(' '.join(line)))

    with io.open(training_target_file, 'w', encoding='utf8') as f_y_train:
        for line in y_train:
            f_y_train.write(line + '\n')

    with io.open(dev_source_file, 'w', encoding='utf8') as f_x_dev:
        for line in x_dev_seq:
            f_x_dev.write('{}\n'.format(' '.join(line)))

    with io.open(dev_target_file, 'w', encoding='utf8') as f_y_dev:
        for line in y_dev:
            f_y_dev.write(line + '\n')

    return np.concatenate(x_train_seq + x_dev_seq + y_train_seq + y_dev_seq).flatten()


def load_test_data(data_testset, input_concat=False):
    """Generate source and target files in the required input format for the model testing.
    """

    test_source_file = os.path.join(config.DATA_DIR, 'test_source.txt')
    test_source_dict_file = os.path.join(config.DATA_DIR, 'test_source_dict.json')
    test_target_file = os.path.join(config.DATA_DIR, 'test_target.txt')
    test_reference_file = os.path.join(config.METRICS_DIR, 'test_references.txt')

    dataset = init_test_data(data_testset)
    dataset_name = dataset['dataset_name']
    x_test, y_test = dataset['data']
    _, _, slot_sep, val_sep, val_sep_end = dataset['separators']

    # Preprocess the MRs
    x_test = [preprocess_mr(x, dataset['separators']) for x in x_test]

    # Produce sequences of extracted words from the meaning representations (MRs) in the testset
    x_test_seq = []
    x_test_dict = []
    for i, mr in enumerate(x_test):
        slot_ctr = 0
        emph_idxs = set()
        # contrast_idxs = set()
        # concession_idxs = set()
        mr_dict = OrderedDict()
        mr_dict_cased = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_end)

            if slot == EMPH_TOKEN:
                emph_idxs.add(slot_ctr)
            # elif slot == CONTRAST_TOKEN:
            #     contrast_idxs.add(slot_ctr)
            # elif slot == CONCESSION_TOKEN:
            #     concession_idxs.add(slot_ctr)
            else:
                mr_dict[slot] = value
                mr_dict_cased[slot] = value_orig
                slot_ctr += 1

        # Build an MR dictionary with original values
        x_test_dict.append(mr_dict_cased)

        # Delexicalize the MR
        delex_sample(mr_dict, dataset=dataset_name, mr_only=True, input_concat=input_concat)

        slot_ctr = 0

        # Convert the dictionary to a list
        x_test_seq.append([])
        for key, val in mr_dict.items():
            # Insert the emphasis token where appropriate
            if slot_ctr in emph_idxs:
                x_test_seq[i].append(EMPH_TOKEN)
            # Insert the contrast token where appropriate
            # if slot_ctr in contrast_idxs:
            #     x_test_seq[i].append(CONTRAST_TOKEN)
            # # Insert the concession token where appropriate
            # if slot_ctr in concession_idxs:
            #     x_test_seq[i].append(CONCESSION_TOKEN)

            if len(val) > 0:
                x_test_seq[i].extend([key] + val.split())
            else:
                x_test_seq[i].append(key)

            slot_ctr += 1

        if input_concat:
            # Append a sequence-end token to be paired up with seq2seq's sequence-end token when concatenating
            x_test_seq[i].append('<STOP>')

    with io.open(test_source_file, 'w', encoding='utf8') as f_x_test:
        for line in x_test_seq:
            f_x_test.write('{}\n'.format(' '.join(line)))

    with io.open(test_source_dict_file, 'w', encoding='utf8') as f_x_test_dict:
        json.dump(x_test_dict, f_x_test_dict)

    if len(y_test) > 0:
        with io.open(test_target_file, 'w', encoding='utf8') as f_y_test:
            for line in y_test:
                f_y_test.write(line + '\n')

        # Reference file for calculating metrics for test predictions
        with io.open(test_reference_file, 'w', encoding='utf8') as f_y_test:
            for i, line in enumerate(y_test):
                if i > 0 and x_test[i] != x_test[i - 1]:
                    f_y_test.write('\n')
                f_y_test.write(line + '\n')


def generate_vocab_file(token_sequences, vocab_filename, vocab_size=10000):
    vocab_file = os.path.join(config.DATA_DIR, vocab_filename)

    distr = FreqDist(token_sequences)
    vocab = distr.most_common(min(len(distr), vocab_size - 3))      # cap the vocabulary size

    vocab_with_reserved_tokens = ['<pad>', '<EOS>'] + list(map(lambda tup: tup[0], vocab)) + ['UNK']

    with io.open(vocab_file, 'w', encoding='utf8') as f_vocab:
        for token in vocab_with_reserved_tokens:
            f_vocab.write('{}\n'.format(token))


def get_vocabulary(token_sequences, vocab_size=10000):
    distr = FreqDist(token_sequences)
    vocab = distr.most_common(min(len(distr), vocab_size))          # cap the vocabulary size

    vocab_set = set(map(lambda tup: tup[0], vocab))

    return vocab_set


# TODO: generalize and utilize in the loading functions
def tokenize_mr(mr):
    """Produces a (delexicalized) sequence of tokens from the input MR.
    Method used in the client to preprocess a single MR that is sent to the service for utterance generation.
    """

    slot_sep = ','
    val_sep = '['
    val_sep_end = ']'
    
    mr_seq = []
    slot_ctr = 0
    emph_idxs = set()
    mr_dict = OrderedDict()
    mr_dict_cased = OrderedDict()

    # Extract the slot-value pairs into a dictionary
    for slot_value in mr.split(slot_sep):
        slot, value, _, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_end)

        if slot == EMPH_TOKEN:
            emph_idxs.add(slot_ctr)
        else:
            mr_dict[slot] = value
            mr_dict_cased[slot] = value_orig
            slot_ctr += 1

    # Delexicalize the MR
    delex_sample(mr_dict, mr_only=True)

    slot_ctr = 0

    # Convert the dictionary to a list
    for key, val in mr_dict.items():
        # Insert the emphasis token where appropriate
        if slot_ctr in emph_idxs:
            mr_seq.append(EMPH_TOKEN)

        if len(val) > 0:
            mr_seq.extend([key] + val.split())
        else:
            mr_seq.append(key)

        slot_ctr += 1

    return mr_seq, mr_dict_cased


def load_training_data_for_eval(data_trainset, data_model_outputs_train, vocab_size, max_input_seq_len, max_output_seq_len, delex=False):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_end = None

    if '/rest_e2e/' in data_trainset or '\\rest_e2e\\' in data_trainset:
        x_train, y_train_1 = read_rest_e2e_dataset_train(data_trainset)
        y_train_2 = read_predictions(data_model_outputs_train)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_end = ']'
    elif '/tv/' in data_trainset or '\\tv\\' in data_trainset:
        x_train, y_train_1, y_train_2 = read_tv_dataset_train(data_trainset)
        if data_model_outputs_train is not None:
            y_train_2 = read_predictions(data_model_outputs_train)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_trainset or '\\laptop\\' in data_trainset:
        x_train, y_train_1, y_train_2 = read_laptop_dataset_train(data_trainset)
        if data_model_outputs_train is not None:
            y_train_2 = read_predictions(data_model_outputs_train)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_train_1 = [preprocess_utterance(y) for y in y_train_1]
    y_train_2 = [preprocess_utterance(y) for y in y_train_2]
    

    # produce sequences of extracted words from the meaning representations (MRs) in the trainset
    x_train_seq = []
    for i, mr in enumerate(x_train):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value

        if delex == True:
            # delexicalize the MR and the utterance
            y_train_1[i] = delex_sample(mr_dict, y_train_1[i], dataset=dataset_name, utterance_only=True)
            y_train_2[i] = delex_sample(mr_dict, y_train_2[i], dataset=dataset_name)

        # convert the dictionary to a list
        x_train_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_train_seq[i].extend([key, val])
            else:
                x_train_seq[i].append(key)


    # create source vocabulary
    if os.path.isfile('data/eval_vocab_source.json'):
        with io.open('data/eval_vocab_source.json', 'r', encoding='utf8') as f_x_vocab:
            x_vocab = json.load(f_x_vocab)
    else:
        x_distr = FreqDist([x_token for x in x_train_seq for x_token in x])
        x_vocab = x_distr.most_common(min(len(x_distr), vocab_size - 2))        # cap the vocabulary size
        with io.open('data/eval_vocab_source.json', 'w', encoding='utf8') as f_x_vocab:
            json.dump(x_vocab, f_x_vocab, ensure_ascii=False)

    x_idx2word = [word[0] for word in x_vocab]
    x_idx2word.insert(0, '<PADDING>')
    x_idx2word.append('<NA>')
    x_word2idx = {word: idx for idx, word in enumerate(x_idx2word)}

    # create target vocabulary
    if os.path.isfile('data/eval_vocab_target.json'):
        with io.open('data/eval_vocab_target.json', 'r', encoding='utf8') as f_y_vocab:
            y_vocab = json.load(f_y_vocab)
    else:
        y_distr = FreqDist([y_token for y in y_train_1 for y_token in y] + [y_token for y in y_train_2 for y_token in y])
        y_vocab = y_distr.most_common(min(len(y_distr), vocab_size - 2))        # cap the vocabulary size
        with io.open('data/eval_vocab_target.json', 'w', encoding='utf8') as f_y_vocab:
            json.dump(y_vocab, f_y_vocab, ensure_ascii=False)

    y_idx2word = [word[0] for word in y_vocab]
    y_idx2word.insert(0, '<PADDING>')
    y_idx2word.append('<NA>')
    y_word2idx = {token: idx for idx, token in enumerate(y_idx2word)}


    # produce sequences of indexes from the MRs in the training set
    x_train_enc = token_seq_to_idx_seq(x_train_seq, x_word2idx, max_input_seq_len)

    # produce sequences of indexes from the utterances in the training set
    y_train_1_enc = token_seq_to_idx_seq(y_train_1, y_word2idx, max_output_seq_len)

    # produce sequences of indexes from the utterances in the training set
    y_train_2_enc = token_seq_to_idx_seq(y_train_2, y_word2idx, max_output_seq_len)

    # produce the list of the target labels in the training set
    labels_train = np.concatenate((np.ones(len(y_train_1_enc)), np.zeros(len(y_train_2_enc))))


    return (np.concatenate((np.array(x_train_enc), np.array(x_train_enc))),
            np.concatenate((np.array(y_train_1_enc), np.array(y_train_2_enc))),
            labels_train)


def load_dev_data_for_eval(data_devset, data_model_outputs_dev, vocab_size, max_input_seq_len, max_output_seq_len, delex=True):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_end = None

    if '/rest_e2e/' in data_devset or '\\rest_e2e\\' in data_devset:
        x_dev, y_dev_1 = read_rest_e2e_dataset_dev(data_devset)
        y_dev_2 = read_predictions(data_model_outputs_dev)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_end = ']'
    elif '/tv/' in data_devset or '\\tv\\' in data_devset:
        x_dev, y_dev_1, y_dev_2 = read_tv_dataset_dev(data_devset)
        if data_model_outputs_dev is not None:
            y_dev_2 = read_predictions(data_model_outputs_dev)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_devset or '\\laptop\\' in data_devset:
        x_dev, y_dev_1, y_dev_2 = read_laptop_dataset_dev(data_devset)
        if data_model_outputs_dev is not None:
            y_dev_2 = read_predictions(data_model_outputs_dev)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_dev_1 = [preprocess_utterance(y) for y in y_dev_1]
    y_dev_2 = [preprocess_utterance(y) for y in y_dev_2]
    

    # produce sequences of extracted words from the meaning representations (MRs) in the devset
    x_dev_seq = []
    for i, mr in enumerate(x_dev):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value
            
        if delex == True:
            # delexicalize the MR and the utterance
            y_dev_1[i] = delex_sample(mr_dict, y_dev_1[i], dataset=dataset_name, utterance_only=True)
            y_dev_2[i] = delex_sample(mr_dict, y_dev_2[i], dataset=dataset_name)

        # convert the dictionary to a list
        x_dev_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_dev_seq[i].extend([key, val])
            else:
                x_dev_seq[i].append(key)


    # load the source vocabulary
    with io.open('data/eval_vocab_source.json', 'r', encoding='utf8') as f_x_vocab:
        x_vocab = json.load(f_x_vocab)

    x_idx2word = [word[0] for word in x_vocab]
    x_idx2word.insert(0, '<PADDING>')
    x_idx2word.append('<NA>')
    x_word2idx = {word: idx for idx, word in enumerate(x_idx2word)}

    # load the target vocabulary
    with io.open('data/eval_vocab_target.json', 'r', encoding='utf8') as f_y_vocab:
        y_vocab = json.load(f_y_vocab)

    y_idx2word = [word[0] for word in y_vocab]
    y_idx2word.insert(0, '<PADDING>')
    y_idx2word.append('<NA>')
    y_word2idx = {token: idx for idx, token in enumerate(y_idx2word)}
    

    # produce sequences of indexes from the MRs in the devset
    x_dev_enc = token_seq_to_idx_seq(x_dev_seq, x_word2idx, max_input_seq_len)

    # produce sequences of indexes from the utterances in the devset
    y_dev_1_enc = token_seq_to_idx_seq(y_dev_1, y_word2idx, max_output_seq_len)

    # produce sequences of indexes from the utterances in the devset
    y_dev_2_enc = token_seq_to_idx_seq(y_dev_2, y_word2idx, max_output_seq_len)

    # produce the list of the target labels in the devset
    labels_dev = np.concatenate((np.ones(len(y_dev_1_enc)), np.zeros(len(y_dev_2_enc))))


    return (np.concatenate((np.array(x_dev_enc), np.array(x_dev_enc))),
            np.concatenate((np.array(y_dev_1_enc), np.array(y_dev_2_enc))),
            labels_dev)


def load_test_data_for_eval(data_testset, data_model_outputs_test, vocab_size, max_input_seq_len, max_output_seq_len, delex=False):
    dataset_name = ''
    slot_sep = ''
    val_sep = ''
    val_sep_end = None

    if '/rest_e2e/' in data_testset or '\\rest_e2e\\' in data_testset:
        x_test, _ = read_rest_e2e_dataset_test(data_testset)
        y_test = read_predictions(data_model_outputs_test)
        dataset_name = 'rest_e2e'
        slot_sep = ','
        val_sep = '['
        val_sep_end = ']'
    elif '/tv/' in data_testset or '\\tv\\' in data_testset:
        x_test, _, y_test = read_tv_dataset_test(data_testset)
        if data_model_outputs_test is not None:
            y_test = read_predictions(data_model_outputs_test)
        dataset_name = 'tv'
        slot_sep = ';'
        val_sep = '='
    elif '/laptop/' in data_testset or '\\laptop\\' in data_testset:
        x_test, _, y_test = read_laptop_dataset_test(data_testset)
        if data_model_outputs_test is not None:
            y_test = read_predictions(data_model_outputs_test)
        dataset_name = 'laptop'
        slot_sep = ';'
        val_sep = '='
    else:
        raise FileNotFoundError

    # parse the utterances into lists of words
    y_test = [preprocess_utterance(y) for y in y_test]
    #y_test_1 = [preprocess_utterance(y) for y in y_test_1]
    #y_test_2 = [preprocess_utterance(y) for y in y_test_2]
    

    # produce sequences of extracted words from the meaning representations (MRs) in the testset
    x_test_seq = []
    for i, mr in enumerate(x_test):
        mr_dict = OrderedDict()
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value

        if delex == True:
            # delexicalize the MR and the utterance
            y_test[i] = delex_sample(mr_dict, y_test[i], dataset=dataset_name)
            #y_test_1[i] = delex_sample(mr_dict, y_test_1[i], dataset=dataset_name, utterance_only=True)
            #y_test_2[i] = delex_sample(mr_dict, y_test_2[i], dataset=dataset_name)

        # convert the dictionary to a list
        x_test_seq.append([])
        for key, val in mr_dict.items():
            if len(val) > 0:
                x_test_seq[i].extend([key, val])
            else:
                x_test_seq[i].append(key)


    # load the source vocabulary
    with io.open('data/eval_vocab_source.json', 'r', encoding='utf8') as f_x_vocab:
        x_vocab = json.load(f_x_vocab)

    x_idx2word = [word[0] for word in x_vocab]
    x_idx2word.insert(0, '<PADDING>')
    x_idx2word.append('<NA>')
    x_word2idx = {word: idx for idx, word in enumerate(x_idx2word)}

    # load the target vocabulary
    with io.open('data/eval_vocab_target.json', 'r', encoding='utf8') as f_y_vocab:
        y_vocab = json.load(f_y_vocab)

    y_idx2word = [word[0] for word in y_vocab]
    y_idx2word.insert(0, '<PADDING>')
    y_idx2word.append('<NA>')
    y_word2idx = {token: idx for idx, token in enumerate(y_idx2word)}


    # produce sequences of indexes from the MRs in the test set
    x_test_enc = token_seq_to_idx_seq(x_test_seq, x_word2idx, max_input_seq_len)

    # produce sequences of indexes from the utterances in the test set
    y_test_enc = token_seq_to_idx_seq(y_test, y_word2idx, max_output_seq_len)
    #y_test_1_enc = token_seq_to_idx_seq(y_test_1, y_word2idx, max_output_seq_len)
    #y_test_2_enc = token_seq_to_idx_seq(y_test_2, y_word2idx, max_output_seq_len)

    # produce the list of the target labels in the test set
    labels_test = np.ones(len(y_test_enc))
    #labels_test = np.concatenate((np.ones(len(y_test_1_enc)), np.zeros(len(y_test_2_enc))))


    return (np.array(x_test_enc),
            np.array(y_test_enc),
            labels_test,
            x_idx2word,
            y_idx2word)

    #return (np.concatenate((np.array(x_test_enc), np.array(x_test_enc))),
    #        np.concatenate((np.array(y_test_1_enc), np.array(y_test_2_enc))),
    #        labels_test,
    #        x_idx2word,
    #        y_idx2word)


# ---- AUXILIARY FUNCTIONS ----


def init_training_data(data_trainset, data_devset):
    if 'rest_e2e' in data_trainset and 'rest_e2e' in data_devset:
        x_train, y_train = read_rest_e2e_dataset_train(data_trainset)
        x_dev, y_dev = read_rest_e2e_dataset_dev(data_devset)
        dataset_name = 'rest_e2e'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ', '
        val_sep = '['
        val_sep_end = ']'
    elif 'video_game' in data_trainset and 'video_game' in data_devset:
        x_train, y_train = read_video_game_dataset_train(data_trainset)
        x_dev, y_dev = read_video_game_dataset_dev(data_devset)
        dataset_name = 'video_game'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ', '
        val_sep = '['
        val_sep_end = ']'
    elif 'tv' in data_trainset and 'tv' in data_devset:
        x_train, y_train, _ = read_tv_dataset_train(data_trainset)
        x_dev, y_dev, _ = read_tv_dataset_dev(data_devset)
        dataset_name = 'tv'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ';'
        val_sep = '='
        val_sep_end = None
    elif 'laptop' in data_trainset and 'laptop' in data_devset:
        x_train, y_train, _ = read_laptop_dataset_train(data_trainset)
        x_dev, y_dev, _ = read_laptop_dataset_dev(data_devset)
        dataset_name = 'laptop'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ';'
        val_sep = '='
        val_sep_end = None
    elif 'hotel' in data_trainset and 'hotel' in data_devset:
        x_train, y_train, _ = read_hotel_dataset_train(data_trainset)
        x_dev, y_dev, _ = read_hotel_dataset_dev(data_devset)
        dataset_name = 'hotel'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ';'
        val_sep = '='
        val_sep_end = None
    else:
        raise ValueError('Unexpected file name or path: {0}, {1}'.format(data_trainset, data_devset))

    return {
        'dataset_name': dataset_name,
        'data': (x_train, y_train, x_dev, y_dev),
        'separators': (da_sep, da_sep_end, slot_sep, val_sep, val_sep_end)
    }


def init_test_data(data_testset):
    if 'rest_e2e' in data_testset:
        x_test, y_test = read_rest_e2e_dataset_test(data_testset)
        dataset_name = 'rest_e2e'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ', '
        val_sep = '['
        val_sep_end = ']'
    elif 'video_game' in data_testset:
        x_test, y_test = read_video_game_dataset_test(data_testset)
        dataset_name = 'video_game'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ', '
        val_sep = '['
        val_sep_end = ']'
    elif 'tv' in data_testset:
        x_test, y_test, _ = read_tv_dataset_test(data_testset)
        dataset_name = 'tv'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ';'
        val_sep = '='
        val_sep_end = None
    elif 'laptop' in data_testset:
        x_test, y_test, _ = read_laptop_dataset_test(data_testset)
        dataset_name = 'laptop'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ';'
        val_sep = '='
        val_sep_end = None
    elif 'hotel' in data_testset:
        x_test, y_test, _ = read_hotel_dataset_test(data_testset)
        dataset_name = 'hotel'
        da_sep = '('
        da_sep_end = ')'
        slot_sep = ';'
        val_sep = '='
        val_sep_end = None
    else:
        raise ValueError('Unexpected file name or path: {0}'.format(data_testset))

    return {
        'dataset_name': dataset_name,
        'data': (x_test, y_test),
        'separators': (da_sep, da_sep_end, slot_sep, val_sep, val_sep_end)
    }


def read_rest_e2e_dataset_train(data_trainset):
    # read the training data from file
    df_train = pd.read_csv(data_trainset, header=0, encoding='utf8')    # names=['mr', 'ref']
    x_train = df_train.mr.tolist()
    y_train = df_train.ref.tolist()

    return x_train, y_train


def read_rest_e2e_dataset_dev(data_devset):
    # read the development data from file
    df_dev = pd.read_csv(data_devset, header=0, encoding='utf8')        # names=['mr', 'ref']
    x_dev = df_dev.mr.tolist()
    y_dev = df_dev.ref.tolist()

    return x_dev, y_dev


def read_rest_e2e_dataset_test(data_testset):
    # read the test data from file
    df_test = pd.read_csv(data_testset, header=0, encoding='utf8')      # names=['mr', 'ref']
    x_test = df_test.iloc[:, 0].tolist()
    y_test = []
    if df_test.shape[1] > 1:
        y_test = df_test.iloc[:, 1].tolist()

    return x_test, y_test


def read_video_game_dataset_train(data_trainset):
    # read the training data from file
    df_train = pd.read_csv(data_trainset, header=0, encoding='utf8')    # names=['mr', 'ref']
    x_train = df_train.mr.tolist()
    y_train = df_train.ref.tolist()

    return x_train, y_train


def read_video_game_dataset_dev(data_devset):
    # read the development data from file
    df_dev = pd.read_csv(data_devset, header=0, encoding='utf8')        # names=['mr', 'ref']
    x_dev = df_dev.mr.tolist()
    y_dev = df_dev.ref.tolist()

    return x_dev, y_dev


def read_video_game_dataset_test(data_testset):
    # read the test data from file
    df_test = pd.read_csv(data_testset, header=0, encoding='utf8')      # names=['mr', 'ref']
    x_test = df_test.iloc[:, 0].tolist()
    y_test = []
    if df_test.shape[1] > 1:
        y_test = df_test.iloc[:, 1].tolist()

    return x_test, y_test


def read_tv_dataset_train(path_to_trainset):
    with io.open(path_to_trainset, encoding='utf8') as f_trainset:
        # Skip the comment block at the beginning of the file
        f_trainset, _ = skip_comment_block(f_trainset, '#')

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()
    y_train_alt = df_train.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')
        
    # convert plural nouns to "[noun] -s" or "[noun] -es" form
    for i, utt in enumerate(y_train):
        y_train[i] = replace_plural_nouns(utt)
    for i, utt in enumerate(y_train_alt):
        y_train_alt[i] = replace_plural_nouns(utt)
        
    return x_train, y_train, y_train_alt


def read_tv_dataset_dev(path_to_devset):
    with io.open(path_to_devset, encoding='utf8') as f_devset:
        # Skip the comment block at the beginning of the file
        f_devset, _ = skip_comment_block(f_devset, '#')

        # read the development data from file
        df_dev = pd.read_json(f_devset, encoding='utf8')

    x_dev = df_dev.iloc[:, 0].tolist()
    y_dev = df_dev.iloc[:, 1].tolist()
    y_dev_alt = df_dev.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_dev):
        x_dev[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')
        
    # convert plural nouns to "[noun] -s" or "[noun] -es" form
    for i, utt in enumerate(y_dev):
        y_dev[i] = replace_plural_nouns(utt)
    for i, utt in enumerate(y_dev_alt):
        y_dev_alt[i] = replace_plural_nouns(utt)

    return x_dev, y_dev, y_dev_alt


def read_tv_dataset_test(path_to_testset):
    with io.open(path_to_testset, encoding='utf8') as f_testset:
        # Skip the comment block at the beginning of the file
        f_testset, _ = skip_comment_block(f_testset, '#')

        # read the test data from file
        df_test = pd.read_json(f_testset, encoding='utf8')

    x_test = df_test.iloc[:, 0].tolist()
    y_test = df_test.iloc[:, 1].tolist()
    y_test_alt = df_test.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_test):
        x_test[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')

    return x_test, y_test, y_test_alt


def read_laptop_dataset_train(path_to_trainset):
    with io.open(path_to_trainset, encoding='utf8') as f_trainset:
        # Skip the comment block at the beginning of the file
        f_trainset, _ = skip_comment_block(f_trainset, '#')

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()
    y_train_alt = df_train.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')

    return x_train, y_train, y_train_alt


def read_laptop_dataset_dev(path_to_devset):
    with io.open(path_to_devset, encoding='utf8') as f_devset:
        # Skip the comment block at the beginning of the file
        f_devset, _ = skip_comment_block(f_devset, '#')

        # read the development data from file
        df_dev = pd.read_json(f_devset, encoding='utf8')

    x_dev = df_dev.iloc[:, 0].tolist()
    y_dev = df_dev.iloc[:, 1].tolist()
    y_dev_alt = df_dev.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_dev):
        x_dev[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')

    return x_dev, y_dev, y_dev_alt


def read_laptop_dataset_test(path_to_testset):
    with io.open(path_to_testset, encoding='utf8') as f_testset:
        # Skip the comment block at the beginning of the file
        f_testset, _ = skip_comment_block(f_testset, '#')

        # read the test data from file
        df_test = pd.read_json(f_testset, encoding='utf8')

    x_test = df_test.iloc[:, 0].tolist()
    y_test = df_test.iloc[:, 1].tolist()
    y_test_alt = df_test.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_test):
        x_test[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')

    return x_test, y_test, y_test_alt


def read_hotel_dataset_train(path_to_trainset):
    with io.open(path_to_trainset, encoding='utf8') as f_trainset:
        # Skip the comment block at the beginning of the file
        f_trainset, _ = skip_comment_block(f_trainset, '#')

        # read the training data from file
        df_train = pd.read_json(f_trainset, encoding='utf8')

    x_train = df_train.iloc[:, 0].tolist()
    y_train = df_train.iloc[:, 1].tolist()
    y_train_alt = df_train.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_train):
        x_train[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')

    return x_train, y_train, y_train_alt


def read_hotel_dataset_dev(path_to_devset):
    with io.open(path_to_devset, encoding='utf8') as f_devset:
        # Skip the comment block at the beginning of the file
        f_devset, _ = skip_comment_block(f_devset, '#')

        # read the development data from file
        df_dev = pd.read_json(f_devset, encoding='utf8')

    x_dev = df_dev.iloc[:, 0].tolist()
    y_dev = df_dev.iloc[:, 1].tolist()
    y_dev_alt = df_dev.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_dev):
        x_dev[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')

    return x_dev, y_dev, y_dev_alt


def read_hotel_dataset_test(path_to_testset):
    with io.open(path_to_testset, encoding='utf8') as f_testset:
        # Skip the comment block at the beginning of the file
        f_testset, _ = skip_comment_block(f_testset, '#')

        # read the test data from file
        df_test = pd.read_json(f_testset, encoding='utf8')

    x_test = df_test.iloc[:, 0].tolist()
    y_test = df_test.iloc[:, 1].tolist()
    y_test_alt = df_test.iloc[:, 2].tolist()

    # TODO: remove from here and use the universal DA extraction instead
    # transform the MR to contain the DA type as the first slot
    for i, mr in enumerate(x_test):
        x_test[i] = preprocess_mr_for_tv_laptop(mr, '(', ';', '=')

    return x_test, y_test, y_test_alt


def read_predictions(path_to_predictions):
    # read the test data from file
    with io.open(path_to_predictions, encoding='utf8') as f_predictions:
        y_pred = f_predictions.readlines()

    return y_pred


def skip_comment_block(fd, comment_symbol):
    """Reads the initial lines of the file (represented by the file descriptor) corresponding to a comment block.
    All consecutive lines starting with the given symbol are considered to be part of the comment block.
    """

    comment_block = ''

    line_beg = fd.tell()
    line = fd.readline()
    while line != '':
        if not line.startswith(comment_symbol):
            fd.seek(line_beg)
            break

        comment_block += line
        line_beg = fd.tell()
        line = fd.readline()

    return fd, comment_block


def replace_plural_nouns(utt):
    stemmer = WordNetLemmatizer()

    pos_tags = nltk.pos_tag(nltk.word_tokenize(utt))
    tokens_to_replace = []
    tokens_new = []

    for token, tag in pos_tags:
        #if tag == 'NNS':
        if token in ['inches', 'watts']:
            tokens_to_replace.append(token)
            tokens_new.append(split_plural_noun(token, stemmer))
        
    for token_to_replace, token_new in zip(tokens_to_replace, tokens_new):
        utt = utt.replace(token_to_replace, token_new)

    return utt


def split_plural_noun(word, stemmer):
    stem = stemmer.lemmatize(word)
    if stem not in word or stem == word:
        return word

    suffix = word.replace(stem, '')

    return stem + ' -' + suffix


def replace_commas_in_mr_values(mr, val_sep, val_sep_end):
    mr_new = ''
    val_beg_cnt = 0
    val_end_cnt = 0

    for c in mr:
        # If comma inside a value, replace the comma with placeholder
        if c == ',' and val_beg_cnt > val_end_cnt:
            mr_new += config.COMMA_PLACEHOLDER
            continue

        # Keep track of value beginning and end
        if c == val_sep:
            val_beg_cnt += 1
        elif c == val_sep_end:
            val_end_cnt += 1

        mr_new += c

    return mr_new


def put_back_commas_in_mr_values(mrs):
    return [mr.replace(config.COMMA_PLACEHOLDER, ',') for mr in mrs]


def preprocess_da_in_mr(mr, separators):
    # Unpack separators
    da_sep, da_sep_end, slot_sep, val_sep, val_sep_end = separators

    # If no DA indication is expected in the data, return the MR unchanged
    if da_sep is None:
        return mr

    # Verify if DA type is indicated at the beginning of the MR
    da_sep_idx = mr.find(da_sep)
    slot_sep_idx = mr.find(slot_sep)
    val_sep_idx = mr.find(val_sep)
    if da_sep_idx < 0 or 0 <= slot_sep_idx < da_sep_idx or 0 <= val_sep_idx < da_sep_idx:
        return mr

    # Extract the DA type from the beginning of the MR
    da_type = mr[:da_sep_idx].lstrip('?')      # Strip the '?' symbol present in Laptop and TV datasets
    slot_value_pairs = mr[da_sep_idx + 1:]
    if da_sep_end is not None:
        slot_value_pairs = slot_value_pairs.rstrip(da_sep_end)

    # Convert the extracted DA to the slot-value form and prepend it to the remainder of the MR
    mr_new = 'da' + val_sep + da_type
    if val_sep_end is not None:
        mr_new += val_sep_end
    if len(slot_value_pairs) > 0:
        mr_new += slot_sep + slot_value_pairs

    return mr_new


# TODO: merge with the above function
def preprocess_mr_for_tv_laptop(mr, da_sep, slot_sep, val_sep):
    sep_idx = mr.find(da_sep)
    da_type = mr[:sep_idx].lstrip('?')
    slot_value_pairs = mr[sep_idx:].strip('()')

    mr_new = 'da=' + da_type
    if len(slot_value_pairs) > 0:
        mr_new += slot_sep + slot_value_pairs

    mr_modified = ''
    for slot_value in mr_new.split(slot_sep):
        slot, _, _, value_orig = parse_slot_and_value(slot_value, val_sep)
        # If the value is enclosed in apostrophes, remove them
        if value_orig.startswith('\'') and value_orig.endswith('\''):
            value_orig = value_orig[1:-1]

        mr_modified += slot + val_sep + value_orig + slot_sep

    mr_new = mr_modified[:-1]

    if da_type in ['compare', 'suggest']:
        slot_counts = {}
        mr_modified = ''
        for slot_value in mr_new.split(slot_sep):
            slot, _, _, value_orig = parse_slot_and_value(slot_value, val_sep)
            if slot in ['da', 'position']:
                mr_modified += slot
            else:
                slot_counts[slot] = slot_counts.get(slot, 0) + 1
                mr_modified += slot + str(slot_counts[slot])

            mr_modified += val_sep + value_orig + slot_sep

        mr_new = mr_modified[:-1]

    return mr_new


def preprocess_mr(mr, separators):
    # Transform the MR to list the DA type as the first slot, if its indication is present in the MR
    mr_new = preprocess_da_in_mr(mr, separators)

    # Replace commas in values if comma is the slot separator
    if separators[2].strip() == ',' and separators[4] is not None:
        mr_new = replace_commas_in_mr_values(mr_new, separators[3], separators[4])

    return mr_new


def preprocess_utterance(utt):
    return ' '.join(word_tokenize(utt.lower()))


def parse_slot_and_value(slot_value, val_sep, val_sep_end=None):
    sep_idx = slot_value.find(val_sep)
    if sep_idx > -1:
        # Parse the slot
        slot = slot_value[:sep_idx].strip()
        # Parse the value
        if val_sep_end is not None:
            value = slot_value[sep_idx + 1:-1].strip()
        else:
            value = slot_value[sep_idx + 1:].strip()
    else:
        # Parse the slot
        if val_sep_end is not None:
            slot = slot_value[:-1].strip()
        else:
            slot = slot_value.strip()
        # Set the value to the empty string
        value = ''

    slot_processed = slot.replace(' ', '').lower()
    if not slot_processed.startswith('__'):
        slot_processed = slot_processed.replace('_', '')

    value = value.replace(config.COMMA_PLACEHOLDER, ',')
    # TODO: fix the cases where a period is in the value
    # TODO: (e.g., the suggest DA file (2 slots) or verify_attribute DA file (4 slots) in the video game dataset)
    value_processed = ' '.join(word_tokenize(value.lower()))

    return slot_processed, value_processed, slot, value


def delex_sample(mr, utterance=None, dataset=None, slots_to_delex=None, mr_only=False, input_concat=False, utterance_only=False):
    """Delexicalizes a single sample (MR and the corresponding utterance).
    By default, the slots 'name', 'near' and 'food' are delexicalized (for the E2E dataset).

    All fields (E2E): name, near, area, food, customer rating, familyFriendly, eatType, priceRange
    """

    if not mr_only and utterance is None:
        raise ValueError('the \'utterance\' argument must be provided when \'mr_only\' is False.')

    if slots_to_delex is not None:
        delex_slots = slots_to_delex
    else:
        if dataset == 'rest_e2e':
            delex_slots = ['name', 'near', 'food']
            # delex_slots = ['name', 'releaseyear', 'expreleasedate', 'developer']        # counterfeit video_game
        elif dataset == 'video_game':
            delex_slots = ['name', 'releaseyear', 'expreleasedate', 'developer']
        elif dataset == 'tv':
            delex_slots = ['name', 'family', 'hdmiport', 'screensize', 'price', 'audio', 'resolution', 'powerconsumption', 'color', 'count']
        elif dataset == 'laptop':
            delex_slots = ['name', 'family', 'processor', 'memory', 'drive', 'battery', 'weight', 'dimension', 'design', 'platform', 'warranty', 'count']
        elif dataset == 'hotel':
            delex_slots = ['name', 'address', 'postcode', 'area', 'near', 'phone', 'count']
        else:
            # By default, assume the dataset is 'rest_e2e'
            delex_slots = ['name', 'near', 'food']

    # Sort the slots to be delexed in a descending order of their value's length (to avoid delexing of a value that is
    #   a substring of another value to be delexed)
    delex_slots_sorted = [(s, v) for s, v in mr.items()
                          if s.rstrip(string.digits) in delex_slots and v not in ['dontcare', 'none', '']]
    delex_slots_sorted = [s for s, v in sorted(delex_slots_sorted, key=lambda x: len(x[1]), reverse=True)]

    mr_update = {}

    # for slot, value in mr.items():
    for slot in delex_slots_sorted:
        value = mr[slot]
        if value not in ['dontcare', 'none', '']:
            # Assemble a placeholder token for the value
            placeholder = create_placeholder(slot, value)

            values_alt = [value]
            # Specify special rules for individual slots, including alternative representations of the values
            if slot == 'address':
                if 'street' in value:
                    values_alt.append(re.sub(r'\b{}\b'.format('street'), 'st', value))
                elif 'avenue' in value:
                    values_alt.append(re.sub(r'\b{}\b'.format('avenue'), 'ave', value))
            elif slot == 'name':
                # If name is contained in the developer slot value, delexicalize the developer slot first
                if not mr_only and 'developer' in mr and value in mr['developer']:
                    dev_placeholder = create_placeholder('developer', mr['developer'])
                    dev_val_preproc = ' '.join(word_tokenize(mr['developer']))
                    utterance = re.sub(r'\b{}\b'.format(dev_val_preproc), dev_placeholder, utterance)
                    mr_update['developer'] = dev_placeholder
            elif slot in ['developer', 'expreleasedate']:
                values_alt = [value.replace(';', ',')]

            utterance_delexed = utterance
            if not mr_only:
                for val in values_alt:
                    # Replace the value (whole-word matches only) with the placeholder
                    utterance_delexed = re.sub(r'\b{}\b'.format(val), placeholder, utterance)
                    if utterance_delexed != utterance:
                        break

            # Do not replace value with a placeholder token unless there is an exact match in the utterance
            if slot not in mr_update and (mr_only or utterance_delexed != utterance or slot == 'name'):
                mr_update[slot] = placeholder
                utterance = utterance_delexed
        else:
            if input_concat:
                mr_update[slot] = value.replace(' ', '_')

    if not utterance_only:
        for slot, new_value in mr_update.items():
            mr[slot] = new_value

    if not mr_only:
        # Tokenize punctuation missed by tokenizer (such as after years and numbers in titles) before delexicalization
        utterance = utterance.replace(config.DELEX_SUFFIX + ',', config.DELEX_SUFFIX + ' ,')
        utterance = utterance.replace(config.DELEX_SUFFIX + '.', config.DELEX_SUFFIX + ' .')

        return utterance


def counterfeit_sample(mr, utt, target_dataset=None, slots_to_replace=None, slot_value_dict=None):
    """Counterfeits a single E2E sample (MR and the corresponding utterance).
    """

    mr_counterfeit = {}
    utt_counterfeit = utt

    if slots_to_replace is None:
        if target_dataset == 'rest_e2e':
            slots_to_replace = ['name', 'near', 'food']
        elif target_dataset == 'video_game':
            slots_to_replace = ['name', 'releaseyear', 'expreleasedate', 'developer']
        elif target_dataset == 'tv':
            slots_to_replace = ['name', 'family', 'hdmiport', 'screensize', 'price', 'audio', 'resolution', 'powerconsumption', 'color', 'count']
        elif target_dataset == 'laptop':
            slots_to_replace = ['name', 'family', 'processor', 'memory', 'drive', 'battery', 'weight', 'dimension', 'design', 'platform', 'warranty', 'count']
        elif target_dataset == 'hotel':
            slots_to_replace = ['name', 'address', 'postcode', 'area', 'near', 'phone', 'count']
        else:
            slots_to_replace = []

    if target_dataset == 'video_game':
        for slot_orig, value_orig in mr.items():
            slot_counterfeit = slot_orig
            value_counterfeit = value_orig

            if slot_orig.rstrip(string.digits) in slots_to_replace:
                # Substitute the slot with the corresponding slot from the target domain
                slot_counterfeit = e2e_slot_to_video_game_slot(slot_orig)
                while slot_counterfeit in mr_counterfeit:
                    slot_counterfeit = e2e_slot_to_video_game_slot(slot_orig)

                if slot_orig == 'food':
                    # If value mentioned in the MR verbatim, replace with a sampled value from the target domain
                    if value_orig in utt_counterfeit:
                        value_counterfeit = random.choice(slot_value_dict[slot_counterfeit])
                        value_realization = value_counterfeit
                        utt_counterfeit = re.sub(value_orig, value_realization, utt_counterfeit)

                    # Replace related keywords/phrases with alternatives matching the target domain
                    if slot_counterfeit == 'releaseyear':
                        phrase_counterfeit1 = random.choice(['was released in', 'came out in'])
                        phrase_counterfeit2 = random.choice(['released in', 'from'])
                    elif slot_counterfeit == 'expreleasedate':
                        phrase_counterfeit1 = random.choice(['will be released on', 'is expected to come out', 'is coming out on'])
                        phrase_counterfeit2 = random.choice(['to be released on', 'expected to be released on', 'slated for release on'])
                    else:
                        phrase_counterfeit1 = ''
                        phrase_counterfeit2 = ''

                    utt_counterfeit = re.sub(r'\bserves\b', phrase_counterfeit1, utt_counterfeit)
                    utt_counterfeit = re.sub(r'\bserving\b', phrase_counterfeit2, utt_counterfeit)
                    utt_counterfeit = re.sub(r'\bprovides\b', phrase_counterfeit1, utt_counterfeit)
                    utt_counterfeit = re.sub(r'\bproviding\b', phrase_counterfeit2, utt_counterfeit)
                    utt_counterfeit = re.sub(r'\bfood\b', '', utt_counterfeit)
                elif slot_orig == 'customerrating':
                    # If value mentioned in the MR verbatim, replace with a sampled value from the target domain
                    if value_orig in utt_counterfeit:
                        value_counterfeit = random.choice(slot_value_dict[slot_counterfeit])
                        value_realization = value_counterfeit
                        utt_counterfeit = re.sub(value_orig, value_realization, utt_counterfeit)

                    # Replace related keywords/phrases with alternatives matching the target domain
                    if slot_counterfeit == 'rating':
                        phrase_counterfeit = 'rating'
                    elif slot_counterfeit == 'esrb':
                        phrase_counterfeit = 'esrb rating'
                    else:
                        phrase_counterfeit = ''

                    for w in ['customer ratings', 'customer rating', 'ratings', 'rating']:
                        utt_counterfeit_sub = re.sub(r'\b{}\b'.format(w), phrase_counterfeit, utt_counterfeit)
                        if utt_counterfeit_sub != utt_counterfeit:
                            utt_counterfeit = utt_counterfeit_sub
                            break
                elif slot_orig == 'pricerange':
                    # If value mentioned in the MR verbatim, replace with a sampled value from the target domain
                    if value_orig in utt_counterfeit:
                        value_counterfeit = random.choice(slot_value_dict[slot_counterfeit])
                        if ',' in value_counterfeit:
                            value_items = [val.strip() for val in value_counterfeit.split(',')]
                            value_items_shuffled = random.sample(value_items, len(value_items))
                            value_realization = ', '.join(value_items_shuffled[:-1]) + ' and ' + value_items_shuffled[-1]
                        else:
                            value_realization = value_counterfeit
                        utt_counterfeit = re.sub(value_orig, value_realization, utt_counterfeit)

                    # Replace related keywords/phrases with alternatives matching the target domain
                    if slot_counterfeit == 'playerperspective':
                        phrase_counterfeit = 'perspective'
                    else:
                        phrase_counterfeit = ''

                    for w in ['price range', 'priced', 'prices', 'price']:
                        utt_counterfeit_sub = re.sub(r'\b{}\b'.format(w), phrase_counterfeit, utt_counterfeit)
                        if utt_counterfeit_sub != utt_counterfeit:
                            utt_counterfeit = utt_counterfeit_sub
                            break
                elif slot_orig == 'familyfriendly':
                    if slot_counterfeit == 'hasmultiplayer':
                        phrase_counterfeit = 'multiplayer'
                    elif slot_counterfeit == 'availableonsteam':
                        phrase_counterfeit = 'steam'
                    elif slot_counterfeit == 'haslinuxrelease':
                        phrase_counterfeit = 'linux'
                    elif slot_counterfeit == 'hasmacrelease':
                        phrase_counterfeit = 'mac'
                    else:
                        phrase_counterfeit = ''

                    for w in ['families', 'children', 'kids', 'family', 'child', 'kid']:
                        utt_counterfeit_sub = re.sub(r'\b{}\b'.format(w), phrase_counterfeit, utt_counterfeit)
                        if utt_counterfeit_sub != utt_counterfeit:
                            utt_counterfeit = utt_counterfeit_sub
                            break

                    for w in ['-friendly', ' friendly']:
                        utt_counterfeit = re.sub(r'\b{}\b'.format(w), ' supporting', utt_counterfeit)

                    utt_counterfeit = re.sub(r'\ballow', 'offer', utt_counterfeit)
                elif slot_orig == 'area':
                    # If value mentioned in the MR verbatim, replace with a sampled value from the target domain
                    if value_orig in utt_counterfeit:
                        value_counterfeit = random.choice(slot_value_dict[slot_counterfeit])
                        if ',' in value_counterfeit:
                            value_items = [val.strip() for val in value_counterfeit.split(',')]
                            value_items_shuffled = random.sample(value_items, len(value_items))
                            value_realization = ', '.join(value_items_shuffled[:-1]) + ' and ' + value_items_shuffled[-1]
                        else:
                            value_realization = value_counterfeit
                        utt_counterfeit = re.sub(value_orig, value_realization, utt_counterfeit)

                    # Replace related keywords/phrases with alternatives matching the target domain
                    if slot_counterfeit == 'platforms':
                        phrase_counterfeit = random.choice(['available for', 'available on', 'released for', 'released on'])
                    else:
                        phrase_counterfeit = ''

                    for w in ['located in']:
                        utt_counterfeit_sub = re.sub(r'\b{}\b'.format(w), phrase_counterfeit, utt_counterfeit)
                        if utt_counterfeit_sub != utt_counterfeit:
                            utt_counterfeit = utt_counterfeit_sub
                            break

                    for w in ['area']:
                        phrase_counterfeit = 'platform' + ('s' if ',' in value_counterfeit else '')
                        utt_counterfeit = re.sub(r'\b{}\b'.format(w), phrase_counterfeit, utt_counterfeit)
                elif slot_orig == 'eattype':
                    # If value mentioned in the MR verbatim, replace with a sampled value from the target domain
                    if value_orig in utt_counterfeit:
                        value_counterfeit = random.choice(slot_value_dict[slot_counterfeit])
                        if ',' in value_counterfeit:
                            value_items = [val.strip() for val in value_counterfeit.split(',')]
                            value_items_shuffled = random.sample(value_items, len(value_items))
                            value_realization = ' '.join(value_items_shuffled) + ' game'
                        else:
                            value_realization = value_counterfeit + ' game'
                        utt_counterfeit = re.sub(value_orig, value_realization, utt_counterfeit)
                elif slot_orig == 'near':
                    if slot_counterfeit == 'developer':
                        phrase_counterfeit = random.choice(['developed by', 'made by'])
                    else:
                        phrase_counterfeit = ''

                    for w in ['located near', 'situated by']:
                        utt_counterfeit_sub = re.sub(r'\b{}\b'.format(w), phrase_counterfeit, utt_counterfeit)
                        if utt_counterfeit_sub != utt_counterfeit:
                            utt_counterfeit = utt_counterfeit_sub
                            break

                    utt_counterfeit = re.sub(r'\bnear\b', random.choice(['by', 'from']), utt_counterfeit)

            mr_counterfeit[slot_counterfeit] = value_counterfeit

        # Replace general keywords/phrases with alternatives matching the target domain
        for w in ['place', 'venue', 'establishment', 'eatery', 'restaurant']:
            utt_counterfeit = re.sub(r'\b{}\b'.format(w), 'game', utt_counterfeit)
        utt_counterfeit = re.sub(r'\bnear\b'.format(w), 'for', utt_counterfeit)
    elif target_dataset == 'hotel':
        for slot_orig, value_orig in mr.items():
            slot_counterfeit = slot_orig
            value_counterfeit = value_orig

            if slot_orig.rstrip(string.digits) in slots_to_replace:
                # Substitute the slot with the corresponding slot from the target domain
                slot_counterfeit = e2e_slot_to_hotel_slot(slot_orig)
                while slot_counterfeit in mr_counterfeit:
                    slot_counterfeit = e2e_slot_to_hotel_slot(slot_orig)

                if slot_orig == 'familyfriendly':
                    if slot_counterfeit == 'acceptscreditcards':
                        phrase_counterfeit = 'credit card'
                    elif slot_counterfeit == 'dogsallowed':
                        phrase_counterfeit = 'dog'
                    elif slot_counterfeit == 'hasinternet':
                        phrase_counterfeit = 'internet'
                    else:
                        phrase_counterfeit = ''

                    for w in ['families', 'children', 'kids']:
                        utt_counterfeit = re.sub(r'\b{}\b'.format(w),
                                                 phrase_counterfeit + 's' if phrase_counterfeit != 'internet' else phrase_counterfeit,
                                                 utt)
                        if utt_counterfeit != utt:
                            break
                    if utt_counterfeit == utt:
                        for w in ['family', 'child', 'kid']:
                            utt_counterfeit_sub = re.sub(r'\b{}\b'.format(w), phrase_counterfeit, utt_counterfeit)
                            if utt_counterfeit_sub != utt_counterfeit:
                                utt_counterfeit = utt_counterfeit_sub
                                break
                elif slot_orig == 'customerrating' or slot_orig == 'food':
                    if slot_counterfeit == 'address':
                        phrase_counterfeit = 'address'
                    elif slot_counterfeit == 'phone':
                        phrase_counterfeit = 'phone number'
                    elif slot_counterfeit == 'postcode':
                        phrase_counterfeit = 'postcode'
                    else:
                        phrase_counterfeit = ''

                    if slot_orig == 'customerrating':
                        for w in ['customer rating of', 'customer ratings', 'customer rating', 'ratings', 'rating']:
                            utt_counterfeit_sub = re.sub(r'\b{}\b'.format(w), phrase_counterfeit, utt_counterfeit)
                            if utt_counterfeit_sub != utt_counterfeit:
                                utt_counterfeit = utt_counterfeit_sub
                                break
                    elif slot_orig == 'food':
                        utt_counterfeit = re.sub(r'\b{}\b'.format('food'), phrase_counterfeit, utt_counterfeit)
                else:
                    raise AttributeError('provided domain does not exist')

            mr_counterfeit[slot_counterfeit] = value_counterfeit

    return mr_counterfeit, utt_counterfeit


def create_placeholder(slot, value):
    """Assemble a placeholder token for the given slot value."""

    vowels = 'aeiou'

    placeholder = config.DELEX_PREFIX
    value = value.lower()

    if value[0] in vowels:
        placeholder += 'vow_'
    else:
        placeholder += 'con_'

    if slot in ['name', 'developer']:
        if value.startswith(('the ', 'a ', 'an ')):
            placeholder += 'det_'
    elif slot == 'food':
        if 'food' not in value:
            placeholder += 'cuisine_'

    placeholder += (slot + config.DELEX_SUFFIX)

    return placeholder


def e2e_slot_to_hotel_slot(slot):
    """Map an E2E slot onto a slot in the Hotel domain. If there are multiple tokens in the corresponding category
    in the Hotel domain, randomly pick one from that category.
    """

    slot_map = {
        'food': ['address', 'phone', 'postcode'],
        'customerrating': ['address', 'phone', 'postcode'],
        'familyfriendly': ['acceptscreditcards', 'dogsallowed', 'hasinternet'],
        'eattype': ['type']
    }

    if slot in slot_map:
        if len(slot_map[slot]) == 1:
            return slot_map[slot][0]
        else:
            return random.choice(slot_map[slot])
    else:
        return slot


def e2e_slot_to_video_game_slot(slot):
    """Map an E2E slot onto a slot in the Video Game domain. If there are multiple tokens in the corresponding category
    in the Video Game domain, randomly pick one from that category.
    """

    slot_map = {
        'food': ['releaseyear', 'expreleasedate'],      # delexed
        'customerrating': ['rating', 'esrb'],
        'pricerange': ['playerperspective'],
        'familyfriendly': ['hasmultiplayer', 'availableonsteam', 'haslinuxrelease', 'hasmacrelease'],   # boolean
        'area': ['platforms'],
        'eattype': ['genres'],
        'near': ['developer']       # delexed
    }

    if slot in slot_map:
        if len(slot_map[slot]) == 1:
            return slot_map[slot][0]
        else:
            return random.choice(slot_map[slot])
    else:
        return slot


def token_seq_to_idx_seq(token_seqences, token2idx, max_output_seq_len):
    # produce sequences of indexes from the utterances in the training set
    idx_sequences = np.zeros((len(token_seqences), max_output_seq_len), dtype=np.int32)       # padding implicitly present, as the index of the padding token is 0
    for i, token_seq in enumerate(token_seqences):
        for j, token in enumerate(token_seq):
            # truncate long utterances
            if j >= max_output_seq_len:
                break

            # represent each token with the corresponding index
            if token in token2idx:
                idx_sequences[i][j] = token2idx[token]
            else:
                idx_sequences[i][j] = token2idx['<NA>']

    return idx_sequences


# ---- SCRIPTS ----

def count_unique_mrs(dataset, filename):
    """Counts unique MRs in the datasets and prints the statistics. (Requires the initial comment blocks in
    the TV and Laptop data files to be manually removed first.)
    """

    if filename.lower().endswith('.csv'):
        df = pd.read_csv(os.path.join(config.DATA_DIR, dataset, filename), header=0, encoding='utf8')
    elif filename.lower().endswith('.json'):
        df = pd.read_json(os.path.join(config.DATA_DIR, dataset, filename), encoding='utf8')
    else:
        raise ValueError('Unexpected file type. Please provide a CSV or a JSON file as input.')

    # Remove delexicalized placeholders, if present
    df.iloc[:, 0] = df.iloc[:, 0].replace(r'__.*?__', '', regex=True)

    print('Unique MRs (' + dataset + ' -> ' + filename + '):  ', end='')
    print(len(df.iloc[:, 0].unique()), '/', len(df.iloc[:, 0]))


def count_mr_overlap(dataset, filename1, filename2):
    """Counts unique MRs in the datasets and prints the statistics. (Requires the initial comment blocks in
    the TV and Laptop data files to be manually removed first.)
    """

    if filename1.lower().endswith('.csv') and filename2.lower().endswith('.csv'):
        df1 = pd.read_csv(os.path.join(config.DATA_DIR, dataset, filename1), header=0, encoding='utf8')
        df2 = pd.read_csv(os.path.join(config.DATA_DIR, dataset, filename2), header=0, encoding='utf8')
    elif filename1.lower().endswith('.json') and filename2.lower().endswith('.json'):
        df1 = pd.read_json(os.path.join(config.DATA_DIR, dataset, filename1), encoding='utf8')
        df2 = pd.read_json(os.path.join(config.DATA_DIR, dataset, filename2), encoding='utf8')
    else:
        raise ValueError('Unexpected file type. Please provide two CSV or two JSON files as input.')

    # Remove delexicalized placeholders, if present
    df1.iloc[:, 0] = df1.iloc[:, 0].replace(r'__.*?__', '', regex=True)
    df2.iloc[:, 0] = df2.iloc[:, 0].replace(r'__.*?__', '', regex=True)

    # Identify the samples whose MR matches one in the other file
    df1_overlap = df1[df1.mr.isin(df2.mr)]
    df2_overlap = df2[df2.mr.isin(df1.mr)]

    print('Overlapping MRs (' + dataset + '):')
    print('-> ' + filename1 + ':\t' + str(len(df1_overlap)) + ' out of ' + str(len(df1)))
    print('-> ' + filename2 + ':\t' + str(len(df2_overlap)) + ' out of ' + str(len(df2)))
    print()


def verify_slot_order(dataset, filename):
    """Verifies whether the slot order in all MRs corresponds to the desired order.
    """

    slots_ordered = ['name', 'eattype', 'food', 'pricerange', 'customerrating', 'area', 'familyfriendly', 'near']
    mrs_dicts = []

    # Read in the data
    data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, _, _, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value_orig

        mrs_dicts.append(mr_dict)

    for mr_dict in mrs_dicts:
        slots = list(mr_dict.keys())
        cur_idx = 0

        for slot in slots:
            if slot in slots_ordered:
                slot_idx = slots.index(slot)
                rightmost_idx = slots_ordered.index(slot)

                if slot_idx <= rightmost_idx and rightmost_idx >= cur_idx:
                    cur_idx = rightmost_idx
                else:
                    print('TEST FAILED: {0} has index {1} in the MR, but the order requires index {2}.'.format(
                        slot, slot_idx, slots_ordered.index(slot)))


def filter_samples_by_da_type_csv(dataset, filename, das_to_keep):
    """Create a new CSV data file by filtering only those samples in the given dataset that contain an MR
    with one of the desired DA types.
    """

    if not filename.lower().endswith('.csv'):
        raise ValueError('Unexpected file type. Please provide a CSV file as input.')

    data_filtered = []

    # Read in the data
    data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Append the opening parenthesis to the DA names, so as to avoid matching DAs whose names have these as prefixes
    das_to_keep = tuple(da + '(' for da in das_to_keep)

    # Filter MRs with the desired DA types only
    for mr, utt in zip(mrs, utterances):
        if mr.startswith(das_to_keep):
            data_filtered.append([mr, utt])

    # Save the filtered dataset to a new file
    filename_out = os.path.splitext(filename)[0] + ' [filtered].csv'
    pd.DataFrame(data_filtered).to_csv(os.path.join(config.DATA_DIR, dataset, filename_out),
                                       header=['mr', 'ref'],
                                       index=False,
                                       encoding='utf8')


def filter_samples_by_da_type_json(dataset, filename, das_to_keep):
    """Create a new JSON data file by filtering only those samples in the given dataset that contain an MR
    with one of the desired DA types.
    """

    if not filename.lower().endswith('.json'):
        raise ValueError('Unexpected file type. Please provide a JSON file as input.')

    data_filtered = []

    with io.open(os.path.join(config.DATA_DIR, dataset, filename), encoding='utf8') as f_dataset:
        # Skip and store the comment at the beginning of the file
        f_dataset, comment_block = skip_comment_block(f_dataset, '#')

        # Read the dataset from file
        data = json.load(f_dataset, encoding='utf8')

    # Append the opening parenthesis to the DA names, so as to avoid matching DAs whose names have these as prefixes
    das_to_keep = tuple(da + '(' for da in das_to_keep)

    # Filter MRs with the desired DA types only
    for sample in data:
        mr = sample[0]
        if mr.startswith(das_to_keep):
            data_filtered.append(sample)

    # Save the filtered dataset to a new file
        filename_out = os.path.splitext(filename)[0] + ' [filtered].json'
    with io.open(os.path.join(config.DATA_DIR, dataset, filename_out), 'w', encoding='utf8') as f_dataset_filtered:
        f_dataset_filtered.write(comment_block)
        json.dump(data_filtered, f_dataset_filtered, indent=4, ensure_ascii=False)


def filter_samples_by_slot_count_csv(dataset, filename, min_count=None, max_count=None, eliminate_position_slot=True):
    """Create a new CSV data file by filtering only those samples in the given dataset that contain an MR
    with the number of slots in the desired range.
    """

    if not filename.lower().endswith('.csv'):
        raise ValueError('Unexpected file type. Please provide a CSV file as input.')

    data_filtered = []

    # Read in the data
    data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs, utterances = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    for mr, utt in zip(mrs, utterances):
        mr_dict = OrderedDict()
        cur_min_count = min_count or 0
        cur_max_count = max_count or 20

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            _, _, slot_orig, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot_orig] = value_orig

        if 'da' in mr_dict:
            cur_min_count += 1
            cur_max_count += 1
        if 'position' in mr_dict:
            if eliminate_position_slot:
                if mr_dict['position'] == 'inner':
                    continue
                elif mr_dict['position'] == 'outer':
                    mr = mr.replace(', position[outer]', '')
            cur_min_count += 1
            cur_max_count += 1

        if min_count is not None and len(mr_dict) < cur_min_count or \
                max_count is not None and len(mr_dict) > cur_max_count:
            continue

        data_filtered.append([mr, utt])

    # Save the filtered dataset to a new file
    filename_out = ''.join(filename.split('.')[:-1])
    if min_count is not None:
        filename_out += '_min{}'.format(min_count)
    if max_count is not None:
        filename_out += '_max{}'.format(max_count)
    filename_out += '_slots.csv'

    pd.DataFrame(data_filtered).to_csv(os.path.join(config.DATA_DIR, dataset, filename_out),
                                       header=['mr', 'ref'],
                                       index=False,
                                       encoding='utf8')


def filter_samples_by_slot_count_json(dataset, filename, min_count=None, max_count=None, eliminate_position_slot=True):
    """Create a new JSON data file by filtering only those samples in the given dataset that contain an MR
    with the number of slots in the desired range.
    """

    if not filename.lower().endswith('.json'):
        raise ValueError('Unexpected file type. Please provide a JSON file as input.')

    data_filtered = []

    with io.open(os.path.join(config.DATA_DIR, dataset, filename), encoding='utf8') as f_dataset:
        # Skip and store the comment at the beginning of the file
        f_dataset, comment_block = skip_comment_block(f_dataset, '#')

        # Read the dataset from file
        data = json.load(f_dataset, encoding='utf8')

    data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Filter MRs with a number of slots in the desired range only
    for sample in data:
        mr = sample[0]

        mr_dict = OrderedDict()
        cur_min_count = min_count or 0
        cur_max_count = max_count or 20

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            _, _, slot_orig, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot_orig] = value_orig

        if 'da' in mr_dict:
            cur_min_count += 1
            cur_max_count += 1
        if 'position' in mr_dict:
            if eliminate_position_slot:
                if mr_dict['position'] == 'inner':
                    continue
                elif mr_dict['position'] == 'outer':
                    mr = mr.replace(';position=outer', '')
            cur_min_count += 1
            cur_max_count += 1

        if min_count is not None and len(mr_dict) < cur_min_count or \
                max_count is not None and len(mr_dict) > cur_max_count:
            continue

        data_filtered.append([mr, sample[1], sample[2]])

    # Save the filtered dataset to a new file
    filename_out = ''.join(filename.split('.')[:-1])
    if min_count is not None:
        filename_out += '_min{}'.format(min_count)
    if max_count is not None:
        filename_out += '_max{}'.format(max_count)
    filename_out += '_slots.json'

    with io.open(os.path.join(config.DATA_DIR, dataset, filename_out), 'w', encoding='utf8') as f_dataset_filtered:
        f_dataset_filtered.write(comment_block)
        json.dump(data_filtered, f_dataset_filtered, indent=4, ensure_ascii=False)


def counterfeit_dataset_from_e2e(filename, target_dataset, out_type='csv', slot_value_dict_path=None):
    """Creates a counterfeit target dataset from the E2E restaurant dataset by mapping the E2E slots onto similar
    slots in the target domain. Boolean slots are handled by heuristically replacing the corresponding mention
    in the reference utterance to reflect the slot from the target domain that replaced the original E2E one.
    The counterfeit dataset is stored in a JSON format.
    """

    source_slots = ['name', 'eattype', 'food', 'pricerange', 'customerrating', 'area', 'familyfriendly', 'near']

    data_counterfeit = []
    data_out = []

    # Read in the data
    data_cont = init_test_data(os.path.join(config.E2E_DATA_DIR, filename))
    mrs, utterances = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the utterances
    utterances = [preprocess_utterance(utt) for utt in utterances]

    if slot_value_dict_path is not None:
        with open(slot_value_dict_path, 'r', encoding='utf8') as f_slot_values:
            slot_value_dict = json.load(f_slot_values)
    else:
        slot_value_dict = None

    for mr, utt in zip(mrs, utterances):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value

        # Delexicalize the MR and the utterance
        data_counterfeit.append(counterfeit_sample(mr_dict, utt,
                                                   target_dataset=target_dataset,
                                                   slots_to_replace=source_slots,
                                                   slot_value_dict=slot_value_dict))

    if target_dataset in ['video_game']:
        for mr, utt in data_counterfeit:
            mr_str = mr_to_string(mr, da='inform')
            data_out.append([mr_str, utt])
    elif target_dataset in ['laptop', 'tv', 'hotel']:
        for mr, utt in data_counterfeit:
            mr_str = 'inform('
            for slot, val in mr.items():
                mr_str += slot + '=\'' + val + '\';'
            mr_str = mr_str[:-1] + ')'

            data_out.append([mr_str, utt, utt])

    # Save the counterfeit dataset to a new file
    if out_type == 'csv':
        filename_out = os.path.splitext(filename)[0] + ' [counterfeit {}].csv'.format(target_dataset)
        df_out = pd.DataFrame(data_out, columns=['mr', 'ref'])
        df_out.to_csv(os.path.join(config.E2E_DATA_DIR, filename_out), index=False, encoding='utf8')
    elif out_type == 'json':
        filename_out = os.path.splitext(filename)[0] + ' [counterfeit {}].json'.format(target_dataset)
        with io.open(os.path.join(config.E2E_DATA_DIR, filename_out), 'w', encoding='utf8') as f_dataset_counterfeit:
            json.dump(data_out, f_dataset_counterfeit, indent=4, ensure_ascii=False)


def get_vocab_overlap(dataset1, filename_train1, filename_dev1, dataset2, filename_train2, filename_dev2):
    """Calculates the word overlap between the vocabularies of two datasets.
    """

    data_trainset1 = os.path.join(config.DATA_DIR, dataset1, filename_train1)
    data_devset1 = os.path.join(config.DATA_DIR, dataset1, filename_dev1)
    data_trainset2 = os.path.join(config.DATA_DIR, dataset2, filename_train2)
    data_devset2 = os.path.join(config.DATA_DIR, dataset2, filename_dev2)

    dataset1 = load_training_data(data_trainset1, data_devset1)
    dataset2 = load_training_data(data_trainset2, data_devset2)

    vocab1 = get_vocabulary(dataset1)
    vocab2 = get_vocabulary(dataset2)

    common_vocab = vocab1.intersection(vocab2)

    print('Size of vocab 1:', len(vocab1))
    print('Size of vocab 2:', len(vocab2))
    print('Number of common words:', len(common_vocab))

    print('Common words:')
    print(common_vocab)


def pool_slot_values(dataset, filenames):
    """Gathers all possible values for each slot type in the dataset.
    """

    # slots_to_pool = ['eattype', 'pricerange', 'customerrating', 'familyfriendly']
    slots_to_pool = None
    slot_poss_values = {}

    # Read in the data
    if len(filenames) == 1:
        data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filenames[0]))
        mrs, utterances = data_cont['data']
    else:
        data_cont = init_training_data(os.path.join(config.DATA_DIR, dataset, filenames[0]),
                                       os.path.join(config.DATA_DIR, dataset, filenames[1]))
        x_train, y_train, x_dev, y_dev = data_cont['data']
        mrs, utterances = (x_train + x_dev), (y_train + y_dev)

    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs
    mrs = [preprocess_mr(mr, data_cont['separators']) for mr in mrs]

    for i, mr in enumerate(mrs):
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, _, _, value_orig = parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value_orig

        # For each slot gather all possible values
        for slot, value in mr_dict.items():
            slot = slot.rstrip(string.digits)
            if slots_to_pool is None or slot in slots_to_pool:
                if slot not in slot_poss_values:
                    slot_poss_values[slot] = set()
                if len(value) > 0:
                    slot_poss_values[slot].add(value)

    # Convert the value sets to lists (and make thus the dictionary serializable into JSON)
    for slot in slot_poss_values.keys():
        slot_poss_values[slot] = sorted(list(slot_poss_values[slot]))

    # Store the dictionary to a file
    with open(os.path.join(config.DATA_DIR, dataset, 'slot_values.json'), 'w', encoding='utf8') as f_slot_values:
        json.dump(slot_poss_values, f_slot_values, indent=4, sort_keys=True, ensure_ascii=False)


def generate_joint_vocab():
    """Generates a joint vocabulary for multiple datasets.
    """

    data_trainset = os.path.join(config.VIDEO_GAME_DATA_DIR, 'train.csv')
    data_devset = os.path.join(config.VIDEO_GAME_DATA_DIR, 'valid.csv')
    data_video_game = load_training_data(data_trainset, data_devset, skip_if_exist=False)

    # data_trainset = os.path.join(config.HOTEL_DATA_DIR, 'train.json')
    # data_devset = os.path.join(config.HOTEL_DATA_DIR, 'valid.json')
    # data_hotel = load_training_data(data_trainset, data_devset, skip_if_exist=False)
    #
    # data_trainset = os.path.join(config.LAPTOP_DATA_DIR, 'train.json')
    # data_devset = os.path.join(config.LAPTOP_DATA_DIR, 'valid.json')
    # data_laptop = load_training_data(data_trainset, data_devset, skip_if_exist=False)
    #
    # data_trainset = os.path.join(config.TV_DATA_DIR, 'train.json')
    # data_devset = os.path.join(config.TV_DATA_DIR, 'valid.json')
    # data_tv = load_training_data(data_trainset, data_devset, skip_if_exist=False)

    data_trainset = os.path.join(config.E2E_DATA_DIR, 'trainset_e2e [denoised] [counterfeit video_game].csv')
    data_devset = os.path.join(config.E2E_DATA_DIR, 'devset_e2e [denoised] [counterfeit video_game].csv')
    data_rest = load_training_data(data_trainset, data_devset, skip_if_exist=False)

    # generate_vocab_file(np.concatenate((data_rest, data_tv, data_laptop, data_hotel, data_video_game)),
    generate_vocab_file(np.concatenate((data_rest, data_video_game)),
                        vocab_filename='vocab.lang_gen.tokens')


def augment_mrs_with_da_type(dataset, filename, da_type):
    # Read in the data
    df = pd.read_csv(os.path.join(config.DATA_DIR, dataset, filename), header=0, encoding='utf8')
    mrs = df.mr.tolist()

    df['mr'] = [add_da_info_to_mr(mr, da_type) for mr in mrs]

    filename_out = os.path.splitext(filename)[0] + ' [with DA].csv'
    df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


def add_da_info_to_mr(mr, da_type):
    return da_type + '(' + mr + ')'


def delex_dataset(dataset, files, slots_to_delex=None, mr_only=False):

    if not isinstance(files, list):
        files = [str(files)]

    for filename in files:
        # Read in the data
        data_cont = init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
        dataset_name = data_cont['dataset_name']
        mrs_orig, utterances_orig = data_cont['data']
        _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

        # Preprocess the MRs and utterances
        mrs = [preprocess_mr(mr, data_cont['separators']) for mr in mrs_orig]
        utterances = [preprocess_utterance(utt) for utt in utterances_orig]

        # Produce sequences of extracted words from the meaning representations (MRs) in the testset
        mrs_delex = []
        utterances_delex = []
        for i, mr in enumerate(mrs):
            mr_dict = OrderedDict()

            # Extract the slot-value pairs into a dictionary
            for slot_value in mr.split(slot_sep):
                slot, value, _, _ = parse_slot_and_value(slot_value, val_sep, val_sep_end)

                mr_dict[slot] = value

            # Delexicalize the MR

            if mr_only:
                delex_sample(mr_dict, utterance=utterances[i], dataset=dataset_name, mr_only=True,
                             slots_to_delex=slots_to_delex)
            else:
                utterances_delex.append(delex_sample(mr_dict, utterance=utterances[i], dataset=dataset_name,
                                                     slots_to_delex=slots_to_delex))

            mrs_delex.append(mr_to_string(mr_dict))

        new_df = pd.DataFrame(columns=['mr', 'ref'])
        new_df['mr'] = mrs_delex
        if mr_only:
            new_df['ref'] = utterances_orig
        else:
            new_df['ref'] = utterances_delex

        suffix = ' [delex' + (', MR only' if mr_only else '') + ']'
        filename_out = os.path.splitext(filename)[0] + suffix + os.path.splitext(filename)[1]
        file_out = os.path.join(config.DATA_DIR, dataset, filename_out)

        new_df.to_csv(file_out, index=False, encoding='utf8')


def mr_to_string(mr_dict, da=None):
    """Convert an MR represented by a dictionary to a flat textual form. The input MR is expected to be an OrderedDict
    of slots and values.
    """

    slot_value_pairs = []

    # If there is a "da" slot in the MR dictionary, pop it and use its value to indicate the DA type of the MR
    if 'da' in mr_dict:
        if da is None:
            da = mr_dict.pop('da', None)
        else:
            assert mr_dict['da'] == da
            mr_dict.pop('da', None)

    # Format the slot-value pairs
    for slot, val in mr_dict.items():
        slot_value_pairs.append(slot + '[{0}]'.format(str(val.strip()) if val is not None else ''))

    # Concatenate the formatted slot-value pairs to form a textual MR
    mr = ', '.join(slot_value_pairs)

    if da is not None:
        # Prepend the DA, and enclose the list of the MR's slot-value pairs in parentheses
        mr = da + '(' + mr + ')'

    return mr


# ---- MAIN ----

def main():
    # count_unique_mrs('rest_e2e', 'trainset_e2e [delex, MR only].csv')
    # count_unique_mrs('rest_e2e', 'devset_e2e [delex, MR only].csv')
    # count_unique_mrs('rest_e2e', 'testset_e2e [delex, MR only].csv')

    # count_unique_mrs('video_game', 'train [delex, MR only].csv')
    # count_unique_mrs('video_game', 'valid [delex, MR only].csv')
    # count_unique_mrs('video_game', 'test [delex, MR only].csv')

    # ----------

    # count_mr_overlap('rest_e2e', 'trainset_e2e.csv', 'devset_e2e.csv')
    # count_mr_overlap('rest_e2e', 'trainset_e2e.csv', 'testset_e2e.csv')
    # count_mr_overlap('rest_e2e', 'devset_e2e.csv', 'testset_e2e.csv')

    # count_mr_overlap('video_game', 'train [delex, MR only].csv', 'valid [delex, MR only].csv')
    # count_mr_overlap('video_game', 'train [delex, MR only].csv', 'test [delex, MR only].csv')
    # count_mr_overlap('video_game', 'valid [delex, MR only].csv', 'test [delex, MR only].csv')

    # ----------

    # verify_slot_order('rest_e2e', 'trainset_e2e_utt_split.csv')

    # ----------

    das_to_keep = ['inform']

    filter_samples_by_da_type_csv('video_game', 'valid.csv', das_to_keep)
    # filter_samples_by_da_type_json('tv', 'train.json', das_to_keep)

    # ----------

    # filter_samples_by_slot_count_csv('rest_e2e', 'testset_e2e.csv', min_count=3, max_count=4)
    # filter_samples_by_slot_count_json('hotel', 'test_filtered.json', min_count=3, max_count=4)

    # ----------

    # slot_value_dict_path = os.path.join(config.VIDEO_GAME_DATA_DIR, 'slot_values_train.json')

    # counterfeit_dataset_from_e2e('testset_e2e_min3_max4_slots.csv', 'hotel', format='json')
    # counterfeit_dataset_from_e2e('trainset_e2e [denoised].csv', 'video_game', out_type='csv',
    #                              slot_value_dict_path=slot_value_dict_path)

    # ----------

    # get_vocab_overlap('rest_e2e', 'trainset_e2e.csv', 'devset_e2e.csv',
    #                   'hotel', 'train.json', 'valid.json')
    # get_vocab_overlap('laptop', 'train.json', 'valid.json',
    #                   'tv', 'train.json', 'valid.json')

    # ----------

    # pool_slot_values('rest_e2e', ['trainset_e2e.csv', 'devset_e2e.csv'])
    # pool_slot_values('laptop', ['train.json', 'valid.json'])
    # pool_slot_values('video_game', ['train.csv', 'valid.csv'])

    # ----------

    # generate_joint_vocab()

    # ----------

    # augment_mrs_with_da_type('rest_e2e', 'trainset_e2e [denoised].csv', 'inform')
    # augment_mrs_with_da_type('video_game', 'dataset.csv', 'inform')

    # ----------

    # delex_dataset('rest_e2e', ['devset_e2e.csv'], slots_to_delex=['name', 'near'], mr_only=True)
    # delex_dataset('video_game', ['valid.csv'], slots_to_delex=['name', 'developer'], mr_only=True)

    # ----------

    # x_test, y_test = read_laptop_dataset_test('data/tv/test.json')
    # print(x_test)
    # print()
    # print(y_test)
    # print()
    # print(len(x_test), len(y_test))

    # ----------

    # if len(y_test) > 0:
    #    with io.open('data/predictions_baseline.txt', 'w', encoding='utf8') as f_y_test:
    #        for line in y_test:
    #            f_y_test.write(line + '\n')

    # Produce a file from the predictions in the TV/Laptop dataset format by replacing the baseline utterances (in the 3rd column)
    # with io.open('eval/predictions-tv/predictions_ensemble_2way_2.txt', 'r', encoding='utf8') as f_predictions:
    #     with io.open('data/tv/test.json', encoding='utf8') as f_testset:
    #         # Skip the comment block at the beginning of the file
    #         f_testset, _ = skip_comment_block(f_testset, '#')
    #
    #         # read the test data from file
    #         df = pd.read_json(f_testset, encoding='utf8')
    #
    #     df.iloc[:, 2] = f_predictions.readlines()
    #     df.to_json('data/tv/test_pred.json', orient='values')

    # Produce a file from the predictions in the TV/Laptop dataset format by replacing the baseline utterances (in the 3rd column)
    # with io.open('eval/predictions-laptop/predictions_ensemble_2way_1.txt', 'r', encoding='utf8') as f_predictions:
    #     with io.open('data/laptop/test.json', encoding='utf8') as f_testset:
    #         # Skip the comment block at the beginning of the file
    #         f_testset, _ = skip_comment_block(f_testset, '#')
    #
    #         # read the test data from file
    #         df = pd.read_json(f_testset, encoding='utf8')
    #
    #     df.iloc[:, 2] = f_predictions.readlines()
    #     df.to_json('data/laptop/test_pred.json', orient='values')


if __name__ == '__main__':
    main()
