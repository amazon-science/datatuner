import copy
import heapq
import json
import logging
import math
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import mlflow
import mlflow.tracking
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from datatuner.classification.consistency_classifier import ConsistencyClassifier, dataset_fields
from datatuner.lm.data_loader import get_dataset_from_file, get_inputs
from datatuner.lm.metrics import aggregate_metrics
from datatuner.lm.model_loader import get_model_directory, load_pretrained
from datatuner.lm.reranker import Reranker
from datatuner.lm.utils import (
    custom_deep_copy,
    load_task_config,
    should_ignore_in_score,
    should_stop_further_beam_exploration,
)
from datatuner.ops.mlflow import get_finished_models
from datatuner.utils import arr_part_matches_string, ewm_mean, flatten, geo_mean, get_curr_time
from tqdm import tqdm

logger = logging.getLogger(__file__)
DEBUG = logging.getLogger().getEffectiveLevel() == logging.DEBUG


def top_filtering(probs, tokenizer, top_k=0, top_p=0.0, dec_dropout=0, threshold=0, seed=42, filter_value=0):
    """ Filter a distribution of probs using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            probs: probs distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep probs
    """
    assert (
            probs.dim() == 1
    )  # TODO: Only works for batch size 1 currently
    top_k = min(top_k, probs.size(-1))

    r = random.random()

    if r < dec_dropout:
        max_ind = probs.argmax()
        decoded = tokenizer.decode([max_ind.item()])
        if decoded[0] == " " and decoded.strip().islower():
            probs[max_ind] = 0

    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
        probs[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probabilities = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = filter_value

    indices_to_remove = probs < threshold
    probs[indices_to_remove] = filter_value

    return probs


class Beam(object):
    # For comparison of prefixes, the tuple starting with (prefix_probability, complete_sentence, ...) is used.
    def __init__(self, beam_width, tokenizer):
        self.heap = list()
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.counter = 0
        self.max_ratio_in_heap = 1000

    def add(self, prob, complete, prefix, item, input_ids, token_type_ids, all_probs):
        # counter is used so that the heap does not have to compare tensors. The 3rd tuple element
        # is always unique across beams
        if DEBUG:

            decoded = self.tokenizer.decode(prefix)
            logger.debug(decoded)
            logger.debug(f"combined prob: {prob}")
            logger.debug(all_probs)
            logger.debug("\n")

            logger.debug("beam elements:")
            n_best_items = heapq.nlargest(len(self.heap), self.heap)
            for i in range(len(n_best_items)):
                text = self.tokenizer.decode(n_best_items[i][3]["prefix"])
                logger.debug(f"{n_best_items[i][0]} {text}")

            logger.debug("\n")
        payload = {
            "prefix": prefix,
            "item": item,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "all_probs": all_probs,
        }
        heapq.heappush(self.heap, (prob, complete, self.counter, payload))
        self.counter += 1

    def get_cons_rep(self, element, mr_key):
        cand = {
            "data": self.tokenizer.decode(element["item"][mr_key]),
            "text": self.tokenizer.decode(element["prefix"]),
        }
        return cand

    def clean_beam(self, cons_classifier, cons_cache, cons_dataset):
        new_heap = []

        # dataset_fields
        largest_prob = heapq.nlargest(1, self.heap)[0][0]

        if cons_classifier is not None:
            mr_key = dataset_fields[cons_dataset]["data"]
            # Create the candidates for consistency classifier
            consistency_cands = []
            beam_ind_to_cand_ind = {}
            # All beam elements are included
            for i in range(len(self.heap)):
                # Only consider complete beam elements
                if self.heap[i][1] is not True:  # and self.heap[i][0] == largest_prob:
                    continue
                # Create a map to know which beam element was included in the beam
                cand = self.get_cons_rep(self.heap[i][3], mr_key)
                # Avoid classifying what we already classified
                if str(cand) in cons_cache:
                    continue
                else:
                    beam_ind_to_cand_ind[i] = len(consistency_cands)
                    consistency_cands.append(cand)
                    # cons_cache[str(cand)] = True

            if len(consistency_cands) > 0:
                # Results format: {"preds":[...], "preds_prob":[...]}
                cons_results = cons_classifier.evaluate(consistency_cands)

            all_cons_results = []

            for i in range(len(self.heap)):
                cand = self.get_cons_rep(self.heap[i][3], mr_key)

                if str(cand) in cons_cache:
                    all_cons_results.append(cons_cache[str(cand)])
                elif i in beam_ind_to_cand_ind:
                    cand_ind = beam_ind_to_cand_ind[i]
                    cons_for_item = {
                        "pred": cons_results["preds"][cand_ind],
                        "prob": cons_results["preds_prob"][cand_ind],
                    }
                    all_cons_results.append(cons_for_item)
                    cons_cache[str(cand)] = cons_for_item
                else:
                    all_cons_results.append(None)

        for i, item in enumerate(self.heap):
            if cons_classifier is not None:
                # Only if there was a consistency result for this item
                if all_cons_results[i] is not None:
                    pred, pred_prob = all_cons_results[i]["pred"], all_cons_results[i]["prob"]
                    # Save the prediction
                    self.heap[i][3]["cons_prediction"] = {"pred": pred, "prob": pred_prob}
                    if pred in ["omission"] and pred_prob > 0.5:
                        if DEBUG:
                            logger.debug(
                                f"removed {consistency_cands[cand_ind]} as the prediction was {pred} with probability {pred_prob}"
                            )
                        continue
                    elif pred == "accurate" and pred_prob > 0.5:
                        # Add a factor to not allow correct to be removed during cleaning based on lowest probability
                        self.heap[i] = list(self.heap[i])
                        self.heap[i][0] = 1000 + self.heap[i][0]
                        self.heap[i] = tuple(self.heap[i])

            # Remove beam components with probability lower than 1/max_ratio_in_heap times the highest beam component probability
            if True or item[0] > largest_prob / self.max_ratio_in_heap:
                new_heap.append(item)
        self.heap = new_heap

        while len(self.heap) > self.beam_width:
            prob, _, _, payload = heapq.heappop(self.heap)
            if DEBUG:
                logger.debug("removing")
                logger.debug(self.tokenizer.decode(payload["prefix"]))
                logger.debug(prob)
                logger.debug(payload["all_probs"])
                logger.debug("\n")

    def __iter__(self):
        return iter(self.heap)


def sample_sequence(
        item,
        tokenizer,
        model,
        args,
        task_config,
        out_name,
        next_stop_token,
        avoided_cache,
        filtered_words=None,
        options=None,
        attentuation_factor=0,
        prev_beam=None,
        avoid_repeating=[],
        reranker=None,
        cons_classifier=None,
):
    """Generate a sequence from the given context"""
    # The next token after the sentence we want to predict should be a special token (e.g. </query>, <eos>, etc.).
    # So we stop on that token.
    # TODO: fix the hardcoded newline character
    special_tokens_ids = tokenizer.convert_tokens_to_ids([next_stop_token, "<pad>"])
    item[out_name] = []
    hid_cache = None
    targ_cache = None
    # Consistency checking cache
    cons_cache = {}

    # The previous beam is carried over in case we have previous tasks (such as entity tagging before query generation)
    if prev_beam is None:
        input_ids, token_type_ids = get_inputs(item, args.device, tokenizer, task_config)
        prev_beam = Beam(args.beam_width, tokenizer)
        prev_beam.add(1.0, False, [], custom_deep_copy(item), input_ids, token_type_ids, [])

    else:
        for i in range(len(prev_beam.heap)):
            prev_item = prev_beam.heap[i][3]["item"]

            # We fill the non-learnt fields. Some fields can be filled from external sources; so they are not learnt)
            # We keep what the beam has already populated.
            for key in item:
                if key not in prev_item:
                    prev_item[key] = item[key]

            input_ids, token_type_ids = get_inputs(prev_item, args.device, tokenizer, task_config)
            # TODO: ignore if prev_beam is not complete

            prev_beam.heap[i] = (
                prev_beam.heap[i][0],
                False,  # The beam is no more complete
                prev_beam.heap[i][2],
                {
                    "prefix": [],
                    "item": prev_item,
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "cons_prediction": {},
                    "all_probs": prev_beam.heap[i][3]["all_probs"],
                },
            )

    # Used to allow reducing repetition for tokens in `avoid_repeating`
    word_counts = defaultdict(lambda: 0)
    # Used to prevent beam elements from extending beyond max length
    beam_len = 0
    total_preds = 0

    while True:
        curr_beam = Beam(args.beam_width, tokenizer)

        # To do batch decoding at each beam step, we map from the index in the beam to the index in the model outputs
        # (complete components are not included)
        beam_map = {}
        all_input_ids = None
        all_token_type_ids = None
        for ind, x in enumerate(prev_beam):
            # if not complete
            if x[1] is False:
                beam_map[ind] = len(beam_map)
                if all_input_ids is None:
                    all_input_ids = x[3]["input_ids"]
                    all_token_type_ids = x[3]["token_type_ids"]
                else:
                    all_input_ids = torch.cat((all_input_ids, x[3]["input_ids"]))
                    all_token_type_ids = torch.cat((all_token_type_ids, x[3]["token_type_ids"]))

        # If we have model inputs to predict based on them
        if all_input_ids is not None:
            all_model_outputs = model(all_input_ids, token_type_ids=all_token_type_ids)

        for beam_ind, (prefix_prob, complete, _, payload) in enumerate(prev_beam):

            prefix, item, input_ids, token_type_ids, all_probs = (
                payload["prefix"],
                payload["item"],
                payload["input_ids"],
                payload["token_type_ids"],
                payload["all_probs"],
            )

            # No need to do anything for complete ones
            if complete:
                curr_beam.add(prefix_prob, True, prefix, item, input_ids, token_type_ids, all_probs)
            else:
                if DEBUG:
                    try:
                        logger.debug(f"input_ids: {tokenizer.decode(input_ids.cpu().numpy()[0])}")
                    except:
                        logger.debug(f"input_ids: {input_ids}")

                # prediction_scores before softmax is all_model_outputs[0]
                model_outputs_0 = all_model_outputs[0][beam_map[beam_ind]: beam_map[beam_ind] + 1, ]

                total_preds += 1
                # Higher temperature -> the softer probability distribution. Lower temperature (<1): peaked probability
                # distribution (one dominates all)
                logits = model_outputs_0[0, -1, :] / (args.temperature if args.temperature > 0 else 1)

                probs = F.softmax(logits, dim=-1)
                # TODO: the reranker model compatible with the batched beam search
                if reranker is not None and args.aux_weight > 0 and len(prefix) > 0:
                    reranker_ids, reranker_token_types, _ = reranker.create_input(prefix, item)
                    if DEBUG:
                        try:
                            logger.debug(f"reranker_ids: {reranker.tokenizer.decode(reranker_ids.cpu().numpy()[0])}")
                        except:
                            logger.debug(f"reranker_ids: {reranker_ids}")
                    reranker_model_outputs = reranker.model(reranker_ids, token_type_ids=reranker_token_types)
                    reranker_logits = reranker_model_outputs[0][0, -1, :] / args.temperature

                    reranker_probs = F.softmax(reranker_logits, dim=-1)
                    reranker_prob_len = reranker_probs.shape[0]
                    reranker_probs = F.pad(
                        input=reranker_probs, pad=(0, probs.shape[0] - reranker_prob_len), mode="constant", value=0
                    )
                    reranker_probs[reranker_prob_len:] = probs[reranker_prob_len:]

                    reranker_logits = F.pad(
                        input=reranker_logits, pad=(0, probs.shape[0] - reranker_prob_len), mode="constant", value=0
                    )
                    reranker_logits[reranker_prob_len:] = logits[reranker_prob_len:]
                    if args.reranking_mode == "average":
                        probs = args.aux_weight * reranker_probs + (1 - args.aux_weight) * probs

                    elif args.reranking_mode == "max":
                        probs = torch.max(reranker_probs, probs)

                    probs = top_filtering(
                        probs, tokenizer, top_k=args.top_k, top_p=args.top_p, dec_dropout=args.dec_dropout
                    )

                if args.cache_pointer:
                    # Get the hidden state corresponding to the last layer
                    current_hidden = all_model_outputs[2][beam_map[beam_ind]: beam_map[beam_ind] + 1, ][-1][-1][-1]

                    if hid_cache is None:
                        hid_cache = current_hidden.unsqueeze(0)
                        targ_cache = custom_deep_copy(probs.unsqueeze(0))
                    else:
                        # Compute dot products between the cached hidden states (of previous predictions) and the current hidden state
                        all_dot_prods = torch.mv(args.cache_theta * hid_cache, current_hidden)
                        # Get the softmax representing the similarity between our current state and each of the previous hidden states
                        softmaxed = F.softmax(all_dot_prods, dim=-1).unsqueeze(1)
                        # Expand softmaxed to have the shape of the targets cache. Then multiply it with the targ_cache.
                        # Doing that, each previous target distrbution will be multiplied with a factor corresponding to the similarity
                        # between the current hidden state and the corresponding previous hidden state.
                        #
                        # Finally, we perform the sum to combine the weighted previous target distributions.
                        p_cache = (softmaxed.expand_as(targ_cache) * targ_cache).sum(0).squeeze()

                        # We zero the indices of very low probabilities
                        indices_to_remove = p_cache < p_cache.max().item() / 2
                        p_cache[indices_to_remove] = 0

                        # Compute the new target probabilities
                        aux_prob = 1 - p_cache
                        aux_prob[indices_to_remove] = 0
                        # Normalize to sum to one
                        aux_prob = aux_prob / aux_prob.sum()
                        probs = (1 - args.cache_lambda) * probs + args.cache_lambda * aux_prob

                        hid_cache = torch.cat((hid_cache, custom_deep_copy(current_hidden.unsqueeze(0))), dim=0)
                        targ_cache = torch.cat((targ_cache, custom_deep_copy(probs.unsqueeze(0))), dim=0)

                # Used to prevent the same token from being predicted in multiple iterations at the same time step
                avoided_cache = []

                # Avoid completing the beam too early
                if beam_len < args.min_length:
                    avoided_cache.extend(special_tokens_ids)

                for s in range(args.per_step_predictions):
                    # Avoid repeating same token at same time step
                    probs[avoided_cache] = 0
                    if DEBUG:
                        logger.debug(f"sum_prob: {probs.sum()}")
                    if probs.sum() <= args.min_prob:
                        break

                    next_tok = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
                    topk = torch.topk(probs, 10)
                    if DEBUG:
                        logger.debug(
                            f"topk:  {list(zip(tokenizer.convert_ids_to_tokens(topk.indices.cpu().numpy()), topk.values.cpu().numpy()))}"
                        )

                    next_token_id = next_tok.item()
                    next_prob = probs[next_token_id].item()
                    avoided_cache.append(next_token_id)
                    word_counts[next_token_id] += 1
                    next_token_str = tokenizer.convert_ids_to_tokens(next_token_id)

                    # We can decrease the probability of repetition by decreasing the probability of specific tokens.
                    if arr_part_matches_string(next_token_str, avoid_repeating):
                        next_prob = next_prob / (1 + attentuation_factor * math.log(word_counts[next_token_id]))

                    # We add the last token id to the vector
                    new_input_ids = torch.cat((input_ids, next_tok.unsqueeze(0)), dim=1)
                    new_token_type_ids = torch.cat((token_type_ids, token_type_ids[:, -1:]), dim=1)

                    if next_prob <= args.min_token_prob and len(curr_beam.heap) > 0:
                        break

                    if should_ignore_in_score(prefix, tokenizer, next_token_str, next_token_id, next_prob):
                        if DEBUG:
                            logger.debug(f"ignoring: {next_token_str} {next_prob}")
                        new_all_probs = all_probs
                    else:
                        new_all_probs = all_probs + [next_prob]

                    if args.prob_window > 0:
                        part_new_all_probs = new_all_probs[::-1][: args.prob_window][::-1]
                    else:
                        part_new_all_probs = new_all_probs

                    if args.averaging == "arithmetic":
                        combined_ppl = np.mean(part_new_all_probs)
                    elif args.averaging == "geometric":
                        combined_ppl = geo_mean(part_new_all_probs)
                    elif args.averaging == "default":
                        combined_ppl = np.log(np.prod(part_new_all_probs)) / (
                                len(part_new_all_probs) ** args.beam_alpha
                        )
                    elif args.averaging == "ewm":
                        combined_ppl = ewm_mean(part_new_all_probs, alpha=args.ewm_alpha)
                    elif args.averaging == "min":
                        combined_ppl = min(part_new_all_probs)
                    else:
                        raise Exception("Unknown averaging")

                    # We deep copy as multiple branches can be created from each beam element.
                    # We don't want to multiple branches to overwrite each other.
                    new_item = custom_deep_copy(item)

                    if DEBUG:
                        logger.debug(f"next: {next_prob} {next_token_str}")

                    if next_token_id in special_tokens_ids:
                        coverage_penalty = 0
                        if args.add_coverage_penalty:
                            context_len = len(input_ids[0]) - len(prefix)
                            context = input_ids[0][:context_len].cpu().numpy()
                            input_tokens = tokenizer.convert_ids_to_tokens(context)
                            logger.debug(next_token_str)
                            layer = -1
                            att_vec = -1
                            all_attn = None
                            for attn_head in range(16):
                                # 0 is for first in batch
                                curr_vec = (
                                    all_model_outputs[att_vec][layer][beam_map[beam_ind]: beam_map[beam_ind] + 1, ][0][
                                        attn_head
                                    ]
                                        .cpu()
                                        .numpy()[context_len - 1:, :context_len]
                                        .sum(axis=0)
                                )
                                curr_vec[np.where(curr_vec > 1)] = 1
                                if all_attn is None:
                                    all_attn = curr_vec
                                else:
                                    all_attn += curr_vec

                            all_attn = np.round(all_attn, 2)
                            sum_all_attn = all_attn.sum()
                            scored_data = list(zip(all_attn, input_tokens))
                            logger.debug(scored_data)

                            scored_data = [
                                (x, y) for x, y in scored_data if ("<" in y) and (y not in ["<data>", "<text>"])
                            ]
                            if len(scored_data) > 0:
                                coverage_penalty = scored_data[-1][0]
                            else:
                                coverage_penalty = 0

                            logger.debug(sorted(scored_data, reverse=True))

                            logger.debug(f"summ_all_attn: {sum_all_attn}")
                            logger.debug(f"prefix: {tokenizer.decode(prefix)}")
                            logger.debug(f"coverage_penalty: {coverage_penalty}")

                        new_item[out_name] = prefix
                        # If next word is the end token then mark prefix as complete
                        curr_beam.add(
                            combined_ppl + coverage_penalty,
                            True,
                            custom_deep_copy(prefix),
                            new_item,
                            custom_deep_copy(new_input_ids),
                            custom_deep_copy(new_token_type_ids),
                            new_all_probs,
                        )

                    else:
                        # If next word is a non-end token then mark prefix as incomplete
                        curr_beam.add(
                            combined_ppl,
                            False,
                            prefix + [next_token_id],
                            new_item,
                            custom_deep_copy(new_input_ids),
                            custom_deep_copy(new_token_type_ids),
                            new_all_probs,
                        )

                    # If this token is a word continuation of the previous token (i.e. not new word), stop adding other
                    # tokens (we want to diversify on word starts and not on word variations). For example if expect the
                    # word "Meghan" to be predicted, and we got "Me" in the previous token, we don't want to add both
                    # "ghan" and "eghan" to the beam search.
                    # We stop on the first prediction (as it's usually the most likely)
                    # TODO: validate this strategy in languages without a space separating words
                    if should_stop_further_beam_exploration(
                            prefix, tokenizer, next_token_str, next_token_id, next_prob
                    ):
                        break

            prev_beam = curr_beam

        curr_beam.clean_beam(cons_classifier, cons_cache, args.cons_classifier)

        beam_len += 1

        n_best_items = heapq.nlargest(len(curr_beam.heap), curr_beam.heap)
        # If the top `args.min_complete_in_beam` are complete or we exceeded max beam length, we stop
        if all(x[1] is True for x in n_best_items[: args.min_complete_in_beam]) or beam_len > args.max_length:
            break

    n_best_items = heapq.nlargest(len(curr_beam.heap), curr_beam.heap)

    try:
        for i in range(len(n_best_items)):

            if n_best_items[i][1] is True:

                text = tokenizer.decode(n_best_items[i][3]["prefix"])

    except:
        pass
    # Return the best items' decoded text
    n_best_items = [x[3]["prefix"] for x in n_best_items]

    return n_best_items, prev_beam


result_cache = {}


def process_one_item(
        item,
        tokenizer,
        model,
        task_config,
        args,
        metrics_results=None,
        metrics_fields=[],
        input_cache=None,
        avoided_cache=None,
        reranker=None,
        cons_classifier=None,
):
    """Process one item during evaluation, either from file or form user's input"""
    matching = True
    full_data_shape = copy.deepcopy(task_config["data_shape"])
    current_task_config = {"data_shape": []}
    prev_beam = None
    i = 0

    new_item = {}

    with torch.no_grad():
        key = None

        while i < len(full_data_shape):
            current_data_item = full_data_shape[i]
            key = current_data_item["id"]
            # If this is a field we learnt, we do not load. We break the loop and generate it
            if current_data_item["learn"] is True and current_data_item["type"] == "text":
                # Here we generate any field that is learnt
                try:
                    next_stop_token = full_data_shape[i + 1]["id"]
                except:
                    next_stop_token = "<eos>"

                # Save the original for comparison
                try:
                    if item[key] and type(item[key][0]) == list:
                        item[key] = item[key][-1]
                    new_item[f"original_{key}"] = tokenizer.decode(item[key])
                except:
                    pass

                # Generate the field value
                search_key = str([item[x] for x in item if x != key])
                if search_key in result_cache:
                    n_best_items, prev_beam = result_cache[search_key]
                else:
                    n_best_items, prev_beam = sample_sequence(
                        item,
                        tokenizer,
                        model,
                        args,
                        current_task_config,
                        key,
                        next_stop_token,
                        avoided_cache,
                        prev_beam=prev_beam,
                        reranker=reranker,
                        cons_classifier=cons_classifier,
                    )
                    result_cache[search_key] = (n_best_items, prev_beam)

                try:
                    new_item[key] = [
                        tokenizer.decode(n_best_items[i], skip_special_tokens=True) for i in range(len(n_best_items))
                    ]

                    if reranker is not None and args.nbest > 1 and not args.no_postranking:
                        n_best_items_reranked = reranker.rerank(n_best_items, item)
                        new_item["reranked_" + key] = [
                            tokenizer.decode(n_best_items_reranked[i], skip_special_tokens=True)
                            for i in range(len(n_best_items_reranked))
                        ]
                except:
                    raise
                    new_item[key] = ""

                if f"original_{key}" in new_item:
                    if new_item[key] != new_item[f"original_{key}"]:
                        matching = False
            else:
                current_task_config["data_shape"] += [current_data_item]

                if current_data_item["type"] == "text":
                    new_item[key] = tokenizer.decode(item[key])

            i += 1

    return new_item, matching


@st.cache(show_spinner=False, allow_output_mutation=True, persist=False)
def setup(args):
    """Setup the models and tokenizer and return them"""
    out_folder = None

    assert args.model_checkpoint
    model_directory, is_local = get_model_directory(args.model_checkpoint)
    if not args.input:

        if not args.out_folder:

            out_folder = Path(f"eval_results/{get_curr_time()}")

        else:
            out_folder = Path(args.out_folder)

        out_folder.mkdir(parents=True, exist_ok=True)

        EVAL_ARGS_FILE = out_folder / "eval_args.json"
        json.dump(vars(args), open(EVAL_ARGS_FILE, "w"), indent=2)

    if not is_local:
        mlflow.start_run(args.model_checkpoint, nested=True)

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, tokenizer = load_pretrained(model_directory, model_type=args.model_type, smoothing=args.smoothing)

    model.to(args.device)
    model.eval()

    task_config = load_task_config(args.task_config or (model_directory / "task_config.json"))
    learned_fields = [x["id"] for x in task_config["data_shape"] if x["learn"] is True and x["type"] != "special"]
    input_text_fields = [x["id"] for x in task_config["data_shape"] if x["learn"] is False and x["type"] == "text"]

    if args.reranker is not None:
        reranker_model_directory, is_local = get_model_directory(args.reranker)
        reranker = Reranker(reranker_model_directory, args.device, is_local=is_local)
    else:
        reranker = None

    if args.cons_classifier:
        dataset = args.cons_classifier
        cons_classifier = ConsistencyClassifier(
            {
                "model_name_or_path": f"/home/ec2-user/{dataset}_consistency_roberta-large_lower",
                "model_type": "roberta",
                "model_name": "roberta-large",
                "task_name": "mnli",
                "data_dir": f"/home/ec2-user/DataTuner/data/{dataset}_consistency/",
                "output_dir": "tmp",
                "no_cuda": True,
                "overwrite_cache": True,
                "do_lower_case": True,
            }
        )
    else:
        cons_classifier = None

    return (
        model,
        tokenizer,
        task_config,
        learned_fields,
        input_text_fields,
        reranker,
        out_folder,
        is_local,
        cons_classifier,
    )


@st.cache(show_spinner=False, allow_output_mutation=True, persist=False)
def load_test_data(filename="/home/ec2-user/DataTuner/data/ldc/test.json", max_items=100):
    """Load test data from file"""
    logger.info("loading test data")
    examples = json.load(open(filename))[:max_items]
    return examples


def get_run_info(client, model_checkpoint):
    """Get run info from mlflow as a dictionary; used to display the info in the streamlit interface"""
    if not model_checkpoint:
        return {"Model": "None"}
    elif model_checkpoint.startswith("/home"):
        return {"Model": Path(model_checkpoint).name}
    else:
        run_info = client.get_run(model_checkpoint)
        params = run_info.data.params
        dataset = Path(params["dataset_path"]).name
        return {
            "PT Model": Path(params["model_checkpoint"]).name,
            "Dataset": Path(dataset).name,
            "LR": params["lr"],
            "Task": Path(params["task_config"]).stem,
            "run_id": run_info.info.run_id,
        }


def run():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None, help="gpt or gpt2")
    parser.add_argument("--input", dest="input", action="store_true")
    parser.add_argument("--cache_pointer", dest="cache_pointer", action="store_true")
    parser.add_argument("--cache_theta", type=float, default=0.0001, help="factor used in the cache pointer mechanism")
    parser.add_argument(
        "--cache_lambda", type=float, default=0, help="weight of the cache probs when cache_pointer is used"
    )
    parser.add_argument(
        "--boost_factor", type=float, default=1, help="weight of the cache probs when cache_pointer is used"
    )
    parser.add_argument(
        "--ignore_existing",
        dest="ignore_existing",
        action="store_true",
        help="ignore previous runs, overwrite the test output files, and start from scratch",
    )

    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--reranking_mode", type=str, default="average", help="Reranking mode")
    parser.add_argument("--out_folder", type=str, default="", help="subfolder name of the eval results folder")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    parser.add_argument("--task_config", type=str, help="Path to the tokenization config file", default=None)

    parser.add_argument("--filename", type=str, default="data/instances_dev.pkl", help="File to use for decoding")
    parser.add_argument("--reranker", type=str, default=None, help="model used for reranking (in question answering)")
    parser.add_argument("--no_sample", action="store_true", help="Set to use greedy decoding instead of sampling")
    parser.add_argument(
        "--no_postranking", action="store_true", help="Set to disable post reranking in the presence of reranker"
    )
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--nbest", type=int, default=5, help="Number of times to run the output generation")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam search width")
    parser.add_argument(
        "--per_step_predictions", type=int, default=2, help="Number of predictions per step of beam search"
    )
    parser.add_argument(
        "--min_complete_in_beam",
        type=int,
        default=10,
        help="Minimum number of complete beam search elements to terminate beam",
    )
    parser.add_argument(
        "--aux_weight", type=float, default=0.5, help="auxiliary model weight (used if a reranker is provided)"
    )
    parser.add_argument(
        "--min_prob",
        type=float,
        default=0.00,
        help="minimum cumulative probability of available tokens to be used on decoding in beam search",
    )
    parser.add_argument(
        "--min_token_prob",
        type=float,
        default=0.00,
        help="minimum probability of token to be used on decoding in beam search",
    )
    parser.add_argument("--prob_window", type=int, default=0, help="Probability window")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument(
        "--log_every", type=int, default=50, help="frequency of logging the output and computing the metrics"
    )
    parser.add_argument("--dec_dropout", type=float, default=0.0, help="Decoding dropout")
    parser.add_argument("--averaging", type=str, default="default", help="averaging method")
    parser.add_argument("--ewm_alpha", type=int, default=0.5, help="value of com for the EWM average")
    parser.add_argument(
        "--beam_alpha", type=float, default=0.75, help="value of alpha for length penalty in beam search"
    )
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)"
    )
    parser.add_argument("--frac", type=float, default=1.0, help="fraction of test data to consider")
    parser.add_argument("--max_data", type=int, default=0, help="Number of data items (0 includes everything)")
    parser.add_argument("--smoothing", action="store_true", help="If true use label smoothing")
    parser.add_argument("--add_coverage_penalty", action="store_true", help="Add coverage penalty while decoding")
    parser.add_argument("--no_mlflow_logging", action="store_true", help="If true disable logging to mlflow")
    parser.add_argument(
        "--cons_classifier", type=str, default=None, help="consistency classifier checkpoint, to use during decoding)"
    )

    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    args = parser.parse_args()
    args.min_complete_in_beam = min(args.min_complete_in_beam, args.beam_width)

    if st._is_running_with_streamlit:
        examples = load_test_data()

        st.header("DataTuner Demo")

        args.nbest = 3
        args.beam_width = 3
        args.per_step_predictions = 2
        args.input = True

        client = mlflow.tracking.MlflowClient()

        args.model_checkpoint = st.sidebar.selectbox(
            "Model", get_finished_models([5]), 0, lambda x: " ".join(list(get_run_info(client, x).values()))
        )

        if False:
            args.reranker = st.sidebar.selectbox(
                "Reranker",
                ["/home/ec2-user/data/distilgpt2/", "/home/ec2-user/data/gpt2/", "/home/ec2-user/data/gpt2-medium/"],
                -1,
                lambda x: " ".join(list(get_run_info(client, x).values())),
            )

        st.write(f"**Main Model**: {get_run_info(client, args.model_checkpoint)}")
        st.write(f"**Auxiliary Model**: {get_run_info(client, args.reranker)}")

    else:
        pass

    model, tokenizer, task_config, learned_fields, input_text_fields, reranker, out_folder, is_local, cons_classifier = setup(
        args
    )
    if args.input:

        def process_input(inst):
            if st._is_running_with_streamlit:
                write = st.write
            else:
                write = print

            new_item, _ = process_one_item(
                inst,
                tokenizer,
                model,
                task_config,
                args,
                input_cache=input_cache,
                avoided_cache=avoided_cache,
                reranker=reranker,
                cons_classifier=cons_classifier,
            )

            for key in learned_fields:
                write("**Answers:**")
                if type(new_item[key]) == list:

                    for x in new_item[key]:
                        write(x)
                    if reranker is not None and not args.no_postranking:
                        write("\n**Answers Reranked from Pretrained Model:**")
                        for x in new_item["reranked_" + key]:
                            write(x)

                else:
                    text_to_print = f'{key}: {new_item[key]}'
                    write(text_to_print)

        input_cache = {}
        avoided_cache = defaultdict(lambda: 0)

        inst = {}
        empty = False
        if st._is_running_with_streamlit:
            args.cons_classifier = "ldc"
            mr_key = dataset_fields[args.cons_classifier]["data"] if args.cons_classifier else "linearized_amr"
            option = st.selectbox("Select an example", examples, 0, lambda x: x[mr_key])

            args.repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 10.0, float(args.repetition_penalty))
            args.cache_lambda = st.sidebar.slider("Cache Lambda", 0.0, 1.0, float(args.cache_lambda))
            args.boost_factor = st.sidebar.slider("boost_factor", 0.0, 3.0, float(args.boost_factor))
            args.cache_theta = st.sidebar.slider("Cache Theta", 0.0, 1.0, float(args.cache_theta))

            args.reranking_mode = st.sidebar.selectbox("Auxiliary Mode", ["average", "max"], 0)
            args.averaging = st.sidebar.selectbox("Averaging Method", ["arithmetic", "geometric", "ewm", "min"], 0)

            args.aux_weight = st.sidebar.slider("Weight of Auxiliary Model", 0.0, 1.0, 0.0)
            args.min_prob = st.sidebar.slider("Min Probability", 0.0, 1.0, float(args.min_prob))
            args.min_token_prob = st.sidebar.slider("min_token_prob", 0.0, 1.0, float(args.min_token_prob))
            args.ewm_alpha = st.sidebar.slider("ewm_alpha", 0.0, 1.0, float(args.ewm_alpha))
            args.prob_window = st.sidebar.slider("prob_window", 0, 100, args.prob_window)

            args.top_k = st.sidebar.slider("top_k", 0, 100, int(args.top_k))
            args.nbest = st.sidebar.slider("nbest", 1, 10, int(args.nbest))
            args.beam_width = st.sidebar.slider("beam_width", 1, 10, int(args.beam_width))

            args.top_p = st.sidebar.slider("top_p", 0.0, 1.0, float(args.top_p))
            args.dec_dropout = st.sidebar.slider("Decoding Dropout", 0.0, 1.0, float(args.dec_dropout))
            args.temperature = st.sidebar.slider("temperature", 0.0, 3.0, float(args.temperature))
            args.per_step_predictions = st.sidebar.slider("per_step_predictions", 1, 5, int(args.per_step_predictions))
            args.no_sample = bool(st.sidebar.slider("no_sample", 0, 1, 1))

            for key in input_text_fields:
                text_input = st.text_area(key, option[key] if type(option[key]) == str else "; ".join(option[key]))
                if not text_input:
                    empty = True
                inst[key] = tokenizer.encode(text_input)

            for key in learned_fields:
                st.write(f"{key}: {option[key] if type(option[key]) == str else option[key][-1]}")

            if not empty:
                process_input(inst)

        else:
            while True:
                inst = {}
                for key in input_text_fields:
                    text_input = input(f"{key}>> ")
                    inst[key] = tokenizer.encode(text_input)

                process_input(inst)

    else:

        infile = Path(args.filename)

        data = get_dataset_from_file(tokenizer, infile, task_config, args.max_data)
        outfilename = f"generated.json"
        metrics_results = defaultdict(list)

        out_filepath = out_folder / outfilename
        metrics_fields = task_config["metrics_fields"] if "metrics_fields" in task_config else []
        output_to_metrics = {}
        for out_entity in task_config["data_shape"]:
            if "metrics" in out_entity:
                output_to_metrics[out_entity["id"]] = out_entity["metrics"]

        def write_output(final=False):

            stats = aggregate_metrics(all_outputs, learned_fields, metrics_fields, output_to_metrics, final=final)
            for key in stats:
                if "total" in stats[key]:
                    logger.info(f"{key}: {stats[key]['total']}")

            (
                    out_folder
                    / f"stats._{infile.stem}_{args.max_data}_{'reranked' if args.reranker else ''}_generated.json"
            ).write_text(json.dumps(stats, indent=2))

            out_filepath.write_text(json.dumps(all_outputs, indent=2))
            logger.info(f"written to {out_filepath}")
            key = learned_fields[-1]
            # Check if first item in beam is equal to original
            not_matching_items = [
                item for item in all_outputs if item["original_" + key] != item[key + (" " * len("original_"))][0]
            ]
            (out_folder / f"non_matching_{outfilename}").write_text(json.dumps(not_matching_items, indent=2))
            return stats

        if not args.ignore_existing and out_filepath.exists():
            all_outputs = json.load(open(out_filepath, "r"))
            skip = len(all_outputs)
            for s in range(skip):
                if "extra_fields" in task_config:
                    for field in task_config["extra_fields"]:
                        all_outputs[s][field] = data[s][field]
        else:
            all_outputs = []
            skip = 0

        logger.info(f"skipping {skip} items that were already analyzed")
        for i, inst in enumerate(tqdm(data)):
            original_inst = custom_deep_copy(inst)
            try:
                if random.random() > args.frac:
                    continue

                if i < skip:
                    continue

                new_item, matching = process_one_item(
                    inst,
                    tokenizer,
                    model,
                    task_config,
                    args,
                    metrics_results=metrics_results,
                    metrics_fields=task_config["metrics_fields"] if "metrics_fields" in task_config else [],
                    avoided_cache=defaultdict(lambda: 0),
                    reranker=reranker,
                    cons_classifier=cons_classifier,
                )

                for key in learned_fields:
                    new_key = key + " " * len("original_")
                    new_item[new_key] = new_item[key]
                    orig_key = "original_" + key
                    orig = new_item[orig_key]
                    del new_item[key]
                    del new_item[orig_key]
                    new_item[orig_key] = orig

                if "extra_fields" in task_config:
                    for field in task_config["extra_fields"]:
                        new_item[field] = inst[field]

                if not matching:
                    logger.info(json.dumps(new_item, indent=2))

                all_outputs.append(new_item)

                if len(all_outputs) % args.log_every == 0:
                    write_output()

            except Exception as e:
                new_item = {}

                for key in learned_fields:

                    orig_key = "original_" + key

                    new_item[orig_key] = original_inst[key]
                    if new_item[orig_key] and type(new_item[orig_key][0]) == list:
                        new_item[orig_key] = new_item[orig_key][-1]
                for key in input_text_fields:
                    new_item[key] = original_inst[key]
                for key in new_item:
                    new_item[key] = tokenizer.decode(new_item[key])

                for key in learned_fields:
                    new_key = key + " " * len("original_")
                    new_item[new_key] = [""]

                all_outputs.append(new_item)
                logger.error(e)
                raise
                import ipdb

                ipdb.set_trace()

        stats = write_output(final=True)

        if not is_local and not args.no_mlflow_logging:
            mlflow.log_artifact(out_folder, "evaluation")
            flattened_stats = flatten(stats)
            flattened_stats = {k: flattened_stats[k] for k in flattened_stats if k.count("-") <= 3}
            mlflow.log_metrics(flattened_stats)


if __name__ == "__main__":
    run()
