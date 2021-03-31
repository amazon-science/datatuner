import copy
import json
import logging
from pathlib import Path

import ftfy
import torch

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def load_task_config(filename):
    """Load the task configuration from file"""
    task_config = json.load(open(filename, "r"))
    return task_config


def is_middle_token(tokenizer, token_str, prefix):
    try:
        tokenizer_name = str(type(tokenizer))

        if len(prefix) == 0:
            return False

        prev_token_str = tokenizer.decode(prefix[-1])

        # If the previous token is not alphanumeric, it's not a middle token
        if not prev_token_str[-1].isalnum():
            return False

        # The prev and current tokens should be of same type.
        if not (
                (prev_token_str[-1].isalpha() and token_str[0].isalpha())
                or (prev_token_str[-1].isdigit() and token_str[0].isdigit())
        ):
            return False

        if "GPT2" in tokenizer_name:
            return not (token_str[0] in [" ", "\u0120"])
        elif "OpenAIGPT" in tokenizer_name:
            return not prefix[-1].endswith("</w>")
        else:
            raise Exception("non-supported tokenizer")
    except:
        return False


def is_added_token(tokenizer, token_id):
    return token_id >= len(tokenizer.decoder)


def should_stop_further_beam_exploration(prefix, tokenizer, next_token_str, next_token_id, next_prob, prob_thresh=0.9):
    """We stop exploring the beam further if the current string is a word continuation as we don't expect better
    continuations to appear.
    Example 1: if we get "Who is the res" and the next token is "ponsible", we stop exploring.
    Example 2: if we get "The airport code is 12" and the next token is "4", we stop exploring.
    Example 3: if we get "The airport code is twenty" and the next token is ".", we stop exploring.
    Example 4: if we get "The airport code is 123" and the next token is ".", we stop exploring.
    """
    return (
            # The token is a middle token
            is_middle_token(tokenizer, next_token_str, prefix)
            #  is not a special token
            and not is_added_token(tokenizer, next_token_id)
            and next_prob > prob_thresh
    )


def should_ignore_in_score(prefix, tokenizer, next_token_str, next_token_id, next_prob, prob_thresh=0.9):
    return (
        # Probability is high enough
        # next_prob > prob_thresh
        # The token is a middle token
            is_middle_token(tokenizer, next_token_str, prefix)
            # is alphanumeric (avoid punctuations)
            and next_token_str.strip()[0].isalnum()
            #  is not a special token
            and not is_added_token(tokenizer, next_token_id)
            and next_prob > prob_thresh
    )


def custom_deep_copy(d):
    if type(d) == dict:
        new_d = {}
        for key in d:
            try:
                new_d[key] = torch.clone(d[key])
            except:
                new_d[key] = copy.deepcopy(d[key])
        return new_d
    else:
        try:
            return torch.clone(d)
        except:
            return copy.deepcopy(d)


def fix_text_in_dir(directory):
    """Fix text encoding with ftfy for the data splits withdirectory"""
    directory = Path(directory)
    for split in ["train.json", "validation.json", "test.json"]:
        data = json.load(open(directory / split))
        for item in data:
            for k in item:
                if type(item[k]) == str:
                    item[k] = ftfy.fix_text(item[k])
                elif type(item[k]) == list:
                    item[k] = [ftfy.fix_text(x) for x in item[k]]
        json.dump(data, open(directory / split, "w"), indent=2)
