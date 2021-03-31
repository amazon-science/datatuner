from pathlib import Path

import torch
import torch.nn.functional as F
from datatuner.lm.data_loader import get_inputs
from datatuner.lm.model_loader import load_pretrained
from datatuner.lm.utils import custom_deep_copy, load_task_config, should_ignore_in_score
from datatuner.utils import geo_mean


class Reranker:
    def __init__(self, model_folder, device, is_local=True):
        with torch.no_grad():
            self.model, self.tokenizer = load_pretrained(model_folder, model_type="gpt2")
            self.model_folder = Path(model_folder)
            try:
                self.task_config = load_task_config(model_folder / "task_config.json")
            except:
                self.task_config = None
            self.device = device
            self.is_local = is_local
            self.model.to(self.device)
            self.model.eval()

            self.NEWLINE = [198]
            self.SPACE = [220]

    def remove_unsupported_tokens(self, ids):
        new_ids = []
        for j in ids:
            try:
                self.tokenizer.decode(j)
                new_ids.append(j)
            except:
                new_ids.append(self.SPACE[0])
        return new_ids

    def create_input(self, input_ids, item):
        if not self.is_local:
            assert item is not None
            item = custom_deep_copy(item)
            item.update({"answer_text": input_ids})
            input_ids, token_type_ids = get_inputs(item, self.device, self.tokenizer, self.task_config)
            context_len = len(input_ids[0]) - len(input_ids)

        else:
            if "linearized_amr" in item:
                context = []
                context_len = 0

            input_ids = self.tokenizer.encode(
                self.tokenizer.decode(self.remove_unsupported_tokens(context + input_ids))
            )

            input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
            token_type_ids = None

        return input_ids, token_type_ids, context_len

    def score(self, input_ids, item):

        input_ids, token_type_ids, context_len = self.create_input(input_ids, item)

        model_outputs = self.model(input_ids, token_type_ids=token_type_ids)
        probs = F.softmax(model_outputs[0][0], dim=1)
        x = []
        for i in range(context_len, len(input_ids[0])):
            next_token_id = input_ids[0][i].item()
            next_token_str = self.tokenizer.decode(next_token_id)
            prefix = input_ids[0][context_len:i]
            next_prob = probs[i - 1][next_token_id].item()
            if (
                    not should_ignore_in_score(prefix, self.tokenizer, next_token_str, next_token_id, next_prob)
                    and input_ids[0][i] != self.SPACE[0]
            ):
                x.append(next_prob)
        score = geo_mean(x)
        return score

    def rerank(self, nbest_items, item):
        with torch.no_grad():
            scores = []

            for input_ids in nbest_items:
                scores.append(-self.score(input_ids, item))

            nbest_items = [x for _, _, x in sorted(zip(scores, list(range(0, len(nbest_items))), nbest_items))]

            return nbest_items
