import logging
import sys

from datatuner.classification.run_classifier import evaluate, main
from transformers.data.processors.utils import InputExample

logger = logging.getLogger(__name__)

dataset_fields = {
    "webnlg": {"text": "text", "data": "modifiedtripleset", "original_data": "raw_modifiedtripleset"},
    "ldc": {"text": "answer_text", "data": "linearized_amr", "original_data": "raw_amr"},
    "viggo": {"text": "ref", "data": "new_mr", "original_data": "mr"},
    "e2e": {"text": "ref", "data": "new_mr", "original_data": "mr"},
}


def get_data_fields():
    out = []
    for x in dataset_fields:
        out.append(dataset_fields[x]["data"])
    return out


class ConsistencyClassifier:
    def __init__(self, args_dict):
        self.args_dict = args_dict
        sys.argv = [sys.argv[0]]
        _, self.model, self.tokenizer, self.args = main(args_dict)
        self.cache = {}

    def evaluate(self, items, set_type="test"):
        examples = []
        for (i, item) in enumerate(items):
            guid = "%s-%s" % (set_type, str(i))
            text_a = item["data"]
            text_b = item["text"]
            if self.args_dict["do_lower_case"]:
                text_a = text_a.lower()
                text_b = text_b.lower()
            label = "accurate"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        self.args.examples = examples

        results = evaluate(self.args, self.model, self.tokenizer, prefix="")
        return results
