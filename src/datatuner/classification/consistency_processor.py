import csv
import os
import sys
from pathlib import Path

from transformers.data.processors.utils import DataProcessor, InputExample


class ConsistencyProcessor(DataProcessor):
    """Processor for the Consistency Classification data set."""

    def __init__(self, do_lower_case):
        self.do_lower_case = do_lower_case
        super(DataProcessor, self).__init__()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="|", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "validation.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        labels = (Path(data_dir) / "labels.txt").read_text().split("\n")
        labels = [x for x in labels if x]
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # order: ["label","data","text"]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, str(i))
            text_a = line[1]
            text_b = line[2]
            if self.do_lower_case:
                text_a = text_a.lower()
                text_b = text_b.lower()
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
