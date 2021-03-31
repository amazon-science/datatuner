from datatuner.lm.special_token_generator import get_custom_tags


def tokenize(self, text, **kwargs):
    """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """

    def split_on_token(tok, text):
        result = []
        split_text = text.split(tok)
        for i, sub_text in enumerate(split_text):
            sub_text = sub_text.strip()
            if i == 0 and not sub_text:
                result += [tok]
            elif i == len(split_text) - 1:
                if sub_text:
                    result += [sub_text]
                else:
                    pass
            else:
                if sub_text:
                    result += [sub_text]
                result += [tok]
        return result

    def split_on_tokens(tok_list, text):
        if not text:
            return []
        if not tok_list:
            return self._tokenize(text, **kwargs)

        tokenized_text = []
        text_list = [text]
        for tok in tok_list:
            tokenized_text = []
            for sub_text in text_list:
                if sub_text not in self.added_tokens_encoder and sub_text not in self.all_special_tokens:
                    tokenized_text += split_on_token(tok, sub_text)
                else:
                    tokenized_text += [sub_text]
            text_list = tokenized_text

        return sum(
            (
                self._tokenize(token, **kwargs)
                if token not in self.added_tokens_encoder and token not in self.all_special_tokens
                else [token]
                for token in tokenized_text
            ),
            [],
        )

    def get_special_tokens(s):
        candidates = get_custom_tags(s)
        return [cand for cand in candidates if cand in self.added_tokens_encoder.keys()]

    # The below becomes very slow when we scale to thousands of special tokens (e.g. many node types/predicates)
    # self.added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens

    all_added = list(self.added_tokens_encoder.keys())

    TOO_LARGE_NUM_TOKENS_THRESHOLD = 10
    # If we have a large number of special tokens, our current hack is to use task specific regexes to decide
    # candidates from the sentence first, and then match these candidates with the special tokens in the encoder.
    # That way we reduce the number of special tokens per iteration to a handful instead of thousands.
    if len(all_added) > TOO_LARGE_NUM_TOKENS_THRESHOLD:
        current_added_tokens = get_special_tokens(text)
    else:
        # Otherwise, we simply take all the added tokens, as is the original case in the library
        current_added_tokens = all_added

    added_tokens = current_added_tokens + self.all_special_tokens

    tokenized_text = split_on_tokens(added_tokens, text)
    return tokenized_text
