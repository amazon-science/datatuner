from datatuner.lm.model_loader import load_pretrained_tokenizer
from fire import Fire


def launch(model_checkpoint, model_type="gpt2"):
    tokenizer = load_pretrained_tokenizer(model_checkpoint, model_type)

    while True:
        tokenized = tokenizer.tokenize(input("text >>> "))

        print(tokenized)
        print(tokenizer.convert_tokens_to_ids(tokenized))


if __name__ == "__main__":
    Fire(launch)
