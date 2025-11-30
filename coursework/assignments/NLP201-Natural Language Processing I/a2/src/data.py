import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "A2-Data 2")



FILE_NAMES = {
    "train": "1b_benchmark.train.tokens",
    "dev":   "1b_benchmark.dev.tokens",
    "test":  "1b_benchmark.test.tokens",
}

def load_split(split: str):
    """
    Load one split ('train', 'dev', or 'test') and return a list of sentences (strings).
    """
    if split not in FILE_NAMES:
        raise ValueError(f"Unknown split: {split}. Expected one of: {list(FILE_NAMES.keys())}")

    file_path = os.path.join(DATA_DIR, FILE_NAMES[split])

    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f]

    return sentences


class SimpleTokenizer:
    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.vocab = None      # set of tokens
        self.freq = None       # Counter: token -> count

    def tokenize(self, line: str):
        """
        Dataset is already whitespace-tokenized.
        """
        return line.strip().split()

    def build_vocabulary(self, train_sentences):
        """
        Build vocabulary from training sentences only.
        Rare tokens (freq < min_freq) will be mapped to <UNK>.
        """
        freq = Counter()

        # Count token frequencies
        for line in train_sentences:
            tokens = self.tokenize(line)
            freq.update(tokens)

        # Save raw frequency dictionary
        self.freq = freq

        # Keep only tokens with freq >= min_freq
        vocab = {tok for tok, c in freq.items() if c >= self.min_freq}

        # Required special tokens
        vocab.add("<UNK>")
        vocab.add("<STOP>")  
        # <START> used in sequences, not in vocab

        self.vocab = vocab

    def preprocess_train(self, sentences):
        assert self.freq is not None
        processed = []
        for line in sentences:
            tokens = self.tokenize(line)
            new_tokens = ["<START>", "<START>"]
            for tok in tokens:
                if self.freq[tok] < self.min_freq:
                    new_tokens.append("<UNK>")
                else:
                    new_tokens.append(tok)
            new_tokens.append("<STOP>")
            processed.append(new_tokens)

        return processed

    def preprocess_other(self, sentences):
        assert self.vocab is not None
        processed = []
        for line in sentences:
            tokens = self.tokenize(line)
            new_tokens = ["<START>", "<START>"]
            for tok in tokens:
                if tok in self.vocab:
                    new_tokens.append(tok)
                else:
                    new_tokens.append("<UNK>")
            new_tokens.append("<STOP>")
            processed.append(new_tokens)

        return processed



