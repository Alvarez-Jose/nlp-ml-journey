import csv
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

@dataclass
class Example:
    tokens: List[str]
    labels: List[str]
    
def load_train_data(path: str) -> List[Example]:
    examples = []
    with open(path, newline = "", encoding = "utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = row["sentence"].strip().split()
            labels = row["labels"].strip().split()
            assert len(tokens) == len(labels), "Tokens and labels must align"
            examples.append(Example(tokens=tokens, labels=labels))
    return examples

def load_test_data(path: str) -> List[Example]:
    examples = [] 
    with open(path, newline = "", encoding = "utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = row["sentence"].strip().split()
            examples.append(Example(tokens = tokens, labels = []))
    return examples

def build_vocab(examples: List[Example], min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for ex in examples:
        counter.update(ex.tokens)
        
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def build_tag_vocab(examples: List[Example]) -> Dict[str, int]:
    tags = set()
    for ex in examples:
        for t in ex.labels:
            tags.add(t)
    tag2id = {PAD_TOKEN: 0}
    for tag in sorted(tags):
        tag2id[tag] = len(tag2id)
    return tag2id

def encode_example(
    examples: List[Example], 
    word2id: Dict[str, int],
    tag2id: Dict[str, int],
    max_len: int
) -> Tuple[List[List[int]], List[List[int]]]:
    X_ids = []
    Y_ids = []
    for ex in examples:
        # words
        w_ids = [word2id.get(tok, word2id[UNK_TOKEN]) for tok in ex.tokens]
        # labels (a list that is empty)
        if ex.labels:
            y_ids = [tag2id[tag] for tag in ex.labels]
        else:
            y_ids = []
            
        # padding
        if len(w_ids) > max_len:
            w_ids = w_ids[:max_len]
            y_ids = y_ids[:max_len] if y_ids else y_ids
        else:
            pad_len = max_len - len(w_ids)
            w_ids = w_ids + [word2id[PAD_TOKEN]] * pad_len
            if y_ids:
                y_ids = y_ids + [tag2id[PAD_TOKEN]] * pad_len
        X_ids.append(w_ids)
        Y_ids.append(y_ids)
    return X_ids, Y_ids