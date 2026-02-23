import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer

class AnnotationDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer_name='bert-base-uncased', max_length=128):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # make the mapping
        unique_labels = set()
        for label_seq in labels:
            unique_labels.update(label_seq)
        self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}