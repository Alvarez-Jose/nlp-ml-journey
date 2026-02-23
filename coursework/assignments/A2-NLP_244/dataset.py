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
        
        # Create label mapping
        unique_labels = set()
        for label_seq in labels:
            unique_labels.update(label_seq)
        self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Save label mappings
        np.save('label2id.npy', self.label2id)
        np.save('id2label.npy', self.id2label)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokenized input
        word_ids = encoding.word_ids()
        aligned_labels = [-100] * len(word_ids)  # -100 is ignore index for loss
        
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                aligned_labels[i] = -100
            elif word_idx != previous_word_idx:
                # First token of a word
                if word_idx < len(labels):
                    aligned_labels[i] = self.label2id.get(labels[word_idx], -100)
            else:
                # Subsequent tokens - assign I-label if it's an entity
                if word_idx < len(labels) and labels[word_idx] != 'O':
                    # Use I-label for continuation
                    i_label = labels[word_idx].replace('B-', 'I-')
                    aligned_labels[i] = self.label2id.get(i_label, -100)
                else:
                    aligned_labels[i] = -100
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels)
        }

# Load preprocessed data
train_sequences = np.load('train_sequences.npy', allow_pickle=True)
train_labels = np.load('train_labels.npy', allow_pickle=True)
val_sequences = np.load('val_sequences.npy', allow_pickle=True)
val_labels = np.load('val_labels.npy', allow_pickle=True)

# Create datasets
train_dataset = AnnotationDataset(train_sequences, train_labels)
val_dataset = AnnotationDataset(val_sequences, val_labels)

# Save datasets
torch.save(train_dataset, 'train_dataset.pt')
torch.save(val_dataset, 'val_dataset.pt')