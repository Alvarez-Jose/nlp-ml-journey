import torch
import torch.nn as nn
from transformers import AutoModel  # Fixed: AutoModel not Automodel

class NERModel(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-uncased'):
        super(NERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# You can test the model creation with:
if __name__ == "__main__":
    # Quick test
    import numpy as np
    label2id = np.load('label2id.npy', allow_pickle=True).item()
    num_labels = len(label2id)
    model = NERModel(num_labels=num_labels)
    print(f"✅ Model initialized with {num_labels} labels")
    print(f"Model architecture:\n{model}")