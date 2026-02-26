import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from model import NERModel
from dataset import AnnotationDataset

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, id2label):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            # Collect only valid labels (not -100)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        predictions.append(preds[i, j].item())
                        true_labels.append(labels[i, j].item())
    
    return predictions, true_labels

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load datasets
train_dataset = torch.load('train_dataset.pt', weights_only=False)
val_dataset = torch.load('val_dataset.pt', weights_only=False)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize model
model = NERModel(num_labels=len(train_dataset.label2id)).to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 5  # 5 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop
train_losses = []
val_f1_scores = []

for epoch in range(5):
    print(f"\nEpoch {epoch + 1}/5")
    
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    train_losses.append(train_loss)
    
    predictions, true_labels = evaluate(model, val_loader, device, val_dataset.id2label)
    
    # Calculate metrics
    report = classification_report(true_labels, predictions, 
                                 target_names=list(val_dataset.id2label.values()),
                                 output_dict=True)
    
    val_f1 = report['macro avg']['f1-score']
    val_f1_scores.append(val_f1)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation F1: {val_f1:.4f}")

# Save model
torch.save(model.state_dict(), 'ner_model.pt')

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(val_f1_scores)
plt.title('Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()