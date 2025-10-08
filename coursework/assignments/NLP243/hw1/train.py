"""
STUDENT IMPLEMENTATION REQUIRED

This file contains the training loop that you need to implement for HW1.
You should complete the train_model function by implementing the training logic
including optimizer setup, loss function, training loop, and model saving.

TODO: Implement the training loop in the train_model function
"""

# define your training loop here
import torch
from eval import evaluate_metrics

def train_model(model, predict_fn, train_loader, val_loader, device, save_path='best_model.pt'):
    model = model.to(device)
    
    model.train()
    
    # train

    # save best model (so far)
    torch.save(model.state_dict(), save_path)

    # load best model from training run
    model.load_state_dict(torch.load(save_path))
    f1_weighted, precision, recall, f1, support = evaluate_metrics(model, val_loader, predict_fn, device)
    print(f"*** Best (weighted) F1: {f1_weighted} ***")
    print(f'*** Best model weights saved at {save_path} ***')
    
    return model
    