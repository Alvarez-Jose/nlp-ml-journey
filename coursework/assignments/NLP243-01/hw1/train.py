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
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

def train_model(model, predict_fn, train_loader, val_loader, test_loader, device, save_path='best_model.pt'):
    model = model.to(device)
    # train
    # save best model (so far)
    # use current loaders inside 
    with torch.no_grad():
        # Gather positives per class across the training set
        pos = torch.zeros(model.num_labels, dtype=torch.float32)
        total = 0
        for _, yb in train_loader:
            pos += yb.sum(dim=0).cpu()
            total += yb.size(0)
        neg = total - pos
        pos_weight = (neg / pos.clamp_min(1.0)).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # defines the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) #Adam optimization algorithm parameters using a learning rate of 0.001
    #optimizer = optim.SGD(model.parameters(), lr=1e-2) # SGD Accelerates momentum and adds weight_decay implements L2 regularization to prevent overfitting
    num_epochs = 25 # how many passes epoch passes for the training 
    best_f1 = -1.0 # compares the current F1 to best_F1 & -1 ensures the first epoch beats it

    for epoch in range(num_epochs):
        model.train()
        model.eval()
        with torch.no_grad():
            # load best model from training run
            f1_weighted, *_= evaluate_metrics(model, val_loader, predict_fn, device)

        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            torch.save(model.state_dict(), save_path)


    # load best model from training run
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    f1_train, *_ = evaluate_metrics(model, train_loader, predict_fn, device)
    f1_val, *_   = evaluate_metrics(model, val_loader, predict_fn, device)
    f1_test, *_  = evaluate_metrics(model, test_loader, predict_fn, device)

    print(f"*** Best (weighted) F1: {best_f1:.4f} ***")
    print(f"Train F1: {f1_train:.4f}")
    print(f"Val   F1: {f1_val:.4f}")
    print(f"Test  F1: {f1_test:.4f}")
    print(f'*** Best model weights saved at {save_path} ***')
    
    return model

