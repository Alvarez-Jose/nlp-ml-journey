"""
TODO write your training loop here.
Things to take care with:
    - make sure to use the correct loss function for the task
    - make sure that the targets are correct (each token should predict the next token in the sequence)
    - there should be no loss for padding tokens.
"""

def train_model(model, dataloader, save_path='best_model.pt'):
    
    model.train()
    
    
    torch.save(model.state_dict(), save_path)