from data import GPTTokenizedData
from model import get_best_model_definition
from train import train_model
from evaluation import evaluate_perplexity




import torch

def main():
    # get dataloaders (data.py)
    tokenized = GPTTokenizedData()
    dataloaders = tokenized.dataloaders # all 3 dataloaders in a dictionary with keys 'train', 'test', 'val'
    vocab_size = tokenized.vocab_size

    # instantiate model (model.py)
    model = get_best_model_definition(vocab_size)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_token_id = tokenized.tokenizer.pad_token_id
    train_model(
        model=model,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["val"],
        pad_token_id=pad_token_id,
        num_epochs=5,
        device=device,
        save_path="best_model.pt"
    )
    
    # reload the best checkpoint before evaluating
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
                                     
    # evaluate perplexity for all three splits (evaluate.py)
    for split in ["train","val","test"]:
        ppl, avg_loss = evaluate_perplexity(model, dataloaders[split])
        print(f"{split.capitalize()} Perplexity: {ppl:.2f} | Loss: {avg_loss:.4f}")
       

if __name__ == "__main__":
    main()
