import torch
from util import get_model_device
from data import GPTTokenizedData
from model import get_best_model_definition
from torch import nn
from torch.cuda.amp import autocast
import math
import argparse

# Enable cudnn autotune for speed
torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate_perplexity(model, dataloader):
    """
    Evaluate perplexity of an autoregressive transformer using teacher forcing
    on (inputs -> targets shifted by 1). Keeps original structure.
    """
    model.eval()
    device = get_model_device(model)

    loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        # move tensors to GPU if available
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        padding_mask = batch['attention_mask'].to(device, non_blocking=True)

        # shift input and targets
        targets = input_ids[:, 1:]
        inputs = input_ids[:, :-1]
        target_padding_mask = padding_mask[:, 1:]
        input_padding_mask  = padding_mask[:, :-1]

        # ignore padding in loss
        targets = targets.masked_fill(target_padding_mask == 0, -100).view(-1)

        # AMP (works even if CPU; no device_type arg needed)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(inputs, input_padding_mask)  # (B, S, V)

        B, S, V = logits.shape
        logits = logits.view(-1, V)

        total_loss += loss_fn(logits, targets).item()
        total_tokens += target_padding_mask.sum().item()

    perplexity = math.exp(total_loss / max(total_tokens, 1))
    return perplexity, total_loss


def parse_arguments():
    """
    Parse command line arguments for model evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained transformer model")
    parser.add_argument('--model_path', default='./best_model.pt', help='Path to the trained model file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    return parser.parse_args()


def main():
    args = parse_arguments()

    tokenized = GPTTokenizedData(args.batch_size)
    dataloaders = tokenized.dataloaders

    vocab_size = tokenized.vocab_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_best_model_definition(vocab_size).to(device)

    # load model weights directly to the correct device
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    ppl, loss = evaluate_perplexity(model, dataloaders['test'])
    print(f"Perplexity: {ppl:.4f}, Total loss: {loss:.4f}")


if __name__ == "__main__":
    main()


