import torch
from util import get_model_device
from data import GPTTokenizedData
from model import get_best_model_definition
from torch import nn
import math
import argparse

def evaluate_perplexity(model, dataloader):
    model.eval()
    device = get_model_device(model)

    IGNORE = -100
    loss_fn_sum = nn.CrossEntropyLoss(reduction="sum", ignore_index=IGNORE)

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_mask: Tensor = batch["attention_mask"].to(device)

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            tgt_mask = attention_mask[:, 1:]  # 1=real, 0=pad
            targets = targets.masked_fill(tgt_mask == 0, IGNORE)

            logits = model(inputs, attention_mask=attention_mask[:, :-1])  # (B,S-1,V)
            V = logits.size(-1)

            loss_sum = loss_fn_sum(
                logits.reshape(-1, V),
                targets.reshape(-1),
            )

            total_loss += float(loss_sum.item())
            total_tokens += int((targets != IGNORE).sum().item())

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return ppl, avg_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained transformer LM")
    parser.add_argument("--model_path", default="./best_model.pt", help="Path to state_dict .pt")
    parser.add_argument("--batch_size", type=int, default=64, help="Eval batch size")
    return parser.parse_args()


def main():
    args = parse_arguments()
    tokenized = GPTTokenizedData(args.batch_size)
    dataloaders = tokenized.dataloaders
    vocab_size = tokenized.vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_best_model_definition(vocab_size)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict):
        model.load_state_dict(state, strict=True)
    model.to(device)

    ppl, avg_loss = evaluate_perplexity(model, dataloaders["test"])
    print(f"Test Perplexity: {ppl:.4f} | Avg NLL: {avg_loss:.6f}")

if __name__ == "__main__":
    main()



