import torch
from torch import nn, Tensor
from torch.optim import Adam
import math

def build_shifted_ll(input_ids: Tensor):
    S = input_ids.size(1)
    inputs = input_ids[:, :S-1]
    labels = input_ids[:, 1:S]
    return inputs, labels

def trim_mask(attention_mask: Tensor):
    S = attention_mask.size(1)
    attn_in = attention_mask[:, :S-1]
    attn_tgt = attention_mask[:, 1:S]
    return attn_in, attn_tgt

def compute_loss_from_logits(logits, labels, pad_token_id):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fn(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1)
    )
    return loss

def train_model(model, train_loader, valid_loader, pad_token_id, num_epochs, device, save_path='best_model.pt'):

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)

    IGNORE = -100
    loss_fn_sum = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction='sum')

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss_sum, total_tokens = 0.0, 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # shift inputs/labels for next-token prediction (B, S-1)
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]    

            # mask labels where padding = 0
            label_mask = attention_mask[:, 1:]  # (B, S-1)
            labels = labels.masked_fill(label_mask == 0, IGNORE)

            optimizer.zero_grad()

            logits = model(inputs, attention_mask=attention_mask[:, :-1])  # (B, S-1, V)

            # summed loss over valid tokens
            B, S, V = logits.shape
            loss_sum = loss_fn_sum(logits.reshape(-1, V), labels.reshape(-1))

            # mean loss per non-ignored token for stable grads
            tokens_this_batch = int((labels != IGNORE).sum().item())
            (loss_sum / max(tokens_this_batch, 1)).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_sum += loss_sum.item()
            total_tokens   += tokens_this_batch

        avg_train_loss = total_loss_sum / max(total_tokens, 1)
        train_ppl = math.exp(avg_train_loss)

        # validate 
        model.eval()
        val_loss_sum, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                inputs = input_ids[:, :-1]
                labels = input_ids[:, 1:]
                label_mask = attention_mask[:, 1:]
                labels = labels.masked_fill(label_mask == 0, IGNORE)

                logits = model(inputs, attention_mask=attention_mask[:, :-1])
                B, S, V = logits.shape

                loss_sum = loss_fn_sum(logits.reshape(-1, V), labels.reshape(-1))
                tokens_this_batch = int((labels != IGNORE).sum().item())

                val_loss_sum += loss_sum.item()
                val_tokens   += tokens_this_batch

        avg_val_loss = val_loss_sum / max(val_tokens, 1)
        val_ppl = math.exp(avg_val_loss)

        # save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[SAVED] {save_path} @ epoch {epoch+1} | val_loss={avg_val_loss:.4f} ppl={val_ppl:.2f}")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (PPL {train_ppl:.2f}) | "
            f"Val Loss: {avg_val_loss:.4f} (PPL {val_ppl:.2f})"
        )
