import torch
from torch import nn, Tensor
from torch.optim import Adam
import math
from torch.optim import AdamW
from typing import Optional

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

def train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    pad_token_id: Optional[int],
    num_epochs: int,
    device: str,
    save_path: str = "best_model.pt",
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    warmup = max( int(0.10 * total_steps), 400 )

    def lr_scale(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))  # in (0,1]

    IGNORE = -100
    loss_fn_sum = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss_sum, total_tokens = 0.0, 0

        for batch in train_loader:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_mask: Tensor = batch["attention_mask"].to(device)

            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            label_mask = attention_mask[:, 1:]
            labels = labels.masked_fill(label_mask == 0, IGNORE)

            scale = lr_scale(global_step)
            for g in optimizer.param_groups:
                g["lr"] = 2e-4 * scale

            logits = model(inputs, attention_mask=attention_mask[:, :-1])  # (B,S-1,V)
            V = logits.size(-1)

            loss_sum = loss_fn_sum(logits.reshape(-1, V), labels.reshape(-1))
            tokens_this_batch = int((labels != IGNORE).sum().item())

            optimizer.zero_grad(set_to_none=True)
            (loss_sum / max(tokens_this_batch, 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1

            total_loss_sum += float(loss_sum.item())
            total_tokens += tokens_this_batch

        avg_train_loss = total_loss_sum / max(total_tokens, 1)
        train_ppl = math.exp(avg_train_loss)

        model.eval()
        val_loss_sum, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids: Tensor = batch["input_ids"].to(device)
                attention_mask: Tensor = batch["attention_mask"].to(device)

                inputs = input_ids[:, :-1]
                labels = input_ids[:, 1:]
                label_mask = attention_mask[:, 1:]
                labels = labels.masked_fill(label_mask == 0, IGNORE)

                logits = model(inputs, attention_mask=attention_mask[:, :-1])
                V = logits.size(-1)

                loss_sum = loss_fn_sum(logits.reshape(-1, V), labels.reshape(-1))
                tokens_this_batch = int((labels != IGNORE).sum().item())

                val_loss_sum += float(loss_sum.item())
                val_tokens += tokens_this_batch

        avg_val_loss = val_loss_sum / max(val_tokens, 1)
        val_ppl = math.exp(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[SAVED] {save_path} @ epoch {epoch+1} | val_loss={avg_val_loss:.4f} ppl={val_ppl:.2f}")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (PPL {train_ppl:.2f}) | "
            f"Val Loss: {avg_val_loss:.4f} (PPL {val_ppl:.2f})"
        )