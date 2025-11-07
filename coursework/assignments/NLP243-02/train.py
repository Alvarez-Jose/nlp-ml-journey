import torch
from torch import nn, Tensor
from torch.optim import Adam
import math
from torch.cuda.amp import autocast, GradScaler

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
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    return loss

def _is_cuda_device(device):
    # Works whether you pass "cuda" or torch.device("cuda")
    return (isinstance(device, str) and device == "cuda") or (
        isinstance(device, torch.device) and device.type == "cuda"
    )

def train_model(model, train_loader, valid_loader, pad_token_id, num_epochs, device, save_path='best_model.pt'):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    use_cuda = _is_cuda_device(device)
    scaler = GradScaler(enabled=use_cuda)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_tokens = 0.0, 0
        

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch.get("labels", input_ids).to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # shifting for next token prediction
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            mask = attention_mask[:, :-1]      

            # forward pass
            with autocast(enabled=use_cuda):
                logits = model(inputs, attention_mask=mask)
                # gather the loss
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Tracking the token-weighted average
            label_mask = attention_mask[:, 1:]
            tokens_this_batch = int(label_mask.sum().item())
            total_loss   += loss.item() * tokens_this_batch 
            total_tokens += tokens_this_batch       

        avg_train_loss = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(avg_train_loss)

        model.eval()
        val_loss_sum, val_tokens = 0.0, 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                inputs = input_ids[:, :-1]
                labels = input_ids[:, 1:]
                mask = attention_mask[:, :-1]

                logits = model(inputs, attention_mask=mask)

                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

                label_mask = attention_mask[:, 1:]
                tokens_this_batch = int(label_mask.sum().item())
                val_loss_sum += loss.item() * tokens_this_batch
                val_tokens   += tokens_this_batch

        avg_val_loss = val_loss_sum / max(val_tokens, 1)
        val_ppl = math.exp(avg_val_loss)

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (PPL {train_ppl:.2f}) | "
            f"Val Loss: {avg_val_loss:.4f} (PPL {val_ppl:.2f})"
        )
