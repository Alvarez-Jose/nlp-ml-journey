from torch import nn
import torch
import math

class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0

        position = torch.arange(max_seq_len).unsqueeze(-1) # S, 1
        # inside sine / cosine we have pos * (10_000**-2m/d)
        # for stability, calculate instead exp(-2m/d * log(10_000))
        # multiplier shape D/2, then S, 1 * D/2 -> S, D/2
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier) # S, D/2
        pe[:, 1::2] = torch.cos(position * multiplier)

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape B, S, D
        batch_seq_len = x.shape[1]
        assert batch_seq_len <= self.max_seq_len, f"batch_seq_len {batch_seq_len} > max_seq_len {self.max_seq_len}"
        pe = self.pe[:batch_seq_len, :].to(device=x.device, dtype=x.dtype)
        # broadcast over batch
        x = x + pe.unsqueeze(0)
        
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int =None, scale: bool = True):
        super().__init__()
        # Instantiate the internal nn.Embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model,padding_idx=padding_idx)
        self.scale = math.sqrt(d_model) if scale else 1.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, S)
        # output: (B, S, D)
        return self.embedding(input_ids) * self.scale
    
class EmbeddingwithPositions(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, padding_idx: int = None, emb_dropout: float = 0.1, scale_tokens: bool = True):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx, scale=scale_tokens)
        self.pos = SinusoidalPositions(max_seq_len=max_seq_len, d_model=d_model, dropout=emb_dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok(input_ids)
        x = self.pos(x)
        return x



def get_best_model_definition(
        vocab_size: int, 
        d_model: int = 256, 
        max_seq_len: int = 512, 
        padding_idx: int = None, 
        emb_dropout: float = 0.1,
) -> nn.Module:
    """
    This is the model that will be used in the evaluation script
    Ensure it matches the .pt file provided there
    """
    return EmbeddingwithPositions(
        vocab_size=vocab_size,
        d_model=d_model, 
        max_seq_len=max_seq_len,
        padding_idx=padding_idx,
        emb_dropout=emb_dropout,
        scale_tokens=True
    )