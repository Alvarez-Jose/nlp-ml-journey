from torch import nn
import torch
import math
from typing import Optional
from torch import nn
import torch, math
from typing import Optional

class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0
        # (S,1)
        pos = torch.arange(max_seq_len).unsqueeze(1)
        # (D/2,)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)) 
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (S,D)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        if S > self.max_seq_len:
            raise ValueError(f"seq len {S} > max_seq_len {self.max_seq_len}")
        return self.dropout(x + self.pe[:S].to(x.device, x.dtype))

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None, scale: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model) if scale else 1.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * self.scale

class EmbeddingWithPositions(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        padding_idx: Optional[int] = None,
        emb_dropout: float = 0.1,
        scale_tokens: bool = True,
    ):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx, scale=scale_tokens)
        self.pos = SinusoidalPositions(max_seq_len, d_model, dropout=emb_dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.pos(self.tok(input_ids))

class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        max_seq_len: int = 512,
        n_layer: int = 4,
        n_head: int = 6,
        padding_idx: Optional[int] = None,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = EmbeddingWithPositions(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            padding_idx=padding_idx,
            emb_dropout=emb_dropout,
            scale_tokens=True,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_init_weights)
        # weight tying
        self.lm_head.weight = self.embed.tok.embedding.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # (B,S,D)
        x = self.embed(input_ids)
        S = x.size(1)

        causal: torch.Tensor = torch.triu(
            torch.ones((S, S), dtype=torch.bool, device=x.device), diagonal=1
        )
        key_pad = (attention_mask == 0) if attention_mask is not None else None

        x = self.encoder(x, mask=causal, src_key_padding_mask=key_pad)
        x = self.ln_f(x)
        return self.lm_head(x)

def get_best_model_definition(
    vocab_size: int,
    d_model: int = 384,
    max_seq_len: int = 512,
    padding_idx: Optional[int] = None,
    emb_dropout: float = 0.1,
) -> nn.Module:
    return TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        n_layer=4,
        n_head=6,
        padding_idx=padding_idx,
        emb_dropout=emb_dropout,
    )
