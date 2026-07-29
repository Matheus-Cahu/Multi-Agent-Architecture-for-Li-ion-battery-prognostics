import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class InvertedEmbedding(nn.Module):
    """Projeta cada variável (série temporal de comprimento seq_len) em um token de dimensão d_model."""

    def __init__(self, seq_len, d_model):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)

    def forward(self, x):
        x = x.transpose(-2, -1)  # (..., seq_len, n_vars) -> (..., n_vars, seq_len)
        x = self.proj(x)         # (..., n_vars, d_model)
        return x


class VariateAttention(nn.Module):
    """Self-attention entre as variáveis (tokens invertidos)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, H):
        out, attn_weights = self.attn(H, H, H)
        return out, attn_weights


class iTransformerBlock(nn.Module):
    """Bloco encoder reutilizável do iTransformer.

    Fluxo: VariateAttention -> Add & Norm -> FeedForward -> Add & Norm.
    Recebe e devolve tensores no formato (batch, n_vars, d_model), permitindo
    empilhar vários blocos em sequência.
    """

    def __init__(self, d_model, n_heads, d_ff=None, activation=None):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model

        self.attn = VariateAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation if activation is not None else nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, H):
        attn_out, attn_weights = self.attn(H)
        H = self.norm1(H + attn_out)      # Add & Norm

        ff_out = self.ff(H)
        H = self.norm2(H + ff_out)        # Add & Norm

        return H, attn_weights


class iTransformer(nn.Module):
    """iTransformer completo: embedding invertido + N blocos + projeção de previsão."""

    def __init__(self, seq_len, pred_len, d_model=128, n_heads=8,
                 n_blocks=1, d_ff=None):
        super().__init__()
        self.embedding = InvertedEmbedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            iTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)
        )
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        H = self.embedding(x)             # (batch, n_vars, d_model)
        for block in self.blocks:
            H, _ = block(H)
        pred = self.projection(H)         # (batch, n_vars, pred_len)
        pred = pred.transpose(-2, -1)     # (batch, pred_len, n_vars)
        return pred

class JanelaDataset(Dataset):
    def __init__(self, serie, T, S):
        self.serie = serie      # so a serie longa [T_total, N], sem copias
        self.T = T
        self.S = S
    def __len__(self):
        return len(self.serie) - self.T - self.S
    def __getitem__(self, i):
        x = self.serie[i : i+self.T]                 # recorta na hora
        y = self.serie[i+self.T : i+self.T+self.S]
        return x, y
