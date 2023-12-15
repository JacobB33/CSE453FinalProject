from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from configs import ModelConfig
import math
import tiktoken


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        norm = self._norm(x.float()).type_as(x)
        return self.weights * norm

# Batch size, sequence length, ....
class PositionalEncoding(nn.Module):
    """
    Module implementing positional encoding for Transformer models.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, x.size(1) - 1]
        return self.dropout(x)


class AttentionModule(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.embedding_size, cfg.embedding_size * 3)
        self.fc = nn.Linear(cfg.embedding_size, cfg.embedding_size)
        self.head_dim = cfg.embedding_size // cfg.nheads
        self.pos_embed = PositionalEncoding(cfg.embedding_size, cfg.dropout, cfg.max_seq_len)

    def forward(self, x: Tensor, mask: Tensor = None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.qkv(x).chunk(3, dim=-1)
        xq, xk = self.pos_embed(xq), self.pos_embed(xk)

        xq = xq.view(batch_size, seq_len, self.cfg.nheads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.cfg.nheads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.cfg.nheads, self.head_dim)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)  # Swap the seq_len and nheads dimension
        # This gives us (batch_size, nheads, seq_len, head_dim)

        # Compute the attention scores
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # This results in shape (batch_size, nheads, seq_len, seq_len) with the attention scores

        # Mask out the scores. Note the mask should have a 1 where it is masked out and a 0 if not
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)
        scores = torch.softmax(scores, dim=-1)  # Softmax over the last dimension
        output = torch.matmul(scores, xv)  # Shape (batch_size, nheads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.cfg.embedding_size)
        return self.fc(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.in_one = nn.Linear(dim, hidden_dim)
        self.in_two = nn.Linear(dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor):
        return self.output(F.silu(self.in_one(x)) * self.in_two(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg

        self.attn = AttentionModule(cfg)

        self.atn_norm = RMSNorm(cfg.embedding_size, eps=cfg.norm_eps)
        self.ff_norm = RMSNorm(cfg.embedding_size, eps=cfg.norm_eps)

        self.feed_forward = FeedForward(dim=cfg.embedding_size, hidden_dim=4 * cfg.embedding_size)

    def forward(self, x: Tensor, mask: Optional[Tensor]):
        h = x + self.attn(self.atn_norm(x), mask)
        out = h + self.feed_forward(self.ff_norm(h))
        return out


class   TransformerModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_size)
        self.layers = nn.ModuleList()

        for _ in range(cfg.n_layers):
            self.layers.append(TransformerBlock(cfg))
        self.norm = RMSNorm(cfg.embedding_size, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.embedding_size, cfg.vocab_size)

    def forward(self, tokens: Tensor, targets=None, start_pose=0):
        bsz, seqlen = tokens.shape

        h = self.embedding(tokens)
        mask = None
        if seqlen > 1:
            mask = torch.zeros((1, 1, seqlen, seqlen), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pose + 1)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)
        if targets is None:
            return self.output(h).float()
        h = self.output(h).float()

        return h, F.cross_entropy(h.reshape(-1, self.config.vocab_size), targets.flatten())
