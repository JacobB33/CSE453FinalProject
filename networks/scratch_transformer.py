import torch
from torch import nn, Tensor
import torch.nn.functional as F
from configs import ModelConfig
import math
class PositionalEncoding(nn.Module):
    """
    Module implementing positional encoding for Transformer models.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
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
        batch_size, seq_len, _ = x.shape()
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


        
        
        