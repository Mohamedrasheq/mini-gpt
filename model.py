"""
The model: a GPT (decoder-only transformer) with the three parts most recent small models use:

  RoPE     positions are added by ROTATING each head's query/key vectors (no position table).
           Works better than a learned position table and adds no parameters.
  SwiGLU   the feed-forward layer uses a gate: (SiLU(x W1) * x W3) W2, instead of GELU(x W1) W2.
           Hidden size 1,408 (about 8/3 x 512) keeps the parameter count close to GELU's 4 x 512.
  RMSNorm  a simpler LayerNorm: only rescales, no mean subtraction and no bias.
No biases in the linear layers; the output layer shares its weights with the word table.

Size (default): 8 layers x 512 wide x 8 heads, 512-token context, 8,192 tokens
  word table 8,192 x 512 = 4.19M | each layer ~3.21M (attention 1.05M + feed-forward 2.16M) | total ~29.9M
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    ff_hidden: int = 1408
    rope_base: float = 10000.0


def rope_tables(head_dim, length, base):
    """cos / sin of the rotation angle for every position and every pair of dimensions."""
    freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angle = torch.outer(torch.arange(length).float(), freq)          # (length, head_dim/2)
    return angle.cos(), angle.sin()


def rotate(x, cos, sin):
    """Rotate each pair (x1, x2) of a head's vector by its position's angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head, self.head_dim = c.n_head, c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = rotate(q, cos[:T], sin[:T]), rotate(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.n_embd, c.ff_hidden, bias=False)       # gate (goes through Swish / SiLU)
        self.w3 = nn.Linear(c.n_embd, c.ff_hidden, bias=False)       # value
        self.w2 = nn.Linear(c.ff_hidden, c.n_embd, bias=False)       # back to model width

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.attn = nn.RMSNorm(c.n_embd), Attention(c)
        self.norm2, self.ff = nn.RMSNorm(c.n_embd), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ff(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = nn.RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                       # weight tying
        cos, sin = rope_tables(c.n_embd // c.n_head, c.block_size, c.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():                     # smaller start for residual outputs
            if name.endswith(("proj.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    m = GPT(Config())
    total = sum(p.numel() for p in m.parameters())
    emb = m.tok_emb.weight.numel()
    print(f"parameters: {total:,} | word table {emb:,} ({emb / total:.0%}) | layers {total - emb:,}")
    x = torch.randint(0, 8192, (2, 64))
    logits, loss = m(x, x)
    print(f"output {tuple(logits.shape)} | starting loss {loss.item():.2f} (random guess = ln 8192 = {math.log(8192):.2f})")
