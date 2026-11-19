"""
Step 4: one self-attention head on top of the baseline.

New compared with the bigram:
  - token embedding    : each character -> a list of N_EMBD numbers (its "meaning")
  - position embedding : each position 0..BLOCK_SIZE-1 -> N_EMBD numbers ("I am the 5th character")
  - one attention head : each character looks back at earlier characters and mixes in what's useful
  - output layer       : turns the mixed result into 65 scores for the next character

Run: .venv/bin/python attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device, generate, tok, train

# ---------------- settings ----------------
BATCH_SIZE = 32
BLOCK_SIZE = 32       # how far back the model can look (characters)
N_EMBD = 64           # size of each character's number list
HEAD_SIZE = 64        # size of query / key / value
STEPS = 5000
LR = 1e-3
EVAL_EVERY = 500
SEED = 1337
torch.manual_seed(SEED)


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # "what do I contain?"
        self.query = nn.Linear(n_embd, head_size, bias=False)  # "what am I looking for?"
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "what I pass on"
        # lower-triangle mask: position t may only look at positions 0..t (never the future)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.last_weights = None

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)            # (B, T, head)
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5          # (B, T, T): how well each pair matches
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # hide the future
        weights = F.softmax(scores, dim=-1)                              # each row adds up to 1
        self.last_weights = weights.detach()
        return weights @ v                                               # weighted mix of values


class AttentionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.head = Head(N_EMBD, HEAD_SIZE, BLOCK_SIZE)
        self.lm_head = nn.Linear(HEAD_SIZE, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))  # what + where
        x = self.head(x)                                                          # look back
        logits = self.lm_head(x)                                                  # scores for next char
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


if __name__ == "__main__":
    model = AttentionModel(tok.vocab_size).to(device)
    settings = dict(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, n_embd=N_EMBD, head_size=HEAD_SIZE,
                    steps=STEPS, lr=LR, seed=SEED)
    model = train(model, "attention", settings, STEPS, LR, BATCH_SIZE, BLOCK_SIZE, EVAL_EVERY)

    print("\n--- text AFTER training ---")
    print(generate(model, BLOCK_SIZE, 400))

    # ---- look inside: where does the last character look? ----
    text = "First Citizen:\nBefore we proceed"
    idx = torch.tensor([tok.encode(text[-BLOCK_SIZE:])], device=device)
    model.eval()
    with torch.no_grad():
        model(idx)
    w = model.head.last_weights[0, -1].cpu()          # attention of the LAST character
    chars = text[-BLOCK_SIZE:]
    top = torch.topk(w, 5)
    print(f"\n--- attention of the last character {chars[-1]!r} in {chars!r} ---")
    for p, i in zip(top.values, top.indices):
        print(f"  looks at position {i.item():2d} {chars[i]!r:6s} with weight {p:.0%}")
