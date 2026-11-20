"""
Step 3: the baseline model (bigram). It guesses the next character from the CURRENT character only.

The whole model is one table of 65 x 65 numbers:
  row = the current character, columns = a score for every possible next character.
Training adjusts the table so likely pairs ("q" -> "u", "t" -> "h") get high scores.

Run: .venv/bin/python step03_bigram.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from step02_char_tokenizer import CharTokenizer

# ---------------- settings ----------------
BATCH_SIZE = 32        # text chunks per training step
BLOCK_SIZE = 8         # characters per chunk (the bigram only uses 1 at a time anyway)
STEPS = 3000
LR = 1e-2
EVAL_EVERY = 300
EVAL_BATCHES = 100     # average the loss over this many batches, so the number is steady
SEED = 1337
OUT = Path("results/bigram")

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tok = CharTokenizer.load()
data = {s: torch.from_numpy(np.fromfile(f"data/{s}.bin", dtype=np.uint8).astype(np.int64))
        for s in ("train", "val", "test")}


def get_batch(split):
    """Random chunks of text. x = input characters, y = the same chunk shifted by one (the answers)."""
    d = data[split]
    starts = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in starts])
    return x.to(device), y.to(device)


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # one row per character: the scores ("logits") for every possible next character
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)                       # (batch, time, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new):
        for _ in range(n_new):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)          # only the last character matters
            nxt = torch.multinomial(probs, num_samples=1)        # pick one, by its probability
            idx = torch.cat([idx, nxt], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(EVAL_BATCHES)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def sample(model, n=300):
    start = torch.zeros((1, 1), dtype=torch.long, device=device)   # start from a newline
    return tok.decode(model.generate(start, n)[0].tolist())


model = BigramModel(tok.vocab_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
random_loss = math.log(tok.vocab_size)

print(f"Device: {device} | parameters: {n_params:,} | random-guess loss: ln({tok.vocab_size}) = {random_loss:.3f}")
print("\n--- text BEFORE training (random table) ---")
print(sample(model, 200))

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
history = []
t0 = time.time()
print("\n--- training ---")
for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0:
        l = estimate_loss(model)
        history.append({"step": step, **l})
        print(f"step {step:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} "
              f"| val perplexity {math.exp(l['val']):5.1f} | {time.time() - t0:4.0f}s")
    if step == STEPS:
        break
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n--- text AFTER training ---")
text_after = sample(model, 400)
print(text_after)

# What did it learn? The most likely next character after a few letters
probs = F.softmax(model.table.weight.detach().cpu(), dim=-1)
print("\n--- what the table learned: most likely next character ---")
for ch in ["q", "t", "h", "T", ".", "\n"]:
    top = torch.topk(probs[tok.stoi[ch]], 3)
    nxt = ", ".join(f"{tok.itos[i.item()]!r} {p:.0%}" for p, i in zip(top.values, top.indices))
    print(f"  after {ch!r:5s} -> {nxt}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "history.json").write_text(json.dumps(
    {"model": "bigram", "params": n_params, "random_loss": random_loss,
     "settings": {"batch_size": BATCH_SIZE, "block_size": BLOCK_SIZE, "steps": STEPS, "lr": LR, "seed": SEED},
     "history": history, "sample": text_after}, indent=2))
print(f"\nSaved -> {OUT}/history.json")
