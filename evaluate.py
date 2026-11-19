"""
Step 6: score the trained model on the TEST set (text it has never seen, never used for choices).

How: walk through the whole text in windows of block_size characters, no random sampling,
so every character is scored exactly once and the number is the same every run.
Val is scored the same way, so the two numbers can be compared fairly.

Run: .venv/bin/python evaluate.py            (default: gpt_full)
     .venv/bin/python evaluate.py small
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from common import data, device
from gpt import GPT, SIZES


def load_model(size):
    c = SIZES[size]
    model = GPT(c).to(device)
    path = Path(f"results/gpt_{size}/best.pt")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, c


@torch.no_grad()
def full_pass_loss(model, split, block_size, batch_size=32):
    """Average loss over every character in the split (windows of block_size, side by side)."""
    d = data[split]
    n_windows = (len(d) - 1) // block_size
    starts = torch.arange(n_windows) * block_size
    total, count = 0.0, 0
    for i in range(0, n_windows, batch_size):
        s = starts[i:i + batch_size]
        x = torch.stack([d[j:j + block_size] for j in s]).to(device)
        y = torch.stack([d[j + 1:j + block_size + 1] for j in s]).to(device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
        count += y.numel()
    return total / count, count


if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "full"
    model, c = load_model(size)
    print(f"Model: gpt_{size} | parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    scores = {}
    for split in ("val", "test"):
        loss, n = full_pass_loss(model, split, c.block_size)
        scores[split] = {"loss": round(loss, 4), "perplexity": round(math.exp(loss), 3), "characters": n}
        print(f"{split:5s}: loss {loss:.3f} | perplexity {math.exp(loss):.2f} | {n:,} characters scored")

    gap = scores["test"]["loss"] - scores["val"]["loss"]
    print(f"\nTest - val gap: {gap:+.3f}  (close to 0 = the val score was honest)")
    print("Reference: random 4.174 | bigram 2.478 | 1-head attention 2.330 (val)")

    out = Path(f"results/gpt_{size}/test_scores.json")
    out.write_text(json.dumps({"model": f"gpt_{size}", **scores}, indent=2))
    print(f"Saved -> {out}")
