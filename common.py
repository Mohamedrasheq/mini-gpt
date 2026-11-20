"""Shared helpers for every model from step 4 on: data, batches, loss estimate, text sampling."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from step02_char_tokenizer import CharTokenizer
from step07_bpe_tokenizer import BPETokenizer

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = CharTokenizer.load()
data = {s: torch.from_numpy(np.fromfile(f"data/{s}.bin", dtype=np.uint8).astype(np.int64))
        for s in ("train", "val", "test")}
TOKENIZERS = {"char": (tok, data)}

# step 7: the same text cut into BPE pieces (1,024-piece vocabulary), once step07 has been run
if Path("data/bpe_merges.json").exists() and Path("data/train_bpe.bin").exists():
    bpe_tok = BPETokenizer.load()
    data_bpe = {s: torch.from_numpy(np.fromfile(f"data/{s}_bpe.bin", dtype=np.uint16).astype(np.int64))
                for s in ("train", "val", "test")}
    TOKENIZERS["bpe"] = (bpe_tok, data_bpe)

# step 8: TinyStories (4,096-token BPE + end-of-story token). 1 GB of tokens, so it is read from disk
# on demand (memmap) instead of being loaded into memory.
STORIES = Path("data/tinystories")
if (STORIES / "bpe_merges.json").exists() and (STORIES / "train.bin").exists():
    from step08_tinystories_tokenizer import StoryTokenizer
    story_tok = StoryTokenizer.load()
    data_stories = {s: np.memmap(STORIES / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val", "test")}
    TOKENIZERS["stories"] = (story_tok, data_stories)

# characters per token on val: turns a per-token loss into a per-character one
CHARS_PER_TOKEN = {"char": 1.0}
if "bpe" in TOKENIZERS:
    CHARS_PER_TOKEN["bpe"] = len(data["val"]) / len(data_bpe["val"])
if "stories" in TOKENIZERS:
    CHARS_PER_TOKEN["stories"] = len((STORIES / "val.txt").read_text(encoding="utf-8")) / len(data_stories["val"])


def get_batch(split, batch_size, block_size, tokenizer="char"):
    """Random chunks of text. x = input tokens, y = the same chunk shifted by one (the answers)."""
    d = TOKENIZERS[tokenizer][1][split]
    starts = torch.randint(len(d) - block_size - 1, (batch_size,))
    if isinstance(d, np.ndarray):   # memmap on disk (TinyStories)
        x = torch.from_numpy(np.stack([d[i:i + block_size] for i in starts.tolist()]).astype(np.int64))
        y = torch.from_numpy(np.stack([d[i + 1:i + block_size + 1] for i in starts.tolist()]).astype(np.int64))
    else:
        x = torch.stack([d[i:i + block_size] for i in starts])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, batch_size, block_size, eval_batches=100, splits=("train", "val"), tokenizer="char"):
    model.eval()
    out = {}
    for split in splits:
        losses = [model(*get_batch(split, batch_size, block_size, tokenizer))[1].item() for _ in range(eval_batches)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def generate(model, block_size, n_new, start="\n", temperature=1.0, top_k=None, tokenizer="char"):
    """Write n_new tokens, one at a time, each time feeding back what was written so far."""
    model.eval()
    tk = TOKENIZERS[tokenizer][0]
    idx = torch.tensor([tk.encode(start)], dtype=torch.long, device=device)
    for _ in range(n_new):
        logits, _ = model(idx[:, -block_size:])          # the model can only see block_size characters
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return tk.decode(idx[0].tolist())


def train(model, name, settings, steps, lr, batch_size, block_size, eval_every, sample_every=None):
    """Plain training loop with validation, best-model saving and a history file."""
    out = Path(f"results/{name}")
    out.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {name} | device: {device} | parameters: {n_params:,} | "
          f"random-guess loss: {math.log(tok.vocab_size):.3f} | bigram baseline val loss: 2.478")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history, samples, best = [], [], float("inf")
    t0 = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0:
            l = estimate_loss(model, batch_size, block_size)
            mark = ""
            if l["val"] < best:
                best = l["val"]
                torch.save(model.state_dict(), out / "best.pt")
                mark = "  (best, saved)"
            history.append({"step": step, **l})
            print(f"step {step:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} "
                  f"| val perplexity {math.exp(l['val']):5.1f} | {time.time() - t0:5.0f}s{mark}")
        if sample_every and step % sample_every == 0 and step > 0:
            s = generate(model, block_size, 200)
            samples.append({"step": step, "text": s})
        if step == steps:
            break
        x, y = get_batch("train", batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    (out / "history.json").write_text(json.dumps(
        {"model": name, "params": n_params, "settings": settings, "best_val_loss": best,
         "seconds": round(time.time() - t0), "history": history, "samples": samples}, indent=2))
    print(f"Best val loss {best:.3f} (perplexity {math.exp(best):.1f}) -> {out}/best.pt, history.json")
    return model
