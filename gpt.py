"""
Step 5: a real GPT (GPT-2 design, small). Written from scratch in plain PyTorch.

  text -> token + position embeddings
       -> N x Block:  x = x + Attention(LayerNorm(x))      (multi-head, causal)
                      x = x + FeedForward(LayerNorm(x))    (width -> 4x width -> width, GELU)
       -> final LayerNorm -> output layer (shares weights with the token embedding)

Training recipe (the one most small-GPT projects use):
  AdamW, weight decay only on weight matrices, warm-up then cosine learning rate,
  gradient clipping at 1.0, validation every N steps, keep the best checkpoint.

Run:
  .venv/bin/python gpt.py small          # ~0.8M parameters, a few minutes
  .venv/bin/python gpt.py full           # ~10.7M parameters (nanoGPT's Shakespeare config)
  .venv/bin/python gpt.py full --time    # only time 20 steps, to estimate the full run
  .venv/bin/python gpt.py full_bpe       # step 8b: same model as full, but reads BPE pieces
"""

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import TOKENIZERS, data, device, estimate_loss, generate, get_batch


@dataclass
class Config:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int     # how many characters the model can see
    dropout: float
    batch_size: int
    steps: int
    lr: float
    min_lr: float
    warmup: int
    beta2: float
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    sample_every: int = 1000
    seed: int = 1337
    vocab_size: int = 65
    tokenizer: str = "char"   # "char" (65 characters) or "bpe" (1,024 pieces, step 8)
    patience: int = 0         # early stopping: stop after this many checks without a new best (0 = never)


SIZES = {
    # small: like nanoGPT's laptop config
    "small": Config(n_layer=4, n_head=4, n_embd=128, block_size=64, dropout=0.1, batch_size=32,
                    steps=5000, lr=1e-3, min_lr=1e-4, warmup=200, beta2=0.99),
    # full: nanoGPT's train_shakespeare_char config
    "full": Config(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2, batch_size=64,
                   steps=5000, lr=1e-3, min_lr=1e-4, warmup=100, beta2=0.99, eval_every=250),
    # step 8b: exactly "full", only the tokenizer changes. Stops early once val stops improving.
    "full_bpe": Config(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2, batch_size=64,
                       steps=5000, lr=1e-3, min_lr=1e-4, warmup=100, beta2=0.99, eval_every=250,
                       vocab_size=1024, tokenizer="bpe", patience=4),
}


# ------------------------------------------------------------------ model
class CausalSelfAttention(nn.Module):
    """Multi-head attention: n_head small heads look back in parallel, then their results are joined."""

    def __init__(self, c):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)      # query, key, value for all heads at once
        self.proj = nn.Linear(c.n_embd, c.n_embd)          # join the heads back together
        self.dropout = c.dropout
        self.resid_drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # the same maths as step 4 (scores -> mask the future -> softmax -> mix values), done fast
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    """The 'thinking' part: each character processes what attention gathered."""

    def __init__(self, c):
        super().__init__()
        self.up = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.down = nn.Linear(4 * c.n_embd, c.n_embd)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.ff = FeedForward(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # look back      (residual: add, don't replace)
        x = x + self.ff(self.ln2(x))     # think about it
        return x


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying: input and output share one table

        # GPT-2 style start values: small random weights; residual outputs scaled down by depth
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


# ------------------------------------------------------------------ training
def lr_at(step, c):
    """Warm up linearly, then follow a cosine curve down to min_lr."""
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    progress = (step - c.warmup) / max(1, c.steps - c.warmup)
    return c.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (c.lr - c.min_lr)


def make_optimizer(model, c):
    """Weight decay only on 2-D weight matrices (not on biases or LayerNorm)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": c.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=c.lr, betas=(0.9, c.beta2))


def run(size, time_only=False):
    c = SIZES[size]
    torch.manual_seed(c.seed)
    model = GPT(c).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = f"gpt_{size}"
    out = Path(f"results/{name}")
    out.mkdir(parents=True, exist_ok=True)
    optimizer = make_optimizer(model, c)
    print(f"Model: {name} | device: {device} | parameters: {n_params:,} | layers {c.n_layer}, heads {c.n_head}, "
          f"width {c.n_embd}, context {c.block_size}")
    # BPE loss is per piece; divide by characters-per-piece to compare with the character models
    chars_per_token = len(data["val"]) / len(TOKENIZERS[c.tokenizer][1]["val"])
    print("Targets (per character): random 4.174 | bigram 2.478 | attention 2.330 | gpt_full val 1.385")
    if c.tokenizer != "char":
        print(f"Tokenizer: {c.tokenizer}, {c.vocab_size} pieces, {chars_per_token:.2f} characters per piece "
              f"-> context {c.block_size} pieces ≈ {c.block_size * chars_per_token:.0f} characters")

    if time_only:
        for i in range(25):
            if i == 5:
                if device == "mps":
                    torch.mps.synchronize()
                t0 = time.time()
            x, y = get_batch("train", c.batch_size, c.block_size, c.tokenizer)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        per_step = (time.time() - t0) / 20
        est_eval = (c.steps // c.eval_every + 1) * 200 * per_step / 3   # eval = forward only, ~1/3 cost
        print(f"~{per_step * 1000:.0f} ms per step -> {c.steps} steps ≈ {(c.steps * per_step + est_eval) / 60:.0f} min "
              f"(including validation)")
        return

    history, samples, best, t0, since_best = [], [], float("inf"), time.time(), 0
    for step in range(c.steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, c)

        if step % c.eval_every == 0:
            l = estimate_loss(model, c.batch_size, c.block_size, tokenizer=c.tokenizer)
            mark = ""
            if l["val"] < best:
                best, since_best = l["val"], 0
                torch.save(model.state_dict(), out / "best.pt")
                mark = "  (best, saved)"
            else:
                since_best += 1
            per_char = l["val"] / chars_per_token
            history.append({"step": step, "lr": lr_at(step, c), **l, "val_per_char": per_char})
            extra = f" | val per char {per_char:.3f}" if c.tokenizer != "char" else ""
            print(f"step {step:5d} | train {l['train']:.3f} | val {l['val']:.3f} | val ppl {math.exp(l['val']):5.2f}"
                  f"{extra} | lr {lr_at(step, c):.1e} | {time.time() - t0:5.0f}s{mark}", flush=True)
            if c.patience and since_best >= c.patience:
                print(f"Early stop: no new best for {c.patience} checks ({c.patience * c.eval_every} steps).")
                samples.append({"step": step, "text": generate(model, c.block_size, round(250 / chars_per_token),
                                                               tokenizer=c.tokenizer)})
                break

        if step % c.sample_every == 0:
            samples.append({"step": step, "text": generate(model, c.block_size, round(250 / chars_per_token),
                                                           tokenizer=c.tokenizer)})

        if step == c.steps:
            break
        x, y = get_batch("train", c.batch_size, c.block_size, c.tokenizer)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        optimizer.step()

    # reload the best checkpoint and write a final sample from it
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    final = generate(model, c.block_size, round(600 / chars_per_token), tokenizer=c.tokenizer)
    (out / "history.json").write_text(json.dumps(
        {"model": name, "params": n_params, "config": asdict(c), "best_val_loss": best,
         "best_val_per_char": best / chars_per_token, "chars_per_token": chars_per_token,
         "seconds": round(time.time() - t0), "history": history, "samples": samples, "final_sample": final},
        indent=2))
    print(f"\nBest val loss {best:.3f} (perplexity {math.exp(best):.2f}) | per character {best / chars_per_token:.3f} "
          f"(gpt_full: 1.385) | {time.time() - t0:.0f}s total")
    print(f"Saved -> {out}/best.pt, history.json")
    print("\n--- text from the best model ---")
    print(final)


if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    run(size, time_only="--time" in sys.argv)
