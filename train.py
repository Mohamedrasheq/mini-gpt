"""
Train the ~30M model (model.py) on the FineWeb-Edu + TinyStories mix (prepare_data.py).

Made for one NVIDIA GPU (e.g. a Colab GPU runtime); also runs on an Apple GPU or the CPU for small tests.
  - 16-bit math (bf16) on GPUs that support it; the master weights stay 32-bit
  - each batch: ~90% FineWeb-Edu sequences, ~10% TinyStories sequences
  - 262,144 tokens per step (micro-batches x gradient accumulation); steps = training tokens / that
  - warm-up then cosine learning rate, AdamW (weight decay on 2D weights only), gradient clip 1.0
  - checks val loss (FineWeb-Edu and TinyStories) every 250 steps, keeps best.pt
  - saves a full checkpoint (model + optimizer + step) every 500 steps: if Colab disconnects,
    run the same command again and it continues from the last checkpoint

Run:
  python train.py --data /content/data --out /content/drive/MyDrive/mini-gpt/model7
  python train.py --data ... --out ... --speed_test     # 30 steps: speed and full-run time estimate
  python train.py --smoke                               # tiny model + quick data (prepare_data.py --quick)
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, Config
from tokenizer import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/fineweb")
p.add_argument("--out", default="results/model7")
p.add_argument("--tokenizer", default="data/tokenizer_8k.json")
p.add_argument("--tokens", type=float, default=0, help="training tokens (default: all in the train files)")
p.add_argument("--batch_tokens", type=int, default=262_144)
p.add_argument("--micro", type=int, default=64, help="sequences per micro-batch (lower it if out of memory)")
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--min_lr", type=float, default=1.5e-4)
p.add_argument("--warmup", type=int, default=300)
p.add_argument("--stories_share", type=float, default=0.10)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--eval_batches", type=int, default=20)
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--sample_every", type=int, default=1000)
p.add_argument("--compile", type=int, default=-1, help="1/0; default: on for NVIDIA GPUs")
p.add_argument("--speed_test", action="store_true")
p.add_argument("--units_per_hour", type=float, default=0, help="speed test: also estimate cloud cost (units per hour)")
p.add_argument("--smoke", action="store_true")
p.add_argument("--steps", type=int, default=0, help="stop after this many steps (testing)")
args = p.parse_args()

if args.smoke:                                   # tiny everything: a quick check of the whole code path
    args.data, args.out = "data/fineweb/quick", "results/smoke_test"
    args.tokenizer = "data/tokenizer_8k_quick.json"
    args.batch_tokens, args.micro, args.warmup = 8 * 128 * 2, 8, 5
    args.eval_every, args.eval_batches, args.save_every, args.sample_every = 10, 4, 10, 20
    args.steps = args.steps or 30
    cfg = Config(block_size=128, n_layer=2, n_head=4, n_embd=128, ff_hidden=352)
else:
    cfg = Config()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
compile_model = args.compile == 1 or (args.compile == -1 and device == "cuda")

data = Path(args.data)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
fw = {s: np.memmap(data / f"fw_{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
st = {s: np.memmap(data / f"st_{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
tok = Tokenizer.load(args.tokenizer)
assert tok.vocab_size == cfg.vocab_size, f"tokenizer has {tok.vocab_size} pieces, model expects {cfg.vocab_size}"

T = cfg.block_size
accum = max(1, args.batch_tokens // (args.micro * T))
step_tokens = accum * args.micro * T
train_tokens = args.tokens or (len(fw["train"]) + len(st["train"]))
max_steps = math.ceil(train_tokens / step_tokens)
log_file = None if args.speed_test else open(out / "train.log", "a")   # a speed test leaves no log


def log(msg):
    print(msg, flush=True)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def rows(arr, starts):
    x = torch.from_numpy(np.stack([arr[i:i + T + 1].astype(np.int64) for i in starts]))
    return x[:, :-1], x[:, 1:]


def get_batch(rng):
    """One micro-batch: each sequence comes from TinyStories with probability stories_share."""
    from_st = rng.random(args.micro) < args.stories_share
    parts = []
    for src, n in ((st["train"], int(from_st.sum())), (fw["train"], int((~from_st).sum()))):
        if n:
            parts.append(rows(src, rng.integers(0, len(src) - T - 1, n)))
    x, y = torch.cat([a for a, _ in parts]), torch.cat([b for _, b in parts])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast(device, enabled=False)


@torch.no_grad()
def evaluate(model):
    """Average loss on the same fixed val windows every time (comparable between checks)."""
    model.eval()
    res = {}
    for name, arr in (("fineweb", fw["val"]), ("stories", st["val"])):
        rng = np.random.default_rng(0)
        losses = []
        for _ in range(args.eval_batches):
            x, y = rows(arr, rng.integers(0, len(arr) - T - 1, args.micro))
            with autocast():
                _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        res[name] = sum(losses) / len(losses)
    model.train()
    return res


@torch.no_grad()
def sample(model, start, n=120, temperature=0.8, top_k=50):
    model.eval()
    idx = torch.tensor([tok.encode(start)], device=device)
    for _ in range(n):
        with autocast():
            logits, _ = model(idx[:, -T:])
        logits = logits[:, -1].float() / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        if nxt.item() == tok.eot:
            break
        idx = torch.cat([idx, nxt], 1)
    model.train()
    return tok.decode(idx[0].tolist())


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, max_steps - args.warmup))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)                        # never leaves a half-written file


# ---------------------------------------------------------------- model, optimizer, resume
torch.manual_seed(1337)
model = GPT(cfg).to(device)
n_params = sum(q.numel() for q in model.parameters())
decay = [q for q in model.parameters() if q.dim() >= 2]
no_decay = [q for q in model.parameters() if q.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=device == "cuda")

step, best, history = 0, float("inf"), {"evals": [], "config": asdict(cfg), "args": vars(args)}
ckpt_path = out / "ckpt.pt"
if ckpt_path.exists() and not args.speed_test:
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    step, best, history = ck["step"], ck["best"], ck["history"]
    log(f"Resumed from step {step:,} (best FineWeb val so far {best:.4f})")
rng = np.random.default_rng(1337 + step)         # different batches after a resume

run = torch.compile(model) if compile_model else model
log(f"device {device} | 16-bit (bf16): {use_bf16} | compiled: {compile_model} | parameters {n_params:,}")
log(f"train tokens {train_tokens:,} (FineWeb-Edu {len(fw['train']):,} + TinyStories {len(st['train']):,}) | "
    f"{step_tokens:,} tokens per step ({args.micro} x {T} x {accum} accumulation) | {max_steps:,} steps")

# ---------------------------------------------------------------- training loop
stop = max_steps if not args.steps else min(max_steps, args.steps)
if args.speed_test:
    stop = step + 30
t_last, tok_since, speed_log = time.time(), 0, []
while step < stop:
    if step % args.eval_every == 0 and not args.speed_test:
        ev = evaluate(run)
        history["evals"].append({"step": step, **ev})
        mark = ""
        if ev["fineweb"] < best:
            best = ev["fineweb"]
            save(out / "best.pt", {"model": model.state_dict(), "config": asdict(cfg), "step": step, "val": ev})
            mark = "  <- best, saved"
        log(f"  eval step {step:,}: val FineWeb-Edu {ev['fineweb']:.4f} | val TinyStories {ev['stories']:.4f}{mark}")
    if step % args.sample_every == 0 and step > 0 and not args.speed_test:
        log("  sample: " + sample(run, "The").replace("\n", " ")[:400])
    if step % args.save_every == 0 and step > 0 and not args.speed_test:
        save(ckpt_path, {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                         "best": best, "history": history})

    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    total = 0.0
    for _ in range(accum):
        x, y = get_batch(rng)
        with autocast():
            _, loss = run(x, y)
        (loss / accum).backward()
        total += loss.item() / accum
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    tok_since += step_tokens

    if step % 10 == 0 or step == stop:
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t_last
        speed = tok_since / dt
        speed_log.append(speed)
        left = (max_steps - step) * step_tokens / speed
        log(f"step {step:>6,}/{max_steps:,} | loss {total:.4f} | lr {lr:.2e} | grad norm {norm:.2f} | "
            f"{speed / 1e3:,.0f}K tokens/s | about {left / 3600:.1f} h left")
        t_last, tok_since = time.time(), 0

if args.speed_test:
    speed = float(np.median(speed_log[1:] or speed_log))        # skip the first (compile / warm-up)
    hours = train_tokens / speed / 3600
    mem = f" | GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device == "cuda" else ""
    log(f"\nSPEED TEST: {speed / 1e3:,.0f}K tokens/s{mem}")
    cost = f" = about {hours * args.units_per_hour:.0f} units at {args.units_per_hour}/hour" if args.units_per_hour else ""
    log(f"Full run ({train_tokens / 1e9:.2f}B tokens): about {hours:.1f} h{cost}")
else:
    ev = evaluate(run)
    history["evals"].append({"step": step, **ev})
    if ev["fineweb"] < best:
        best = ev["fineweb"]
        save(out / "best.pt", {"model": model.state_dict(), "config": asdict(cfg), "step": step, "val": ev})
    save(ckpt_path, {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                     "best": best, "history": history})
    (out / "history.json").write_text(json.dumps(history, indent=2))
    log(f"Stopped at step {step:,}. Final val FineWeb-Edu {ev['fineweb']:.4f} | TinyStories {ev['stories']:.4f} "
        f"| best FineWeb-Edu {best:.4f}")
    log("sample: " + sample(run, "The").replace("\n", " ")[:400])
