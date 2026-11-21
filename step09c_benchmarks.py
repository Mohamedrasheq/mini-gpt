"""
Step 9c: run the same 8 tests on our models and compare them.

Same code, same data, same scoring for every model, so the numbers are fair to compare.
Every test works the same way: the model writes nothing, it only scores text.
    context + each possible answer  ->  log-probability of the answer's tokens  ->  highest wins

  test          what it checks                          choices  random  scored by
  tinystories   predicting unseen stories (loss)        -        -       loss per character (lower = better)
  fineweb_edu   predicting unseen web pages (loss)      -        -       loss per character (lower = better)
  blimp         grammar: correct vs wrong sentence      2        50%     total log-prob
  hellaswag     common sense: sensible scene ending     4        25%     log-prob per character
  lambada       last word of a passage (needs context)  -        0%      the model's top pick = the right word
  piqa          everyday physical sense                 2        50%     log-prob per character
  arc_easy      school science questions                3-5      ~25%    log-prob per character
  winogrande    who does "she" / "it" refer to          2        50%     log-prob of the rest of the sentence

Loss is compared per CHARACTER, not per token: the models cut text into different tokens.
The question prompts follow lm-evaluation-harness, the tool most model releases use.
Datasets come from Hugging Face (test only, never trained on).

Run:
  .venv/bin/python step09c_benchmarks.py --model stories     # model 6 (step 8)
  .venv/bin/python step09c_benchmarks.py --model fineweb     # model 7 (step 10)
  .venv/bin/python step09c_benchmarks.py --table             # side-by-side table
  add --quick to try only 200 items per test (nothing saved); --only <tests> to run some tests
Saves results/benchmarks/<model>.json and results/benchmarks/compare.md
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import TOKENIZERS, device
from step06_evaluate import load_model
from step09_blimp import DIR as BLIMP_DIR, download as blimp_download
from step09b_hellaswag import download as hellaswag_download

DATA = Path("data/benchmarks")
OUT = Path("results/benchmarks")
ROWS_API = "https://datasets-server.huggingface.co/rows"
SOURCES = {                                   # name: (Hugging Face dataset, config, split)
    "lambada": ("EleutherAI/lambada_openai", "default", "test"),
    "piqa": ("baber/piqa", "default", "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
}
TOKEN_BUDGET = 2 ** 27                        # tokens x vocabulary per batch (keeps memory low)


# ---------------------------------------------------------------- data

def hf_rows(name):
    """Download one test set from Hugging Face once, 100 rows per request (resumes if stopped)."""
    path = DATA / f"{name}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    dataset, config, split = SOURCES[name]
    pages = DATA / f"{name}_pages"
    pages.mkdir(parents=True, exist_ok=True)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        cache = pages / f"{offset}.json"
        for attempt in range(10):
            if cache.exists():
                break
            url = f"{ROWS_API}?dataset={dataset}&config={config}&split={split}&offset={offset}&length=100"
            got = subprocess.run(["curl", "-s", "--fail", "-L", url], capture_output=True, text=True)
            if got.returncode == 0:
                cache.write_text(got.stdout)
                break
            time.sleep(5 * (attempt + 1))     # the server limits fast requests: wait, retry
        else:
            raise SystemExit(f"{name}: download stopped at row {offset:,}. Run again to continue.")
        page = json.loads(cache.read_text())
        total = page["num_rows_total"]
        rows += [r["row"] for r in page["rows"]]
        offset += 100
        print(f"\r  downloading {name}: {len(rows):,} of {total:,}", end="", flush=True)
        time.sleep(0.3)
    print()
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


# ---------------------------------------------------------------- models

class LM:
    """One interface for every model: encode text, start token, context length, logits."""

    def __init__(self, name, ckpt=None, tokenizer=None):
        self.name = name
        if name == "fineweb":                 # step 10: ~30M model, 8K tokenizer
            from step08_tinystories_tokenizer import StoryTokenizer
            from step10_model import GPT, Config
            ck = torch.load(ckpt or "results/gpt_fineweb/best.pt", map_location=device, weights_only=True)
            net = GPT(Config(**ck["config"])).to(device).eval()
            net.load_state_dict(ck["model"])
            tok = StoryTokenizer.load(tokenizer or "data/fineweb/tokenizer_8k.json")
            self.encode, self.start, self.max_len = tok.encode, tok.eot, net.c.block_size
            self.vocab, self.forward = tok.vocab_size, lambda x: net(x)[0]
        else:
            net, c = load_model(name)
            tok = TOKENIZERS[c.tokenizer][0]
            self.encode, self.start, self.max_len = tok.encode, tok.eot, c.block_size
            self.vocab, self.forward = tok.vocab_size, lambda x: net(x)[0]
        self.params = sum(p.numel() for p in net.parameters())


@torch.no_grad()
def score(lm, pairs):
    """For each (context, answer): log-prob of the answer's tokens, and whether every one was the top pick."""
    enc = [([lm.start] + lm.encode(ctx), lm.encode(ans)) for ctx, ans in pairs]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]))   # similar lengths together
    logp_out, greedy_out = [0.0] * len(enc), [False] * len(enc)
    i = 0
    while i < len(order):
        n = max(1, TOKEN_BUDGET // (lm.vocab * lm.max_len))
        while i + n < len(order) and n * min(lm.max_len + 1, sum(map(len, enc[order[i + n]]))) * lm.vocab < TOKEN_BUDGET:
            n += 1
        idx = order[i:i + n]
        seqs, first = [], []
        for k in idx:
            c, a = enc[k]
            s = (c + a)[-(lm.max_len + 1):]   # too long: drop the oldest context tokens
            seqs.append(s)
            first.append(max(1, len(s) - len(a)))
        width = max(map(len, seqs))
        x = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros(len(seqs), width - 1, dtype=torch.bool)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
            mask[j, first[j] - 1:len(s) - 1] = True          # only the answer's tokens count
        x, mask = x.to(device), mask.to(device)
        logits = lm.forward(x[:, :-1]).float()
        target = x[:, 1:]
        tok_logp = -F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        top = logits.argmax(-1) == target
        lp = (tok_logp * mask).sum(1).tolist()
        gr = (top | ~mask).all(1).tolist()
        for j, k in enumerate(idx):
            logp_out[k], greedy_out[k] = lp[j], gr[j]
        i += n
    return logp_out, greedy_out


def pick(scores, answers, per_char):
    """Index of the best answer: by total log-prob, or log-prob per character."""
    vals = [s / len(a) for s, a in zip(scores, answers)] if per_char else scores
    return vals.index(max(vals))


def multiple_choice(lm, questions):
    """questions: list of (context, [answers], right index). Returns acc and acc_norm (%)."""
    pairs = [(ctx, a) for ctx, answers, _ in questions for a in answers]
    s, _ = score(lm, pairs)
    acc = acc_norm = pos = 0
    for ctx, answers, right in questions:
        part = s[pos:pos + len(answers)]
        pos += len(answers)
        acc += pick(part, answers, False) == right
        acc_norm += pick(part, answers, True) == right
    n = len(questions)
    return {"acc": round(100 * acc / n, 2), "acc_norm": round(100 * acc_norm / n, 2), "items": n}


# ---------------------------------------------------------------- the 7 tests

def test_tinystories(lm, quick):
    """Loss per character on the TinyStories test split, in windows of the model's full context."""
    text = Path("data/tinystories/test.txt").read_text(encoding="utf-8")
    if quick:
        text = text[:200_000]
    if lm.name == "stories" and not quick:    # same tokens as test.txt, already encoded in step 8b
        ids = np.fromfile("data/tinystories/test.bin", dtype=np.uint16).astype(np.int64).tolist()
    else:
        ids = lm.encode(text)
    return loss_per_char(lm, ids, len(text))


def test_fineweb_edu(lm, quick):
    """Loss per character on 5,000 held-out FineWeb-Edu documents (file 013, never trained on)."""
    from step08_tinystories_tokenizer import EOT
    from step10b_data import FW_URL, HELDOUT_DOCS, HELDOUT_FILE, fetch, parquet_docs
    raw = Path("data/fineweb/raw")
    raw.mkdir(parents=True, exist_ok=True)
    path = fetch(FW_URL.format(HELDOUT_FILE), raw / f"fw_{HELDOUT_FILE}.parquet")
    it = parquet_docs(path, skip=HELDOUT_DOCS)            # the second 5,000 = step 10's test documents
    docs = [next(it) for _ in range(200 if quick else HELDOUT_DOCS)]
    ids = [i for d in docs for i in lm.encode(d + EOT)]
    return loss_per_char(lm, ids, sum(len(d) for d in docs))


def loss_per_char(lm, ids, n_chars):
    """Total loss over all tokens / number of characters (fair between different tokenizers)."""
    L, total = lm.max_len, 0.0
    windows = [ids[k:k + L + 1] for k in range(0, len(ids) - 1, L)]
    full, last = [w for w in windows if len(w) == L + 1], [w for w in windows if 1 < len(w) < L + 1]
    per_batch = max(1, TOKEN_BUDGET // (lm.vocab * L))
    with torch.no_grad():
        for group in [full[b:b + per_batch] for b in range(0, len(full), per_batch)] + [[w] for w in last]:
            x = torch.tensor(group, device=device)
            logits = lm.forward(x[:, :-1]).float()
            total += F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="sum").item()
    return {"loss_per_char": round(total / n_chars, 4), "chars_per_token": round(n_chars / len(ids), 2),
            "items": n_chars}


def test_blimp(lm, quick):
    blimp_download()
    files = sorted(BLIMP_DIR.glob("*.jsonl"))
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    if quick:
        rows = rows[::len(rows) // 200][:200]
    pairs = [("", r[k]) for r in rows for k in ("sentence_good", "sentence_bad")]
    s, _ = score(lm, pairs)
    ok = sum(s[2 * k] > s[2 * k + 1] for k in range(len(rows)))
    return {"acc": round(100 * ok / len(rows), 2), "items": len(rows)}


def test_hellaswag(lm, quick):
    rows = [json.loads(line) for line in hellaswag_download().read_text().splitlines() if line.strip()]
    if quick:
        rows = rows[:200]
    return multiple_choice(lm, [(r["ctx"], [" " + e for e in r["endings"]], int(r["label"])) for r in rows])


def test_lambada(lm, quick):
    rows = hf_rows("lambada")[:200 if quick else None]
    pairs = [(t[:t.rindex(" ")], t[t.rindex(" "):]) for t in (r["text"] for r in rows)]
    s, greedy = score(lm, pairs)
    n_tokens = sum(len(lm.encode(a)) for _, a in pairs)
    return {"acc": round(100 * sum(greedy) / len(rows), 2),
            "perplexity": round(math.exp(-sum(s) / n_tokens), 2), "items": len(rows)}


def test_piqa(lm, quick):
    rows = hf_rows("piqa")[:200 if quick else None]
    return multiple_choice(lm, [(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]],
                                 int(r["label"])) for r in rows])


def test_arc_easy(lm, quick):
    rows = hf_rows("arc_easy")[:200 if quick else None]
    return multiple_choice(lm, [(f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                                 r["choices"]["label"].index(r["answerKey"])) for r in rows])


def test_winogrande(lm, quick):
    """Put each option in the blank; score the words AFTER the blank (same words for both options)."""
    rows = hf_rows("winogrande")[:200 if quick else None]
    pairs = []
    for r in rows:
        cut = r["sentence"].index("_")
        rest = " " + r["sentence"][cut + 1:].strip()
        pairs += [(r["sentence"][:cut] + r["option1"], rest), (r["sentence"][:cut] + r["option2"], rest)]
    s, _ = score(lm, pairs)
    ok = sum((0 if s[2 * k] > s[2 * k + 1] else 1) == int(r["answer"]) - 1 for k, r in enumerate(rows))
    return {"acc": round(100 * ok / len(rows), 2), "items": len(rows)}


TESTS = {"tinystories": test_tinystories, "fineweb_edu": test_fineweb_edu, "blimp": test_blimp, "hellaswag": test_hellaswag,
         "lambada": test_lambada, "piqa": test_piqa, "arc_easy": test_arc_easy, "winogrande": test_winogrande}

# what goes in the table: (test, metric, random-guess value, higher is better?)
HEADLINE = [("tinystories", "loss_per_char", None, False), ("fineweb_edu", "loss_per_char", None, False), ("blimp", "acc", 50, True),
            ("hellaswag", "acc_norm", 25, True), ("lambada", "acc", 0, True), ("piqa", "acc_norm", 50, True),
            ("arc_easy", "acc_norm", 25, True), ("winogrande", "acc", 50, True)]


NAMES = {"stories": "Model 6", "fineweb": "Model 7"}


def table():
    res = {m: json.loads((OUT / f"{m}.json").read_text()) for m in NAMES if (OUT / f"{m}.json").exists()}
    if not res:
        raise SystemExit("Run --model ... first.")
    head = " | ".join(f"{NAMES[m]} ({r['parameters'] / 1e6:.1f}M)" for m, r in res.items())
    lines = [f"| Test | Metric | Random | {head} |", "|---|---|---|" + "---|" * len(res)]
    for test, metric, rand, higher in HEADLINE:
        vals = {m: r["tests"].get(test, {}).get(metric) for m, r in res.items()}
        known = [v for v in vals.values() if v is not None]
        best = (max if higher else min)(known) if known else None
        unit = "" if metric == "loss_per_char" else "%"
        cells = " | ".join("-" if v is None else f"**{v}{unit}**" if v == best and len(known) > 1 else f"{v}{unit}"
                           for v in vals.values())
        lines.append(f"| {test} | {metric} | {'-' if rand is None else f'{rand}%'} | {cells} |")
    text = "\n".join(lines)
    print(text)
    (OUT / "compare.md").write_text(text + "\n")
    print(f"\nSaved -> {OUT}/compare.md  (best value in bold; loss: lower is better)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stories", help="stories (model 6) or fineweb (model 7)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--table", action="store_true")
    p.add_argument("--only", nargs="*", help="run only these tests (merged into saved results)")
    p.add_argument("--ckpt", help="fineweb only: checkpoint path (default results/gpt_fineweb/best.pt)")
    p.add_argument("--tokenizer", help="fineweb only: tokenizer path (default data/fineweb/tokenizer_8k.json)")
    p.add_argument("--save_as", help="results name (default: the model name)")
    args = p.parse_args()
    if args.table:
        table()
        raise SystemExit

    lm = LM(args.model, args.ckpt, args.tokenizer)
    print(f"Model: {args.model} | parameters: {lm.params:,} | context: {lm.max_len} tokens | device: {device}\n")
    results, t_all = {}, time.time()
    for name, fn in TESTS.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        results[name] = fn(lm, args.quick)
        shown = "  ".join(f"{k} {v}" for k, v in results[name].items())
        print(f"  {name:12s} {shown}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nAll done in {time.time() - t_all:.0f}s")

    if args.quick:
        print("--quick: not saved.")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        out = OUT / f"{args.save_as or args.model}.json"
        saved = json.loads(out.read_text()) if out.exists() else {"tests": {}}
        saved["tests"].update(results)            # --only keeps the other tests' earlier results
        out.write_text(json.dumps({"model": args.model, "parameters": lm.params, "tests": saved["tests"]}, indent=2))
        print(f"Saved -> {out}")
