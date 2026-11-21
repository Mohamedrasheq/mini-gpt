"""
Step 9b: test the stories model's common sense with HellaSwag.

HellaSwag has 10,042 questions (the validation set; the test answers are hidden). Each question
is the start of a short scene plus 4 possible endings, and the file says which ending is right:
    start : "A man is sitting on a roof. He"
    endings: "is using wrap to wrap a pair of skis." / "is ripping level tiles off." (right) / ...

The model writes nothing. For each ending it scores how likely that ending is after the start
(log-probability of the ending's tokens only). The highest-scoring ending is the model's pick.
Two scores, both standard:
    acc       pick by total log-probability (favours short endings)
    acc_norm  pick by log-probability per character of the ending (the number most models report)
Random guessing = 25%; humans about 95%.

Source: https://huggingface.co/datasets/Rowan/hellaswag (Zellers et al., 2019, MIT licence). Used for testing only.

Run:
  .venv/bin/python step09b_hellaswag.py            # all 10,042 questions
  .venv/bin/python step09b_hellaswag.py --quick    # first 200 questions, a quick check
Saves results/gpt_<size>/hellaswag.json (size: stories by default, or --size full_bpe etc.)
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from common import TOKENIZERS, device
from step06_evaluate import load_model

DIR = Path("data/hellaswag")
URL = "https://datasets-server.huggingface.co/rows?dataset=Rowan/hellaswag&config=default&split=validation"
BATCH = 256


def download():
    """Fetch the validation set once (skipped if already there), 100 rows per request."""
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / "hellaswag_val.jsonl"
    if path.exists():
        return path
    pages = DIR / "pages"                                  # each page is kept, so a re-run resumes
    pages.mkdir(exist_ok=True)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        cache = pages / f"{offset}.json"
        for attempt in range(10):
            if cache.exists():
                break
            got = subprocess.run(["curl", "-s", "--fail", "-L", f"{URL}&offset={offset}&length=100"],
                                 capture_output=True, text=True)
            if got.returncode == 0:
                cache.write_text(got.stdout)
                break
            time.sleep(5 * (attempt + 1))                  # the server limits fast requests: wait, retry
        else:
            raise SystemExit(f"Download stopped at row {offset:,}. Run the script again to continue.")
        page = json.loads(cache.read_text())
        total = page["num_rows_total"]
        rows += [r["row"] for r in page["rows"]]
        offset += 100
        print(f"\r  downloading HellaSwag: {len(rows):,} of {total:,}", end="", flush=True)
        time.sleep(0.3)
    print()
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@torch.no_grad()
def ending_scores(model, tok, start_id, block_size, pairs):
    """For each (start, ending): total log-probability of the ending's tokens, given the start."""
    scores = []
    for i in range(0, len(pairs), BATCH):
        seqs, first = [], []
        for ctx, end in pairs[i:i + BATCH]:
            c = [start_id] + tok.encode(ctx)
            e = tok.encode(" " + end)
            s = (c + e)[-(block_size + 1):]                  # too long: drop the oldest start tokens
            seqs.append(s)
            first.append(max(1, len(s) - len(e)))            # where the ending begins
        width = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros(len(seqs), width - 1)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
            mask[j, first[j] - 1:len(s) - 1] = 1             # only the ending's tokens count
        x, mask = x.to(device), mask.to(device)
        logits, _ = model(x[:, :-1])
        logp = F.log_softmax(logits.float(), dim=-1)
        token_logp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        scores.extend((token_logp * mask).sum(dim=1).tolist())
    return scores


def pct(passed, total):
    return 100 * passed / total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="stories")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    rows = [json.loads(line) for line in download().read_text().splitlines() if line.strip()]
    if args.quick:
        rows = rows[:200]

    model, c = load_model(args.size)
    tok = TOKENIZERS[c.tokenizer][0]
    start_id = getattr(tok, "eot", None)
    if start_id is None:
        start_id = tok.encode("\n")[0]                       # character models: start after a newline
    print(f"Model: gpt_{args.size} | parameters: {sum(q.numel() for q in model.parameters()):,} | device: {device}")
    print(f"HellaSwag: {len(rows):,} questions x 4 endings\n")

    t0 = time.time()
    pairs = [(r["ctx"], e) for r in rows for e in r["endings"]]
    scores = ending_scores(model, tok, start_id, c.block_size, pairs)

    ok, ok_norm = 0, 0
    by_source = defaultdict(lambda: [0, 0])                  # [right (acc_norm), total]
    for k, r in enumerate(rows):
        s = scores[4 * k:4 * k + 4]
        s_norm = [v / len(" " + e) for v, e in zip(s, r["endings"])]
        pick, pick_norm = s.index(max(s)), s_norm.index(max(s_norm))
        label = int(r["label"])
        ok += pick == label
        ok_norm += pick_norm == label
        source = "WikiHow (how-to steps)" if r["source_id"].startswith("wikihow") else "ActivityNet (video scenes)"
        by_source[source][0] += pick_norm == label
        by_source[source][1] += 1

    n = len(rows)
    print("By source (acc_norm):")
    for source, (right, tot) in sorted(by_source.items()):
        print(f"  {pct(right, tot):5.1f}%  {source} ({tot:,} questions)")
    print(f"\nHellaSwag acc     : {pct(ok, n):.1f}%  ({ok:,} of {n:,})")
    print(f"HellaSwag acc_norm: {pct(ok_norm, n):.1f}%  ({ok_norm:,} of {n:,}) | {time.time() - t0:.0f}s")
    print("Reference (acc_norm): random 25% | humans ~95%")

    if args.quick:
        print("\n--quick: not saved.")
    else:
        out = Path(f"results/gpt_{args.size}/hellaswag.json")
        out.write_text(json.dumps({
            "model": f"gpt_{args.size}", "questions": n,
            "acc": round(pct(ok, n), 2), "acc_norm": round(pct(ok_norm, n), 2),
            "by_source_acc_norm": {k: round(pct(*v), 2) for k, v in by_source.items()}}, indent=2))
        print(f"Saved -> {out}")
