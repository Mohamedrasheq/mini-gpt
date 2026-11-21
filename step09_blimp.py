"""
Step 9: test the stories model's grammar with BLiMP (Benchmark of Linguistic Minimal Pairs).

BLiMP has 67 files x 1,000 sentence pairs. Each pair differs in one small thing, and the file says
which sentence is correct:
    good: "Paula references Robert."      bad: "Paula reference Robert."

The model writes nothing. It scores both sentences: the total log-probability of every token,
starting right after an end-of-story token (so the first word is scored too). A pair passes when
the correct sentence scores higher. Score = pairs passed / all pairs.
Random guessing = 50%; humans about 89%.

Source: https://github.com/alexwarstadt/blimp (Warstadt et al., 2020). Used for testing only.

Run:
  .venv/bin/python step09_blimp.py            # all 67,000 pairs
  .venv/bin/python step09_blimp.py --quick    # 2 files only, a quick check
Saves results/gpt_<size>/blimp.json (size: stories by default, or --size full_bpe etc.)
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

DIR = Path("data/blimp")
API = "https://api.github.com/repos/alexwarstadt/blimp/contents/data"
RAW = "https://raw.githubusercontent.com/alexwarstadt/blimp/master/data/"
BATCH = 256


def download():
    """Fetch the 67 BLiMP files once (skipped if already there)."""
    DIR.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(["curl", "-s", "-H", "User-Agent: mini-gpt", API],
                             capture_output=True, text=True, check=True).stdout
    names = sorted(x["name"] for x in json.loads(listing) if x["name"].endswith(".jsonl"))
    missing = [n for n in names if not (DIR / n).exists()]
    for n in missing:
        subprocess.run(["curl", "-s", "--fail", "-o", str(DIR / n), RAW + n], check=True)
    print(f"BLiMP: {len(names)} files in {DIR} ({len(missing)} downloaded now)")
    return names


@torch.no_grad()
def sentence_scores(model, tok, start_id, sentences):
    """Total log-probability of each sentence, scored after a start token."""
    scores = []
    for i in range(0, len(sentences), BATCH):
        seqs = [[start_id] + tok.encode(s) for s in sentences[i:i + BATCH]]
        width = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros(len(seqs), width - 1)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s) - 1] = 1                       # which positions are real tokens
        x, mask = x.to(device), mask.to(device)
        logits, _ = model(x[:, :-1])
        logp = F.log_softmax(logits.float(), dim=-1)
        token_logp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)   # log P(each next token)
        scores.extend((token_logp * mask).sum(dim=1).tolist())
    return scores


def pct(passed, total):
    return 100 * passed / total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="stories")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    names = download()
    if args.quick:
        names = names[:2]

    model, c = load_model(args.size)
    tok = TOKENIZERS[c.tokenizer][0]
    start_id = getattr(tok, "eot", None)
    if start_id is None:
        start_id = tok.encode("\n")[0]                     # character models: start after a newline
    print(f"Model: gpt_{args.size} | parameters: {sum(q.numel() for q in model.parameters()):,} | device: {device}\n")

    t0 = time.time()
    by_file, by_term, by_field = {}, defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for n in names:
        rows = [json.loads(line) for line in (DIR / n).read_text().splitlines() if line.strip()]
        good = sentence_scores(model, tok, start_id, [r["sentence_good"] for r in rows])
        bad = sentence_scores(model, tok, start_id, [r["sentence_bad"] for r in rows])
        passed = sum(g > b for g, b in zip(good, bad))
        uid, term = rows[0]["UID"], rows[0]["linguistics_term"]
        field = rows[0]["field"].replace("/", "_")      # one file spells it "syntax/semantics"
        by_file[uid] = {"term": term, "field": field, "passed": passed, "total": len(rows)}
        for key, table in ((term, by_term), (field, by_field)):
            table[key][0] += passed
            table[key][1] += len(rows)
        print(f"  {pct(passed, len(rows)):5.1f}%  {uid}", flush=True)

    total_pass = sum(v["passed"] for v in by_file.values())
    total = sum(v["total"] for v in by_file.values())
    print(f"\nBy grammar area:")
    for term, (ok, tot) in sorted(by_term.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print(f"  {pct(ok, tot):5.1f}%  {term} ({tot:,} pairs)")
    print(f"\nBy field:")
    for field, (ok, tot) in sorted(by_field.items()):
        print(f"  {pct(ok, tot):5.1f}%  {field}")
    print(f"\nBLiMP overall: {pct(total_pass, total):.1f}%  ({total_pass:,} of {total:,} pairs) | {time.time() - t0:.0f}s")
    print("Reference: random 50% | humans ~89%")

    if args.quick:
        print("\n--quick: not saved.")
    else:
        out = Path(f"results/gpt_{args.size}/blimp.json")
        out.write_text(json.dumps({
            "model": f"gpt_{args.size}", "overall": round(pct(total_pass, total), 2),
            "passed": total_pass, "total": total,
            "by_term": {k: round(pct(*v), 2) for k, v in by_term.items()},
            "by_field": {k: round(pct(*v), 2) for k, v in by_field.items()},
            "by_file": by_file}, indent=2))
        print(f"Saved -> {out}")
