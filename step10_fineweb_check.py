"""
Step 10 (check first): can model 6's tokenizer handle FineWeb-Edu, or do we need a new one?

FineWeb-Edu = English web pages that read like teaching material (Hugging Face, 2024, ODC-By 1.0).
This script downloads a ~50 MB sample (spread across the whole 10B-token sample) and measures:
  1. chars per token   : how much text fits in one token (higher = better, fewer cuts)
  2. whole words       : share of words that stay ONE token (higher = better)
for model 6's tokenizer (learned on children's stories), for fresh tokenizers learned on
FineWeb-Edu (4,096 / 8,192 / 16,384 pieces).

Decision:
  model 6's tokenizer close to a fresh 4,096 one  -> Option A: keep it, continue training model 6
  much worse (more than ~10% fewer chars/token)   -> Option B: new tokenizer, train from scratch

Nothing is trained on the GPU. The fresh tokenizers are only for measuring (not saved).

Run:  .venv/bin/python step10_fineweb_check.py
Saves results/fineweb_check.json; the sample stays in data/fineweb/ (gitignored).
"""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from step07_bpe_tokenizer import SPLIT, BPETokenizer
from step08_tinystories_tokenizer import EOT, StoryTokenizer, learn_merges

DIR = Path("data/fineweb")
ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train"
TOTAL_DOCS = 9_672_101                 # documents in sample-10BT
PAGES = 150                            # 150 x 100 documents, spread evenly
TARGET_MB = 50
LEARN_MB = 10                          # fresh tokenizers learn from this much FineWeb text
FRESH_SIZES = (4096, 8192, 16384)
EXAMPLES = ["photosynthesis", "government", "temperature", "approximately", "independence", "the"]


def download():
    """~50 MB of FineWeb-Edu: 100 documents from each of 150 places across the sample (resumes if stopped)."""
    out = DIR / "sample.txt"
    if out.exists():
        return out.read_text(encoding="utf-8").split(f"\n{EOT}\n")
    pages = DIR / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    docs, size = [], 0
    for p in range(PAGES):
        cache = pages / f"{p}.json"
        offset = p * (TOTAL_DOCS // PAGES)
        for attempt in range(10):
            if cache.exists():
                break
            got = subprocess.run(["curl", "-s", "--fail", "-L", f"{ROWS_API}&offset={offset}&length=100"],
                                 capture_output=True, text=True)
            if got.returncode == 0:
                cache.write_text(got.stdout)
                break
            time.sleep(5 * (attempt + 1))  # the server limits fast requests: wait, retry
        else:
            raise SystemExit(f"Download stopped at page {p}. Run again to continue.")
        for r in json.loads(cache.read_text())["rows"]:
            if not r.get("truncated_cells"):
                docs.append(r["row"]["text"].strip())
                size += len(docs[-1].encode("utf-8"))
        print(f"\r  downloading FineWeb-Edu: {size / 1e6:.0f} of ~{TARGET_MB} MB", end="", flush=True)
        if size >= TARGET_MB * 1e6:
            break
        time.sleep(0.3)
    print()
    out.write_text(f"\n{EOT}\n".join(docs), encoding="utf-8")
    return docs


def chunk_counts(text):
    """Each regex chunk once, with how often it appears (BPE works chunk by chunk, so this is exact)."""
    return Counter(m for part in text.split(EOT) for m in SPLIT.findall(part))


def measure(encode, counts):
    """chars per token, and share of words (letters only) that stay one token."""
    chars = tokens = words = whole = 0
    for chunk, n in counts.items():
        k = len(encode(chunk))
        chars += n * len(chunk)
        tokens += n * k
        if chunk.strip().isalpha():
            words += n
            whole += n * (k == 1)
    return chars / tokens, 100 * whole / words


def pieces(encode, decode_piece, word):
    return " | ".join(decode_piece(i) for i in encode(" " + word))


if __name__ == "__main__":
    docs = download()
    cut = int(len(docs) * 0.8)                         # 80% for learning, 20% held out for measuring
    learn_text = "\n\n".join(docs[:cut])
    test_text = "\n\n".join(docs[cut:])
    print(f"FineWeb-Edu sample: {len(docs):,} documents | learn part {len(learn_text) / 1e6:.0f}M chars | "
          f"held-out part {len(test_text) / 1e6:.0f}M chars")
    stories_val = Path("data/tinystories/val.txt").read_text(encoding="utf-8")[:10_000_000]

    fw_counts, st_counts = chunk_counts(test_text), chunk_counts(stories_val)
    rows = []

    # 1. model 6's tokenizer
    m6 = StoryTokenizer.load()
    cpt_st, _ = measure(m6.encode, st_counts)
    cpt_fw, whole_fw = measure(m6.encode, fw_counts)
    rows.append(("model 6 tokenizer (stories)", 4096, cpt_fw, whole_fw))
    examples = {"model 6 tokenizer (stories)": [pieces(m6.encode, m6.piece, w) for w in EXAMPLES]}

    # 2. fresh tokenizers learned on FineWeb-Edu (learn once up to 16,384; smaller = the first joins)
    print(f"\nLearning fresh tokenizers from {LEARN_MB} MB of FineWeb-Edu...")
    t0 = time.time()
    merges = learn_merges(learn_text[:LEARN_MB * 1_000_000], max(FRESH_SIZES) - 257)
    print(f"  done in {time.time() - t0:.0f}s")
    for size in FRESH_SIZES:
        tk = BPETokenizer(merges[:size - 257])
        cpt, whole = measure(tk.encode, fw_counts)
        name = f"fresh, learned on FineWeb ({size:,})"
        rows.append((name, size, cpt, whole))
        examples[name] = [pieces(tk.encode, tk.piece, w) for w in EXAMPLES]

    print(f"\nModel 6's tokenizer on its own data (TinyStories): {cpt_st:.2f} chars per token")
    print("\nOn FineWeb-Edu (held-out part):")
    print(f"  {'tokenizer':36s} | {'vocab':>6} | {'chars per token':>15} | {'whole words':>11}")
    for name, size, cpt, whole in rows:
        print(f"  {name:36s} | {size:>6,} | {cpt:>15.2f} | {whole:>10.1f}%")

    print("\nHow some words are cut:")
    for name, cuts in examples.items():
        print(f"  {name}")
        for w, c in zip(EXAMPLES, cuts):
            print(f"    {w:15s} -> {c}")

    fresh4k = rows[1][2]
    gap = 100 * (fresh4k - cpt_fw) / fresh4k
    print(f"\nModel 6's tokenizer vs a fresh 4,096 one on FineWeb-Edu: {gap:.0f}% fewer chars per token")
    print("Suggestion: " + ("Option A (keep the tokenizer, continue model 6)" if gap <= 10
                            else "Option B (new tokenizer, train from scratch)"))

    out = Path("results/fineweb_check.json")
    out.write_text(json.dumps({
        "sample_docs": len(docs), "stories_chars_per_token_model6": round(cpt_st, 3),
        "fineweb": [{"tokenizer": n, "vocab": s, "chars_per_token": round(c, 3), "whole_words_pct": round(w, 1)}
                    for n, s, c, w in rows],
        "gap_vs_fresh_4096_pct": round(gap, 1), "examples": examples}, indent=2))
    print(f"Saved -> {out}")
