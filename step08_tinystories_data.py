"""
Step 8a: get TinyStories (V2, written by GPT-4) and split it into train / val / test.

Source : https://huggingface.co/datasets/roneneldan/TinyStories  (license: CDLA-Sharing-1.0)
Files  : TinyStoriesV2-GPT4-train.txt (2.2 GB) and TinyStoriesV2-GPT4-valid.txt (22 MB)
         Stories are separated by the line <|endoftext|>.

Split:
  train = the whole official train file
  val   = first half of the official valid file  (used during training to pick the best checkpoint)
  test  = second half of the official valid file (used once, at the end)
  The valid file is cut at a story boundary, so no story is split in two.

The data stays local (data/tinystories/ is in .gitignore); anyone can re-download it with this script.

Run: .venv/bin/python step08_tinystories_data.py
"""

import subprocess
from collections import Counter
from pathlib import Path

OUT = Path("data/tinystories")
BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train_raw.txt": "TinyStoriesV2-GPT4-train.txt", "valid_raw.txt": "TinyStoriesV2-GPT4-valid.txt"}
EOT = "<|endoftext|>"


def download():
    OUT.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        path = OUT / local
        print(f"Downloading {remote} -> {path}")
        # -C - resumes a half-finished download; curl skips the work if the file is already complete
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(path), BASE + remote], check=False)


def stories(text):
    return [s.strip() for s in text.split(EOT) if s.strip()]


if __name__ == "__main__":
    download()

    train = stories((OUT / "train_raw.txt").read_text(encoding="utf-8"))
    valid = stories((OUT / "valid_raw.txt").read_text(encoding="utf-8"))
    half = len(valid) // 2
    parts = {"train": train, "val": valid[:half], "test": valid[half:]}

    print("\n=== Split ===")
    for name, items in parts.items():
        joined = f"\n{EOT}\n".join(items) + f"\n{EOT}\n"   # keep the end-of-story marker between stories
        (OUT / f"{name}.txt").write_text(joined, encoding="utf-8")
        n_chars = sum(len(s) for s in items)
        print(f"{name:5s}: {len(items):>9,} stories | {n_chars:>13,} characters | "
              f"avg {n_chars / len(items):,.0f} characters per story -> {OUT}/{name}.txt")

    # leak check: the same story text in train and val/test
    train_set = set(train)
    for name in ("val", "test"):
        dup = sum(s in train_set for s in parts[name])
        print(f"stories in {name} that also appear word-for-word in train: {dup} ({dup / len(parts[name]):.1%})")

    # characters: what the tokenizer will have to handle
    sample = "".join(train[:200_000])
    counts = Counter(sample)
    rare = [c for c, n in counts.items() if not c.isascii()]
    print(f"\nUnique characters (first 200K stories): {len(counts)} | non-ASCII: {len(rare)} "
          f"e.g. {''.join(sorted(rare, key=lambda c: -counts[c])[:20])!r}")

    print("\n--- first train story ---")
    print(train[0])
