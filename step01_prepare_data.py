"""
Step 1: look at the training text and split it into train / validation / test.

Input : data/shakespeare.txt   (Tiny Shakespeare, ~1 MB of plays)
Output: data/train.txt, data/val.txt, data/test.txt

Split by POSITION (first 90% / next 5% / last 5%), cut at a blank line so no speech
is broken in half. Random lines would leak: near-identical lines ("All: Speak, speak.")
would land in both train and test.

Run: .venv/bin/python step01_prepare_data.py
"""

from collections import Counter
from pathlib import Path

DATA = Path("data")
SPLITS = {"train": 0.90, "val": 0.05, "test": 0.05}

text = (DATA / "shakespeare.txt").read_text(encoding="utf-8")

# ---------- 1. Basic size ----------
lines = text.splitlines()
words = text.split()
print("=== Size ===")
print(f"characters : {len(text):,}")
print(f"lines      : {len(lines):,}")
print(f"words      : {len(words):,}")
print(f"speeches   : {text.count(chr(10) * 2):,}  (blocks separated by a blank line)")

# ---------- 2. Vocabulary: every unique character ----------
chars = sorted(set(text))
print(f"\n=== Vocabulary: {len(chars)} unique characters ===")
print(repr("".join(chars)))

counts = Counter(text)
print("\nMost common :", ", ".join(f"{c!r} {n:,}" for c, n in counts.most_common(8)))
print("Least common:", ", ".join(f"{c!r} {n}" for c, n in counts.most_common()[-6:]))


# ---------- 3. Split by position, at blank lines ----------
def cut_at_blank_line(pos: int) -> int:
    """Move a cut point forward to the next blank line, so speeches stay whole."""
    nxt = text.find("\n\n", pos)
    return len(text) if nxt == -1 else nxt + 2


cut1 = cut_at_blank_line(int(len(text) * SPLITS["train"]))
cut2 = cut_at_blank_line(int(len(text) * (SPLITS["train"] + SPLITS["val"])))
parts = {"train": text[:cut1], "val": text[cut1:cut2], "test": text[cut2:]}

print("\n=== Split (by position) ===")
for name, part in parts.items():
    (DATA / f"{name}.txt").write_text(part, encoding="utf-8")
    print(f"{name:5s}: {len(part):>9,} chars ({len(part) / len(text):.1%})  -> data/{name}.txt")

# Characters that appear in val/test but never in train can't be learned
train_chars = set(parts["train"])
for name in ("val", "test"):
    unseen = set(parts[name]) - train_chars
    print(f"characters in {name} never seen in train: {sorted(unseen) or 'none'}")

# ---------- 4. A peek at each split ----------
for name, part in parts.items():
    print(f"\n--- start of {name} ---")
    print(part[:160].rstrip())
