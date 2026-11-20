"""
Step 7: our own BPE tokenizer (byte-level, the same idea as GPT-2), written from scratch.

How it learns (train text only):
  1. Start with 256 pieces: one for every byte (so ANY text can be encoded, nothing is "unknown").
  2. Cut the text into chunks first (words, spaces, punctuation), so a piece never joins two words.
  3. Count every pair of neighbouring pieces. Join the most common pair into one new piece.
  4. Repeat until the vocabulary has VOCAB_SIZE pieces.

How it encodes new text: apply the same joins, in the same order they were learned.

Run: .venv/bin/python step07_bpe_tokenizer.py
  - learns the pieces, compares vocabulary sizes, shows examples
  - saves data/bpe_merges.json and data/{train,val,test}_bpe.bin (uint16)
"""

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path("data")
VOCAB_SIZE = 1024
COMPARE_SIZES = (512, 1024, 2048)

# words (with their leading space), numbers, punctuation, and spaces / newlines, kept apart
SPLIT = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def merge(ids, pair, new_id):
    """Replace every (a, b) neighbour pair in ids with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges):
        self.merges = [tuple(p) for p in merges]                     # learned order matters
        self.rank = {p: 256 + i for i, p in enumerate(self.merges)}  # pair -> new piece id
        self.pieces = {i: bytes([i]) for i in range(256)}            # id -> bytes
        for (a, b), i in self.rank.items():
            self.pieces[i] = self.pieces[a] + self.pieces[b]
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size, verbose=False):
        # count each unique chunk once (fast): {(bytes...): how often}
        chunks = Counter(tuple(m.encode("utf-8")) for m in SPLIT.findall(text))
        words = {w: list(w) for w in chunks}
        merges = []
        for k in range(vocab_size - 256):
            pairs = Counter()
            for w, ids in words.items():
                f = chunks[w]
                for p in zip(ids, ids[1:]):
                    pairs[p] += f
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = 256 + k
            merges.append(best)
            for w in words:
                if len(words[w]) > 1:
                    words[w] = merge(words[w], best, new_id)
            if verbose and (k + 1) % 256 == 0:
                print(f"  {256 + k + 1} pieces learned")
        return cls(merges)

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # join the pair that was learned earliest
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = merge(ids, pair, self.rank[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        return [i for chunk in SPLIT.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids):
        return b"".join(self.pieces[int(i)] for i in ids).decode("utf-8", errors="replace")

    def piece(self, i):
        return self.pieces[int(i)].decode("utf-8", errors="replace")

    def save(self, path=DATA / "bpe_merges.json"):
        Path(path).write_text(json.dumps(self.merges))

    @classmethod
    def load(cls, path=DATA / "bpe_merges.json"):
        return cls(json.loads(Path(path).read_text()))


def show(tok, text):
    ids = tok.encode(text)
    parts = " | ".join(repr(tok.piece(i))[1:-1] for i in ids)
    print(f"  {text!r}\n    {len(text)} characters -> {len(ids)} pieces:  {parts}")


if __name__ == "__main__":
    text = {s: (DATA / f"{s}.txt").read_text(encoding="utf-8") for s in ("train", "val", "test")}

    # ---------- learn once up to the largest size; smaller sizes = the first merges ----------
    t0 = time.time()
    print(f"Learning up to {max(COMPARE_SIZES)} pieces from train ({len(text['train']):,} characters)...")
    big = BPETokenizer.train(text["train"], max(COMPARE_SIZES), verbose=True)
    print(f"  done in {time.time() - t0:.0f}s")

    print("\nWhich vocabulary size?  (val text; 'rare' = pieces seen fewer than 50 times in train)")
    print(f"  {'size':>5} | {'chars per piece':>15} | {'train pieces':>12} | {'rare pieces':>11}")
    for size in COMPARE_SIZES:
        tk = BPETokenizer(big.merges[:size - 256])
        tr = tk.encode(text["train"])
        counts = Counter(tr)
        rare = sum(1 for i in range(size) if counts[i] < 50)
        cpp = len(text["val"]) / len(tk.encode(text["val"]))
        print(f"  {size:>5} | {cpp:>15.2f} | {len(tr):>12,} | {rare:>5} ({rare / size:.0%})")
    print(f"  (character tokenizer for reference: 1.00 chars per piece, {len(text['train']):,} train pieces)")

    tok = BPETokenizer(big.merges[:VOCAB_SIZE - 256])
    tok.save()
    print(f"\nChosen: {VOCAB_SIZE} pieces -> data/bpe_merges.json")

    print("\nFirst 20 joins it learned (the most common pairs):")
    print("  " + "  ".join(repr(tok.piece(256 + i)) for i in range(20)))
    longest = sorted(range(256, tok.vocab_size), key=lambda i: -len(tok.pieces[i]))[:15]
    print("Longest pieces:")
    print("  " + "  ".join(repr(tok.piece(i)) for i in longest))

    print("\nExamples:")
    for s in ["ROMEO: What news?", "To be, or not to be, that is the question.",
              "First Citizen:\nBefore we proceed any further, hear me speak.", "Hello, ChatGPT!"]:
        show(tok, s)

    print()
    for name in ("train", "val", "test"):
        ids = tok.encode(text[name])
        assert tok.decode(ids) == text[name], f"round trip failed for {name}"
        np.array(ids, dtype=np.uint16).tofile(DATA / f"{name}_bpe.bin")
        print(f"{name:5s}: {len(text[name]):>9,} characters -> {len(ids):>7,} pieces "
              f"({len(text[name]) / len(ids):.2f} chars each) -> data/{name}_bpe.bin   (round trip OK)")
