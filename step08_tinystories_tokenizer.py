"""
Step 8b: a BPE tokenizer for TinyStories, then encode every split into token files.

Same method as step 7 (byte-level BPE, our own code), with three changes for big data:
  1. Learn the pieces from a SAMPLE of train (spread across the whole file), not all 2.2 GB.
     Plain-Python BPE on 2.2 GB would take hours; a sample gives nearly the same pieces.
  2. A faster learning loop: after each join, only the words that contain that pair are updated.
  3. A special token <|endoftext|> that marks the end of each story.
Encoding the 2.2 GB train file runs on all CPU cores, in blocks, so memory stays low.

Run:
  .venv/bin/python step08_tinystories_tokenizer.py            # full run (sample, compare sizes, encode all)
  .venv/bin/python step08_tinystories_tokenizer.py --quick    # small check only (seconds), nothing saved

Saves (in data/tinystories/):
  bpe_merges.json                 the learned joins (the tokenizer itself)
  train.bin, val.bin, test.bin    the stories as token numbers (uint16)
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from step07_bpe_tokenizer import SPLIT, BPETokenizer, merge

DIR = Path("data/tinystories")
EOT = "<|endoftext|>"
VOCAB_SIZE = 4096                     # 256 bytes + 3,839 joins + 1 end-of-story token
COMPARE_SIZES = (2048, 4096, 8192)
SAMPLE_MB = 40                        # how much train text to learn the joins from
BLOCK_BYTES = 8 * 1024 * 1024         # encode the big file in 8 MB blocks


class StoryTokenizer(BPETokenizer):
    """Step 7's BPE plus one special token: <|endoftext|> = the last id."""

    def __init__(self, merges):
        super().__init__(merges)
        self.eot = 256 + len(self.merges)
        self.pieces[self.eot] = EOT.encode()

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + 1

    def encode(self, text):
        ids = []
        for i, part in enumerate(text.split(EOT)):
            if i > 0:
                ids.append(self.eot)
            ids.extend(super().encode(part))
        return ids

    @classmethod
    def load(cls, path=DIR / "bpe_merges.json"):
        return cls(json.loads(Path(path).read_text()))

    def save(self, path=DIR / "bpe_merges.json"):
        Path(path).write_text(json.dumps(self.merges))


def learn_merges(text, n_merges):
    """BPE learning, updating only the words that contain the chosen pair (much faster than step 7's loop)."""
    chunks = Counter(m for part in text.split(EOT) for m in SPLIT.findall(part))
    words = [list(c.encode("utf-8")) for c in chunks]
    freq = list(chunks.values())
    pairs, where = Counter(), defaultdict(set)
    for w, ids in enumerate(words):
        for p in zip(ids, ids[1:]):
            pairs[p] += freq[w]
            where[p].add(w)
    print(f"  {len(words):,} unique chunks in the sample")

    merges, t0 = [], time.time()
    for k in range(n_merges):
        best = max(pairs, key=pairs.get)
        if pairs[best] <= 0:
            break
        new_id = 256 + k
        merges.append(best)
        for w in list(where[best]):
            old = words[w]
            for p in zip(old, old[1:]):
                pairs[p] -= freq[w]
            new = merge(old, best, new_id)
            words[w] = new
            for p in zip(new, new[1:]):
                pairs[p] += freq[w]
                where[p].add(w)
        del pairs[best]
        if (k + 1) % 1024 == 0:
            print(f"  {256 + k + 1:,} pieces learned ({time.time() - t0:.0f}s)")
    return merges


def read_sample(path, mb):
    """Take mb x 1 MB slices spread evenly across the file, cut to whole stories."""
    size = path.stat().st_size
    parts = []
    with open(path, "rb") as f:
        for i in range(mb):
            f.seek(int(size * i / mb))
            block = f.read(1024 * 1024).decode("utf-8", errors="ignore")
            a, b = block.find(EOT), block.rfind(EOT)
            if a != -1 and b > a:
                parts.append(block[a + len(EOT):b])
    return EOT.join(parts)


def blocks(path):
    """Yield the file as text blocks that end exactly after an <|endoftext|> marker."""
    marker = EOT.encode()
    with open(path, "rb") as f:
        rest = b""
        while True:
            data = f.read(BLOCK_BYTES)
            if not data:
                if rest:
                    yield rest.decode("utf-8")
                return
            data = rest + data
            cut = data.rfind(marker)
            if cut == -1:
                rest = data
                continue
            cut += len(marker)
            yield data[:cut].decode("utf-8")
            rest = data[cut:]


_tok = None


def _init_worker(merges):
    global _tok
    _tok = StoryTokenizer(merges)


def _encode_block(text):
    return np.array(_tok.encode(text), dtype=np.uint16)


def encode_file(tok, src, dst, workers):
    t0, n_tokens, n_bytes = time.time(), 0, src.stat().st_size
    with open(dst, "wb") as out, Pool(workers, _init_worker, (tok.merges,)) as pool:
        for i, arr in enumerate(pool.imap(_encode_block, blocks(src))):
            arr.tofile(out)
            n_tokens += len(arr)
            done = min(1.0, (i + 1) * BLOCK_BYTES / n_bytes)
            if (i + 1) % 25 == 0:
                left = (time.time() - t0) / done * (1 - done)
                print(f"    {done:.0%} | {n_tokens:,} tokens | about {left / 60:.0f} min left", flush=True)
    return n_tokens, time.time() - t0


def show(tok, text):
    ids = tok.encode(text)
    print(f"  {len(text)} characters -> {len(ids)} tokens: " + " | ".join(repr(tok.piece(i))[1:-1] for i in ids))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    sample_mb = 2 if quick else SAMPLE_MB
    sizes = (512, 1024) if quick else COMPARE_SIZES
    vocab = 1024 if quick else VOCAB_SIZE

    print(f"Learning joins from a {sample_mb} MB sample of train (spread across the file)...")
    t0 = time.time()
    sample = read_sample(DIR / "train.txt", sample_mb)
    big = learn_merges(sample, max(sizes) - 257)
    print(f"  done in {time.time() - t0:.0f}s")

    val = (DIR / "val.txt").read_text(encoding="utf-8")
    sample_counts_text = sample[: 10 * 1024 * 1024]
    print("\nWhich vocabulary size?  (chars per token on val; 'rare' = seen < 50 times in 10 MB of train)")
    print(f"  {'size':>5} | {'chars per token':>15} | {'rare pieces':>11}")
    for size in sizes:
        tk = StoryTokenizer(big[:size - 257])
        counts = Counter(tk.encode(sample_counts_text))
        rare = sum(1 for i in range(256, size) if counts[i] < 50)       # learned pieces only
        print(f"  {size:>5} | {len(val) / len(tk.encode(val)):>15.2f} | {rare:>5} ({rare / (size - 256):.0%})")

    tok = StoryTokenizer(big[:vocab - 257])
    print(f"\nChosen: {tok.vocab_size} tokens (end-of-story token = {tok.eot})")
    print("Longest pieces: " + "  ".join(repr(tok.piece(i)) for i in
                                         sorted(range(256, tok.eot), key=lambda i: -len(tok.pieces[i]))[:12]))
    print("Examples:")
    for s in ["Once upon a time, there was a little girl named Lily.",
              "Tom said, “I am sorry,” and gave the ball back.<|endoftext|>"]:
        show(tok, s)
    assert tok.decode(tok.encode(val[:100_000])) == val[:100_000], "round trip failed"
    print("Round trip OK")

    if quick:
        print("\n--quick: stopping here, nothing saved.")
        sys.exit()

    tok.save()
    print(f"Saved -> {DIR}/bpe_merges.json")
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"\nEncoding all splits with {workers} CPU workers (train takes the longest)...")
    for name in ("val", "test", "train"):
        n, secs = encode_file(tok, DIR / f"{name}.txt", DIR / f"{name}.bin", workers)
        chars = len((DIR / f"{name}.txt").read_text(encoding="utf-8")) if name != "train" else None
        extra = f" | {chars / n:.2f} chars per token" if chars else ""
        print(f"  {name:5s}: {n:>12,} tokens{extra} | {secs / 60:.1f} min -> {DIR}/{name}.bin", flush=True)
    print("\nDone.")
