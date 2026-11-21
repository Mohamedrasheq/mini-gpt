"""
Byte-level BPE tokenizer, written from scratch: 8,192 tokens, with an end-of-text token.

How it learns:
  1. Start with 256 pieces, one for every byte, so any text can be encoded.
  2. Cut the text into chunks first (words, numbers, punctuation, spaces), so a piece never joins two words.
  3. Count every pair of neighbouring pieces; join the most common pair into a new piece. Repeat.
How it encodes: apply the same joins in the order they were learned.
The last id (8,191) is <|endoftext|>, which ends every document.

It's learned from a 40 MB sample of the training mix: 90% FineWeb-Edu + 10% TinyStories.
The result is saved in data/tokenizer_8k.json, so you only need this script to re-create it.

Run:
  .venv/bin/python tokenizer.py            # learn from 40 MB, save data/tokenizer_8k.json
  .venv/bin/python tokenizer.py --quick    # learn from 4 MB, save data/tokenizer_8k_quick.json (for tests)
"""

import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

EOT = "<|endoftext|>"
VOCAB_SIZE = 8192                         # 256 bytes + 7,935 joins + 1 end-of-text token
DATA = Path("data")

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


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(p) for p in merges]                     # learned order matters
        self.rank = {p: 256 + i for i, p in enumerate(self.merges)}  # pair -> new piece id
        self.pieces = {i: bytes([i]) for i in range(256)}            # id -> bytes
        for (a, b), i in self.rank.items():
            self.pieces[i] = self.pieces[a] + self.pieces[b]
        self.eot = 256 + len(self.merges)
        self.pieces[self.eot] = EOT.encode()
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + 1

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))   # earliest join first
            if pair not in self.rank:
                break
            ids = merge(ids, pair, self.rank[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for i, part in enumerate(text.split(EOT)):
            if i > 0:
                ids.append(self.eot)
            ids.extend(t for chunk in SPLIT.findall(part) for t in self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        return b"".join(self.pieces[int(i)] for i in ids).decode("utf-8", errors="replace")

    def piece(self, i):
        return self.pieces[int(i)].decode("utf-8", errors="replace")

    def save(self, path):
        Path(path).write_text(json.dumps(self.merges))

    @classmethod
    def load(cls, path=DATA / "tokenizer_8k.json"):
        return cls(json.loads(Path(path).read_text()))


def learn_merges(text, n_merges):
    """BPE learning that only updates the chunks containing the chosen pair (fast enough for 40 MB)."""
    chunks = Counter(m for part in text.split(EOT) for m in SPLIT.findall(part))
    words = [list(c.encode("utf-8")) for c in chunks]
    freq = list(chunks.values())
    pairs, where = Counter(), defaultdict(set)
    for w, ids in enumerate(words):
        for p in zip(ids, ids[1:]):
            pairs[p] += freq[w]
            where[p].add(w)
    print(f"  {len(words):,} unique chunks in the sample")
    merges = []
    for k in range(n_merges):
        best = max(pairs, key=pairs.get)
        if pairs[best] <= 0:
            break
        merges.append(best)
        for w in list(where[best]):
            old = words[w]
            for p in zip(old, old[1:]):
                pairs[p] -= freq[w]
            words[w] = merge(old, best, 256 + k)
            for p in zip(words[w], words[w][1:]):
                pairs[p] += freq[w]
                where[p].add(w)
        del pairs[best]
        if (k + 1) % 1024 == 0:
            print(f"  {256 + k + 1:,} pieces learned")
    return merges


# ---------------------------------------------------------------- the learning sample

ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train"
TOTAL_DOCS = 9_672_101                    # documents in FineWeb-Edu sample-10BT
PAGES, SAMPLE_MB = 150, 50                # 100 documents from each of up to 150 places across the sample
STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"


def fineweb_sample(folder=DATA / "fineweb" / "tokenizer_sample"):
    """~50 MB of FineWeb-Edu documents spread across the whole sample (resumes if stopped)."""
    out = folder / "sample.txt"
    if out.exists():
        return out.read_text(encoding="utf-8").split(f"\n{EOT}\n")
    folder.mkdir(parents=True, exist_ok=True)
    docs, size = [], 0
    for p in range(PAGES):
        cache = folder / f"page_{p}.json"
        for attempt in range(10):
            if cache.exists():
                break
            url = f"{ROWS_API}&offset={p * (TOTAL_DOCS // PAGES)}&length=100"
            got = subprocess.run(["curl", "-s", "--fail", "-L", url], capture_output=True, text=True)
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
        print(f"\r  downloading the FineWeb-Edu sample: {size / 1e6:.0f} of ~{SAMPLE_MB} MB", end="", flush=True)
        if size >= SAMPLE_MB * 1e6:
            break
        time.sleep(0.3)
    print()
    out.write_text(f"\n{EOT}\n".join(docs), encoding="utf-8")
    return docs


def stories_sample(mb, path=DATA / "fineweb" / "tokenizer_sample" / "stories.txt"):
    """mb MB of TinyStories from the start of the official train file, cut to whole stories."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-L", "--fail", "-s", "-S", "-r", f"0-{int(mb * 1e6) + 100_000}", "-o", str(path),
                        STORIES_URL], check=True)
    text = path.read_bytes().decode("utf-8", errors="ignore")
    return text[:text.rfind(EOT, 0, int(mb * 1e6))]


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    mb = 4 if quick else 40
    fw = EOT.join(fineweb_sample())[:int(mb * 1e6 * 0.9)]
    st = stories_sample(mb * 0.1)
    text = fw + EOT + st
    print(f"Learning {VOCAB_SIZE:,} pieces from {len(text) / 1e6:.0f}M characters "
          f"({len(fw) / 1e6:.0f}M FineWeb-Edu + {len(st) / 1e6:.0f}M TinyStories)...")
    tok = Tokenizer(learn_merges(text, VOCAB_SIZE - 257))
    print(f"  vocabulary {tok.vocab_size:,} | end-of-text token = {tok.eot}")

    check = fineweb_sample()[-1][:20_000]                    # a document kept out of the learning text
    assert tok.decode(tok.encode(check)) == check, "round trip failed"
    for s in ["The government measured the temperature approximately.",
              "Once upon a time, there was a little girl named Lily."]:
        ids = tok.encode(s)
        print(f"  {len(s)} chars -> {len(ids)} tokens: " + " | ".join(tok.piece(i) for i in ids))
    path = DATA / ("tokenizer_8k_quick.json" if quick else "tokenizer_8k.json")
    tok.save(path)
    print(f"Round trip OK. Saved -> {path}")
