"""
Step 2: character tokenizer. Text -> numbers (encode) and numbers -> text (decode).

Each unique character in the TRAINING text gets one number (its position in the sorted list).
The vocabulary is built from train only, the same rule as the data split: nothing from val/test
is used to build any part of the model.

Run: .venv/bin/python tokenizer.py
  - builds data/vocab.json
  - encodes train / val / test into data/*.bin (one byte per character), ready for training
  - prints a demo and checks that decode(encode(text)) == text
"""

import json
from pathlib import Path

import numpy as np

DATA = Path("data")


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string -> id
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # id -> string

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @classmethod
    def load(cls, path=DATA / "vocab.json"):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path=DATA / "vocab.json"):
        Path(path).write_text(json.dumps(self.chars, ensure_ascii=False), encoding="utf-8")

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


if __name__ == "__main__":
    train = (DATA / "train.txt").read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(train)
    tok.save()
    print(f"Vocabulary size: {tok.vocab_size}  -> data/vocab.json")

    # ---------- the full table ----------
    print("\nCharacter -> number")
    row = [f"{ch!r:>5}={i:<2}" for i, ch in enumerate(tok.chars)]
    for k in range(0, len(row), 8):
        print("  " + "  ".join(row[k:k + 8]))

    # ---------- demo ----------
    for sample in ["hi", "Hello, world!", "To be, or not to be"]:
        ids = tok.encode(sample)
        print(f"\n{sample!r}\n  encode -> {ids}\n  decode -> {tok.decode(ids)!r}")

    # ---------- encode every split and check the round trip ----------
    print()
    for name in ("train", "val", "test"):
        text = (DATA / f"{name}.txt").read_text(encoding="utf-8")
        ids = np.array(tok.encode(text), dtype=np.uint8)  # 65 < 256, so 1 byte each
        assert tok.decode(ids) == text, f"round trip failed for {name}"
        ids.tofile(DATA / f"{name}.bin")
        print(f"{name:5s}: {len(ids):>9,} tokens -> data/{name}.bin   (round trip OK)")

    # ---------- what the model will actually see ----------
    first = np.fromfile(DATA / "train.bin", dtype=np.uint8)[:40]
    print("\nFirst 40 tokens of train, as the model sees them:")
    print(" ", first.tolist())
    print("  =", repr(tok.decode(first)))
