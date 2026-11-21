"""
Step 10a: an 8,192-piece BPE tokenizer for the FineWeb-Edu + TinyStories mix.

Same method as step 8b (byte-level BPE, our own code, end-of-text token), learned from the same
mix the model will train on: 90% FineWeb-Edu (the sample from step 10's check) + 10% TinyStories.
Step 10's check measured 8,192 pieces: 3.82 characters per token on FineWeb-Edu (model 6's: 2.72).

Run:
  .venv/bin/python step10a_tokenizer.py            # learn from 40 MB, save
  .venv/bin/python step10a_tokenizer.py --quick    # learn from 4 MB, save to a test file
Saves data/fineweb/tokenizer_8k.json (or tokenizer_8k_quick.json)
"""

import sys
import time
from pathlib import Path

from step08_tinystories_tokenizer import EOT, StoryTokenizer, learn_merges, read_sample
from step10_fineweb_check import download as fineweb_sample

DIR = Path("data/fineweb")
VOCAB_SIZE = 8192                     # 256 bytes + 7,935 joins + 1 end-of-text token
LEARN_MB = 40
STORIES_SHARE = 0.10


def tokenizer_path(quick=False):
    return DIR / ("tokenizer_8k_quick.json" if quick else "tokenizer_8k.json")


def load(quick=False):
    return StoryTokenizer.load(tokenizer_path(quick))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    mb = 4 if quick else LEARN_MB
    fw_chars = int(mb * 1e6 * (1 - STORIES_SHARE))
    fw = EOT.join(fineweb_sample())[:fw_chars]
    st = read_sample(Path("data/tinystories/train.txt"), max(1, round(mb * STORIES_SHARE)))
    text = fw + EOT + st
    print(f"Learning {VOCAB_SIZE:,} pieces from {len(text) / 1e6:.0f}M characters "
          f"({len(fw) / 1e6:.0f}M FineWeb-Edu + {len(st) / 1e6:.0f}M TinyStories)...")
    t0 = time.time()
    tok = StoryTokenizer(learn_merges(text, VOCAB_SIZE - 257))
    print(f"  done in {time.time() - t0:.0f}s | vocabulary {tok.vocab_size:,} | end-of-text token = {tok.eot}")

    check = fineweb_sample()[-1][:20_000]               # a document kept out of the learning text
    assert tok.decode(tok.encode(check)) == check, "round trip failed"
    for s in ["The government measured the temperature approximately.",
              "Once upon a time, there was a little girl named Lily."]:
        ids = tok.encode(s)
        print(f"  {len(s)} chars -> {len(ids)} tokens: " + " | ".join(tok.piece(i) for i in ids))
    print("Round trip OK")
    tok.save(tokenizer_path(quick))
    print(f"Saved -> {tokenizer_path(quick)}")
