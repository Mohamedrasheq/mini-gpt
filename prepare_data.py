"""
Download FineWeb-Edu + TinyStories and turn them into token files for training.

Mix: ~1.8B FineWeb-Edu tokens (90%) + ~0.2B TinyStories tokens (10%), with the 8K tokenizer (tokenizer.py).

  FineWeb-Edu (sample-10BT, 14 parquet files of ~2.15 GB each; ODC-By 1.0)
    train      : files 000-002, read in order until 1.8B tokens
    val / test : 5,000 + 5,000 documents from file 013 (never used for training or the tokenizer)
  TinyStories (V2, GPT-4; CDLA-Sharing-1.0)
    train      : the start of the official train file, until 0.2B tokens
    val / test : first / second half of the official valid file

Each document ends with the end-of-text token. Encoding runs on all CPU cores.
Made to run on Colab (fast download from Hugging Face), works on any computer.

Run:
  python prepare_data.py                       # full: ~7 GB download, writes ~4 GB of token files
  python prepare_data.py --quick               # small test: file 013 only (0.54 GB), 20M + 2M tokens
  python prepare_data.py --out /content/data   # where to write (default data/fineweb)
Writes fw_train.bin, fw_val.bin, fw_test.bin, st_train.bin, st_val.bin, st_test.bin (uint16)
"""

import argparse
import os
import subprocess
import time
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tokenizer import DATA, EOT, Tokenizer

FW_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{}_00000.parquet"
ST_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt"
HELDOUT_FILE, HELDOUT_DOCS = "013", 5000
DOCS_PER_JOB = 1000

_tok = None


def _init(merges):
    global _tok
    _tok = Tokenizer(merges)


def _encode_docs(docs):
    ids = []
    for d in docs:
        ids.extend(_tok.encode(d))
        ids.append(_tok.eot)
    return np.array(ids, dtype=np.uint16)


def fetch(url, path, max_bytes=None):
    """Download once (resumes a half-finished file). max_bytes: only the start of the file."""
    if path.exists() and (max_bytes is None or path.stat().st_size >= max_bytes):
        return path
    cmd = ["curl", "-L", "--fail", "-s", "-S", "-o", str(path), url]
    cmd += ["-r", f"0-{max_bytes - 1}"] if max_bytes else ["-C", "-"]
    print(f"  downloading {url.rsplit('/', 1)[-1]}{' (start only)' if max_bytes else ''}...", flush=True)
    subprocess.run(cmd, check=True)
    return path


def parquet_docs(path, skip=0):
    import pyarrow.parquet as pq
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=DOCS_PER_JOB, columns=["text"]):
        for d in batch.column("text").to_pylist():
            n += 1
            if n > skip:
                yield d


def jobs(docs):
    job = []
    for d in docs:
        job.append(d)
        if len(job) == DOCS_PER_JOB:
            yield job
            job = []
    if job:
        yield job


def write_tokens(pool, docs, dst, limit=None):
    """Encode documents on all cores and write them to dst, stopping at `limit` tokens."""
    t0, n, todo = time.time(), 0, jobs(docs)
    with open(dst, "wb") as out:
        while not (limit and n >= limit):
            group = list(islice(todo, 4 * pool._processes))   # a few jobs at a time: memory stays low
            if not group:
                break
            for arr in pool.map(_encode_docs, group):
                if limit and n + len(arr) > limit:
                    arr = arr[:limit - n]
                arr.tofile(out)
                n += len(arr)
                if n // 50_000_000 != (n - len(arr)) // 50_000_000:
                    rate = n / (time.time() - t0)
                    left = f" | about {(limit - n) / rate / 60:.0f} min left" if limit else ""
                    print(f"    {dst.name}: {n / 1e6:,.0f}M tokens{left}", flush=True)
                if limit and n >= limit:
                    break
    print(f"  {dst.name:13s} {n:>13,} tokens ({time.time() - t0:.0f}s)", flush=True)
    return n


def stories(text):
    return [s.strip() for s in text.split(EOT) if s.strip()]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fineweb")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    out = Path(args.out) / ("quick" if args.quick else "")
    raw = Path(args.out) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    fw_target, st_target = (20_000_000, 2_000_000) if args.quick else (1_800_000_000, 200_000_000)
    train_files = [HELDOUT_FILE] if args.quick else ["000", "001", "002"]
    tok = Tokenizer.load(DATA / ("tokenizer_8k_quick.json" if args.quick else "tokenizer_8k.json"))
    workers = os.cpu_count() or 2
    print(f"Tokenizer: {tok.vocab_size:,} pieces | workers: {workers} | output: {out}\n")

    print("Downloading...")
    held = fetch(FW_URL.format(HELDOUT_FILE), raw / f"fw_{HELDOUT_FILE}.parquet")
    files = [fetch(FW_URL.format(f), raw / f"fw_{f}.parquet") for f in train_files]
    st_valid = fetch(ST_URL.format("valid"), raw / "st_valid.txt")
    st_train = fetch(ST_URL.format("train"), raw / "st_train_start.txt",
                     max_bytes=10_000_000 if args.quick else 900_000_000)   # ~0.2B tokens needs ~0.8 GB

    with Pool(workers, _init, (tok.merges,)) as pool:
        print("\nFineWeb-Edu (held-out documents from file 013):")
        it = parquet_docs(held)
        write_tokens(pool, (next(it) for _ in range(HELDOUT_DOCS)), out / "fw_val.bin")
        write_tokens(pool, (next(it) for _ in range(HELDOUT_DOCS)), out / "fw_test.bin")

        print("FineWeb-Edu train:")
        skip = 2 * HELDOUT_DOCS if args.quick else 0          # quick mode trains on the rest of file 013
        def train_docs():
            for i, f in enumerate(files):
                yield from parquet_docs(f, skip=skip if i == 0 else 0)
        n_fw = write_tokens(pool, train_docs(), out / "fw_train.bin", limit=fw_target)

        print("TinyStories:")
        valid = stories(st_valid.read_text(encoding="utf-8"))
        half = len(valid) // 2
        write_tokens(pool, iter(valid[:half]), out / "st_val.bin")
        write_tokens(pool, iter(valid[half:]), out / "st_test.bin")
        start = st_train.read_bytes().decode("utf-8", errors="ignore")
        start = start[:start.rfind(EOT)]                       # drop the story cut off at the end
        n_st = write_tokens(pool, iter(stories(start)), out / "st_train.bin", limit=st_target)

    print(f"\nTrain mix: {n_fw:,} FineWeb-Edu + {n_st:,} TinyStories = {n_fw + n_st:,} tokens "
          f"({100 * n_st / (n_fw + n_st):.0f}% stories)")
    if n_fw < fw_target or n_st < st_target:
        print("WARNING: fewer tokens than planned (see above).")
    print("Done.")
