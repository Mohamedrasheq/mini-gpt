"""
Try the trained model: give it a start, it continues the text.
(Not a real chat: it only knows how to continue Shakespeare-style text.)

Run:
  .venv/bin/python chat.py                          # type starts one by one (empty line = quit)
  .venv/bin/python chat.py "ROMEO:"                 # one start, then exit
  .venv/bin/python chat.py "ROMEO:" --temperature 0.5 --top_k 10 --length 400 --seed 1

Settings:
  --temperature  low (0.5) = safe, repetitive | 1.0 = normal | high (1.5) = creative, messy
  --top_k        only pick from the k most likely next characters (e.g. 10); default: all 65
  --length       how many characters to write (default 300)
  --seed         same seed + same settings = same text
  --size         full (default) or small
"""

import argparse

import torch

from common import generate, tok
from evaluate import load_model


def continue_text(model, c, start, args):
    unknown = sorted(set(ch for ch in start if ch not in tok.stoi))
    if unknown:
        print(f"  (the model doesn't know these characters, removed: {unknown})")
        start = "".join(ch for ch in start if ch in tok.stoi)
    if not start:
        start = "\n"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    return generate(model, c.block_size, args.length, start=start,
                    temperature=args.temperature, top_k=args.top_k)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("start", nargs="?")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--size", default="full")
    args = p.parse_args()

    model, c = load_model(args.size)
    settings = f"temperature {args.temperature}, top_k {args.top_k or 'all'}, length {args.length}"

    if args.start is not None:
        print(continue_text(model, c, args.start.replace("\\n", "\n"), args))
    else:
        print(f"gpt_{args.size} ready ({settings}). Type a start, e.g. ROMEO:  (empty line = quit)")
        while True:
            start = input("\n> ")
            if not start.strip():
                break
            print("-" * 60)
            print(continue_text(model, c, start, args))
            print("-" * 60)
