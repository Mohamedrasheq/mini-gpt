"""
Try a trained model: give it a start, it continues the text.
(Not a real chat: a base model only continues text in the style it was trained on.)

Run:
  .venv/bin/python step06_chat.py                                  # Shakespeare model, type starts one by one
  .venv/bin/python step06_chat.py "ROMEO:"                         # one start, then exit
  .venv/bin/python step06_chat.py --size stories                   # TinyStories model (step 8)
  .venv/bin/python step06_chat.py "Once upon a time, a dog" --size stories --temperature 0.8 --top_k 50

Settings:
  --size         full (default), small, full_bpe or stories
  --temperature  low (0.5) = safe, repetitive | 1.0 = normal | high (1.5) = creative, messy
  --top_k        only pick from the k most likely next tokens (e.g. 10 for characters, 50 for stories)
  --length       about how many characters to write (default 300; stories: 800)
  --seed         same seed + same settings = same text
For the stories model, the text stops at the end of the first story (<|endoftext|>).
Empty line = quit.
"""

import argparse

import torch

from common import CHARS_PER_TOKEN, TOKENIZERS, generate
from step06_evaluate import load_model

EOT = "<|endoftext|>"


def continue_text(model, c, start, args):
    tk = TOKENIZERS[c.tokenizer][0]
    if c.tokenizer == "char":                     # the character model only knows 65 characters
        unknown = sorted(set(ch for ch in start if ch not in tk.stoi))
        if unknown:
            print(f"  (the model doesn't know these characters, removed: {unknown})")
            start = "".join(ch for ch in start if ch in tk.stoi)
    if not start:
        start = c.start
    if args.seed is not None:
        torch.manual_seed(args.seed)
    length = args.length or (800 if c.tokenizer == "stories" else 300)
    n_tokens = max(1, round(length / CHARS_PER_TOKEN[c.tokenizer]))
    text = generate(model, c.block_size, n_tokens, start=start,
                    temperature=args.temperature, top_k=args.top_k, tokenizer=c.tokenizer)
    if c.tokenizer == "stories" and EOT in text:
        text = text[:text.index(EOT)].rstrip() + "\n[end of story]"
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("start", nargs="?")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--length", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--size", default="full")
    args = p.parse_args()

    model, c = load_model(args.size)
    settings = f"temperature {args.temperature}, top_k {args.top_k or 'all'}"
    example = "Once upon a time, there was a dog" if c.tokenizer == "stories" else "ROMEO:"

    if args.start is not None:
        print(continue_text(model, c, args.start.replace("\\n", "\n"), args))
    else:
        print(f"gpt_{args.size} ready ({settings}). Type a start, e.g. {example}  (empty line = quit)")
        while True:
            start = input("\n> ")
            if not start.strip():
                break
            print("-" * 60)
            print(continue_text(model, c, start, args))
            print("-" * 60)
