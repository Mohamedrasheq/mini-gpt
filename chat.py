"""
Try model 7: give it a start, it continues the text.
(A base model, not a chat assistant: it continues text in the style of what it read.)

Run:
  .venv/bin/python chat.py                                     # type starts one by one
  .venv/bin/python chat.py "The water cycle is"                # one start, then exit
  .venv/bin/python chat.py "Once upon a time" --temperature 0.7 --seed 1

Settings: --temperature (0.7 = safer, 1.0 = more varied), --top_k (pick from the k likeliest tokens),
--tokens (how many to write), --seed (same seed + same settings = same text). Empty line = quit.
Needs results/model7/best.pt (from train.py) and data/tokenizer_8k.json.
"""

import argparse

import torch

from model import GPT, Config
from tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(ckpt="results/model7/best.pt", tokenizer="data/tokenizer_8k.json"):
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    model = GPT(Config(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, Tokenizer.load(tokenizer)


@torch.no_grad()
def continue_text(model, tok, start, n_tokens=200, temperature=0.8, top_k=50):
    T = model.c.block_size
    idx = torch.tensor([tok.encode(start)], device=device)
    for _ in range(n_tokens):
        logits = model(idx[:, -T:])[0][:, -1].float() / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        if nxt.item() == tok.eot:                       # the model ended the text
            break
        idx = torch.cat([idx, nxt], 1)
    return tok.decode(idx[0].tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("start", nargs="?")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    model, tok = load()

    def run(start):
        if args.seed is not None:
            torch.manual_seed(args.seed)
        return continue_text(model, tok, start, args.tokens, args.temperature, args.top_k)

    if args.start:
        print(run(args.start))
    else:
        print("Model 7 ready. Type a start, e.g. 'The water cycle is'  (empty line = quit)")
        while (start := input("\n> ")).strip():
            print("-" * 60 + "\n" + run(start) + "\n" + "-" * 60)
