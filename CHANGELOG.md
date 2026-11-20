# Changelog

## 1.0.0 — 2026-11-20

First release: eight steps from a character bigram to a GPT that writes short stories, all written
from scratch in plain PyTorch.

### Models

- **Shakespeare (steps 1-7):** bigram, one attention head, GPT small, GPT full (10.8M parameters,
  test loss 1.536 per character) and the same GPT on a 1,024-piece BPE (test 1.628 per character).
- **TinyStories (step 8):** a 12.3M-parameter GPT with a 4,096-token BPE, trained on 246M tokens.
  Test loss 1.364 per token (perplexity 3.91), val 1.366, no overfitting.

### Code

- Character tokenizer and byte-level BPE tokenizer written from scratch, with an end-of-story token.
- GPT-2 style model: pre-LayerNorm blocks, multi-head causal attention, GELU feed-forward,
  weight tying, AdamW with warm-up and cosine learning rate, best-checkpoint saving, early stopping.
- Deterministic full-pass evaluation on val and test, text generation with temperature and top-k,
  and an HTML report of every model (`results/report.html`).
- Runs on an NVIDIA GPU (CUDA), an Apple GPU (MPS) or the CPU.
