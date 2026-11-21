# Changelog

## 2.0.0 — 2026-11-21

The repo now holds only model 7, with plain file names. The step-by-step history (models 1-6,
steps 1-10) stays in the [v1.1.0 release](https://github.com/Mohamedrasheq/mini-gpt/tree/v1.1.0).

### Changed

- Files renamed: `tokenizer.py`, `prepare_data.py`, `model.py`, `train.py`, `evaluate.py`,
  `chat.py`, `train_colab.ipynb`. The tokenizer code from steps 7, 8 and 10a is merged into
  `tokenizer.py`; the BLiMP and HellaSwag downloads are part of `evaluate.py`.
- Results live in `results/model7/` and `results/benchmarks/`; the tokenizer in `data/tokenizer_8k.json`.

### Added

- GPT-2 124M as a reference in `evaluate.py` (`--model gpt2`), scored by the same script, and a
  Model 7 vs GPT-2 table in the README.

### Removed

- Steps 1-9 and 10 (check, tokenizer) as separate files, Tiny Shakespeare, the models 1-6 results
  and the HTML report. All of them are in v1.1.0.

## 1.1.0 — 2026-11-21

Model 7: a 30M-parameter GPT trained on 2 billion tokens of English, and a test suite to measure it.

### Models

- **Model 7 (step 10):** 29.9M parameters (8 layers, 512 wide, 512-token context) with RoPE, SwiGLU
  and RMSNorm, an 8,192-token BPE, trained on 1.8B FineWeb-Edu + 0.2B TinyStories tokens.
  BLiMP 76.1%, ARC-Easy 39.9%, PIQA 58.2%, HellaSwag 27.9%, LAMBADA 22.1%;
  FineWeb-Edu test loss 0.790 per character. Better than model 6 on every general-English test.

### Code

- Step 9: BLiMP grammar test, HellaSwag, and one script for 8 tests (TinyStories and FineWeb-Edu
  loss per character, BLiMP, HellaSwag, LAMBADA, PIQA, ARC-Easy, WinoGrande) with a comparison table.
- Step 10: tokenizer fit check, 8K tokenizer, FineWeb-Edu + TinyStories data pipeline (multi-core
  encoding, held-out val/test), model-7 design, single-GPU training (bf16, gradient accumulation,
  resumable checkpoints, speed test), a Colab notebook and a script to try the model.
- Logs no longer include device names or run times.

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
