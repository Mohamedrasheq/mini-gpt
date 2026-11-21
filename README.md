# mini-gpt

**Version 1.1.0** · [Changelog](CHANGELOG.md) · MIT license

Small GPT language models built from scratch in plain PyTorch, one step at a time: from a
character bigram on Shakespeare to a 30M-parameter model trained on 2 billion tokens of English.

No pre-trained weights and no model libraries: every part (tokenizers, attention, GPT block,
training loop, evaluation) is written here and trained from random numbers.

```text
The water cycle is one of the main advantages of the environment. It helps in the development of a
good water supply for a population. With this, a river is formed for a variety of different forms
of life.
```
*Written by model 7 (29.9M parameters). The English is fluent; the facts are not reliable, as
expected from a model this small.*

## Model 7 (latest)

| | |
|---|---|
| Parameters | 29,893,120 (8 layers, 512 wide, 8 heads, 512-token context) |
| Design | GPT with RoPE, SwiGLU and RMSNorm, weight tying, no biases |
| Tokenizer | our own byte-level BPE, 8,192 tokens |
| Training data | 2.0B tokens: 90% FineWeb-Edu + 10% TinyStories |
| Training | 7,630 steps × 262,144 tokens, AdamW, warm-up + cosine, bf16 mixed precision |

### Test scores

All tests run with the same script (`step09c_benchmarks.py`), on data the model never trained on.
Multiple-choice tests are scored zero-shot: the model scores each answer, the highest wins.

| Test | What it checks | Random | Model 7 |
|---|---|---:|---:|
| BLiMP | grammar: pick the correct sentence of a pair | 50% | **76.1%** |
| ARC-Easy | school science questions (acc_norm) | 25% | **39.9%** |
| PIQA | everyday physical common sense (acc_norm) | 50% | **58.2%** |
| HellaSwag | pick the sensible ending of a scene (acc_norm) | 25% | **27.9%** |
| LAMBADA | predict the last word of a passage | 0% | **22.1%** |
| WinoGrande | who a pronoun refers to | 50% | 50.7% |
| FineWeb-Edu test loss | per character, lower is better | - | **0.790** |
| TinyStories test loss | per character, lower is better | - | **0.397** |

WinoGrande stays at chance: this test needs much larger models.

## How it grew

| # | Model | Data | Parameters | Key result |
|---|---|---|---:|---|
| 1 | Bigram | Shakespeare (1 MB) | 4,225 | val loss 2.478 per character |
| 2 | 1-head attention | Shakespeare | 22,721 | val loss 2.330 |
| 3 | GPT small | Shakespeare | 809,856 | val loss 1.564 |
| 4 | GPT full (characters) | Shakespeare | 10,770,816 | val loss 1.385, test 1.536 |
| 5 | GPT full (1,024-piece BPE) | Shakespeare | 11,139,072 | test 1.628 per character (too little data for BPE) |
| 6 | GPT stories | TinyStories (246M tokens) | 12,318,720 | test 1.364 per token, BLiMP 60.4% |
| **7** | **GPT FineWeb** | **FineWeb-Edu + TinyStories (2.0B tokens)** | **29,893,120** | **BLiMP 76.1%** |

Model 6 writes clean children's stories but knows little else. Model 7 changed three things at
once (richer data, a bigger tokenizer and a larger, modern design) and improved on every
general-English test. Model 6 vs 7 on all tests: [`results/benchmarks/compare.md`](results/benchmarks/compare.md).

`results/report.html` has loss curves and samples for models 1-6.

## Steps

Every file is numbered in the order it was built. Read and run them top to bottom.
Earlier steps are kept on purpose: each one is a smaller model that the next step improves on.

| Step | File | What it does | Model |
|---|---|---|---|
| 1 | `step01_prepare_data.py` | Splits Shakespeare into train / val / test (90 / 5 / 5) by position | - |
| 2 | `step02_char_tokenizer.py` | Character tokenizer: 65 characters, built from train only | - |
| 3 | `step03_bigram.py` | Baseline: guesses the next character from the current one only | 1 |
| 4 | `step04_attention.py` | One self-attention head that looks back 32 characters | 2 |
| 5 | `step05_gpt.py` | The GPT (GPT-2 style): multi-head causal attention, feed-forward, pre-LayerNorm, residuals, weight tying, AdamW, warm-up + cosine, early stopping | 3-6 |
| 6 | `step06_evaluate.py` | Scores a trained GPT on val and test (every token once) | - |
| 6 | `step06_chat.py` | Type a start, the model continues it (models 1-6) | - |
| 6 | `step06_report.py` | Builds `results/report.html` | - |
| 7 | `step07_bpe_tokenizer.py` | Byte-level BPE tokenizer written from scratch | 5 |
| 8 | `step08_tinystories_data.py` | Downloads TinyStories V2 and splits it into train / val / test | - |
| 8 | `step08_tinystories_tokenizer.py` | 4,096-token BPE with an end-of-story token; encodes all 2.2 GB | 6 |
| 9 | `step09_blimp.py` | Grammar test: 67,000 BLiMP sentence pairs, with a breakdown by grammar area | - |
| 9 | `step09b_hellaswag.py` | Common-sense test: 10,042 HellaSwag questions | - |
| 9 | `step09c_benchmarks.py` | All 8 tests for any of our models, and a side-by-side table | - |
| 10 | `step10_fineweb_check.py` | Measures how well each tokenizer fits FineWeb-Edu (why we need a new one) | - |
| 10 | `step10a_tokenizer.py` | 8,192-token BPE learned from the FineWeb-Edu + TinyStories mix | 7 |
| 10 | `step10b_data.py` | Downloads and encodes 1.8B FineWeb-Edu + 0.2B TinyStories tokens | 7 |
| 10 | `step10_model.py` | The model-7 design: RoPE, SwiGLU, RMSNorm | 7 |
| 10 | `step10_train.py` | Training for one GPU: bf16, gradient accumulation, data mix, resumable checkpoints | 7 |
| 10 | `step10_colab.ipynb` | Runs steps 10b and 10 on a Colab GPU | 7 |
| 10 | `step10_chat.py` | Type a start, model 7 continues it | 7 |

`common.py` is not a step: it holds the helpers steps 4-6 share (data, batches, loss estimate,
text generation).

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Shakespeare (models 1-5)
.venv/bin/python step01_prepare_data.py
.venv/bin/python step02_char_tokenizer.py
.venv/bin/python step03_bigram.py
.venv/bin/python step04_attention.py
.venv/bin/python step05_gpt.py small
.venv/bin/python step05_gpt.py full               # --time estimates the run on your hardware first
.venv/bin/python step06_evaluate.py full
.venv/bin/python step06_chat.py --temperature 0.8 --top_k 10
.venv/bin/python step07_bpe_tokenizer.py
.venv/bin/python step05_gpt.py full_bpe

# TinyStories (model 6, ~4 GB of disk)
.venv/bin/python step08_tinystories_data.py
.venv/bin/python step08_tinystories_tokenizer.py
.venv/bin/python step05_gpt.py stories
.venv/bin/python step06_evaluate.py stories
.venv/bin/python step06_chat.py --size stories --temperature 0.8 --top_k 50

# Tests (step 9)
.venv/bin/python step09c_benchmarks.py --model stories   # add --quick for a small check
.venv/bin/python step09c_benchmarks.py --table

# FineWeb-Edu (model 7): best on a cloud GPU, see step10_colab.ipynb
.venv/bin/python step10_fineweb_check.py
.venv/bin/python step10a_tokenizer.py
.venv/bin/python step10b_data.py                         # ~7 GB download, ~4 GB of token files
.venv/bin/python step10_train.py --speed_test            # estimates the full run first
.venv/bin/python step10_train.py
.venv/bin/python step09c_benchmarks.py --model fineweb
.venv/bin/python step10_chat.py "The water cycle is"
```

The code picks the best device it finds: an NVIDIA GPU (CUDA), an Apple GPU (MPS) or the CPU.
Tested with Python 3.14 and PyTorch 2.15. Trained weights are not in the repo.

## Next

- Train model 7's design on more tokens (4-6B), then a slightly larger model.
- Publish the trained weights with a model card.

## Data

- [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
  (public domain), included as `data/shakespeare.txt`.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) V2 (Eldan & Li, 2023; CDLA-Sharing-1.0).
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` (Hugging Face, 2024; ODC-By 1.0).
- Tests: [BLiMP](https://github.com/alexwarstadt/blimp) (Warstadt et al., 2020),
  [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) (Zellers et al., 2019),
  [LAMBADA](https://huggingface.co/datasets/EleutherAI/lambada_openai) (Paperno et al., 2016),
  [PIQA](https://huggingface.co/datasets/baber/piqa) (Bisk et al., 2020),
  [ARC](https://huggingface.co/datasets/allenai/ai2_arc) (Clark et al., 2018),
  [WinoGrande](https://huggingface.co/datasets/allenai/winogrande) (Sakaguchi et al., 2020).

Only Tiny Shakespeare is included; the scripts download everything else. Test sets are used for
testing only, never for training or tuning.

## Credits

The model design follows GPT-2 (Radford et al., 2019) and, for model 7, RoPE (Su et al., 2021),
SwiGLU (Shazeer, 2020) and RMSNorm (Zhang & Sennrich, 2019). The training setup follows Andrej
Karpathy's nanoGPT, minbpe, llama2.c and build-nanogpt work, and the test prompts follow
EleutherAI's lm-evaluation-harness. The code here is written independently.

## License

MIT
