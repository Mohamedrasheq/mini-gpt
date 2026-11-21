# mini-gpt

**Version 2.0.0** · [Changelog](CHANGELOG.md) · MIT license

A 30M-parameter GPT language model built from scratch in plain PyTorch and trained on 2 billion
tokens of English: our own tokenizer, model, data pipeline, training loop and test suite.

No pre-trained weights and no model libraries: every part is written here and trained from
random numbers.

```text
The water cycle is one of the main advantages of the environment. It helps in the development of a
good water supply for a population. With this, a river is formed for a variety of different forms
of life.
```
*Written by model 7. The English is fluent; the facts are not reliable, as expected from a model
this small.*

## Model 7

| | |
|---|---|
| Parameters | 29,893,120 (8 layers, 512 wide, 8 heads, 512-token context) |
| Design | GPT with RoPE, SwiGLU and RMSNorm, weight tying, no biases |
| Tokenizer | our own byte-level BPE, 8,192 tokens |
| Training data | 2.0B tokens: 90% FineWeb-Edu + 10% TinyStories |
| Training | 7,630 steps × 262,144 tokens, AdamW, warm-up + cosine, bf16 mixed precision |

## Results

Zero-shot scores from `evaluate.py`, on test data neither model trained on. GPT-2 124M (OpenAI,
2019) is scored by the same script, in the same way, as a reference point.

| Test | What it checks | Random | Model 7 (30M) | GPT-2 (124M) |
|---|---|---:|---:|---:|
| BLiMP | grammar: pick the correct sentence of a pair | 50% | 76.1% | **82.5%** |
| ARC-Easy | school science questions (acc_norm) | 25% | 39.9% | **40.5%** |
| PIQA | everyday physical common sense (acc_norm) | 50% | 58.2% | **62.5%** |
| HellaSwag | pick the sensible ending of a scene (acc_norm) | 25% | 27.9% | **29.9%** |
| LAMBADA | predict the last word of a passage | 0% | 22.1% | **33.7%** |
| WinoGrande | who a pronoun refers to | 50% | **50.7%** | 50.4% |
| FineWeb-Edu test loss | per character, lower is better | - | 0.790 | **0.719** |
| TinyStories test loss | per character, lower is better | - | **0.397** | 0.622 |

- **Model 7 is about 4× smaller** than GPT-2 and was trained on about 2B tokens of FineWeb-Edu and
  TinyStories; GPT-2 was trained on WebText (about 40 GB of web pages linked from Reddit).
- **Close to GPT-2:** ARC-Easy (−0.6) and HellaSwag (−2.0). **Behind:** grammar (BLiMP, −6.4),
  PIQA (−4.3) and long-passage LAMBADA (−11.6). WinoGrande is at chance for both models.
- **How it's measured:** each answer is scored by log-probability and the highest wins; loss is
  per character so different tokenizers compare fairly. Prompts follow lm-evaluation-harness.
  Other tools can give slightly different numbers for the same model because of prompt formatting.
- Reproduce: `evaluate.py --model model7`, `evaluate.py --model gpt2`, `evaluate.py --table`.

## How it grew

This repo started with a character bigram. Each version below is a model that the next one
improved on. Their code is in the [v1.1.0 release](https://github.com/Mohamedrasheq/mini-gpt/tree/v1.1.0).

| # | Model | Data | Parameters | Key result |
|---|---|---|---:|---|
| 1 | Bigram | Tiny Shakespeare (1 MB) | 4,225 | val loss 2.478 per character |
| 2 | 1-head attention | Tiny Shakespeare | 22,721 | val loss 2.330 |
| 3 | GPT small | Tiny Shakespeare | 809,856 | val loss 1.564 |
| 4 | GPT (characters) | Tiny Shakespeare | 10,770,816 | test loss 1.536 per character |
| 5 | GPT (1,024-piece BPE) | Tiny Shakespeare | 11,139,072 | test 1.628 per character: too little data for BPE |
| 6 | GPT stories | TinyStories (246M tokens) | 12,318,720 | clear children's stories, BLiMP 60.4% |
| **7** | **GPT FineWeb** | **FineWeb-Edu + TinyStories (2.0B tokens)** | **29,893,120** | **BLiMP 76.1%** |

## Files

| File | What it does |
|---|---|
| `tokenizer.py` | Byte-level BPE written from scratch; learns the 8,192-token vocabulary (saved in `data/tokenizer_8k.json`) |
| `prepare_data.py` | Downloads FineWeb-Edu and TinyStories, encodes 1.8B + 0.2B tokens on all CPU cores, holds out val / test |
| `model.py` | The model: RoPE, SwiGLU, RMSNorm |
| `train.py` | Training for one GPU: bf16, gradient accumulation, 90/10 data mix, resumable checkpoints, speed test |
| `train_colab.ipynb` | Runs `prepare_data.py` and `train.py` on a Colab GPU |
| `evaluate.py` | The 8 tests for any model, and the comparison table |
| `chat.py` | Type a start, the model continues it |

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python prepare_data.py                    # ~7 GB download, ~4 GB of token files
.venv/bin/python train.py --speed_test              # estimates the full run on your GPU first
.venv/bin/python train.py                           # best on a cloud GPU: see train_colab.ipynb
.venv/bin/python evaluate.py --model model7         # the 8 tests (add --quick for a small check)
.venv/bin/python chat.py "The water cycle is"
```

`tokenizer.py` only needs to run if you want to re-learn the tokenizer; the learned one is included.
For the GPT-2 reference, also `pip install transformers`. The code picks the best device it finds:
an NVIDIA GPU (CUDA), an Apple GPU (MPS) or the CPU. Tested with Python 3.14 and PyTorch 2.15.
Trained weights are not in the repo.

## Next

- Train the same design on more tokens (4-6B), then a slightly larger model.
- Publish the trained weights with a model card.

## Data

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` (Hugging Face, 2024; ODC-By 1.0).
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) V2 (Eldan & Li, 2023; CDLA-Sharing-1.0).
- Tests: [BLiMP](https://github.com/alexwarstadt/blimp) (Warstadt et al., 2020),
  [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) (Zellers et al., 2019),
  [LAMBADA](https://huggingface.co/datasets/EleutherAI/lambada_openai) (Paperno et al., 2016),
  [PIQA](https://huggingface.co/datasets/baber/piqa) (Bisk et al., 2020),
  [ARC](https://huggingface.co/datasets/allenai/ai2_arc) (Clark et al., 2018),
  [WinoGrande](https://huggingface.co/datasets/allenai/winogrande) (Sakaguchi et al., 2020).

The scripts download all data. Test sets are used for testing only, never for training or tuning.

## Credits

The design follows GPT-2 (Radford et al., 2019), RoPE (Su et al., 2021), SwiGLU (Shazeer, 2020) and
RMSNorm (Zhang & Sennrich, 2019). The training setup follows Andrej Karpathy's nanoGPT, minbpe,
llama2.c and build-nanogpt work, and the test prompts follow EleutherAI's lm-evaluation-harness.
The code here is written independently.

## License

MIT
