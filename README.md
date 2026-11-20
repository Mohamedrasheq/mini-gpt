# mini-gpt

**Version 1.0.0** · [Changelog](CHANGELOG.md) · MIT license

Small GPT language models built from scratch in plain PyTorch, one step at a time: first on
Tiny Shakespeare, then on TinyStories. Everything trains on a single consumer GPU.

No pre-trained weights and no model libraries: every part (tokenizers, attention, GPT block,
training loop, evaluation) is written here and trained from random numbers.

```text
Once upon a time, there was a smooth stone. It stood in a row near a big tree. The stone wanted to
be just like the other stones.
One day, a little girl found the smooth stone. She saw the shiny stone and said, "Wow! This stone is
so pretty. I will buy it for you." The girl took the stone home. [...] The stone was happy and so was
the girl. They became friends and lived happily ever after.
```
*Written by model 6 (12.3M parameters, trained from scratch on TinyStories).*

## Results

### Shakespeare (models 1-5)

All five read the same 1 MB of text and are scored on the same held-out data. Loss is per character
(lower is better). Perplexity = e^loss, roughly "how many characters the model is choosing between".

| Model | Parameters | Val loss | Perplexity | Test loss |
|---|---:|---:|---:|---:|
| Random guess | 0 | 4.174 | 65.0 | - |
| 1. Bigram | 4,225 | 2.478 | 11.9 | - |
| 2. 1-head attention | 22,721 | 2.330 | 10.3 | - |
| 3. GPT small | 809,856 | 1.564 | 4.78 | 1.688 |
| **4. GPT full** (characters) | **10,770,816** | **1.385** | **3.99** | **1.536** |
| 5. GPT full BPE (1,024 word pieces) | 11,139,072 | 1.453 | 4.28 | 1.628 |

- GPT full uses the same settings as nanoGPT's Shakespeare character model (reference val ≈ 1.47).
- It reached its best val at step 1,750 of 5,000 and then overfit. The best checkpoint is the one kept.
- Test is mostly *The Tempest* (62% of speeches by characters never seen in training), so test loss
  is higher than val loss (*The Taming of the Shrew*).
- **BPE lost on this data.** In word pieces, 1 MB of text is only 411K training tokens; the model
  overfit by step 750. Its loss is converted to per character to compare fairly.

### TinyStories (model 6)

| Model | Parameters | Tokens seen | Val loss | Test loss | Test perplexity |
|---|---:|---:|---:|---:|---:|
| **6. GPT stories** | **12,318,720** | 246M | **1.366** | **1.364** | **3.91** |

- Loss is per token (4,096-token BPE), so it can't be compared with the Shakespeare table.
- Train and val stayed together the whole run (no overfitting), and test matches val (gap -0.003).
- With word pieces and 2,000× more data, the same design went from made-up words to clear English.

Open `results/report.html` for loss curves, text written during training and samples at
different temperatures.

## Steps

Every file is numbered in the order it was built. Read and run them top to bottom.
Earlier steps are kept on purpose: each one is a smaller model that the next step improves on.

| Step | File | What it does | Model |
|---|---|---|---|
| 1 | `step01_prepare_data.py` | Splits Shakespeare into train / val / test (90 / 5 / 5) by position, cut at blank lines | - |
| 2 | `step02_char_tokenizer.py` | Character tokenizer: 65 characters, built from train only | - |
| 3 | `step03_bigram.py` | Baseline: guesses the next character from the current one only | 1 |
| 4 | `step04_attention.py` | One self-attention head that looks back 32 characters | 2 |
| 5 | `step05_gpt.py` | The GPT (GPT-2 design): multi-head causal attention, feed-forward, pre-LayerNorm, residuals, weight tying, AdamW, warm-up + cosine schedule, best-checkpoint saving, early stopping | 3, 4, 5, 6 |
| 6 | `step06_evaluate.py` | Scores a trained GPT on val and test (every token once) | - |
| 6 | `step06_chat.py` | Type a start, the model continues it (temperature, top-k) | - |
| 6 | `step06_report.py` | Builds `results/report.html` | - |
| 7 | `step07_bpe_tokenizer.py` | Byte-level BPE tokenizer written from scratch (1,024 pieces for Shakespeare) | 5 |
| 8 | `step08_tinystories_data.py` | Downloads TinyStories V2 and splits it into train / val / test | - |
| 8 | `step08_tinystories_tokenizer.py` | Learns a 4,096-token BPE (with an end-of-story token) and encodes all 2.2 GB | 6 |

`common.py` is not a step: it holds the helpers every step from 4 on shares (data, batches, loss
estimate, text generation).

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Shakespeare
.venv/bin/python step01_prepare_data.py           # split the text
.venv/bin/python step02_char_tokenizer.py         # characters -> numbers
.venv/bin/python step03_bigram.py                 # seconds
.venv/bin/python step04_attention.py              # seconds
.venv/bin/python step05_gpt.py small              # about a minute
.venv/bin/python step05_gpt.py full               # add --time first to estimate the run on your hardware
.venv/bin/python step06_evaluate.py full          # val and test score
.venv/bin/python step06_chat.py --temperature 0.8 --top_k 10   # type a start, e.g. ROMEO:
.venv/bin/python step07_bpe_tokenizer.py          # learn BPE pieces, encode the text
.venv/bin/python step05_gpt.py full_bpe           # the GPT again, reading BPE pieces

# TinyStories (~4 GB of disk)
.venv/bin/python step08_tinystories_data.py       # download (2.2 GB) and split
.venv/bin/python step08_tinystories_tokenizer.py  # learn the 4,096-token BPE, encode everything
.venv/bin/python step05_gpt.py stories            # the longest run; --time estimates it first
.venv/bin/python step06_evaluate.py stories
.venv/bin/python step06_chat.py --size stories --temperature 0.8 --top_k 50

.venv/bin/python step06_report.py                 # results/report.html
```

It picks the best device it finds: an NVIDIA GPU (CUDA), an Apple GPU (MPS) or the CPU.
Tested with Python 3.14 and PyTorch 2.15.

## Next

- A grammar test (BLiMP-style sentence pairs) to measure English the same way across models.
- A bigger model on real-world English (FineWeb-Edu) on a cloud GPU.
- Modern parts (RMSNorm, RoPE, SwiGLU), one at a time.

## Data

- [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
  (about 1.1M characters of Shakespeare's plays, public domain), included as `data/shakespeare.txt`.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) V2 (Eldan & Li, 2023; CDLA-Sharing-1.0),
  not included: `step08_tinystories_data.py` downloads it.

## Credits

The model design follows GPT-2 (Radford et al., 2019) and the training setup follows Andrej
Karpathy's nanoGPT, minbpe and llama2.c work. The code here is written independently.

## License

MIT
