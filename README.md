# mini-gpt

A small GPT language model built from scratch in plain PyTorch, one step at a time, and trained on
Tiny Shakespeare on a MacBook.

No pre-trained weights and no model libraries: every part (tokenizer, attention, GPT block,
training loop, evaluation) is written here and trained from random numbers.

```text
ROMEO:
What news?

JULIET:
What! what, wilt thou love?

JULIET:
Good lords, I say thou hast scorn to have thee,
```
*Written by the 10.8M-parameter model (temperature 0.5).*

## Results

All models were trained on the same text and scored on the same held-out data.
The loss is per character (lower is better). Perplexity = e^loss, roughly "how many characters
the model is choosing between".

| Model | Parameters | Val loss | Perplexity | Test loss | Train time (M-series Mac) |
|---|---:|---:|---:|---:|---:|
| Random guess | 0 | 4.174 | 65.0 | - | - |
| Bigram | 4,225 | 2.478 | 11.9 | - | 4 s |
| 1-head attention | 22,721 | 2.330 | 10.3 | - | 12 s |
| GPT small | 809,856 | 1.564 | 4.78 | 1.688 | 69 s |
| **GPT full** | **10,770,816** | **1.385** | **3.99** | **1.536** | 70 min |

- GPT full uses the same settings as nanoGPT's Shakespeare character model (reference val ≈ 1.47).
- The test text is mostly *The Tempest*: 62% of its speeches are by characters never seen in
  training, so test loss is higher than val loss (*The Taming of the Shrew*).
- GPT full reached its best val at step 1,750 of 5,000 and then overfit (train kept falling, val
  rose). The best checkpoint is the one kept.

Open `results/report.html` for curves, samples during training and samples at different temperatures.

## Steps

| Step | File | What it does |
|---|---|---|
| 1 | `prepare_data.py` | Splits the text into train / val / test (90 / 5 / 5) by position, cut at blank lines |
| 2 | `tokenizer.py` | Character tokenizer: 65 characters, built from train only |
| 3 | `bigram.py` | Baseline: guesses the next character from the current one only |
| 4 | `attention.py` | One self-attention head that looks back 32 characters |
| 5 | `gpt.py` | The GPT (GPT-2 design): multi-head causal attention, feed-forward, pre-LayerNorm, residuals, weight tying, AdamW, warm-up + cosine schedule, best-checkpoint saving |
| 6 | `evaluate.py`, `chat.py`, `make_report.py` | Test-set score, text generation from a prompt, HTML report |
| 8 | `bpe_tokenizer.py` | Byte-level BPE tokenizer written from scratch (1,024 pieces, ~2.3 characters per piece) |

`common.py` holds the shared helpers: batches, loss estimate, text generation.

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python prepare_data.py      # split the text
.venv/bin/python tokenizer.py         # characters -> numbers
.venv/bin/python bpe_tokenizer.py     # BPE pieces -> numbers

.venv/bin/python bigram.py
.venv/bin/python attention.py
.venv/bin/python gpt.py small         # ~1 min
.venv/bin/python gpt.py full          # ~1 hour on an M-series Mac (--time to estimate first)

.venv/bin/python evaluate.py full     # val and test score
.venv/bin/python chat.py --temperature 0.8 --top_k 10   # type a start, e.g. ROMEO:
.venv/bin/python make_report.py       # results/report.html
```

It uses the Apple GPU (MPS) when available, otherwise the CPU.

## In progress

- **Step 8b:** the same GPT reading BPE pieces instead of characters (`gpt.py full_bpe`), compared
  by loss per character.
- **Next:** more data (TinyStories), then modern parts (RMSNorm, RoPE, SwiGLU).

## Data

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
(about 1.1M characters of Shakespeare's plays, public domain), included as `data/shakespeare.txt`.

## Credits

The model design follows GPT-2 (Radford et al., 2019) and the training setup follows Andrej
Karpathy's nanoGPT and "Let's build GPT" lecture. The code here is written independently.

## License

MIT
