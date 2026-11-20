"""
Build results/report.html: one local page with every result so far (no internet needed to view it).

Reads results/*/history.json and results/gpt_*/test_scores.json, and writes a few fresh
text samples from model 4 (Shakespeare) and model 6 (TinyStories) at different temperatures
(fixed seed, so the same every run).

Run: .venv/bin/python step06_report.py   then open results/report.html
"""

import html
import json
import math
from pathlib import Path

import torch

from common import generate
from step06_chat import continue_text
from step06_evaluate import load_model

R = Path("results")
OUT = R / "report.html"


def load(name):
    return json.loads((R / name / "history.json").read_text())


hist = {n: load(n) for n in ("bigram", "attention", "gpt_small", "gpt_full", "gpt_full_bpe", "gpt_stories")}
test = {n: json.loads((R / n / "test_scores.json").read_text())
        for n in ("gpt_small", "gpt_full", "gpt_full_bpe", "gpt_stories")}
full = hist["gpt_full"]
best_row = min(full["history"], key=lambda h: h["val"])
last_row = full["history"][-1]
esc = html.escape


# ------------------------------------------------------------------ fresh samples from model 4
model, c = load_model("full")
SETTINGS = [(0.5, None, "Safe: clean but repetitive"),
            (0.8, 10, "Balanced: our recommended setting"),
            (1.0, None, "Normal: as trained"),
            (1.5, None, "Wild: creative but broken spelling")]
temp_samples = []
for t, k, note in SETTINGS:
    torch.manual_seed(1)
    temp_samples.append((t, k, note, generate(model, c.block_size, 300, start="ROMEO:", temperature=t, top_k=k)))

story_model, sc = load_model("stories")
STORY_START = "Once upon a time, there was a little cat"
STORY_SETTINGS = [(0.5, None, "Safe: very clean, a bit plain"),
                  (0.8, 50, "Balanced: our recommended setting"),
                  (1.0, None, "Normal: as trained"),
                  (1.5, None, "Wild: more surprises, more mistakes")]
story_samples = []
for t, k, note in STORY_SETTINGS:
    a = type("A", (), {"seed": 1, "length": 900, "temperature": t, "top_k": k})()
    story_samples.append((t, k, note, continue_text(story_model, sc, STORY_START, a)))


# ------------------------------------------------------------------ charts (plain SVG)
def line_chart(history, best_step, w=760, h=320, ymin=0.5, ymax=4.5, yticks=(1, 2, 3, 4), xstep=1000,
               zone=True):
    pad_l, pad_r, pad_t, pad_b = 48, 16, 16, 36
    steps = [p["step"] for p in history]
    sx = lambda s: pad_l + (s - steps[0]) / (steps[-1] - steps[0]) * (w - pad_l - pad_r)
    sy = lambda v: pad_t + (ymax - v) / (ymax - ymin) * (h - pad_t - pad_b)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Training and validation loss">']
    if zone:   # overfitting zone
        parts.append(f'<rect x="{sx(best_step):.1f}" y="{pad_t}" width="{sx(steps[-1]) - sx(best_step):.1f}" '
                     f'height="{h - pad_t - pad_b}" class="zone"/>')
        parts.append(f'<text x="{(sx(best_step) + sx(steps[-1])) / 2:.1f}" y="{pad_t + 18}" class="zone-label" '
                     f'text-anchor="middle">overfitting: memorising, not learning</text>')
    for v in yticks:
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                     f'<text x="{pad_l - 8}" y="{sy(v) + 4:.1f}" text-anchor="end" class="tick">{v}</text>')
    for s in range(0, steps[-1] + 1, xstep):
        parts.append(f'<text x="{sx(s):.1f}" y="{h - 12}" text-anchor="middle" class="tick">{s}</text>')
    for key, cls in (("train", "s-train"), ("val", "s-val")):
        pts = " ".join(f"{sx(p['step']):.1f},{sy(p[key]):.1f}" for p in history)
        parts.append(f'<polyline points="{pts}" class="{cls}"/>')
    b = next(p for p in history if p["step"] == best_step)
    parts.append(f'<circle cx="{sx(best_step):.1f}" cy="{sy(b["val"]):.1f}" r="5" class="best-dot"/>'
                 f'<text x="{sx(best_step) - 8:.1f}" y="{sy(b["val"]) + 22:.1f}" text-anchor="end" class="ann">'
                 f'best: step {best_step}, val {b["val"]:.3f} (saved)</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(rows, w=760):
    # rows: (label, value, highlight)
    bar_h, gap, pad_l, pad_r = 26, 12, 150, 70
    h = len(rows) * (bar_h + gap) + 10
    vmax = 4.5
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Validation loss by model">']
    for i, (label, v, hi) in enumerate(rows):
        y = 5 + i * (bar_h + gap)
        bw = v / vmax * (w - pad_l - pad_r)
        parts.append(f'<text x="{pad_l - 10}" y="{y + bar_h / 2 + 5}" text-anchor="end" class="lab">{esc(label)}</text>'
                     f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="4" class="{"bar-hi" if hi else "bar"}"/>'
                     f'<text x="{pad_l + bw + 8:.1f}" y="{y + bar_h / 2 + 5}" class="val">{v:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ page pieces
random_loss = math.log(65)
models = [
    ("Random guess", 0, random_loss, None, "-", "Picks any of 65 characters"),
    ("Bigram", hist["bigram"]["params"], min(p["val"] for p in hist["bigram"]["history"]), None, "4 s",
     "Sees only the current character"),
    ("1-head attention", hist["attention"]["params"], hist["attention"]["best_val_loss"], None,
     f'{hist["attention"]["seconds"]} s', "Looks back 32 characters"),
    ("GPT small", hist["gpt_small"]["params"], hist["gpt_small"]["best_val_loss"], test["gpt_small"]["test"]["loss"],
     f'{hist["gpt_small"]["seconds"]} s', "4 layers, looks back 64"),
    ("GPT full (model 4)", full["params"], full["best_val_loss"], test["gpt_full"]["test"]["loss"],
     f'{full["seconds"] / 60:.0f} min', "6 layers, looks back 256"),
    ("GPT full BPE (model 5)", hist["gpt_full_bpe"]["params"], hist["gpt_full_bpe"]["best_val_per_char"],
     test["gpt_full_bpe"]["test"]["loss_per_char"], f'{hist["gpt_full_bpe"]["seconds"] / 60:.0f} min',
     "Same, reads 1,024 word pieces"),
]

table_rows = "".join(
    f'<tr class="{"hi" if "model 4" in m else ""}"><td>{esc(m)}</td><td>{note}</td><td class="n">{p:,}</td>'
    f'<td class="n">{v:.3f}</td><td class="n">{math.exp(v):.2f}</td>'
    f'<td class="n">{"-" if t is None else f"{t:.3f}"}</td></tr>'
    for m, p, v, t, s, note in models)

curve_rows = "".join(
    f'<tr class="{"hi" if r["step"] == best_row["step"] else ""}"><td class="n">{r["step"]}</td>'
    f'<td class="n">{r["train"]:.3f}</td><td class="n">{r["val"]:.3f}</td><td class="n">{r["val"] - r["train"]:+.3f}</td>'
    f'<td class="n">{math.exp(r["val"]):.2f}</td></tr>'
    for r in full["history"])

progress_cards = "".join(
    f'<div class="card"><div class="card-h">Step {s["step"]}'
    f'{" (random start)" if s["step"] == 0 else ""}{" (overfitting)" if s["step"] > best_row["step"] else ""}</div>'
    f'<pre>{esc(s["text"].strip())}</pre></div>'
    for s in full["samples"])

temp_cards = "".join(
    f'<div class="card"><div class="card-h">Temperature {t}{f", top-k {k}" if k else ""}</div>'
    f'<div class="card-sub">{esc(note)}</div><pre>{esc(txt)}</pre></div>'
    for t, k, note, txt in temp_samples)

tg = test["gpt_full"]

# ------------------------------------------------------------------ model 6 (TinyStories) and model 5 (BPE)
EOT = "<|endoftext|>"
st = hist["gpt_stories"]
s_best = min(st["history"], key=lambda h: h["val"])
s_last = st["history"][-1]
ts = test["gpt_stories"]
scfg = st["config"]

story_progress_cards = "".join(
    f'<div class="card"><div class="card-h">Step {x["step"]}{" (random start)" if x["step"] == 0 else ""}</div>'
    f'<pre>{esc(x["text"].split(EOT)[0].strip())}</pre></div>'
    for x in st["samples"])
story_final = esc(st["final_sample"].split(EOT)[0].strip())
story_temp_cards = "".join(
    f'<div class="card"><div class="card-h">Temperature {t}{f", top-k {k}" if k else ""}</div>'
    f'<div class="card-sub">{esc(note)}</div><pre>{esc(txt)}</pre></div>'
    for t, k, note, txt in story_samples)

bp = hist["gpt_full_bpe"]
b_best = min(bp["history"], key=lambda h: h["val"])
b_last = bp["history"][-1]
tb = test["gpt_full_bpe"]

tiles = [
    ("Model 6 parameters", f'{st["params"] / 1e6:.1f} M', "our own, trained from random"),
    ("Test loss (honest)", f'{ts["test"]["loss"]:.3f}', f'perplexity {ts["test"]["perplexity"]:.2f} (of 4,096 tokens)'),
    ("Test - val gap", f'{ts["test"]["loss"] - ts["val"]["loss"]:+.3f}', "no overfitting"),
    ("Tokens read", f'{scfg["steps"] * scfg["batch_size"] * scfg["block_size"] / 1e6:.0f}M', "less than half of the training set"),
]

tiles_html = "".join(f'<div class="tile"><div class="t-l">{a}</div><div class="t-v">{b}</div><div class="t-s">{c_}</div></div>'
                     for a, b, c_ in tiles)
cfg = full["config"]
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mini GPT Report</title>
<style>
:root {{ --bg:#fafaf8; --panel:#fff; --ink:#1d1d1f; --mute:#6b6b70; --line:#e4e4e0; --train:#8a8a90; --val:#d9480f;
        --accent:#d9480f; --hi:#fff3eb; --zone:rgba(217,72,15,.07); --bar:#c9c9ce; }}
@media (prefers-color-scheme: dark) {{
 :root {{ --bg:#141416; --panel:#1c1c1f; --ink:#ececee; --mute:#9a9aa2; --line:#2e2e33; --train:#8a8a92; --val:#ff8a4c;
         --accent:#ff8a4c; --hi:#2a1d15; --zone:rgba(255,138,76,.09); --bar:#45454c; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system,system-ui,sans-serif }}
main {{ max-width:860px; margin:0 auto; padding:32px 16px 64px }}
h1 {{ font-size:28px; margin:0 0 4px }} h2 {{ font-size:19px; margin:40px 0 8px }}
.sub {{ color:var(--mute); margin:0 0 24px }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px }}
.t-l {{ color:var(--mute); font-size:13px }} .t-v {{ font-size:26px; font-weight:650; margin:2px 0 }} .t-s {{ color:var(--mute); font-size:12.5px }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; overflow-x:auto }}
ul.points {{ margin:8px 0 0; padding-left:20px }} ul.points li {{ margin:3px 0 }}
table {{ width:100%; border-collapse:collapse; font-size:14px }}
th,td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); white-space:nowrap }}
th {{ color:var(--mute); font-weight:600; font-size:12.5px }} td.n,th.n {{ text-align:right; font-variant-numeric:tabular-nums }}
tr.hi td {{ background:var(--hi); font-weight:600 }}
.chart {{ width:100%; height:auto; display:block }}
.grid {{ stroke:var(--line) }} .tick,.lab,.val,.ann {{ fill:var(--mute); font-size:12px }} .lab {{ fill:var(--ink); font-size:13px }}
.ann {{ fill:var(--accent); font-weight:600 }}
.s-train {{ fill:none; stroke:var(--train); stroke-width:2; stroke-dasharray:5 4 }} .s-val {{ fill:none; stroke:var(--val); stroke-width:2.5 }}
.best-dot {{ fill:var(--accent) }} .zone {{ fill:var(--zone) }} .zone-label {{ fill:var(--accent); font-size:12px }}
.bar {{ fill:var(--bar) }} .bar-hi {{ fill:var(--accent) }}
.legend {{ display:flex; gap:18px; color:var(--mute); font-size:13px; margin-top:6px }}
.sw {{ display:inline-block; width:18px; height:3px; vertical-align:middle; margin-right:6px }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; min-width:0 }}
.card-h {{ font-weight:650; font-size:14px }} .card-sub {{ color:var(--mute); font-size:12.5px }}
pre {{ white-space:pre-wrap; word-break:break-word; font:12.5px/1.5 ui-monospace,Menlo,monospace; margin:8px 0 0 }}
details summary {{ cursor:pointer; color:var(--mute); margin-top:10px }}
</style></head><body><main>

<h1>Mini GPT: results report</h1>
<p class="sub">Six GPT-style language models written from scratch in PyTorch and trained on a single consumer GPU. The best one, model 6, writes short stories in simple English.</p>

<div class="tiles">{tiles_html}</div>

<h2>Verdict</h2>
<div class="panel"><ul class="points">
<li><b>Model 6 (TinyStories) works:</b> correct grammar, a beginning, middle and end, dialogue, and it keeps track of who is who. It knows where a story ends.</li>
<li><b>Still weak at:</b> small logic slips ("I will buy it for you" for a stone she found). Normal for 12M parameters.</li>
<li><b>What made the difference:</b> data. 1 MB of Shakespeare gave made-up words; 2.2 GB of simple stories gave real English.</li>
<li><b>Honest score:</b> test {ts["test"]["loss"]:.3f} vs val {ts["val"]["loss"]:.3f}: it learned patterns, not stories.</li>
</ul></div>

<h2>1. All models</h2>
<div class="panel">
<p class="sub" style="margin:0 0 8px"><b>Shakespeare models</b>, loss per character (lower is better). Models 1-5 read the same 1 MB of text.</p>
{bar_chart([(m, v, "model 4" in m) for m, _, v, _, _, _ in models])}
<table><tr><th>Model</th><th>What it can do</th><th class="n">Parameters</th><th class="n">Val loss</th><th class="n">Perplexity</th><th class="n">Test loss</th></tr>{table_rows}</table>
<p class="sub" style="margin:16px 0 8px"><b>TinyStories model</b>, loss per token (a different dataset, so not comparable with the table above).</p>
<table><tr><th>Model</th><th>What it can do</th><th class="n">Parameters</th><th class="n">Val loss</th><th class="n">Perplexity</th><th class="n">Test loss</th></tr>
<tr class="hi"><td>GPT stories (model 6)</td><td>6 layers, 4,096 word pieces, sees ~1,000 characters</td><td class="n">{st["params"]:,}</td><td class="n">{st["best_val_loss"]:.3f}</td><td class="n">{math.exp(st["best_val_loss"]):.2f}</td><td class="n">{ts["test"]["loss"]:.3f}</td></tr></table>
</div>

<h2>2. Model 6: TinyStories</h2>
<div class="panel">{line_chart(st["history"], s_best["step"], ymin=1, ymax=8.6, yticks=(2, 4, 6, 8), xstep=2500, zone=False)}
<div class="legend"><span><span class="sw" style="background:var(--train)"></span>train loss</span>
<span><span class="sw" style="background:var(--val)"></span>val loss</span></div>
<ul class="points">
<li>Train and val stay on top of each other the whole run ({s_last["train"]:.3f} vs {s_last["val"]:.3f} at the end): <b>no overfitting</b>. It read {scfg["steps"] * scfg["batch_size"] * scfg["block_size"] / 1e6:.0f}M tokens, less than half the training set, so it never saw a story twice.</li>
<li>Best val {s_best["val"]:.3f} at step {s_best["step"]:,}; the last checks only wobble around it.</li>
</ul>
<table><tr><th>Split</th><th>What it is</th><th class="n">Loss</th><th class="n">Perplexity</th><th class="n">Tokens</th></tr>
<tr><td>Val</td><td>13,815 stories</td><td class="n">{ts["val"]["loss"]:.3f}</td><td class="n">{ts["val"]["perplexity"]:.2f}</td><td class="n">{ts["val"]["tokens"]:,}</td></tr>
<tr class="hi"><td>Test</td><td>13,815 other stories, never seen</td><td class="n">{ts["test"]["loss"]:.3f}</td><td class="n">{ts["test"]["perplexity"]:.2f}</td><td class="n">{ts["test"]["tokens"]:,}</td></tr></table>
</div>

<h3 style="margin:24px 0 8px">The story written at the end of training</h3>
<div class="card"><pre>{story_final}</pre></div>

<h3 style="margin:24px 0 8px">Stories while it trained</h3>
<p class="sub">Start: "Once upon a time". From random tokens to a full story.</p>
<div class="cards">{story_progress_cards}</div>

<h3 style="margin:24px 0 8px">Different temperatures</h3>
<p class="sub">Start: "{STORY_START}". Same seed, only the setting changes.</p>
<div class="cards">{story_temp_cards}</div>

<h2>3. Model 5: BPE on Shakespeare (a useful failure)</h2>
<div class="panel">{line_chart(bp["history"], b_best["step"], ymin=0.5, ymax=7.5, yticks=(1, 3, 5, 7), xstep=500)}
<ul class="points">
<li>Same model as model 4, but reading 1,024 BPE word pieces instead of 65 characters.</li>
<li>Best val at step {b_best["step"]}, then heavy overfitting: train {b_last["train"]:.2f} vs val {b_last["val"]:.2f} when early stopping ended it.</li>
<li>Per character it lost to model 4: test {tb["test"]["loss_per_char"]:.3f} vs {tg["test"]["loss"]:.3f}. In pieces, 1 MB of text is only 411K training tokens, far too few for 11M parameters.</li>
<li>Lesson: word pieces need more data. With TinyStories (558M tokens), the same idea worked.</li>
</ul></div>

<h2>4. Model 4: characters on Shakespeare</h2>
<div class="panel">{line_chart(full["history"], best_row["step"])}
<div class="legend"><span><span class="sw" style="background:var(--train)"></span>train loss (text it learns from)</span>
<span><span class="sw" style="background:var(--val)"></span>val loss (text it never trains on)</span></div>
<ul class="points">
<li>Both lines fall together until <b>step {best_row["step"]}</b>, then train keeps falling ({best_row["train"]:.2f} → {last_row["train"]:.2f}) while val rises ({best_row["val"]:.3f} → {last_row["val"]:.3f}). It's memorising.</li>
<li>We kept the step-{best_row["step"]} checkpoint, not the last one.</li>
<li>Test {tg["test"]["loss"]:.3f} vs val {tg["val"]["loss"]:.3f}: test is mostly <i>The Tempest</i> (62% of speeches by characters never seen in training), val mostly <i>The Taming of the Shrew</i> (7%). GPT small shows the same gap, so it comes from the data.</li>
</ul>
<details><summary>All checkpoints</summary>
<table><tr><th class="n">Step</th><th class="n">Train</th><th class="n">Val</th><th class="n">Gap</th><th class="n">Val perplexity</th></tr>{curve_rows}</table>
</details></div>

<h3 style="margin:24px 0 8px">Text while it trained</h3>
<div class="cards">{progress_cards}</div>

<h3 style="margin:24px 0 8px">Different temperatures</h3>
<p class="sub">Start: "ROMEO:". Same seed, only the setting changes.</p>
<div class="cards">{temp_cards}</div>

<h2>5. Setup</h2>
<div class="panel"><ul class="points">
<li><b>Model design (all GPTs):</b> GPT-2 style: token + position embeddings, pre-LayerNorm blocks with multi-head causal attention and a GELU feed-forward, weight tying, AdamW with warm-up then cosine learning rate, gradient clip 1.0, best-checkpoint saving.</li>
<li><b>Models 1-5:</b> Tiny Shakespeare (1.1M characters), split by position into 90% train / 5% val / 5% test.</li>
<li><b>Model 6:</b> TinyStories V2 (2.7M stories, 2.2 GB). Train = the official train file; val and test = the two halves of the official validation file (0 exact duplicates with train). Our own byte-level BPE with 4,096 tokens, learned from a 40 MB train sample, {scfg["n_layer"]} layers, {scfg["n_head"]} heads, width {scfg["n_embd"]}, context {scfg["block_size"]} tokens, dropout {scfg["dropout"]}, {scfg["steps"]:,} steps of batch {scfg["batch_size"]}.</li>
<li><b>Next:</b> a grammar test to measure English across models, then a bigger model on real-world English (FineWeb-Edu).</li>
</ul></div>

</main></body></html>"""

OUT.write_text(page, encoding="utf-8")
print(f"Saved -> {OUT}")
