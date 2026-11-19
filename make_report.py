"""
Build results/report.html: one local page with every result so far (no internet needed to view it).

Reads results/*/history.json and results/gpt_*/test_scores.json, and writes a few fresh
text samples from model 4 (gpt_full) at different temperatures (fixed seed, so the same every run).

Run: .venv/bin/python make_report.py   then open results/report.html
"""

import html
import json
import math
from pathlib import Path

import torch

from common import generate
from evaluate import load_model

R = Path("results")
OUT = R / "report.html"


def load(name):
    return json.loads((R / name / "history.json").read_text())


hist = {n: load(n) for n in ("bigram", "attention", "gpt_small", "gpt_full")}
test = {n: json.loads((R / n / "test_scores.json").read_text()) for n in ("gpt_small", "gpt_full")}
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


# ------------------------------------------------------------------ charts (plain SVG)
def line_chart(history, best_step, w=760, h=320):
    pad_l, pad_r, pad_t, pad_b = 48, 16, 16, 36
    steps = [p["step"] for p in history]
    ymin, ymax = 0.5, 4.5
    sx = lambda s: pad_l + (s - steps[0]) / (steps[-1] - steps[0]) * (w - pad_l - pad_r)
    sy = lambda v: pad_t + (ymax - v) / (ymax - ymin) * (h - pad_t - pad_b)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Training and validation loss">']
    # overfitting zone
    parts.append(f'<rect x="{sx(best_step):.1f}" y="{pad_t}" width="{sx(steps[-1]) - sx(best_step):.1f}" '
                 f'height="{h - pad_t - pad_b}" class="zone"/>')
    parts.append(f'<text x="{(sx(best_step) + sx(steps[-1])) / 2:.1f}" y="{pad_t + 18}" class="zone-label" '
                 f'text-anchor="middle">overfitting: memorising, not learning</text>')
    for v in (1, 2, 3, 4):
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                     f'<text x="{pad_l - 8}" y="{sy(v) + 4:.1f}" text-anchor="end" class="tick">{v}</text>')
    for s in range(0, steps[-1] + 1, 1000):
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
]

table_rows = "".join(
    f'<tr class="{"hi" if "model 4" in m else ""}"><td>{esc(m)}</td><td>{note}</td><td class="n">{p:,}</td>'
    f'<td class="n">{v:.3f}</td><td class="n">{math.exp(v):.2f}</td>'
    f'<td class="n">{"-" if t is None else f"{t:.3f}"}</td><td class="n">{s}</td></tr>'
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
tiles = [
    ("Parameters", f'{full["params"] / 1e6:.1f} M', "our own, trained from random"),
    ("Best val loss", f'{full["best_val_loss"]:.3f}', f'step {best_row["step"]} of {last_row["step"]}'),
    ("Test loss (honest)", f'{tg["test"]["loss"]:.3f}', f'perplexity {tg["test"]["perplexity"]:.2f}'),
    ("Training time", f'{full["seconds"] / 60:.0f} min', "Mac GPU (MPS)"),
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

<h1>Mini GPT: test report</h1>
<p class="sub">Model 4 (GPT full): a 10.8M-parameter GPT written from scratch and trained on Tiny Shakespeare, one character at a time.</p>

<div class="tiles">{tiles_html}</div>

<h2>Verdict</h2>
<div class="panel"><ul class="points">
<li><b>Good at:</b> layout, speaker names, spelling, old-English style. It even pairs ROMEO with JULIET and LADY CAPULET.</li>
<li><b>Weak at:</b> meaning. Sentences sound like Shakespeare but don't make sense.</li>
<li><b>Why:</b> it reads single characters and learned from only 1 MB of text, so most of its effort goes on spelling.</li>
<li><b>Best settings for writing:</b> temperature about 0.8, top-k 10.</li>
</ul></div>

<h2>1. Training curve (model 4)</h2>
<div class="panel">{line_chart(full["history"], best_row["step"])}
<div class="legend"><span><span class="sw" style="background:var(--train)"></span>train loss (text it learns from)</span>
<span><span class="sw" style="background:var(--val)"></span>val loss (text it never trains on)</span></div>
<ul class="points">
<li>Both lines fall together until <b>step {best_row["step"]}</b>, where the model is learning real patterns.</li>
<li>After that, train keeps falling ({best_row["train"]:.2f} → {last_row["train"]:.2f}) but val rises ({best_row["val"]:.3f} → {last_row["val"]:.3f}). It's memorising.</li>
<li>We kept the step-{best_row["step"]} checkpoint, not the last one. The last ~{(1 - best_row["step"] / last_row["step"]) * 100:.0f}% of training added nothing.</li>
</ul>
<details><summary>All checkpoints</summary>
<table><tr><th class="n">Step</th><th class="n">Train</th><th class="n">Val</th><th class="n">Gap</th><th class="n">Val perplexity</th></tr>{curve_rows}</table>
</details></div>

<h2>2. Test score: the honest number</h2>
<div class="panel">
<table><tr><th>Split</th><th>What it is</th><th class="n">Loss</th><th class="n">Perplexity</th><th class="n">Characters</th></tr>
<tr><td>Val</td><td>mostly <i>The Taming of the Shrew</i></td><td class="n">{tg["val"]["loss"]:.3f}</td><td class="n">{tg["val"]["perplexity"]:.2f}</td><td class="n">{tg["val"]["characters"]:,}</td></tr>
<tr class="hi"><td>Test</td><td>mostly <i>The Tempest</i>, never seen</td><td class="n">{tg["test"]["loss"]:.3f}</td><td class="n">{tg["test"]["perplexity"]:.2f}</td><td class="n">{tg["test"]["characters"]:,}</td></tr></table>
<ul class="points">
<li>Every character is scored once, in order (no random batches), so the number is the same every run.</li>
<li>Test is {tg["test"]["loss"] - tg["val"]["loss"]:+.3f} worse because it's a different play: 62% of its speeches are by characters never seen in training (val: 7%).</li>
<li>GPT small shows the same gap ({test["gpt_small"]["val"]["loss"]:.3f} → {test["gpt_small"]["test"]["loss"]:.3f}), so this comes from the data, not a bug.</li>
<li>Perplexity {tg["test"]["perplexity"]:.2f} means that, on new text, the model is about as unsure as picking between ~{tg["test"]["perplexity"]:.0f} equally likely characters (out of 65).</li>
</ul></div>

<h2>3. How we got here: all models</h2>
<div class="panel">{bar_chart([(m, v, "model 4" in m) for m, _, v, _, _, _ in models])}
<p class="sub" style="margin:6px 0 12px">Best val loss, where lower is better.</p>
<table><tr><th>Model</th><th>What it can do</th><th class="n">Parameters</th><th class="n">Val loss</th><th class="n">Perplexity</th><th class="n">Test loss</th><th class="n">Time</th></tr>{table_rows}</table>
</div>

<h2>4. Text while it trained (model 4)</h2>
<p class="sub">Same start every time, written at each checkpoint during training.</p>
<div class="cards">{progress_cards}</div>

<h2>5. Text at different temperatures (best model)</h2>
<p class="sub">Start: "ROMEO:". Same seed, only the setting changes.</p>
<div class="cards">{temp_cards}</div>

<h2>6. Setup</h2>
<div class="panel"><ul class="points">
<li><b>Data:</b> Tiny Shakespeare (1.1M characters), split by position into 90% train / 5% val / 5% test.</li>
<li><b>Tokenizer:</b> 65 characters, built from the train text only.</li>
<li><b>Model:</b> {cfg["n_layer"]} layers, {cfg["n_head"]} heads, width {cfg["n_embd"]}, context {cfg["block_size"]} characters, dropout {cfg["dropout"]}, GPT-2 design.</li>
<li><b>Training:</b> {cfg["steps"]} steps, batch {cfg["batch_size"]}, AdamW, learning rate {cfg["lr"]} with warm-up then cosine to {cfg["min_lr"]}, weight decay {cfg["weight_decay"]}, gradient clip {cfg["grad_clip"]}.</li>
<li><b>Next ideas:</b> a BPE tokenizer (word pieces), more data (TinyStories), stopping earlier.</li>
</ul></div>

</main></body></html>"""

OUT.write_text(page, encoding="utf-8")
print(f"Saved -> {OUT}")
