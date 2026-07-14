#!/usr/bin/env python3
"""
paper_figures_v2.py -- figures for the corrected paper.

The previous figures are all void: they were computed under the user-major ordering
described in Section 6.6 and must not be reused.

Every number here is read from a measured artifact, not retyped by hand where a file exists.

Figures:
  fig_cutoff   the money figure -- the ranking of methods CROSSES as K moves
  fig_stream   Protocol B: counting beats gradient continual learning on all four datasets
  fig_ordering the ordering error inverts the conclusion
  fig_revisit  the counter's accuracy is a monotone function of the revisit rate
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figs_v2"
OUT.mkdir(exist_ok=True)

# Okabe-Ito, colour-vision safe
BLUE, VERM, GREEN, ORANGE, GREY, PURPLE = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#666666", "#CC79A7")
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

K = [1, 5, 10, 20]

# ---------------------------------------------------------------- fig_cutoff
# Panel A: OUR runs, Protocol A (GETNext's own evaluation instances, full catalogue)
OURS = {
    "COUNT (same info)":      [.2344, .5396, .6505, .7222],
    "COUNT (T0 only)":        [.2368, .5249, .6205, .6843],
    "GETNext (as released)":  [.2459, .4709, .5538, .6126],
    "GETNext (index-fixed)":  [.2052, .4153, .4978, .5656],
    "Popularity":             [.0086, .0286, .0431, .0593],
}
# Panel B: PUBLISHED numbers, the authors' own 80/10/10 split (NOT comparable to panel A)
PUB = {
    "STHGCN (SIGIR'23)":  [.2734, .5361, .6244, None],
    "GETNext (SIGIR'22)": [.2435, .5089, .6143, None],
    "LLM4POI (SIGIR'24)": [.3372, .3982, .5010, None],
}
# Panel C: candidate re-ranking protocol (C=50, shuffled) -- LLM can only be run here
CAND = {
    "COUNT":                  [.1889, .4889, .6202, None],
    "Qwen2.5-7B (zero-shot)": [.0354, .1293, .2283, None],
    "Chance":                 [.0154, .0771, .1541, None],
}
CEILING = .7707

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

ax = axes[0]
for name, ys, c, ls, m in [
    ("COUNT (same info)", OURS["COUNT (same info)"], BLUE, "-", "o"),
    ("COUNT (T0 only)", OURS["COUNT (T0 only)"], BLUE, "--", "s"),
    ("GETNext (as released)", OURS["GETNext (as released)"], VERM, "-", "^"),
    ("GETNext (index-fixed)", OURS["GETNext (index-fixed)"], VERM, ":", "v"),
    ("Popularity", OURS["Popularity"], GREY, "-.", "x"),
]:
    ax.plot(K, ys, ls, color=c, marker=m, ms=4, lw=1.6, label=name)
ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels(K)
ax.set_xlabel("K"); ax.set_ylabel("Acc@K")
ax.set_ylim(-0.02, 0.86)
ax.set_title("(a) We ran these.\nGETNext's protocol, its own instances", fontsize=9)
ax.legend(fontsize=6.4, loc="upper left", frameon=False, borderpad=0.2)

# Inset: the sign of (COUNT - GETNext) flips between K=1 and K=5. On the Acc@K axes the two
# curves nearly touch at K=1, which is exactly the point a reader would miss.
ins = ax.inset_axes([0.55, 0.06, 0.42, 0.30])
delta = [100 * (c - g) / g for c, g in
         zip(OURS["COUNT (same info)"], OURS["GETNext (as released)"])]
cols = [VERM if d < 0 else BLUE for d in delta]
ins.bar(range(len(K)), delta, color=cols, width=0.62, edgecolor="white", lw=0.5)
ins.axhline(0, color="black", lw=0.7)
ins.set_xticks(range(len(K))); ins.set_xticklabels(K, fontsize=6)
ins.tick_params(axis="y", labelsize=6, pad=1)
ins.set_title(r"COUNT $-$ GETNext (%)", fontsize=6.5, pad=2)
ins.grid(False)
for i, d in enumerate(delta):
    ins.annotate(f"{d:+.0f}", (i, d), ha="center", fontsize=5.6,
                 va="bottom" if d > 0 else "top",
                 xytext=(0, 1 if d > 0 else -1), textcoords="offset points")
ins.set_ylim(min(delta) * 1.9, max(delta) * 1.35)

ax = axes[1]
for name, ys, c, m in [
    ("STHGCN (SIGIR'23)", PUB["STHGCN (SIGIR'23)"], GREEN, "s"),
    ("GETNext (SIGIR'22)", PUB["GETNext (SIGIR'22)"], VERM, "^"),
    ("LLM4POI (SIGIR'24)", PUB["LLM4POI (SIGIR'24)"], PURPLE, "D"),
]:
    kk = [k for k, y in zip(K, ys) if y is not None]
    yy = [y for y in ys if y is not None]
    ax.plot(kk, yy, "-", color=c, marker=m, ms=4, lw=1.6, label=name)
ax.annotate("LLM4POI's paper\nreports Acc@1\nand stops here",
            xy=(1.03, .3372), xytext=(1.9, .285), fontsize=6.8, color=PURPLE,
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.8))
ax.annotate("and here it is below\nboth models it\n'supersedes'",
            xy=(9.5, .5010), xytext=(3.1, .445), fontsize=6.8, color=PURPLE,
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.8))
ax.set_xscale("log"); ax.set_xticks([1, 5, 10]); ax.set_xticklabels([1, 5, 10])
ax.set_xlabel("K"); ax.set_ylabel("Acc@K")
ax.set_ylim(.20, .70)
ax.set_title("(b) Published numbers.\nThe LLM wins at K=1, loses at K=10", fontsize=9)
ax.legend(fontsize=6.4, loc="upper left", frameon=False, borderpad=0.2)

ax = axes[2]
for name, ys, c, m in [
    ("COUNT", CAND["COUNT"], BLUE, "o"),
    ("Qwen2.5-7B (zero-shot)", CAND["Qwen2.5-7B (zero-shot)"], PURPLE, "D"),
    ("Chance (shuffled cands.)", CAND["Chance"], GREY, "x"),
]:
    kk = [k for k, y in zip(K, ys) if y is not None]
    yy = [y for y in ys if y is not None]
    ax.plot(kk, yy, "-", color=c, marker=m, ms=4, lw=1.6, label=name)
ax.axhline(CEILING, color=GREEN, ls="--", lw=1.0)
ax.annotate("retriever ceiling (recall@50)", xy=(1.05, CEILING - .055), fontsize=6.8,
            color=GREEN)
ax.annotate("captures 76% of\nthe available signal", xy=(10, .6202), xytext=(2.0, .655),
            fontsize=6.8, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
ax.annotate("captures 12%", xy=(10, .2283), xytext=(3.2, .30), fontsize=6.8, color=PURPLE,
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.8))
ax.set_xscale("log"); ax.set_xticks([1, 5, 10]); ax.set_xticklabels([1, 5, 10])
ax.set_xlabel("K"); ax.set_ylabel("Acc@K")
ax.set_ylim(-0.02, 0.86)
ax.set_title("(c) Same 50 candidates.\nThe 7B model is barely above chance", fontsize=9)
ax.legend(fontsize=6.4, loc="center left", frameon=False, borderpad=0.2)

fig.savefig(OUT / "fig_cutoff.pdf")
plt.close(fig)
print("-> fig_cutoff.pdf")

# ---------------------------------------------------------------- fig_stream
# One estimator for the whole paper: per-instance mean, expected-rank ties (see gen_tables.py).
# That is `acc@10m`, not `acc@10` (per-round macro average, torch.topk index tie-breaking).
ACC = "acc@10m"
DS = ["nyc", "tky", "gowalla_ca", "brightkite"]
DISP = ["FSQ-NYC", "FSQ-TKY", "Gowalla-CA", "Brightkite-US"]
rows = {}
for d in DS:
    R = pd.read_csv(ROOT / f"results_chrono_{d}_gru.csv")
    rows[d] = R.groupby("policy")[ACC].mean().to_dict()
cnt = pd.read_csv(ROOT / "results_chrono_count.csv").set_index("dataset")[ACC].to_dict()

ORDER = [("static", "static (frozen)", GREY),
         ("periodic-4", "periodic-4", VERM),
         ("selective-gated", "selective", VERM),
         ("always+replay", "always+replay", VERM),
         ("EWC", "EWC", VERM),
         ("ADER", "ADER", VERM),
         ("GIRAM", "GIRAM (memory)", BLUE),
         ("__count__", "COUNT (no neural model)", GREEN)]

fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), sharey=False)
for ax, d, disp in zip(axes, DS, DISP):
    vals, labs, cols = [], [], []
    for key, lab, c in ORDER:
        v = cnt[d] if key == "__count__" else rows[d].get(key, np.nan)
        vals.append(v); labs.append(lab); cols.append(c)
    y = np.arange(len(vals))
    hatch = ["" if c != VERM else "//" for c in cols]
    for i, (v, c, h) in enumerate(zip(vals, cols, hatch)):
        ax.barh(i, v, color=c, hatch=h, edgecolor="white", lw=0.6, height=0.72)
    ax.axvline(rows[d]["static"], color=GREY, ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labs if d == "nyc" else [""] * len(labs), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Acc@10")
    ax.set_title(disp, fontsize=9)
fig.suptitle("Protocol B (every transition, cold users kept). Hatched = updates the backbone by gradient.",
             fontsize=8, y=1.04)
fig.savefig(OUT / "fig_stream.pdf")
plt.close(fig)
print("-> fig_stream.pdf")

# ---------------------------------------------------------------- fig_ordering
fig, ax = plt.subplots(figsize=(4.6, 3.0))
meth = ["static", "always+replay", "GIRAM (memory)"]
before = [.3161, .2233, .3239]
after = [.3161, .4191, .5752]
x = np.arange(len(meth)); w = 0.36
ax.bar(x - w / 2, before, w, label="user-major rounds (the bug)", color=VERM, hatch="//",
       edgecolor="white", lw=0.6)
ax.bar(x + w / 2, after, w, label="chronological rounds (correct)", color=BLUE,
       edgecolor="white", lw=0.6)
ax.axhline(.3161, color=GREY, ls=":", lw=1)
ax.annotate("static is bit-identical\nunder both orderings\n(the invariant that proves\nonly the order changed)",
            xy=(0, .3161), xytext=(0.55, .085), fontsize=6.6, color=GREY,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.7))
ax.set_xticks(x); ax.set_xticklabels(meth, fontsize=8)
ax.set_ylabel("Acc@10 (NYC)")
ax.set_title("A sort key inverts the conclusion", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.savefig(OUT / "fig_ordering.pdf")
plt.close(fig)
print("-> fig_ordering.pdf")

# ---------------------------------------------------------------- fig_revisit
stats = {s["dataset"]: s for s in json.loads((ROOT / "dataset_stats.json").read_text())}
name_map = {"Foursquare-NYC": "nyc", "Foursquare-TKY": "tky",
            "Gowalla-CA": "gowalla_ca", "Brightkite-US": "brightkite"}
fig, ax = plt.subplots(figsize=(4.3, 3.0))
for disp, key in name_map.items():
    rv = stats[disp]["revisit_rate"]; acc = cnt[key]
    ax.scatter(rv, acc, s=55, color=BLUE, zorder=3)
    ax.annotate(disp, (rv, acc), textcoords="offset points", xytext=(6, -3), fontsize=7.5)
xs = np.array([stats[d]["revisit_rate"] for d in name_map])
ys = np.array([cnt[name_map[d]] for d in name_map])
o = np.argsort(xs)
ax.plot(xs[o], ys[o], "-", color=BLUE, alpha=0.35, lw=1.2, zorder=2)
ax.set_xlabel("revisit rate (a property of the data)")
ax.set_ylabel("COUNT  Acc@10")
ax.set_title("The counter's accuracy is the data's,\nnot the model's", fontsize=9)
ax.set_xlim(0.35, 0.92); ax.set_ylim(0.15, 0.78)
fig.savefig(OUT / "fig_revisit.pdf")
plt.close(fig)
print("-> fig_revisit.pdf")

print(f"\nAll figures -> {OUT}")
