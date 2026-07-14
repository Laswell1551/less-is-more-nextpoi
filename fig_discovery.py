#!/usr/bin/env python3
"""
fig_discovery.py -- the paper's two central figures.

fig_discovery : the decomposition on the standard benchmark's own NYC instances.
fig_score     : where the score actually comes from, on all four datasets. This is the one
                that carries the argument -- returns are 35-76% of the INSTANCES but
                96-99.7% of the SCORE, because return accuracy is 40-100x higher than
                discovery accuracy for every method.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figs_v2"
OUT.mkdir(exist_ok=True)

BLUE, VERM, GREEN, GREY, PURPLE, ORANGE = (
    "#0072B2", "#D55E00", "#009E73", "#666666", "#CC79A7", "#E69F00")
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})

d = json.loads((ROOT / "discovery_summary.json").read_text())
pa = d["protocol_A_full_catalogue"]
cr = d["candidate_reranking_50"]

# Shares are derived from the counts, never stored. An earlier revision of this script read
# pa["revisit_share"], a key gen_tables.py stopped emitting when the artifact was rebuilt --
# so the script raised, nobody could regenerate, and the figures silently stayed on the
# pre-expected-rank numbers while the prose moved on. Derive, do not duplicate.
pa["revisit_share"] = pa["revisit"] / pa["instances"]
pa["discovery_share"] = pa["discovery"] / pa["instances"]

METHODS = [
    ("Popularity",   pa["acc@10"]["popularity"],                      GREY,   ""),
    ("GETNext",      pa["acc@10"]["GETNext (official, as-released)"], VERM,   ""),
    ("Qwen-7B$^\\dagger$", cr["acc@10"]["Qwen2.5-7B (LoRA-T0, frozen)"], PURPLE, "///"),
    ("COUNT",        pa["acc@10"]["COUNT (same info as GETNext)"],     BLUE,   ""),
]

# ======================================================================= fig_discovery
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1),
                         gridspec_kw={"width_ratios": [0.6, 1, 1]})

ax = axes[0]
rev, dis = pa["revisit_share"], pa["discovery_share"]
ax.bar([0], [rev], color=BLUE, width=0.5)
ax.bar([0], [dis], bottom=[rev], color=GREEN, width=0.5)
ax.text(0, rev / 2, f"{rev:.0%}\nreturns", ha="center", va="center", color="white",
        fontsize=10, fontweight="bold")
ax.text(0, rev + dis / 2, f"{dis:.0%}\nnew", ha="center", va="center", color="white",
        fontsize=8.5)
ax.set_xlim(-0.45, 0.45); ax.set_xticks([]); ax.set_ylim(0, 1)
ax.set_ylabel("share of evaluation instances")
ax.set_title("(a) What the benchmark\nis made of", fontsize=9)
ax.grid(False)

for k, (ax, key, ylim, title) in enumerate([
        (axes[1], "revisit", 0.99, "(b) On returns (79%)\nthe counter is simply better"),
        (axes[2], "discovery", 0.056, "(c) On discoveries (21%)\neverything is below popularity")]):
    vals = [m[1][key] for m in METHODS]
    for i, m in enumerate(METHODS):
        ax.bar(i, m[1][key], color=m[2], hatch=m[3], edgecolor="white", lw=0.8, width=0.66)
    fmt = "{:.3f}" if key == "revisit" else "{:.4f}"
    for i, v in enumerate(vals):
        ax.text(i, v + ylim * 0.02, fmt.format(v), ha="center", fontsize=7.4)
    if key == "discovery":
        ax.axhline(vals[0], color=GREY, ls="--", lw=1.1)
        ax.annotate("popularity — no personalisation,\nno learning — is the best of the four",
                    xy=(1.0, vals[0]), xytext=(0.25, 0.0355), fontsize=7.0,
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m[0] for m in METHODS], fontsize=7.4, rotation=18, ha="right")
    ax.set_ylim(0, ylim); ax.set_ylabel("Acc@10")
    ax.set_title(title, fontsize=9)

fig.text(0.5, -0.13,
         r"$\dagger$ The language model is scored in the candidate re-ranking protocol "
         r"(a 7B model cannot rank a 4,561-POI catalogue per instance), so its "
         r"return/discovery $\it{contrast}$ is comparable but its $\it{level}$ is not; "
         r"it is hatched for that reason.",
         ha="center", fontsize=6.8, color="#333333")
fig.savefig(OUT / "fig_discovery.pdf")
plt.close(fig)
print("-> fig_discovery.pdf")

# ======================================================================= fig_score
sd = json.loads((ROOT / "score_decomposition.json").read_text())
dst = json.loads((ROOT / "discovery_stream.json").read_text())
DS = ["nyc", "tky", "gowalla_ca", "brightkite"]
LAB = ["Foursquare-NYC", "Foursquare-TKY", "Gowalla-CA", "Brightkite-US"]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})

# left: returns as a share of instances vs as a share of the score
ax = axes[0]
inst = [dst[k]["return_share"] for k in DS]
score = [sd[k]["COUNT"] for k in DS]
y = np.arange(len(DS)); h = 0.34
ax.barh(y - h / 2, inst, h, color=GREY, label="of the INSTANCES")
ax.barh(y + h / 2, score, h, color=BLUE, label="of the SCORE")
for i, (a, b) in enumerate(zip(inst, score)):
    ax.text(a + .015, i - h / 2, f"{a:.0%}", va="center", fontsize=7.6, color="#333")
    ax.text(b + .015, i + h / 2, f"{b:.1%}", va="center", fontsize=7.6, color=BLUE,
            fontweight="bold")
ax.axhspan(1.55, 2.45, color=ORANGE, alpha=0.14, lw=0)
ax.set_yticks(y); ax.set_yticklabels(LAB, fontsize=8); ax.invert_yaxis()
ax.set_xlim(0, 1.28); ax.set_ylim(3.75, -0.95)      # headroom above the top bar for the legend
ax.set_xlabel("returns, as a share of ...")
ax.set_title("Returns are a minority of what happens\nand almost all of what is scored",
             fontsize=9)
ax.legend(fontsize=7.2, loc="upper right", frameon=False, ncol=2,
          title="returns, as a share of:", title_fontsize=7,
          borderaxespad=0.2, columnspacing=1.0, handlelength=1.4)

# right: why -- return accuracy is an order of magnitude above discovery accuracy
ax = axes[1]
ret = [dst[k]["acc@10"]["COUNT"]["return"] for k in DS]
dis = [dst[k]["acc@10"]["COUNT"]["discovery"] for k in DS]
x = np.arange(len(DS)); w = 0.34
ax.bar(x - w / 2, ret, w, color=BLUE, label="COUNT on returns")
ax.bar(x + w / 2, dis, w, color=GREEN, label="COUNT on discoveries")
for i, (a, b) in enumerate(zip(ret, dis)):
    ax.text(i - w / 2, a + .015, f"{a:.2f}", ha="center", fontsize=7.4)
    ax.text(i + w / 2, b + .015, f"{b:.3f}", ha="center", fontsize=7.0, color=GREEN)
    ax.annotate(f"{a/b:.0f}$\\times$", xy=(i, max(a, b) + 0.10), ha="center", fontsize=7.6,
                color="#8a5a00", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["NYC", "TKY", "Gowalla", "Brightkite"], fontsize=8)
ax.set_ylim(0, 1.10); ax.set_ylabel("Acc@10")
ax.set_title("...because returns are 40-100$\\times$ easier.\nThat is why the sum is a return metric",
             fontsize=9)
ax.legend(fontsize=7.2, loc="upper center", frameon=False, ncol=1)

fig.savefig(OUT / "fig_score.pdf")
plt.close(fig)
print("-> fig_score.pdf")
for k, l in zip(DS, LAB):
    print(f"   {l:16s} returns {dst[k]['return_share']:5.1%} of instances -> "
          f"{sd[k]['COUNT']:5.1%} of COUNT's score")
