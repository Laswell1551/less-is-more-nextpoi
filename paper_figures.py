#!/usr/bin/env python3
"""Publication figure set for "Less is More" (ANCHOR), from the real results.
Outputs vector PDFs + PNG previews into experiments/figs_paper/.
  fig_main      : Acc@10 per method, colored by the fine-tune-backbone fault line.
  fig_pareto    : accuracy vs. churn -- the method-value view (ANCHOR dominates).
  fig_cost      : forgetting and churn (the price of fine-tuning).
  fig_regime    : warm/cold composition + drift (answers the regime critique).
  fig_cold      : warm vs. cold-start accuracy (hard for every method).
  fig_redundancy: neural retraining changes much, gains nothing.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figs_paper"; FIGS.mkdir(exist_ok=True)
DATASETS = ["nyc", "tky", "gowalla_ca", "brightkite"]
NAMES = {"nyc": "Foursquare-NYC", "tky": "Foursquare-TKY", "gowalla_ca": "Gowalla-CA",
         "brightkite": "Brightkite-US"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False, "legend.fontsize": 8.5,
})

ORDER = ["static", "GIRAM", "GIRAM-VAE", "selective-gated", "EWC", "ADER", "periodic-4", "always+replay"]
LABEL = {"static": "static", "GIRAM": "memory", "GIRAM+": "memory+", "GIRAM-VAE": "full GIRAM",
         "selective-gated": "selective", "EWC": "EWC", "ADER": "ADER", "periodic-4": "periodic",
         "always+replay": "always"}
FT = {"static": 0, "GIRAM": 0, "GIRAM+": 0, "GIRAM-VAE": 0, "selective-gated": 0,
      "EWC": 1, "ADER": 1, "periodic-4": 1, "always+replay": 1}
# Colour-vision-safe palette (Okabe-Ito). Blue vs. vermillion is separable under
# deuteranopia/protanopia/tritanopia, unlike the green/red pair it replaces. Colour is
# never the only channel: fine-tuning bars are hatched and its lines are dashed.
C_NOFT, C_FT, C_STATIC = "#0072B2", "#D55E00", "#333333"
HATCH_FT = "//"
# Okabe-Ito categorical set, used where >3 series must be told apart.
OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def load(n):
    return pd.read_csv(ROOT / f"results_seed_{n}_gru.csv")


def stat(df, col):
    g = df.groupby("policy")[col]
    return g.mean(), g.std()


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)


def fig_main():
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
    for j, n in enumerate(DATASETS):
        m, s = stat(load(n), "acc@10")
        xs = np.arange(len(ORDER))
        col = [C_STATIC if p == "static" else (C_FT if FT[p] else C_NOFT) for p in ORDER]
        b = ax[j].bar(xs, [m[p] for p in ORDER], yerr=[s[p] for p in ORDER], color=col,
                      capsize=2, edgecolor="white", linewidth=0.4)
        for p, bar in zip(ORDER, b):                    # hatch = second, non-colour channel
            if FT[p]:
                bar.set_hatch(HATCH_FT); bar.set_edgecolor("white")
        ax[j].axhline(m["static"], ls="--", c=C_STATIC, lw=0.9, zorder=0)
        ax[j].set_xticks(xs); ax[j].set_xticklabels([LABEL[p] for p in ORDER], rotation=45, ha="right")
        ax[j].set_title(NAMES[n]); ax[j].set_ylabel("Acc@10" if j == 0 else "")
        ax[j].margins(y=0.12)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C_STATIC),
               plt.Rectangle((0, 0), 1, 1, facecolor=C_NOFT),
               plt.Rectangle((0, 0), 1, 1, facecolor=C_FT, hatch=HATCH_FT, edgecolor="white")]
    fig.legend(handles, ["static (no update)", "adapts without fine-tuning backbone",
                         "fine-tunes backbone (hatched)"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(); _save(fig, "fig_main")


def fig_pareto():
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for j, n in enumerate(DATASETS):
        df = load(n); ma, _ = stat(df, "acc@10"); mc, _ = stat(df, "churn")
        for p in ORDER:
            c = C_STATIC if p == "static" else (C_FT if FT[p] else C_NOFT)
            mk = "*" if p == "static" else ("o" if not FT[p] else "s")
            ax[j].scatter(mc[p], ma[p], c=c, marker=mk, s=90 if p == "static" else 55,
                          edgecolor="white", linewidth=0.5, zorder=3)
            ax[j].annotate(LABEL[p], (mc[p], ma[p]), fontsize=7, xytext=(3, 3),
                           textcoords="offset points")
        ax[j].set_title(NAMES[n]); ax[j].set_xlabel("recommendation churn")
        ax[j].set_ylabel("Acc@10" if j == 0 else "")
    fig.tight_layout(); _save(fig, "fig_pareto")


def fig_cost():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    methods = [p for p in ORDER if p != "static"]
    for a, col, ttl in [(ax[0], "forget_drop", "Forgetting (Acc drop on $T_0$ probe)"),
                        (ax[1], "churn", "Recommendation churn")]:
        w = 0.2
        for i, n in enumerate(DATASETS):
            m, _ = stat(load(n), col)
            cols = [C_FT if FT[p] else C_NOFT for p in methods]
            bars = a.bar(np.arange(len(methods)) + i * w, [m[p] for p in methods], w,
                         color=cols, alpha=min(0.5 + 0.16 * i, 1.0), edgecolor="white", linewidth=0.3)
            for p, bar in zip(methods, bars):
                if FT[p]:
                    bar.set_hatch(HATCH_FT)
        a.set_xticks(np.arange(len(methods)) + w)
        a.set_xticklabels([LABEL[p] for p in methods], rotation=45, ha="right")
        a.set_title(ttl); a.axhline(0, c="k", lw=0.6)
    fig.tight_layout(); _save(fig, "fig_cost")


def fig_regime():
    reg = json.load(open(ROOT / "regime_stats.json"))
    RDS = [d for d in DATASETS if d in reg]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
    xs = np.arange(len(RDS))
    warm = [reg[n]["frac_warm_user"] for n in RDS]
    cold = [reg[n]["frac_cold_user"] for n in RDS]
    ax[0].bar(xs, warm, label="warm users", color="#4c72b0")
    ax[0].bar(xs, cold, bottom=warm, label="cold (new) users", color="#dd8452")
    ax[0].set_xticks(xs); ax[0].set_xticklabels([NAMES[n] for n in RDS], rotation=18, ha="right")
    ax[0].set_ylabel("share of test transitions"); ax[0].set_title("Composition"); ax[0].legend()
    for n in RDS:
        ax[1].plot(range(1, 6), reg[n]["drift_js_T0_to_Tk"], "-o", label=NAMES[n], lw=1.6)
    ax[1].set_xlabel("increment $T_k$"); ax[1].set_ylabel("JS drift from $T_0$")
    ax[1].set_title("Drift magnitude"); ax[1].legend()
    fig.tight_layout(); _save(fig, "fig_regime")


def fig_cold():
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    keys = ["static", "GIRAM", "selective-gated", "always+replay"]
    w = 0.2
    for i, n in enumerate(DATASETS):
        df = load(n); warm, _ = stat(df, "acc_warm"); cold, _ = stat(df, "acc_cold")
        ax.bar(np.arange(len(keys)) + i * w, [warm[k] for k in keys], w,
               color="#4c72b0", alpha=min(0.5 + 0.16 * i, 1.0), label=("warm" if i == 1 else None))
        ax.bar(np.arange(len(keys)) + i * w, [cold[k] for k in keys], w,
               color="#dd8452", alpha=min(0.5 + 0.16 * i, 1.0), label=("cold-start" if i == 1 else None))
    ax.set_xticks(np.arange(len(keys)) + w); ax.set_xticklabels([LABEL[k] for k in keys], rotation=20, ha="right")
    ax.set_ylabel("Acc@10"); ax.legend()          # title lives in the caption, not on the figure
    fig.tight_layout(); _save(fig, "fig_cold")


def fig_redundancy():
    red = json.load(open(ROOT / "neural_redundancy.json"))
    RDS = [d for d in DATASETS if d in red]
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    xs = np.arange(len(RDS))
    chg = [red[n]["update_changes_top1_frac"] for n in RDS]
    gain = [red[n]["top1_acc_refit"] - red[n]["top1_acc_static"] for n in RDS]
    ax.bar(xs - 0.2, chg, 0.4, label="fraction of top-1 changed", color=OI[0])
    b2 = ax.bar(xs + 0.2, gain, 0.4, label="top-1 accuracy gain", color=OI[1])
    for bar in b2:
        bar.set_hatch("//"); bar.set_edgecolor("white")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([NAMES[n] for n in RDS], rotation=18, ha="right")
    ax.legend()                                    # title lives in the caption
    fig.tight_layout(); _save(fig, "fig_redundancy")


def fig_rounds():
    """Per-round Acc@10 trajectory: static flat, fine-tuning decays."""
    import matplotlib.lines as mlines
    files = {n: ROOT / f"results_rounds_{n}_gru.csv" for n in DATASETS}
    if not any(f.exists() for f in files.values()):
        return
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.0))
    for j, n in enumerate(DATASETS):
        if not files[n].exists():
            continue
        df = pd.read_csv(files[n])
        for p in ORDER:
            sub = df[df.policy == p].sort_values("round")
            if sub.empty:
                continue
            c = C_STATIC if p == "static" else (C_FT if FT[p] else C_NOFT)
            ax[j].plot(sub["round"], sub["acc"], color=c,
                       ls="--" if FT[p] else "-",     # dash = second, non-colour channel
                       lw=2.3 if p == "static" else 1.0, alpha=0.9 if p == "static" else 0.65)
        ax[j].set_title(NAMES[n]); ax[j].set_xlabel("round")
        ax[j].set_ylabel("Acc@10" if j == 0 else "")
    handles = [mlines.Line2D([], [], color=C_STATIC, lw=2),
               mlines.Line2D([], [], color=C_NOFT, lw=2),
               mlines.Line2D([], [], color=C_FT, lw=2, ls="--")]
    fig.legend(handles, ["static", "freeze backbone", "fine-tune backbone (dashed)"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(); _save(fig, "fig_rounds")


def fig_lr():
    """Accuracy (top) and forgetting (bottom) vs learning rate: the 'FT loses' result
    is an LR artifact, and at the sweet spot accuracy > static while forgetting -> 0
    (a real sweet spot, not the LR->0 tautology)."""
    j = ROOT / "lr_sweep.json"
    if not j.exists():
        return
    d = json.load(open(j))
    if "acc" not in d[next(iter(d))]:                 # old flat format; wait for re-run
        return
    lrs = [1e-3, 3e-4, 1e-4, 3e-5]; ks = [f"{lr:.0e}" for lr in lrs]
    mc = {"always": (C_FT, "o"), "EWC": (OI[1], "s"), "ADER": (OI[3], "^")}  # colour + marker
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.2))
    for c, ds in enumerate(DATASETS):
        r = d[ds]
        for m, (col, mk) in mc.items():
            ax[0, c].plot(lrs, [r["acc"][m][k] for k in ks], ls="-", marker=mk, color=col, label=m, lw=1.3, ms=4)
            ax[1, c].plot(lrs, [r["forget"][m][k] for k in ks], ls="-", marker=mk, color=col, lw=1.3, ms=4)
        ax[0, c].axhline(r["static"], ls="--", c=C_STATIC, lw=1.1, label="static")
        ax[1, c].axhline(0, c="k", lw=0.6)
        for rr in (0, 1):
            ax[rr, c].set_xscale("log"); ax[rr, c].invert_xaxis()
        ax[0, c].set_title(NAMES[ds]); ax[1, c].set_xlabel("fine-tuning LR")
        ax[0, c].set_ylabel("Acc@10" if c == 0 else ""); ax[1, c].set_ylabel("forgetting" if c == 0 else "")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); _save(fig, "fig_lr")


def fig_regime_crit():
    """Single panel: the failed a-priori criterion -- the new-POI rate is a weak,
    dataset-clustered predictor of the FT-static gap (the negative result of Sec 6)."""
    j = ROOT / "regime_sweep.json"
    if not j.exists():
        return
    rows = list(json.load(open(j)).values())
    cols = dict(zip(DATASETS, OI[:4]))              # colour-vision-safe, 4 categories
    mks = dict(zip(DATASETS, ["o", "s", "^", "D"]))  # marker = second channel
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.2))
    for ds in DATASETS:
        rs = [r for r in rows if r["dataset"] == ds]
        ax.scatter([r["new_poi"] for r in rs], [r["FT_minus_static"] for r in rs],
                   color=cols[ds], marker=mks[ds], s=32, label=NAMES[ds])
    x = np.array([r["new_poi"] for r in rows]); y = np.array([r["FT_minus_static"] for r in rows])
    a, b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, a * xs + b, "k--", lw=1); ax.axhline(0, c="k", lw=0.4)
    ax.set_xlabel("new-POI rate (a priori)"); ax.set_ylabel("FT $-$ static")
    ax.legend(fontsize=6.5)                        # r is reported in the caption
    fig.tight_layout(); _save(fig, "fig_regime_crit")


def fig_inject():
    """The high point (Sec 7): clean geographic injection. Left: only the memory gains;
    fixed-LR FT and the steelman controller tie or lose. Right: on injected (new) POIs,
    fine-tuning is no better than the frozen model; the memory is ~10x better."""
    f1 = ROOT / "inject_rigorous.json"; f2 = ROOT / "inject_rigorous_ft.json"
    if not (f1.exists() and f2.exists()):
        return
    m = json.load(open(f1)); ft = json.load(open(f2)); dss = DATASETS
    x = np.arange(len(dss)); w = 0.25
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
    d_fx = [ft[d]["fixedFT"][1] for d in dss]; d_ct = [ft[d]["controller"][1] for d in dss]
    d_me = [m[d]["mem_minus_static"][0] for d in dss]
    def _hatch(bars, h):
        for bar in bars:
            bar.set_hatch(h); bar.set_edgecolor("white")

    _hatch(ax[0].bar(x - w, d_fx, w, label="fixed-LR FT", color=C_FT), HATCH_FT)
    _hatch(ax[0].bar(x, d_ct, w, label="controller (steelman)", color=OI[1]), "\\\\")
    ax[0].bar(x + w, d_me, w, label="memory (counting)", color=C_NOFT)
    ax[0].axhline(0, c="k", lw=0.6); ax[0].set_xticks(x)
    ax[0].set_xticklabels([NAMES[d] for d in dss], rotation=12, fontsize=7)
    ax[0].set_ylabel(r"$\Delta$Acc@10 vs. static"); ax[0].legend(fontsize=6.5)
    s = [ft[d]["NEWtarget"]["static"] for d in dss]; fx = [ft[d]["NEWtarget"]["fixedFT"] for d in dss]
    ct = [ft[d]["NEWtarget"]["controller"] for d in dss]; me = [m[d]["NEWtarget_memory"] for d in dss]
    ax[1].bar(x - 1.5 * w, s, w, label="static", color=C_STATIC)
    _hatch(ax[1].bar(x - 0.5 * w, fx, w, label="fixed FT", color=C_FT), HATCH_FT)
    _hatch(ax[1].bar(x + 0.5 * w, ct, w, label="controller", color=OI[1]), "\\\\")
    ax[1].bar(x + 1.5 * w, me, w, label="memory", color=C_NOFT)
    ax[1].set_xticks(x); ax[1].set_xticklabels([NAMES[d] for d in dss], rotation=12, fontsize=7)
    ax[1].set_ylabel("Acc@10 on new POIs"); ax[1].legend(fontsize=6.5)
    fig.tight_layout(); _save(fig, "fig_inject")


if __name__ == "__main__":
    fig_main(); fig_pareto(); fig_cost(); fig_regime(); fig_cold(); fig_redundancy(); fig_rounds()
    fig_lr(); fig_regime_crit(); fig_inject()
    print(f"figures -> {FIGS}")
