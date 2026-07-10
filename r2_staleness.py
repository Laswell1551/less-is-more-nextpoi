#!/usr/bin/env python3
"""
r2_staleness.py -- the *correct* test of R2 (decision temporal sparsity), i.e. the
direct justification for change-gated lazy updates: STALENESS TOLERANCE.

We freeze a transparent personalized recommender at block T0 and evaluate it on
T1..T5 WITHOUT updating ("frozen"), against the same model refit cumulatively on
T0..T(k-1) before each block ("refit"). If the frozen model stays close to refit,
the optimal decision is temporally stable -> updates can be sparse/lazy.

Recommender = personalized 1st-order Markov (last POI -> next), backing off to
personalized popularity then global popularity. No GPU, no learning -- a clean,
hard-to-game reference. Also reports the frozen-vs-refit top-1 agreement
(= fraction of decisions an update would NOT change).

Usage: python r2_staleness.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "figs"
DATASETS = ["nyc", "tky", "gowalla_ca", "brightkite"]
COLORS = {"nyc": "#1f77b4", "tky": "#d62728", "gowalla_ca": "#2ca02c",
          "brightkite": "#9467bd"}
BLOCKS = [f"T{i}" for i in range(6)]
K = 10


def load(name):
    return pd.read_csv(PROC / name / "checkins.csv.gz").sort_values(
        ["user_idx", "unix_s"]).reset_index(drop=True)


def build_markov(df):
    """Return (trans, upop, gpop_top) from transitions within trajectories."""
    trans = defaultdict(lambda: defaultdict(Counter))
    upop = defaultdict(Counter)
    gpop = Counter()
    u = df.user_idx.to_numpy(); p = df.poi_idx.to_numpy(); t = df.traj_id.to_numpy()
    pu = pp = pt = None
    for i in range(len(df)):
        ui, pi, ti = u[i], p[i], t[i]
        gpop[pi] += 1; upop[ui][pi] += 1
        if pt == ti and pu == ui:
            trans[ui][pp][pi] += 1
        pu, pp, pt = ui, pi, ti
    gpop_top = gpop.most_common(200)
    return trans, upop, gpop_top


def predict(u, last, model, k=K):
    trans, upop, gpop_top = model
    s = {}
    tl = trans.get(u, {}).get(last)
    if tl:
        for q, c in tl.items():
            s[q] = s.get(q, 0) + c * 1e12
    up = upop.get(u)
    if up:
        for q, c in up.items():
            s[q] = s.get(q, 0) + c * 1e6
    for q, c in gpop_top:
        s[q] = s.get(q, 0) + c
    return [q for q, _ in sorted(s.items(), key=lambda x: -x[1])[:k]]


def transitions(df):
    """Yield (user, last_poi, next_poi) within trajectories."""
    u = df.user_idx.to_numpy(); p = df.poi_idx.to_numpy(); t = df.traj_id.to_numpy()
    out = []
    for i in range(1, len(df)):
        if t[i] == t[i - 1] and u[i] == u[i - 1]:
            out.append((u[i], p[i - 1], p[i]))
    return out


def evaluate(test_trans, model, ref_model=None, k=K):
    """Acc@k, MRR; if ref_model given, also top-1 agreement (frozen vs refit)."""
    hit = rr = agree = n = 0
    for u, last, nxt in test_trans:
        topk = predict(u, last, model, k)
        if nxt in topk:
            hit += 1; rr += 1.0 / (topk.index(nxt) + 1)
        if ref_model is not None:
            r1 = predict(u, last, ref_model, 1)
            f1 = topk[0] if topk else None
            agree += int(bool(r1) and r1[0] == f1)
        n += 1
    res = {"acc@%d" % k: hit / max(n, 1), "mrr": rr / max(n, 1), "n": n}
    if ref_model is not None:
        res["top1_agree"] = agree / max(n, 1)
    return res


def run(name):
    df = load(name)
    by = {b: df[df.block == b] for b in BLOCKS}
    test = {b: transitions(by[b]) for b in BLOCKS[1:]}
    frozen = build_markov(by["T0"])
    refit = {}                       # model trained on T0..T(k-1), evaluated on Tk
    acc_frozen, acc_refit, agree = [], [], []
    for k in range(1, 6):
        cum = pd.concat([by[f"T{j}"] for j in range(k)])
        refit[k] = build_markov(cum)
        rf = evaluate(test[f"T{k}"], frozen, ref_model=refit[k])
        rr = evaluate(test[f"T{k}"], refit[k])
        acc_frozen.append(rf["acc@%d" % K]); acc_refit.append(rr["acc@%d" % K])
        agree.append(rf["top1_agree"])
    gap = float(np.mean(np.array(acc_refit) - np.array(acc_frozen)))
    out = {
        "acc_frozen_T1toT5": [round(x, 4) for x in acc_frozen],
        "acc_refit_T1toT5": [round(x, 4) for x in acc_refit],
        "mean_lazy_gap": round(gap, 4),
        "retention_at_T5": round(acc_frozen[-1] / max(acc_refit[-1], 1e-9), 4),
        "mean_top1_agreement": round(float(np.mean(agree)), 4),
    }
    print(f"[{name}]  frozen@T0 Acc@10 across T1..T5 = "
          f"{[round(x,3) for x in acc_frozen]}")
    print(f"          refit      Acc@10 across T1..T5 = "
          f"{[round(x,3) for x in acc_refit]}")
    print(f"          lazy gap = {gap:.4f}  | T5 retention = {out['retention_at_T5']*100:.1f}% "
          f"| update-changes-top1 only {(1-out['mean_top1_agreement'])*100:.1f}% of steps")
    return acc_frozen, acc_refit, out


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    head = {}
    plt.figure(figsize=(5.2, 3.8))
    x = list(range(1, 6))
    for n in DATASETS:
        af, ar, out = run(n)
        head[n] = out
        plt.plot(x, ar, "-o", color=COLORS[n], lw=1.5, label=f"{n} refit")
        plt.plot(x, af, "--s", color=COLORS[n], lw=1.2, mfc="none", label=f"{n} frozen@T0")
    plt.xlabel("evaluation block T_k"); plt.ylabel(f"Acc@{K}")
    plt.title("R2 (corrected): staleness tolerance\nfrozen-at-T0 vs cumulatively refit")
    plt.legend(fontsize=7, ncol=3); plt.tight_layout()
    plt.savefig(FIGS / "obs_R2_staleness.png", dpi=160); plt.close()
    json.dump(head, open(ROOT / "observations_R2_staleness.json", "w"), indent=2)
    print(f"\nfigure -> {FIGS}\\obs_R2_staleness.png")


if __name__ == "__main__":
    main()
