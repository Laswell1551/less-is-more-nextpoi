#!/usr/bin/env python3
"""Promote existing multi-seed results into LaTeX-ready table numbers (no new runs).
  (A) full robustness grid: method x backbone x dataset  (Acc@10 mean)
  (B) significance + effect size: GRU, method vs static, per dataset (Delta, paired p, Cohen d)
  (C) observation per-dataset stats: revisit, cold-user, drift, rho_chg, retention
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DS = ["nyc", "tky", "gowalla_ca", "brightkite"]
BB = ["gru", "attn", "getnext"]
METH = ["static", "always+replay", "periodic-4", "EWC", "ADER",
        "selective-gated", "GIRAM", "GIRAM-VAE"]


def load(ds, bb):
    f = ROOT / f"results_seed_{ds}_{bb}.csv"
    return pd.read_csv(f) if f.exists() else None


print("=" * 70, "\n(A) FULL ROBUSTNESS  Acc@10 mean (method x backbone x dataset)\n" + "=" * 70)
for bb in BB:
    print(f"\n-- backbone={bb} --")
    print(f"{'method':18s}" + "".join(f"{d:>12s}" for d in DS))
    for m in METH:
        row = f"{m:18s}"
        for ds in DS:
            R = load(ds, bb)
            if R is None or m not in set(R.policy):
                row += f"{'-':>12s}"
            else:
                row += f"{R[R.policy == m]['acc@10'].mean():>12.4f}"
        print(row)

print("\n" + "=" * 70, "\n(B) SIGNIFICANCE + EFFECT SIZE  (GRU; method vs static, per dataset)\n" + "=" * 70)
print(f"{'dataset':12s}{'method':16s}{'Delta':>9s}{'p(paired)':>11s}{'Cohen d':>9s}")
for ds in DS:
    R = load(ds, "gru")
    if R is None:
        continue
    s = R[R.policy == "static"].sort_values("seed")["acc@10"].to_numpy()
    for m in METH[1:]:
        if m not in set(R.policy):
            continue
        y = R[R.policy == m].sort_values("seed")["acc@10"].to_numpy()
        diff = y - s
        d = diff.mean() / (diff.std(ddof=1) + 1e-12)          # paired Cohen's d
        t, p = stats.ttest_rel(y, s)
        print(f"{ds:12s}{m:16s}{diff.mean():>+9.4f}{p:>11.2e}{d:>9.2f}")

print("\n" + "=" * 70, "\n(C) OBSERVATION STATS per dataset\n" + "=" * 70)
summ = json.load(open(ROOT / "data" / "processed" / "summary.json"))
reg = json.load(open(ROOT / "regime_stats.json")) if (ROOT / "regime_stats.json").exists() else {}
red = json.load(open(ROOT / "neural_redundancy.json")) if (ROOT / "neural_redundancy.json").exists() else {}
print(f"{'dataset':12s}{'revisit':>9s}{'cold-usr':>9s}{'driftJS':>9s}{'rho_chg':>9s}")
for ds in DS:
    rv = summ.get(ds, {}).get("revisit_rate", float("nan"))
    cold = reg.get(ds, {}).get("frac_cold_user", float("nan"))
    drift = (reg.get(ds, {}).get("drift_js_T0_to_Tk", [float("nan")]) or [float("nan")])[-1]
    rc = red.get(ds, {}).get("update_changes_top1_frac", float("nan"))
    print(f"{ds:12s}{rv:>9.3f}{cold:>9.3f}{drift:>9.3f}{rc:>9.3f}")

print("\n" + "=" * 70, "\n(D) MULTI-K + MRR  (GRU, mean over seeds; streaming)\n" + "=" * 70)
KEYM = ["static", "always+replay", "EWC", "ADER", "selective-gated", "GIRAM", "GIRAM-VAE"]
for ds in DS:
    R = load(ds, "gru")
    if R is None or "mrr" not in R.columns:
        continue
    print(f"\n-- {ds} --")
    print(f"{'method':16s}{'Acc@1':>8s}{'Acc@5':>8s}{'Acc@10':>8s}{'Acc@20':>8s}{'MRR':>8s}")
    g = R.groupby("policy")
    for m in KEYM:
        if m not in set(R.policy):
            continue
        s = g.get_group(m)
        print(f"{m:16s}{s['acc@1'].mean():>8.4f}{s['acc@5'].mean():>8.4f}"
              f"{s['acc@10'].mean():>8.4f}{s['acc@20'].mean():>8.4f}{s['mrr'].mean():>8.4f}")
