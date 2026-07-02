#!/usr/bin/env python3
"""
revision_analysis.py -- Round-1 revision item (1): characterize the streaming REGIME,
to answer the Devil's Advocate CRITICAL ("the setup may be tilted toward static").

For each dataset we report, on the test stream T1..T5:
  - warm/cold composition: share of test transitions from users / to POIs already
    seen in the base block T0 (warm), vs new (cold);
  - drift magnitude: JS divergence of the global POI distribution between T0 and each
    Tk, and per-user T0->T5 drift.
These say how warm-dominated and how mild-drift the regime is -> how much the
static-strong result is baked into the setup vs a genuine finding.

Pure pandas. -> prints + regime_stats.json
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
DATASETS = ["nyc", "tky", "gowalla_ca", "brightkite"]


def js(c1, c2):
    keys = set(c1) | set(c2)
    p = np.array([c1.get(k, 0) for k in keys], float); p /= p.sum()
    q = np.array([c2.get(k, 0) for k in keys], float); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def transitions(df):
    df = df.sort_values(["user_idx", "unix_s"])
    u = df.user_idx.to_numpy(); p = df.poi_idx.to_numpy(); t = df.traj_id.to_numpy()
    keep = (t[1:] == t[:-1]) & (u[1:] == u[:-1])
    return u[1:][keep], p[:-1][keep], p[1:][keep]   # user, last, target


def main():
    out = {}
    for name in DATASETS:
        df = pd.read_csv(PROC / name / "checkins.csv.gz")
        T0 = df[df.block == "T0"]
        warm_u = set(T0.user_idx.unique()); warm_p = set(T0.poi_idx.unique())
        test = df[df.block != "T0"]
        u, last, tgt = transitions(test)
        n = len(u)
        warm_user = np.isin(u, list(warm_u))
        warm_tgt = np.isin(tgt, list(warm_p))
        # drift: global POI distribution T0 vs Tk
        p0 = Counter(T0.poi_idx)
        drift = [round(js(p0, Counter(df[df.block == f"T{k}"].poi_idx)), 3) for k in range(1, 6)]
        # per-user T0->T5 drift (users present in both, >=10 checkins each side)
        T5 = df[df.block == "T5"]
        u0 = {uid: Counter(g) for uid, g in T0.groupby("user_idx")["poi_idx"]}
        u5 = {uid: Counter(g) for uid, g in T5.groupby("user_idx")["poi_idx"]}
        both = [uid for uid in u5 if uid in u0
                and sum(u0[uid].values()) >= 10 and sum(u5[uid].values()) >= 10]
        per_user = [js(u0[uid], u5[uid]) for uid in both]
        stats = {
            "test_transitions": int(n),
            "frac_warm_user": round(float(warm_user.mean()), 4),
            "frac_warm_target_poi": round(float(warm_tgt.mean()), 4),
            "frac_fully_warm": round(float((warm_user & warm_tgt).mean()), 4),
            "frac_cold_user": round(float((~warm_user).mean()), 4),
            "drift_js_T0_to_Tk": drift,
            "per_user_drift_T0_T5_median": round(float(np.median(per_user)), 3) if per_user else None,
            "n_users_in_both_blocks": len(both),
        }
        out[name] = stats
        print(f"[{name}] warm-user {stats['frac_warm_user']*100:.1f}% | warm-target {stats['frac_warm_target_poi']*100:.1f}% "
              f"| fully-warm {stats['frac_fully_warm']*100:.1f}% | cold-user {stats['frac_cold_user']*100:.1f}%")
        print(f"        global drift JS T0->T1..T5 = {drift}  | per-user T0->T5 median JS = {stats['per_user_drift_T0_T5_median']}")
    json.dump(out, open(ROOT / "regime_stats.json", "w"), indent=2)
    print("-> regime_stats.json")


if __name__ == "__main__":
    main()
