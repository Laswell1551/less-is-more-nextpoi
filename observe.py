#!/usr/bin/env python3
"""
observe.py -- three structural observations on real check-in data (paper, Sec. 3).

R1  Low-rank user-POI structure:
        SVD energy of the user x POI interaction matrix; cumulative energy vs rank.
R2  Temporal sparsity of preference drift:
        per-user Jensen-Shannon divergence of the POI-visit distribution between
        consecutive fixed-size windows (most users stable, only a few drift).
R3  Periodicity + spatial concentration:
        ACF of the hourly aggregate check-in volume (peaks at 24h / 168h) and the
        POI popularity rank-size long tail.

Outputs: experiments/figs/obs_R{1,2,3}.png  and  experiments/observations.json
Pure CPU / pandas / scipy. Run after preprocess.py.

Usage: python observe.py            (all datasets)
       python observe.py --datasets nyc tky
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "figs"
DATASETS = ["nyc", "tky", "gowalla_ca"]
COLORS = {"nyc": "#1f77b4", "tky": "#d62728", "gowalla_ca": "#2ca02c"}
WIN = 20            # window size (check-ins) for the drift measure
SVD_K = 200        # number of singular components to probe


def load(name):
    f = PROC / name / "checkins.csv.gz"
    return pd.read_csv(f).sort_values(["user_idx", "unix_s"]).reset_index(drop=True)


# --------------------------------------------------------------- R1 low rank
def r1_lowrank(df):
    u = df.user_idx.to_numpy()
    p = df.poi_idx.to_numpy()
    nU, nP = u.max() + 1, p.max() + 1
    M = csr_matrix((np.ones(len(df)), (u, p)), shape=(nU, nP), dtype=np.float64)
    frob2 = float(M.multiply(M).sum())                       # ||M||_F^2 (exact)
    k = min(SVD_K, min(M.shape) - 1)
    s = svds(M, k=k, return_singular_vectors=False)
    s = np.sort(s)[::-1]
    cum = np.cumsum(s ** 2) / frob2                           # cumulative energy of top-k
    def rank_for(th):
        idx = np.searchsorted(cum, th)
        return int(idx + 1) if idx < len(cum) else None
    return {
        "ranks": np.arange(1, k + 1), "cum_energy": cum,
        "energy_top1": round(float(cum[0]), 4),
        "energy_top10": round(float(cum[min(9, k - 1)]), 4),
        "rank@80": rank_for(0.80), "rank@90": rank_for(0.90),
        "n_users": int(nU), "n_pois": int(nP),
    }


# ------------------------------------------------------ R2 preference drift
def _js(c1, c2):
    """Jensen-Shannon divergence (log2, in [0,1]) between two POI counters."""
    keys = set(c1) | set(c2)
    p = np.array([c1.get(k, 0) for k in keys], float); p /= p.sum()
    q = np.array([c2.get(k, 0) for k in keys], float); q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def r2_drift(df, win=WIN):
    drifts = []
    for _, g in df.groupby("user_idx", sort=False):
        pois = g.poi_idx.to_numpy()
        nw = len(pois) // win
        if nw < 2:
            continue
        windows = [Counter(pois[i * win:(i + 1) * win]) for i in range(nw)]
        for a, b in zip(windows[:-1], windows[1:]):
            drifts.append(_js(a, b))
    d = np.sort(np.asarray(drifts))
    # concentration (Lorenz): share of total drift carried by the most-changed pairs
    csum = np.cumsum(d) / d.sum()
    frac_stable = float((d < 0.10).mean())                   # ~zero-drift fraction
    # share of total drift from the top-10% most-changed window pairs:
    top10_share = float(1 - csum[int(0.90 * len(d)) - 1]) if len(d) > 10 else float("nan")
    return {
        "drift_sorted": d, "lorenz": csum,
        "median_drift": round(float(np.median(d)), 4),
        "frac_drift_lt_0.10": round(frac_stable, 4),
        "top10pct_change_share": round(top10_share, 4),
        "n_pairs": int(len(d)),
    }


# ------------------------------------------------- R3 periodicity + long tail
def r3_period_tail(df):
    hours = ((df.unix_s.to_numpy() - df.unix_s.min()) // 3600).astype(int)
    counts = np.bincount(hours, minlength=hours.max() + 1).astype(float)
    x = counts - counts.mean()
    full = np.correlate(x, x, mode="full")[len(x) - 1:]
    acf = full / full[0]
    maxlag = min(200, len(acf) - 1)
    pop = df.poi_idx.value_counts().to_numpy().astype(float)   # sorted desc
    cov = np.cumsum(pop) / pop.sum()
    top1 = float(cov[max(0, int(0.01 * len(pop)) - 1)])
    top10 = float(cov[max(0, int(0.10 * len(pop)) - 1)])
    return {
        "acf": acf[:maxlag + 1], "acf_24h": round(float(acf[24]), 3),
        "acf_168h": round(float(acf[168]), 3) if len(acf) > 168 else None,
        "pop_sorted": pop, "pop_cov": cov,
        "top1pct_poi_share": round(top1, 4), "top10pct_poi_share": round(top10, 4),
    }


# ----------------------------------------------------------------- plotting
def plot_all(R):
    FIGS.mkdir(parents=True, exist_ok=True)
    # R1
    plt.figure(figsize=(5, 3.6))
    for n in R:
        r = R[n]["R1"]
        plt.plot(r["ranks"], r["cum_energy"], color=COLORS[n], label=f"{n} (r@90%={r['rank@90']})")
    plt.axhline(0.9, ls="--", c="grey", lw=0.8)
    plt.xlabel("rank (top singular components)"); plt.ylabel("cumulative energy")
    plt.title("R1: low-rank user-POI structure"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGS / "obs_R1.png", dpi=160); plt.close()

    # R2: CDF of drift + Lorenz concentration
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for n in R:
        r = R[n]["R2"]; d = r["drift_sorted"]
        ax[0].plot(d, np.linspace(0, 1, len(d)), color=COLORS[n],
                   label=f"{n} (<0.1: {r['frac_drift_lt_0.10']*100:.0f}%)")
        ax[1].plot(np.linspace(0, 1, len(r["lorenz"])), r["lorenz"], color=COLORS[n], label=n)
    ax[0].axvline(0.1, ls="--", c="grey", lw=0.8)
    ax[0].set_xlabel("consecutive-window JS drift"); ax[0].set_ylabel("CDF")
    ax[0].set_title("R2a: preference drift is small"); ax[0].legend(fontsize=8)
    ax[1].plot([0, 1], [0, 1], ls=":", c="k", lw=0.8)
    ax[1].set_xlabel("window pairs (sorted by drift)"); ax[1].set_ylabel("cum. share of total drift")
    ax[1].set_title("R2b: change is concentrated (Lorenz)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "obs_R2.png", dpi=160); plt.close()

    # R3: ACF + rank-size
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for n in R:
        r = R[n]["R3"]
        ax[0].plot(np.arange(len(r["acf"])), r["acf"], color=COLORS[n], lw=0.9,
                   label=f"{n} (24h={r['acf_24h']})")
        rank = np.arange(1, len(r["pop_sorted"]) + 1) / len(r["pop_sorted"])
        ax[1].loglog(rank, r["pop_sorted"], color=COLORS[n], label=n)
    for L in (24, 48, 72, 168):
        ax[0].axvline(L, ls="--", c="grey", lw=0.5)
    ax[0].set_xlabel("lag (hours)"); ax[0].set_ylabel("ACF of check-in volume")
    ax[0].set_title("R3a: diurnal/weekly periodicity"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("POI rank (fraction)"); ax[1].set_ylabel("visit count")
    ax[1].set_title("R3b: popularity long tail"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "obs_R3.png", dpi=160); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    R, head = {}, {}
    for n in args.datasets:
        print(f"[{n}] computing observations ...")
        df = load(n)
        R[n] = {"R1": r1_lowrank(df), "R2": r2_drift(df), "R3": r3_period_tail(df)}
        head[n] = {
            "R1_rank@90": R[n]["R1"]["rank@90"], "R1_energy_top10": R[n]["R1"]["energy_top10"],
            "R2_frac_drift_lt_0.10": R[n]["R2"]["frac_drift_lt_0.10"],
            "R2_top10pct_change_share": R[n]["R2"]["top10pct_change_share"],
            "R3_acf_24h": R[n]["R3"]["acf_24h"], "R3_acf_168h": R[n]["R3"]["acf_168h"],
            "R3_top10pct_poi_share": R[n]["R3"]["top10pct_poi_share"],
        }
        print(f"   R1 rank@90%={R[n]['R1']['rank@90']}/{R[n]['R1']['n_pois']} pois "
              f"(top-10 comp={R[n]['R1']['energy_top10']*100:.1f}%)")
        print(f"   R2 drift<0.1: {R[n]['R2']['frac_drift_lt_0.10']*100:.1f}%  | "
              f"top-10% pairs carry {R[n]['R2']['top10pct_change_share']*100:.1f}% of all drift  "
              f"(median {R[n]['R2']['median_drift']})")
        print(f"   R3 ACF@24h={R[n]['R3']['acf_24h']} ACF@168h={R[n]['R3']['acf_168h']}  | "
              f"top-10% POIs = {R[n]['R3']['top10pct_poi_share']*100:.1f}% of visits")
    plot_all(R)
    json.dump(head, open(ROOT / "observations.json", "w"), indent=2)
    print(f"\nfigures -> {FIGS}\\obs_R1/2/3.png   headlines -> observations.json")


if __name__ == "__main__":
    main()
