#!/usr/bin/env python3
"""
clean_split_control.py -- does the boundary overlap change any conclusion?

THE PROBLEM (ours, self-reported)
Blocks are assigned by a trajectory's START time, so a trajectory beginning just before the
cut carries its later check-ins past it. check_leakage.py measures the consequence: 4-7% of
evaluation check-ins occur earlier than the last training check-in. Small, but we cite
Bellogin et al. on leaking splits, so we do not get to wave it away.

THE CONTROL
Assign each SAMPLE to a block by the timestamp of its TARGET, not by its trajectory's start.
Then no training target is later than any evaluation target: the split is strictly clean.

Note what this does NOT do: it does not truncate the input sequences. A sample whose target
falls after the cut may still have earlier check-ins in its context window, and it should --
that is the user's genuine history, and a deployed model would have it. What we remove is the
only thing that was actually wrong: training on targets that postdate evaluation targets.

If static / GIRAM / COUNT keep their ordering and their margins, the boundary effect is
immaterial and we say so with numbers. If they do not, the protocol changes.

Usage:  python clean_split_control.py --datasets nyc tky
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import learned_continual as L
from count_baseline import run_count

ROOT = Path(__file__).resolve().parent


def samples_by_target_time(df, n_pois):
    """Build samples over WHOLE trajectories, and return each sample's target timestamp.

    learned_continual.make_samples drops the timestamps, and rebuilding trajectories from a
    pre-filtered block would truncate a straddling trajectory's context. So we replicate its
    emission order over the full frame and carry the target times along.
    """
    PAD = n_pois
    users, seqs, tgts, hours, dows, times = [], [], [], [], [], []
    for _, g in df.groupby("traj_id", sort=False):
        p = g.poi_idx.to_numpy(); u = g.user_idx.to_numpy()
        h = g.hour.to_numpy(); d = g.dow.to_numpy(); t = g.unix_s.to_numpy()
        for i in range(1, len(p)):
            s = p[max(0, i - L.L):i]
            if len(s) < L.L:
                s = np.concatenate([np.full(L.L - len(s), PAD), s])
            users.append(u[i]); seqs.append(s); tgts.append(p[i])
            hours.append(h[i]); dows.append(d[i]); times.append(t[i])
    S = {"user": torch.tensor(users), "seq": torch.tensor(np.stack(seqs)),
         "tgt": torch.tensor(tgts), "hour": torch.tensor(hours),
         "dow": torch.tensor(dows)}
    return S, np.asarray(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nyc", "tky"])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--epochs0", type=int, default=8)
    args = ap.parse_args()

    # This script used to only PRINT, and clean_split_control.json was then hand-copied from the
    # log. That is how discovery_summary.json went stale without anyone noticing. Artifacts are
    # written by the code that computes them, or they are not artifacts.
    RESULTS = {}

    for ds in args.datasets:
        name = ds
        df = L.load(name)
        n_users = int(df.user_idx.max() + 1); n_pois = int(df.poi_idx.max() + 1)
        S_all, t_all = samples_by_target_time(df, n_pois)

        # the cut: the same 50% point of the timeline the original blocks used
        cut = df.unix_s.min() + 0.5 * (df.unix_s.max() - df.unix_s.min())
        is_train = t_all < cut
        idx_tr = np.flatnonzero(is_train)
        idx_te = np.flatnonzero(~is_train)
        order_te = idx_te[np.argsort(t_all[idx_te], kind="stable")]   # chronological stream

        S0 = L.subset(S_all, idx_tr)
        S_test = L.subset(S_all, order_te)
        warm = np.unique(S0["user"].numpy())

        assert t_all[idx_tr].max() <= t_all[order_te].min(), "split is not clean"
        print(f"\n=== {name}: STRICTLY CLEAN split (blocks by target time) ===")
        print(f"  train targets {len(idx_tr):,}  (last  {pd.to_datetime(t_all[idx_tr].max(), unit='s')})")
        print(f"  eval  targets {len(order_te):,}  (first {pd.to_datetime(t_all[order_te].min(), unit='s')})")

        rounds = np.array_split(np.arange(len(S_test["tgt"])), args.rounds)
        rng = np.random.default_rng(0)
        pj = rng.choice(len(S_test["tgt"]), size=min(512, len(S_test["tgt"])), replace=False)
        forget_S = L.subset(S0, rng.choice(len(S0["tgt"]),
                                           size=min(2000, len(S0["tgt"])), replace=False))
        margs = (n_users, n_pois, 128, "gru")

        torch.manual_seed(0); np.random.seed(0)
        base = L.NextPOI(*margs).to(L.DEVICE)
        L.fit(base, S0, epochs=args.epochs0)
        bs = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        common = (bs, S_test, rounds, pj, forget_S, margs)

        res = {
            "static": L.run_policy("static", *common, lambda r, a, e: False, 30, S0, 1024,
                                   np.random.default_rng(123), warm),
            "always+replay": L.run_policy("always", *common, lambda r, a, e: True, 30, S0,
                                          1024, np.random.default_rng(123), warm),
            "GIRAM": L.run_giram("GIRAM", *common, warm, 0.5),
        }
        cnt = run_count("count", S0, S_test, rounds, pj, forget_S, n_users, n_pois, warm)

        # acc@10m == per-instance mean, expected-rank ties (ranking.py) -- the ONE estimator the
        # paper uses everywhere. NOT acc@10, which is a per-round macro average with torch.topk
        # index-order tie-breaking. Neural rows are identical under both (their float scores do
        # not tie); only the counter moves. See the note above stream_table() in gen_tables.py.
        ACC = "acc@10m"
        print(f"\n  {'method':16s} {'Acc@10':>8s}   vs static")
        st = res["static"][ACC]
        for k in ("static", "always+replay", "GIRAM"):
            v = res[k][ACC]
            print(f"  {k:16s} {v:8.4f}   {100*(v-st)/st:+7.1f}%")
        print(f"  {'COUNT':16s} {cnt[ACC]:8.4f}   {100*(cnt[ACC]-st)/st:+7.1f}%")

        out = {k: round(float(res[k][ACC]), 4) for k in ("static", "always+replay", "GIRAM")}
        out["COUNT"] = round(float(cnt[ACC]), 4)
        out["n_train"] = int(len(idx_tr))
        out["n_eval"] = int(len(idx_te))
        RESULTS[ds] = out
        print("\n  Compare with the trajectory-start split (Table 5 of the paper) to see")
        print("  whether the boundary overlap moved anything.")

    blob = {"note": "Strictly-clean split (evaluation blocks assigned by TARGET timestamp, so no "
                    "evaluation target precedes any training target). Acc@10 = per-instance mean "
                    "with expected-rank ties (acc@10m), the same estimator as every other table. "
                    "Counter memory in float64. WRITTEN BY THIS SCRIPT -- do not hand-edit.",
            "datasets": RESULTS}
    (ROOT / "clean_split_control.json").write_text(json.dumps(blob, indent=2))
    print("\n-> clean_split_control.json")


if __name__ == "__main__":
    main()
