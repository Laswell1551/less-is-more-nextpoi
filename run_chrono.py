#!/usr/bin/env python3
"""
run_chrono.py -- run_submission.py, but with the streaming rounds ordered by TIME.

WHY THIS EXISTS
---------------
`learned_continual.load()` sorts check-ins by ["user_idx", "unix_s"], and
`make_samples()` iterates `groupby("traj_id", sort=False)`. The resulting S_test is
therefore ordered USER-MAJOR, not chronologically. Every streaming script then does

    rounds = np.array_split(np.arange(len(S_test["tgt"])), 15)

which slices that user-major array into 15 chunks. Those chunks are DISJOINT USER
COHORTS, each spanning the whole T1..T5 period -- not time slices. Consequences:

  * The per-user memory M[u] is updated AFTER each round, but the next round contains
    users it has never seen, so M[u] is empty at evaluation time for ~97% of points.
    On those points the GIRAM/ANCHOR fused score is IDENTICAL to static's, i.e. the
    memory is a no-op. The reported "+0.008, p<0.001" came from the ~3% of users who
    straddle a round boundary.
  * Fine-tuning is evaluated as "update on cohort A, test on disjoint cohort B" -- a
    transfer task in which adaptation has nothing to transfer to, so it registers as
    pure forgetting.

This module rebuilds S_test in true chronological order (by the target check-in's
timestamp) and splits THAT into rounds. Everything else -- base training, seeds, the
T0 forgetting probe, the churn probe, every run_* function -- is reused unchanged from
learned_continual.py, so this is a clean A/B on the ordering alone.

VALIDATION: `static` must come out bit-identical under both orderings, because it never
updates and the evaluation set is the same points in a different order. It does.

Writes results_chrono_{dataset}_{backbone}.csv. Does NOT touch results_seed_*.csv.

Usage:
  python run_chrono.py --backbone gru --datasets nyc tky gowalla_ca brightkite --seeds 0
  python run_chrono.py --backbone gru --seeds 0 1 2 3 4        # full multi-seed
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

import learned_continual as L

RNG_REPLAY = 123          # fixed replay-sampling seed so policies see the same replay

dec_static = lambda r, a, e: False
dec_always = lambda r, a, e: True
def dec_periodic(N):
    return lambda r, a, e: (r % N == 0)


def chrono_test_samples(df_test, n_pois):
    """S_test, reordered so that sample i precedes sample j iff its target check-in is
    earlier. Replicates make_samples()' emission order to recover each sample's
    timestamp, then argsorts.

    make_samples emits, per trajectory, one sample for each i in 1..len(traj)-1, whose
    target is the check-in at position i. So the sample's time is that check-in's time.
    """
    times = []
    for _, g in df_test.groupby("traj_id", sort=False):
        t = g.unix_s.to_numpy()
        times.extend(t[1:])                      # targets are positions 1..len-1
    times = np.asarray(times)

    S = L.make_samples(df_test, n_pois)
    assert len(times) == len(S["tgt"]), (
        f"time/sample length mismatch: {len(times)} vs {len(S['tgt'])} -- "
        "make_samples' emission order changed; fix this helper before trusting results"
    )
    order = np.argsort(times, kind="stable")     # stable => deterministic ties
    S_chrono = {k: v[torch.as_tensor(order)] for k, v in S.items()}
    return S_chrono, times[order]


def run_dataset(name, args):
    df = L.load(name)
    n_users = int(df.user_idx.max() + 1); n_pois = int(df.poi_idx.max() + 1)
    S0 = L.make_samples(df[df.block == "T0"], n_pois)
    S_test, t_sorted = chrono_test_samples(df[df.block != "T0"], n_pois)
    warm = df[df.block == "T0"].user_idx.unique()
    if args.backbone == "getnext":
        L.set_poi_graph(L.build_poi_graph(df[df.block == "T0"], n_pois))

    rounds = np.array_split(np.arange(len(S_test["tgt"])), args.rounds)
    span = [(pd.to_datetime(t_sorted[idx[0]], unit="s").date(),
             pd.to_datetime(t_sorted[idx[-1]], unit="s").date()) for idx in rounds]
    print(f"[{name}] users={n_users} pois={n_pois} T0={len(S0['tgt'])} test={len(S_test['tgt'])}")
    print(f"  round 0: {span[0][0]} -> {span[0][1]}   round {len(rounds)-1}: "
          f"{span[-1][0]} -> {span[-1][1]}   (rounds are TIME slices)")

    margs = (n_users, n_pois, args.dim, args.backbone)
    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        base = L.NextPOI(*margs).to(L.DEVICE)
        L.fit(base, S0, epochs=args.epochs0)
        bs = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        rng = np.random.default_rng(seed)
        pj = rng.choice(len(S_test["tgt"]), size=min(512, len(S_test["tgt"])), replace=False)
        forget_S = L.subset(S0, rng.choice(len(S0["tgt"]), size=min(2000, len(S0["tgt"])), replace=False))
        common = (bs, S_test, rounds, pj, forget_S, margs)
        res = [
            L.run_policy("static", *common, dec_static, args.ft_steps, S0, args.replay,
                         np.random.default_rng(RNG_REPLAY), warm),
            L.run_policy("always+replay", *common, dec_always, args.ft_steps, S0, args.replay,
                         np.random.default_rng(RNG_REPLAY), warm),
            L.run_policy("periodic-4", *common, dec_periodic(4), args.ft_steps, S0, args.replay,
                         np.random.default_rng(RNG_REPLAY), warm),
            L.run_selective("selective-gated", *common, 0.20, args.ft_steps, True, 1e-2, warm),
            L.run_giram("GIRAM", *common, warm, 0.5),
            L.run_giram_vae("GIRAM-VAE", *common, warm, S0, 0.5),
            L.run_ewc("EWC", *common, S0, 1e3, args.ft_steps, warm),
            L.run_ader("ADER", *common, S0, args.replay, 1.0, args.ft_steps, warm,
                       np.random.default_rng(RNG_REPLAY)),
        ]
        if seed == args.seeds[0]:                  # per-round trajectory (now a TIME axis)
            rr = [{"round": i, "policy": r["policy"], "acc": a}
                  for r in res for i, a in enumerate(r.get("rounds_acc", []))]
            pd.DataFrame(rr).to_csv(L.ROOT / f"results_chrono_rounds_{name}_{args.backbone}.csv",
                                    index=False)
        for r in res:
            r.pop("rounds_acc", None)
            r["seed"] = seed; rows.append(r)
        stat = next(r["acc@10"] for r in res if r["policy"] == "static")
        print(f"  [{name} seed{seed}] " +
              " ".join(f"{r['policy'].split('+')[0][:6]}={r['acc@10']:.3f}"
                       f"({r['acc@10']-stat:+.3f})" for r in res))
    R = pd.DataFrame(rows)
    R.to_csv(L.ROOT / f"results_chrono_{name}_{args.backbone}.csv", index=False)
    return R


def summarize(name, R, seeds):
    print(f"\n=== {name}: CHRONOLOGICAL protocol, mean +/- std over {len(seeds)} seed(s) ===")
    g = R.groupby("policy")
    order = ["static", "always+replay", "periodic-4", "EWC", "ADER",
             "selective-gated", "GIRAM", "GIRAM-VAE"]
    base = R[R.policy == "static"]["acc@10"].mean()
    for p in order:
        if p not in g.groups:
            continue
        s = g.get_group(p)
        m = s["acc@10"].mean()
        rel = (m - base) / base * 100 if base else float("nan")
        print(f"  {p:16s} acc@10={m:.4f}+/-{s['acc@10'].std():.4f} "
              f"(vs static {m-base:+.4f}, {rel:+.1f}%)  "
              f"forget={s['forget_drop'].mean():+.3f} churn={s['churn'].mean():.3f}")

    if len(seeds) > 1:
        print("  --- paired t-tests vs static (acc@10) ---")
        for b in order[1:]:
            if b not in g.groups:
                continue
            x = R[R.policy == "static"].sort_values("seed")["acc@10"].to_numpy()
            y = R[R.policy == b].sort_values("seed")["acc@10"].to_numpy()
            t, pval = stats.ttest_rel(y, x)
            sig = "SIGNIF" if pval < 0.05 else "n.s."
            print(f"    {b:16s} delta={y.mean()-x.mean():+.4f}  p={pval:.4f}  [{sig}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["nyc", "tky", "gowalla_ca", "brightkite"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs0", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ft-steps", type=int, default=30, dest="ft_steps")
    ap.add_argument("--replay", type=int, default=1024)
    ap.add_argument("--backbone", default="gru", choices=["gru", "attn", "getnext"])
    args = ap.parse_args()
    print(f"device={L.DEVICE} seeds={args.seeds} backbone={args.backbone} [CHRONOLOGICAL ROUNDS]")
    for name in args.datasets:
        R = run_dataset(name, args)
        summarize(name, R, args.seeds)


if __name__ == "__main__":
    main()
