#!/usr/bin/env python3
"""
run_submission.py -- submission-grade multi-seed + significance for the
"Less is More" diagnosis. Reuses the validated run_* functions from
learned_continual.py; runs N seeds x 3 datasets, reports mean +/- std and
paired t-tests for the key claims:
   static vs GIRAM-core   (is the ~1% gain significant?)
   static vs selective    (does forgetting-free updating match static?)
   static vs always+replay (does naive global FT hurt?)

Run (activate the `poi` env first; see README):
  python run_submission.py --backbone gru --datasets nyc tky gowalla_ca brightkite
"""
import argparse
import numpy as np
import pandas as pd
import torch
from scipy import stats

import learned_continual as L

SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["nyc", "tky", "gowalla_ca"]
RNG_REPLAY = 123          # fixed replay-sampling seed so policies see the same replay

dec_static = lambda r, a, e: False
dec_always = lambda r, a, e: True
def dec_periodic(N):
    return lambda r, a, e: (r % N == 0)


def run_dataset(name, args):
    df = L.load(name)
    n_users = int(df.user_idx.max() + 1); n_pois = int(df.poi_idx.max() + 1)
    S0 = L.make_samples(df[df.block == "T0"], n_pois)
    S_test = L.make_samples(df[df.block != "T0"], n_pois)
    warm = df[df.block == "T0"].user_idx.unique()
    if args.backbone == "getnext":
        L.set_poi_graph(L.build_poi_graph(df[df.block == "T0"], n_pois))
    rounds = np.array_split(np.arange(len(S_test["tgt"])), args.rounds)
    margs = (n_users, n_pois, args.dim, args.backbone)
    rows = []
    for seed in SEEDS:
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
            L.run_giram_plus("GIRAM+", *common, warm, 0.5),
            L.run_giram_vae("GIRAM-VAE", *common, warm, S0, 0.5),
            L.run_ewc("EWC", *common, S0, 1e3, args.ft_steps, warm),
            L.run_ader("ADER", *common, S0, args.replay, 1.0, args.ft_steps, warm,
                       np.random.default_rng(RNG_REPLAY)),
        ]
        if seed == SEEDS[0]:                       # save per-round trajectory once
            rr = [{"round": i, "policy": r["policy"], "acc": a}
                  for r in res for i, a in enumerate(r.get("rounds_acc", []))]
            pd.DataFrame(rr).to_csv(L.ROOT / f"results_rounds_{name}_{args.backbone}.csv", index=False)
        for r in res:
            r.pop("rounds_acc", None)
            r["seed"] = seed; rows.append(r)
        print(f"  [{name} seed{seed}] " +
              " ".join(f"{r['policy'].split('+')[0][:6]}={r['acc@10']:.3f}" for r in res))
    R = pd.DataFrame(rows)
    R.to_csv(L.ROOT / f"results_seed_{name}_{args.backbone}.csv", index=False)
    return R


def summarize(name, R):
    print(f"\n=== {name}: mean +/- std over {len(SEEDS)} seeds ===")
    g = R.groupby("policy")
    order = ["static", "always+replay", "periodic-4", "EWC", "ADER", "selective-gated", "GIRAM", "GIRAM+", "GIRAM-VAE"]
    for p in order:
        s = g.get_group(p)
        print(f"  {p:16s} acc@10={s['acc@10'].mean():.4f}+/-{s['acc@10'].std():.4f} "
              f"forget={s['forget_drop'].mean():+.3f} churn={s['churn'].mean():.3f}")

    def paired(a, b):
        x = R[R.policy == a].sort_values("seed")["acc@10"].to_numpy()
        y = R[R.policy == b].sort_values("seed")["acc@10"].to_numpy()
        t, pval = stats.ttest_rel(y, x)
        return y.mean() - x.mean(), pval
    print("  --- paired t-tests vs static (acc@10) ---")
    for b in ["GIRAM", "selective-gated", "always+replay", "EWC", "ADER"]:
        d, pval = paired("static", b)
        sig = "SIGNIF" if pval < 0.05 else "n.s."
        print(f"    {b:16s} delta={d:+.4f}  p={pval:.4f}  [{sig}]")
    return g["acc@10"].agg(["mean", "std"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--epochs0", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ft-steps", type=int, default=30, dest="ft_steps")
    ap.add_argument("--replay", type=int, default=1024)
    ap.add_argument("--backbone", default="gru", choices=["gru", "attn", "getnext"])
    args = ap.parse_args()
    print(f"device={L.DEVICE} seeds={SEEDS} backbone={args.backbone}")
    for name in args.datasets:
        R = run_dataset(name, args)
        summarize(name, R)


if __name__ == "__main__":
    main()
