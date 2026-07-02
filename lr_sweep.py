#!/usr/bin/env python3
"""Learning-rate sweep for the fine-tuning baselines (referee #1, #2). Records BOTH
Acc@10 and T0-probe forgetting at each LR, over 3 seeds, on all four datasets. The
accuracy curve shows the 'FT loses' result is an LR artifact; the forgetting curve
distinguishes a real sweet spot (forgetting -> 0 AND accuracy > static) from the
LR->0 tautology (nothing learned). Output feeds fig_lr (2 rows: accuracy, forgetting).
"""
import json
from pathlib import Path

import numpy as np
import torch

import learned_continual as L

DS = ["nyc", "tky", "gowalla_ca", "brightkite"]
LRS = [1e-3, 3e-4, 1e-4, 3e-5]
SEEDS = [0, 1, 2]
KS = [f"{lr:.0e}" for lr in LRS]


def setup(ds, seed):
    df = L.load(ds); nu = int(df.user_idx.max() + 1); npo = int(df.poi_idx.max() + 1)
    S0 = L.make_samples(df[df.block == "T0"], npo)
    St = L.make_samples(df[df.block != "T0"], npo)
    warm = df[df.block == "T0"].user_idx.unique(); margs = (nu, npo, 128, "gru")
    torch.manual_seed(seed); np.random.seed(seed)
    b = L.NextPOI(*margs).to(L.DEVICE); L.fit(b, S0, epochs=8)
    bs = {k: v.detach().cpu().clone() for k, v in b.state_dict().items()}
    rng = np.random.default_rng(seed); pj = rng.choice(len(St["tgt"]), 512, replace=False)
    fS = L.subset(S0, rng.choice(len(S0["tgt"]), min(2000, len(S0["tgt"])), replace=False))
    rounds = np.array_split(np.arange(len(St["tgt"])), 15)
    return (bs, St, rounds, pj, fS, margs), warm, S0


def main():
    out = {}
    for ds in DS:
        A = {m: {k: [] for k in KS} for m in ["always", "EWC", "ADER"]}
        Fg = {m: {k: [] for k in KS} for m in ["always", "EWC", "ADER"]}
        st = []
        for seed in SEEDS:
            com, warm, S0 = setup(ds, seed)
            st.append(L.run_policy("s", *com, (lambda r, a, e: False), 30, S0, 1024,
                                   np.random.default_rng(123), warm)["acc@10"])
            for lr in LRS:
                k = f"{lr:.0e}"
                r = L.run_policy("a", *com, (lambda r, a, e: True), 30, S0, 1024,
                                 np.random.default_rng(123), warm, ft_lr=lr)
                A["always"][k].append(r["acc@10"]); Fg["always"][k].append(r["forget_drop"])
                r = L.run_ewc("e", *com, S0, 1e3, 30, warm, ft_lr=lr)
                A["EWC"][k].append(r["acc@10"]); Fg["EWC"][k].append(r["forget_drop"])
                r = L.run_ader("d", *com, S0, 1024, 1.0, 30, warm, np.random.default_rng(123), ft_lr=lr)
                A["ADER"][k].append(r["acc@10"]); Fg["ADER"][k].append(r["forget_drop"])
        out[ds] = {"static": round(float(np.mean(st)), 4),
                   "acc": {m: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for m, d in A.items()},
                   "forget": {m: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for m, d in Fg.items()}}
        best = max(out[ds]["acc"][m][k] for m in A for k in KS)
        print(f"{ds}: static={out[ds]['static']:.4f}  best-FT={best:.4f}  "
              f"min-forget@best-lr (always 3e-5)={out[ds]['forget']['always']['3e-05']:+.3f}")
    json.dump(out, open(L.ROOT / "lr_sweep.json", "w"), indent=2)
    print("-> lr_sweep.json")


if __name__ == "__main__":
    main()
