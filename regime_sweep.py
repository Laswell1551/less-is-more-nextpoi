#!/usr/bin/env python3
"""Adversarial protocol (referee #3, #11): manufacture conditions where fine-tuning
SHOULD win, by shrinking the base block. A smaller base means the stream carries more
genuinely new POIs the frozen model never saw -> fine-tuning (which can learn them)
should overtake static. We sweep base_frac in {0.10,0.20,0.35,0.50} on all four
datasets and record, per split: the cheap a-priori observable new-POI rate, and the
streaming Acc@10 of static / tuned-FT(lr=1e-4) / memory. Output feeds the a-priori
criterion (does new-POI rate predict the FT-static gap without running FT?).
"""
import json
from pathlib import Path

import numpy as np
import torch

import learned_continual as L

DS = ["nyc", "tky", "gowalla_ca", "brightkite"]
BFS = [0.10, 0.20, 0.35, 0.50]


def split_at(df, bf):
    ts = df.groupby("traj_id")["unix_s"].transform("min").to_numpy()
    q = np.quantile(ts, bf); m = ts <= q
    return df[m], df[~m]


def main():
    out = {}
    for ds in DS:
        df = L.load(ds); npo = int(df.poi_idx.max() + 1); nu = int(df.user_idx.max() + 1)
        for bf in BFS:
            t0, stream = split_at(df, bf)
            S0 = L.make_samples(t0, npo); St = L.make_samples(stream, npo)
            if S0 is None or St is None or len(St["tgt"]) < 1000:
                continue
            warm = t0.user_idx.unique(); t0p = set(int(x) for x in t0.poi_idx.unique())
            new_poi = float(np.mean([int(int(t) not in t0p) for t in St["tgt"].numpy()]))
            margs = (nu, npo, 128, "gru")
            torch.manual_seed(0); np.random.seed(0)
            b = L.NextPOI(*margs).to(L.DEVICE); L.fit(b, S0, epochs=8)
            bs = {k: v.detach().cpu().clone() for k, v in b.state_dict().items()}
            rng = np.random.default_rng(0)
            pj = rng.choice(len(St["tgt"]), min(512, len(St["tgt"])), replace=False)
            fS = L.subset(S0, rng.choice(len(S0["tgt"]), min(2000, len(S0["tgt"])), replace=False))
            rounds = np.array_split(np.arange(len(St["tgt"])), 15)
            com = (bs, St, rounds, pj, fS, margs)
            st = L.run_policy("s", *com, (lambda r, a, e: False), 30, S0, 1024,
                              np.random.default_rng(123), warm)["acc@10"]
            ft = L.run_policy("f", *com, (lambda r, a, e: True), 30, S0, 1024,
                              np.random.default_rng(123), warm, ft_lr=1e-4)["acc@10"]
            mem = L.run_giram("m", *com, warm, 0.5)["acc@10"]
            out[f"{ds}@{bf:.2f}"] = {"dataset": ds, "base_frac": bf, "new_poi": round(new_poi, 3),
                                     "static": round(st, 4), "FT@1e-4": round(ft, 4),
                                     "memory": round(mem, 4), "FT_minus_static": round(ft - st, 4)}
            print(f"{ds} bf={bf:.2f}  new-POI={new_poi:.3f}  static={st:.4f}  "
                  f"FT={ft:.4f}  mem={mem:.4f}  FT-static={ft - st:+.4f}")
    json.dump(out, open(L.ROOT / "regime_sweep.json", "w"), indent=2)
    print("-> regime_sweep.json")


if __name__ == "__main__":
    main()
