#!/usr/bin/env python3
"""
count_baseline.py -- the non-neural counting baseline, measured in the SAME harness as
every neural method (learned_continual's seg_eval / stream_rich / churn probe / T0 probe,
macro-averaged over the same chronological rounds).

This matters: llm_prep.py measures the counter with a micro-average over points, while
learned_continual macro-averages over rounds. Those two numbers are not comparable and
must not sit in the same table. This module removes that objection by making the counter a
drop-in `model` for the existing harness.

The model:
    score(u, p) = M[u][p] + eps * popularity_norm[p]
        M   : per-user visit counts, decayed by gamma once per round
        pop : global visit counts
    - seeded on T0 (the counter's "training" is counting the base period, the exact
      analogue of the static model being trained on T0)
    - updated only AFTER each round is scored (predict-then-update, no leakage)
    - no embeddings, no gradients, no neural network; O(1) per observed check-in

The popularity term is a strict tie-break (eps is tiny), so the ranking is
"the user's own history by recency-weighted frequency, then global popularity".
"""
import argparse

import numpy as np
import pandas as pd
import torch

import learned_continual as L
from run_chrono import chrono_test_samples

EPS = 1e-6


class CountModel:
    """Duck-types the `model(seq, user, hour, dow) -> [B, n_pois]` interface that
    learned_continual's evaluation helpers expect, so the counter is scored by exactly the
    same code path as the GRU/GETNext backbones."""

    def __init__(self, n_users, n_pois, device):
        # float64, NOT torch's default float32. The eps*popularity term is a strict tie-break,
        # and in float32 it is rounded away for any POI with more than ~8 (decayed) visits --
        # exactly the POIs at the top of the ranking. See dtype_control.py and getnext_compare.py.
        self.M = torch.zeros(n_users, n_pois, device=device, dtype=torch.float64)
        self.pop = torch.zeros(n_pois, device=device, dtype=torch.float64)
        self.n_pois = n_pois

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, seq, user, hour, dow):
        pop_n = self.pop / (self.pop.max() + 1e-9)
        return self.M[user] + EPS * pop_n.unsqueeze(0)

    def observe(self, users, tgts, decay=None):
        if decay is not None:
            self.M.mul_(decay)
        ones = torch.ones(len(tgts), device=self.M.device, dtype=self.M.dtype)
        self.M.index_put_((users, tgts), ones, accumulate=True)
        self.pop.index_put_((tgts,), ones, accumulate=True)


def run_count(name, S0, S_test, rounds, pj, forget_S, n_users, n_pois, warm, decay=0.9):
    dev = L.DEVICE
    m = CountModel(n_users, n_pois, dev)
    m.observe(S0["user"].to(dev), S0["tgt"].to(dev))          # seed on T0, no decay

    accs, churn, prev = [], [], None
    HW = NW = HC = NC = 0
    rich = L._rich0()

    f0 = L.seg_eval(m, forget_S, np.array([]))                # T0 probe, before the stream
    f0 = f0[2] / max(f0[3], 1)

    for r, idx in enumerate(rounds):
        Sr = L.subset(S_test, idx)
        hw, nw, hc, nc = L.seg_eval(m, Sr, warm)              # predict ...
        accs.append((hw + hc) / max(nw + nc, 1))
        HW += hw; NW += nw; HC += hc; NC += nc
        L.stream_rich(lambda s, uu, hh, dd: m(s, uu, hh, dd), Sr, rich)

        m.observe(Sr["user"].to(dev), Sr["tgt"].to(dev), decay=decay)   # ... then update

        top = L.probe_topk(m, S_test, pj)
        if prev is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, prev)])))
        prev = top

    fN = L.seg_eval(m, forget_S, np.array([]))
    fN = fN[2] / max(fN[3], 1)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": 0, "forget_drop": round(float(f0 - fN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4),
            "avg_users_upd": "-", **L._rich_fin(rich)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["nyc", "tky", "gowalla_ca", "brightkite"])
    ap.add_argument("--rounds", type=int, default=15)
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        df = L.load(name)
        n_users = int(df.user_idx.max() + 1); n_pois = int(df.poi_idx.max() + 1)
        S0 = L.make_samples(df[df.block == "T0"], n_pois)
        S_test, _ = chrono_test_samples(df[df.block != "T0"], n_pois)
        rounds = np.array_split(np.arange(len(S_test["tgt"])), args.rounds)
        warm = df[df.block == "T0"].user_idx.unique()

        rng = np.random.default_rng(0)
        pj = rng.choice(len(S_test["tgt"]), size=min(512, len(S_test["tgt"])), replace=False)
        forget_S = L.subset(S0, rng.choice(len(S0["tgt"]), size=min(2000, len(S0["tgt"])),
                                           replace=False))
        r = run_count("count (non-neural)", S0, S_test, rounds, pj, forget_S,
                      n_users, n_pois, warm)
        r["dataset"] = name
        r.pop("rounds_acc", None)
        rows.append(r)
        print(f"[{name:11s}] acc@10={r['acc@10']:.4f}  acc@1={r['acc@1']:.4f}  "
              f"acc@5={r['acc@5']:.4f}  mrr={r['mrr']:.4f}  "
              f"warm={r['acc_warm']:.4f} cold={r['acc_cold']:.4f}  "
              f"forget={r['forget_drop']:+.3f}  churn={r['churn']:.3f}")
    pd.DataFrame(rows).to_csv(L.ROOT / "results_chrono_count.csv", index=False)
    print("\n-> results_chrono_count.csv")


if __name__ == "__main__":
    main()
