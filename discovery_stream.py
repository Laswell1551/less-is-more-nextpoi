#!/usr/bin/env python3
"""
discovery_stream.py -- the return/discovery decomposition for the models we train ourselves,
on all four datasets, under the streaming protocol.

WHY THIS EXISTS
The decisive decomposition (discovery_analysis.py) uses the official GETNext implementation
on its own evaluation instances, and it is solid on NYC. It is not available on TKY: the
released code does not learn on our TKY split (the loss stays near ln|P| and then diverges to
NaN), and we could not determine why. It trains normally on the authors' own TKY data, so we
do not attribute the failure to their code, and we do not report a GETNext TKY number.

But the claim under test -- "no method does discovery" -- does not depend on that one
implementation. Here we ask the same question of the models we train ourselves, which do
train on all four datasets: a frozen neural backbone, the strongest published continual method
(GIRAM), the counter, and a popularity floor. If every learned model is also at or below
popularity on the discovery slice, the claim holds independently of GETNext.

DEFINITION (identical to discovery_analysis.py)
  return    = the target POI is one this user has visited before (at any earlier point)
  discovery = the target POI is one this user has NEVER visited

Usage:  python discovery_stream.py --datasets nyc tky gowalla_ca brightkite
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ranking as RK
import pandas as pd
import torch

import learned_continual as L
from run_chrono import chrono_test_samples
from count_baseline import CountModel

ROOT = Path(__file__).resolve().parent
DISP = {"nyc": "Foursquare-NYC", "tky": "Foursquare-TKY",
        "gowalla_ca": "Gowalla-CA", "brightkite": "Brightkite-US"}


@torch.no_grad()
def ranks_of(score_fn, S, idx, dev, bs=512):
    """Rank of the true target under score_fn, for the samples in `idx`."""
    out = []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i:i + bs], dtype=torch.long)
        seq = S["seq"][j].to(dev); u = S["user"][j].to(dev)
        h = S["hour"][j].to(dev); d = S["dow"][j].to(dev)
        t = S["tgt"][j].to(dev)
        sc = score_fn(seq, u, h, d)
        # expected rank under random tie-breaking (see ranking.py). The old
        # (sc > s*).sum()+1 placed the target first among its ties, which flatters the
        # counter (it ties constantly) and not the neural model (it never does).
        r = RK.expected_rank_torch(sc, t)
        out.append(r.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["nyc", "tky", "gowalla_ca", "brightkite"])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--epochs0", type=int, default=8)
    args = ap.parse_args()

    dev = L.DEVICE
    allout = {}
    for name in args.datasets:
        df = L.load(name)
        n_users = int(df.user_idx.max() + 1); n_pois = int(df.poi_idx.max() + 1)
        S0 = L.make_samples(df[df.block == "T0"], n_pois)
        S_test, _ = chrono_test_samples(df[df.block != "T0"], n_pois)
        rounds = np.array_split(np.arange(len(S_test["tgt"])), args.rounds)
        margs = (n_users, n_pois, 128, "gru")

        # --- label every evaluation instance: return or discovery ----------------------
        # "seen" accumulates causally: T0 first, then each round BEFORE it is scored.
        seen = [set() for _ in range(n_users)]
        for u, t in zip(S0["user"].numpy(), S0["tgt"].numpy()):
            seen[u].add(int(t))
        is_ret = np.zeros(len(S_test["tgt"]), dtype=bool)
        u_all = S_test["user"].numpy(); t_all = S_test["tgt"].numpy()
        for idx in rounds:
            for i in idx:
                is_ret[i] = int(t_all[i]) in seen[u_all[i]]
            for i in idx:
                seen[u_all[i]].add(int(t_all[i]))

        # --- the models -----------------------------------------------------------------
        torch.manual_seed(0); np.random.seed(0)
        base = L.NextPOI(*margs).to(dev)
        L.fit(base, S0, epochs=args.epochs0)
        base.eval()

        # popularity floor (T0 counts, frozen -- same information as the static model)
        pop = torch.zeros(n_pois, device=dev, dtype=torch.float64)
        pop.index_put_((S0["tgt"].to(dev),),
                       torch.ones(len(S0["tgt"]), device=dev, dtype=torch.float64),
                       accumulate=True)
        pop_n = (pop / (pop.max() + 1e-9)).unsqueeze(0)

        # the counter, replayed over the same rounds (predict-then-update)
        cm = CountModel(n_users, n_pois, dev)
        cm.observe(S0["user"].to(dev), S0["tgt"].to(dev))

        R = {"static (neural)": [], "popularity": [], "COUNT": []}
        order = []
        for idx in rounds:
            order.append(idx)
            R["static (neural)"].append(ranks_of(lambda s, u, h, d: base(s, u, h, d),
                                                 S_test, idx, dev))
            R["popularity"].append(ranks_of(lambda s, u, h, d: pop_n.expand(len(s), -1),
                                            S_test, idx, dev))
            R["COUNT"].append(ranks_of(lambda s, u, h, d: cm(s, u, h, d), S_test, idx, dev))
            cm.observe(S_test["user"][torch.as_tensor(idx)].to(dev),
                       S_test["tgt"][torch.as_tensor(idx)].to(dev), decay=0.9)

        order = np.concatenate(order)
        ret = is_ret[order]
        res = {}
        for k, chunks in R.items():
            r = np.concatenate(chunks)
            res[k] = {
                "all": round(float((r <= 10).mean()), 4),
                "return": round(float((r[ret] <= 10).mean()), 4),
                "discovery": round(float((r[~ret] <= 10).mean()), 4),
            }
        allout[name] = {"n": int(len(order)), "return_share": round(float(ret.mean()), 4),
                        "acc@10": res}

        print(f"\n=== {DISP[name]} ===")
        print(f"  {ret.mean():.1%} returns / {1-ret.mean():.1%} discoveries "
              f"({len(order):,} instances)")
        print(f"  {'method':18s} {'Acc@10 all':>11s} {'on RETURN':>11s} {'on DISCOVERY':>13s}")
        for k in ("popularity", "static (neural)", "COUNT"):
            v = res[k]
            print(f"  {k:18s} {v['all']:11.4f} {v['return']:11.4f} {v['discovery']:13.4f}")

    (ROOT / "discovery_stream.json").write_text(json.dumps(allout, indent=2))
    print("\n-> discovery_stream.json")
    print("\nIf the neural model is at or below the popularity floor in the discovery column")
    print("on every dataset, then 'no method does discovery' does not rest on GETNext alone.")


if __name__ == "__main__":
    main()
