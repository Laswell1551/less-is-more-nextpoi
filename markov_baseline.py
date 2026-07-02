#!/usr/bin/env python3
"""Non-neural strong baseline (referee cleanup): a personalized first-order Markov
recommender backing off to global transitions then popularity, evaluated under the
SAME streaming predict-then-update protocol (15 rounds) on the deduped data. The
reference pi of Obs. 1 was never scored in the main table; this puts it there.
"""
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import learned_continual as L

KS = [1, 5, 10, 20]


def init(S, ut, gt, pop, npo):
    seq = S["seq"].numpy(); u = S["user"].numpy(); t = S["tgt"].numpy()
    for i in range(len(t)):
        p = int(seq[i, -1]); tg = int(t[i])
        if p < npo:
            ut[(int(u[i]), p)][tg] += 1; gt[p][tg] += 1
        pop[tg] += 1


def eval_ds(ds):
    df = L.load(ds); npo = int(df.poi_idx.max() + 1)
    S0 = L.make_samples(df[df.block == "T0"], npo)
    St = L.make_samples(df[df.block != "T0"], npo)
    ut = defaultdict(Counter); gt = defaultdict(Counter); pop = Counter()
    init(S0, ut, gt, pop, npo)
    seq = St["seq"].numpy(); u = St["user"].numpy(); t = St["tgt"].numpy()
    rounds = np.array_split(np.arange(len(St["tgt"])), 15)
    hits = {k: 0 for k in KS}; rr = 0.0; N = 0
    for idx in rounds:
        toppop = [x for x, _ in pop.most_common(20)]
        for i in idx:                                   # predict (hard backoff)
            p = int(seq[i, -1]); uu = int(u[i]); tg = int(t[i])
            d = ut.get((uu, p)) or gt.get(p) or pop
            ranked = [x for x, _ in d.most_common(20)]
            ranked += [x for x in toppop if x not in ranked]
            rank = ranked.index(tg) + 1 if tg in ranked[:20] else 99
            for k in KS:
                hits[k] += int(rank <= k)
            rr += 1.0 / rank if rank <= 20 else 0.0
            N += 1
        for i in idx:                                   # update
            p = int(seq[i, -1]); uu = int(u[i]); tg = int(t[i])
            if p < npo:
                ut[(uu, p)][tg] += 1; gt[p][tg] += 1
            pop[tg] += 1
    res = {f"acc@{k}": round(hits[k] / N, 4) for k in KS}; res["mrr"] = round(rr / N, 4)
    return res


def main():
    DS = sys.argv[1:] or ["nyc", "tky", "gowalla_ca", "brightkite"]
    out = {}
    for ds in DS:
        out[ds] = eval_ds(ds)
        print(ds, out[ds])
    if Path(L.ROOT / "markov_baseline.json").exists():
        out = {**json.load(open(L.ROOT / "markov_baseline.json")), **out}
    json.dump(out, open(L.ROOT / "markov_baseline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
