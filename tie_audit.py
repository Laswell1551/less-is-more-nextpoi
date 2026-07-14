#!/usr/bin/env python3
"""
tie_audit.py -- how much of the counter's margin is a tie-breaking artifact?

THE WORRY
The counter's rank is computed as (s > s[target]).sum() + 1. That is the OPTIMISTIC convention:
when several POIs share the target's score, the target is placed FIRST among them.

The counter's scores tie constantly. Its score is

    s = <integer visit count>  +  1e-6 * <popularity, normalised>

and `popularity` is itself an integer count, so two POIs with the same visit count AND the same
global popularity have EXACTLY equal scores. GETNext's scores are floats from a neural network
and essentially never tie.

So the optimistic convention gives the counter a benefit that it denies GETNext. If the counter's
margin is an artifact of that asymmetry, the paper is wrong and we need to know now.

WHAT THIS MEASURES
The counter's metrics under all three tie conventions:
    optimistic  rank = (s >  s*).sum() + 1                 (best rank in the tie group)
    pessimistic rank = (s >= s*).sum()                     (worst rank in the tie group)
    expected    rank = (s >  s*).sum() + (n_ties + 1) / 2  (the defensible one)
plus the share of instances where the target is tied at all, and the size of its tie group.

The paper must report the EXPECTED convention, or the pessimistic one, but never the optimistic
one against an opponent that cannot tie.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GN = ROOT.parent / "baselines" / "GETNext" / "dataset" / "OURS-NYC"
EPS = 1e-6


def build():
    """GETNext's exact Protocol-A instances -- by CALLING getnext_compare, not re-deriving them.

    This function used to rebuild the instance set itself, and it got it wrong: it kept cold
    users, kept trajectories GETNext drops as too short, and never filtered unseen POIs out of
    the sequence. It produced 11,328 instances where GETNext scores 5,550 -- a strictly harder
    set -- and then compared the counter's score on THOSE against GETNext's score on ITS 5,550.
    Two methods, two different denominators, one table. That is the precise error this paper
    documents in other people's code, committed in the script whose whole job was to police it.

    There is exactly one definition of the Protocol-A instance set, and it lives in
    getnext_compare.build_instances(). Everything else imports it.
    """
    import getnext_compare as GC

    inst_raw, train, poi_list, _, _, _ = GC.build_instances("OURS-NYC")
    poi_ix = {p: i for i, p in enumerate(poi_list)}
    users = sorted(set(train.user_id.astype(int)) | {i["user"] for i in inst_raw})
    u_ix = {u: i for i, u in enumerate(users)}

    M0 = np.zeros((len(users), len(poi_list)), dtype=np.float64)
    pop0 = np.zeros(len(poi_list), dtype=np.float64)
    for u, p in zip(train.user_id.astype(int), train.POI_id):
        if p in poi_ix:
            M0[u_ix[u], poi_ix[p]] += 1.0
            pop0[poi_ix[p]] += 1.0

    inst = [{"u": u_ix[e["user"]], "tgt": poi_ix[e["target"]],
             "prefix": [poi_ix[p] for p in e["prefix"] if p in poi_ix]}
            for e in inst_raw]
    return M0, pop0, inst, len(poi_list)


def main():
    M0, pop0, inst, n_pois = build()
    pn0 = pop0 / (pop0.max() + 1e-9)
    print(f"{len(inst):,} instances over {n_pois:,} POIs "
          f"-- getnext_compare.py's own set (expect 5,550 / 4,561)\n")

    rows = {}
    for tag in ("count-static", "count-traj"):
        opt, pes, exp = [], [], []
        n_tied, tie_sizes = 0, []
        for e in inst:
            s = M0[e["u"]] + EPS * pn0
            if tag == "count-traj" and e["prefix"]:
                s = s.copy()
                for p in e["prefix"]:
                    s[p] += 1.0
            st = s[e["tgt"]]
            gt = int((s > st).sum())            # strictly better
            eq = int((s == st).sum())           # tied WITH the target (includes the target)
            opt.append(gt + 1)
            pes.append(gt + eq)
            exp.append(gt + (eq + 1) / 2)
            if eq > 1:
                n_tied += 1
                tie_sizes.append(eq)

        n = len(inst)
        def met(r):
            r = np.asarray(r, dtype=np.float64)
            return {f"acc@{k}": round(float((r <= k).mean()), 4) for k in (1, 5, 10, 20)} | \
                   {"mrr": round(float((1.0 / r).mean()), 4)}

        rows[tag] = {"optimistic": met(opt), "pessimistic": met(pes), "expected": met(exp),
                     "pct_target_tied": round(100 * n_tied / n, 1),
                     "median_tie_group": int(np.median(tie_sizes)) if tie_sizes else 0}

        r = rows[tag]
        print(f"{tag}:  target is tied with >=1 other POI in {r['pct_target_tied']}% of "
              f"instances (median tie group {r['median_tie_group']})")
        for conv in ("optimistic", "pessimistic", "expected"):
            m = r[conv]
            print(f"   {conv:12s} " + "  ".join(f"{k}={m[k]:.4f}" for k in
                                                ("acc@1", "acc@5", "acc@10", "acc@20", "mrr")))
        print()

    gn = json.loads((ROOT / "getnext_row.json").read_text()) if (ROOT / "getnext_row.json").exists() else None
    if gn:
        print(f"GETNext (as released, per-instance, float logits -- ties are negligible):")
        print("   " + "  ".join(f"{k}={gn[k]:.4f}" for k in
                                ("acc@1", "acc@5", "acc@10", "acc@20", "mrr")))
        print()
        for conv in ("optimistic", "expected", "pessimistic"):
            c = rows["count-traj"][conv]
            d10 = (c["acc@10"] / gn["acc@10"] - 1) * 100
            d1 = (c["acc@1"] / gn["acc@1"] - 1) * 100
            print(f"   COUNT-traj ({conv:11s}) vs GETNext:  Acc@10 {d10:+6.1f}%   Acc@1 {d1:+6.1f}%")

    (ROOT / "tie_audit.json").write_text(json.dumps(rows, indent=2))
    print("\n-> tie_audit.json")


if __name__ == "__main__":
    main()
