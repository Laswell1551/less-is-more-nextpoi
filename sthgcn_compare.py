#!/usr/bin/env python3
"""
sthgcn_compare.py -- score the counter on STHGCN's OWN evaluation instances.

WHY THIS IS NECESSARY (and why the obvious comparison would be wrong)

Our counter reaches Acc@10 = .6211 on the 5,550 instances that GETNext scores. STHGCN, run on the
same chronological split, will report its own Acc@10. It is tempting to put those two numbers side
by side. That would be wrong, and it is precisely the class of error this paper is about.

STHGCN applies DIFFERENT filters from GETNext (min_poi_freq 9, min_user_freq 9, 1440-minute
sessions), so it scores a DIFFERENT instance set. Two accuracies measured on two different sets of
instances are not comparable, however similar the protocols look.

So we do for STHGCN exactly what we did for GETNext: reconstruct its evaluation instance set from
its own preprocessed output, and score the counter on precisely those instances, with precisely the
information STHGCN has at inference.

    count-static : the base-period counts only (STRICTLY LESS than STHGCN sees)
    count-traj   : base-period counts + the current trajectory's prefix (EXACTLY what STHGCN sees)
    count-stream : base-period counts + everything strictly earlier in the stream (MORE)
    popularity   : global counts, no personalisation

Ties are broken at the EXPECTED rank (ranking.py), never optimistically -- the counter's integer
scores tie constantly and STHGCN's float scores do not, so the optimistic convention would subsidise
us and not it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ranking as RK

ROOT = Path(__file__).resolve().parent
ST = ROOT.parent / "baselines" / "STHGCN" / "data"
EPS = 1e-6


def load(dataset):
    d = ST / dataset / "preprocessed"      # STHGCN writes its split files here
    need = ["train_sample.csv", "validate_sample.csv", "test_sample.csv"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise SystemExit(
            f"STHGCN has not finished preprocessing {dataset} yet -- missing {missing}.\n"
            f"(It writes these into {d} after the hypergraph is built.)")
    tr = pd.read_csv(d / "train_sample.csv")
    va = pd.read_csv(d / "validate_sample.csv")
    te = pd.read_csv(d / "test_sample.csv")
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ours_nyc")
    args = ap.parse_args()

    tr, va, te = load(args.dataset)
    print(f"[{args.dataset}] STHGCN's own splits: "
          f"train {len(tr):,}  val {len(va):,}  test {len(te):,}")
    print(f"   columns: {list(te.columns)[:10]}")

    # ---- the output space is what STHGCN can predict: the POIs it saw in training -------------
    pois = sorted(tr.PoiId.unique())
    pix = {p: i for i, p in enumerate(pois)}
    users = sorted(set(tr.UserId) | set(te.UserId))
    uix = {u: i for i, u in enumerate(users)}
    print(f"   POI catalogue (train POIs) : {len(pois):,}")
    print(f"   users                      : {len(users):,}")

    # ---- the counter's memory, built from the training portion ONLY ---------------------------
    M0 = np.zeros((len(users), len(pois)), dtype=np.float64)
    pop0 = np.zeros(len(pois), dtype=np.float64)
    for u, p in zip(tr.UserId.to_numpy(), tr.PoiId.to_numpy()):
        if p in pix and u in uix:
            M0[uix[u], pix[p]] += 1.0
            pop0[pix[p]] += 1.0
    pn0 = pop0 / (pop0.max() + 1e-9)

    # ---- STHGCN's evaluation instances --------------------------------------------------------
    # test_sample.csv is one row per scored instance; the row IS the target check-in.
    tcol = "UTCTimeOffsetEpoch" if "UTCTimeOffsetEpoch" in te.columns else None
    # STHGCN names its trajectory column pseudo_session_trajectory_id. count-traj must see the
    # SAME trajectory prefix STHGCN sees at inference, so this has to be the right column.
    traj = next((c for c in ("pseudo_session_trajectory_id", "trajectory_id")
                 if c in te.columns), None)
    assert traj, f"no trajectory column in test_sample.csv: {list(te.columns)}"
    print(f"   trajectory column          : {traj}")

    inst, dropped_user, dropped_poi = [], 0, 0
    for r in te.itertuples():
        if r.UserId not in uix:
            dropped_user += 1          # cold user: no memory, and STHGCN has no embedding either
            continue
        if r.PoiId not in pix:
            dropped_poi += 1           # target outside STHGCN's output space: a guaranteed miss
            continue
        inst.append({"u": uix[r.UserId], "tgt": pix[r.PoiId],
                     "t": getattr(r, tcol) if tcol else 0,
                     "traj": getattr(r, traj) if traj else None})
    n = len(inst)
    print(f"\n   evaluation instances scored : {n:,}")
    print(f"   dropped, cold user          : {dropped_user:,}")
    print(f"   dropped, target POI unseen  : {dropped_poi:,}")

    # ---- the trajectory prefix that STHGCN sees at inference -----------------------------------
    #
    # CAREFUL. test_sample.csv holds ONE row per test trajectory -- its LAST check-in, the target
    # (STHGCN's only_keep_last). The trajectory's earlier check-ins -- the query sequence the model
    # actually conditions on -- are NOT in test_sample.csv. They live in sample.csv, tagged
    # 'ignore' precisely because they are input rather than a scored sample.
    #
    # Building the prefix from test_sample.csv therefore yields an EMPTY prefix for every instance,
    # which silently collapses count-traj onto count-static. It did, on the first run: both scored
    # .5662 exactly. count-traj is the like-for-like row -- the one that gives the counter exactly
    # what STHGCN gets -- so getting this wrong would have made the whole comparison meaningless.
    full = pd.read_csv(ST / args.dataset / "preprocessed" / "sample.csv")
    prefix = {}
    for tid, g in full.groupby(traj, sort=False):
        g = g.sort_values(tcol)
        prefix[tid] = [(t, pix[p]) for t, p in zip(g[tcol], g.PoiId) if p in pix]
    n_with_prefix = sum(1 for e in inst if len(prefix.get(e["traj"], [])) > 1)
    print(f"   instances with a non-empty trajectory prefix : {n_with_prefix:,} / {n:,}")

    # ---- the stream, for count-stream ----------------------------------------------------------
    ev = full.sort_values(tcol)
    ev_u = ev.UserId.to_numpy(); ev_p = ev.PoiId.to_numpy()
    ev_t = ev[tcol].to_numpy()

    order = np.argsort([i["t"] for i in inst], kind="stable")
    M = M0.copy(); pop = pop0.copy()
    cur = 0
    res = {k: [] for k in ("popularity", "count-static", "count-traj", "count-stream")}

    for j in order:
        e = inst[j]
        while cur < len(ev_t) and ev_t[cur] < e["t"]:
            p, u = ev_p[cur], ev_u[cur]
            if p in pix and u in uix:
                M[uix[u], pix[p]] += 1.0
                pop[pix[p]] += 1.0
            cur += 1
        pn = pop / (pop.max() + 1e-9)

        # count-traj: the base-period counts PLUS the check-ins of this trajectory that occur
        # STRICTLY BEFORE the target -- i.e. exactly the query sequence STHGCN conditions on.
        s_traj = M0[e["u"]] + EPS * pn0
        pre = [p for t, p in prefix.get(e["traj"], []) if t < e["t"]]
        if pre:
            s_traj = s_traj.copy()
            for p in pre:
                s_traj[p] += 1.0

        for tag, s in (("popularity", pn0),
                       ("count-static", M0[e["u"]] + EPS * pn0),
                       ("count-traj", s_traj),
                       ("count-stream", M[e["u"]] + EPS * pn)):
            res[tag].append(RK.expected_rank_np(s, e["tgt"]))

    out = {}
    print(f"\n   {'method':16s} {'Acc@1':>8s} {'Acc@5':>8s} {'Acc@10':>8s} {'Acc@20':>8s} {'MRR':>8s}")
    for tag in ("popularity", "count-static", "count-traj", "count-stream"):
        r = np.asarray(res[tag], dtype=np.float64)
        # the dropped instances are guaranteed misses for ANY method with a fixed output space,
        # and STHGCN drops them too -- so we score on the same denominator it does.
        m = {f"acc@{k}": round(float((r <= k).mean()), 4) for k in (1, 5, 10, 20)}
        m["mrr"] = round(float((1.0 / r).mean()), 4)
        out[tag] = m
        print(f"   {tag:16s} " + " ".join(f"{m[k]:8.4f}" for k in
                                          ("acc@1", "acc@5", "acc@10", "acc@20", "mrr")))

    blob = {"dataset": args.dataset, "n_instances": n, "n_pois": len(pois),
            "dropped_cold_user": dropped_user, "dropped_unseen_poi": dropped_poi,
            "tie_breaking": "expected rank (ranking.py) -- never optimistic",
            "note": ("The counter scored on STHGCN's OWN evaluation instances, which are NOT "
                     "GETNext's 5,550: STHGCN applies different filters (min_poi_freq 9, "
                     "min_user_freq 9, 1440-min sessions). Comparing an accuracy measured here "
                     "with one measured on GETNext's instances would be meaningless."),
            "results": out}
    (ROOT / f"sthgcn_compare_{args.dataset}.json").write_text(json.dumps(blob, indent=2))
    print(f"\n-> sthgcn_compare_{args.dataset}.json")
    print("\n   Compare STHGCN's reported Acc@10 with count-traj ABOVE -- not with the .6211 that")
    print("   the counter scores on GETNext's instances. Different instance set, different number.")


if __name__ == "__main__":
    main()
