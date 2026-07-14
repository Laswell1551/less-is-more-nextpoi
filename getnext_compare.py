#!/usr/bin/env python3
"""
getnext_compare.py -- apples-to-apples comparison against official GETNext.

WHY THIS FILE EXISTS
--------------------
GETNext's evaluation is NOT our evaluation, and the two numbers must never be put in the
same table without this reconciliation:

  * GETNext scores ONLY THE LAST TIMESTEP of each trajectory
        utils.top_k_acc_last_timestep:  y_true = y_true_seq[-1]
    i.e. one evaluation instance per trajectory.
  * GETNext DROPS from validation: users unseen in train, POIs unseen in train, and
    trajectories left with fewer than `short_traj_thres` (=2) transitions.
  * Our protocol scores EVERY transition of EVERY trajectory, keeps cold users, and keeps
    unseen-POI targets (which are guaranteed misses for any model).

So we reconstruct GETNext's exact evaluation instances from the very CSVs it reads, and
score the counting baselines on precisely those instances, over the same POI catalogue
(the POIs present in train, which is GETNext's output space).

TWO COUNTERS, because they answer different questions:
  count-static  : counts from T0 only, then frozen. GETNext is also trained on T0 and then
                  frozen, so this isolates MODEL QUALITY -- counting vs deep learning, with
                  neither one allowed to look at the stream.
  count-stream  : counts from T0 plus every check-in strictly earlier than the instance
                  being predicted. This is the deployed method. It is NOT a fair
                  model-quality comparison against a frozen GETNext -- it is the point of
                  the paper (the counter can be updated for free; the network cannot).

Usage:
  python getnext_compare.py --name OURS-NYC
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GN = ROOT.parent / "baselines" / "GETNext" / "dataset"
SHORT_TRAJ_THRES = 2          # GETNext's default


def build_instances(name):
    """Reproduce GETNext's TrajectoryDatasetVal, exactly."""
    d = GN / name
    train = pd.read_csv(d / f"{name}_train.csv", encoding="latin-1")
    val = pd.read_csv(d / f"{name}_val.csv", encoding="latin-1")

    train_users = set(train.user_id.astype(str))
    train_pois = set(train.POI_id)

    val = val.copy()
    val["ts"] = pd.to_datetime(val["local_time"]).astype("int64") // 10**9

    inst = []
    dropped_user = dropped_short = 0
    for tid, g in val.groupby("trajectory_id", sort=False):
        u = str(tid).split("_")[0]
        if u not in train_users:                     # "Ignore user if not in training set"
            dropped_user += 1
            continue
        keep = g[g.POI_id.isin(train_pois)]          # "Ignore poi if not in training set"
        pois = keep.POI_id.tolist()
        ts = keep.ts.tolist()
        if len(pois) - 1 < SHORT_TRAJ_THRES:         # input_seq shorter than the threshold
            dropped_short += 1
            continue
        # GETNext scores label_seq[-1] == pois[-1], given input_seq == pois[:-1].
        # The instance's time is the TARGET's own timestamp (after POI filtering).
        # `prefix` is exactly GETNext's input_seq -- the only post-T0 information it sees.
        # `traj_id` is carried so that GETNext's own per-instance prediction dump can be
        # joined back onto these instances (discovery_analysis.py).
        inst.append({"user": int(u), "target": pois[-1], "t": int(ts[-1]),
                     "prefix": pois[:-1], "traj_id": str(tid)})

    # every val check-in, in time order -- the stream the counter is allowed to observe
    events = val[["user_id", "POI_id", "ts"]].sort_values("ts", kind="stable")
    return inst, train, sorted(train_pois), dropped_user, dropped_short, events


def counters(name, inst, train, poi_list, events):
    """Score count-static and count-stream on GETNext's exact instances."""
    poi_ix = {p: i for i, p in enumerate(poi_list)}
    P = len(poi_list)
    # index EVERY user seen anywhere: cold users' trajectories are dropped from the
    # instances, but their check-ins still occur in the stream and still move popularity
    users = sorted(set(train.user_id.astype(int)) |
                   set(events.user_id.astype(int)) |
                   {i["user"] for i in inst})
    u_ix = {u: i for i, u in enumerate(users)}
    U = len(users)

    # ---- T0 counts (this is the counter's entire "training") --------------------
    # float64, NOT float32. The counter's score is  M[u,p] + 1e-6 * pop(p)/max(pop), and the
    # second term is a strict tie-break. float32's representable step near a value v is about
    # v*1.2e-7, so for any POI the user has visited more than ~8 times the step exceeds the
    # entire eps term and the tie-break is ROUNDED AWAY -- two such POIs then collide exactly,
    # near the TOP of the ranking, where collisions cost accuracy. Measured (dtype_control.py,
    # same instances, only the dtype changed): float32 ties on 49.1% of targets and the
    # optimistic convention inflates Acc@10 by 4.7%; float64 ties on 31.5% and inflates by 0.5%.
    # We reported float32 numbers for a long time. See sec:silent-failures.
    M0 = np.zeros((U, P), dtype=np.float64)
    pop0 = np.zeros(P, dtype=np.float64)
    for u, p in zip(train.user_id.astype(int), train.POI_id):
        if p in poi_ix:
            M0[u_ix[u], poi_ix[p]] += 1.0
            pop0[poi_ix[p]] += 1.0

    # Causal sweep. The stream counter must absorb EVERY val check-in that happened before
    # the instance being scored -- not merely the instances GETNext happens to score (it
    # scores only one transition per trajectory, but all the other check-ins still occurred).
    ev_u = events.user_id.astype(int).to_numpy()
    ev_p = events.POI_id.to_numpy()
    ev_t = events.ts.to_numpy()

    order = np.argsort([i["t"] for i in inst], kind="stable")
    M = M0.copy(); pop = pop0.copy()
    pn0 = pop0 / (pop0.max() + 1e-9)

    res = {k: [] for k in ("count-static", "count-traj", "count-stream", "popularity")}
    res_opt = {k: [] for k in ("count-static", "count-traj", "count-stream", "popularity")}
    EPS = 1e-6
    cur = 0                                      # events strictly before the current instance
    for j in order:
        e = inst[j]
        while cur < len(ev_t) and ev_t[cur] < e["t"]:
            p = ev_p[cur]
            if p in poi_ix:                      # POIs outside GETNext's output space
                M[u_ix[ev_u[cur]], poi_ix[p]] += 1.0
                pop[poi_ix[p]] += 1.0
            cur += 1

        ui = u_ix[e["user"]]; ti = poi_ix[e["target"]]
        pn = pop / (pop.max() + 1e-9)

        # count-traj: T0 counts + ONLY the current trajectory's prefix. This is EXACTLY the
        # information GETNext has at inference (T0 in its weights, input_seq at its input),
        # so it is the strict like-for-like model-quality comparison.
        s_traj = M0[ui] + EPS * pn0
        if e["prefix"]:
            s_traj = s_traj.copy()
            for p in e["prefix"]:
                s_traj[poi_ix[p]] += 1.0

        for tag, s in (("count-static", M0[ui] + EPS * pn0),
                       ("count-traj", s_traj),
                       ("count-stream", M[ui] + EPS * pn),
                       ("popularity", pn)):
            # TIE-BREAKING. The counter's score is <integer visit count> + EPS*<normalised
            # popularity>, and popularity is itself an integer count -- so two POIs with the
            # same visit count AND the same popularity have EXACTLY equal scores. The target
            # is tied with at least one other POI in ~40% of instances.
            #
            # (s > s*).sum() + 1 would place the target FIRST among its ties -- the optimistic
            # convention. GETNext's scores are neural-network floats and essentially never tie,
            # so the optimistic convention would hand the counter an advantage it denies its
            # opponent. We therefore use the EXPECTED rank under random tie-breaking, which is
            # what a real deployment would get. Both are recorded so the gap is auditable.
            st = s[ti]
            gt = int((s > st).sum())          # strictly better
            eq = int((s == st).sum())         # tied with the target, INCLUDING the target
            res[tag].append(gt + (eq + 1) / 2.0)   # expected rank
            res_opt[tag].append(gt + 1)            # optimistic rank (reported for audit only)

    n = len(inst)
    out = {}
    for tag, ranks in res.items():
        opt = res_opt[tag]
        n_tied = sum(1 for a, b in zip(ranks, opt) if a != b)
        acc = lambda k: sum(1 for r in ranks if r <= k) / n
        # MRR truncated at rank 20, because GETNext's dump only holds its top-20 and a
        # truncated MRR cannot be compared with an untruncated one. Reporting the counter's
        # full-catalogue MRR beside GETNext's top-20 MRR would flatter the counter by giving
        # it credit for ranks GETNext is not even allowed to express. Same cutoff, both sides.
        out[tag] = {"acc@1": round(acc(1), 4), "acc@5": round(acc(5), 4),
                    "acc@10": round(acc(10), 4), "acc@20": round(acc(20), 4),
                    "mrr@20": round(sum(1.0 / r for r in ranks if r <= 20) / n, 4),
                    "mrr_full": round(sum(1.0 / r for r in ranks) / n, 4),
                    "pct_target_tied": round(100.0 * n_tied / n, 1),
                    "OPTIMISTIC_acc@10_not_used": round(sum(1 for r in opt if r <= 10) / n, 4),
                    "OPTIMISTIC_acc@1_not_used": round(sum(1 for r in opt if r <= 1) / n, 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="OURS-NYC")
    args = ap.parse_args()

    inst, train, poi_list, du, ds, events = build_instances(args.name)
    print(f"[{args.name}] GETNext's exact evaluation set")
    print(f"  instances (= surviving val trajectories, LAST timestep only): {len(inst):,}")
    print(f"  trajectories dropped -- cold user: {du:,} | too short after POI filter: {ds:,}")
    print(f"  POI catalogue (train POIs = GETNext's output space): {len(poi_list):,}")
    print()

    out = counters(args.name, inst, train, poi_list, events)
    print(f"  {'method':16s} {'Acc@1':>8s} {'Acc@5':>8s} {'Acc@10':>8s} {'Acc@20':>8s} {'MRR':>8s}")
    for tag in ("popularity", "count-static", "count-traj", "count-stream"):
        m = out[tag]
        print(f"  {tag:16s} {m['acc@1']:8.4f} {m['acc@5']:8.4f} {m['acc@10']:8.4f} "
              f"{m['acc@20']:8.4f} {m['mrr@20']:8.4f} {m['mrr_full']:8.4f}")
    print("\n  count-static : T0 only, frozen -- the fair model-quality comparison vs a "
          "GETNext that is also trained on T0 and frozen.")
    print("  count-stream : T0 + all check-ins strictly before each instance -- the deployed "
          "method (the counter updates for free; the network cannot).")

    import json
    (ROOT / f"getnext_compare_{args.name}.json").write_text(json.dumps(
        {"name": args.name, "n_instances": len(inst),
         "dropped_cold_user_trajs": du, "dropped_short_trajs": ds,
         "n_pois": len(poi_list), "results": out}, indent=2))
    print(f"\n-> getnext_compare_{args.name}.json")


if __name__ == "__main__":
    main()
