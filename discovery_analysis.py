#!/usr/bin/env python3
"""
discovery_analysis.py -- what is the benchmark actually rewarding?

THE OBJECTION THIS ANSWERS
"Your counter only predicts revisits. A POI *recommender* is supposed to help people
discover places they have not been -- a system that only ever returns your own haunts is
useless." That objection is correct about the counter, and it would be fatal, EXCEPT that it
applies to everything else too. So we measure it rather than argue.

For every evaluation instance we ask whether the target POI is one the user has visited
before (a REVISIT) or not (a DISCOVERY), and we report each method's accuracy separately on
the two. The interesting number is the discovery column: if every method -- the counter, the
neural backbones, the 7B language model -- is near zero there, then no method in this
literature does discovery, all reported progress is progress at predicting revisits, and the
benchmark is measuring something other than what the task is named for.

Runs on the Protocol-A instance set (GETNext's own), so the numbers line up with Table 3.

Usage:  python discovery_analysis.py --name OURS-NYC
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ranking as RK
import pandas as pd

from getnext_compare import build_instances

ROOT = Path(__file__).resolve().parent
EPS = 1e-6


def load_getnext_preds(path):
    """GETNext's own per-instance top-20, dumped from its saved best-epoch checkpoint by
    baselines/GETNext/eval_dump.py (which reuses GETNext's model and validation loop)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    idx2poi = {v: k for k, v in d["poi_id2idx"].items()}
    out = {}
    for p in d["preds"]:
        out[str(p["traj_id"])] = {
            "target": idx2poi[p["target_idx"]],
            "top20": [idx2poi[i] for i in p["top20_idx"]],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="OURS-NYC")
    ap.add_argument("--getnext-preds", nargs="*", default=[], dest="gn",
                    help="label=path/to/preds.json, e.g. as-released=../baselines/GETNext/preds_asreleased.json")
    args = ap.parse_args()

    GN = {}
    for spec in args.gn:
        lab, path = spec.split("=", 1)
        GN[lab] = load_getnext_preds(path)
        print(f"  loaded {len(GN[lab]):,} GETNext predictions for '{lab}'")

    inst, train, poi_list, du, ds, events = build_instances(args.name)
    poi_ix = {p: i for i, p in enumerate(poi_list)}
    P = len(poi_list)
    users = sorted(set(train.user_id.astype(int)) | set(events.user_id.astype(int)))
    u_ix = {u: i for i, u in enumerate(users)}

    # what each user had already visited, in T0
    seen = [set() for _ in users]
    # float64: in float32 the eps*popularity tie-break is rounded away above ~8 visits.
    # See getnext_compare.py and dtype_control.py.
    M0 = np.zeros((len(users), P), dtype=np.float64)
    pop0 = np.zeros(P, dtype=np.float64)
    for u, p in zip(train.user_id.astype(int), train.POI_id):
        if p in poi_ix:
            seen[u_ix[u]].add(poi_ix[p])
            M0[u_ix[u], poi_ix[p]] += 1.0
            pop0[poi_ix[p]] += 1.0

    ev_u = events.user_id.astype(int).to_numpy()
    ev_p = events.POI_id.to_numpy()
    ev_t = events.ts.to_numpy()

    M = M0.copy(); pop = pop0.copy()
    seen_now = [set(s) for s in seen]
    pn0 = pop0 / (pop0.max() + 1e-9)

    order = np.argsort([i["t"] for i in inst], kind="stable")
    cur = 0
    rows = []
    for j in order:
        e = inst[j]
        while cur < len(ev_t) and ev_t[cur] < e["t"]:
            p = ev_p[cur]
            if p in poi_ix:
                ui = u_ix[ev_u[cur]]
                M[ui, poi_ix[p]] += 1.0
                pop[poi_ix[p]] += 1.0
                seen_now[ui].add(poi_ix[p])
            cur += 1

        ui = u_ix[e["user"]]; ti = poi_ix[e["target"]]
        is_revisit = ti in seen_now[ui]          # has this user been here before, ever?

        pn = pop / (pop.max() + 1e-9)
        s_traj = M0[ui] + EPS * pn0
        for p in e["prefix"]:
            s_traj[poi_ix[p]] += 1.0
        scores = {
            "COUNT (same info)": s_traj,
            "COUNT (stream)": M[ui] + EPS * pn,
            "Popularity": pn,
        }
        r = {"revisit": bool(is_revisit)}
        for k, s in scores.items():
            # expected rank under random tie-breaking -- NOT (s > s*).sum()+1, which
            # would place the target first among its ties. The counter ties on ~40% of
            # instances (integer counts); GETNext's float logits never do. See ranking.py.
            r[k] = RK.expected_rank_np(s, ti)
        # GETNext: rank of the target within its own dumped top-20 (>20 => a miss at any K<=20).
        # POI ids come back from the dump as strings (JSON keys) but pandas parsed ours as
        # int64, so both sides are normalised to str before anything is compared.
        for lab, preds in GN.items():
            p = preds.get(e["traj_id"])
            if p is None:
                r[f"GETNext ({lab})"] = 10 ** 6
                continue
            tgt = str(e["target"])
            assert str(p["target"]) == tgt, (
                f"instance mismatch on {e['traj_id']}: GETNext's target {p['target']!r} != "
                f"ours {tgt!r} -- the two instance sets are not the same")
            top = [str(x) for x in p["top20"]]
            r[f"GETNext ({lab})"] = top.index(tgt) + 1 if tgt in top else 10 ** 6
        rows.append(r)

    R = pd.DataFrame(rows)
    n = len(R)
    n_rev = int(R.revisit.sum()); n_dis = n - n_rev

    print(f"\n[{args.name}] {n:,} evaluation instances (Protocol A)")
    print(f"  REVISIT   (target seen by this user before): {n_rev:,}  ({n_rev/n:.1%})")
    print(f"  DISCOVERY (target NEVER seen by this user):  {n_dis:,}  ({n_dis/n:.1%})")
    print()
    print(f"  {'method':24s} {'Acc@10 all':>11s} {'on REVISIT':>11s} {'on DISCOVERY':>13s}")
    out = {}
    cols = (["Popularity"] + [f"GETNext ({l})" for l in GN]
            + ["COUNT (same info)", "COUNT (stream)"])
    for k in cols:
        a_all = (R[k] <= 10).mean()
        a_rev = (R[R.revisit][k] <= 10).mean() if n_rev else float("nan")
        a_dis = (R[~R.revisit][k] <= 10).mean() if n_dis else float("nan")
        print(f"  {k:24s} {a_all:11.4f} {a_rev:11.4f} {a_dis:13.4f}")
        out[k] = {"all": round(float(a_all), 4), "revisit": round(float(a_rev), 4),
                  "discovery": round(float(a_dis), 4)}

    print()
    print("  A counter CANNOT rank an unvisited POI above a visited one, so its discovery")
    print("  accuracy is bounded by the popularity back-off alone. The question the paper")
    print("  must answer is what a deep model or a 7B LLM scores in that same column --")
    print("  if they are also near zero, the benchmark is not measuring discovery at all,")
    print("  and the field's reported progress is progress at predicting revisits.")
    print()
    print(f"  Share of the counter's Acc@10 hits that are revisits: "
          f"{(R[R.revisit]['COUNT (stream)'] <= 10).sum() / max((R['COUNT (stream)'] <= 10).sum(), 1):.1%}")

    (ROOT / f"discovery_{args.name}.json").write_text(json.dumps(
        {"name": args.name, "n": n, "n_revisit": n_rev, "n_discovery": n_dis,
         "results": out}, indent=2))
    print(f"\n-> discovery_{args.name}.json")


if __name__ == "__main__":
    main()
