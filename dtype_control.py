#!/usr/bin/env python3
"""
dtype_control.py -- is the counter's tie rate a property of the DATA, or of float32?

THE HYPOTHESIS
The counter scores  s = M[u,p] + eps * pop(p)/max(pop),  eps = 1e-6.

The second term is meant to be a strict tie-break. But getnext_compare.py builds M and pop as
float32, and float32's representable step near a value v is about v * 1.2e-7. So:

    v =  1  ->  step 1.2e-7   the eps term (<=1e-6) survives, ~8 distinct levels
    v =  8  ->  step 9.5e-7   the eps term is at the edge of representable
    v = 16  ->  step 1.9e-6   the eps term is BELOW the step: it is rounded away entirely

If that is right, then for any POI the user has visited more than ~8 times the popularity
back-off silently does nothing, two such POIs collide exactly, and the counter ties far more
often than its own definition says it should. The tie rate would then be an artifact of the
storage type, not a fact about check-in data -- and the "optimistic convention inflated us"
story would be measuring float32, not measuring ties.

THE CONTROL
Same instances, same counter, same expected-rank estimator. The ONLY thing that changes is the
dtype of M and pop. If the tie rate moves, the dtype is the cause. If it does not, it isn't.

Nothing here is inferred. Both numbers are measured.
"""
import numpy as np

import getnext_compare as GC

EPS = 1e-6


def run(dtype):
    inst, train, poi_list, _, _, _ = GC.build_instances("OURS-NYC")
    poi_ix = {p: i for i, p in enumerate(poi_list)}
    users = sorted(set(train.user_id.astype(int)) | {i["user"] for i in inst})
    u_ix = {u: i for i, u in enumerate(users)}

    M0 = np.zeros((len(users), len(poi_list)), dtype=dtype)
    pop0 = np.zeros(len(poi_list), dtype=dtype)
    for u, p in zip(train.user_id.astype(int), train.POI_id):
        if p in poi_ix:
            M0[u_ix[u], poi_ix[p]] += 1.0
            pop0[poi_ix[p]] += 1.0
    pn0 = pop0 / (pop0.max() + 1e-9)

    exp, opt, n_tied, tie_sizes = [], [], 0, []
    for e in inst:
        s = M0[u_ix[e["user"]]] + EPS * pn0          # <- the dtype under test
        if e["prefix"]:
            s = s.copy()
            for p in e["prefix"]:
                s[poi_ix[p]] += 1.0
        st = s[poi_ix[e["target"]]]
        gt = int((s > st).sum())
        eq = int((s == st).sum())                    # includes the target itself
        exp.append(gt + (eq + 1) / 2.0)
        opt.append(gt + 1)
        if eq > 1:
            n_tied += 1
            tie_sizes.append(eq)

    n = len(inst)
    a10 = lambda r: float(np.mean(np.asarray(r) <= 10))
    return {"acc@10_expected": round(a10(exp), 4), "acc@10_optimistic": round(a10(opt), 4),
            "pct_tied": round(100 * n_tied / n, 1),
            "median_tie_group": int(np.median(tie_sizes)) if tie_sizes else 0, "n": n}


if __name__ == "__main__":
    import json
    from pathlib import Path

    gn = 0.5527                                      # GETNext as-released, per-instance Acc@10
    print("count-traj on GETNext's 5,550 instances. Only the dtype of M and pop changes.\n")
    print(f"  {'dtype':>10s} {'%tied':>7s} {'med.grp':>8s} {'Acc@10 (exp)':>13s} "
          f"{'Acc@10 (opt)':>13s} {'opt inflates by':>16s} {'vs GETNext (exp)':>18s}")
    out = {"note": "Same instances, same counter, same estimator. ONLY the storage dtype of the "
                   "counter's M and pop changes. float32 was what the paper reported for a long "
                   "time; float64 is what Eq. (countscore) actually says.",
           "getnext_asreleased_acc@10": gn}
    for dt, nm in ((np.float32, "float32"), (np.float64, "float64")):
        r = run(dt)
        r["optimistic_inflates_pct"] = round(
            100 * (r["acc@10_optimistic"] / r["acc@10_expected"] - 1), 1)
        r["vs_getnext_pct"] = round(100 * (r["acc@10_expected"] / gn - 1), 1)
        out[nm] = r
        print(f"  {nm:>10s} {r['pct_tied']:6.1f}% {r['median_tie_group']:8d} "
              f"{r['acc@10_expected']:13.4f} {r['acc@10_optimistic']:13.4f} "
              f"{r['optimistic_inflates_pct']:15.1f}% {r['vs_getnext_pct']:17.1f}%")
    Path(__file__).resolve().parent.joinpath("dtype_control.json").write_text(
        json.dumps(out, indent=2))
    print("\n  float32 is what every reported number in the paper was computed in.")
    print("  float64 is what Eq. (countscore) actually says.")
    print("\n-> dtype_control.json")
