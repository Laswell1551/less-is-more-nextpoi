#!/usr/bin/env python3
"""
hashseed_experiment.py -- does GETNext's reported accuracy depend on PYTHONHASHSEED?

THE CLAIM WE ARE TESTING (and which we nearly published on no evidence at all)
An earlier draft asserted: "Re-evaluating a single saved checkpoint under PYTHONHASHSEED in
{0,1,7} moves Acc@1 over .2414-.2471 and MRR over .3448-.3480 -- a 2.4% relative spread."

We went looking for the source of those numbers. They existed only inside a hardcoded print()
string in getnext_transformer_check.py. No experiment had ever been run. So we ran one.

THE MECHANISM (verified in the released code, not assumed)
    eval loop:  for traj_id in set(df['trajectory_id'].tolist()):   <- a set() over STRING ids
Python randomises string hash order per process, so the trajectory order -- and hence, with
shuffle=False, the BATCH COMPOSITION -- changes from run to run. And because the released
transformer applies its causal mask across the mini-batch rather than across time
(sec:getnext-bug), a prediction depends on which other users share its batch.

THE CONTROL THAT MAKES IT MEAN ANYTHING
Predictions also move between two IDENTICAL runs, purely from GPU floating-point
non-determinism. So "the rank changed when I varied the hash seed" proves nothing on its own.
We therefore compare:

    TREATMENT : 3 runs, PYTHONHASHSEED = 0, 1, 7     (batch composition varies + float noise)
    CONTROL   : 3 runs, PYTHONHASHSEED = 0, 0, 0     (float noise ONLY)

Only the EXCESS of treatment over control is attributable to hash order.

We also report the metric-relevant quantity, which is not "did the rank move at all" -- float
noise moves rank 137 to 139 and nobody cares -- but "did the rank cross a scoring boundary",
i.e. did the instance flip between hit and miss at K.
"""
import json
import statistics as st
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GN = ROOT.parent / "baselines" / "GETNext"


def load(pattern):
    runs = {}
    for f in sorted(GN.glob(pattern)):
        P = json.loads(f.read_text())["preds"]
        runs[f.name] = {
            "ranks": {r["traj_id"]: r["target_rank"] for r in P},
            "order": [r["traj_id"] for r in P],
            "n": len(P),
        }
    return runs


def summarise(runs, label):
    ks = list(runs)
    if len(ks) < 2:
        return None
    common = set(runs[ks[0]]["ranks"])
    for k in ks[1:]:
        common &= set(runs[k]["ranks"])
    common = sorted(common)
    n = len(common)

    order_identical = all(runs[k]["order"] == runs[ks[0]]["order"] for k in ks[1:])

    rank_moved = sum(1 for t in common if len({runs[k]["ranks"][t] for k in ks}) > 1) / n

    flips = {}
    for K in (1, 5, 10, 20):
        f = sum(1 for t in common
                if len({runs[k]["ranks"][t] <= K for k in ks}) > 1) / n
        flips[K] = f

    accs = {K: [sum(1 for t in common if runs[k]["ranks"][t] <= K) / n for k in ks]
            for K in (1, 10)}

    out = {"runs": ks, "n": n, "batch_order_identical_across_runs": order_identical,
           "frac_rank_moved_at_all": round(rank_moved, 4),
           "frac_flipped_hit_miss": {f"@{K}": round(v, 4) for K, v in flips.items()},
           "acc@1_values": [round(v, 4) for v in accs[1]],
           "acc@10_values": [round(v, 4) for v in accs[10]],
           "acc@10_sd": round(st.pstdev(accs[10]), 5)}

    print(f"{label}")
    print(f"   runs                                : {', '.join(ks)}")
    print(f"   batch/instance order identical      : {order_identical}")
    print(f"   rank moved at all                   : {rank_moved:.1%} of instances")
    print(f"   FLIPPED hit<->miss  @1 / @10 / @20  : "
          f"{flips[1]:.2%} / {flips[10]:.2%} / {flips[20]:.2%}")
    print(f"   Acc@10 across runs                  : "
          f"{min(accs[10]):.4f}-{max(accs[10]):.4f}  (sd {st.pstdev(accs[10]):.5f})")
    print(f"   Acc@1  across runs                  : "
          f"{min(accs[1]):.4f}-{max(accs[1]):.4f}")
    print()
    return out


def main():
    ctrl = load("fixedseed_*.json")     # PYTHONHASHSEED = 0, 0, 0
    trt = load("hashseed_*.json")       # PYTHONHASHSEED = 0, 1, 7

    if not ctrl:
        print("!! fixedseed_*.json not found -- run the control first (3 evals, PYTHONHASHSEED=0)")
        return

    print("=" * 78)
    c = summarise(ctrl, "CONTROL   -- PYTHONHASHSEED fixed at 0 (float non-determinism ONLY)")
    t = summarise(trt, "TREATMENT -- PYTHONHASHSEED = 0 / 1 / 7 (hash order + float noise)")

    print("=" * 78)
    ex10 = t["frac_flipped_hit_miss"]["@10"] - c["frac_flipped_hit_miss"]["@10"]
    ex1 = t["frac_flipped_hit_miss"]["@1"] - c["frac_flipped_hit_miss"]["@1"]
    print("EXCESS ATTRIBUTABLE TO HASH ORDER (treatment minus control):")
    print(f"   instances flipping hit<->miss @1  : {ex1:+.2%}")
    print(f"   instances flipping hit<->miss @10 : {ex10:+.2%}")
    print()
    if c["batch_order_identical_across_runs"] and not t["batch_order_identical_across_runs"]:
        print("   Fixing the hash seed DOES fix the batch order; varying it does not. So the")
        print("   mechanism (set() over string ids -> batch composition) is real.")
    if abs(ex10) < 0.01 and abs(ex1) < 0.01:
        print("   But the EXCESS is within noise: GPU floating-point non-determinism alone")
        print("   already moves this many instances. We therefore CANNOT attribute the spread")
        print("   to hash order, and we do not claim it. What survives is the weaker, measured")
        print("   statement: GETNext's evaluation is not reproducible run-to-run, from at least")
        print("   two independent causes, one of which is a direct consequence of the defect.")
    else:
        print("   The excess is real: varying the hash seed changes the reported metric by more")
        print("   than repeated identical runs do. Batch composition -- and therefore an")
        print("   environment variable -- is a determinant of GETNext's reported accuracy.")

    (ROOT / "hashseed_experiment.json").write_text(json.dumps(
        {"control_PYTHONHASHSEED_fixed": c, "treatment_PYTHONHASHSEED_varied": t,
         "excess_flip_at_1": round(ex1, 4), "excess_flip_at_10": round(ex10, 4)}, indent=2))
    print("\n-> hashseed_experiment.json")


if __name__ == "__main__":
    main()
