#!/usr/bin/env python3
"""
getnext_row.py -- GETNext's rows in the central table, computed with the SAME estimator as
every other row in that table.

WHY THIS EXISTS
Two estimator mismatches were silently corrupting the comparison, and both are the kind of
error this paper is about:

  1. BATCH-AVERAGED vs PER-INSTANCE. GETNext's own validation loop reports metrics averaged over
     BATCHES: top_k_acc_last_timestep returns a per-batch hit-rate and train.py takes the mean of
     those. When the last batch is short that is not a per-instance mean. On NYC the gap is real
     (batch-averaged Acc@10 = .5531 at its selected epoch; per-instance = .5544 over the same
     5,550 instances). Every other row in the table is a per-instance mean, so GETNext's must be
     too.

  2. TRUNCATED vs FULL-CATALOGUE MRR. The counter's rank is computed over the whole catalogue as
     (s > s[target]).sum() + 1. If GETNext's MRR were computed from a top-20 dump, ranks past 20
     would contribute 0 and its MRR would be an underestimate -- flattering the counter. So
     eval_dump.py now records the target's FULL-CATALOGUE rank using that identical expression,
     and we use it here.

Neither correction favours us. The first makes GETNext look slightly better; the second removes
an advantage the counter should never have had. We make both anyway, because they are correct.

Usage:  python getnext_row.py                 # both checkpoints
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GN = ROOT.parent / "baselines" / "GETNext"

JOBS = [("as released",      "preds_asreleased.json", "getnext_row.json"),
        ("index-corrected",  "preds_fixed.json",      "getnext_row_fixed.json")]

# GETNext's evaluation is deterministic ONLY once PYTHONHASHSEED is fixed. Its val set is built by
# iterating set(df['trajectory_id']) over STRING ids, whose order Python randomises per process;
# with shuffle=False that fixes the batch composition; and because the released transformer attends
# ACROSS the mini-batch (sec:getnext-bug), a prediction depends on its batch-mates.
#
# Measured (hashseed_experiment.py): three runs at a FIXED seed are bit-identical (sd 0.00000);
# three runs at seeds {0,1,7} re-rank 59.5% of instances and move Acc@10 over .5510-.5548.
#
# Reporting one arbitrary hash seed would therefore be reporting one arbitrary draw. We report the
# MEAN over the three seeds, with the spread, and we say so in the caption.
HASHSEED_RUNS = ["hashseed_0.json", "hashseed_1.json", "hashseed_7.json"]


def score(path):
    P = json.loads(path.read_text())
    rows = P["preds"]
    n = len(rows)
    assert n == 5550, f"expected the 5,550 Protocol-A instances, got {n}"
    assert "target_rank" in rows[0], (
        f"{path.name} has no 'target_rank' -- it was written by a pre-patch eval_dump.py. "
        f"Re-run eval_dump.py so the full-catalogue rank is recorded; a top-20 dump cannot "
        f"produce an MRR comparable with the counter's.")

    ranks = [r["target_rank"] for r in rows]
    out = {f"acc@{k}": round(sum(1 for r in ranks if r <= k) / n, 4) for k in (1, 5, 10, 20)}
    out["mrr"] = round(sum(1.0 / r for r in ranks) / n, 4)
    out["n_instances"] = n
    out["estimator"] = ("per-instance mean; MRR over the full catalogue via "
                        "(logits > logits[target]).sum()+1 -- identical to the counter's")

    # cross-check: the top-20 membership dump must agree with the recorded rank
    bad = 0
    for r in rows:
        in20 = r["target_idx"] in r["top20_idx"]
        if in20 != (r["target_rank"] <= 20):
            bad += 1
    assert bad == 0, f"{bad} instances where target_rank disagrees with top20 membership"
    return out


def main():
    KS = ("acc@1", "acc@5", "acc@10", "acc@20", "mrr")

    # ---- as released: MEAN over PYTHONHASHSEED in {0,1,7}, because one seed is one arbitrary draw
    runs = [score(GN / f) for f in HASHSEED_RUNS if (GN / f).exists()]
    if len(runs) == len(HASHSEED_RUNS):
        row = {k: round(st.mean(r[k] for r in runs), 4) for k in KS}
        row["sd"] = {k: round(st.pstdev([r[k] for r in runs]), 5) for k in KS}
        row["n_instances"] = runs[0]["n_instances"]
        row["n_hashseeds"] = len(runs)
        row["estimator"] = (
            "per-instance mean over the 5,550 Protocol-A instances; MRR over the FULL catalogue via "
            "(logits > logits[target]).sum()+1, identical to the counter's; and averaged over "
            "PYTHONHASHSEED in {0,1,7}. The last part is necessary, not fussy: at a FIXED hash seed "
            "this evaluation is bit-identical run to run (sd 0.00000), but the hash seed determines "
            "the batch composition and the released transformer attends across the mini-batch, so a "
            "single seed reports one arbitrary draw. See hashseed_experiment.py.")
        (ROOT / "getnext_row.json").write_text(json.dumps(row, indent=2))
        print(f"GETNext (as released), mean over PYTHONHASHSEED {{0,1,7}}, "
              f"{row['n_instances']:,} instances:")
        print("   " + "  ".join(f"{k}={row[k]:.4f}" for k in KS))
        print("   sd " + "  ".join(f"{k}={row['sd'][k]:.5f}" for k in KS))
        print("   -> getnext_row.json\n")
    else:
        print("!! hashseed runs missing; falling back to a single run (NOT recommended)")
        row = score(GN / "preds_asreleased.json")
        (ROOT / "getnext_row.json").write_text(json.dumps(row, indent=2))

    # ---- index-corrected: single run (it is used only to show the fix DEGRADES the model)
    p = GN / "preds_fixed.json"
    if p.exists():
        row = score(p)
        row["note"] = ("single run. Used only to show that correcting the index misalignment "
                       "degrades the model; the conclusion is far larger than the hash-seed spread.")
        (ROOT / "getnext_row_fixed.json").write_text(json.dumps(row, indent=2))
        print(f"GETNext (index-corrected), {row['n_instances']:,} instances:")
        print("   " + "  ".join(f"{k}={row[k]:.4f}" for k in KS))
        print("   -> getnext_row_fixed.json")


if __name__ == "__main__":
    main()
