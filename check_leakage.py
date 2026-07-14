#!/usr/bin/env python3
"""
check_leakage.py -- how much do the T0 and T1-T5 blocks overlap in wall-clock time?

Blocks are assigned by a trajectory's START time (so that no trajectory straddles a
boundary), which means a trajectory that began just before the cut can carry check-ins past
it. The training block can therefore contain check-ins that occur LATER than the earliest
evaluation check-ins. That is a boundary artifact of trajectory-level splitting, not a
design choice, and it has to be measured rather than assumed small.

We report, for each dataset:
  * the last T0 check-in and the first post-T0 check-in
  * the share of post-T0 check-ins that occur BEFORE the last T0 check-in
    (these are the ones a T0-trained model could, in principle, have seen the future of)
  * the share of T0 check-ins that occur AFTER the first post-T0 check-in
If those shares are tiny, the split is chronologically clean for practical purposes and we
say so with a number. If they are not, the protocol needs changing.
"""
import json
from pathlib import Path

import pandas as pd

PROC = Path(__file__).resolve().parent / "data" / "processed"
NAMES = {"nyc": "Foursquare-NYC", "tky": "Foursquare-TKY",
         "gowalla_ca": "Gowalla-CA", "brightkite": "Brightkite-US"}

rows = []
for k, disp in NAMES.items():
    df = pd.read_csv(PROC / k / "checkins.csv.gz")
    t0 = df[df.block == "T0"]
    te = df[df.block != "T0"]

    last_t0 = t0.unix_s.max()
    first_te = te.unix_s.min()

    te_before = int((te.unix_s < last_t0).sum())
    t0_after = int((t0.unix_s > first_te).sum())

    overlap_days = max(0, (last_t0 - first_te) / 86400)
    span_days = (df.unix_s.max() - df.unix_s.min()) / 86400

    rows.append({
        "dataset": disp,
        "last_T0": str(pd.to_datetime(last_t0, unit="s")),
        "first_post_T0": str(pd.to_datetime(first_te, unit="s")),
        "overlap_days": round(overlap_days, 1),
        "overlap_pct_of_span": round(100 * overlap_days / span_days, 2),
        "post_T0_checkins_before_last_T0": te_before,
        "post_T0_pct": round(100 * te_before / len(te), 2),
        "T0_checkins_after_first_post_T0": t0_after,
        "T0_pct": round(100 * t0_after / len(t0), 2),
    })

R = pd.DataFrame(rows)
print(R[["dataset", "overlap_days", "overlap_pct_of_span",
         "post_T0_pct", "T0_pct"]].to_string(index=False))
print()
for r in rows:
    print(f"{r['dataset']:16s} last T0 = {r['last_T0'][:10]}   "
          f"first post-T0 = {r['first_post_T0'][:10]}   "
          f"overlap = {r['overlap_days']:.1f} d")
print()
worst = max(r["post_T0_pct"] for r in rows)
print(f"Worst case: {worst:.2f}% of post-T0 check-ins fall before the last T0 check-in.")
Path("leakage_check.json").write_text(json.dumps(rows, indent=2))
print("-> leakage_check.json")
