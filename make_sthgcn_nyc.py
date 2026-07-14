#!/usr/bin/env python3
"""
make_sthgcn_nyc.py -- feed official STHGCN our chronological split.

STHGCN dispatches on dataset name: 'nyc' goes to preprocess_nyc(), which reads
raw/NYC_{train,val,test}.csv -- i.e. GETNext's own preprocessed CSVs -- and takes the split
from file membership. (So STHGCN's published NYC numbers are computed on GETNext's split;
the two papers share a protocol.) The 15 columns it renames positionally are exactly the
15 our make_getnext_data.py already emits, in the same order, so no reformatting is needed.

We therefore only have to re-cut our export into three files:
    NYC_train.csv  our T0, minus a validation tail
    NYC_val.csv    the last 10% of T0 by time (STHGCN early-stops on it)
    NYC_test.csv   our T1-T5 stream  <- the thing we actually score
Everything the model is evaluated on is strictly later than everything it trained on.

NOTE: the chained-assignment defect documented in make_sthgcn_data.py lives in
FileReader.split_train_test, which preprocess_nyc does NOT call. It affects STHGCN's TKY and
CA paths, not this one. We state it that narrowly.

Usage:  python make_sthgcn_nyc.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
GN = ROOT.parent / "baselines" / "GETNext" / "dataset" / "OURS-NYC"
OUT = ROOT.parent / "baselines" / "STHGCN" / "data" / "ours_nyc" / "raw"

tr = pd.read_csv(GN / "OURS-NYC_train.csv", encoding="latin-1")   # our T0
te = pd.read_csv(GN / "OURS-NYC_val.csv", encoding="latin-1")     # our T1-T5

# carve the validation set out of the TAIL of T0, by trajectory, so no trajectory is split
tr["_t"] = pd.to_datetime(tr["local_time"])
order = tr.groupby("trajectory_id")["_t"].min().sort_values()
n_val = int(0.10 * len(order))
val_traj = set(order.index[-n_val:])
va = tr[tr.trajectory_id.isin(val_traj)].drop(columns="_t")
tr = tr[~tr.trajectory_id.isin(val_traj)].drop(columns="_t")

OUT.mkdir(parents=True, exist_ok=True)
tr.to_csv(OUT / "NYC_train.csv", index=False)
va.to_csv(OUT / "NYC_val.csv", index=False)
te.to_csv(OUT / "NYC_test.csv", index=False)

print(f"-> {OUT}")
for tag, d in (("train (T0)", tr), ("val (T0 tail)", va), ("test (T1-T5)", te)):
    print(f"  {tag:15s} {len(d):>7,} check-ins  {d.trajectory_id.nunique():>6,} trajectories")
print(f"\n  train ends   {pd.to_datetime(tr.local_time).max()}")
print(f"  val   spans  {pd.to_datetime(va.local_time).min()} .. {pd.to_datetime(va.local_time).max()}")
print(f"  test  starts {pd.to_datetime(te.local_time).min()}")
