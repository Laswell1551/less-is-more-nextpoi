#!/usr/bin/env python3
"""
make_sthgcn_data.py -- export OUR chronological split into the raw format official STHGCN
reads, plus a companion split file so its train/validation/test match ours exactly.

STHGCN's FileReader (for nyc/tky) expects the raw TSMC2014 tab-separated layout:
    UserId, PoiId, PoiCategoryId, PoiCategoryName, Latitude, Longitude, TimezoneOffset, UTCTime
with UTCTime formatted "%a %b %d %H:%M:%S +0000 %Y". We regenerate exactly that from our
processed, 10-core-filtered, deduplicated check-ins, so STHGCN sees our data and our
preprocessing.

TWO THINGS ABOUT ITS SPLIT
1. STHGCN re-splits internally at 80/10/90 by check-in rank in time. We must override that
   with our own T0 / T1-T5 blocks, or the comparison is not on the same split.
2. Its `split_train_test` assigns the split with pandas CHAINED ASSIGNMENT:
       df.iloc[validation_index:test_index]['SplitTag'] = 'validation'
   which writes to a temporary copy. On any modern pandas this is a silent no-op: every row
   stays 'train' and the validation and test sets come out EMPTY. (Verified on pandas 3.0.3;
   the repo pins pandas 0.24.2, where it may have written through.) The patch in
   sthgcn_patch.py fixes this with .loc AND drives the split from our blocks.

Usage:
  python make_sthgcn_data.py --dataset nyc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT.parent / "baselines" / "STHGCN" / "data"

NAME = {"nyc": "ours_nyc", "tky": "ours_tky"}


def recover_local_offset(df):
    """Same whole-hour offset recovery as make_getnext_data (disambiguated by the stored
    day-of-week, because (local_hour - utc_hour) mod 24 is ambiguous for negative offsets)."""
    utc = pd.to_datetime(df["unix_s"], unit="s")
    off_pos = (df["hour"].to_numpy() - utc.dt.hour.to_numpy()) % 24
    dow = df["dow"].to_numpy()
    ok_pos = (utc + pd.to_timedelta(off_pos, unit="h")).dt.dayofweek.to_numpy() == dow
    ok_neg = (utc + pd.to_timedelta(off_pos - 24, unit="h")).dt.dayofweek.to_numpy() == dow
    assert (ok_pos | ok_neg).all() and not (ok_pos & ok_neg).any()
    return utc, np.where(ok_pos, off_pos, off_pos - 24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    args = ap.parse_args()

    df = pd.read_csv(PROC / args.dataset / "checkins.csv.gz")
    df = df.sort_values(["user_idx", "unix_s"]).reset_index(drop=True)
    utc, off_h = recover_local_offset(df)

    cat = (df["cat"].fillna("Unknown").astype(str)
           .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
           .replace("", "Unknown"))

    raw = pd.DataFrame({
        "UserId": df["user_idx"].to_numpy(),
        "PoiId": df["poi_idx"].astype(str).to_numpy(),
        "PoiCategoryId": pd.factorize(cat)[0],
        "PoiCategoryName": cat.to_numpy(),
        "Latitude": df["lat"].to_numpy(),
        "Longitude": df["lon"].to_numpy(),
        "TimezoneOffset": off_h * 60,
        # STHGCN parses this with "%a %b %d %H:%M:%S +0000 %Y"
        "UTCTime": utc.dt.strftime("%a %b %d %H:%M:%S +0000 %Y").to_numpy(),
    })

    d = OUT / NAME[args.dataset] / "raw"
    d.mkdir(parents=True, exist_ok=True)
    fn = f"dataset_{NAME[args.dataset]}.txt"
    raw.to_csv(d / fn, sep="\t", header=False, index=False)

    # companion: the split we want STHGCN to use, keyed by (UserId, UTCTime) which is unique.
    # dtype "<U10" is load-bearing: np.where over "train"/"test" infers "<U5", and assigning
    # "validation" into that array SILENTLY TRUNCATES it to "valid".
    split = np.where(df["block"].to_numpy() == "T0", "train", "test").astype("<U10")
    # carve a validation slice out of the TAIL of T0 (STHGCN early-stops on it). Taking it
    # from the tail keeps it chronologically adjacent to the test period, and keeps the
    # training set strictly earlier than everything it is evaluated on.
    t0 = np.flatnonzero(split == "train")
    t0_by_time = t0[np.argsort(df["unix_s"].to_numpy()[t0], kind="stable")]
    n_val = int(0.1 * len(t0_by_time))
    split[t0_by_time[-n_val:]] = "validation"

    comp = pd.DataFrame({"UserId": raw["UserId"], "UTCTime": raw["UTCTime"],
                         "SplitTag": split})
    comp.to_csv(d.parent / "our_split.csv", index=False)

    print(f"[{args.dataset}] -> {d/fn}")
    print(f"  {len(raw):,} check-ins  |  {raw.UserId.nunique():,} users  "
          f"|  {raw.PoiId.nunique():,} POIs")
    for tag in ("train", "validation", "test"):
        n = int((split == tag).sum())
        print(f"  {tag:11s} {n:>7,}  ({n/len(split):.1%})")
    print(f"  -> {d.parent/'our_split.csv'}  (companion; sthgcn_patch.py applies it)")


if __name__ == "__main__":
    main()
