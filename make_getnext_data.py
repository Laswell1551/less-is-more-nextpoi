#!/usr/bin/env python3
"""
make_getnext_data.py -- export OUR chronological splits into the official GETNext CSV schema,
so the official GETNext code can be run on exactly our data and protocol.

Mapping:
    train.csv = our T0 block          (the base period the static model is trained on)
    val.csv   = our T1..T5 stream     (GETNext's train.py scores --data-val; there is no
                                       test path in it, so val IS the evaluation set)

That makes official-GETNext-on-val precisely our `static` protocol: train once on T0,
evaluate over the whole post-T0 stream. It is the head-to-head we need against the counter.

WHY FULL-PRECISION LOCAL TIME MATTERS: GETNext's default time feature is `norm_in_day_time`
(fraction of the local day). Our processed file keeps only the integer local `hour`, so we
recover the exact local timestamp from unix_s: the Foursquare tz offsets for NYC/TKY are
whole hours, so offset = (local_hour - utc_hour) mod 24 recovers it exactly. We assert that
the reconstructed local hour and day-of-week match the stored ones. Feeding GETNext a
degraded time feature would hobble the very baseline we are trying to give its best shot.

Usage:
  python make_getnext_data.py --dataset nyc
  python make_getnext_data.py --dataset all
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT.parent / "baselines" / "GETNext" / "dataset"

NAME = {"nyc": "OURS-NYC", "tky": "OURS-TKY",
        "gowalla_ca": "OURS-CA", "brightkite": "OURS-BK"}


def to_getnext(df):
    utc = pd.to_datetime(df["unix_s"], unit="s")

    # Recover the whole-hour local offset. (local_hour - utc_hour) mod 24 is ambiguous:
    # for a negative offset like NYC's -4h it returns +20h, which lands on the right HOUR
    # but the wrong DAY. The two candidates differ by exactly 24h, hence by exactly one
    # weekday, so the stored `dow` picks out the true one.
    off_pos = (df["hour"].to_numpy() - utc.dt.hour.to_numpy()) % 24          # 0..23
    dow = df["dow"].to_numpy()

    l_pos = utc + pd.to_timedelta(off_pos, unit="h")
    l_neg = utc + pd.to_timedelta(off_pos - 24, unit="h")
    ok_pos = l_pos.dt.dayofweek.to_numpy() == dow
    ok_neg = l_neg.dt.dayofweek.to_numpy() == dow
    assert not (ok_pos & ok_neg).any(), "offset disambiguation is not unique"
    assert (ok_pos | ok_neg).all(), (
        f"neither candidate offset reproduces the stored day-of-week on "
        f"{int((~(ok_pos | ok_neg)).sum())} rows")

    off_h = np.where(ok_pos, off_pos, off_pos - 24)
    local = utc + pd.to_timedelta(off_h, unit="h")

    assert int((local.dt.hour.to_numpy() != df["hour"].to_numpy()).sum()) == 0
    assert int((local.dt.dayofweek.to_numpy() != dow).sum()) == 0

    # ASCII-fold the category names. GETNext's loaders call pd.read_csv() with no encoding
    # argument, so a non-UTF-8 byte in a venue name (e.g. "Caf\xe9") makes their own code
    # raise UnicodeDecodeError under modern pandas. The name is only ever used as a label
    # (POI_catid_code carries the signal), so folding it is lossless for the model.
    cat = (df["cat"].fillna("Unknown").astype(str)
           .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
           .replace("", "Unknown"))
    cat_code = pd.factorize(cat)[0]

    secs = (local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second).to_numpy()
    out = pd.DataFrame({
        "user_id": df["user_idx"].to_numpy(),
        "POI_id": df["poi_idx"].astype(str).to_numpy(),
        "POI_catid": cat.to_numpy(),
        "POI_catid_code": cat_code,
        "POI_catname": cat.to_numpy(),
        "latitude": df["lat"].to_numpy(),
        "longitude": df["lon"].to_numpy(),
        "timezone": off_h * 60,
        "UTC_time": utc.dt.strftime("%Y-%m-%d %H:%M:%S+00:00").to_numpy(),
        "local_time": local.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "day_of_week": local.dt.dayofweek.to_numpy(),
        "norm_in_day_time": secs / 86400.0,
        # GETNext does trajectory_id.split('_')[0] to recover the user, then looks it up in
        # user_id2idx_dict. Our own traj_id prefix is NOT the user_idx (user 0 carries
        # traj_id "1_2"), so it must be re-minted as "<user_idx>_<n>" or their lookup
        # KeyErrors. The caller supplies the re-minted id.
        "trajectory_id": df["gn_traj_id"].to_numpy(),
    })
    day0 = local.dt.normalize().min()
    out["norm_day_shift"] = (local.dt.normalize() - day0).dt.days.to_numpy()
    out["norm_relative_time"] = out["norm_day_shift"] + out["norm_in_day_time"]
    return out


def export(name):
    df = pd.read_csv(PROC / name / "checkins.csv.gz")
    # GETNext walks each user's rows in file order to build trajectories -> sort accordingly
    df = df.sort_values(["user_idx", "unix_s"]).reset_index(drop=True)

    # Re-mint trajectory ids as "<user_idx>_<k>" (GETNext parses the user out of this).
    # Done on the FULL frame before splitting so a trajectory keeps one id across blocks
    # (preprocess assigns a trajectory to a block by its start time, so none straddle).
    traj_user = df.groupby("traj_id", sort=False)["user_idx"].first()
    k = traj_user.groupby(traj_user).cumcount()
    gn = traj_user.astype(str) + "_" + k.astype(str)
    df["gn_traj_id"] = df["traj_id"].map(gn)
    assert df["gn_traj_id"].notna().all()
    # every re-minted id must resolve back to its own user, or GETNext silently mis-attributes
    chk = df["gn_traj_id"].str.split("_").str[0].astype(int)
    assert (chk == df["user_idx"]).all(), "trajectory_id prefix does not match user_idx"

    tr = to_getnext(df[df.block == "T0"])
    va = to_getnext(df[df.block != "T0"])

    d = OUT / NAME[name]
    d.mkdir(parents=True, exist_ok=True)
    tr.to_csv(d / f"{NAME[name]}_train.csv", index=False)
    va.to_csv(d / f"{NAME[name]}_val.csv", index=False)

    tr_u, va_u = set(tr.user_id), set(va.user_id)
    tr_p, va_p = set(tr.POI_id), set(va.POI_id)
    cold_u = len(va_u - tr_u)
    unseen_rows = int((~va.POI_id.isin(tr_p)).sum())
    print(f"[{name}] -> {d}")
    print(f"  train(T0)   {len(tr):>7,} check-ins | {len(tr_u):>5,} users | {len(tr_p):>6,} POIs")
    print(f"  val(T1-T5)  {len(va):>7,} check-ins | {len(va_u):>5,} users | {len(va_p):>6,} POIs")
    print(f"  GETNext's own filtering would DROP: {cold_u} cold users "
          f"({cold_u/max(len(va_u),1):.1%} of val users), "
          f"{unseen_rows:,} rows whose POI is unseen in train ({unseen_rows/len(va):.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    args = ap.parse_args()
    names = list(NAME) if args.dataset == "all" else [args.dataset]
    for n in names:
        export(n)


if __name__ == "__main__":
    main()
