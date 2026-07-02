#!/usr/bin/env python3
"""
preprocess.py -- raw check-ins -> canonical Next-POI trajectories + continual blocks.

Pipeline per dataset:
  parse -> (Gowalla) California bbox filter -> iterative k-core filtering
  -> gap-based trajectory segmentation -> contiguous re-indexing
  -> chronological block split (GIRAM-style) -> write canonical csv.gz + id maps + stats.

Block split (matches GIRAM'25 so our continual results are directly comparable):
  T0 (base) = first 50% of the global timeline; T1..T5 = remaining 50% in 5 equal
  chronological slices. Each trajectory is assigned by its START time, so no
  trajectory straddles a block boundary.

Canonical columns (data/processed/<name>/checkins.csv.gz):
  user_idx, poi_idx, cat, lat, lon, unix_s, hour, dow, traj_id, block

Usage:
  python preprocess.py --dataset all
  python preprocess.py --dataset nyc --min-user 10 --min-poi 10 --gap-hours 24
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# (lat_min, lat_max, lon_min, lon_max) for the Gowalla California subset (~GETNext/GIRAM)
CA_BBOX = (32.5, 42.0, -124.5, -114.0)

CONFIG = {
    "nyc":        {"src": "foursquare", "file": "dataset_TSMC2014_NYC.txt"},
    "tky":        {"src": "foursquare", "file": "dataset_TSMC2014_TKY.txt"},
    "gowalla_ca": {"src": "gowalla",    "file": "loc-gowalla_totalCheckins.txt",
                   "bbox": CA_BBOX, "tz_off_min": -480},  # approx PST for hour/dow
    "brightkite": {"src": "brightkite", "file": "loc-brightkite_totalCheckins.txt",
                   "bbox": (24.0, 50.0, -125.0, -66.0), "tz_off_min": -360},  # continental US
}


# --------------------------------------------------------------------------- parse
def _to_unix_and_local(ts_utc, off_min):
    """ts_utc: tz-aware UTC Series. Returns (unix_s int64, local-naive datetime)."""
    naive_utc = ts_utc.dt.tz_convert("UTC").dt.tz_localize(None)      # naive wall-clock UTC
    unix_s = (naive_utc.astype("int64") // 10**9)
    local = naive_utc + pd.to_timedelta(off_min, unit="m")           # off_min: Series or scalar
    return unix_s, local


def parse_foursquare(path):
    cols = ["user", "poi", "cat_id", "cat", "lat", "lon", "tz_off", "utc"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols,
                     encoding="latin-1", quoting=3)
    ts = pd.to_datetime(df["utc"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    unix_s, local = _to_unix_and_local(ts, df["tz_off"].astype("int64"))
    out = pd.DataFrame({
        "user": df["user"].astype(str), "poi": df["poi"].astype(str),
        "cat": df["cat"].astype(str), "lat": df["lat"], "lon": df["lon"],
        "unix_s": unix_s, "hour": local.dt.hour, "dow": local.dt.dayofweek,
    })
    return out


def parse_gowalla(path, bbox=None, tz_off_min=0):
    cols = ["user", "time", "lat", "lon", "poi"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()]
    if bbox:
        la0, la1, lo0, lo1 = bbox
        m = (df.lat >= la0) & (df.lat <= la1) & (df.lon >= lo0) & (df.lon <= lo1)
        df, ts = df[m], ts[m]
    unix_s, local = _to_unix_and_local(ts, tz_off_min)
    out = pd.DataFrame({
        "user": df["user"].astype(str), "poi": df["poi"].astype(str),
        "cat": "NA", "lat": df["lat"], "lon": df["lon"],
        "unix_s": unix_s.values, "hour": local.dt.hour.values, "dow": local.dt.dayofweek.values,
    })
    return out


# ------------------------------------------------------------------ clean / segment
def kcore_filter(df, min_poi, min_user):
    while True:
        n0 = len(df)
        pf = df["poi"].value_counts()
        df = df[df["poi"].isin(pf.index[pf >= min_poi])]
        uf = df["user"].value_counts()
        df = df[df["user"].isin(uf.index[uf >= min_user])]
        if len(df) == n0:
            return df


def segment_trajectories(df, gap_hours, min_len):
    df = df.sort_values(["user", "unix_s"]).reset_index(drop=True)
    gap = df.groupby("user")["unix_s"].diff()
    new = (gap.isna()) | (gap > gap_hours * 3600)
    df["traj_seq"] = new.groupby(df["user"]).cumsum().astype(int)
    df["traj_id"] = df["user"] + "_" + df["traj_seq"].astype(str)
    keep = df.groupby("traj_id")["poi"].transform("size") >= min_len
    return df[keep].copy()


def assign_blocks(df, base_frac=0.5, n_incr=5):
    tstart = df.groupby("traj_id")["unix_s"].transform("min").to_numpy()
    qs = np.quantile(tstart, [base_frac + (1 - base_frac) * i / n_incr for i in range(n_incr + 1)])
    conds = [tstart <= qs[0]]
    labels = ["T0"]
    for i in range(n_incr):
        conds.append((tstart > qs[i]) & (tstart <= qs[i + 1]))
        labels.append(f"T{i+1}")
    df["block"] = np.select(conds, labels, default=f"T{n_incr}")
    return df


def reindex(df):
    umap = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    pmap = {p: i for i, p in enumerate(sorted(df["poi"].unique()))}
    df["user_idx"] = df["user"].map(umap).astype(int)
    df["poi_idx"] = df["poi"].map(pmap).astype(int)
    return df, umap, pmap


# ------------------------------------------------------------------------- stats
def compute_stats(name, df, n_raw, params):
    n_ck, n_u, n_p = len(df), df.user_idx.nunique(), df.poi_idx.nunique()
    n_tr = df.traj_id.nunique()
    d = df.sort_values(["user_idx", "unix_s"]).reset_index(drop=True)
    same = (d.user_idx.values[1:] == d.user_idx.values[:-1]) & \
           (d.traj_id.values[1:] == d.traj_id.values[:-1])
    self_trans = float(((d.poi_idx.values[1:] == d.poi_idx.values[:-1]) & same).sum() / max(same.sum(), 1))
    revisit = float(d.duplicated(subset=["user_idx", "poi_idx"]).mean())   # not user's first visit
    top5 = float(d.groupby("user_idx")["poi_idx"]
                 .apply(lambda s: s.value_counts().head(5).sum() / len(s)).mean())
    by_block = (df.groupby("block")
                .agg(checkins=("poi_idx", "size"), trajectories=("traj_id", "nunique"),
                     users=("user_idx", "nunique"), pois=("poi_idx", "nunique"))
                .reindex([f"T{i}" for i in range(6)]).fillna(0).astype(int))
    return {
        "dataset": name, "params": params,
        "raw_checkins": int(n_raw), "checkins": int(n_ck),
        "users": int(n_u), "pois": int(n_p), "trajectories": int(n_tr),
        "avg_traj_len": round(n_ck / max(n_tr, 1), 3),
        "density_pct": round(100 * n_ck / max(n_u * n_p, 1), 5),
        "span_days": round((df.unix_s.max() - df.unix_s.min()) / 86400, 1),
        # R2 decision-sparsity teasers (model-free):
        "self_transition_rate": round(self_trans, 4),
        "revisit_rate": round(revisit, 4),
        "top5_poi_coverage": round(top5, 4),
        "by_block": by_block.to_dict("index"),
    }


def print_stats(s):
    print(f"\n=== {s['dataset']} ===")
    print(f"  raw->kept check-ins : {s['raw_checkins']:,} -> {s['checkins']:,}")
    print(f"  users / POIs / traj : {s['users']:,} / {s['pois']:,} / {s['trajectories']:,}")
    print(f"  avg traj len        : {s['avg_traj_len']}   density {s['density_pct']}%   span {s['span_days']}d")
    print(f"  R2 teasers          : self-transition {s['self_transition_rate']:.3f} | "
          f"revisit {s['revisit_rate']:.3f} | top5-coverage {s['top5_poi_coverage']:.3f}")
    print(f"  blocks (checkins)   : " +
          "  ".join(f"{k}={v['checkins']:,}" for k, v in s["by_block"].items()))


# ------------------------------------------------------------------------- driver
def process(name, args):
    cfg = CONFIG[name]
    path = RAW / cfg["file"]
    if not path.exists():
        print(f"[{name}] raw missing: {path}  (run download_data.py first)", file=sys.stderr)
        return None
    print(f"[{name}] parsing {path.name} ...")
    if cfg["src"] == "foursquare":
        df = parse_foursquare(path)
    else:
        df = parse_gowalla(path, cfg.get("bbox"), cfg.get("tz_off_min", 0))
    n_raw = len(df)
    # collapse consecutive duplicate POIs within each user (standard next-POI practice):
    # a user repeatedly checking in at the same place is one visit, not a sequence of transitions.
    df = df.sort_values(["user", "unix_s"]).reset_index(drop=True)
    uu = df["user"].to_numpy(); pp = df["poi"].to_numpy()
    dup = np.concatenate([[False], (uu[1:] == uu[:-1]) & (pp[1:] == pp[:-1])])
    df = df[~dup].reset_index(drop=True)
    df = kcore_filter(df, args.min_poi, args.min_user)
    if df.empty:
        print(f"[{name}] empty after filtering -- loosen thresholds", file=sys.stderr)
        return None
    df = segment_trajectories(df, args.gap_hours, args.min_traj)
    df = assign_blocks(df)
    df, umap, pmap = reindex(df)
    df = df.sort_values(["user_idx", "unix_s"]).reset_index(drop=True)

    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["user_idx", "poi_idx", "cat", "lat", "lon", "unix_s", "hour", "dow", "traj_id", "block"]
    df[cols].to_csv(out_dir / "checkins.csv.gz", index=False, compression="gzip")
    json.dump(umap, open(out_dir / "user2idx.json", "w"))
    json.dump(pmap, open(out_dir / "poi2idx.json", "w"))
    params = {"min_poi": args.min_poi, "min_user": args.min_user,
              "gap_hours": args.gap_hours, "min_traj": args.min_traj}
    stats = compute_stats(name, df, n_raw, params)
    json.dump(stats, open(out_dir / "stats.json", "w"), indent=2)
    print_stats(stats)
    print(f"  -> {out_dir/'checkins.csv.gz'}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                    choices=["all"] + list(CONFIG))
    ap.add_argument("--min-poi", type=int, default=10, dest="min_poi")
    ap.add_argument("--min-user", type=int, default=10, dest="min_user")
    ap.add_argument("--gap-hours", type=int, default=24, dest="gap_hours")
    ap.add_argument("--min-traj", type=int, default=2, dest="min_traj")
    args = ap.parse_args()

    names = list(CONFIG) if args.dataset == "all" else [args.dataset]
    summary = {}
    for nm in names:
        s = process(nm, args)
        if s:
            summary[nm] = {k: s[k] for k in
                           ["users", "pois", "trajectories", "checkins",
                            "self_transition_rate", "revisit_rate", "top5_poi_coverage"]}
    if summary:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
        print(f"\nsummary -> {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
