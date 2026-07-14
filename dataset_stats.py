"""Descriptive statistics for the Data section. Computed on the released splits."""
import json
import numpy as np, pandas as pd
from pathlib import Path

PROC = Path("data/processed")
NAMES = {"nyc": "Foursquare-NYC", "tky": "Foursquare-TKY",
         "gowalla_ca": "Gowalla-CA", "brightkite": "Brightkite-US"}
rows = []
for k, disp in NAMES.items():
    df = pd.read_csv(PROC / k / "checkins.csv.gz").sort_values(["user_idx", "unix_s"])
    n_u, n_p, n_c = df.user_idx.nunique(), df.poi_idx.nunique(), len(df)
    span = (df.unix_s.max() - df.unix_s.min()) / 86400
    start = pd.to_datetime(df.unix_s.min(), unit="s").strftime("%Y-%m")
    end = pd.to_datetime(df.unix_s.max(), unit="s").strftime("%Y-%m")

    # revisit rate: share of check-ins to a POI this user has visited before
    seen, rev = set(), 0
    for u, p in zip(df.user_idx.to_numpy(), df.poi_idx.to_numpy()):
        if (u, p) in seen:
            rev += 1
        seen.add((u, p))
    # self-transition rate (consecutive same POI) -- the artifact dedup removes
    same = ((df.poi_idx.to_numpy()[1:] == df.poi_idx.to_numpy()[:-1]) &
            (df.user_idx.to_numpy()[1:] == df.user_idx.to_numpy()[:-1])).mean()

    t0_u = set(df[df.block == "T0"].user_idx.unique())
    te_u = set(df[df.block != "T0"].user_idx.unique())
    cold = len(te_u - t0_u) / max(len(te_u), 1)
    t0_p = set(df[df.block == "T0"].poi_idx.unique())
    new_poi = (~df[df.block != "T0"].poi_idx.isin(t0_p)).mean()

    rows.append({
        "dataset": disp, "users": n_u, "pois": n_p, "checkins": n_c,
        "trajectories": df.traj_id.nunique(),
        "avg_traj_len": round(n_c / df.traj_id.nunique(), 2),
        "density_pct": round(100 * n_c / (n_u * n_p), 3),
        "span_days": int(span), "window": f"{start}..{end}",
        "categories": int(df.cat.nunique()),
        "revisit_rate": round(rev / n_c, 3),
        "self_transition": round(float(same), 3),
        "T0_checkins": int((df.block == "T0").sum()),
        "cold_user_share_post_T0": round(cold, 3),
        "new_poi_rate_post_T0": round(float(new_poi), 3),
    })

R = pd.DataFrame(rows)
print(R.to_string(index=False))
print("\nTOTALS: users=%d  POIs=%d  check-ins=%d  trajectories=%d" %
      (R.users.sum(), R.pois.sum(), R.checkins.sum(), R.trajectories.sum()))
Path("dataset_stats.json").write_text(json.dumps(rows, indent=2))
print("-> dataset_stats.json")
