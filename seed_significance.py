"""5-seed summary + paired significance for the corrected chronological grid.

COUNT is deterministic (no seed variance), so it is compared against each neural method with a
one-sample t-test on the per-seed differences rather than a paired two-sample test.
"""
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(".")
DS = ["nyc", "tky", "gowalla_ca", "brightkite"]
DISP = {"nyc": "FSQ-NYC", "tky": "FSQ-TKY", "gowalla_ca": "Gowalla-CA", "brightkite": "Brightkite-US"}
# acc@10m == the per-instance mean with expected-rank ties (ranking.py). This is the one
# estimator the whole paper uses; see the note above stream_table() in gen_tables.py. The
# `acc@10` column is a per-ROUND macro average with torch.topk (index-order) tie-breaking,
# and quoting it here while Protocol A uses the expected rank would put two estimators in one
# paper. Neural rows are identical under both; only the counter, whose integer scores tie, moves.
ACC = "acc@10m"
cnt = pd.read_csv("results_chrono_count.csv").set_index("dataset")[ACC].to_dict()
ORDER = ["static", "periodic-4", "selective-gated", "always+replay", "EWC", "ADER",
         "GIRAM-VAE", "GIRAM"]

for d in DS:
    R = pd.read_csv(f"results_chrono_{d}_gru.csv")
    g = R.groupby("policy")[ACC]
    st = R[R.policy == "static"].sort_values("seed")[ACC].to_numpy()
    print(f"\n=== {DISP[d]} (5 seeds) ===")
    print(f"  {'method':17s} {'Acc@10':>15s}  {'vs static':>10s}  {'p':>9s}   {'vs COUNT':>10s}  {'p':>9s}")
    C = cnt[d]
    for p in ORDER:
        if p not in g.groups:
            continue
        v = R[R.policy == p].sort_values("seed")[ACC].to_numpy()
        m, s = v.mean(), v.std(ddof=1)
        if p == "static":
            ps = pv = float("nan"); ds_ = 0.0
        else:
            ds_ = m - st.mean()
            pv = stats.ttest_rel(v, st).pvalue
        # vs COUNT: COUNT is a constant, so a one-sample test on (method - COUNT)
        dc = m - C
        pc = stats.ttest_1samp(v - C, 0.0).pvalue
        print(f"  {p:17s} {m:.4f} +/- {s:.4f}  {ds_:+10.4f}  {pv:9.2e}   {dc:+10.4f}  {pc:9.2e}")
    print(f"  {'COUNT (no NN)':17s} {C:.4f} (deterministic)")
