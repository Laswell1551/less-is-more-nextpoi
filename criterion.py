#!/usr/bin/env python3
"""A-priori freeze-vs-fine-tune criterion (the constructive product, referee-driven).
Tests whether a CHEAP observable computable WITHOUT running fine-tuning---the new-POI
rate (fraction of stream targets that are POIs unseen in the base block)---predicts the
realized FT-minus-static gap across all (dataset x base_frac) conditions. If it does,
'fine-tune iff new-POI rate > tau' is an a-priori decision rule, turning the paper's
post-hoc 'predictive axis' into a computable criterion.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
J = ROOT / "regime_sweep.json"
if not J.exists():
    print("regime_sweep.json not found yet -- run regime_sweep.py first.")
    raise SystemExit

d = json.load(open(J))
rows = sorted(d.values(), key=lambda r: r["new_poi"])
x = np.array([r["new_poi"] for r in rows])               # cheap a-priori observable
y = np.array([r["FT_minus_static"] for r in rows])       # realized gap (needs FT to know)

# correlation + linear fit y ~ a*x + b
r = float(np.corrcoef(x, y)[0, 1])
a, b = np.polyfit(x, y, 1)
yhat = a * x + b
ss = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
tau = float(-b / a) if a != 0 else float("nan")          # new-POI crossover where gap=0

# decision-rule accuracy: predict 'fine-tune' iff new_poi > tau; correct iff matches sign(gap)
pred = x > tau
true = y > 0
acc = float(np.mean(pred == true))

print(f"conditions: {len(rows)}")
print(f"corr(new-POI, FT-static gap) = {r:+.3f}   linear R^2 = {ss:.3f}")
print(f"gap ~ {a:+.3f}*new_poi {b:+.3f}   ->  crossover tau = {tau:.3f}")
print(f"decision rule 'fine-tune iff new-POI > {tau:.3f}' agrees with realized winner in "
      f"{acc*100:.0f}% of conditions")
print("\n new_poi  FT-static  winner")
for rr in rows:
    print(f"  {rr['new_poi']:.3f}   {rr['FT_minus_static']:+.4f}   "
          f"{'fine-tune' if rr['FT_minus_static']>0 else 'freeze'}   ({rr['dataset']}@{rr['base_frac']})")
json.dump({"corr": round(r, 3), "r2": round(ss, 3), "slope": round(a, 3),
           "intercept": round(b, 3), "tau": round(tau, 3), "rule_acc": round(acc, 3)},
          open(ROOT / "criterion.json", "w"), indent=2)
