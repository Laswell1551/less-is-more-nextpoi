#!/usr/bin/env python3
"""
mechanism_demo.py -- early controlled mechanism demo on a transparent Markov backbone.
(Exploratory / superseded by the learned-backbone experiments; kept for transparency.)

Online continual next-POI on a transparent personalized-Markov backbone. We stream
T1..T5 in R chronological rounds under a strict predict-then-update protocol, and
compare four "when to update" policies:

  static     : never update after T0                       (update_rate = 0)
  always     : every active user updates every round       (update_rate = 1, upper acc)
  periodic-N : every active user updates every N-th round   (naive lazy baseline)
  CALM       : user updates only when its novelty (fraction of *unseen* transitions
               this round) exceeds threshold tau           (change-gated lazy update)

Metrics: Acc@10, MRR (accuracy) | update_rate (fraction of user-rounds updated)
         | churn (round-to-round Jaccard change of a user's top-10 for a fixed
           probe context -- recommendation instability *induced by updating*).

The thesis: CALM reaches ~always accuracy at a fraction of the update_rate and
churn -- cashing in the "~70% of updates don't change the decision" observation.

CPU only. Run after preprocess.py.  ->  results_mechanism.csv + figs/mechanism_pareto.png
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "figs"
DATASETS = ["nyc", "tky", "gowalla_ca"]
COLORS = {"nyc": "#1f77b4", "tky": "#d62728", "gowalla_ca": "#2ca02c"}
K = 10
R = 15                       # streaming rounds over T1..T5
TAU_GRID = [0.10, 0.20, 0.30, 0.45, 0.60, 0.80]   # novelty gate
THETA_GRID = [0.20, 0.35, 0.50, 0.65, 0.80]       # error/drift gate (recent per-user acc)
PERIODIC = [2, 3, 4, 6]
GTOP = 50                    # global-popularity backoff depth


def load(name):
    return pd.read_csv(PROC / name / "checkins.csv.gz").sort_values(
        ["user_idx", "unix_s"]).reset_index(drop=True)


def build_T0(df0):
    trans = defaultdict(lambda: defaultdict(Counter))
    upop = defaultdict(Counter)
    gpop = Counter()
    u = df0.user_idx.to_numpy(); p = df0.poi_idx.to_numpy(); t = df0.traj_id.to_numpy()
    for i in range(len(df0)):
        gpop[p[i]] += 1; upop[u[i]][p[i]] += 1
        if i and t[i] == t[i - 1] and u[i] == u[i - 1]:
            trans[u[i]][p[i - 1]][p[i]] += 1
    return {"trans": trans, "upop": upop, "gpop": gpop}


def predict(m, u, last, gtop, k=K):
    s = {}
    tl = m["trans"].get(u, {}).get(last)
    if tl:
        for q, c in tl.items():
            s[q] = s.get(q, 0) + c * 1e12
    up = m["upop"].get(u)
    if up:
        for q, c in up.items():
            s[q] = s.get(q, 0) + c * 1e6
    for q, c in gtop:
        s[q] = s.get(q, 0) + c
    return [q for q, _ in sorted(s.items(), key=lambda x: -x[1])[:k]]


def fold(m, u, pairs):
    for last, nxt in pairs:
        m["trans"][u][last][nxt] += 1
        m["upop"][u][nxt] += 1
        m["gpop"][nxt] += 1


def test_transitions(df):
    u = df.user_idx.to_numpy(); p = df.poi_idx.to_numpy()
    t = df.traj_id.to_numpy(); ts = df.unix_s.to_numpy()
    out = [(u[i], p[i - 1], p[i], ts[i]) for i in range(1, len(df))
           if t[i] == t[i - 1] and u[i] == u[i - 1]]
    out.sort(key=lambda x: x[3])
    return out


def make_rounds(trans, R):
    chunks = np.array_split(np.arange(len(trans)), R)
    rounds_eval, rounds_user = [], []
    for idx in chunks:
        ev = [(trans[i][0], trans[i][1], trans[i][2]) for i in idx]
        by = defaultdict(list)
        for u, last, nxt in ev:
            by[u].append((last, nxt))
        rounds_eval.append(ev); rounds_user.append(by)
    return rounds_eval, rounds_user


def probe_context(model0):
    """Each user's fixed probe = its most frequent 'last' POI in T0 transitions."""
    probe = {}
    for u, lasts in model0["trans"].items():
        probe[u] = max(lasts, key=lambda l: sum(lasts[l].values()))
    return probe


# decision functions: (model, u, pairs, r, ema) -> bool
def dec_static(m, u, pairs, r, ema): return False
def dec_always(m, u, pairs, r, ema): return True
def dec_periodic(N):
    return lambda m, u, pairs, r, ema: (r % N == 0)
def dec_calm(tau):                       # gate on novelty (fraction of unseen transitions)
    def f(m, u, pairs, r, ema):
        tl = m["trans"].get(u, {})
        nov = sum(1 for last, nxt in pairs if nxt not in tl.get(last, {})) / len(pairs)
        return nov > tau
    return f
def dec_error(theta):                    # gate on recent per-user accuracy (drift-triggered)
    return lambda m, u, pairs, r, ema: ema.get(u, 0.0) < theta


def run_policy(df0, rounds_eval, rounds_user, probe, decide):
    m = build_T0(df0)
    hit = rr = n = upd = opp = 0
    churn_sum = churn_n = 0
    prev = {}; ema = {}
    for r in range(len(rounds_eval)):
        gtop = m["gpop"].most_common(GTOP)
        rhit = defaultdict(lambda: [0, 0])
        for u, last, nxt in rounds_eval[r]:               # predict (stale)
            tk = predict(m, u, last, gtop)
            rhit[u][1] += 1
            if nxt in tk:
                hit += 1; rr += 1.0 / (tk.index(nxt) + 1); rhit[u][0] += 1
            n += 1
        for u, (h, c) in rhit.items():                    # per-user recent accuracy (EMA, observed feedback)
            a = h / c
            ema[u] = a if u not in ema else 0.5 * ema[u] + 0.5 * a
        for u, pairs in rounds_user[r].items():           # then update
            opp += 1
            if decide(m, u, pairs, r, ema):
                fold(m, u, pairs); upd += 1
        gtop = m["gpop"].most_common(GTOP)                # churn probe (post-update)
        for u in rounds_user[r]:
            pl = probe.get(u)
            if pl is None:
                continue
            tk = set(predict(m, u, pl, gtop))
            if u in prev:
                churn_sum += 1 - len(tk & prev[u]) / max(len(tk | prev[u]), 1)
                churn_n += 1
            prev[u] = tk
    return {"acc@10": hit / max(n, 1), "mrr": rr / max(n, 1),
            "update_rate": upd / max(opp, 1), "churn": churn_sum / max(churn_n, 1)}


def run_dataset(name):
    df = load(name)
    df0 = df[df.block == "T0"]
    dft = df[df.block != "T0"]
    rounds_eval, rounds_user = make_rounds(test_transitions(dft), R)
    probe = probe_context(build_T0(df0))
    policies = [("static", dec_static), ("always", dec_always)]
    policies += [(f"periodic-{N}", dec_periodic(N)) for N in PERIODIC]
    policies += [(f"CALMnov@{tau}", dec_calm(tau)) for tau in TAU_GRID]
    policies += [(f"CALMerr@{th}", dec_error(th)) for th in THETA_GRID]
    rows = []
    for label, dec in policies:
        res = run_policy(df0, rounds_eval, rounds_user, probe, dec)
        res.update(dataset=name, policy=label)
        rows.append(res)
        print(f"  {label:12s} acc@10={res['acc@10']:.3f} mrr={res['mrr']:.3f} "
              f"upd_rate={res['update_rate']:.2f} churn={res['churn']:.3f}")
    return rows


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    allrows = []
    for n in DATASETS:
        print(f"[{n}]")
        allrows += run_dataset(n)
    df = pd.DataFrame(allrows)[["dataset", "policy", "acc@10", "mrr", "update_rate", "churn"]]
    df.to_csv(ROOT / "results_mechanism.csv", index=False)

    # Pareto: acc vs update_rate (CALM curve), churn as marker size
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    head = {}
    for j, n in enumerate(DATASETS):
        d = df[df.dataset == n]
        always_acc = float(d[d.policy == "always"]["acc@10"].iloc[0])
        always_churn = float(d[d.policy == "always"]["churn"].iloc[0])
        for pref, ls, fill in [("CALMnov", "-o", COLORS[n]), ("CALMerr", "-^", "none")]:
            c = d[d.policy.str.startswith(pref)].sort_values("update_rate")
            if len(c):
                ax[j].plot(c["update_rate"], c["acc@10"], ls, color=COLORS[n],
                           mfc=fill, label=pref)
        peri = d[d.policy.str.startswith("periodic-")].sort_values("update_rate")
        ax[j].plot(peri["update_rate"], peri["acc@10"], ":sk", lw=0.8, mfc="none", label="periodic")
        for pol, mk in [("static", "x"), ("always", "*")]:
            row = d[d.policy == pol]
            if len(row):
                ax[j].scatter(row["update_rate"], row["acc@10"], marker=mk, s=80, c="k", label=pol)
        ax[j].axhline(always_acc, ls="--", c="grey", lw=0.8)
        ax[j].set_title(n); ax[j].set_xlabel("update rate"); ax[j].set_ylabel("Acc@10")
        ax[j].legend(fontsize=7)
        # sweet spot: smallest-update CALM reaching >=97% of always acc
        calm = d[d.policy.str.startswith("CALM")]
        ok = calm[calm["acc@10"] >= 0.97 * always_acc]
        if len(ok):
            sp = ok.sort_values("update_rate").iloc[0]
            head[n] = {"always_acc": round(always_acc, 4),
                       "calm_tau": sp["policy"], "calm_acc": round(float(sp["acc@10"]), 4),
                       "calm_update_rate": round(float(sp["update_rate"]), 3),
                       "calm_churn": round(float(sp["churn"]), 3),
                       "always_churn": round(always_churn, 3),
                       "churn_reduction": round(1 - float(sp["churn"]) / max(always_churn, 1e-9), 3)}
    plt.tight_layout(); plt.savefig(FIGS / "mechanism_pareto.png", dpi=160); plt.close()
    json.dump(head, open(ROOT / "mechanism_sweetspot.json", "w"), indent=2)
    print("\n=== CALM sweet spot (>=97% of always acc) ===")
    for n, h in head.items():
        print(f"  {n}: acc {h['calm_acc']} (always {h['always_acc']}) at update_rate "
              f"{h['calm_update_rate']} | churn {h['calm_churn']} vs {h['always_churn']} "
              f"(-{h['churn_reduction']*100:.0f}%)  [{h['calm_tau']}]")
    print(f"\nresults -> results_mechanism.csv | figure -> figs/mechanism_pareto.png")


if __name__ == "__main__":
    main()
