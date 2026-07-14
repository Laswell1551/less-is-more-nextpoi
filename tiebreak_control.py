#!/usr/bin/env python3
"""
tiebreak_control.py -- is the language model's contribution actually just TIE-BREAKING?

THE SETUP
Fusing the 7B model's scores into the counter at a small weight lifts Acc@10 from .6202 (the
counter alone) to .6263 at alpha=0.05. Taken at face value, that says a fine-tuned 7B language
model carries information the counter lacks, worth +1.0%.

THE SUSPICION
The counter's score over the 50 candidates is a per-user visit count -- an INTEGER. Candidates
the user has never visited all score exactly 0 and are exactly tied. Any continuous signal added
on top, however uninformative, will break those ties and can only help. So the +1.0% might be
worth nothing at all: it might be what ANY tie-breaker would buy.

THE CONTROL
Replace the language model's scores with pure noise -- a uniform random vector, containing zero
information about the target -- and sweep the same alpha. If random noise reproduces the +1.0%,
then the language model contributed nothing, and the honest claim is the one we originally made:
its accuracy-optimal contribution is zero.

We also try a popularity tie-break (informative, but free) as a middle case.

This is exactly the control this paper argues the field should run, so we run it on ourselves.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
N_SEEDS = 5


def expected_rank(f, t):
    st = f[t]
    return int((f > st).sum()) + (int((f == st).sum()) + 1) / 2.0


def sweep(ev, S_alt, label, n=None):
    """S_alt[i] is the score vector to fuse in place of the LLM's, for instance i."""
    n = n or len(ev)
    out = {}
    for a in ALPHAS:
        ranks = []
        for s, e in zip(S_alt, ev):
            if e["tgt_pos"] < 0:
                ranks.append(None)
                continue
            m = np.asarray(e["mem"], dtype=np.float64)
            p_mem = m / (m.sum() + 1e-9)
            p_alt = np.exp(s - s.max()); p_alt /= p_alt.sum() + 1e-12
            f = a * p_alt + (1 - a) * p_mem
            ranks.append(expected_rank(f, e["tgt_pos"]))
        r = [x for x in ranks if x is not None]
        out[a] = sum(1 for x in r if x <= 10) / n
    best_a = max(out, key=lambda k: out[k])
    print(f"  {label:34s} " + "  ".join(f"{a:.2f}:{out[a]:.4f}" for a in ALPHAS))
    print(f"  {'':34s} best alpha={best_a:.2f}  Acc@10={out[best_a]:.4f}  "
          f"lift over counter-alone = {(out[best_a]/out[0.0]-1)*100:+.2f}%")
    return out, best_a


def main():
    ev = [json.loads(l) for l in open(ROOT / "llm_eval_nyc.jsonl", encoding="utf-8")]
    sf = ROOT / "llm_scores_static_nyc.jsonl"
    S_llm = [np.asarray(json.loads(l), dtype=np.float64) for l in open(sf, encoding="utf-8")]
    assert len(S_llm) == len(ev)
    n_cand = len(S_llm[0])
    print(f"[nyc] {len(ev)} instances, {n_cand} candidates\n")

    print("A. THE REAL LANGUAGE MODEL (Qwen2.5-7B, LoRA-T0):")
    llm, _ = sweep(ev, S_llm, "LLM scores")

    print("\nB. THE CONTROL -- PURE NOISE (zero information about the target):")
    lifts = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        S_rand = [rng.standard_normal(n_cand) for _ in ev]
        r, ba = sweep(ev, S_rand, f"random noise (seed {seed})")
        lifts.append(r[ba] / r[0.0] - 1)

    print("\nC. A FREE, INFORMATIVE TIE-BREAK -- global popularity:")
    pop = np.zeros(n_cand)
    S_pop = []
    for e in ev:
        # candidate popularity is already available to the retriever; use it as the alt score
        S_pop.append(np.asarray(e.get("cand_pop", [0.0] * n_cand), dtype=np.float64))
    has_pop = any(s.any() for s in S_pop)
    if has_pop:
        sweep(ev, S_pop, "popularity")
    else:
        print("  (no cand_pop field in llm_eval_nyc.jsonl -- skipped)")

    print("\n" + "=" * 78)
    m = float(np.mean(lifts)) * 100
    s = float(np.std(lifts)) * 100
    llm_lift = (max(llm.values()) / llm[0.0] - 1) * 100
    print(f"  The language model's best lift over the counter alone : {llm_lift:+.2f}%")
    print(f"  PURE NOISE's best lift over the counter alone         : {m:+.2f}% +- {s:.2f}% "
          f"({N_SEEDS} seeds)")
    print()
    if llm_lift <= m + 2 * s:
        print("  => The language model's contribution is NOT distinguishable from a random")
        print("     tie-breaker. It carries no usable information beyond breaking the counter's")
        print("     ties. The accuracy-optimal contribution of the 7B model is, in substance, ZERO.")
    else:
        print("  => The language model beats a random tie-breaker. Its contribution is real,")
        print(f"     though small: {llm_lift:+.2f}% against {m:+.2f}% for noise.")

    (ROOT / "tiebreak_control.json").write_text(json.dumps({
        "llm_sweep": {str(k): v for k, v in llm.items()},
        "llm_best_lift_pct": round(llm_lift, 3),
        "noise_best_lift_pct_mean": round(m, 3),
        "noise_best_lift_pct_sd": round(s, 3),
        "n_seeds": N_SEEDS,
    }, indent=2))
    print("\n-> tiebreak_control.json")


if __name__ == "__main__":
    main()
