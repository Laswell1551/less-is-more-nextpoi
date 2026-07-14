#!/usr/bin/env python3
"""
llm_alpha_sweep.py -- does ANY fusion weight let the LLM improve on the counter?

The obvious objection to "fusing the fine-tuned 7B with the counter makes it worse" is that
we picked the fusion weight. It is a fair objection and it cannot be waved away, because
    p_fused(alpha) = alpha * p_llm + (1 - alpha) * p_memory
recovers the pure counter exactly at alpha = 0. So max_alpha (fused) >= counter BY
CONSTRUCTION, and a single alpha proves nothing. The only thing that settles whether the
language model contributes anything is the whole curve.

This runs entirely offline from the raw per-instance LLM scores dumped by llm_audit
(llm_scores_static_<ds>.jsonl) and the memory scores already in llm_eval_<ds>.jsonl, so the
sweep is free.

  alpha = 0  -> the counter alone
  alpha = 1  -> the LLM alone
  argmax     -> what the LLM is actually worth

Usage:  python llm_alpha_sweep.py --dataset nyc
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    args = ap.parse_args()

    ev = [json.loads(l) for l in open(ROOT / f"llm_eval_{args.dataset}.jsonl",
                                      encoding="utf-8")]
    sf = ROOT / f"llm_scores_static_{args.dataset}.jsonl"
    if not sf.exists():
        raise SystemExit(f"{sf.name} not found -- re-run llm_audit.py with --variants static "
                         f"(it now dumps the raw scores)")
    S = [np.asarray(json.loads(l), dtype=np.float64) for l in open(sf, encoding="utf-8")]
    assert len(S) == len(ev), f"{len(S)} score rows vs {len(ev)} instances"

    alphas = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0]
    n = len(ev)
    print(f"[{args.dataset}] {n} instances, {len(S[0])} candidates\n")
    print(f"  {'alpha':>6}  {'Acc@1':>7} {'Acc@5':>7} {'Acc@10':>7} {'MRR':>7}   note")

    best = (-1, None)
    for a in alphas:
        ranks = []
        for s, e in zip(S, ev):
            if e["tgt_pos"] < 0:
                ranks.append(None)                      # retriever miss: nobody can hit it
                continue
            p_llm = np.exp(s - s.max()); p_llm /= p_llm.sum() + 1e-12
            m = np.asarray(e["mem"], dtype=np.float64)
            p_mem = m / (m.sum() + 1e-9)
            f = a * p_llm + (1 - a) * p_mem
            # expected rank under random tie-breaking. At alpha=0 the score IS the
            # per-user count, so every unvisited candidate ties at 0; the optimistic
            # convention would place the target first among them. See ranking.py.
            _s = f[e["tgt_pos"]]
            ranks.append(int((f > _s).sum()) + (int((f == _s).sum()) + 1) / 2.0)
        r = [x for x in ranks if x is not None]
        acc = lambda k: sum(1 for x in r if x <= k) / n
        mrr = sum(1.0 / x for x in r) / n
        note = ("<- the counter alone" if a == 0 else
                "<- the LLM alone" if a == 1 else "")
        print(f"  {a:6.2f}  {acc(1):7.4f} {acc(5):7.4f} {acc(10):7.4f} {mrr:7.4f}   {note}")
        if acc(10) > best[0]:
            best = (acc(10), a)

    print()
    a0 = best[1]
    if a0 == 0.0:
        print("  The best fusion weight is ZERO. No amount of the language model's score")
        print("  improves on the counter -- it can only dilute it.")
    else:
        print(f"  Best alpha = {a0} (Acc@10 {best[0]:.4f}); the counter alone is the alpha=0 row.")
        print("  Report this honestly: the LLM does add something at this weight.")

    # ---- revisit vs discovery -----------------------------------------------------------
    # A candidate carries mem > 0 exactly when this user has visited it before, so the memory
    # vector labels each instance for free: the target is a REVISIT if mem[tgt_pos] > 0 and a
    # DISCOVERY otherwise. The discovery column is the one that decides what the benchmark is
    # really measuring -- a counter cannot rank an unvisited POI above a visited one, so if
    # the 7B model cannot either, then nothing in this literature does discovery.
    print("\n  --- revisit vs discovery (the target is somewhere this user has never been) ---")
    print(f"  {'method':28s} {'Acc@10 all':>11s} {'REVISIT':>9s} {'DISCOVERY':>11s}")
    for tag, a in (("Qwen2.5-7B (LoRA-T0), alpha=1", 1.0), ("COUNT, alpha=0", 0.0)):
        hit_all = hit_rev = hit_dis = 0
        n_rev = n_dis = 0
        for s, e in zip(S, ev):
            m = np.asarray(e["mem"], dtype=np.float64)
            rev = e["tgt_pos"] >= 0 and m[e["tgt_pos"]] > 0
            if e["tgt_pos"] >= 0:
                n_rev += int(rev); n_dis += int(not rev)
            else:
                n_dis += 1                      # retriever miss: the user had never been there
            if e["tgt_pos"] < 0:
                continue
            p_llm = np.exp(s - s.max()); p_llm /= p_llm.sum() + 1e-12
            p_mem = m / (m.sum() + 1e-9)
            f = a * p_llm + (1 - a) * p_mem
            _s = f[e["tgt_pos"]]
            hit = int((f > _s).sum()) + (int((f == _s).sum()) + 1) / 2.0 <= 10
            hit_all += hit
            if rev:
                hit_rev += hit
            else:
                hit_dis += hit
        print(f"  {tag:28s} {hit_all/n:11.4f} {hit_rev/max(n_rev,1):9.4f} "
              f"{hit_dis/max(n_dis,1):11.4f}")
    print(f"\n  instances: {n_rev:,} revisit / {n_dis:,} discovery")


if __name__ == "__main__":
    main()
