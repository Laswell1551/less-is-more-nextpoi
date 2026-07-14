#!/usr/bin/env python3
"""
llm_prep.py -- shared causal retriever + non-neural counting baselines + LLM prompt export.

Everything here runs on the CHRONOLOGICAL stream (see run_chrono.py for why the original
user-major ordering was wrong).

Three jobs:

1. SHARED CANDIDATE RETRIEVER. An LLM cannot score 5k-16k POIs per prediction, so the LLM
   comparison is a candidate re-ranking protocol. To keep it fair, ONE causal retriever
   supplies the candidate set and EVERY method (LLM, GRU, memory, popularity) re-ranks the
   SAME set. The retriever uses only data strictly before the current round:
       per-user decayed visit counts (decay 0.9/round)  ->  fills up to 60% of the slots
       global popularity counts                          ->  fills the rest
   We report recall@K, which is the ceiling every method in that protocol shares.

2. NON-NEURAL COUNTING BASELINES, full-catalogue (not candidate-restricted):
       memory-only : rank POIs by the user's decayed visit count, backing off to global
                     popularity. No neural network, no gradients, no embeddings.
       popularity  : rank by global popularity alone.
   These are the "less is more" floor. Under the buggy user-major ordering the per-user
   memory was structurally dead (empty for ~97% of eval points), so this baseline was
   never actually measured.

3. LLM PROMPT EXPORT. Stratified eval subsample across rounds, with the candidate list
   SHUFFLED per example (fixed seed) so that an LLM cannot exploit candidate ordering --
   LLM rankers are known to be position-biased (Hou et al., ECIR 2024). The same shuffled
   set is handed to every method.

Usage:
  python llm_prep.py --dataset nyc --n-eval 1000 --n-cand 50
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import learned_continual as L
from run_chrono import chrono_test_samples

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------- retriever
class CausalRetriever:
    """Per-user decayed visit counts + global popularity. Updated ONLY with observed
    (past) rounds, so it never sees the round it is asked to retrieve for."""

    def __init__(self, n_users, n_pois, decay=0.9, user_frac=0.6):
        # float64: in float32 the eps*popularity tie-break is rounded away above ~8 visits, which
        # changes which POIs collide and therefore which candidates get retrieved.
        # See dtype_control.py and getnext_compare.py.
        self.M = np.zeros((n_users, n_pois), dtype=np.float64)   # per-user decayed counts
        self.pop = np.zeros(n_pois, dtype=np.float64)            # global counts
        self.decay = decay
        self.user_frac = user_frac
        self._pop_order = None            # cached argsort(-pop); invalidated on observe()

    def _order(self):
        if self._pop_order is None:
            self._pop_order = np.argsort(-self.pop, kind="stable")
        return self._pop_order

    def observe(self, users, tgts):
        """Fold one round's observed check-ins into the retriever (called AFTER eval)."""
        self.M *= self.decay
        np.add.at(self.M, (users, tgts), 1.0)
        np.add.at(self.pop, tgts, 1.0)
        self._pop_order = None

    def seed_from(self, users, tgts):
        """Warm the retriever on T0 without decay (T0 is the base period)."""
        np.add.at(self.M, (users, tgts), 1.0)
        np.add.at(self.pop, tgts, 1.0)
        self._pop_order = None

    def candidates(self, u, n_cand):
        """Top-n_cand POI ids for user u: their own history first, then global popularity."""
        n_user = int(n_cand * self.user_frac)
        mu = self.M[u]
        hist = np.flatnonzero(mu)
        if len(hist) > n_user:
            hist = hist[np.argpartition(-mu[hist], n_user)[:n_user]]
            hist = hist[np.argsort(-mu[hist])]
        else:
            hist = hist[np.argsort(-mu[hist])]
        need = n_cand - len(hist)
        if need > 0:
            top_pop = self._order()[: n_cand + len(hist)]      # cached: no per-call sort
            fill = top_pop[~np.isin(top_pop, hist)][:need]
            cand = np.concatenate([hist, fill])
        else:
            cand = hist
        return cand.astype(np.int64)

    def memory_scores(self, u, cand):
        return self.M[u][cand]

    def pop_scores(self, cand):
        return self.pop[cand]


def _rank_metrics(score, tgt_pos):
    """Rank of the true item under `score` (higher=better), 1-indexed. Ties broken
    pessimistically (count all items scoring strictly higher, plus half the ties)."""
    s_true = score[tgt_pos]
    higher = int((score > s_true).sum())
    ties = int((score == s_true).sum()) - 1
    return higher + ties / 2.0 + 1.0


def export_train(S0, n_pois, n_users, cats, lats, lons, n_cand, n_train, rng, chunks=10):
    """T0 training prompts for the LLM's LoRA stage, with CAUSAL candidates.

    T0 is swept CHRONOLOGICALLY in `chunks`; examples in chunk i get candidates from a
    retriever that has seen only chunks 0..i-1. Chunk 0 is skipped (no history yet). This
    mirrors exactly how candidates are produced at evaluation time, so the LoRA head is not
    trained on a candidate distribution it will never see.

    NOTE: `S0` MUST already be in time order. make_samples() emits user-major order, and
    chunking THAT gives disjoint user cohorts rather than time slices -- the retriever then
    knows nothing about the next chunk's users, the target almost never lands in the
    candidate list, and the export collapses (409 of 8000 survived when we made exactly this
    mistake). It is the same defect this paper documents in Section 6.6; the caller passes a
    chronologically sorted S0.
    """
    ret = CausalRetriever(n_users, n_pois)
    u_all = S0["user"].numpy(); t_all = S0["tgt"].numpy()
    seq_all = S0["seq"].numpy(); h_all = S0["hour"].numpy(); d_all = S0["dow"].numpy()
    order = np.arange(len(t_all))
    parts = np.array_split(order, chunks)

    out = []
    per_chunk = max(1, n_train // (chunks - 1))
    for ci, idx in enumerate(parts):
        if ci > 0:
            take = rng.choice(len(idx), size=min(per_chunk, len(idx)), replace=False)
            for i in take:
                j = idx[i]
                u = int(u_all[j]); t = int(t_all[j])
                cand = ret.candidates(u, n_cand)
                if t not in cand:
                    continue                    # unlearnable: target not even a candidate
                cand_sh = rng.permutation(cand)
                tgt_pos = int(np.flatnonzero(cand_sh == t)[0])
                seq = [int(p) for p in seq_all[j] if p != n_pois]
                out.append({
                    "user": u, "target": t, "tgt_pos": tgt_pos,
                    "hour": int(h_all[j]), "dow": int(d_all[j]),
                    "seq": seq[-10:], "seq_cat": [cats[p] for p in seq[-10:]],
                    "cand": [int(c) for c in cand_sh],
                    "cand_cat": [cats[c] for c in cand_sh],
                    "cand_ll": [[round(float(lats[c]), 5), round(float(lons[c]), 5)]
                                for c in cand_sh],
                    "mem": [float(x) for x in ret.memory_scores(u, cand_sh)],
                })
        ret.observe(u_all[idx], t_all[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--n-cand", type=int, default=50, dest="n_cand")
    ap.add_argument("--n-eval", type=int, default=1000, dest="n_eval")
    ap.add_argument("--n-train", type=int, default=8000, dest="n_train")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = L.load(args.dataset)
    n_users = int(df.user_idx.max() + 1); n_pois = int(df.poi_idx.max() + 1)
    S0 = L.make_samples(df[df.block == "T0"], n_pois)
    S_test, _ = chrono_test_samples(df[df.block != "T0"], n_pois)
    rounds = np.array_split(np.arange(len(S_test["tgt"])), args.rounds)

    # POI metadata for the prompts (Foursquare carries venue categories; SNAP sets do not)
    meta = (df.groupby("poi_idx")[["cat", "lat", "lon"]].first()
              .reindex(range(n_pois)))
    cats = meta["cat"].fillna("Unknown").astype(str).to_numpy()
    lats = meta["lat"].to_numpy(); lons = meta["lon"].to_numpy()
    has_cat = df["cat"].notna().any() and df["cat"].nunique() > 1

    ret = CausalRetriever(n_users, n_pois)
    ret.seed_from(S0["user"].numpy(), S0["tgt"].numpy())        # base period is observed

    rng = np.random.default_rng(args.seed)
    per_round = max(1, args.n_eval // len(rounds))

    hit = {10: 0, 50: 0, 100: 0}
    mem_r, pop_r, n_tot = [], [], 0
    mem_cand_r = []
    export = []

    for r, idx in enumerate(rounds):
        u_all = S_test["user"].numpy()[idx]
        t_all = S_test["tgt"].numpy()[idx]
        seq_all = S_test["seq"].numpy()[idx]
        h_all = S_test["hour"].numpy()[idx]
        d_all = S_test["dow"].numpy()[idx]

        # ---- full-catalogue counting baselines + retriever recall (on ALL points) ----
        for u, t in zip(u_all, t_all):
            mu = ret.M[u]
            # memory-only, backing off to popularity: user counts dominate, pop breaks ties
            score = mu + 1e-6 * (ret.pop / (ret.pop.max() + 1e-9))
            mem_r.append(_rank_metrics(score, t))
            pop_r.append(_rank_metrics(ret.pop, t))
            n_tot += 1
        for K in (10, 50, 100):
            for u, t in zip(u_all, t_all):
                if t in ret.candidates(u, K):
                    hit[K] += 1

        # ---- stratified eval subsample -> LLM prompts ----
        take = rng.choice(len(idx), size=min(per_round, len(idx)), replace=False)
        for i in take:
            u = int(u_all[i]); t = int(t_all[i])
            cand = ret.candidates(u, args.n_cand)
            if t not in cand:                      # retriever miss: every method misses it
                cand_sh = rng.permutation(cand)
                tgt_pos = -1
            else:
                cand_sh = rng.permutation(cand)    # shuffle => no position bias for the LLM
                tgt_pos = int(np.flatnonzero(cand_sh == t)[0])

            mem_cand_r.append(_rank_metrics(ret.memory_scores(u, cand_sh), tgt_pos)
                              if tgt_pos >= 0 else args.n_cand + 1)

            seq = [int(p) for p in seq_all[i] if p != n_pois]
            export.append({
                "round": r, "user": u, "target": t, "tgt_pos": tgt_pos,
                "hour": int(h_all[i]), "dow": int(d_all[i]),
                "seq": seq[-10:],
                "seq_cat": [cats[p] for p in seq[-10:]],
                "cand": [int(c) for c in cand_sh],
                "cand_cat": [cats[c] for c in cand_sh],
                "cand_ll": [[round(float(lats[c]), 5), round(float(lons[c]), 5)] for c in cand_sh],
                "mem": [float(x) for x in ret.memory_scores(u, cand_sh)],
            })

        ret.observe(u_all, t_all)                  # predict-then-update

    def acc(ranks, k):
        return float(np.mean([r <= k for r in ranks]))
    def mrr(ranks):
        return float(np.mean([1.0 / r for r in ranks]))

    out = {
        "dataset": args.dataset, "n_pois": n_pois, "n_users": n_users,
        "n_test": int(n_tot), "has_category": bool(has_cat),
        "retriever_recall": {f"@{K}": round(hit[K] / n_tot, 4) for K in (10, 50, 100)},
        "full_catalogue": {
            "memory_only": {"acc@1": round(acc(mem_r, 1), 4), "acc@5": round(acc(mem_r, 5), 4),
                            "acc@10": round(acc(mem_r, 10), 4), "mrr": round(mrr(mem_r), 4)},
            "popularity": {"acc@1": round(acc(pop_r, 1), 4), "acc@5": round(acc(pop_r, 5), 4),
                           "acc@10": round(acc(pop_r, 10), 4), "mrr": round(mrr(pop_r), 4)},
        },
        "candidate_restricted": {
            "n_cand": args.n_cand, "n_eval": len(export),
            "memory_only": {"acc@1": round(acc(mem_cand_r, 1), 4),
                            "acc@5": round(acc(mem_cand_r, 5), 4),
                            "acc@10": round(acc(mem_cand_r, 10), 4),
                            "mrr": round(mrr(mem_cand_r), 4)},
        },
    }
    print(json.dumps(out, indent=2))

    (ROOT / f"llm_prep_{args.dataset}.json").write_text(json.dumps(out, indent=2))
    with open(ROOT / f"llm_eval_{args.dataset}.jsonl", "w", encoding="utf-8") as f:
        for e in export:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # S0 in TIME order (make_samples gives user-major; see export_train's docstring)
    S0_chrono, _ = chrono_test_samples(df[df.block == "T0"], n_pois)
    tr = export_train(S0_chrono, n_pois, n_users, cats, lats, lons, args.n_cand, args.n_train,
                      np.random.default_rng(args.seed + 99))
    with open(ROOT / f"llm_train_{args.dataset}.jsonl", "w", encoding="utf-8") as f:
        for e in tr:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n-> llm_prep_{args.dataset}.json")
    print(f"-> llm_eval_{args.dataset}.jsonl   ({len(export)} eval prompts, "
          f"{args.n_cand} candidates each)")
    print(f"-> llm_train_{args.dataset}.jsonl  ({len(tr)} T0 training prompts, causal candidates)")


if __name__ == "__main__":
    main()
