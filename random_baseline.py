#!/usr/bin/env python3
"""
random_baseline.py -- the chance level for the candidate re-ranking protocol.

Because llm_prep shuffles the candidate order per instance, a ranker with no information
has an exactly computable expected score, and every method must be read against it. Without
this control, "the LLM scores 0.228" is not interpretable.

For an instance whose target IS among the C candidates, a uniformly random permutation puts
it at each rank with probability 1/C, so
    E[Acc@K] = K / C ,      E[RR] = (1/C) * sum_{k=1..C} 1/k = H_C / C .
Instances whose target is NOT among the candidates (a retriever miss) are counted as misses
by every method, so the overall expectation scales by the retriever's recall.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    args = ap.parse_args()

    ev = [json.loads(l) for l in open(ROOT / f"llm_eval_{args.dataset}.jsonl",
                                      encoding="utf-8")]
    n = len(ev)
    C = len(ev[0]["cand"])
    hit = sum(1 for e in ev if e["tgt_pos"] >= 0)      # retriever recall on this subsample
    r = hit / n

    H = sum(1.0 / k for k in range(1, C + 1))
    out = {
        "dataset": args.dataset, "n_eval": n, "n_cand": C,
        "retriever_recall": round(r, 4),
        "chance": {
            "acc@1": round(r * 1 / C, 4),
            "acc@5": round(r * 5 / C, 4),
            "acc@10": round(r * 10 / C, 4),
            "mrr": round(r * H / C, 4),
        },
        "ceiling": {"acc@any": round(r, 4)},
    }
    print(json.dumps(out, indent=2))
    print(f"\n  Every method in this protocol is bounded below by chance and above by the")
    print(f"  retriever's recall ({r:.4f}). A method scoring near the chance row has")
    print(f"  extracted essentially nothing from the prompt.")
    (ROOT / f"random_baseline_{args.dataset}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
