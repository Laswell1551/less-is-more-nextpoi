#!/usr/bin/env python3
"""
llm_scaling.py -- how much fine-tuning data would the LLM need to reach a counter?

WHY THIS EXISTS
Our LoRA arm is trained on far fewer examples than LLM4POI uses, so the obvious objection to
"a counter beats the fine-tuned 7B model" is "you did not train it enough". That objection
has to be answered with a curve, not an assurance.

We train ONE LoRA adapter incrementally over the T0 prompts and evaluate it at checkpoints,
so the training cost is paid once rather than once per point. The result is Acc@10 as a
function of the number of fine-tuning examples, with the counter's score and the chance level
drawn as horizontal references. If the curve saturates below the counter, the objection is
closed; if it is still climbing, we say so and report the extrapolation honestly.

Usage:
  python llm_scaling.py --dataset nyc --n-eval 300
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import llm_audit as LA

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--n-cand", type=int, default=50, dest="n_cand")
    ap.add_argument("--n-eval", type=int, default=300, dest="n_eval")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--checkpoints", nargs="+", type=int,
                    default=[250, 500, 1000, 2000, 3500, 5500])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tr = [json.loads(l) for l in open(ROOT / f"llm_train_{args.dataset}.jsonl",
                                      encoding="utf-8")]
    ev = [json.loads(l) for l in open(ROOT / f"llm_eval_{args.dataset}.jsonl",
                                      encoding="utf-8")]
    # a FIXED evaluation subset, identical at every checkpoint, so the curve is not noise
    rng = np.random.default_rng(0)
    sub = [ev[i] for i in sorted(rng.choice(len(ev), size=min(args.n_eval, len(ev)),
                                            replace=False))]
    cps = [c for c in args.checkpoints if c <= len(tr)]
    print(f"[{args.dataset}] {len(tr)} train prompts available, checkpoints {cps}, "
          f"eval on a fixed {len(sub)}-instance subset", flush=True)

    tok = AutoTokenizer.from_pretrained(LA.MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(LA.MODEL, quantization_config=bnb,
                                                dtype=torch.bfloat16, device_map="cuda:0")
    base.config.use_cache = False
    m = LA.add_lora(base)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    sc = LA.Scorer(m, tok, args.n_cand)

    def evaluate(tag):
        t0 = time.time()
        ranks = [LA.rank_of(sc.score(LA.chat(tok, LA.build_prompt(e, args.dataset,
                                                                 args.n_cand))),
                            e["tgt_pos"]) for e in sub]
        r = LA.metrics(ranks, len(sub))
        r["sec"] = round(time.time() - t0, 1)
        print(f"  n_train={tag:>5}  acc@1={r['acc@1']:.4f} acc@5={r['acc@5']:.4f} "
              f"acc@10={r['acc@10']:.4f} mrr={r['mrr']:.4f}  ({r['sec']}s)", flush=True)
        return r

    out = {"dataset": args.dataset, "n_eval": len(sub), "n_cand": args.n_cand,
           "lr": args.lr, "points": {}}
    out["points"]["0"] = evaluate(0)          # the LoRA is initialised to a no-op => zero-shot

    prev = 0
    for c in cps:
        t0 = time.time()
        loss = LA.lora_step_batch(m, tok, tr[prev:c], args.dataset, args.n_cand, opt)
        print(f"  [+{c-prev} examples, loss={loss:.3f}, {time.time()-t0:.0f}s]", flush=True)
        out["points"][str(c)] = evaluate(c)
        prev = c
        (ROOT / f"llm_scaling_{args.dataset}.json").write_text(json.dumps(out, indent=2))

    print(f"\n-> llm_scaling_{args.dataset}.json")


if __name__ == "__main__":
    main()
