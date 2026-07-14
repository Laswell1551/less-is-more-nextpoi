#!/usr/bin/env python3
"""
llm_audit.py -- put a 7B LLM through the SAME freeze-vs-fine-tune audit as every other
method, on the SAME chronological stream, re-ranking the SAME causal candidate list.

This is the editor's demand ("the baselines you used were quite stale -- you should use or
address not using large language models") answered by running one, not by citing one.

VARIANTS
  llm-zs      zero-shot, frozen. No training at all.
  llm-static  QLoRA on T0, then frozen for the whole stream.
              = the LLM4POI recipe (trajectory-in-prompt + LoRA) and our `static` analogue.
  llm-ft      QLoRA on T0, then CONTINUALLY LoRA-fine-tuned after every round.
              = the continual-learning analogue. Run at two rates (see --ft-lr) because our
              whole point is that the forgetting story is rate-dependent.
  llm-anchor  llm-static + the per-user counting memory fused into the candidate scores.

SCORING (exact, and free of the position bias LLM rankers are known to have)
  Candidates are presented as a numbered list 00..NN, SHUFFLED per example by llm_prep, so
  no method can exploit candidate order (Hou et al., ECIR'24). We score candidate i by the
  exact log-likelihood of its two-digit label:
        log P(d1 | prompt) + log P(d2 | prompt, d1)
  Qwen tokenizes digits individually, so every label is exactly 2 tokens and no
  length-normalisation bias is possible. One batched forward per prediction: we run
  [prompt+"0", ..., prompt+"D"] and read P(d1) at position -2 and P(d2|d1) at -1.

FAIRNESS
  Neural baselines are masked to the SAME candidate set and re-scored, so the LLM is not
  being handed an easier task. The retriever's recall@K (reported by llm_prep) is the shared
  ceiling for every method in this protocol.

Usage:
  python llm_audit.py --dataset nyc --variants zs static ft anchor
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent

# The weights live OUTSIDE the project tree on purpose: this repo sits inside a BaiduSync
# folder and is itself a git repo, so a 14 GB model parked under experiments/ would be
# uploaded to the cloud and offered up for commit. It also loads far faster off a local
# disk than through the sync client's file locks.
MODEL = str(Path.home() / "models" / "Qwen2.5-7B-Instruct")

CITY = {"nyc": "New York City", "tky": "Tokyo",
        "gowalla_ca": "California", "brightkite": "the United States"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- prompt
def build_prompt(e, ds, n_cand):
    hist = "\n".join(
        f"  {c}" for c in e["seq_cat"][-8:]) or "  (no recent history)"
    lines = []
    for i, (c, ll) in enumerate(zip(e["cand_cat"], e["cand_ll"])):
        lines.append(f"{i:02d}. {c} ({ll[0]:.3f}, {ll[1]:.3f})")
    cands = "\n".join(lines)
    return (
        f"A user is checking in at places in {CITY[ds]}.\n\n"
        f"Their recent check-ins, oldest first:\n{hist}\n\n"
        f"It is now {DOW[e['dow']]} at {e['hour']:02d}:00.\n\n"
        f"Candidate places (id. category (lat, lon)):\n{cands}\n\n"
        f"Which candidate will the user visit next? "
        f"Answer with the two-digit id only."
    )


def chat(tok, prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------------- scoring
class Scorer:
    def __init__(self, model, tok, n_cand):
        self.model, self.tok, self.n_cand = model, tok, n_cand
        self.n_d1 = (n_cand + 9) // 10                      # first-digit range
        self.digit_ids = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
        assert all(len(x) == 1 for x in self.digit_ids), (
            "tokenizer does not emit single-token digits; the 2-digit scoring "
            "scheme assumes it does -- switch to full-continuation scoring")
        self.digit_ids = [x[0] for x in self.digit_ids]

    @torch.no_grad()
    def score(self, text):
        """-> np.array [n_cand] of exact log P(label_i | prompt).

        ONE forward over the batch [prompt+"0", ..., prompt+"D-1"], reading the logits at the
        last two positions:
            position -2 (the final prompt token) -> P(d1 | prompt)
            position -1 (after digit a)          -> P(d2 | prompt, d1=a)
        This is mathematically identical to scoring every candidate's full continuation, and
        every label is exactly 2 tokens, so there is no length-normalisation bias.

        `logits_to_keep=2` is what makes it affordable. By default the LM head runs at EVERY
        position, producing a [D, L, 152064] tensor -- for a ~900-token prompt that is 7e8
        values and 3 GB once cast to fp32, and it dominated the runtime (107 s per
        prediction). We need two positions, so we ask for two.

        We deliberately do NOT reuse a batch-expanded KV cache here, even though it would cut
        the body compute by D. That version disagreed with the reference implementation by up
        to 0.63 nats -- enough to move a ranking -- so it is not worth the speed.
        """
        tok, model = self.tok, self.model
        base = tok(text, return_tensors="pt").input_ids.to(model.device)
        D = self.n_d1

        d1s = torch.tensor([self.digit_ids[d] for d in range(D)],
                           device=model.device).unsqueeze(1)          # [D, 1]
        ids = torch.cat([base.expand(D, -1), d1s], dim=1)             # [D, L+1]
        lg = model(ids, logits_to_keep=2).logits.float()              # [D, 2, V]

        lp1 = F.log_softmax(lg[0, -2], -1)                            # P(d1 | prompt)
        lp2 = F.log_softmax(lg[:, -1], -1)                            # [D, V]

        out = np.full(self.n_cand, -1e9, dtype=np.float64)
        for a in range(D):
            for b in range(10):
                i = 10 * a + b
                if i < self.n_cand:
                    out[i] = (lp1[self.digit_ids[a]] + lp2[a, self.digit_ids[b]]).item()
        return out


# --------------------------------------------------------------------------- metrics
def rank_of(score, tgt_pos):
    """Expected rank under random tie-breaking (see ranking.py).

    NOT (score > s*).sum()+1. The LLM's log-prob scores essentially never tie, so for the LLM
    the two agree -- but the COUNTER is scored in this same protocol, and its score over the 50
    candidates is a per-user visit count, so every candidate the user has never visited scores
    exactly 0 and they all tie. The optimistic convention would place the target FIRST among
    them, handing the counter Acc@10 credit for targets about which it knows nothing.
    """
    if tgt_pos < 0:
        return None                                   # retriever miss: counts as a miss
    s = score[tgt_pos]
    gt = int((score > s).sum())
    eq = int((score == s).sum())                      # includes the target itself
    return gt + (eq + 1) / 2.0


def metrics(ranks, n):
    """ranks: list of rank-or-None, length n (None => retriever miss => never a hit)."""
    r = [x for x in ranks if x is not None]
    acc = lambda k: sum(1 for x in r if x <= k) / n
    mrr = sum(1.0 / x for x in r) / n
    return {"acc@1": round(acc(1), 4), "acc@5": round(acc(5), 4),
            "acc@10": round(acc(10), 4), "mrr": round(mrr, 4), "n": n}


# --------------------------------------------------------------------------- LoRA
def add_lora(model, r=16, alpha=32):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    # casts norms/embeddings to fp32 and wires up gradient checkpointing correctly for a
    # 4-bit base; without it a 7B QLoRA at ~900-token prompts OOMs on 16 GB.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, cfg)


def lora_step_batch(model, tok, examples, ds, n_cand, opt, accum=4):
    """One pass over `examples`: cross-entropy on the 2 label tokens only.

    Gradients are ACCUMULATED, not batched. The obvious implementation --
        losses = [model(x).loss for x in chunk]; torch.stack(losses).mean().backward()
    -- keeps `chunk` full computation graphs alive simultaneously and OOMs a 16 GB card on a
    7B QLoRA at these prompt lengths. Backward-ing each example immediately frees its graph,
    so peak memory is that of a single example regardless of the effective batch size.
    """
    model.train()
    digit = lambda d: tok.encode(str(d), add_special_tokens=False)[0]
    tot, nb = 0.0, 0
    opt.zero_grad(set_to_none=True)
    # Instances the retriever missed (tgt_pos = -1) carry no learnable label -- the target is
    # not among the candidates, so there is no correct id to predict. They are still scored
    # (as misses) at evaluation time, but they must not be trained on: f"{-1:02d}" is "-1"
    # and int('-') raises. The T0 export already drops them; the continual arm trains on the
    # evaluation stream, so it has to drop them here.
    examples = [e for e in examples if e["tgt_pos"] >= 0]
    for i, e in enumerate(examples):
        text = chat(tok, build_prompt(e, ds, n_cand))
        base = tok(text, return_tensors="pt").input_ids.to(model.device)
        lab = f"{e['tgt_pos']:02d}"
        lab_ids = torch.tensor([[digit(int(lab[0])), digit(int(lab[1]))]],
                               device=model.device)
        ids = torch.cat([base, lab_ids], dim=1)
        tgt = ids.clone()
        tgt[:, :base.shape[1]] = -100                        # loss on the 2 label tokens only
        loss = model(ids, labels=tgt).loss / accum
        loss.backward()                                      # frees this example's graph now
        tot += loss.item() * accum; nb += 1
        if (i + 1) % accum == 0:
            opt.step(); opt.zero_grad(set_to_none=True)
    if nb % accum:
        opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()
    return tot / max(nb, 1)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--variants", nargs="+",
                    default=["zs", "static", "ft", "ft-low", "anchor"])
    ap.add_argument("--n-cand", type=int, default=50, dest="n_cand")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--train-n", type=int, default=4000, dest="train_n")
    ap.add_argument("--ft-lr", type=float, default=2e-4, dest="ft_lr")
    ap.add_argument("--ft-lr-low", type=float, default=2e-5, dest="ft_lr_low")
    ap.add_argument("--ft-per-round", type=int, default=60, dest="ft_per_round")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    ev = [json.loads(l) for l in open(ROOT / f"llm_eval_{args.dataset}.jsonl",
                                      encoding="utf-8")]
    tr = [json.loads(l) for l in open(ROOT / f"llm_train_{args.dataset}.jsonl",
                                      encoding="utf-8")]
    print(f"[{args.dataset}] {len(ev)} eval / {len(tr)} train prompts, "
          f"{args.n_cand} candidates, model={MODEL}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda:0")
        m.config.use_cache = False
        return m

    results = {}

    def evaluate(sc, tag, fuse_mem=False, dump=None):
        """`dump`: path to save the RAW per-instance LLM scores. Saving them lets the
        memory-fusion weight alpha be swept offline for free. Without the raw scores, a
        single alpha is one data point, and the obvious objection -- "you did not tune the
        fusion weight; maybe some alpha>0 beats the counter" -- cannot be answered, because
        alpha->0 recovers the counter exactly and so max_alpha(fused) >= counter by
        construction. The sweep is the only thing that settles whether the LLM adds anything.
        """
        t0 = time.time(); ranks = []; raw = []
        for k, e in enumerate(ev):
            s = sc.score(chat(tok, build_prompt(e, args.dataset, args.n_cand)))
            if dump is not None:
                raw.append([round(float(x), 5) for x in s])
            if fuse_mem:
                m = np.asarray(e["mem"], dtype=np.float64)
                p_llm = np.exp(s - s.max()); p_llm /= p_llm.sum() + 1e-12
                p_mem = m / (m.sum() + 1e-9)
                s = np.log(args.alpha * p_llm + (1 - args.alpha) * p_mem + 1e-12)
            ranks.append(rank_of(s, e["tgt_pos"]))
            if (k + 1) % 100 == 0:
                print(f"    {tag}: {k+1}/{len(ev)}  ({(time.time()-t0)/(k+1):.2f}s/pred)",
                      flush=True)
        r = metrics(ranks, len(ev))
        r["sec"] = round(time.time() - t0, 1)
        results[tag] = r
        if dump is not None:
            with open(dump, "w", encoding="utf-8") as f:
                for row in raw:
                    f.write(json.dumps(row) + "\n")
            print(f"    raw scores -> {Path(dump).name}", flush=True)
        print(f"  {tag:12s} acc@1={r['acc@1']:.4f} acc@5={r['acc@5']:.4f} "
              f"acc@10={r['acc@10']:.4f} mrr={r['mrr']:.4f}  ({r['sec']}s)", flush=True)

    # ---- zero-shot ----------------------------------------------------------
    if "zs" in args.variants:
        m = fresh(); m.eval()
        evaluate(Scorer(m, tok, args.n_cand), "llm-zs")
        del m; torch.cuda.empty_cache()

    # ---- LoRA on T0, then frozen (and its memory-fused variant) -------------
    need_static = {"static", "anchor"} & set(args.variants)
    if need_static or {"ft", "ft-low"} & set(args.variants):
        m = add_lora(fresh())
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                lr=args.ft_lr)
        t0 = time.time()
        loss = lora_step_batch(m, tok, tr[:args.train_n], args.dataset, args.n_cand, opt)
        print(f"  [T0 LoRA] {min(args.train_n, len(tr))} examples, "
              f"loss={loss:.3f} ({time.time()-t0:.0f}s)", flush=True)
        sd = {k: v.detach().clone() for k, v in m.state_dict().items() if "lora" in k}
        m.save_pretrained(str(ROOT / f"lora_t0_{args.dataset}"))   # so it need not be retrained
        print(f"  [T0 LoRA adapter saved -> lora_t0_{args.dataset}/]", flush=True)

        if "static" in args.variants:
            evaluate(Scorer(m, tok, args.n_cand), "llm-static",
                     dump=ROOT / f"llm_scores_static_{args.dataset}.jsonl")
        if "anchor" in args.variants:
            evaluate(Scorer(m, tok, args.n_cand), "llm-anchor", fuse_mem=True)

        # ---- continual LoRA, at two rates -----------------------------------
        for tag, lr in [("llm-ft", args.ft_lr), ("llm-ft-low", args.ft_lr_low)]:
            key = tag.replace("llm-", "")
            if key not in args.variants:
                continue
            m.load_state_dict(sd, strict=False)             # back to the T0 checkpoint
            opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
            sc = Scorer(m, tok, args.n_cand)
            by_round = {}
            for e in ev:
                by_round.setdefault(e["round"], []).append(e)
            ranks, t0 = [], time.time()
            for r in sorted(by_round):
                for e in by_round[r]:                        # predict ...
                    ranks.append(rank_of(
                        sc.score(chat(tok, build_prompt(e, args.dataset, args.n_cand))),
                        e["tgt_pos"]))
                pool = [x for x in by_round[r]]              # ... then update on that round
                lora_step_batch(m, tok, pool[:args.ft_per_round], args.dataset,
                                args.n_cand, opt)
                print(f"    {tag}: round {r} done ({time.time()-t0:.0f}s)", flush=True)
            res = metrics(ranks, len(ev)); res["sec"] = round(time.time() - t0, 1)
            res["ft_lr"] = lr
            results[tag] = res
            print(f"  {tag:12s} acc@1={res['acc@1']:.4f} acc@5={res['acc@5']:.4f} "
                  f"acc@10={res['acc@10']:.4f} mrr={res['mrr']:.4f}", flush=True)
        del m; torch.cuda.empty_cache()

    out = ROOT / f"llm_audit_{args.dataset}.json"
    # Record WHICH model, not WHERE it sits on this disk. MODEL resolves to an absolute path
    # under the user's home directory, and dumping it here once put a real username into a
    # public artifact. The artifact should identify the checkpoint, not the machine.
    out.write_text(json.dumps({"dataset": args.dataset, "model": "Qwen/Qwen2.5-7B-Instruct",
                               "n_cand": args.n_cand, "results": results}, indent=2))
    print(f"\n-> {out.name}")


if __name__ == "__main__":
    main()
