#!/usr/bin/env python3
"""
fig_llmscale.py -- "you did not train it enough": the objection, answered with a curve.

THE OBJECTION
Our LoRA arm is trained on far fewer examples than LLM4POI uses. The obvious rebuttal to
"a counter beats the fine-tuned 7B model" is therefore "you simply did not train it enough."
That objection must be answered with a curve, not an assurance.

THE EXPERIMENT (llm_scaling.py)
One LoRA adapter, trained incrementally over the base-period prompts, evaluated at checkpoints on
a FIXED 300-instance subset -- the same 300 at every checkpoint, so the curve is signal, not noise.
The n=0 point is the freshly-added adapter, whose B matrix is zero: it is exactly the zero-shot
model, which anchors the curve.

THE REFERENCE LINES
The counter's .6202 is measured on all 990 instances. Plotting it against a curve measured on 300
would be comparing two different denominators -- exactly the error this paper is about. So we
recompute chance, ceiling and the counter on THE SAME 300 instances.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ranking as RK

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figs_v2"
OUT.mkdir(exist_ok=True)

BLUE, VERM, GREY, GREEN = "#0072B2", "#D55E00", "#666666", "#009E73"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def reference_lines_on_the_same_subset(n_eval=300, n_cand=50):
    """chance / ceiling / COUNT, recomputed on EXACTLY the 300 instances the curve uses."""
    ev = [json.loads(l) for l in open(ROOT / "llm_eval_nyc.jsonl", encoding="utf-8")]
    rng = np.random.default_rng(0)          # identical to llm_scaling.py
    idx = sorted(rng.choice(len(ev), size=min(n_eval, len(ev)), replace=False))
    sub = [ev[i] for i in idx]
    n = len(sub)

    # ceiling: the retriever found the target at all
    recall = sum(1 for e in sub if e["tgt_pos"] >= 0) / n
    # chance: candidates are shuffled, so a random permutation puts the target in the top 10
    # with probability 10/C, conditional on the retriever having found it
    chance = recall * 10.0 / n_cand
    # the counter: rank the candidates by the per-user memory, expected-rank ties
    hits = 0
    for e in sub:
        if e["tgt_pos"] < 0:
            continue
        m = np.asarray(e["mem"], dtype=np.float64)
        if RK.expected_rank_np(m, e["tgt_pos"]) <= 10:
            hits += 1
    count = hits / n
    return {"n": n, "chance": round(chance, 4), "ceiling": round(recall, 4),
            "count": round(count, 4)}


def main():
    d = json.loads((ROOT / "llm_scaling_nyc.json").read_text())
    pts = d["points"]
    xs = sorted(int(k) for k in pts)
    ys = [pts[str(x)]["acc@10"] for x in xs]

    ref = reference_lines_on_the_same_subset(d["n_eval"], d["n_cand"])
    print(f"Reference lines recomputed on the SAME {ref['n']} instances the curve uses:")
    print(f"   chance   {ref['chance']:.4f}")
    print(f"   COUNT    {ref['count']:.4f}")
    print(f"   ceiling  {ref['ceiling']:.4f}")
    print()
    print(f"{'n_train':>8s}  {'Acc@10':>7s}")
    for x, y in zip(xs, ys):
        print(f"{x:8d}  {y:7.4f}")

    tail = [y for x, y in zip(xs, ys) if x >= 2000]
    print(f"\n   From 2,000 to {xs[-1]:,} examples ({xs[-1]/2000:.1f}x the data), Acc@10 goes "
          f"{tail[0]:.4f} -> {tail[-1]:.4f}: {'FLAT' if abs(tail[-1]-tail[0]) < 0.03 else 'STILL MOVING'}.")
    print(f"   The counter, on the same instances, scores {ref['count']:.4f}.")
    gap = ref["count"] - max(ys)
    print(f"   Gap between the counter and the LLM's BEST point: {gap:+.4f} "
          f"({100*gap/max(ys):+.0f}% relative).")

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(ref["ceiling"], color=GREEN, ls=":", lw=1.2)
    ax.text(xs[-1], ref["ceiling"] + .012, "retriever ceiling (recall@50)",
            ha="right", fontsize=7.2, color=GREEN)
    ax.axhline(ref["count"], color=BLUE, ls="--", lw=1.6)
    ax.text(xs[-1], ref["count"] + .012, f"\\textbf{{a counter}} ({ref['count']:.3f})",
            ha="right", fontsize=8, color=BLUE, fontweight="bold")
    ax.axhline(ref["chance"], color=GREY, ls=":", lw=1.2)
    ax.text(xs[-1], ref["chance"] + .012, "chance", ha="right", fontsize=7.2, color=GREY)

    ax.plot(xs, ys, "o-", color=VERM, lw=1.8, ms=5, label="Qwen2.5-7B, LoRA")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=6.6, color=VERM)

    ax.set_xlabel("fine-tuning examples seen")
    ax.set_ylabel("Acc@10")
    ax.set_xlim(-200, xs[-1] + 200)
    ax.set_ylim(0, max(ref["ceiling"], max(ys)) + 0.10)
    ax.set_title("The language model saturates far below a counter", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.savefig(OUT / "fig_llmscale.pdf")
    print("\n-> fig_llmscale.pdf")

    (ROOT / "llm_scaling_summary.json").write_text(json.dumps({
        "reference_on_same_subset": ref,
        "curve": {str(x): y for x, y in zip(xs, ys)},
        "saturates_at": round(float(np.mean(tail)), 4),
        "count_minus_best_llm": round(gap, 4),
    }, indent=2))
    print("-> llm_scaling_summary.json")


if __name__ == "__main__":
    main()
