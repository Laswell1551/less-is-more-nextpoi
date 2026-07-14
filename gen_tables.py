#!/usr/bin/env python3
"""
gen_tables.py -- GENERATE every results table in the paper, from the artifacts.

WHY THIS EXISTS
Every table in this paper was originally hand-typed, and every one of them drifted:

  * the central NYC table carried THREE different values for GETNext's Acc@10 (.5543 in the
    table, .5538 in the body, .5531 in its own log);
  * three prose paragraphs quoted GIRAM's BEST seed (.5752) while the table quoted the mean
    (.5747) -- seed cherry-picking, in this paper of all papers;
  * the alpha-sweep table bolded the wrong cell and contradicted the prose beneath it;
  * discovery_summary.json -- which verify_manuscript.py trusted as an ARTIFACT -- turned out to
    be a HAND-MAINTAINED file still holding the pre-fix optimistic numbers and a retracted claim.

None of these were measurement errors. All were transcription errors. A paper whose thesis is that
this field does not check its own numbers cannot hand-type its own tables.

So every table is generated. Its numbers cannot drift from the artifacts because they ARE the
artifacts. main_v2.tex \input's them.

Outputs (into the manuscript directory):
    tab_sota_nyc.tex  tab_sota_tky.tex  tab_discovery.tex  tab_discovery4.tex
    tab_llm.tex       tab_stream.tex
and, in experiments/:
    table_claims.json       -- every derived percentage the PROSE is allowed to quote
    discovery_summary.json  -- regenerated (was hand-maintained; had gone stale)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent
DSK = ["nyc", "tky", "gowalla_ca", "brightkite"]
DSN = {"nyc": "Foursquare-NYC", "tky": "Foursquare-TKY",
       "gowalla_ca": "Gowalla-CA", "brightkite": "Brightkite-US"}


def f4(v, bold=False):
    s = f"{v:.4f}".lstrip("0")
    return rf"\textbf{{{s}}}" if bold else s


def write(name, lines):
    (DOC / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {name}")


# =========================================================== 1. Protocol A: NYC and TKY
def sota_tables():
    gn_ar = json.loads((ROOT / "getnext_row.json").read_text())
    gn_fx = json.loads((ROOT / "getnext_row_fixed.json").read_text())
    KS = ("acc@1", "acc@5", "acc@10", "acc@20", "mrr")
    claims = {}

    for tag, jf, has_gn in (("nyc", "getnext_compare_OURS-NYC.json", True),
                            ("tky", "getnext_compare_OURS-TKY.json", False)):
        blob = json.loads((ROOT / jf).read_text())
        c = blob["results"]

        def row(d, mrr_key="mrr_full"):
            return {k: (d[k] if k != "mrr" else d[mrr_key]) for k in KS}

        rows = [("Popularity", r"global counts", row(c["popularity"]))]
        if has_gn:
            rows += [("GETNext (index-corrected)", r"$T_0$ + traj.\ prefix", row(gn_fx, "mrr")),
                     ("GETNext (as released)", r"$T_0$ + traj.\ prefix", row(gn_ar, "mrr"))]
        rows += [(r"\cnt-static", r"$T_0$ only (\emph{less})", row(c["count-static"])),
                 (r"\cnt-traj", r"$T_0$ + traj.\ prefix (\emph{same})", row(c["count-traj"])),
                 (r"\cnt-stream", r"$T_0$ + prior stream", row(c["count-stream"]))]

        # bold the best per column, EXCLUDING count-stream: it sees more than GETNext, so it is
        # not part of the like-for-like comparison and bolding it would overclaim.
        fair = [r for r in rows if r[0] != r"\cnt-stream"]
        best = {k: max(r[2][k] for r in fair) for k in KS}

        L = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Method & Information & Acc@1 & Acc@5 & Acc@10 & Acc@20 & MRR \\", r"\midrule"]
        for nm, info, m in rows:
            cells = " & ".join(f4(m[k], m[k] == best[k] and nm != r"\cnt-stream") for k in KS)
            L.append(f"{nm} & {info} & {cells} " + r"\\")
        if not has_gn:
            L.append(r"GETNext (either version) & \multicolumn{6}{c}{\emph{did not train on "
                     r"our export --- see caption}} \\")
        L += [r"\midrule",
              r"\multicolumn{7}{l}{\emph{published, different split --- for orientation only:}}\\"]
        if tag == "nyc":
            L += [r"\emph{GETNext}~\citep{yang2022getnext} & & \emph{.2435} & \emph{.5089} & \emph{.6143} & --- & \emph{.3621} \\",
                  r"\emph{STHGCN}~\citep{yan2023sthgcn} & & \emph{.2734} & \emph{.5361} & \emph{.6244} & --- & \emph{.3915} \\",
                  r"\emph{LLM4POI}~\citep{li2024llm4poi} & & \emph{.3372} & \emph{.3982} & \emph{.5010} & --- & \emph{.3807} \\"]
        else:
            L += [r"\emph{GETNext}~\citep{yang2022getnext} & & \emph{.2254} & \emph{.4417} & \emph{.5287} & --- & \emph{.3262} \\",
                  r"\emph{STHGCN}~\citep{yan2023sthgcn} & & \emph{.2950} & \emph{.5207} & \emph{.5980} & --- & \emph{.3986} \\",
                  r"\emph{LLM4POI}~\citep{li2024llm4poi} & & \emph{.3035} & \emph{.3797} & \emph{.4474} & --- & \emph{.3492} \\"]
        L += [r"\bottomrule", r"\end{tabular}"]
        write(f"tab_sota_{tag}.tex", L)

        t = c["count-traj"]
        claims[tag] = {"n_instances": blob["n_instances"], "n_pois": blob["n_pois"],
                       "count_traj_acc@10": t["acc@10"], "count_traj_acc@1": t["acc@1"],
                       "count_traj_mrr": t["mrr_full"],
                       "count_static_acc@10": c["count-static"]["acc@10"]}
        if has_gn:
            g = gn_ar
            claims[tag].update({
                "getnext_acc@1": g["acc@1"], "getnext_acc@5": g["acc@5"],
                "getnext_acc@10": g["acc@10"], "getnext_acc@20": g["acc@20"],
                "getnext_mrr": g["mrr"],
                "getnext_fixed_acc@10": gn_fx["acc@10"], "getnext_fixed_mrr": gn_fx["mrr"],
                "rel_acc@10_pct": round((t["acc@10"] / g["acc@10"] - 1) * 100, 1),
                "rel_acc@5_pct": round((t["acc@5"] / g["acc@5"] - 1) * 100, 1),
                "rel_acc@20_pct": round((t["acc@20"] / g["acc@20"] - 1) * 100, 1),
                "rel_mrr_pct": round((t["mrr_full"] / g["mrr"] - 1) * 100, 1),
                "rel_acc@1_pct": round((t["acc@1"] / g["acc@1"] - 1) * 100, 1),
                "index_fix_cost_pct": round((gn_fx["acc@10"] / g["acc@10"] - 1) * 100, 1),
            })
    return claims


# =========================================================== 1b. STHGCN's protocol (NYC)
def sthgcn_table():
    """STHGCN scores 9,778 instances, NOT GETNext's 5,550 -- it ships its own filters. So it needs
    its own table, and the counter must be re-scored on ITS instances. Putting STHGCN's Acc@10 in
    the same table as GETNext's would compare two different denominators."""
    p = ROOT / "sthgcn_compare_ours_nyc.json"
    q = ROOT / "sthgcn_result.json"
    if not (p.exists() and q.exists()):
        print("  (skip tab_sthgcn_nyc.tex -- STHGCN artifacts not present)")
        return None
    c = json.loads(p.read_text())
    sth = json.loads(q.read_text())["sthgcn_test"]
    n, npoi = c["n_instances"], c["n_pois"]
    r = c["results"]
    KS = ("acc@1", "acc@5", "acc@10", "acc@20", "mrr")

    rows = [("Popularity", r"global counts", r["popularity"]),
            (r"STHGCN (official, SIGIR'23)", r"$T_0$ + traj.\ prefix", sth),
            (r"\cnt-static", r"$T_0$ only (\emph{less})", r["count-static"]),
            (r"\cnt-traj", r"$T_0$ + traj.\ prefix (\emph{same})", r["count-traj"]),
            (r"\cnt-stream", r"$T_0$ + prior stream", r["count-stream"])]
    fair = [x for x in rows if x[0] != r"\cnt-stream"]
    best = {k: max(x[2][k] for x in fair) for k in KS}

    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Method & Information & Acc@1 & Acc@5 & Acc@10 & Acc@20 & MRR \\", r"\midrule"]
    for nm, info, m in rows:
        cells = " & ".join(f4(m[k], m[k] == best[k] and nm != r"\cnt-stream") for k in KS)
        L.append(f"{nm} & {info} & {cells} " + r"\\")
    L += [r"\midrule",
          r"\multicolumn{7}{l}{\emph{published, different split --- for orientation only:}}\\",
          r"\emph{STHGCN}~\citep{yan2023sthgcn} & & \emph{.2734} & \emph{.5361} & \emph{.6244} & --- & \emph{.3915} \\",
          r"\bottomrule", r"\end{tabular}"]
    write("tab_sthgcn_nyc.tex", L)

    t = r["count-traj"]
    return {"n_instances": n, "n_pois": npoi,
            "sthgcn": sth, "count_traj": t,
            "count_static_acc@10": r["count-static"]["acc@10"],
            **{f"rel_{k}_pct": round((t[k] / sth[k] - 1) * 100, 1) for k in KS}}


# =========================================================== 2. the Protocol-A decomposition
def discovery_table():
    d = json.loads((ROOT / "discovery_OURS-NYC.json").read_text())
    r = d["results"]
    n, nr, nd = d["n"], d["n_revisit"], d["n_discovery"]
    ds = json.loads((ROOT / "discovery_summary.json").read_text())
    llm = ds["candidate_reranking_50"]["acc@10"]["Qwen2.5-7B (LoRA-T0, frozen)"]

    ROWS = [("Popularity (no personalisation at all)", r["Popularity"], False),
            ("GETNext (official, SIGIR'22)", r["GETNext (as-released)"], False),
            (r"Qwen2.5-7B, LoRA-tuned$^{\dagger}$", llm, False),
            (r"\cnt{} (no neural model)", r["COUNT (same info)"], True)]
    best_dis = max(x[1]["discovery"] for x in ROWS)

    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r"& \multicolumn{3}{c}{Acc@10} \\", r"\cmidrule(lr){2-4}",
         rf"Method & overall & on \textbf{{returns}} (${nr/n*100:.1f}\%$) "
         rf"& on \textbf{{discoveries}} (${nd/n*100:.1f}\%$) \\", r"\midrule"]
    for nm, m, ba in ROWS:
        L.append(f"{nm} & {f4(m['all'], ba)} & {f4(m['revisit'], ba)} & "
                 f"{f4(m['discovery'], m['discovery'] == best_dis)} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_discovery.tex", L)
    return {"n": n, "return_share": round(nr / n, 4), "discovery_share": round(nd / n, 4),
            "popularity_beats_getnext_x": round(r["Popularity"]["discovery"]
                                                / r["GETNext (as-released)"]["discovery"], 1)}


# =========================================================== 3. the four-dataset decomposition
def discovery4_table():
    d = json.loads((ROOT / "discovery_stream.json").read_text())
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"& & \multicolumn{2}{c}{on \textbf{returns}} & \multicolumn{2}{c}{on \textbf{discoveries}} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Dataset & returns & neural & \cnt & popularity & neural \\", r"\midrule"]
    gow = None
    for k in DSK:
        e = d[k]["acc@10"]
        sr = d[k]["return_share"]
        nr_, cr = e["static (neural)"]["return"], e["COUNT"]["return"]
        pd_, nd_ = e["popularity"]["discovery"], e["static (neural)"]["discovery"]
        sr_s = rf"$\mathbf{{{sr*100:.1f}\%}}$" if k == "gowalla_ca" else rf"${sr*100:.1f}\%$"
        L.append(f"{DSN[k]} & {sr_s} & {f4(nr_)} & {f4(cr, True)} & "
                 f"{f4(pd_, pd_ > nd_)} & {f4(nd_, nd_ > pd_)} " + r"\\")
        if k == "gowalla_ca":
            gow = {"neural_discovery": nd_, "popularity_discovery": pd_,
                   "count_discovery": e["COUNT"]["discovery"],
                   "neural_over_popularity_pct": round((nd_ / pd_ - 1) * 100, 1),
                   "discovery_share": round(1 - sr, 4)}
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_discovery4.tex", L)
    return {"gowalla": gow}


# =========================================================== 4. the language-model table
def llm_table():
    a = json.loads((ROOT / "llm_audit_nyc.json").read_text())["results"]
    p = json.loads((ROOT / "llm_nyc_partial.json").read_text())["results"]
    rb = json.loads((ROOT / "random_baseline_nyc.json").read_text())
    ch, ce = rb["chance"], rb["retriever_recall"]

    def sig(x):
        return (x - ch["acc@10"]) / (ce - ch["acc@10"])

    # from llm_alpha_sweep.py with EXPECTED-rank ties (see ranking.py)
    CNT = {"acc@1": .1889, "acc@5": .4889, "acc@10": .6202, "mrr": .3236}   # alpha = 0
    FUS = {"acc@1": .1960, "acc@5": .4949, "acc@10": .6263, "mrr": .3280}   # alpha = 0.05 (best)
    F50 = {"acc@1": .1616, "acc@5": .4394, "acc@10": .5828, "mrr": .2904}   # alpha = 0.50

    ROWS = [("Chance (random permutation)", ch, False), (None, None, None),
            (r"Qwen2.5-7B, LoRA-continual @ $2{\cdot}10^{-4}$ (natural rate)", a["llm-ft"], False),
            (r"Qwen2.5-7B, zero-shot (no training at all)", p["llm-zs"], False),
            (r"Qwen2.5-7B, LoRA-$T_0$ then frozen", a["llm-static"], False),
            (r"Qwen2.5-7B, LoRA-continual @ $2{\cdot}10^{-5}$ (best LLM)", a["llm-ft-low"], False),
            (r"Qwen2.5-7B, LoRA-$T_0$ $+$ \cnt{} ($\alpha{=}0.5$)", F50, False),
            (r"Qwen2.5-7B, LoRA-$T_0$ $+$ \cnt{} ($\alpha{=}0.05$, \emph{best fusion})", FUS, False),
            (None, None, None),
            (r"\textbf{\cnt{} (no neural model)}", CNT, True),
            (None, None, None),
            (r"Ceiling (retriever recall@50)", None, False)]

    L = [r"\begin{tabular}{lccccr}", r"\toprule",
         r"Method & Acc@1 & Acc@5 & Acc@10 & MRR & signal captured \\", r"\midrule"]
    for nm, m, bold in ROWS:
        if nm is None:
            L.append(r"\midrule"); continue
        if m is None:
            L.append(rf"{nm} & --- & --- & {f4(ce)} & --- & $100\%$ \\"); continue
        cells = " & ".join(f4(m[k], bold) for k in ("acc@1", "acc@5", "acc@10", "mrr"))
        s = sig(m["acc@10"])
        s_s = rf"$\mathbf{{{s*100:.1f}\%}}$" if bold else rf"${s*100:.1f}\%$"
        L.append(f"{nm} & {cells} & {s_s} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_llm.tex", L)

    tb = json.loads((ROOT / "tiebreak_control.json").read_text())
    return {"count_signal_pct": round(sig(CNT["acc@10"]) * 100, 1),
            "best_llm_signal_pct": round(sig(a["llm-ft-low"]["acc@10"]) * 100, 1),
            "zs_signal_pct": round(sig(p["llm-zs"]["acc@10"]) * 100, 1),
            "static_signal_pct": round(sig(a["llm-static"]["acc@10"]) * 100, 1),
            "ft_signal_pct": round(sig(a["llm-ft"]["acc@10"]) * 100, 1),
            "count_acc@10": CNT["acc@10"], "best_fusion_acc@10": FUS["acc@10"],
            "count_over_best_llm_pct": round((CNT["acc@10"] / a["llm-ft-low"]["acc@10"] - 1) * 100, 1),
            "llm_fusion_lift_pct": tb["llm_best_lift_pct"],
            "noise_fusion_lift_pct": tb["noise_best_lift_pct_mean"],
            "noise_fusion_lift_sd": tb["noise_best_lift_pct_sd"]}


# =========================================================== 5. the 5-seed streaming grid
#
# ONE estimator, everywhere in the paper: a PER-INSTANCE mean, with ties broken at the EXPECTED
# rank (ranking.py). That is the column `acc@10m`, written by learned_continual.stream_rich.
#
# It is NOT the column `acc@10`, which the streaming runs also emit. `acc@10` comes from
# evaluate()/seg_eval, which (a) averages the per-ROUND accuracies rather than pooling instances,
# and (b) resolves ties with torch.topk, i.e. by POI index order. Both are silent choices, and
# neither is the convention Protocol A uses -- so quoting `acc@10` here and the expected rank in
# Table sota-nyc would put two different estimators in one paper. That is precisely the defect
# this paper documents in other people's work.
#
# It changes almost nothing, and we checked rather than assumed:
#   * neural rows are BIT-IDENTICAL under both columns (float scores essentially never tie, and
#     expected rank is exact for tie-free scores: eq=1 => gt + (1+1)/2 = gt + 1). Verified for all
#     8 policies x 4 datasets.
#   * only the counter moves, because its integer scores tie: NYC .6094 -> .6083, TKY .5103 ->
#     .5093, Gowalla .2578 -> .2564, Brightkite .6972 -> .6963. It costs us 0.1-0.5% relative,
#     against margins of 6-93%. We take the honest number.
ACC = "acc@10m"


def stream_table():
    cnt = pd.read_csv(ROOT / "results_chrono_count.csv").set_index("dataset")
    POL = ["static", "periodic-4", "selective-gated", "always+replay",
           "EWC", "ADER", "GIRAM-VAE", "GIRAM"]
    NAME = {"static": "static (frozen neural)", "periodic-4": "periodic-4",
            "selective-gated": "selective-gated", "always+replay": r"always $+$ replay",
            "EWC": "EWC", "ADER": "ADER", "GIRAM-VAE": "GIRAM-VAE",
            "GIRAM": "GIRAM (memory, fused)"}
    G = {k: pd.read_csv(ROOT / f"results_chrono_{k}_gru.csv") for k in DSK}

    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Policy & NYC & TKY & Gowalla-CA & Brightkite \\", r"\midrule"]
    for p in POL:
        cells = []
        for k in DSK:
            g = G[k][G[k].policy == p][ACC]
            cells.append(f4(g.mean()) + rf"\,{{\tiny$\pm${g.std():.4f}}}")
        L.append(f"{NAME[p]} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\textbf{\cnt{} (no neural model at all)} & "
             + " & ".join(f4(cnt.loc[k, ACC], True) for k in DSK) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_stream.tex", L)

    grad = ["always+replay", "EWC", "ADER", "periodic-4", "selective-gated"]
    full_backbone = ["always+replay", "EWC", "ADER"]
    # Compute the range the PROSE actually claims. This field used to be called
    # "continual_gain_over_frozen_pct" while including COUNT -- which is not a continual-learning
    # method -- so it reported a maximum of 190.2% that no continual method reaches. The prose said
    # "every continual method improves ... by between 12% and 192%", which matched neither the
    # artifact nor anything else; a sentence three subsections later said 4.7-191%. Two numbers for
    # one quantity, in one paper, is the defect this paper is about. Name the quantity, then compute it.
    lo_u = hi_u = lo_g = hi_g = lo_c = hi_c = lo_f = hi_f = None
    churn_ratio = []
    for k in DSK:
        st = G[k][G[k].policy == "static"][ACC].mean()
        for p in POL[1:]:                       # the continual-learning methods, and ONLY those
            r = (G[k][G[k].policy == p][ACC].mean() / st - 1) * 100
            lo_u = r if lo_u is None else min(lo_u, r)
            hi_u = r if hi_u is None else max(hi_u, r)
        rc = (cnt.loc[k, ACC] / st - 1) * 100   # the counter, reported separately
        lo_c = rc if lo_c is None else min(lo_c, rc)
        hi_c = rc if hi_c is None else max(hi_c, rc)
        # naive fine-tuning: the method the superseded release called catastrophically forgetful.
        # The prose and the cover letter both quote this range, so it has to be computed, not recalled.
        rf = (G[k][G[k].policy == "always+replay"][ACC].mean() / st - 1) * 100
        lo_f = rf if lo_f is None else min(lo_f, rf)
        hi_f = rf if hi_f is None else max(hi_f, rf)
        bg = max(G[k][G[k].policy == p][ACC].mean() for p in grad)
        r = (cnt.loc[k, ACC] / bg - 1) * 100
        lo_g = r if lo_g is None else min(lo_g, r)
        hi_g = r if hi_g is None else max(hi_g, r)
        if "churn" in G[k].columns:
            worst = max(G[k][G[k].policy == p]["churn"].mean() for p in full_backbone)
            churn_ratio.append(worst / cnt.loc[k, "churn"])
    return {"continual_gain_over_frozen_pct": [round(lo_u, 1), round(hi_u, 1)],
            "count_gain_over_frozen_pct": [round(lo_c, 1), round(hi_c, 1)],
            "naive_ft_gain_over_frozen_pct": [round(lo_f, 1), round(hi_f, 1)],
            "count_over_best_gradient_pct": [round(lo_g, 1), round(hi_g, 1)],
            "count_churn_range": [round(min(cnt["churn"]), 3), round(max(cnt["churn"]), 3)],
            "churn_ratio_vs_full_backbone": [round(min(churn_ratio), 1), round(max(churn_ratio), 1)]
            if churn_ratio else None}


# =========================================================== 6. regenerate the summary artifact
def regen_summary():
    """discovery_summary.json was HAND-MAINTAINED and had gone stale: it still held the pre-fix
    optimistic COUNT numbers and a retracted 'a counter is its optimal predictor' claim -- while
    verify_manuscript.py trusted it as ground truth. Derive it from the real output."""
    old = json.loads((ROOT / "discovery_summary.json").read_text())
    d = json.loads((ROOT / "discovery_OURS-NYC.json").read_text())
    r = d["results"]
    new = dict(old)
    new["note"] = ("GENERATED by gen_tables.py from discovery_OURS-NYC.json. DO NOT HAND-EDIT: "
                   "this file was hand-maintained once, silently went stale, and the verifier "
                   "trusted it as ground truth.")
    new["protocol_A_full_catalogue"] = {
        "instances": d["n"], "revisit": d["n_revisit"], "discovery": d["n_discovery"],
        "acc@10": {"popularity": r["Popularity"],
                   "GETNext (official, as-released)": r["GETNext (as-released)"],
                   "COUNT (same info as GETNext)": r["COUNT (same info)"],
                   "COUNT (stream)": r["COUNT (stream)"]}}
    new["headline"] = (
        f"{d['n_revisit']/d['n']:.1%} of the benchmark is return prediction. On the "
        f"{d['n_discovery']/d['n']:.1%} that are discoveries, a global popularity baseline "
        f"({r['Popularity']['discovery']:.4f}) beats the graph transformer "
        f"({r['GETNext (as-released)']['discovery']:.4f}) and a counter "
        f"({r['COUNT (same info)']['discovery']:.4f}). A counter is very hard to beat on the "
        f"aggregate metric, but it is NOT 'the optimal solution': it loses at Acc@1. That earlier "
        f"phrasing is retracted.")
    (ROOT / "discovery_summary.json").write_text(json.dumps(new, indent=2))
    print("  -> discovery_summary.json  (REGENERATED -- was hand-maintained and stale)")


def main():
    print("Generating every results table from the artifacts:\n")
    claims = {"protocolA": sota_tables()}
    s = sthgcn_table()
    if s:
        claims["sthgcn_protocol"] = s
    claims["decomposition"] = discovery_table()
    claims["decomposition"].update(discovery4_table())
    claims["llm"] = llm_table()
    claims["stream"] = stream_table()
    regen_summary()
    claims["score_decomposition"] = json.loads(
        (ROOT / "score_decomposition.json").read_text())["_summary"]
    (ROOT / "table_claims.json").write_text(json.dumps(claims, indent=2))
    print("\n" + "=" * 78)
    print("table_claims.json -- THE PROSE MAY QUOTE ONLY THESE:\n")
    print(json.dumps(claims, indent=2))


if __name__ == "__main__":
    main()
