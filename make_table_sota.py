#!/usr/bin/env python3
"""
make_table_sota.py -- GENERATE the paper's central table from the artifacts.

WHY THIS EXISTS
The first draft of this table was hand-typed, and it ended up carrying three different values
for GETNext's Acc@10 (.5543 in the table, .5538 in three body paragraphs, .5531 in its own
training log) plus an MRR that compared a truncated estimator against an untruncated one. Every
one of those was a transcription error, not a measurement error. A paper whose thesis is that
this field does not check its numbers cannot hand-type its central table.

So the table is generated. Its numbers cannot drift from the artifacts because they ARE the
artifacts. Output: tab_sota_nyc.tex, \\input{} directly by main_v2.tex.

Every row is measured the same way:
  - the same 5,550 Protocol-A instances,
  - a per-instance mean (never a batch average),
  - a full-catalogue rank for MRR (never truncated at 20).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KS = ("acc@1", "acc@5", "acc@10", "acc@20", "mrr")


def fmt(v, bold=False):
    s = f"{v:.4f}".lstrip("0")
    return rf"\textbf{{{s}}}" if bold else s


def main():
    gc = json.loads((ROOT / "getnext_compare_OURS-NYC.json").read_text())
    cnt = gc["results"]
    n_inst, n_pois = gc["n_instances"], gc["n_pois"]

    gn_ar = json.loads((ROOT / "getnext_row.json").read_text())
    gn_fx = json.loads((ROOT / "getnext_row_fixed.json").read_text())

    def row(d, key="mrr"):
        return {k: d[k] if k != "mrr" else d[key] for k in KS}

    rows = [
        ("Popularity",                "global counts",                     row(cnt["popularity"], "mrr_full")),
        ("GETNext (index-corrected)", r"$T_0$ + traj.\ prefix",            row(gn_fx)),
        ("GETNext (as released)",     r"$T_0$ + traj.\ prefix",            row(gn_ar)),
        (r"\cnt-static",              r"$T_0$ only (\emph{less})",         row(cnt["count-static"], "mrr_full")),
        (r"\cnt-traj",                r"$T_0$ + traj.\ prefix (\emph{same})", row(cnt["count-traj"], "mrr_full")),
        (r"\cnt-stream",              r"$T_0$ + prior stream",             row(cnt["count-stream"], "mrr_full")),
    ]

    # bold the best in each column, EXCLUDING count-stream (it sees more than GETNext, so it is
    # not part of the like-for-like comparison and bolding it would overclaim)
    fair = [r for r in rows if r[0] != r"\cnt-stream"]
    best = {k: max(r[2][k] for r in fair) for k in KS}

    L = []
    L.append(r"\begin{tabular}{llccccc}")
    L.append(r"\toprule")
    L.append(r"Method & Information & Acc@1 & Acc@5 & Acc@10 & Acc@20 & MRR \\")
    L.append(r"\midrule")
    for name, info, m in rows:
        cells = " & ".join(fmt(m[k], bold=(m[k] == best[k] and name != r"\cnt-stream")) for k in KS)
        L.append(f"{name} & {info} & {cells} " + r"\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{7}{l}{\emph{published, different split --- for orientation only:}}\\")
    L.append(r"\emph{GETNext}~\citep{yang2022getnext} & & \emph{.2435} & \emph{.5089} & \emph{.6143} & --- & \emph{.3621} \\")
    L.append(r"\emph{STHGCN}~\citep{yan2023sthgcn} & & \emph{.2734} & \emph{.5361} & \emph{.6244} & --- & \emph{.3915} \\")
    L.append(r"\emph{LLM4POI}~\citep{li2024llm4poi} & & \emph{.3372} & \emph{.3982} & \emph{.5010} & --- & \emph{.3807} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")

    (ROOT.parent / "tab_sota_nyc.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # the derived claims, recomputed here so the prose can never drift from the table
    c, g = cnt["count-traj"], gn_ar
    claims = {
        "n_instances": n_inst, "n_pois": n_pois,
        "acc10_rel_pct": round((c["acc@10"] / g["acc@10"] - 1) * 100, 1),
        "mrr_rel_pct":   round((c["mrr_full"] / g["mrr"] - 1) * 100, 1),
        "acc1_rel_pct":  round((c["acc@1"] / g["acc@1"] - 1) * 100, 1),
        "count_traj_acc10": c["acc@10"], "getnext_acc10": g["acc@10"],
        "count_traj_mrr": c["mrr_full"], "getnext_mrr": g["mrr"],
        "count_traj_acc1": c["acc@1"], "getnext_acc1": g["acc@1"],
        "count_static_acc10": cnt["count-static"]["acc@10"],
        "getnext_fixed_acc10": gn_fx["acc@10"], "getnext_fixed_mrr": gn_fx["mrr"],
        "popularity_acc10": cnt["popularity"]["acc@10"],
    }
    (ROOT / "table_claims.json").write_text(json.dumps(claims, indent=2))

    print("-> tab_sota_nyc.tex")
    print(f"   {n_inst:,} instances, {n_pois:,} POIs\n")
    print(f"   {'method':26s} " + "".join(f"{k:>9s}" for k in KS))
    for name, _, m in rows:
        print(f"   {name:26s} " + "".join(f"{m[k]:9.4f}" for k in KS))
    print()
    print(f"   COUNT-traj vs GETNext(as-rel):  Acc@10 {claims['acc10_rel_pct']:+.1f}%   "
          f"MRR {claims['mrr_rel_pct']:+.1f}%   Acc@1 {claims['acc1_rel_pct']:+.1f}%")
    print("-> table_claims.json  (the prose must quote THESE)")


if __name__ == "__main__":
    main()
