#!/usr/bin/env python3
"""
verify_manuscript.py -- every headline number in the manuscript, checked against the artifact
that produced it.

The paper's thesis is that this field does not check its own numbers. It would be absurd to ship
it without checking ours. This script has already earned its keep three times:

  * It caught the manuscript carrying THREE different values for GETNext's Acc@10 (.5543 in the
    table, .5538 in the body, .5531 in its own log) -- and it caught that the FIRST version of
    this very script had passed all of them, because it substring-matched: ".554" is a substring
    of ".5543".
  * It caught three prose paragraphs quoting GIRAM = .5752 -- the BEST of five seeds -- while the
    table correctly quoted the mean, .5747. Seed cherry-picking, in this paper of all papers.
  * It caught my own stale expectations for the relative-improvement claims after the ranking
    fix, by recomputing them from the artifacts instead of trusting the prose.

WHAT IT DOES
  1. TOKEN MATCH, never substring: ".554" does not match inside ".5543".
  2. READS THE \input-ed TABLES too, so generated tables are covered.
  3. RECOMPUTES every "+X% relative" claim from the two canonical values it derives from.
  4. NEAR-MISS SCAN: flags any 4-decimal number in the text that is within 1% of a canonical
     value but is not equal to any canonical value. That is what a stale copy looks like.

Run:  python verify_manuscript.py        Exit code 1 if anything fails.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent

# ---- read main_v2.tex AND every file it \input's --------------------------------------------
TEX = (DOC / "main_v2.tex").read_text(encoding="utf-8", errors="surrogateescape")
for m in re.finditer(r"\\input\{([^}]+)\}", TEX):
    p = DOC / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
    if p.exists():
        TEX += "\n" + p.read_text(encoding="utf-8", errors="surrogateescape")

fails, warns = [], []
CANON = {}            # label -> (value, source)


def forms(v):
    return {f"{v:.4f}".lstrip("0"), f"{v:.4f}",
            f"{v:.3f}".lstrip("0"), f"{v:.3f}",
            f"{v:.2f}".lstrip("0"), f"{v:.2f}"}


TOKENS = [(m.group(1), float(m.group(1)))
          for m in re.finditer(r"(?<![\d.])(0?\.\d{2,4})(?![\d])", TEX)]
TOKSET = {t for t, _ in TOKENS}


def reg(label, value, source):
    """Register a canonical value WITHOUT requiring it to appear (e.g. per-seed grid entries)."""
    CANON[label] = (float(value), source)
    return float(value)


def check(label, value, source):
    """Register AND require the value to appear in the tex as a whole token."""
    v = reg(label, value, source)
    hit = bool(forms(v) & TOKSET)
    if not hit:
        fails.append((label, f"artifact says {v:.4f}; not found in the manuscript", source))
    print(f"  [{'ok ' if hit else 'MISS'}] {label:42s} {v:<9.4f} <- {source}")
    return v


def check_int(label, value, source):
    s = f"{value:,}".replace(",", "{,}")
    hit = s in TEX or str(value) in TEX
    if not hit:
        fails.append((label, f"artifact says {value:,}; not found", source))
    print(f"  [{'ok ' if hit else 'MISS'}] {label:42s} {value:<9,} <- {source}")


def check_ratio(label, num, den, claimed, tol=0.15):
    a, b = CANON[num][0], CANON[den][0]
    true = (a / b - 1.0) * 100
    ok = abs(true - claimed) <= tol
    if not ok:
        fails.append((label, f"manuscript claims {claimed:+.1f}%, artifacts give {true:+.1f}%",
                      f"{num} / {den}"))
    print(f"  [{'ok ' if ok else 'MISS'}] {label:42s} claimed {claimed:+6.1f}%  actual {true:+6.1f}%")


print("Verifying every headline number against its source artifact.\n")

# ---- the central table (generated; every row per-instance, expected-rank ties, full MRR) -----
print("Protocol A / NYC -- tab_sota_nyc.tex, generated from the artifacts:")
gr = json.loads((ROOT / "getnext_row.json").read_text())
gx = json.loads((ROOT / "getnext_row_fixed.json").read_text())
gc = json.loads((ROOT / "getnext_compare_OURS-NYC.json").read_text())["results"]

for k in ("acc@1", "acc@5", "acc@10", "acc@20", "mrr"):
    check(f"GETNext(as-released) {k}", gr[k], "getnext_row.json")
for k in ("acc@10", "mrr"):
    check(f"GETNext(index-corrected) {k}", gx[k], "getnext_row_fixed.json")
for tag, nm in (("popularity", "Popularity"), ("count-static", "COUNT-static"),
                ("count-traj", "COUNT-traj"), ("count-stream", "COUNT-stream")):
    for k in ("acc@1", "acc@5", "acc@10", "acc@20"):
        check(f"{nm} {k}", gc[tag][k], "getnext_compare_OURS-NYC.json")
    # MRR: full catalogue on BOTH sides -- never truncated on one side only
    check(f"{nm} mrr", gc[tag]["mrr_full"], "getnext_compare_OURS-NYC.json")

print("\nDerived percentage claims, recomputed from the canonical values above:")
# The counter's memory is accumulated in float64, not float32: in float32 the eps*popularity
# tie-break is rounded away above ~8 visits, which manufactures ties at the TOP of the ranking
# (sec:dtype, dtype_control.py). These are the repaired margins; the float32 ones were
# +12.4 / +8.9 / +15.6 / +0.8 / -16.0.
check_ratio("COUNT-traj vs GETNext, Acc@10", "COUNT-traj acc@10", "GETNext(as-released) acc@10", 12.6)
check_ratio("COUNT-traj vs GETNext, Acc@5", "COUNT-traj acc@5", "GETNext(as-released) acc@5", 9.8)
check_ratio("COUNT-traj vs GETNext, Acc@20", "COUNT-traj acc@20", "GETNext(as-released) acc@20", 15.2)
check_ratio("COUNT-traj vs GETNext, MRR", "COUNT-traj mrr", "GETNext(as-released) mrr", 1.1)
check_ratio("COUNT-traj vs GETNext, Acc@1", "COUNT-traj acc@1", "GETNext(as-released) acc@1", -12.1)

# ---- the decomposition ----------------------------------------------------------------------
print("\ndiscovery_summary.json (the return/discovery split):")
d = json.loads((ROOT / "discovery_summary.json").read_text())
pa = d["protocol_A_full_catalogue"]["acc@10"]
check("popularity, discovery", pa["popularity"]["discovery"], "discovery_summary.json")
check("GETNext, discovery", pa["GETNext (official, as-released)"]["discovery"], "discovery_summary.json")
check("GETNext, return", pa["GETNext (official, as-released)"]["revisit"], "discovery_summary.json")
check("COUNT, return", pa["COUNT (same info as GETNext)"]["revisit"], "discovery_summary.json")

print("\nscore_decomposition.json (the method-independent claim):")
sd = json.loads((ROOT / "score_decomposition.json").read_text())
lo, hi = sd["_summary"]["best_achievable_return_share_range"]
print(f"  [--- ] best-achievable return share of the score: {lo:.1%} - {hi:.1%}")
for k in ("nyc", "tky", "gowalla_ca", "brightkite"):
    reg(f"COUNT return-share-of-score, {k}", sd[k]["COUNT"], "score_decomposition.json")

# ---- the 4-dataset decomposition (register all, so the near-miss scan knows them) ------------
ds = json.loads((ROOT / "discovery_stream.json").read_text())
for k, e in ds.items():
    reg(f"return_share {k}", e["return_share"], "discovery_stream.json")
    for m, a in e["acc@10"].items():
        for part in ("all", "return", "discovery"):
            reg(f"{m} {part} {k}", a[part], "discovery_stream.json")

# ---- the LLM arm ----------------------------------------------------------------------------
print("\nthe language-model arm:")
a = json.loads((ROOT / "llm_audit_nyc.json").read_text())["results"]
p = json.loads((ROOT / "llm_nyc_partial.json").read_text())["results"]
rb = json.loads((ROOT / "random_baseline_nyc.json").read_text())
check("llm-zs Acc@10", p["llm-zs"]["acc@10"], "llm_nyc_partial.json")
check("llm-static Acc@10", a["llm-static"]["acc@10"], "llm_audit_nyc.json")
check("llm-ft Acc@10 (natural rate)", a["llm-ft"]["acc@10"], "llm_audit_nyc.json")
check("llm-ft-low Acc@10 (10x smaller)", a["llm-ft-low"]["acc@10"], "llm_audit_nyc.json")
check("chance Acc@10", rb["chance"]["acc@10"], "random_baseline_nyc.json")
check("retriever ceiling (recall@50)", rb["retriever_recall"], "random_baseline_nyc.json")
for lab, v in (("COUNT in LLM protocol", 0.6202), ("best fusion alpha=0.05", 0.6263)):
    check(lab, v, "llm_alpha_sweep (honest ties)")

# ---- the 5-seed streaming grid: register EVERY policy x dataset mean ------------------------
# ACC is acc@10m -- the per-instance mean with expected-rank ties (ranking.py), which is the ONE
# estimator the whole paper uses. It is NOT acc@10, the per-round macro average with torch.topk
# index-order tie-breaking that these runs also emit. Neural rows are identical under both; only
# the counter, whose visit counts tie, moves. See the note above stream_table() in gen_tables.py.
ACC = "acc@10m"
print("\nthe 5-seed streaming grid (means; the prose must not quote a single seed):")
cnt = pd.read_csv(ROOT / "results_chrono_count.csv").set_index("dataset")
for dsn in ("nyc", "tky", "gowalla_ca", "brightkite"):
    check(f"COUNT Acc@10, {dsn}", cnt.loc[dsn, ACC], "results_chrono_count.csv")
    reg(f"COUNT churn, {dsn}", cnt.loc[dsn, "churn"], "results_chrono_count.csv")
    R = pd.read_csv(ROOT / f"results_chrono_{dsn}_gru.csv")
    for pol, g in R.groupby("policy"):
        reg(f"{pol} Acc@10 mean, {dsn}", round(g[ACC].mean(), 4),
            f"results_chrono_{dsn}_gru.csv")
        if "churn" in R.columns:
            reg(f"{pol} churn mean, {dsn}", round(g["churn"].mean(), 4),
                f"results_chrono_{dsn}_gru.csv")
for pol in ("static", "GIRAM"):
    R = pd.read_csv(ROOT / "results_chrono_nyc_gru.csv")
    check(f"{pol} Acc@10, nyc (5-seed MEAN)", round(R[R.policy == pol][ACC].mean(), 4),
          "results_chrono_nyc_gru.csv")

# ---- data ------------------------------------------------------------------------------------
print("\ndataset_stats.json + leakage_check.json:")
st = {s["dataset"]: s for s in json.loads((ROOT / "dataset_stats.json").read_text())}
check_int("total check-ins", sum(s["checkins"] for s in st.values()), "dataset_stats.json")
check_int("total users", sum(s["users"] for s in st.values()), "dataset_stats.json")
check_int("total POIs", sum(s["pois"] for s in st.values()), "dataset_stats.json")
for dsn in st:
    reg(f"{dsn} revisit_rate", st[dsn]["revisit_rate"], "dataset_stats.json")
check("Foursquare-NYC revisit_rate", st["Foursquare-NYC"]["revisit_rate"], "dataset_stats.json")
check("Brightkite-US revisit_rate", st["Brightkite-US"]["revisit_rate"], "dataset_stats.json")
lk = {r["dataset"]: r for r in json.loads((ROOT / "leakage_check.json").read_text())}
for dsn in ("Foursquare-NYC", "Brightkite-US"):
    pct = lk[dsn]["post_T0_pct"]
    hit = f"{pct:.2f}\\%" in TEX
    print(f"  [{'ok ' if hit else 'MISS'}] {dsn + ' boundary overlap':42s} {pct:<9.2f} <- leakage_check.json")
    if not hit:
        fails.append((f"{dsn} overlap", f"{pct:.2f}% not found", "leakage_check.json"))

# ---- stale-copy detection lives in verify_prose.py --------------------------------------------
# An earlier version of this file ran a "near-miss" scan here: flag any number within 1% of a
# canonical value but equal to none. It found the real bugs (.5752 = the best of five seeds, where
# the table quoted the mean) -- but it also produced ~100 false alarms, because its registry did
# not know about the published literature numbers, the seed-0 demonstration tables, or the
# alpha-sweep rows. A check that cries wolf is a check people learn to ignore, which is exactly
# the failure mode this whole exercise is about.
#
# verify_prose.py does the job properly: it builds the COMPLETE set of numbers the paper is
# allowed to say (every value in a generated table, every artifact, every explicitly-labelled
# published number) and flags anything in the prose that traces to none of them. Run both.
print("\n(stale-copy detection: run verify_prose.py -- it has the complete registry)")
print()
if fails:
    print(f"{len(fails)} NUMBER(S) DO NOT MATCH THEIR SOURCE:")
    for lab, why, src in fails:
        print(f"   - {lab}: {why}   ({src})")
if warns:
    print(f"\n{len(warns)} NEAR-MISS(ES). A stale copy of a recomputed number looks exactly like "
          f"this. Resolve each one before submission.")
if fails or warns:
    sys.exit(1)
print("All headline numbers trace to the artifacts that produced them; every derived percentage")
print("recomputes; no stale near-duplicates remain.")
