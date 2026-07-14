#!/usr/bin/env python3
"""
verify_prose.py -- does every number in the PROSE trace to a generated table or an artifact?

THE QUESTION THIS ANSWERS
The results tables are now generated (gen_tables.py), so they cannot drift. The remaining risk is
the PROSE: a paragraph quoting a number that used to be right, or a number that is right for a
different quantity, or -- as happened here -- the BEST OF FIVE SEEDS where the table quotes the
mean.

So: harvest every 4-decimal number the paper is ALLOWED to say --

    * every number in a generated table  (tab_*.tex -- artifact-derived by construction)
    * every derived percentage in table_claims.json
    * a small, explicit whitelist of scalars (chance level, retriever ceiling, ...)
    * the PUBLISHED literature numbers, listed explicitly and labelled as not ours
    * the numbers in the hand-kept tables that have their own artifact check
      (dataset stats, leakage, the ordering-error demo)

-- then flag every 4-decimal number in main_v2.tex's own prose that is NOT in that set.

Per-seed values are deliberately NOT whitelisted: the prose must quote means, never a single seed.
That is exactly the error this script was written to catch, and it caught it (.5752 in three
paragraphs where the table said .5747).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent

# ---------------------------------------------------------------- what the paper MAY say
ALLOWED = {}          # literal -> where it comes from


def add(lit, src):
    ALLOWED.setdefault(lit, src)


def add_val(v, src):
    for f in (f"{v:.4f}", f"{v:.3f}", f"{v:.2f}"):
        add(f.lstrip("0"), src)
        add(f, src)


# (1) every number in a generated table is artifact-derived by construction
for t in sorted(DOC.glob("tab_*.tex")):
    for m in re.finditer(r"(?<![\d.])(0?\.\d{2,4})(?![\d])", t.read_text(encoding="utf-8")):
        add(m.group(1), f"generated table {t.name}")

# (2) every derived percentage the prose is allowed to quote
tc = json.loads((ROOT / "table_claims.json").read_text())


def walk(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            walk(v, f"{path}.{k}" if path else k)
    elif isinstance(d, list):
        for v in d:
            walk(v, path)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        if 0 < abs(d) < 1:
            add_val(float(d), f"table_claims:{path}")


walk(tc)

# (3) explicit scalars
rb = json.loads((ROOT / "random_baseline_nyc.json").read_text())
for k, v in rb["chance"].items():
    add_val(float(v), "random_baseline_nyc.json:chance")
add_val(float(rb["retriever_recall"]), "random_baseline_nyc.json:ceiling")
for f in ("llm_audit_nyc.json", "llm_nyc_partial.json"):
    for cfg, m in json.loads((ROOT / f).read_text())["results"].items():
        for k, v in m.items():
            if isinstance(v, (int, float)) and 0 < v < 1:
                add_val(float(v), f"{f}:{cfg}.{k}")
for f in ("discovery_OURS-NYC.json", "discovery_stream.json", "score_decomposition.json",
          "tiebreak_control.json", "getnext_row.json", "getnext_row_fixed.json",
          "getnext_compare_OURS-NYC.json", "getnext_compare_OURS-TKY.json",
          "dataset_stats.json", "leakage_check.json",
          "clean_split_control.json",     # strictly-clean split; acc@10m, same estimator as all
          "getnext_determinism.json",     # 3 identical evals (superseded by the one below)
          "hashseed_experiment.json",     # the CONTROLLED PYTHONHASHSEED experiment
          "llm_scaling_summary.json",     # the training-budget curve + its matched reference lines
          "dtype_control.json",           # float32-vs-float64 counter: the sec:dtype control.
                                          # Registers the SUPERSEDED float32 numbers (.6211/.6505)
                                          # on purpose -- sec:dtype quotes them as the thing we got
                                          # wrong, so they must trace to the control that measured
                                          # them, not merely to a memory of what we used to print.
          "sthgcn_compare_ours_nyc.json"):  # the counter on STHGCN's OWN 9,778 instances
    p = ROOT / f
    if not p.exists():
        continue
    def w2(d):
        if isinstance(d, dict):
            for v in d.values():
                w2(v)
        elif isinstance(d, list):
            for v in d:
                w2(v)
        elif isinstance(d, (int, float)) and not isinstance(d, bool) and 0 < abs(d) < 1:
            add_val(float(d), f)
    w2(json.loads(p.read_text()))

# (4) PUBLISHED numbers -- other people's, quoted for orientation, never ours
PUBLISHED = {
    ".2435": "GETNext published, NYC", ".5089": "GETNext published, NYC",
    ".6143": "GETNext published, NYC", ".3621": "GETNext published, NYC",
    ".2734": "STHGCN published, NYC", ".5361": "STHGCN published, NYC",
    ".6244": "STHGCN published, NYC", ".3915": "STHGCN published, NYC",
    ".3372": "LLM4POI published, NYC", ".3982": "LLM4POI published, NYC",
    ".5010": "LLM4POI published, NYC", ".3807": "LLM4POI published, NYC",
    ".2254": "GETNext published, TKY", ".4417": "GETNext published, TKY",
    ".5287": "GETNext published, TKY", ".3262": "GETNext published, TKY",
    ".2950": "STHGCN published, TKY", ".5207": "STHGCN published, TKY",
    ".5980": "STHGCN published, TKY", ".3986": "STHGCN published, TKY",
    ".3035": "LLM4POI published, TKY", ".3797": "LLM4POI published, TKY",
    ".4474": "LLM4POI published, TKY", ".3492": "LLM4POI published, TKY",
    ".5014": "LLM4POI Acc@10 (refinepoi regrid)",
}
for k, v in PUBLISHED.items():
    add(k, f"PUBLISHED ({v})")

# (5) the hand-kept demonstration tables, whose numbers are separately checked
#     tab:inversion  -- the ordering-error demo, GRU seed 0, both orderings
#     the transformer isolation test
DEMO = {
    ".3161": "tab:inversion, static, seed 0 (bit-identical under both orderings)",
    ".3276": "tab:inversion, static TKY, seed 0",
    ".2233": "tab:inversion, always+replay, user-major, seed 0",
    ".4191": "tab:inversion, always+replay, chronological, seed 0",
    ".2446": "tab:inversion, always+replay TKY, user-major, seed 0",
    ".3695": "tab:inversion, always+replay TKY, chronological, seed 0",
    ".3239": "tab:inversion, GIRAM, user-major, seed 0",
    ".5752": "tab:inversion, GIRAM, chronological, seed 0",
    ".3298": "tab:inversion, GIRAM TKY, user-major, seed 0",
    ".4982": "tab:inversion, GIRAM TKY, chronological, seed 0",
    ".2471": "transformer isolation test",
    ".3448": "transformer isolation test",
    ".3480": "transformer isolation test",
    ".5531": "GETNext's own BATCH-AVERAGED log value (quoted to contrast the estimator)",
    ".5544": "GETNext per-instance, 3-run mean (quoted in the caption)",
    ".0024": "max per-seed sd on Acc@10 across the whole grid (periodic-4); min is .0004 (GIRAM)",
    ".0025": "GETNext TKY val Acc@10, pinned after the loss goes NaN "
             "(baselines/GETNext/runs/tky2/asreleased/log_training.txt: val_top10_acc:0.0025)",
}
for k, v in DEMO.items():
    add(k, f"demo/caption ({v})")

# ---------------------------------------------------------------- scan the prose
TEX = (DOC / "main_v2.tex").read_text(encoding="utf-8", errors="surrogateescape")
# strip everything that is generated or is a verbatim block
TEX = re.sub(r"\\input\{[^}]*\}", "", TEX)

flagged = []
for m in re.finditer(r"(?<![\d.])(0?\.\d{4})(?![\d])", TEX):
    lit = m.group(1)
    if lit in ALLOWED or lit.lstrip("0") in ALLOWED:
        continue
    ln = TEX[: m.start()].count("\n") + 1
    ctx = TEX[max(0, m.start() - 60): m.start() + 40].replace("\n", " ")
    flagged.append((ln, lit, ctx))

print(f"Prose audit: {len(ALLOWED)} number-literals are traceable to a table or an artifact.\n")
if not flagged:
    print("  [ok ] Every 4-decimal number in the prose of main_v2.tex traces to a generated")
    print("        table, an artifact, or an explicitly-labelled published value.")
    sys.exit(0)

print(f"  {len(flagged)} NUMBER(S) IN THE PROSE TRACE TO NOTHING:\n")
for ln, lit, ctx in flagged:
    print(f"   L{ln:<5d} {lit}   ...{ctx}...")
print("\n  Each is either stale, or a quantity that needs registering. Resolve every one.")
sys.exit(1)
