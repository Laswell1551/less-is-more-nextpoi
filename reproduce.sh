#!/usr/bin/env bash
# Reproduce every table and figure in the paper, in dependency order.
#
# Activate the `poi` conda env first (see README). A CUDA GPU is recommended; the LLM arm needs
# ~10 GB of VRAM. Override the interpreter with:  PYTHON=/path/to/python bash reproduce.sh
#
# NOTE: step 6 (the language model) is the only step that needs a GPU. Its scores are committed,
# so steps 7-9 -- including the fusion sweep and the noise control -- run without one. Pass
# SKIP_LLM=1 to skip step 6 and use the committed scores.
set -euo pipefail
PY=${PYTHON:-python}
SKIP_LLM=${SKIP_LLM:-0}

echo "[1/9] Data  (skipped if data/processed/ is already present)"
if [ ! -d data/processed/nyc ]; then
  $PY download_data.py         # -> data/raw/        (Foursquare TSMC2014, Gowalla, Brightkite)
  $PY preprocess.py            # -> data/processed/  (10-core, 24h trajectories, dedup, T0 + stream)
fi
$PY dataset_stats.py           # -> dataset_stats.json     (Table 3)
$PY check_leakage.py           # -> leakage_check.json     (Table 1)

echo "[2/9] The estimator, and the controls on it"
$PY tie_audit.py               # -> tie_audit.json         optimistic vs expected vs pessimistic
$PY dtype_control.py           # -> dtype_control.json     float32 silently deleted the tie-break

echo "[3/9] The counter, and the neural methods, on a stream ordered BY TIME"
$PY count_baseline.py --datasets nyc tky gowalla_ca brightkite   # -> results_chrono_count.csv
$PY run_chrono.py --backbone gru --datasets nyc tky gowalla_ca brightkite  # 5 seeds
$PY clean_split_control.py     # -> clean_split_control.json   (Table 2: the overlap changes nothing)
$PY seed_significance.py       # paired tests, COUNT vs every neural method

echo "[4/9] The official baselines, each on ITS OWN evaluation instances"
# GETNext scores 5,550 NYC instances; STHGCN scores 9,778. They are NOT the same set, and the two
# numbers must never appear in one table. Each baseline gets its own reconstruction.
$PY getnext_row.py             # -> getnext_row.json, getnext_row_fixed.json  (as-released / index-fixed)
$PY getnext_compare.py --name OURS-NYC
$PY getnext_compare.py --name OURS-TKY
$PY sthgcn_compare.py

echo "[5/9] What the benchmark actually measures"
$PY discovery_analysis.py --name OURS-NYC \
      --getnext-preds as-released=../baselines/GETNext/preds_asreleased.json
$PY discovery_stream.py --datasets nyc tky gowalla_ca brightkite
$PY score_decomposition.py     # -> score_decomposition.json   (the identity the paper turns on)

echo "[6/9] Language model  (GPU; SKIP_LLM=1 to reuse the committed scores)"
if [ "$SKIP_LLM" = "0" ]; then
  $PY llm_prep.py --dataset nyc --n-eval 1000 --n-cand 50
  $PY llm_audit.py --dataset nyc
  $PY llm_scaling.py
else
  echo "      skipped -- using the committed llm_scores_static_nyc.jsonl"
fi

echo "[7/9] The LLM claims, checkable with no GPU at all"
$PY llm_alpha_sweep.py         # does ANY fusion weight let the LLM improve on the counter?
$PY tiebreak_control.py        # half of what it 'adds' is tie-breaking, not information

echo "[8/9] The defects in the released baseline"
$PY getnext_transformer_check.py   # the transformer does no sequence modelling
$PY hashseed_experiment.py         # its accuracy depends on an environment variable

echo "[9/9] Generate every table, render every figure, then check the manuscript"
$PY gen_tables.py              # -> tab_*.tex + table_claims.json.  NO table is hand-typed.
$PY fig_discovery.py           # -> figs_v2/fig_discovery.pdf, fig_score.pdf
$PY paper_figures_v2.py        # -> figs_v2/fig_{cutoff,stream,ordering,revisit}.pdf
$PY fig_llmscale.py            # -> figs_v2/fig_llmscale.pdf

$PY verify_manuscript.py       # every headline number traces to the artifact that produced it
$PY verify_prose.py            # every number-literal in the prose traces to a table or artifact
$PY verify_letter.py           # the response letter matches the manuscript it describes

echo
echo "Done. Figures in figs_v2/, tables written next to the manuscript as tab_*.tex,"
echo "artifacts (JSON/CSV) in the current directory. All three verifiers exited 0."
