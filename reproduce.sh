#!/usr/bin/env bash
# Reproduce every table and figure in the paper, in order.
# Activate the `poi` conda env first (see README). A CUDA GPU is strongly recommended.
# Override the interpreter with: PYTHON=/path/to/python bash reproduce.sh
set -euo pipefail
PY=${PYTHON:-python}

echo "[1/8] Data  (skipped if data/processed/ already present)"
if [ ! -d data/processed/nyc ]; then
  $PY download_data.py     # -> data/raw/      (Foursquare, Gowalla, Brightkite)
  $PY preprocess.py        # -> data/processed/ (10-core, 24h trajectories, T0-T5)
fi

echo "[2/8] Observations 1-3  (redundancy, forgetting, regime / cold-start)"
$PY observe.py
$PY r2_staleness.py
$PY revision_analysis.py
$PY revision_neural_redundancy.py

echo "[3/8] Main multi-seed comparison  (3 backbones x 4 datasets x 5 seeds)"
for BB in gru attn getnext; do
  $PY run_submission.py --backbone "$BB" --datasets nyc tky gowalla_ca brightkite
done

echo "[4/8] Learning-rate sweep (the artifact)  +  Markov baseline"
$PY lr_sweep.py
$PY markov_baseline.py

echo "[5/8] Base-shrinking probe  +  (failed) a-priori criterion"
$PY regime_sweep.py
$PY criterion.py

echo "[6/8] Forgetting-feedback controller  +  clean geographic injection"
$PY controller_eval.py
$PY inject_rigorous.py
$PY inject_rigorous_ft.py

echo "[7/8] Measured efficiency  (wall-clock + peak GPU memory)"
$PY efficiency.py

echo "[8/8] Assemble tables  +  render all figures"
$PY make_tables.py
$PY paper_figures.py

echo
echo "Done. Figures in figs_paper/, table numbers printed above,"
echo "per-seed CSVs and analysis JSONs in the current directory."
