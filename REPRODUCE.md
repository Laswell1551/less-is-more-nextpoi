# Reproduction map: paper element → script → output

Run `bash reproduce.sh` for everything, or run the script in the relevant row.
Figures are rendered by `paper_figures.py` from the JSON/CSV artifacts listed here;
table numbers are assembled by `make_tables.py` (which reads the same artifacts).

## Tables

| Paper table | Script(s) | Output artifact(s) |
|-------------|-----------|--------------------|
| Dataset statistics | `download_data.py`, `preprocess.py` | `data/processed/<ds>/stats.json` |
| Observation statistics (revisit, cold-user, drift, redundancy) | `observe.py`, `r2_staleness.py`, `revision_analysis.py` | `observations.json`, `observations_R2_staleness.json`, `regime_stats.json` |
| Main comparison (Acc@10) | `run_submission.py --backbone {gru,attn,getnext}` | `results_seed_<ds>_<bb>.csv` |
| — Markov row (Table 4: the **frozen** Markov) | `markov_baseline.py --frozen` | `markov_baseline_frozen.json` |
| — Markov, continually updated (not in Table 4) | `markov_baseline.py` | `markov_baseline.json` |
| Significance (paired Δ, p) | `run_submission.py` → `make_tables.py` | `results_seed_*.csv` |
| Multi-K (Acc@1/5/10/20, MRR) | `run_submission.py` → `make_tables.py` | `results_seed_*.csv` |
| Robustness grid (8 methods × 3 backbones × 4 datasets) | `run_submission.py` (all backbones) | `results_seed_*_{gru,attn,getnext}.csv` |
| Cost (forgetting, churn, update steps) | `run_submission.py` | `results_seed_*.csv` |
| Forgetting-feedback controller | `controller_eval.py` | `controller_eval.json` |
| Manufactured-headroom injection (Δ vs static, new/old split) | `inject_rigorous.py`, `inject_rigorous_ft.py` | `inject_rigorous.json`, `inject_rigorous_ft.json` |
| Measured efficiency (ms/round, peak GPU mem) | `efficiency.py` | `efficiency.json` |
| Sensitivity (memory decay γ, fusion α) | `run_submission.py` with varied `run_giram` args | `results_seed_*.csv` |

## Figures (all rendered by `paper_figures.py`)

| Paper figure | Source script → artifact | `paper_figures.py` function |
|--------------|--------------------------|-----------------------------|
| Redundancy (much change, no gain) | `revision_neural_redundancy.py` → `neural_redundancy.json` | `fig_redundancy` |
| Price of fine-tuning (forgetting, churn) | `run_submission.py` → `results_seed_*.csv` | `fig_cost` |
| Regime (drift / cold-user share) | `revision_analysis.py` → `regime_stats.json` | `fig_regime` |
| Warm vs. cold-start | `run_submission.py` → `results_seed_*.csv` | `fig_cold` |
| Acc@10 per method | `run_submission.py` → `results_seed_*.csv` | `fig_main` |
| Accuracy vs. churn (Pareto) | `run_submission.py` → `results_seed_*.csv` | `fig_pareto` |
| Per-round trajectory | `run_submission.py` → `results_rounds_*.csv` | `fig_rounds` |
| Learning-rate artifact (acc + forgetting vs LR) | `lr_sweep.py` → `lr_sweep.json` | `fig_lr` |
| Failed a-priori criterion | `regime_sweep.py` → `regime_sweep.json` | `fig_regime_crit` |
| Manufactured-headroom injection | `inject_rigorous{,_ft}.py` → `inject_rigorous*.json` | `fig_inject` |

The ANCHOR schematic (the framework figure) is a TikZ diagram in the LaTeX source, not
produced by a script.

## Notes

- Determinism: seeds `[0,1,2,3,4]` are fixed in `run_submission.py`; the controller,
  injection, and LR sweeps fix their own seeds. The Markov baseline is deterministic.
- Runtime: the full multi-seed sweep across 3 backbones × 4 datasets takes a few hours on
  a single 16 GB GPU. The committed `*.csv` / `*.json` artifacts let you regenerate every
  table and figure in seconds without rerunning the sweeps.
