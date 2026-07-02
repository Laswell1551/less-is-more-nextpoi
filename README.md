# Less is More: Auditing Continual Learning for Streaming Next-POI Recommendation

Code, data splits, and protocols for the paper
*"Less is More: A Strong Static Baseline and the Limits of Continual Learning for
Streaming Next-POI Recommendation."*

This repository reproduces every table and figure in the paper. It is a compact audit
harness, not a heavy framework: one file (`learned_continual.py`) holds the backbones,
the streaming predict-then-update protocol, all continual-learning baselines, the
per-user memory, the selective update, and the forgetting/churn metrics; the other
scripts are thin drivers that each produce one part of the paper.

> **Anonymized for double-blind review.** Author names, affiliations, and funding are
> omitted here and will be added when the paper is accepted.

## What the paper finds (and this code shows)

1. A model trained once and frozen is a **strong, under-reported baseline**; no continual
   method beats it by more than a few percent, and a transparent first-order **Markov**
   matches the strongest neural backbone.
2. "Fine-tuning catastrophically forgets" is a **learning-rate artifact** (continuous in
   the update step, not a binary freeze-vs-fine-tune divide).
3. Even an auto-tuned, forgetting-safe **controller**, handed *manufactured* new-POI
   headroom via a clean geographic-injection protocol, captures **none** of it; the small
   recoverable signal is captured by **counting** (a per-user memory), not by gradient.

## Installation

```bash
git clone https://github.com/Laswell1551/less-is-more-nextpoi.git
cd less-is-more-nextpoi
conda env create -f environment.yml      # creates the `poi` env (CUDA 12.4)
conda activate poi
# or, with pip + an existing PyTorch/CUDA install:
pip install -r requirements.txt
```

A CUDA GPU is recommended (experiments were run on an RTX 4080 SUPER, 16 GB). The code
falls back to CPU automatically (`DEVICE` in `learned_continual.py`), but the full sweep
is slow on CPU.

## Quick start

The processed splits ship in `data/processed/` (≈19 MB), so you can reproduce the main
results **without** downloading the 1.1 GB of raw data:

```bash
python run_submission.py --backbone gru --datasets nyc tky gowalla_ca brightkite
python make_tables.py            # prints the main / significance / multi-K tables
python paper_figures.py          # writes all figures to figs_paper/
```

To rebuild the splits from scratch (optional):

```bash
python download_data.py          # -> data/raw/   (Foursquare, Gowalla, Brightkite)
python preprocess.py             # -> data/processed/   (10-core, 24h trajectories, T0-T5)
```

To reproduce **everything** in order:

```bash
bash reproduce.sh
```

## Datasets

All four datasets are public; we redistribute only our **preprocessing and chronological
splits**, not the raw logs. `download_data.py` fetches them from the original sources:

| Dataset       | Source                                                            |
|---------------|-------------------------------------------------------------------|
| Foursquare NYC/TKY | Yang et al., TSMC 2015 (TSMC2014 dump)                       |
| Gowalla       | Cho et al., KDD 2011 (SNAP `loc-gowalla`)                         |
| Brightkite    | Cho et al., KDD 2011 (SNAP `loc-brightkite`)                      |

Preprocessing applies 10-core filtering, splits trajectories at 24 h inactivity gaps,
deduplicates consecutive self-transitions, and forms a base block `T0` plus five
chronological increments `T1`–`T5` (see `preprocess.py` and the paper, §3).

## Repository map

**Core (reproduce the paper):**

| File | Role |
|------|------|
| `learned_continual.py` | The harness: backbones (GRU / self-attention / GETNext-style graph), streaming protocol, `run_policy` / `run_selective` / `run_giram` / `run_ewc` / `run_ader` / `run_giram_vae` / `run_controller`, the `InterestVAE`, and Acc@K / MRR / forgetting / churn metrics. |
| `download_data.py`, `preprocess.py` | Raw download and the 10-core / 24h / T0–T5 pipeline. |
| `run_submission.py` | Multi-seed (5) main comparison per backbone → `results_seed_*.csv`, `results_rounds_*.csv`. |
| `observe.py`, `r2_staleness.py`, `revision_analysis.py`, `revision_neural_redundancy.py` | Observations 1–3 (redundancy, forgetting, regime/cold-start). |
| `lr_sweep.py` | The learning-rate sweep (the "artifact" result) → `lr_sweep.json`. |
| `regime_sweep.py`, `criterion.py` | Base-shrinking probe and the (honestly failed) a-priori criterion → `regime_sweep.json`, `criterion.json`. |
| `markov_baseline.py` | First-order Markov baseline → `markov_baseline.json`. |
| `controller_eval.py` | Forgetting-feedback step-size controller, 5 seeds → `controller_eval.json`. |
| `inject_rigorous.py`, `inject_rigorous_ft.py` | Clean geographic new-POI injection (5 seeds, paired tests, new/old target split) → `inject_rigorous*.json`. |
| `efficiency.py` | Measured per-round wall-clock + peak GPU memory → `efficiency.json`. |
| `make_tables.py`, `paper_figures.py` | Assemble table numbers / render all figures. |

**Exploratory / superseded** (kept for transparency, *not* used in the final paper):
`controller_test.py`, `controller_test2.py` (controller tuning, superseded by
`controller_eval.py`); `injection_test.py`, `injection_mem.py` (single-seed injection,
superseded by `inject_rigorous*.py`); `mechanism_demo.py`, `final_metrics.py`.

See **`REPRODUCE.md`** for the exact table/figure → script → output-file mapping.

## Released artifacts

The per-seed result files (`results_seed_*.csv`, `results_rounds_*.csv`) and the analysis
JSONs (`lr_sweep.json`, `controller_eval.json`, `inject_rigorous*.json`, `markov_baseline.json`,
`efficiency.json`, `regime_sweep.json`, `criterion.json`, `observations*.json`, …) are
committed so the figures/tables can be regenerated without rerunning the GPU sweeps.

## Citation

```bibtex
@article{anonymous2026lessismore,
  title  = {Less is More: A Strong Static Baseline and the Limits of Continual
            Learning for Streaming Next-POI Recommendation},
  author = {Anonymous for review},
  year   = {2026},
  note   = {Under review}
}
```

## License

Released under the MIT License (see `LICENSE`). The reproduced GIRAM / variational
generative-retrieval slot (`run_giram_vae`, `InterestVAE`) is a faithful reimplementation
of prior work (Wang et al., 2025), included for the "simple vs. complex" comparison.
