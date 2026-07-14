# Reproduction map: paper element → script → artifact

Every table in the paper is **generated** from the artifacts by `gen_tables.py` and `\input`-ed into
the manuscript. No table is hand-typed. Every figure is rendered from the same artifacts. Three
`verify_*.py` scripts then re-derive every number in the manuscript from those artifacts and fail
loudly on any mismatch.

Run everything, in dependency order: **`bash reproduce.sh`**.

---

## 0. Data

| Step | Script | Output |
|---|---|---|
| Download raw logs (1.1 GB; **not** redistributed) | `download_data.py` | `data/raw/` |
| 10-core, 24 h trajectories, consecutive-duplicate removal, `T0` + stream | `preprocess.py` | `data/processed/<ds>/` *(committed, ~19 MB — you can skip the download)* |
| Dataset statistics (Table 3) | `dataset_stats.py` | `dataset_stats.json` |
| Boundary overlap of our own split (Table 1) | `check_leakage.py` | `leakage_check.json` |
| Strictly-clean split control (Table 2) | `clean_split_control.py` | `clean_split_control.json` |

## 1. The estimator (read this before any number)

| Purpose | Script | Output |
|---|---|---|
| **The one definition of the target's rank.** Expected rank under random tie-breaking. Every experiment imports it. | `ranking.py` | — |
| Optimistic vs. expected vs. pessimistic ties, on GETNext's own instances | `tie_audit.py` | `tie_audit.json` |
| **`float32` vs `float64`: is the counter's tie rate a property of the data, or of the storage type?** (§6.9) | `dtype_control.py` | `dtype_control.json` |

## 2. Tables

| Paper table | Script(s) | Artifact(s) |
|---|---|---|
| **1** Boundary overlap | `check_leakage.py` | `leakage_check.json` |
| **2** Clean-split control | `clean_split_control.py` | `clean_split_control.json` |
| **3** Dataset statistics | `dataset_stats.py` | `dataset_stats.json` |
| **4** Return/discovery decomposition, Protocol A | `discovery_analysis.py --getnext-preds as-released=...` | `discovery_OURS-NYC.json` → `tab_discovery.tex` |
| **5** Decomposition, all four datasets | `discovery_stream.py` | `discovery_stream.json` → `tab_discovery4.tex` |
| **6** NYC, Protocol A (vs. official GETNext) | `getnext_compare.py --name OURS-NYC`, `getnext_row.py` | `getnext_compare_OURS-NYC.json`, `getnext_row{,_fixed}.json` → `tab_sota_nyc.tex` |
| **7** TKY, Protocol A | `getnext_compare.py --name OURS-TKY` | `getnext_compare_OURS-TKY.json` → `tab_sota_tky.tex` |
| **8** NYC, **STHGCN's** protocol (9,778 instances — *not* the 5,550 of Table 6) | `sthgcn_compare.py` | `sthgcn_compare_ours_nyc.json`, `sthgcn_result.json` → `tab_sthgcn_nyc.tex` |
| **9** Candidate re-ranking, Qwen2.5-7B | `llm_prep.py`, `llm_audit.py`, `llm_alpha_sweep.py` | `llm_audit_nyc.json`, `llm_nyc_partial.json`, `random_baseline_nyc.json` → `tab_llm.tex` |
| **10** Protocol B: 5-seed chronological stream | `run_chrono.py`, `count_baseline.py` | `results_chrono_<ds>_gru.csv`, `results_chrono_count.csv` → `tab_stream.tex` |
| **11** The ordering inversion | `run_chrono.py` (vs. the superseded `run_submission.py`) | `results_chrono_*.csv` |
| *(in-text)* protocol swing, score decomposition, dtype control, return rate | `score_decomposition.py`, `dtype_control.py`, `dataset_stats.py` | `score_decomposition.json`, `dtype_control.json`, `dataset_stats.json` |
| Paired significance (COUNT vs. every neural method) | `seed_significance.py` | prints; the counter is deterministic, so it is a one-sample test on the per-seed differences |

**Generate them all:** `python gen_tables.py` → writes `tab_*.tex` into the manuscript directory and
`table_claims.json` (the only derived percentages the prose is permitted to quote).

## 3. Figures

| Paper figure | Script | Reads |
|---|---|---|
| `fig_discovery` — what the benchmark is made of, and what nobody does | `fig_discovery.py` | `discovery_summary.json` |
| `fig_score` — **the argument in two panels**: returns are 35–76% of the *instances* but 95.7–99.7% of the *score* | `fig_discovery.py` | `discovery_stream.json`, `score_decomposition.json` |
| `fig_cutoff` — the ranking of methods is a function of the cutoff | `paper_figures_v2.py` | `getnext_compare_*.json`, `table_claims.json` |
| `fig_stream` — counting beats gradient descent on every dataset | `paper_figures_v2.py` | `results_chrono_*.csv` |
| `fig_ordering` — a sort key decides the answer | `paper_figures_v2.py` | `results_chrono_*.csv` |
| `fig_revisit` — the counter's accuracy is the data's, not the model's | `paper_figures_v2.py` | `dataset_stats.json`, `results_chrono_count.csv` |
| `fig_llmscale` — "you did not train it enough", answered | `fig_llmscale.py` | `llm_scaling_nyc.json`, `llm_scaling_summary.json` |

All figures land in `figs_v2/` (git-ignored — they rebuild from the committed artifacts in seconds).

## 4. The three defects, and the controls that catch them

| Defect | Control | Output |
|---|---|---|
| **Ours:** a sort key turned streaming rounds into disjoint user cohorts (a per-user memory was empty for 97% of scored points) | `run_chrono.py` orders the stream by time. Invariant: *a frozen method must be bit-identical under any permutation of the evaluation set* — it is. | `results_chrono_*.csv` |
| **Released GETNext:** the transformer does no sequence modelling (`batch_first` mismatch → the causal mask runs across the mini-batch); and `list(set(...))` misaligns 4,979 of 4,980 POIs | `getnext_transformer_check.py` — perturbing a trajectory's own earlier check-in moves its own later prediction by `0.000000`, while perturbing *another user's* trajectory moves it by 2.21. `hashseed_experiment.py` — with `PYTHONHASHSEED` fixed the evaluation is bit-identical; changing it re-ranks 59.5% of instances. | `getnext_determinism.json`, `hashseed_experiment.json` |
| **Ours:** `float32` silently deleted the counter's tie-break above ~8 visits — and corrupted our own audit of the tie convention | `dtype_control.py` — same instances, same counter, same estimator, **only the dtype changes** | `dtype_control.json` |

## 5. The LLM arm (and how to check it with no GPU)

`llm_audit.py` needs a GPU (4-bit Qwen2.5-7B, ~10 GB). **You do not need one to check the claim.**
The model's raw per-candidate scores and the exact evaluation prompts are committed:

```bash
python llm_alpha_sweep.py    # the fusion sweep: does ANY weight let the LLM improve the counter?
python tiebreak_control.py   # the noise control: half of what the LLM "adds" is tie-breaking
```

Both read `llm_eval_nyc.jsonl` (990 prompts × 50 shuffled candidates, plus the counter's score
vector) and `llm_scores_static_nyc.jsonl` (the LLM's log-probabilities). CPU, seconds.

To regenerate from scratch (GPU): `llm_prep.py` → `llm_audit.py` → `llm_scaling.py`.
The LoRA adapter (154 MB) and the training prompts (30 MB) are git-ignored; `llm_prep.py` and the
`--lora` path in `llm_audit.py` rebuild both deterministically.

## 6. The manuscript checks itself

```bash
python verify_manuscript.py   # every headline number traces to the artifact that produced it
python verify_prose.py        # all 1,288 number-literals in the prose trace to a table or artifact
python verify_letter.py       # the response letter matches the manuscript it describes
```

Each exits non-zero on any mismatch. Each was written *after* a specific number in the paper turned
out to be stale — see the docstrings, which name the failure that motivated them.
