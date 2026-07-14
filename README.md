# Less Is More: Next-POI Benchmarks Reward Returning, and a Counter Is Hard to Beat

Code, chronological splits, evaluation protocols and result artifacts for the paper
*"Less Is More: Next Point-of-Interest Benchmarks Reward Returning, and a Counter Is Hard to
Beat."*

> **Anonymized for double-blind review.** Author names, affiliations and funding are omitted here
> and will be added on acceptance.

---

## ⚠️ This release supersedes the previous one, and the previous one was wrong

An earlier version of this repository (and its Zenodo archive, DOI
[10.5281/zenodo.21305263](https://doi.org/10.5281/zenodo.21305263)) accompanied a paper titled
*"A Strong Static Baseline and the Limits of Continual Learning."* **Its central conclusions do not
survive, and we withdraw them.** We say so here, at the top, because the archived version is
citable and someone may find it first.

The defect was one sort key. The harness loaded check-ins ordered by `(user, time)` and then cut
the array into 15 "streaming rounds" — so the rounds were not slices of time at all. They were
**disjoint cohorts of users**, each spanning the whole evaluation period. Every "streaming"
experiment was therefore updating on one group of users and testing on a different group. A
per-user memory was empty at scoring time for **97% of scored points** and was inert by
construction; fine-tuning had no adaptation to earn and only forgetting to pay. Both headline
claims were artifacts of that:

| claim in the superseded release | with the stream actually ordered by time |
|---|---|
| "no continual method beats a frozen model by more than a few percent" | continual methods beat it by **12–192%** |
| "naive fine-tuning catastrophically forgets and loses 8–30%" | it **gains** 13–32% |

`ranking.py`, `run_chrono.py` and `experiments/verify_*.py` in this release exist because of that
episode. The paper now reports the sort key as one of its findings (§6.7), together with two other
silent defects — one in a widely used released baseline, one more of our own (see below).

---

## What this paper finds

1. **The benchmark is a return-prediction benchmark.** Split the standard evaluation set by whether
   the user has been to the target POI before: **70–76%** of instances are *returns* on three of
   four datasets.
2. **A per-user visit counter — no neural network, no gradients, O(1) per check-in — beats two
   states of the art**, each run by us on *its own* evaluation instances: **GETNext (SIGIR'22) by
   12.6%** and **STHGCN (SIGIR'23) by 10.8%** at Acc@10. It **loses to both at Acc@1** and draws at
   MRR. Two architectures, two years apart, the same pattern: a counter knows a user's *repertoire*,
   not the *order*.
3. **Nobody is good at discovery, and the metric does not care.** Discovery accuracy tops out at
   `.065` for *any* method we ran. Even a method handed the best return *and* the best discovery
   accuracy anyone achieved would still draw **90–99%** of its score from returns.
4. **Deep models *do* learn discovery — where the data force them to.** On Gowalla-CA (the one
   discovery-dominated dataset) a neural model beats a popularity floor at discovery by **58%**,
   which a counter structurally cannot do. The benchmark pays at most **10.2%** of its score for it.
5. **A 7B language model does not change the picture.** Its best configuration reaches Acc@10
   `.4455` where the counter reaches `.6202`, and the accuracy-optimal weight for *fusing* it into
   the counter buys **0.98%** — of which a **random-noise control** reproduces half.
6. **Three silent defects, each with a mechanism and a control. Two of them are ours.**

## The three silent defects

| # | where | what | control |
|---|---|---|---|
| 1 | **our harness** | a sort key turned "streaming rounds" into user cohorts (see above) | `run_chrono.py`; invariant: *a method that never updates must be invariant to the order of the evaluation stream* |
| 2 | **released GETNext** | `TransformerEncoderLayer` built without `batch_first` while the input is `pad_sequence(batch_first=True)`, and the causal mask is `(batch, batch)` — the exact shape PyTorch validates, so nothing ever raises. **The attention runs across the mini-batch, not across time.** Also: `list(set(...))` misaligns **4,979 of 4,980** POIs between the graph and the sequence model. | `getnext_transformer_check.py` (perturbing a trajectory's own past moves its future by `0.000000`); `hashseed_experiment.py` (changing `PYTHONHASHSEED` alone re-ranks **59.5%** of its evaluation instances) |
| 3 | **our counter** | the `1e-6` popularity term exists *only* to break ties, but we accumulated in **`float32`**, whose representable step above ~8 visits **exceeds** `1e-6`. The tie-break was silently rounded away **at the top of the ranking**. It ties on 49.1% of instances in `float32` and 31.5% in `float64`. **It corrupted our own audit of defect 1**: the optimistic tie convention appeared to flatter us by +4.7%, when honestly it flatters by +0.5%. | `dtype_control.py` — same instances, same counter, same estimator, **only the dtype changes** |

Defect 3 is the one we would most like readers to take away: *a self-audit can be corrupted by the
same bug it is auditing.* Any paper reporting a counting, popularity or frequency baseline should
state the dtype it accumulated in.

## One estimator, everywhere

Every Acc@K and MRR in the paper is a **per-instance mean** with ties broken at the **expected rank**
(`ranking.py`):

```
r(p*) = |{p : s(p) > s(p*)}| + (|{p : s(p) = s(p*)}| + 1) / 2
```

Never `(s > s*).sum() + 1`, which places the target *first* among everything it ties with — a
convention that subsidises a counter (whose integer scores tie) and not a neural network (whose
float scores do not).

## Install

```bash
git clone https://github.com/Laswell1551/less-is-more-nextpoi.git
cd less-is-more-nextpoi
conda env create -f environment.yml      # the `poi` env (PyTorch 2.6 + CUDA 12.4)
conda activate poi
```

A CUDA GPU is recommended (ours: RTX 4080 SUPER, 16 GB). The code falls back to CPU. The LLM arm
needs ~10 GB of VRAM for 4-bit Qwen2.5-7B; **everything else, including the LLM fusion sweep and the
noise control, runs on CPU** — the model's raw per-candidate scores are committed.

## Quick start

Processed splits ship in `data/processed/` (~19 MB), so the main results need **no** raw download:

```bash
# the counter, on the chronological stream, 4 datasets
python count_baseline.py --datasets nyc tky gowalla_ca brightkite

# the counter against the official baselines, each on ITS OWN evaluation instances
python getnext_compare.py --name OURS-NYC     # 5,550 instances
python sthgcn_compare.py                      # 9,778 instances -- a different set entirely

# what the benchmark measures
python discovery_analysis.py --name OURS-NYC --getnext-preds as-released=../baselines/GETNext/preds_asreleased.json
python discovery_stream.py --datasets nyc tky gowalla_ca brightkite
python score_decomposition.py

# the controls that catch the three defects
python dtype_control.py          # float32 vs float64: is the tie rate real?
python tie_audit.py              # optimistic vs expected vs pessimistic ties
python clean_split_control.py    # does our 4-7% boundary overlap manufacture anything?

# tables and figures, generated FROM the artifacts (never hand-typed)
python gen_tables.py
python fig_discovery.py && python paper_figures_v2.py && python fig_llmscale.py
```

To rebuild the splits from scratch:

```bash
python download_data.py     # -> data/raw/     (Foursquare TSMC2014, Gowalla, Brightkite)
python preprocess.py        # -> data/processed/  (10-core, 24 h trajectories, dedup, T0-T5)
```

Everything, in order: `bash reproduce.sh`. Table/figure → script → output mapping: **`REPRODUCE.md`**.

## The paper checks itself

Three scripts re-derive every number in the manuscript from the artifacts and **fail loudly** on any
mismatch. They are not decoration: each one was written after a specific number in the paper turned
out to be stale.

| script | checks |
|---|---|
| `verify_manuscript.py` | every headline number traces to the artifact that produced it; every derived percentage recomputes |
| `verify_prose.py` | **all 1,288** number-literals in the prose trace to a generated table or an artifact |
| `verify_letter.py` | the response letter's structural claims (pages, tables, equations, references) match the manuscript it describes — and that the file itself is not encoding-corrupted |

Every results table is **generated** by `gen_tables.py` from the result files, so a table and the
prose cannot drift apart. They used to: the central NYC table once carried three different values
for GETNext's Acc@10, and three prose paragraphs quoted GIRAM's *best* seed while the table quoted
the mean.

## Datasets

Public, and we redistribute only our **preprocessing and chronological splits**, not the raw logs.

| Dataset | Source |
|---|---|
| Foursquare NYC / TKY | Yang et al., IEEE TSMC 2015 (TSMC2014 dump) |
| Gowalla-CA | Cho et al., KDD 2011 (SNAP `loc-gowalla`) |
| Brightkite-US | Cho et al., KDD 2011 (SNAP `loc-brightkite`) |

10-core filtering, 24 h trajectory segmentation, **consecutive-duplicate removal** (without it,
Brightkite's self-transition rate is 0.73 and every conclusion inverts — see `preprocess.py`), then a
base block `T0` and the post-`T0` stream.

## Repository map

| File | Role |
|---|---|
| `ranking.py` | **One** definition of the target's rank. Every experiment imports it. |
| `count_baseline.py` | The counting baseline, in the same harness and the same metrics as every neural method. `float64` — see defect 3. |
| `learned_continual.py` | Backbones (GRU / self-attention / GETNext-style graph), the predict-then-update protocol, and the continual-learning baselines (replay, periodic, EWC, ADER, GIRAM, GIRAM-VAE, selective). |
| `run_chrono.py` | The 5-seed streaming grid, with the rounds ordered **by time**. |
| `getnext_compare.py`, `sthgcn_compare.py` | Reconstruct each baseline's **own** evaluation instances from **its own code**, then score the counter on precisely those. The single most important discipline in the repo: GETNext scores 5,550 NYC instances, STHGCN 9,778, and putting the two numbers in one table is an error. |
| `discovery_analysis.py`, `discovery_stream.py`, `score_decomposition.py` | The return/discovery split and the arithmetic that follows from it. |
| `llm_prep.py`, `llm_audit.py` | Qwen2.5-7B in five configurations, under a 50-candidate re-ranking protocol with a **chance floor** and a **retriever ceiling**, candidates shuffled per instance. |
| `llm_alpha_sweep.py`, `tiebreak_control.py` | The fusion sweep, and the noise control that halves its apparent gain. **Both run offline, on CPU, from the committed scores.** |
| `dtype_control.py`, `tie_audit.py` | The estimator controls (defect 3, and the tie convention). |
| `getnext_transformer_check.py`, `hashseed_experiment.py` | The controls behind defect 2. |
| `clean_split_control.py`, `check_leakage.py` | Our own split's 4–7% boundary overlap, measured, plus a strictly-clean control showing it changes nothing. |
| `gen_tables.py` | Generates **every** results table from the artifacts. |
| `verify_manuscript.py`, `verify_prose.py`, `verify_letter.py` | The self-checks above. |

**Superseded, kept for transparency, not used in the paper:** `run_submission.py` (user-major
ordering — the defect), `lr_sweep.py`, `regime_sweep.py`, `criterion.py`, `controller_eval.py`,
`inject_rigorous*.py`, `mechanism_demo.py`, `final_metrics.py`, `observe.py`.

## Citation

```bibtex
@article{anonymous2026lessismore,
  title  = {Less Is More: Next Point-of-Interest Benchmarks Reward Returning,
            and a Counter Is Hard to Beat},
  author = {Anonymous for review},
  year   = {2026},
  note   = {Under review. Supersedes the release archived at doi:10.5281/zenodo.21305263,
            whose conclusions we withdraw.}
}
```

## License

MIT (see `LICENSE`). The reproduced GIRAM memory / variational generative-retrieval slot
(`run_giram`, `run_giram_vae`, `InterestVAE` in `learned_continual.py`) is a faithful
reimplementation of prior work (Wang et al., 2025), included as the strongest published continual
baseline.
