# BEDOL PV Forecasting

Reproducibility resources for **BEDOL (bounded energy dynamics with observation-conditioned latents)**, a shared selective state-space framework for 24-step-ahead multisite photovoltaic (PV) power forecasting.

## Methodological scope

BEDOL learns an input-dependent **shared selective-state transition** from historical observations across six South Korean PV sites. **Site identity is introduced downstream of the shared temporal encoder**, rather than conditioning the shared transition itself. A recurrent decoder produces a 24-step forecast. The stochastic variational latent layer is used for **uncertainty representation**; the ablation evidence does not establish an improvement in point accuracy from this layer.

The state update has a bounded invariant domain by construction. A separate analysis provides **sufficient conditions** for local contraction and perturbation control; it does **not** certify global contractivity of the fitted network. BEDOL is a finite-dimensional discrete state-transition model, **not a neural operator in the function-space sense**, and is not presented as a Mamba implementation. Claims of cross-site sharing concern the **six observed sites**, not zero-shot transfer to unseen sites.

## Reproducibility philosophy

This repository supports re-execution of the **released BEDOL core workflow** using six public site CSV files. It does **not** contain every fitted prediction, checkpoint, bootstrap draw, full common-protocol baseline implementation, or complete MCS/SPA analysis underlying the paper. Instead, it provides the data specification, fixed chronological split, validation-selected configuration, compact paper-reference metrics, and purpose-specific notebooks for BEDOL training, evaluation, selected ablations, grouped XAI, and uncertainty diagnostics.

Exact floating-point values can vary slightly across GPU models, CUDA/cuDNN versions, and PyTorch releases. The goal is faithful re-execution of the protocol and recovery of the same qualitative findings, not bit-for-bit equality.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── chronological_splits.csv
│   ├── data_dictionary.csv
│   ├── experiment_config.csv
│   └── site_manifest.csv
├── data/
│   ├── README.md
│   └── 01_..._FILLED.csv through 06_..._FILLED.csv
├── notebooks/
│   ├── 00_data_audit_and_split.ipynb
│   ├── 01_train_bedol_core.ipynb
│   ├── 02_evaluate_and_statistics.ipynb
│   ├── 03_ablation_study.ipynb
│   └── 04_xai_and_uncertainty.ipynb
├── reference/
│   └── paper_reference_metrics.csv
└── src/
    └── bedol_repro.py
```

## Released six-site data files

The following six CSV files are **already included** in `data/`. Keep these filenames unchanged when running the notebooks:

- `01_Busan_New_Port_FILLED.csv`
- `02_Busan_Water_Treatment_FILLED.csv`
- `03_Dangjin_Landfill_Solar_FILLED.csv`
- `04_Donghae_Solar_FILLED.csv`
- `05_Gwangyang_Port_2_FILLED.csv`
- `06_Hadong_Water_Treatment_FILLED.csv`

## Notebook workflow

### 00 — Data audit and chronological split
Checks filenames, hashes, target completeness, quality flags, and the frozen split rule. It writes only compact audit files.

### 01 — Train BEDOL core
Rebuilds leakage-safe preprocessing, trains the frozen BEDOL configuration, saves fresh checkpoints, and generates held-out predictions. `FAST` mode is provided for a quick pipeline check; `PAPER` mode uses seeds 11, 29, and 47 and the longer training budget.

### 02 — Evaluate and statistical inference
Reads the newly generated predictions, computes site-level and macro metrics, compares them with compact reference values, and runs a moving-block bootstrap sanity check.

### 03 — Ablation study
Retrains only the mechanism variants needed to test the main architectural claims: fixed transition, no site embedding, no Bayesian latent, no NLinear anchor, and no teacher forcing. It is intentionally separate because these runs are computationally expensive.

### 04 — XAI and uncertainty
Uses the freshly trained BEDOL checkpoint(s) for grouped blocked-permutation model-reliance analysis and Monte Carlo latent sampling for interval diagnostics. It does not use cached XAI or uncertainty results.

## Validation-locked protocol

- Historical lookback: **168 h**
- Forecast horizon: **24 h**
- Last complete year: **test**
- Preceding complete year: **validation**
- Earlier period: **training**
- Training / validation / test origin strides: **6 / 12 / 6 h**
- Canonical encoder: **SelectiveSSM**
- Final paper seeds: **11, 29, 47**
- Architecture and hyperparameters are selected using validation data only. The held-out test year is not used for model selection.

Primary evaluation respects `recommended_for_evaluation=1`. Filled-target training can use `recommended_for_training_filled=1`. See `data/README.md` and `configs/data_dictionary.csv`.

## Running the reproduction

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in numerical order. Start with `FAST` mode in Notebook 01 to verify the environment before launching the longer paper-style run.

## Reference values

`reference/paper_reference_metrics.csv` contains only a small set of paper-level reference values for sanity checking. It is not a cache of the full results. Newly generated predictions and analyses are written locally under `reproduction_outputs/`.

## Citation

For use of this repository's data and code, cite this repository using the **Cite this repository** option (generated from `CITATION.cff`). The associated manuscript is *BEDOL: Bounded Energy Dynamics with Observation-Conditioned Latents for 24-Step-Ahead Multisite Solar Power Forecasting* (Jihoon Moon and Seungmin Rho); its journal citation and DOI should be added only after publication.

Repository: https://github.com/johnnyone89/bedol-pv-forecasting

## Licensing and reuse

No software or dataset license is currently declared in this repository. Public availability should not be interpreted as permission for unrestricted reuse or redistribution. Verify the rights to the six source datasets before adding a license.
