# BEDOL PV Forecasting

Reproducibility resources for **BEDOL (Bayesian Energy-Dynamics Operator Learning)**, a shared selective state-space framework for 24-step-ahead multi-site photovoltaic power forecasting.

## Reproducibility philosophy

This repository is intentionally organized so that a reader can re-run the method from the released six-site CSV files. It does **not** publish every fitted prediction, checkpoint, bootstrap draw, or intermediate paper artifact. Instead, it provides the data specification, frozen chronological split, validation-selected configuration, compact reference metrics, and a sequence of purpose-specific notebooks.

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
│   └── README.md
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

## English-only data filenames

Rename the six released CSV files exactly as follows and place them in `data/`:

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

## Main frozen protocol

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

Please cite the BEDOL paper once the final bibliographic record is available. This section will be updated after publication.
