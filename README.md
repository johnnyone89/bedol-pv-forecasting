# BEDOL PV Forecasting

Reproducibility resources for **BEDOL (Bayesian Energy-Dynamics Operator Learning)**, a shared selective state-space model for 24-step-ahead multi-site photovoltaic power forecasting.

## What this repository is for

This repository is designed to let readers **re-run the modeling pipeline from the released six-site hourly data** and obtain results that are close to the reported findings. It does **not** store the full set of paper outputs, fitted predictions, bootstrap draws, or every intermediate artifact. Small configuration and reference CSV files are included so that a fresh run can be audited against the paper protocol.

> Reproducibility note: exact floating-point values can vary slightly with GPU model, CUDA/cuDNN, PyTorch version, and stochastic optimization. The notebook fixes random seeds and the chronological split, but the goal is faithful re-execution rather than bit-for-bit reproduction.

## Minimal repository layout

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── experiment_config.csv
│   ├── site_manifest.csv
│   └── chronological_splits.csv
├── reference/
│   └── paper_reference_metrics.csv
└── notebooks/
    └── BEDOL_reproduce_core.ipynb
```

Place the six released `*_FILLED.csv` site files under `data/` (or change `DATA_DIR` in the notebook). The notebook accepts the filenames listed in `configs/site_manifest.csv`.

## Main protocol

- Lookback: **168 h**
- Forecast horizon: **24 h**
- Chronological split: **last 1 year = test; preceding 1 year = validation; earlier data = training**
- Train / validation / test window strides: **6 / 12 / 6 h**
- Canonical BEDOL encoder: **SelectiveSSM**
- Final seeds used in the paper pipeline: **11, 29, 47**
- Architecture and hyperparameters are locked using validation data only; the test year is not used for model selection.

The data release preserves quality and provenance columns. For primary evaluation, use rows marked `recommended_for_evaluation=1`. For filled-target training, `recommended_for_training_filled=1` can be used. The target is `solar_power_filled`.

## How to reproduce

1. Install the environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Put the six site CSV files in `data/`.
3. Open `notebooks/BEDOL_reproduce_core.ipynb`.
4. Run **FAST mode** first to validate the pipeline.
5. Switch to **PAPER mode** for the three-seed, longer training configuration.
6. Compare the newly produced summary table with `reference/paper_reference_metrics.csv`.

The notebook also includes optional switches for the main mechanism checks (fixed transition, no site embedding, no Bayesian latent, no NLinear anchor, no teacher forcing), a moving-block bootstrap comparison, and grouped blocked-permutation XAI. Those switches are off by default so the core reproduction remains practical.

## What is and is not claimed

The released code is intended to reproduce the **method and experimental protocol**. It is not a cache of the published outputs. Small differences in metrics are expected across software/hardware environments. The central qualitative pattern to verify is that the shared, site-conditioned selective dynamics remain competitive under the same chronological protocol and that the fixed-transition / independent or de-conditioned variants are weaker when the full paper experiment is run.

## Data integrity

The six-site release contains 278,232 hourly rows in total across six sites. The site manifest records date ranges, row counts, quality flags, recommended training/evaluation counts, and SHA-256 hashes. See `configs/site_manifest.csv` and the original release manifest for details.

## Citation

If you use this repository, please cite the BEDOL paper once the final bibliographic record is available. The citation block will be updated after publication.
