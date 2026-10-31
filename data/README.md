# Data folder

The following six released site CSV files are already present in this directory. Preserve these filenames for the reproduction workflow:

1. `01_Busan_New_Port_FILLED.csv`
2. `02_Busan_Water_Treatment_FILLED.csv`
3. `03_Dangjin_Landfill_Solar_FILLED.csv`
4. `04_Donghae_Solar_FILLED.csv`
5. `05_Gwangyang_Port_2_FILLED.csv`
6. `06_Hadong_Water_Treatment_FILLED.csv`

The notebooks validate these filenames against `configs/site_manifest.csv`.

## Target and quality columns

- Forecast target: `solar_power_filled`
- Model-imputed target indicator: `target_imputed=1`
- Recommended filled-target training mask: `recommended_for_training_filled=1`
- Recommended primary evaluation mask: `recommended_for_evaluation=1`
- Data-quality annotation: `quality_flag`

The public reproduction workflow regenerates BEDOL-core predictions and selected summary analyses from the released data. It does not require cached paper predictions or precomputed bootstrap draws. The repository does not reproduce every baseline model or all paper-level inferential procedures. Reported test metrics use directly observed, evaluation-eligible targets; filled targets may be used only where permitted for training.
