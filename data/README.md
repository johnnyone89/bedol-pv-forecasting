# Data folder

Place the six released site files in this directory using the following English-only filenames:

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

The public reproduction workflow regenerates predictions and summary results from the released data. It does not require cached paper predictions or precomputed bootstrap draws.
