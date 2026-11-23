# PeakWise

Forecasting next-day **evening electricity use** of Sri Lankan households with ensemble learning. PeakWise compares a 7-day rolling-median baseline with Random Forest (bagging), XGBoost (boosting) and stacking, using a chronological, leakage-tested evaluation on the RECON-SL smart-meter data. A Streamlit dashboard presents the saved results.

**Main finding:** the ensembles improve only slightly on a strong 7-day median, and stacking adds almost nothing over Random Forest.

## Objective

Given a household's own past evenings, how well can its consumption between 18:30 and 22:30 tomorrow be predicted, and do ensemble methods justify their extra complexity over a simple rule?

The **target is energy in kWh over the four-hour window** (not power in kW). A forecast for day D is made at the end of day D−1 using only information up to D−1.

## Data and attribution

The experiments use the 15-minute smart-meter files of **RECON-SL** (*Residential Electricity Consumption*), published by LIRNEasia and collaborators.

- Dataset: Zenodo record [15023048](https://zenodo.org/records/15023048) (DOI 10.21227/n1dk-q860), published 2025-02-28, licence **CC BY 4.0**.
- Authors on the record: Chanuka Ravishan Algama, Isuruni Fernando, Merl Chandana, Pasindu Ranage (LIRNEasia); Dinithi Dissanayake (National University of Singapore); Jesus Ramos (California Policy Lab); Kasun Amarasinghe (Carnegie Mellon University).
- Source repository: [LIRNEasia/SL_electricity_consumption](https://github.com/LIRNEasia/SL_electricity_consumption).

The raw data are not redistributed. The repository holds derived results (audit summaries, metrics and a pseudonymised copy of the test-period predictions), which remain subject to the CC BY 4.0 attribution requirement. Household labels in the dashboard data are salted hashes made with a secret salt that is not stored. This is **pseudonymisation, not strong anonymisation**: the source data are public, so a series could still be matched by its values.

The meter's `importkwh` column is a **cumulative register**, so consumption is the difference between consecutive readings. The audit (`src/peakwise_audit.py`, see `AUDIT.md`) streams all 45.6 million readings, keeps a household-day only if all 17 readings from 18:30 to 22:30 exist and no interval is negative or implausible, and imputes nothing. The result is **464,640 valid household-days** from 2,931 households (2023-10-01 to 2024-12-22). Where the meters' own six-hour peak register (TR2) is available, 99.6% of 335,343 comparable days agree with the extracted target within 0.01 kWh.

## Methods

- **Features** (`src/phase1.py`): 14 per household-day, all computed from the household's own earlier evenings (yesterday, same weekday last week, 7- and 28-day medians, spread, share of zero days, amount of history, calendar). Missing history stays missing.
- **Baselines:** global median, persistence, same weekday last week, 7-, 28-day and all-history medians. The **7-day rolling median** is the strongest and is the reference throughout.
- **Random Forest (bagging):** 300 deep trees on bootstrap samples, averaged; random feature subsets decorrelate the trees.
- **XGBoost (boosting):** shallow trees fitted in sequence to the remaining error, with an absolute-error loss, 1,500 rounds.
- **Stacking:** a non-negative linear meta-learner fitted on out-of-fold predictions of Random Forest, two XGBoost variants and the baseline. The weights are not forced to sum to one.
- **Validation** (`src/phase2.py`): train 2023-10-01 to 2024-09-30, validation 2024-10-01 to 2024-11-10, test 2024-11-11 to 2024-12-22. Tuning used only expanding chronological folds inside the training period (`artifacts/phase2/TUNING_AND_AI_LOG.md`). The test period was read once, by models refitted on training plus validation. Notebook 01 had already printed test-period *baseline* scores, so the test period is not blind for the baselines. `tests/check_leakage.py` and `tests/check_phase2.py` corrupt future targets and require earlier features and stacking predictions to stay unchanged.

## Results

Errors are in kWh per household-evening. Test: 58,404 household-days from 2,120 households; validation: 82,375.

| Model | Test MAE | Test RMSE | Test vs baseline (MAE / RMSE) | Validation MAE | Validation RMSE |
|---|---|---|---|---|---|
| 7-day median (baseline) | 0.2917 | 1.7052 | – | 0.3159 | 1.0106 |
| Random Forest | 0.2862 | 1.6068 | −1.9% / −5.8% | 0.3105 | 0.9321 |
| XGBoost | 0.2997 | 1.8181 | +2.7% / +6.6% | 0.3079 | 1.0342 |
| Stacking | 0.2872 | 1.6064 | −1.5% / −5.8% | 0.3130 | 0.9360 |

- The 7-day median is a strong baseline; Random Forest improves on it modestly (the 95% household-level bootstrap interval for the test MAE difference, −0.0100 to +0.0004 kWh, touches zero; the RMSE gain is clearer).
- XGBoost does not reliably beat the baseline, although it has the lowest median error.
- Stacking gives no gain over Random Forest, which received 0.95 of the meta-learner's weight.
- Unusually high evenings (above the training 99th percentile) are not forecast well by any model: 409 test household-days from 21 households.

## Limitations

- Only **previously observed households** are evaluated; performance for new households is not shown.
- Households come from several Sri Lankan cities, so results should not be assumed to represent the whole country. The survey part of the dataset is not used.
- Test-period targets **cannot be cross-checked** against the meters' TR2 register (no usable six-hour bracket after 31 October 2024).
- Validation and test are six-week windows: they show no seasonality and are probably easier than an average stretch. Whole-day data gaps are common.
- Features use only a household's own history; there is no weather, tariff or occupancy information.
- The stacking meta-learner minimises squared error while the score is MAE; an absolute-error meta-learner was not tried.

## Running the dashboard

Tested with Python 3.9 on macOS.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard reads only `dashboard/data/` (about 1 MB). It needs no raw data and no retraining, and is a retrospective view of saved results, not a live forecasting service.

## Notebooks and reproducing the experiments

`notebooks/` holds `00` exploratory analysis, `01` features, validation and baselines, and `02` ensemble comparison. They are committed with their outputs and can be read on GitHub.

The raw data (about 1.2 GB) must be downloaded separately. Install `requirements-dev.txt` (on macOS XGBoost also needs `brew install libomp`), then:

```bash
mkdir data_zenodo
curl -L -o data_zenodo/data.zip "https://zenodo.org/api/records/15023048/files/data.zip/content"
unzip -q data_zenodo/data.zip -d data_zenodo          # only needed for notebook 00

python src/peakwise_audit.py --zip data_zenodo/data.zip --verify-md5 --out artifacts/audit   # ~2 min
python src/phase1.py                                  # features, splits, baselines
python src/phase2.py all                              # tuning and final fits (~20 min)
python src/build_dashboard_data.py                    # rebuilds dashboard/data/
```

The audit checks the archive md5 (`dcdac5809b4da6b6362519d33112a5f9`). The model table, household-level predictions with the dataset's own identifiers and the large audit CSVs are regenerated by these steps and are not stored in the repository. The trained models are stored in `artifacts/models/` (Random Forest, two XGBoost models and the stacking weights, refitted on training plus validation; about 44 MB). Load the `.joblib` files only from this repository, because joblib files are pickles.

**Reproduction status.** Each stage was run during development, and the dashboard and audit tests were also run on a simulated clean checkout. The whole chain has **not** been re-executed end to end from a fresh clone, and results have not been verified bit-for-bit. Seeds are fixed, but multi-threaded training can change values at floating-point level, so expect numbers to match closely, not necessarily to the last digit. Tuning was run twice: the first run capped XGBoost at 600 rounds and the cap was raised to 1,500 before any test row was read. Dashboard pseudonyms change on each rebuild unless `PEAKWISE_SALT` is set.

## Tests

```bash
python -m unittest tests.test_dashboard tests.test_regressions tests.test_tr2   # no raw data needed
python tests/check_leakage.py && python tests/check_phase2.py                  # need the regenerated model files
```

## Repository layout

```
app.py                  Streamlit entry point
dashboard/              dashboard pages, styling and its ~1 MB result data (dashboard/data/)
src/                    audit (peakwise_audit.py, probe_dbe.py, spot_check.py), phase1.py, phase2.py, build_dashboard_data.py
notebooks/              00 exploratory analysis, 01 features and baselines, 02 ensemble comparison
artifacts/
  models/               trained Random Forest, XGBoost and stacking weights
  phase1/, phase2/      saved metrics and the tuning/AI log
  audit/                audit report and DBE probe summary
tests/                  audit and dashboard tests, leakage checks
AUDIT.md                data audit findings
```

## AI-assisted development

This project was developed with AI assistance. The data audit, experiment code, tests, notebooks and dashboard were drafted and iterated in interactive sessions with Claude (Anthropic), first in the claude.ai chat interface and then in Claude Code, under the author's direction. Design choices, tuning decisions and review corrections are recorded in `artifacts/phase2/TUNING_AND_AI_LOG.md`. Results come from the code and saved outputs in this repository and are covered by the tests above.
