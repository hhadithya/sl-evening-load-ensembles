# PeakWise

Next-day evening electricity consumption forecasting for Sri Lankan households, built as an ensemble-learning study on the
RECON-SL smart-meter dataset. It compares a rolling-median baseline with Random Forest (bagging), XGBoost (boosting) and a stacked
combination, under a chronological, leakage-tested evaluation. The repository contains the data audit, the experiment code and
notebooks, and a Streamlit dashboard that presents the saved results.

The study's main finding is modest: ensembles improve only slightly on a strong 7-day rolling median, and stacking adds almost nothing over Random Forest.

## Objective and context

RECON-SL combines smart-meter readings with household surveys for households in several Sri Lankan cities. Its meters record separate day, peak and off-peak registers, which
is time-of-use metering, and the data audit in this repository confirmed that the meters' peak register corresponds to the window 18:30-22:30 (local time). Evening consumption is therefore
a natural quantity to forecast. The question studied here is narrow and practical: given a household's own past evenings, how well can its consumption between 18:30 and 22:30 on the next day
be predicted, and do ensemble methods justify their extra complexity over a simple rule?

The prediction target is **energy in kWh over the four-hour window** (not instantaneous power in kW). A forecast for day D is made at the end of day D-1 using only
information up to D-1.

## Data and attribution

The experiments use the 15-minute smart-meter files of **RECON-SL**, the *Residential Electricity Consumption* dataset published by LIRNEasia and collaborators.

- Dataset: Zenodo record [15023048](https://zenodo.org/records/15023048) (DOI 10.21227/n1dk-q860), published 2025-02-28, licence **CC BY 4.0**.
- Authors listed on the record: Chanuka Ravishan Algama, Isuruni Fernando, Merl Chandana, Pasindu Ranage (LIRNEasia); Dinithi Dissanayake (National University of Singapore);
  Jesus Ramos (California Policy Lab); Kasun Amarasinghe (Carnegie Mellon University).
- Source repository and documentation: [LIRNEasia/SL_electricity_consumption](https://github.com/LIRNEasia/SL_electricity_consumption).

This repository does not redistribute the raw data. It contains derived results (audit summaries, metrics, figures and a pseudonymised copy of the test-period predictions) that are adapted from the dataset and
shared with the attribution above; they remain subject to the CC BY 4.0 attribution requirement. The dataset's own household identifiers are not published in the stored predictions: the dashboard data replaces them with
salted-hash labels, drawn with a secret salt that is not stored. This is **pseudonymisation, not strong anonymisation**: the source data are public, so someone holding them could still match a series by its values.
A few of the dataset's identifiers appear in the audit summary, `AUDIT.md` and the notebooks, where individual households are discussed as examples.

## Data audit

The meter's `importkwh` column is a **cumulative register**, so consumption in an interval is the difference between consecutive readings. The audit (`peakwise_audit.py`, described in `AUDIT.md`) streams all
45,613,280 15-minute readings with bounded memory, checks timestamps, duplicates and register behaviour, and keeps a household-day only if all 17 readings from 18:30 to 22:30 exist and no interval is negative or
implausible. Nothing is imputed. The result is **464,640 valid household-days** from 2,931 households
(2023-10-01 to 2024-12-22). About 54% of quarter-hours inside household histories are missing, mostly as whole days. The audit was cross-checked against the meters' own
six-hour peak register (TR2) for the days where that register is available: 99.6% of the 335,343 comparable days agree within 0.01 kWh, and shifting the window by one 15-minute slot breaks the agreement.

## Methodology

**Features** (`phase1.py`, 14 per household-day): yesterday's evening, the same weekday last week, the last observed evening and the days since it, 7- and 28-day rolling medians, the 28-day spread and share of zero evenings, counts of observed days, the
all-history median, and day of week, weekend flag and month. All are computed from the household's own earlier evenings, shifted so that day D cannot see day D or later. Missing history is kept as missing, and the number of observed days is given to the models.

**Baselines.** Global median, persistence (yesterday), same weekday last week, 28-day median, all-history median and the **7-day rolling median**, which is the strongest and the reference for every comparison. Every baseline answers on every row through an explicit fallback
(for example, the last observed evening when yesterday is missing).

**Models** (`phase2.py`).
- *Bagging, Random Forest:* 300 deep trees on bootstrap half-samples with a random 40% of features per split (minimum leaf 50 rows), averaged. Reduces variance; the feature subsampling decorrelates the trees.
- *Boosting, XGBoost:* shallow trees fitted sequentially to what the ensemble still gets wrong, with an absolute-error loss (targets the median, which suits a skewed target and an MAE score), depth 6, 1,500 rounds, learning rate 0.05. Reduces bias; regularisation and rounds control overfitting.
- *Stacking:* a non-negative linear meta-learner on **out-of-fold** predictions of Random Forest, two XGBoost variants and the 7-day median. Out-of-fold predictions come from chronological folds, so a base model never predicts rows it was fitted on or that come before its training data. The weights are non-negative and not forced to sum to one.

**Chronological validation.** Training 2023-10-01 to 2024-09-30; validation 2024-10-01 to 2024-11-10; test 2024-11-11 to 2024-12-22. Hyperparameters were tuned only on three expanding chronological folds inside the training period (a small, two-stage search recorded in
`outputs/phase2/TUNING_AND_AI_LOG.md`). Model families were compared on validation; the test period was read once, by models refitted on training plus validation with frozen settings. The Phase 1 notebook had already printed test-period baseline scores, so the test period is not blind for the baselines.
Executable tests (`check_leakage.py`, `check_phase2.py`) corrupt future targets and require every earlier feature and stacking prediction to be unchanged.

## Results

Test period, 58,404 household-days from 2,120 households that appear in the fitting data. MAE, RMSE and median error are in kWh per household-evening; the last column is the change against the 7-day median (MAE / RMSE).

| Model | MAE | RMSE | Median error | vs baseline |
|---|---|---|---|---|
| 7-day rolling median (baseline) | 0.2917 | 1.7052 | 0.1090 | – |
| Random Forest | 0.2862 | 1.6068 | 0.1139 | -1.9% / -5.8% |
| XGBoost | 0.2997 | 1.8181 | 0.1063 | +2.7% / +6.6% |
| Stacking | 0.2872 | 1.6064 | 0.1143 | -1.5% / -5.8% |

Validation period, 82,375 household-days:

| Model | MAE | RMSE | Median error | vs baseline |
|---|---|---|---|---|
| 7-day rolling median (baseline) | 0.3159 | 1.0106 | 0.1165 | – |
| Random Forest | 0.3105 | 0.9321 | 0.1228 | -1.7% / -7.8% |
| XGBoost | 0.3079 | 1.0342 | 0.1138 | -2.6% / +2.3% |
| Stacking | 0.3130 | 0.9360 | 0.1246 | -0.9% / -7.4% |

- **The 7-day rolling median is a strong baseline.** On the same tuning folds it scores MAE 0.4183 and RMSE 1.2522.
- **Random Forest improves modestly:** test MAE -1.9% and RMSE -5.8%. The 95% household-level bootstrap interval for the MAE difference is -0.0100 to +0.0004 kWh, which touches zero; the RMSE gain is the clearer one
  (-0.160 to -0.039 kWh).
- **XGBoost does not reliably beat the baseline** (test MAE +2.7%, RMSE +6.6%), although it has the lowest median error.
- **Stacking does not improve on Random Forest** (test MAE 0.2872 vs 0.2862); the meta-learner placed 0.95 of its weight on Random Forest (validation fit: 0.91).
- **Where models fail.** Ensembles beat the baseline on typical evenings but are worse on exactly-zero and very quiet evenings, and no model forecasts unusually high evenings (above the training 99th percentile) well: mean error -1.9 kWh for the baseline and -4.5 kWh for Random Forest, on 409 household-days from 21 households.
- **The gain is thin across households:** 55% of households improve with Random Forest, and the median household's MAE changes by -0.0025 kWh.

## Limitations

- Evaluation covers **previously observed households**; it does not show performance for entirely new households.
- The households come from several Sri Lankan cities (the meter files list utility branches such as Moratuwa, Nugegoda, Kalutara, Kelaniya and Galle), so the results should not be assumed to represent the whole country. The survey data in the dataset is not used.
- The test-period targets could not be cross-checked against the meters' TR2 register: there is no usable six-hour bracket after 31 October 2024. On validation, the cross-check is confounded with calendar month.
- Validation and test are six-week windows, so they cannot show seasonality and are probably easier than an average stretch of the dataset.
- Data are irregular: whole-day gaps are common, and there is almost no usable data between mid-January and March 2024.
- The extreme-consumption conclusions rest on very few households. Nothing in the data explains *why* models fail there; explanations in the dashboard are labelled as untested.
- Features use only a household's own history. No weather, tariff, holiday or occupancy information is available.
- The stacking meta-learner minimises squared error while the score is MAE; a meta-learner trained on absolute error was not tried.

## Installation and dashboard

Tested with Python 3.9 on macOS.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard reads only `dashboard/data/` (about 1 MB) and needs no raw data, trained models or retraining. It is a retrospective view of saved results, not a live forecasting or monitoring service.
For hosting (for example Streamlit Community Cloud) the main file is `app.py` and the dependencies come from `requirements.txt`. The Forecast Explorer lets a visitor browse individual pseudonymised households for the test period; consider limiting it to its aggregate view in a public deployment.

## Reproducing the experiments

The raw data (about 1.2 GB compressed) must be obtained separately. Install the full environment with `pip install -r requirements-dev.txt` (on macOS, XGBoost also needs `brew install libomp`), then:

```bash
mkdir data_zenodo
curl -L -o data_zenodo/data.zip "https://zenodo.org/api/records/15023048/files/data.zip/content"
unzip -q data_zenodo/data.zip -d data_zenodo        # only needed for the EDA notebook

python peakwise_audit.py --zip data_zenodo/data.zip --verify-md5 --out audit_out   # ~2 min, ~1 GB RAM
python phase1.py                                                                   # features, splits, baselines (~10 s)
python phase2.py all                                                               # tuning and final fits (~20 min)
python build_dashboard_data.py                                                     # rebuilds dashboard/data/
```

The audit verifies the archive checksum (md5 `dcdac5809b4da6b6362519d33112a5f9`). The trained models, the Phase 1 model table, the household-level prediction files with the dataset's own identifiers and the large audit CSVs are regenerated by these steps and are not stored in the repository.
The notebooks in `notebooks/` are committed with their outputs and can be read on GitHub; re-running them requires the steps above.

**Reproduction status, stated plainly.** Each stage was run during development: the full audit on the Zenodo archive, the Phase 1 pipeline, tuning and final fits, and the dashboard build. The unit and leakage tests were run, and the dashboard and audit tests were also run on a simulated clean checkout without the raw data.
The whole chain has **not** been re-executed end to end from a fresh clone, and results have **not** been verified bit-for-bit. Seeds are fixed (`random_state=42`), but multi-threaded training and summation can change values at floating-point level (an identical rebuild of the dashboard data differed from an earlier one by about 1e-16 in the permutation importances).
Phase 2 tuning was run twice: the first run capped XGBoost at 600 rounds, and the cap was raised to 1,500 after several settings hit it (both logs are in `outputs/phase2/`). Expect reproduced numbers to match the reported ones closely, not necessarily to the last digit. The pseudonym labels in the dashboard data change on every rebuild unless `PEAKWISE_SALT` is set.

## Tests

```bash
python -m unittest tests.test_dashboard tests.test_regressions tests.test_tr2   # dashboard and audit tests, no raw data needed
python check_leakage.py && python check_phase2.py                              # need the regenerated Phase 1 and 2 files
```

## Repository layout

| Path | Purpose |
|---|---|
| `peakwise_audit.py`, `probe_dbe.py`, `spot_check.py`, `AUDIT.md`, `audit_out/` | Raw-data audit, its checks and findings, and two small audit summaries |
| `phase1.py`, `check_leakage.py` | Features, chronological splits, baselines, leakage tests |
| `phase2.py`, `check_phase2.py` | Tuning, leakage-safe stacking, evaluation, tests |
| `notebooks/` | `00` exploratory analysis, `01` features and baselines, `02` ensemble comparison |
| `outputs/` | Saved metrics, figures and tuning logs (`outputs/phase2/TUNING_AND_AI_LOG.md`) |
| `build_dashboard_data.py`, `dashboard/`, `app.py`, `.streamlit/` | Streamlit dashboard and its lightweight data |
| `tests/` | Audit and dashboard tests |

## Licence

The original project code is released under the **MIT Licence** (see `LICENSE`). This covers the code only. The RECON-SL dataset and results derived from it are **not** covered by the MIT Licence: they remain under
**CC BY 4.0** and require the attribution given above.

## AI-assisted development

This project was developed with AI assistance: the data audit, experiment code, tests, notebooks and dashboard were drafted and iterated in interactive sessions with Claude (Anthropic), first in the claude.ai chat interface and then in
Claude Code, under the author's direction. The experimental design choices, tuning decisions and corrections made during review are recorded in `outputs/phase2/TUNING_AND_AI_LOG.md`.
Results in this repository are produced by the code and saved outputs it contains and are covered by the tests above.
