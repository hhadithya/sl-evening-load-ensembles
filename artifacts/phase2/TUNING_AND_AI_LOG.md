## 11. Record of tuning decisions and AI-assisted implementation

### Tuning decisions (in the order they were made)

| # | Decision | Evidence it was based on | Used test data? |
|---|---|---|---|
| 1 | Tune only on the 3 chronological folds inside the training period; compare families on validation. | Design fixed before any fit. `load_table(include_test=False)` never loads a test row. | No |
| 2 | Selection rule: lowest mean cross-validated MAE within each family. | Fixed before tuning. | No |
| 3 | XGBoost stage 1 explores *what is optimised* (absolute error / squared error / squared error on log1p) × depth (3, 6). | The score is MAE and the target is right-skewed; the MAE-optimal forecast is the median. | No |
| 4 | XGBoost stage 2 varies regularisation (`min_child_weight` 10/200, `reg_lambda` 10, depth 9 or 5) around the winners. | Stage-1 CV results. | No |
| 5 | Random Forest grid: `min_samples_leaf` {10, 50} × `max_features` {0.4, 1.0} × target {raw, log1p}; tree count not tuned (100 for tuning, 300 final). | More trees only reduce variance; leaf size and feature subsampling set bias and tree correlation. | No |
| 6 | **Rerun with the XGBoost round cap raised from 600 to 1,500.** | First run: several configurations (notably the absolute-error ones) chose the 600 cap, so the result was boundary-limited. The first-run log is kept locally and not in the repository. | No (CV only; test never loaded) |
| 7 | Selected: Random Forest `min_samples_leaf=50, max_features=0.4`, raw target, 300 trees; XGBoost absolute error, depth 6, `min_child_weight=50`, `reg_lambda=10`, 1,500 rounds; second XGBoost squared error on log1p, depth 6, 150 rounds. | Lowest CV MAE in each family. | No |
| 8 | Stacking: non-negative linear meta-learner on out-of-fold predictions of the three tuned models plus the 7-day median. No other meta-learner was tried. | Four correlated inputs and a skewed target: a flexible meta-learner would mostly add variance. | No |
| 9 | Final test: refit on train+validation with frozen settings, evaluate once. | Standard protocol. | Test predicted once |
| 10 | After the test predictions existed, **only** warnings handling, tests and reporting code changed. No model setting, selection or artifact was altered. | `src/phase2.py final` was run once; model files are unchanged since. | n/a |

### Corrections from the independent review, and how each was handled
* `macro_MAE` was the median of per-household MAEs: renamed; `hh_MAE_mean` and `hh_MAE_median` are both reported.
* The Phase 1 notebook had already shown test baselines: acknowledged in the Phase 2 notebook; ensemble selection used folds and validation only.
* Evaluation rows from households absent from training: 125 validation rows (26 households), none in test; reported separately, excluded from headline numbers.
* Baselines with partial coverage: every baseline now answers on every row through an explicit fallback chain, and every comparison uses the same rows.

### Things that went wrong or surprised us (kept deliberately)
* XGBoost was installed but could not load until `libomp` was installed (a missing system library, not a code issue).
* NNLS emitted spurious "divide by zero / overflow / invalid value in matmul" warnings on macOS (Accelerate BLAS) for finite inputs. The weights were verified against an independent least-squares solution, and only that warning is silenced.
* The first draft of the XGBoost search was boundary-limited (decision 6).
* The `tr2_check` robustness check cannot be run on the test period (no six-hour bracket after 31 October 2024) and on validation it is confounded with the calendar month.

### AI-assisted implementation statement
This work was produced with **Claude (Anthropic) via Claude Code**, working interactively with the project owner. The AI wrote the code in `src/phase2.py`, `tests/check_phase2.py` and the notebook builder, proposed the experimental design, ran the experiments and drafted the interpretation. The owner supplied the project scope, the independent review corrections and the decision to treat the test period as read-once. All numbers in this notebook are computed by the code in the repository from the saved artifacts; none were typed in from memory, and the markdown commentary was checked against the printed outputs after execution. The AI's design choices (the search space, the selection rule, the meta-learner, the evaluation population) are recorded above so that they can be challenged. Reproduce everything with `python phase2.py all` followed by running this notebook.
