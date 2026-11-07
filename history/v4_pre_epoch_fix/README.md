# TESS Planet Explorer

Local dark-themed desktop app for real TESS two-minute light curves. Enter a TIC
ID, choose a sector, and inspect a blind BLS transit search plus an experimental
planet-like / false-positive-like classifier. It does not confirm a planet.

## Run the interface

Double-click `launch_app.bat`, or run `python app.py`. Use `python app.py --demo`
to analyze a cached example on startup. Type into the TIC dropdown to filter
saved/catalog targets; use Up/Down and Enter to select. You may enter any numeric
TIC ID manually. Known catalog status and available sectors are shown as metadata.
Sector suggestions update after selection. The dropdown includes example targets
and an editable sector control. Internet is needed only for uncached data.

The main interface shows the individual signal classification, estimated calibrated
probability when calibration checks support displaying it, period, duration, depth,
and light-curve/BLS plots. Aggregate evaluation statistics are kept in this README,
the optional evaluation dialog, and `training_report.json`. Weak periodic detections
show no forced binary classification. CSV and PNG export are available.

## Independent measured result

The **SVM** was selected using training data only. On
**88 entirely new TIC IDs**, opened once after the model and
threshold were frozen:

| Metric | Independent result |
|---|---:|
| Accuracy | 84.09% |
| Balanced accuracy | 84.42% |
| Planet precision | 89.13% |
| Planet recall | 82.00% |
| Planet F1 | 0.854 |
| False-positive recall | 86.84% |
| Accuracy Wilson 95% interval | 75.0%–90.3% |

Confusion matrix, rows=true and columns=predicted, order `[false positive, planet]`:

```text
[[33, 5], [9, 41]]
```

The 70% accuracy target was **met**
by this one independent evaluation. This measures classification of recovered
catalog signals, not accuracy across arbitrary TESS stars or survey discovery
completeness. Per-target predictions are in `outputs/independent_test_predictions.csv`.

## Dataset, labels, and leakage prevention

- 740 distinct targets were assigned before extraction,
  spanning available two-minute SPOC products in sectors 1–55.
- Training: 600 assigned stars,
  345 matched/gated stars used: 155
  vetted false positives and 190 confirmed/known planets.
- Independent test: 140 assigned new stars,
  88 matched/gated stars evaluated. Class counts are
  {'0': 38, '1': 50} (0=false positive, 1=planet).
- One observation per TIC is used. No TIC appears in both training and test.
  Every prior development-test TIC is quarantined from fitting and selection.
  Every independent test TIC is absent from all 118 previously attempted targets.
- Labels come from a saved NASA Exoplanet Archive TOI snapshot: CP/KP positives,
  FP negatives. PC, APC, FA, and uncatalogued stars are not relabeled as either
  class. Only a uniquely eligible catalog signal per TIC is used in this benchmark.
- Each search is blind. Catalog period/epoch are used afterward for association,
  never as classifier inputs or search guides. Association allows 0.5/1/2 period
  aliases, a tight period tolerance, and a BJD-to-BTJD epoch check with propagated
  catalog ephemeris uncertainties and a capped timing tolerance.
- The predeclared detection gate requires BLS SNR >= 6, pink SNR >= 3, at least two
  sampled transit windows, and >=40% of sampled events with local depth SNR >2.
  Unmatched and weak signals are recorded, not assigned fabricated negative labels.
- `data/v4/split_plan.json` fixes IDs before feature extraction. New test FITS files
  were downloaded in advance but not feature-extracted until fitting and selection
  were complete. `frozen_model.json` records model, code, split, and training-feature
  hashes. `evaluation_lock.json` prevents silently evaluating/reselecting again.

The assigned-to-accepted coverage is 345/600
for training and 88/140 for test.
Classification accuracy is conditional on that recovery/gating step. These losses
must be included when discussing end-to-end transit-finding performance.

## Preprocessing, period recovery, and features

Quality flags and finite PDCSAP flux are checked. A broad, iteratively clipped
Savitzky–Golay trend is fitted separately across observing gaps, preserving
downward transits. Upper excursions are removed. A uniform-frequency 20,000-point
BLS search covers 0.6–10 days (also limited by baseline), using an eight-minute
binned search copy. Competing peaks and half/double-period aliases are refined.
Very broad dips exceeding 15% duty cycle are excluded. The selected transit and
secondary windows are masked for a second detrending pass, followed by a final
fit on the original two-minute samples. Every inference uses this same pipeline.

The classifier uses flux-based period, duration, depth, duty cycle, SNR/SDE, peak
competition, odd/even differences, secondary-eclipse strength, event depth
consistency, shape/ingress, noise and red-noise statistics. Centroid diagnostics
measure in-transit motion after removing slow pointing variations. FITS stellar
radius/log(g)/temperature, crowding, and flux fraction support an inferred
companion-radius proxy and duration/density consistency check. These are
approximate vetting diagnostics, not fitted physical planet parameters.

The training period audit compared 42 observations also present in the
old pilot using the same current catalog ephemerides; 3 formerly unmatched
periods now associate correctly. See `outputs/period_recovery_audit.csv`.
190 accepted training planets matched the full catalog period; the
false-positive training set includes 5 half-period and 1
double-period eclipse aliases.

## Training-only model comparison

| Model | Grouped grid-CV balanced accuracy (mean ± SD) | Calibrated training OOF balanced accuracy | Minimum class recall |
|---|---:|---:|---:|
| SVM | 86.0% ± 4.1% | 84.6% | 78.1% |
| Random Forest | 84.6% ± 3.1% | 83.9% | 82.6% |
| Extra Trees | 85.0% ± 2.9% | 83.4% | 79.4% |
| Gradient Boosting | 86.0% ± 3.6% | 83.6% | 82.6% |
| Logistic baseline | 83.9% ± 3.2% | 82.1% | 77.4% |

These are **selection estimates**, not independent results: hyperparameters and
thresholds were chosen using training data. Five grouped folds tune hyperparameters;
separate outer/inner grouped folds generate sigmoid-calibrated training OOF scores
and choose a stable, near-optimal balanced-accuracy threshold. Imputation, scaling,
and transforms are fitted inside each fold. Selection rewards balanced accuracy,
penalizes fold variability, and requires useful recall for both classes where
feasible. SVM is preferred only within 0.01 of the best selection score.

Selected model parameters: `{'classifier__C': 10, 'classifier__gamma': 0.003}`. Decision threshold:
`0.500`. The frozen model is evaluated without refitting on test
stars. `svm_model.joblib` and `models/best_model.joblib` contain the selected model;
`models/svm_candidate.joblib` preserves the SVM candidate.

Training calibration ECE (five bins): 0.048.
Independent calibration ECE: 0.079;
Brier score: 0.105. Probability display is
**enabled** based on predeclared
training and independent calibration checks. Scores refer to this sampled catalog
population; class priors, magnitude selection, stellar metadata errors, other
sectors, blends, noise, and new instrument behavior can affect transfer.

## Historical results, retained in full

| Historical development model | Accuracy | Balanced accuracy | Planet F1 |
|---|---:|---:|---:|
| Initial five-feature baseline (all predictions positive) | 70.0% | 50.0% | 0.824 |
| Revised five-feature development model | 35.0% | 34.5% | 0.435 |
| Previous 11-feature model | 35.0% | 48.8% | 0.235 |

The first result was misleading because it predicted every star positive. The
later 35% results were development diagnostics on 20 observations/15 stars and
were not untouched scientific tests. Full original files/model/report/data tables
are in `history/2026-11-07_v3`; full historical evaluations are embedded in the new
`training_report.json`. The old holdout was not used for the new comparison.

## Reproduce and test

```powershell
python -m pip install -r requirements-tested.txt
python -m unittest test_transit_features.py test_pipeline.py
python verify_project.py
python smoke_test_app.py
python app.py --demo
```

The scientific checks cover transit preservation across gaps, blind period
recovery, BJD epoch association, eclipse/centroid diagnostics, and TIC leakage.
The GUI smoke test runs the actual background worker, inference, plot rendering,
selector manual-entry handling, sector suggestions, and CSV/PNG export.

Training commands are intentionally staged:

```powershell
python prepare_dataset.py
python prefetch_data.py
python train_real_tess.py --stage extract --partition training --workers 8
python train_real_tess.py --stage fit
python train_real_tess.py --stage extract --partition independent_test --workers 8
python train_real_tess.py --stage evaluate
```

The included split/test has already been opened. The final command and further
fitting deliberately refuse to reuse it for model development. For a new study,
archive `data/v4`, select additional untouched stars, and create a new split/protocol;
do not remove a lock merely to keep optimizing this test score. Existing cached
FITS files and configuration-tagged features are retained. No bootstrap uncertainty
intervals or planet confirmation are claimed in this version.

Source references: [NASA TOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html),
[TESS FITS products](https://spacetelescope.github.io/mast_notebooks/notebooks/TESS/beginner_how_to_use_lc/beginner_how_to_use_lc.html),
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
