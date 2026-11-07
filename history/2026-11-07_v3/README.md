# TESS Planet Explorer

Desktop app for downloading real TESS two-minute SPOC light curves, finding
the strongest periodic dip using Box Least Squares, and scoring the signal
with an SVM trained on real catalog-labeled observations.

## Open the interface

**Current model limitation:** the final pilot correctly classified 7 of 20
development observations (35% accuracy, 48.8% balanced accuracy). Planet recall
is 14.3%, and false-positive recall is 83.3%. This model is not reliable enough
for scientific planet/nonplanet decisions. The interface explicitly displays
that limitation. The download, BLS, feature extraction, fitting, and prediction
workflow runs on real data, but usable classification performance has not been
achieved. A larger and better balanced training sample, stronger validation,
and further feature/model work are required.
All 192 light curve files were downloaded. Two detections had transit windows
so wide that there were insufficient out-of-transit samples for vetting; those
were rejected, leaving 190 analyzed observations and 96 catalog-matched rows.

Double-click **launch_app.bat**, or run from this folder:

```powershell
python app.py
```

Use `python app.py --demo` to analyze a cached development-test star on startup.

1. Click **Use held-out test star** for a cached observation that was excluded
   from model fitting and tuning.
2. Or enter a numeric **TIC ID** and **sector**, then click **Analyze TIC**.
3. Inspect the highlighted transit windows, folded profile, and BLS search.
4. **Open local FITS** accepts official TESS two-minute PDCSAP light curves.
5. **Export result** writes the prediction CSV and a matching PNG plot.

Internet is required for uncached data. Previously downloaded FITS files are
reused. Analysis runs in a background thread; the interface remains responsive.
Downloads use the official public TESS AWS mirror with a MAST archive fallback.
This is on-demand analysis of released archive observations, not live streaming
from the satellite. Depending on the computer and connection, expect seconds
to a few minutes for download and BLS analysis.

The model's search range is 0.6–10 days, limited further by observation baseline.
It searches a ten-minute-binned copy on 6,000 trial periods, then refines the
strongest period on the original two-minute samples over a 400-point local grid.
Flux-scatter features and transit plots use the original two-minute samples.
The SVM uses period, duration, depth, flux scatter, duty cycle, transit SNR,
odd/even depth significance, secondary-to-primary depth ratio, center-to-edge
shape ratio, transit-to-transit depth scatter, and a two-hour red-noise ratio.
Zero-valued vetting diagnostics receive a small positive floor for logarithms.
Only the strongest signal is classified. It does not search for every planet
in a system, model grazing geometry, or validate a planetary discovery.

## Real training data and evaluation

See **training_report.json** for the actual sample counts and held-out metrics.
Run `python verify_project.py` to verify saved-model predictions, confirm that
train/test stars do not overlap, and export `outputs/evaluation.png` plus the
per-observation held-out predictions.
The included real-data pilot uses sectors 1–3 SPOC 120-second PDCSAP light curves
from MAST, with current NASA Exoplanet Archive TOI labels:

- Positive: CP (confirmed) or KP (previously known planet).
- Negative: FP (cataloged false positive).
- Candidates, ambiguous cases, and uncatalogued stars are not training labels.

Training runs a **blind BLS search** using the same preprocessing as the app.
Only detections matching a catalog signal are labeled. The catalog period is
used only for label association, never as an SVM feature or to guide the search.
Association allows periods within 1% of the catalog period or its 0.5/2 aliases,
plus an epoch check. Timing tolerance is the larger of 0.1 days or twice the
catalog duration, capped at 15% of the detected period. Epochs are converted
from catalog BJD to TESS BTJD using BJD − 2457000.

Approximate 80/20 train/test partitions keep stars separate. Hyperparameter
tuning, scaling, and sigmoid probability calibration use only the training
partition. Test IDs and per-star predictions are saved for inspection.
Positive features are log-transformed before scaling. Hyperparameters maximize
training grouped-CV balanced accuracy. The binary threshold maximizes balanced
accuracy on training-only out-of-fold calibrated probabilities; it is not fixed
at 0.5. Calibration folds also keep stars separate.
The model retained in `svm_model.joblib` is the evaluated model; test stars
are not added back into training afterward.
Multiple sector observations of one star always stay in the same partition.
Reported classification metrics count observations, and the report separately
lists distinct star counts. Recovery failures are listed separately rather
than being relabeled as negatives.

The first baseline predicted every held-out observation as a planet. Its report
is retained in `outputs/initial_baseline_report.json`. That diagnostic motivated
class-balanced selection and a training-only threshold. Because the first
holdout results were inspected during development, the revised results are
**development holdout** metrics, not a final untouched scientific test. A fresh
star-disjoint evaluation set is needed before claiming generalization accuracy.
The subsequent five-feature development result is also preserved in
`outputs/five_feature_development_report.json`. The final feature set adds
flux-based transit-vetting measurements; it does not use catalog dispositions,
catalog ephemerides, or TIC IDs as classifier inputs.

**Limits:** This is a small, catalog-selected pilot, not the proposed 2,000-curve
study. Metrics apply to recovered catalog signals, not all downloaded stars or
a blind survey. The negatives are astrophysical false positives; pure noise
and systematic artifacts are not a separate trained class. An arbitrary star
will still have a strongest BLS peak, so a model score alone cannot establish
whether that star has a planet. Probability calibration and performance on
other sectors need larger independent samples. Catalog matching is automatic
and merits manual review before scientific use. Bootstrap parameter intervals
and a Random Forest comparison are not part of this version.

## Reproduce training

```powershell
python -m pip install -r requirements.txt
python train_real_tess.py --sectors 1 2 3
```

`requirements-tested.txt` records the direct package versions used for the
included model. Use it when reproducing that artifact in a separate environment.

This downloads sector manifests and eligible labeled light curves, extracts
features, and saves the model and evaluation. It does not download all stars in
each sector. One sector can be selected with `--sector N`; rerunning replaces
the current model/report. Restart the interface afterward. Cached manifests,
catalog snapshot, and per-observation features make repeat runs quicker.
Feature caches include their preprocessing configuration; changed settings
automatically trigger re-extraction.

To update labels, archive/rename `data/toi_catalog.csv` before rerunning so a
fresh catalog snapshot is fetched. Changing preprocessing or features requires
retraining; increment the pipeline configuration version when changing feature
definitions. The app checks model pipeline compatibility.

Important files:

- `app.py`: desktop interface.
- `tess_pipeline.py`: shared downloading, cleaning, and BLS search.
- `train_real_tess.py`: real-data collection, label matching, and training.
- `svm_classifier.py`: scaling, grouped splits, SVM, calibration, inference.
- `data/labeled_features.csv`: matched real training/evaluation rows.
- `data/all_extracted_features.csv`: includes unmatched detections for audit.
- `data/download_errors.csv`: per-target failures, if any.
- `training_report.json`: held-out results, provenance, and software versions.

Original exploratory files are preserved. `SVM_USAGE.md` describes their earlier
Kepler workflow; use this README and the training report for the desktop app.

Data sources: [MAST TESS bulk downloads](https://archive.stsci.edu/tess/bulk_downloads.html),
[TOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html).
