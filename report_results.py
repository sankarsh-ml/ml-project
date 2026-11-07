"""Write a reproducible README and audit from saved reports, without refitting."""
import json
from pathlib import Path
import pandas as pd
from tess_pipeline import ROOT
from training_workflow import match_ephemeris
from experiment import RUN_DIR


def period_audit():
    plan = pd.read_csv(RUN_DIR / "observation_plan.csv", dtype={"tic_id": str})
    old = pd.read_csv(ROOT / "history/2026-11-07_v3/data/all_extracted_features.csv", dtype={"tic_id": str})
    new = pd.read_csv(RUN_DIR / "training_extracted.csv", dtype={"tic_id": str})
    rows = []
    metadata = {(row["tic_id"], row["sector"]): row for row in plan[plan.partition == "training"].to_dict("records")}
    current = {(row["tic_id"], row["sector"]): row for row in new.to_dict("records")}
    for row in old.to_dict("records"):
        key = (row["tic_id"], row["sector"])
        if key not in metadata or key not in current:
            continue
        before = match_ephemeris(row, metadata[key])
        after = current[key]
        rows.append({"tic_id": key[0], "sector": key[1], "old_period_days": row["period_days"],
            "new_period_days": after["period_days"], "catalog_period_days": metadata[key]["pl_orbper"],
            "old_match_current_ephemeris": before["catalog_match"], "new_match": after["catalog_match"],
            "new_alias": after["period_alias"], "new_bls_snr": after["bls_snr"],
            "new_pink_snr": after["pink_snr"], "new_detection_accepted": after["detection_accepted"]})
    audit = pd.DataFrame(rows)
    audit.to_csv(ROOT / "outputs/period_recovery_audit.csv", index=False)
    return audit


def main():
    report = json.loads((ROOT / "training_report.json").read_text())
    plan = json.loads((RUN_DIR / "split_plan.json").read_text())
    audit = period_audit()
    comparison = "\n".join(f"| {row['model']} | {row['grid_cv_balanced_accuracy']:.1%} ± {row['grid_cv_std']:.1%} | {row['balanced_accuracy']:.1%} | {row['minimum_class_recall']:.1%} |"
        for row in report["model_comparison"])
    historical = report["historical_evaluations"]
    history_table = "\n".join(f"| {name} | {r['metrics']['accuracy']:.1%} | {r.get('balanced_accuracy', .5):.1%} | {r['metrics']['1']['f1-score']:.3f} |"
        for name, r in [("Initial five-feature baseline (all predictions positive)", historical[1]),
                        ("Revised five-feature development model", historical[2]),
                        ("Previous 11-feature model", historical[0])])
    m, neg = report["metrics"]["1"], report["metrics"]["0"]
    recovery = report["data_recovery"]
    ci = report["test_accuracy_wilson_95"]
    improved = int((~audit.old_match_current_ephemeris & audit.new_match).sum()) if len(audit) else 0
    regressed = int((audit.old_match_current_ephemeris & ~audit.new_match).sum()) if len(audit) else 0
    training = pd.read_csv(RUN_DIR / "training_labeled.csv")
    full_planets = int(((training.label == 1) & (training.period_alias == 1)).sum())
    half_fp = int(((training.label == 0) & (training.period_alias == .5)).sum())
    double_fp = int(((training.label == 0) & (training.period_alias == 2)).sum())
    readme = f'''# TESS Planet Explorer

Local dark-themed desktop app for real TESS two-minute light curves. Enter a TIC
ID, choose a sector, and inspect a blind BLS transit search plus an experimental
planet-like / false-positive-like classifier. It does not confirm a planet.

## GitHub checkout and large files

Models use Git LFS: install it and run `git lfs install`; use `git lfs pull` after
an existing checkout if needed. `.gitignore` excludes raw FITS downloads, cached
sector manifests, Python caches, temporary files, and runtime logs. The app
downloads requested data on demand. Run `python prefetch_data.py` to restore raw
protocol observations before raw-FITS verification. Dataset plans, feature tables,
models, reports, and historical results remain versioned. The local inventory in
`FILE_MANIFEST.csv` includes external FITS data which are intentionally not in Git.

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

The **{report['selected_model']}** was selected using training data only. On
**{report['test_rows']} entirely new TIC IDs**, opened once after the model and
threshold were frozen:

| Metric | Independent result |
|---|---:|
| Accuracy | {report['accuracy']:.2%} |
| Balanced accuracy | {report['balanced_accuracy']:.2%} |
| Planet precision | {m['precision']:.2%} |
| Planet recall | {m['recall']:.2%} |
| Planet F1 | {m['f1-score']:.3f} |
| False-positive recall | {neg['recall']:.2%} |
| Accuracy Wilson 95% interval | {ci[0]:.1%}–{ci[1]:.1%} |

Confusion matrix, rows=true and columns=predicted, order `[false positive, planet]`:

```text
{report['confusion_matrix_labels_0_1']}
```

The 70% accuracy target was **{'met' if report['accuracy_target_met'] else 'not met'}**
by this one independent evaluation. This measures classification of recovered
catalog signals, not accuracy across arbitrary TESS stars or survey discovery
completeness. Per-target predictions are in `outputs/independent_test_predictions.csv`.

## Dataset, labels, and leakage prevention

- {report['total_assigned_stars']} distinct targets were assigned before extraction,
  spanning available two-minute SPOC products in sectors 1–55.
- Training: {recovery['training']['assigned']} assigned stars,
  {report['train_rows']} matched/gated stars used: {report['training_class_counts']['0']}
  vetted false positives and {report['training_class_counts']['1']} confirmed/known planets.
- Independent test: {recovery['independent_test']['assigned']} assigned new stars,
  {report['test_rows']} matched/gated stars evaluated. Class counts are
  {report['test_class_counts']} (0=false positive, 1=planet).
- One observation per TIC is used. No TIC appears in both training and test.
  All {len(plan['quarantined_development_tic_ids'])} old development and superseded
  test TICs are quarantined from fitting and selection. Every final test TIC is
  absent from all {len(plan['legacy_seen_tic_ids'])} previously assigned/attempted targets.
- Labels come from a saved NASA Exoplanet Archive TOI snapshot: CP/KP positives,
  FP negatives. PC, APC, FA, and uncatalogued stars are not relabeled as either
  class. Only a uniquely eligible catalog signal per TIC is used in this benchmark.
- Each search is blind. Catalog period/epoch are used afterward for association,
  never as classifier inputs or search guides. Association allows 0.5/1/2 period
  aliases, a tight period tolerance, and a BJD-to-BTJD epoch check with propagated
  catalog ephemeris uncertainties and a capped timing tolerance. Epoch projection
  uses the catalog period, not the noisier fitted period from one sector.
- The predeclared detection gate requires BLS SNR >= 6, pink SNR >= 3, at least two
  sampled transit windows, and >=40% of sampled events with local depth SNR >2.
  Unmatched and weak signals are recorded, not assigned fabricated negative labels.
- `data/v5/split_plan.json` fixes IDs before feature extraction. New test FITS files
  were downloaded in advance but not feature-extracted until fitting and selection
  were complete. `frozen_model.json` records model, code, split, and training-feature
  hashes. `evaluation_lock.json` prevents silently evaluating/reselecting again.

The assigned-to-accepted coverage is {report['train_rows']}/{recovery['training']['assigned']}
for training and {report['test_rows']}/{recovery['independent_test']['assigned']} for test.
Classification accuracy is conditional on that recovery/gating step. These losses
must be included when discussing end-to-end transit-finding performance. Counting
only correct labeled classifications over all reserved targets gives
{report['correct_classification_fraction_of_all_reserved']:.1%}; this is an operational
yield, not the classifier's conditional accuracy. The majority-class accuracy
baseline on the classified test subset is {report['majority_baseline_accuracy']:.1%}.

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

The training period audit compared {len(audit)} observations also present in the
old pilot using the same current catalog ephemerides; {improved} formerly unmatched
periods now associate correctly and {regressed} previously matched associations
were lost. See `outputs/period_recovery_audit.csv`.
{full_planets} accepted training planets matched the full catalog period; the
false-positive training set includes {half_fp} half-period and {double_fp}
double-period eclipse aliases.

## Training-only model comparison

| Model | Grouped grid-CV balanced accuracy (mean ± SD) | Calibrated training OOF balanced accuracy | Minimum class recall |
|---|---:|---:|---:|
{comparison}

These are **selection estimates**, not independent results: hyperparameters and
thresholds were chosen using training data. Five grouped folds tune hyperparameters;
separate outer/inner grouped folds generate sigmoid-calibrated training OOF scores
and choose a stable, near-optimal balanced-accuracy threshold. Imputation, scaling,
and transforms are fitted inside each fold. Selection rewards balanced accuracy,
penalizes fold variability, and requires useful recall for both classes where
feasible. SVM is preferred only within 0.01 of the best selection score.

Selected model parameters: `{report['best_parameters']}`. Decision threshold:
`{report['threshold']:.3f}`. The frozen model is evaluated without refitting on test
stars. `svm_model.joblib` and `models/best_model.joblib` contain the selected model;
`models/svm_candidate.joblib` preserves the SVM candidate.

Training calibration ECE (five bins): {report['calibration']['ece_5_bins']:.3f}.
Independent calibration ECE: {report['independent_calibration']['ece_5_bins']:.3f};
Brier score: {report['brier_score']:.3f}. Probability display is
**{'enabled' if report['probability_reliable'] else 'withheld'}** based on predeclared
training and independent calibration checks. Scores refer to this sampled catalog
population; class priors, magnitude selection, stellar metadata errors, other
sectors, blends, noise, and new instrument behavior can affect transfer.

## Historical results, retained in full

| Historical development model | Accuracy | Balanced accuracy | Planet F1 |
|---|---:|---:|---:|
{history_table}

The first result was misleading because it predicted every star positive. The
later 35% results were development diagnostics on 20 observations/15 stars and
were not untouched scientific tests. Full original files/model/report/data tables
are in `history/2026-11-07_v3`; full historical evaluations are embedded in the new
`training_report.json`. The old holdout was not used for the new comparison.

The first expanded v4 run scored 84.09% on 88 classified new stars, but is
**superseded**: a final training audit found that epoch projection used the noisy
fitted period rather than the catalog period across years. After correction,
101 additional training signals associated correctly. The entire 140-star v4
test reservation was quarantined, not used to select the corrected classifier.
Protocol v5 reserved 200 further untouched TICs, froze a new training-selected
model, and opened that test once. The final result above is from those fresh
stars. Full v4 evidence and the reason it was superseded are retained in
`history/v4_pre_epoch_fix` and `training_report.json`.

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
python prepare_followup.py
python prefetch_data.py
python train_real_tess.py --stage extract --partition training --workers 8
python train_real_tess.py --stage fit
python train_real_tess.py --stage extract --partition independent_test --workers 8
python train_real_tess.py --stage evaluate
```

The included split/test has already been opened. The final command and further
fitting deliberately refuse to reuse it for model development. For a new study,
preserve `data/v4` and `data/v5`, select additional untouched stars, and create a new protocol;
do not remove a lock merely to keep optimizing this test score. Existing cached
FITS files and configuration-tagged features are retained. No bootstrap uncertainty
intervals or planet confirmation are claimed in this version.

Source references: [NASA TOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html),
[TESS FITS products](https://spacetelescope.github.io/mast_notebooks/notebooks/TESS/beginner_how_to_use_lc/beginner_how_to_use_lc.html),
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
'''
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "outputs/RESULTS.txt").write_text(
        f"Independent {report['selected_model']}: accuracy {report['accuracy']:.2%}, balanced accuracy {report['balanced_accuracy']:.2%}.\n"
        f"Training {report['train_rows']} distinct TICs; test {report['test_rows']} entirely new TICs.\n"
        f"Planet precision/recall/F1: {m['precision']:.3f}/{m['recall']:.3f}/{m['f1-score']:.3f}.\n"
        f"Confusion matrix [FP, planet]: {report['confusion_matrix_labels_0_1']}\n")
    print("README, result summary, and period audit updated from saved evidence.")


if __name__ == "__main__":
    main()
