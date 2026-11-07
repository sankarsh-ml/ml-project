# TESS Planet Explorer

A small desktop project following the original abstract: download real TESS brightness data, clean and detrend it, find repeating dips with Box Least Squares (BLS), classify the signal with a Random Forest, estimate measurement uncertainty by resampling, and plot the result.

The simplified model achieved **70.37% accuracy and 70.37% balanced accuracy** on 54 accepted signals from 60 newly reserved stars. This is a measured result for this sample, not a guarantee for future stars. The 95% Wilson interval for accuracy is **57.17–80.86%**.

## Run

Use Python 3.11 with Tkinter. Install Git LFS before pulling the model:

```powershell
git lfs install
git lfs pull
python -m pip install -r requirements.txt
python app.py
```

Or double-click `launch_app.bat` after installing dependencies. `python app.py --demo` analyzes a convenient example.

Choose or type a TIC ID, select a sector, and click **Analyze TIC**. The searchable dropdown includes catalog dispositions and known sectors, accepts new IDs, and supports arrows and Enter. You can also open a local TESS SPOC two-minute FITS file. Downloads run in the background; this is on-demand analysis of archived observations, not a live telescope feed.

The dark interface shows a planet-like or false-positive-like verdict, period, duration, depth, bootstrap scatter, and brightness/folded-transit/BLS plots. Weak detections return “Insufficient periodic signal.” Export saves a CSV and matching PNG. A candidate is never presented as a confirmed planet.

## How the code works

1. **Download and clean:** `tess_pipeline.py` downloads official SPOC FITS data from the public TESS mirror, with an STScI fallback. Cached observations work offline. `search_pipeline.py` reads PDCSAP flux, removes nonfinite/flagged points, verifies two-minute cadence, normalizes brightness, and removes slow trends separately across gaps. Dips are excluded from trend fitting.
2. **Find the transit:** BLS searches 0.6–10 days, limited to half the observation baseline. It refines the strongest peaks and half/double-period alternatives, masks transit windows, detrends again, and measures period, epoch, duration, and depth on the original cadence. A fixed signal-quality gate requires multiple sampled events.
3. **Classify:** `transit_features.py` derives 16 brightness-only inputs: period, duration, depth, flux scatter, duty cycle, transit SNR, odd/even difference, secondary eclipse ratio, transit shape, event depth scatter, red-noise ratio, event count, pink-noise SNR, BLS SNR, detection efficiency, and event consistency. `classifier.py` fits one class-balanced Random Forest. There is no SVM layer, stellar-metadata model, or model-family competition.
4. **Estimate confidence:** sigmoid calibration uses training folds. The model produces a calibrated score, but its independent calibration is not established: this test has only 54 accepted signals and calibration error is 0.127. The interface therefore displays **Not validated** for numerical probability. Raw scores remain in exports and evaluation tables; they are not physical planet-confirmation probabilities.
5. **Resample:** 20 residual block-bootstrap fits estimate scatter and 16th–84th percentile intervals for period, duration, and depth. Observation times stay fixed, and one-hour residual blocks preserve some short-term noise correlation. Fits use eight-minute bins and a local BLS grid. These estimates are conditional on the chosen period alias and fixed detrending; they exclude alias uncertainty and preprocessing systematics. Grid resolution can limit very small uncertainties.

`app.py` connects these functions to the Tkinter interface. `target_selector.py` provides the searchable selector. `train_real_tess.py` contains catalog association, training, freezing, and evaluation in one script.

The implementations use [Astropy BLS](https://docs.astropy.org/en/stable/timeseries/bls.html), [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), and the [NASA Exoplanet Archive TOI catalog definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html).

## Real-data evaluation

Training uses **446 distinct stars**: 237 CP/KP planet labels and 209 FP labels. Another **60 previously unused stars** were reserved before fitting, with 30 catalog stars per class. The frozen model accepted 54 ephemeris-matched signals, evenly split between the classes. The resulting labeled dataset contains **500 stars**.

The data span multiple TESS sectors from 1–55; this implementation does **not** claim to have trained on 2,000 observations from one sector. The TOI snapshot contains 8,151 rows. Only CP/KP and explicit FP dispositions supply labels; unresolved candidates and unlisted stars are not treated as negatives. Catalog period and epoch are used only to match the blind detection to its label, including half/double-period aliases. They never enter the classifier.

| Independent test metric | Result |
| --- | ---: |
| Accuracy | 70.37% |
| Balanced accuracy | 70.37% |
| Planet precision | 67.74% |
| Planet recall | 77.78% |
| Planet F1 | 72.41% |
| False-positive recall | 62.96% |
| Majority-class baseline | 50.00% |

Confusion matrix, rows = true class and columns = predicted class, ordered `[false positive, planet-like]`:

```text
[[17, 10],
 [ 6, 21]]
```

Six reserved observations were unmatched or weak and were excluded by the predeclared gate/ephemeris association, rather than forced into a class. Classification coverage is **54/60 = 90%**; correct classifications across all reserved observations are **38/60 = 63.33%**. The 70.37% score applies to recovered catalog signals, not the entire target list or a blind stellar survey. This benchmark also does not measure rejection of arbitrary pure noise.

Five grouped training folds select among six Random Forest configurations by balanced accuracy. The chosen forest has 300 trees, minimum leaf size 10, and all 16 features available at each split. Its training CV balanced accuracy was 77.94%; calibrated training out-of-fold balanced accuracy was 75.80%. Hyperparameters were selected on those training data, so these are development estimates. The classification threshold stays fixed at 0.5. No TIC appears in training and test, and every prior evaluation target is excluded from the new test. No tuning followed this independent evaluation.

The earlier 84% Extra Trees result used additional centroid/stellar information and a different test set. It is superseded and is not the simplified Random Forest's result. Old source copies, notebooks, alternate models, caches, experiment archives, logs, and unused observations have been removed. A short previous-result record remains in the current report for context.

## Reproduce and test

```powershell
python -m unittest test_pipeline test_transit_features -v
python verify_project.py
python smoke_test_app.py
```

Verification checks frozen source/model/table hashes, star separation, original FITS hashes when downloads are present, and exact reproduction of saved test predictions. It writes `outputs/evaluation.png`. The GUI smoke test runs real FITS analysis, the worker queue, classification, all three plots, selector entry, and CSV/PNG export. All 15 unit tests and the end-to-end GUI test passed after cleanup.

The included experiment is already evaluated and deliberately refuses retraining against this opened test. For a **new experiment**, use a separate project copy, reserve additional untouched TICs, update `data/observation_plan.csv` and `data/split.json`, and remove the generated `data/frozen.json` and `data/evaluation_lock.json` there. Never erase these safeguards to reuse the present test for tuning. If rebuilding features, regenerate `data/training_features.csv` before freezing:

```powershell
python train_real_tess.py --stage extract-training
python train_real_tess.py --stage fit
python train_real_tess.py --stage evaluate --workers 4
```

Keep the new training IDs consistent with the new split. These commands are the preparation sequence for an unopened experiment, not commands to rerun on the shipped evaluated split.

## Files retained

- The Python source files, two focused unit-test files, GUI smoke test, and verification script.
- `planet_model.joblib`: the single compressed Random Forest artifact, stored through Git LFS.
- `training_report.json`: current training methodology, class counts, calibration, and independent metrics.
- `data/`: catalog, sector index, one observation plan and split, training/test feature tables, provenance hashes, freeze record, evaluation lock, and test extraction/error records.
- `outputs/`: current independent predictions, confusion plot, and one example CSV/PNG.
- `requirements.txt`, launcher, `.gitignore`, and `.gitattributes`.

Raw FITS files, regenerable sector manifests, Python caches, and logs are ignored by Git. Current training/test observations and cached examples remain locally; missing FITS files download on demand. Git retains earlier commits normally; obsolete files have been removed from the current project, without rewriting repository history.
