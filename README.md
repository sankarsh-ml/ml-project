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

The dark interface shows a planet-like or false-positive-like verdict and just three measurement cards: orbital period in days, transit duration in hours, and brightness drop as a percentage. The plots use readable percentage scales and elapsed observation time. Measurement details opens the bootstrap estimates and dataset membership; About the model explains the classifier and evaluation. The probability card, technical summary block, repeated sector list, and plot toolbar have been removed. Weak detections return "Insufficient periodic signal." Export saves the detailed CSV and matching PNG.

## How the code works

1. **Download and clean:** `tess_pipeline.py` downloads official SPOC FITS data from the public TESS mirror, with an STScI fallback. Cached observations work offline. `search_pipeline.py` reads PDCSAP flux, removes nonfinite/flagged points, verifies two-minute cadence, normalizes brightness, and removes slow trends separately across gaps. Dips are excluded from trend fitting.
2. **Find the transit:** BLS searches 0.6–10 days, limited to half the observation baseline. It refines the strongest peaks and half/double-period alternatives, masks transit windows, detrends again, and measures period, epoch, duration, and depth on the original cadence. A fixed signal-quality gate requires multiple sampled events.
3. **Classify:** `transit_features.py` derives 16 brightness-only inputs: period, duration, depth, flux scatter, duty cycle, transit SNR, odd/even difference, secondary eclipse ratio, transit shape, event depth scatter, red-noise ratio, event count, pink-noise SNR, BLS SNR, detection efficiency, and event consistency. `classifier.py` fits one class-balanced Random Forest. There is no SVM layer, stellar-metadata model, or model-family competition.
4. **Internal score:** sigmoid calibration uses training folds. The model produces a score to choose its class, but independent numerical confidence is not established: this test has only 54 accepted signals and calibration error is 0.127. The interface has no probability card or "Not validated" placeholder. Raw scores remain in CSV exports and evaluation tables for analysis, and are not physical planet-confirmation probabilities.
5. **Resample:** 20 residual block-bootstrap fits estimate scatter and 16th–84th percentile intervals for period, duration, and depth. Observation times stay fixed, and one-hour residual blocks preserve some short-term noise correlation. Fits use eight-minute bins and a local BLS grid. These estimates are conditional on the chosen period alias and fixed detrending; they exclude alias uncertainty and preprocessing systematics. Grid resolution can limit very small uncertainties.

`app.py` connects these functions to the Tkinter interface. `target_selector.py` provides the searchable selector. `train_real_tess.py` contains catalog association, training, freezing, and evaluation in one script.

The implementations use [Astropy BLS](https://docs.astropy.org/en/stable/timeseries/bls.html), [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), and the [NASA Exoplanet Archive TOI catalog definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html).

## Walkthrough: from a star to a prediction

### What you select

A TIC ID is the star's identifier in the TESS Input Catalog. A sector identifies an observing campaign/sky field, so one star can have several sector observations. The TOI catalog lists interesting transit signals; its disposition is an external catalog label. CP/KP are the vetted planet categories used here, while FP is an explicit false positive. A candidate with an unresolved label is not automatically a negative example.

The selector shows catalog information to help you choose a target. The classifier itself receives only measured brightness features. A known planet can therefore receive a false-positive-like prediction when the model makes a mistake. In that case the interface now says **This prediction disagrees with the catalog**. The model does not overwrite or disprove the catalog label.

### Clean the brightness measurements

Each two-minute sample gives an observation time and the star's measured brightness, called flux. PDCSAP is the telescope pipeline's corrected brightness product; the code still needs to reject flagged/nonfinite samples and remove remaining slow variation. It uses the TESS quality bit mask 17087, checks that cadence is about two minutes, and divides flux by its median to put the baseline near 1.

Slow stellar/instrumental changes can be much larger than a transit. A broad two-day Savitzky-Golay trend is fitted separately across large data gaps. Repeated fitting excludes deep downward deviations so a possible transit is not absorbed into the trend. Dividing by that trend leaves the small, quicker dips that BLS searches for. Large upward excursions are removed, while downward dips remain available for analysis. The code is deliberately designed to preserve transits rather than smooth them away.

### Search with Box Least Squares

Imagine sliding a repeating rectangular dip over the brightness series. For every trial period and dip duration, BLS asks how strongly that repeating box is supported by the data and their uncertainties. The search spans 0.6 to at most 10 days, with the upper limit also restricted to half the observation baseline. A uniform-frequency grid uses 20,000 trials on an eight-minute binned copy for speed.

The strongest peak is not always the correct orbital period. Alternating eclipses can produce a half-period solution, and some events can create double-period peaks. The code refines several strong peaks and half/double alternatives. It then masks the chosen transit windows, removes the slow trend again, and refines the final signal on the original two-minute measurements. The period is a best repeating-dip period; interpreting it as an orbital period remains a candidate hypothesis.

Three measurements are especially useful:

- **Period:** time between repeating dips. A 6.2677-day value means the fitted signal repeats about every 6.27 days.
- **Duration:** how long the fitted dip lasts. Internally it is stored in days; the interface multiplies by 24 to show hours.
- **Depth:** fractional loss of brightness. The interface multiplies by 100 to show a percent. A depth of 0.000256 is 0.0256%, or 256 parts per million (ppm). It is not a 25.6% loss of light.

Before classification the fixed gate requires BLS SNR >= 6, pink-noise SNR >= 3, at least two sampled transit windows, and at least 40% of sampled windows with a measurable event. Signals failing that gate receive **Insufficient periodic signal**, rather than a forced binary classification.

### Turn a light curve into 16 inputs

The forest does not read an image of the plot. It receives one row of numerical measurements:

| Input column | Meaning |
| --- | --- |
| `period_days` | Best fitted repetition time. |
| `duration_days` | Length of the fitted dip. |
| `depth` | Fraction of the star's baseline light lost in the dip. |
| `flux_std` | Overall scatter of the detrended brightness. |
| `duty_cycle` | Duration divided by period: the fraction of an orbit spent in the dip. |
| `transit_snr` | Dip depth relative to noise after combining in-transit samples. |
| `odd_even_significance` | Whether alternating dips have substantially different depths, which can indicate eclipsing stars. |
| `secondary_ratio` | Strength of a dip near the opposite phase relative to the primary dip. |
| `shape_ratio` | Comparison of the dip's center and edges; helps distinguish flat-bottomed and V-shaped profiles. |
| `depth_scatter_ratio` | How much individually measured event depths vary relative to the fitted depth. |
| `red_noise_ratio` | Excess correlated noise measured from two-hour averages compared with white-noise expectations. |
| `observed_transits` | Number of transit windows containing enough usable samples to measure an event. |
| `pink_snr` | Transit SNR penalized for correlated noise. |
| `bls_snr` | Strength of the final BLS box fit relative to its uncertainty. |
| `sde` | How far the selected search peak rises above the typical search background. |
| `event_fraction` | Fraction of individually sampled events with event SNR above 2. |

TIC IDs, sector IDs, catalog dispositions, catalog periods/epochs, and true labels are not classifier inputs. The explicit feature list prevents accidental use of those extra table columns.

### Learn with a Random Forest

A decision tree asks a sequence of numerical questions, such as whether a secondary eclipse is strong or whether individual dip depths are consistent. A Random Forest combines many trees fitted to randomly resampled training rows so one tree's peculiarities have less influence. The selected configuration uses 300 trees per forest, a minimum of 10 training samples in each leaf, all 16 features available per split, and class-balanced bootstrap weights.

Training tries two feature-selection settings and three leaf sizes: six configurations in total. Five grouped cross-validation folds compare them using balanced accuracy, so correctly identifying both classes matters. Any missing input values are filled using medians learned inside the fitting fold, rather than from the test data. Hyperparameters come from training data only.

Sigmoid calibration learns how forest scores relate to observed class frequencies on training folds. The saved calibration wrapper averages five calibrated fold models. The final class threshold is fixed at 0.5: scores at or above it produce planet-like, and lower scores produce false-positive-like. A score close to 0.5 triggers the plain-language borderline notice. We do not display it as a percentage probability of a real planet because this small independent sample does not establish reliable numerical confidence.

### Keep training and testing separate

The 446 established training stars are used to learn patterns and select settings. Sixty additional stars were reserved before fitting, with no TIC reused from training or earlier inspected experiments. Their detections were processed blindly, and catalog period/epoch matching was used only to attach the correct known label to the detected signal. The association allows half/double periods and projects catalog epochs using the catalog period, avoiding accumulated phase error from a noisy one-sector BLS period. BTJD timestamps are converted to BJD by adding 2,457,000 for this association.

Before the new test was opened, the model, scientific code, split, and input tables were fingerprinted in `data/frozen.json`. Evaluation writes a lock file and is performed once. The verification script can reproduce the saved score repeatedly; that is different from modifying the model and trying the same test until its score rises. UI readability changes do not retrain the model or change its predictions.

### Estimate measurement uncertainty

The pipeline fits a signal, subtracts it to obtain residual noise, randomly resamples approximately one-hour blocks of that noise, adds the blocks back to the fitted signal, and runs a nearby BLS fit again. Twenty repetitions give a spread of periods, durations, and depths. The return value includes standard deviations and 16th-84th percentile intervals. Measurement details displays the standard deviations, and the CSV includes them.

This bootstrap describes local measurement variation, not model classification accuracy or confidence that a planet exists. It assumes the chosen period alias and fixed detrending, and it uses a finite search grid. If all resamples select the same duration grid point, the measured duration scatter can be zero; the dialog says that it is grid-limited rather than implying perfect precision.

### Read the three plots

- **Brightness over time:** baseline brightness is near 100%, horizontal position is days since the first retained observation, and highlighted points fall inside the fitted transit windows. Gaps are missing observations, not necessarily brightness changes.
- **Repeating dips aligned:** the code wraps all times using the fitted period and puts their transit centers at hour zero. Combining repeated events makes a consistent shallow dip easier to see. The vertical scale is brightness change in percent, with the baseline at zero; the line shows binned medians.
- **Transit period search (BLS):** the horizontal axis is trial period and the vertical axis is BLS signal strength. The marked line identifies the selected period. Other peaks can be harmonics or competing signals; the plot alone does not confirm a planet.

Downloads and numerical work run in a background thread. A queue sends status/results to Tkinter's main thread, which safely updates widgets and plots. The app remains responsive, but selecting a TIC analyzes already archived data rather than streaming a live telescope.

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

## Every file explained

The project has 33 tracked files. The application needs the source, dependency list, and saved model to analyze a star; the data and reports also preserve the evidence behind its reported score.

### Application and science code

| File | What it does and when it runs |
| --- | --- |
| `app.py` | The desktop entry point. `Explorer` builds the dark Tkinter window, validates input, starts background work, handles the worker queue, renders the three plots, and exports results. `render()` changes presentation only: period/duration stay in their original scientific units internally, while depth and plot brightness are converted to percentages for display. `measurement_details()` puts uncertainty and training/test membership in the optional dialog. The catalog-disagreement notice is a presentation warning, not a classifier input. Run with `python app.py`. |
| `target_selector.py` | `parse_tic()` accepts positive TIC numbers, with an optional TIC prefix. `TargetIndex` combines catalog dispositions, the sector index, the observation plan, training records, and downloaded filenames. `SearchableTIC` implements the editable entry/popup, filtering, arrow keys, Enter, and manual IDs. `EXAMPLES` supplies convenient targets. Catalog status in the selector is separate from the model prediction. |
| `tess_pipeline.py` | Defines the project/data paths and downloads observations. `download_tic()` first checks local FITS files. Otherwise `manifest()` reads an official sector filename list and finds the requested TIC, then the downloader tries the official public TESS mirror and an STScI fallback. It checks the FITS signature and uses a temporary file before replacing the final path. Downloaded manifest scripts are parsed, never executed. This module also exposes the shared `analyze()` function. |
| `search_pipeline.py` | Implements the actual brightness analysis. `CONFIG` contains the frozen settings. `read_curve()` loads quality-filtered, normalized PDCSAP flux. `detrend()` removes slow changes while protecting dips and gaps. `bin_curve()` speeds up the initial search. `search()` finds/refines BLS peaks and half/double-period alternatives. `analyze()` re-detrends around masked transit windows, fits the final signal, calculates features, and returns the arrays used by the plots. `detection_accepted()` applies the fixed quality gate. `bootstrap_uncertainty()` resamples residual blocks and refits nearby signals. |
| `transit_features.py` | Defines the exact ordered 16-column input schema. `phase_time()` aligns samples to a repeating period, `mad()` robustly estimates scatter, and `vetting_features()` measures odd/even differences, secondary eclipses, shape, event consistency, and noise from brightness alone. It neither downloads data nor reads catalog planet labels. |
| `classifier.py` | Contains the Random Forest and prediction functions. `feature_matrix()` selects/validates only the 16 approved inputs, ignoring IDs and labels. `grouped_folds()` keeps each star in one fold. `train()` searches six forest configurations, imputes missing values inside each fit, calibrates scores on training folds, and returns the fitted artifact/report. `predict()` loads that artifact and returns its score and class using threshold 0.5. `metrics()` and `calibration_summary()` calculate evaluation statistics. Routine app use calls prediction, not training. |
| `train_real_tess.py` | The offline training/evaluation command. `process_observation()` downloads/analyzes a planned observation and attaches a label only after `match_ephemeris()` checks the detected period/epoch against the catalog. `extract()` collects feature/error tables. `fit()` trains on the reserved training TICs and freezes hashes. `evaluate()` exclusively opens the untouched test once, applies the frozen model, and writes predictions/metrics. `check_frozen()` detects source, data, or model changes. Normal GUI use does not execute this script. |
| `verify_project.py` | Reproduces the saved evaluation without retraining. It checks frozen hashes, test/training separation, catalog association and gates, exact saved predictions, and raw FITS hashes when files exist locally. It also regenerates `outputs/evaluation.png`. Run with `python verify_project.py`. |
| `smoke_test_app.py` | Exercises the real desktop code with a real example FITS file: background worker, queue, classification, bootstrap, plot rendering, selector, and CSV/PNG export. It checks that there are only three measurement cards and no probability placeholder. Separate presentation fixtures check catalog disagreement, borderline predictions, and insufficient-signal display. It captures Tk callback errors so an apparently successful window cannot hide a failed handler. |
| `test_pipeline.py` | Eleven focused tests of detrending, blind BLS recovery, resampling behavior, catalog epoch/alias matching, TIC separation, exclusion of catalog information from features, grouped folds, editable TIC lookup, quality gates, and refusal to train against an opened independent test. Uses synthetic transits where the correct answer is known, plus the actual saved split. |
| `test_transit_features.py` | Four physical feature checks: a secondary eclipse raises the secondary ratio; alternating event depths raise odd/even significance; a V-shaped dip changes the shape ratio; overlapping transit windows with insufficient baseline are rejected. Together the two unit-test files contain 15 tests. |

### Saved model, data, and evaluation evidence

| File | What it contains and why it stays |
| --- | --- |
| `planet_model.joblib` | The fitted median imputer, calibrated Random Forest ensemble, ordered feature names, preprocessing configuration, threshold, and training feature ranges. It is about 1.55 MB, compressed and stored through Git LFS. The calibration wrapper averages five calibrated fold models, each containing a 300-tree forest. The app loads this file rather than learning again every time. Load only a trusted model artifact. |
| `training_report.json` | The machine-readable methodology and results: training class counts, selected hyperparameters, development estimates, calibration checks, independent metrics and confusion matrix, TIC lists, coverage, limitations, bootstrap scope, and verification notes. This is the authoritative current evaluation report. Changing the UI does not create a new model accuracy result. |
| `data/toi_catalog_full.csv` | The saved 8,151-row TOI catalog snapshot, including TIC/TOI identifiers, dispositions, period/epoch/duration information, and metadata. It provides training/evaluation labels and selector labels. Although it contains stellar metadata, the simplified classifier never uses those metadata columns. |
| `data/sector_index.csv` | 15,742 catalog TIC/sector pairs from official observation lists. It suggests known sectors in the dropdown. It is a local availability index, not a prediction or a guarantee that newly published sectors are already indexed. |
| `data/observation_plan.csv` | 506 selected observations: 446 training stars and 60 reserved test stars. Each row identifies the TIC, sector, FITS filename, partition, label, and catalog ephemeris needed to associate a blind detection. It is the acquisition/association plan, not the feature matrix fed to the classifier. |
| `data/split.json` | The immutable experiment assignment: training TICs, new test TICs, previously used TICs that the new test must avoid, random seed, and protocol description. It prevents accidental reuse of previously inspected stars as a supposedly new test. |
| `data/training_features.csv` | The 446 accepted, labeled training observations. It stores the 16 model inputs plus provenance/association fields such as TIC, sector, epoch, source filename, disposition, and gate result. `feature_matrix()` selects the input columns explicitly so the extra fields cannot leak into fitting. |
| `data/training_provenance.csv` | A 446-row link from training TIC/sector to the original FITS filename and its SHA-256 fingerprint. It records exactly which downloaded bytes produced the training data, without placing large FITS files in Git. |
| `data/test_extracted.csv` | All 60 new test observations after blind processing, including weak or unmatched detections. Keeps the exclusion reasons visible through the `catalog_match` and `detection_accepted` columns, plus source hashes. This prevents the 54 accepted signals from hiding the other six reserved observations. |
| `data/test_features.csv` | The 54 accepted, ephemeris-matched test signals used for classification metrics. It contains 27 catalog planets and 27 catalog false positives. This table is used only for the frozen independent evaluation and subsequent reproduction of that score. |
| `data/test_errors.csv` | Download/processing failures, with TIC and error text. It currently has only the header: there were no extraction exceptions. The six exclusions were quality/association decisions, not failed downloads hidden from the report. |
| `data/frozen.json` | SHA-256 fingerprints of the model, training table, split, observation plan, catalog, and five scientific source modules. These fingerprints were fixed before opening the new test. They are integrity checks, not prediction features. `app.py` is presentation code and is outside this scientific freeze. |
| `data/evaluation_lock.json` | The timestamp recording that this test has been opened. Exclusive file creation prevents an unnoticed second evaluation, and `fit()` refuses to train under this opened protocol. Retaining the small file is essential to the credibility of the reported score. |

### Outputs, setup, and repository files

| File | Purpose |
| --- | --- |
| `outputs/demo_prediction.csv` | One example application's full exported feature row, predicted class, raw internal score, reliability flag, and bootstrap standard deviations. Internal scores are retained for analysis but are not shown as planet probabilities in the interface. |
| `outputs/demo_prediction.png` | The same example's three dark-themed plots, exported using the clearer elapsed-time and percentage axes. It is a plot export, not a screenshot of the entire window. |
| `outputs/test_predictions.csv` | The 54 independent TIC/sector pairs with true label, model score, and predicted label. It makes the confusion matrix auditable row by row. |
| `outputs/evaluation.png` | The independent confusion-matrix figure generated by `verify_project.py`. It documents model evaluation, not the prediction for the currently selected GUI star. |
| `requirements.txt` | Exact Python library versions used for the supplied artifact and tests: NumPy/Pandas for arrays/tables, SciPy for detrending/statistics, Astropy for FITS/BLS, scikit-learn for learning/calibration, Joblib for model storage/parallel jobs, Matplotlib for plots, and Requests for downloads. Tkinter comes with the standard Windows Python installation. |
| `launch_app.bat` | A Windows convenience launcher. It changes to this project's directory, starts `python app.py`, and keeps an error visible if startup fails. It does not install dependencies or train a model. |
| `.gitignore` | Excludes raw FITS downloads, regenerated sector manifests, Python caches, logs, temporary files, local environments, and local credentials from Git. Ignored files can still exist and be used locally. |
| `.gitattributes` | Routes saved model binaries through Git LFS and preserves exact text bytes so Windows line-ending conversion cannot silently break the frozen scientific SHA-256 checks. It also configures whitespace checks for those preserved line endings. |
| `README.md` | This human-readable guide: installation, normal use, the pipeline, all 16 inputs, training/evaluation methodology, limitations, and every retained file. Start here when explaining or presenting the project. |

### Local generated files

`data/lightcurves/*.fits` are the actual downloaded telescope observations. A FITS file contains scientific tables and headers, including time, corrected flux, uncertainty, quality flags, TIC, and sector. Current training/test observations and useful examples remain locally; the repository does not upload the large downloads. `data/sector*_download.sh` may be regenerated when an uncached TIC is requested; only filenames are parsed from these lists. `__pycache__/` is Python's disposable compiled cache. None of these needs manual editing.

Git retains earlier commits normally. Obsolete source copies and experiment archives were removed from the current project without rewriting repository history.
