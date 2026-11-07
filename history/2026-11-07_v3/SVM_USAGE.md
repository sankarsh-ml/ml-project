# How the existing project works

**Update:** The project now includes a real TESS training pipeline and desktop
interface. See [README.md](README.md) for the current workflow and
`training_report.json` for actual held-out results. This page explains the
original Kepler exploration and the SVM feature interface.

`ml_project_prepro.ipynb` downloads Kepler-69 long-cadence observations,
stitches the observations, removes missing values and outliers, and divides
out a smooth trend. `ml_project_bls (2).py` includes that process and adds a
Box Least Squares (BLS) search. It tries 100,000 periods between 0.61 and
300 days and 11 transit durations from 0.05 to 0.6 days. The strongest BLS
peak supplies period, transit epoch, duration, depth, and power. Folding
aligns successive suspected transits; binning makes their average shape
easier to see. The script masks that signal and searches for a second one,
then saves the flattened light curve as `kepler69_flat.csv`.

These are exploratory preprocessing and detection files. They do not yet
implement the abstract's TESS dataset, TOI labels, Random Forest, bootstrap
uncertainties, or held-out classification results. A BLS maximum exists even
for noise, so the strongest peak alone is not evidence of a planet.

The notebook clips both high and low outliers at five sigma, which can remove
deep transits. The Python script protects downward dips by clipping only
upper outliers. A flattening window of 901 is a number of samples: about
18.4 days at approximately 29.4-minute Kepler cadence, but only 30 hours at
TESS two-minute cadence. Retune detrending for the cadence and expected transit
durations; inspect whether it suppresses long transits. Transit masking during
refinement can help. The existing 300-day search range is unsuitable as a
default for one approximately 27-day TESS sector: repeated-transit coverage
must constrain the search and candidate selection.

# Added final SVM classifier

Pipeline: cleaned light curve -> BLS -> feature row -> log transform and scaling
-> RBF SVM -> calibrated candidate probability -> binary label at a threshold
selected using training-only cross-validation.

This implements an SVM as the final classifier in place of a Random Forest.
An SVM is not a neural-network layer. If both RF and SVM are required, compare
them on the same held-out stars first. An RF-to-SVM stacking model would need
out-of-fold RF predictions; training the SVM on RF predictions from the RF's
own training rows would leak information.

Install dependencies:

```powershell
python -m pip install -r requirements-svm.txt
```

After the existing BLS calculation, extract a feature row:

```python
import pandas as pd
from svm_classifier import bls_features, predict

row = bls_features(bls_lc, bls)
new_features = pd.DataFrame([row])
# After training on a labeled dataset:
result = predict(new_features, "svm_model.joblib")
print(result[["predicted_label", "planet_candidate_probability"]])
```

For training, run the same preprocessing, search, and feature extraction on
each light curve. Save one row per detected signal with these columns:

```text
tic_id,label,period_days,duration_days,depth,flux_std,duty_cycle,transit_snr,odd_even_significance,secondary_ratio,shape_ratio,depth_scatter_ratio,red_noise_ratio
```

Period and duration are days; depth and flux standard deviation use relative
flux normalized near one. Duty cycle is duration divided by period. TIC ID
and label are metadata and are never model inputs. For multiple signals on
one star, match each signal to the catalog ephemeris, checking period aliases
and transit epoch; do not assign every detected signal the host star's label.

Use vetted positive and negative labels. Absence from TOI does not mean
nonplanet. Planet candidates are not confirmed planets: decide explicitly
whether positives include candidates or only confirmed/known planets.
Exclude ambiguous dispositions. Confirm that a sector has enough labeled
stars in each class before promising a 2,000-example supervised dataset.

Train and then classify new feature rows:

```powershell
python svm_classifier.py train labeled_features.csv svm_model.joblib
python svm_classifier.py predict new_features.csv svm_model.joblib predictions.csv
```

The module reserves approximately 20% of stars for testing, searches SVM
hyperparameters with three grouped training folds, and calibrates probability
with grouped folds. All rows for a TIC stay together, including multiple
signals or resampled versions. Scaling happens inside the fitted pipeline.
Reported results include accuracy, class precision/recall/F1, confusion matrix,
and average precision. Keep the test set untouched when changing parameters
or thresholds. The saved model is the evaluated training-partition model.

The current feature set adds transit SNR, odd-even depth differences, secondary
eclipses, shape, depth consistency, and red-noise measurements to the original
five-feature baseline. Apply every feature definition identically at training
and prediction time and retrain when definitions change.

The calibrated probability estimates the training label, not certainty that
a physical planet exists. Validate calibration on representative real data.
Bootstrap intervals for period/depth/duration are a separate task; naively
resampling individual time samples ignores time-correlated noise. Consider
resampling residual blocks on the original time grid and rerunning the search.

The initial SVM checks used synthetic features. The desktop version now has a
separate real-data pilot; its reported results apply only to the catalog-matched
observations described in README.md, not the proposed 2,000-curve study.
