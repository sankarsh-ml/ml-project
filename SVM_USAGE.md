# Classifier API

The current validated classifier is selected using training-only grouped cross-validation.
See README.md and training_report.json for the final independent result, full model
comparison, calibration checks, and limitations.

The compatibility filename svm_model.joblib contains the selected **Extra Trees**
classifier. The calibrated SVM candidate is preserved as models/svm_candidate.joblib.
The SVM was compared fairly; the deployed model had the stronger training-only
calibrated balanced-accuracy selection score. No family was selected using final
test results.

## Analyze a real TESS observation

```python
import pandas as pd
from tess_pipeline import download_tic, analyze
from svm_classifier import predict

path = download_tic("149603524", sector=1)
result = analyze(path)
features = pd.DataFrame([result["features"]])
prediction = predict(features, "svm_model.joblib")
print(prediction[["predicted_label", "planet_candidate_probability", "probability_reliable"]])
```

Use the full FITS analyzer so that preprocessing, BLS selection, centroid motion,
stellar metadata, and vetting features match training. The older bls_features()
helper remains available for compatibility; it cannot reconstruct all FITS
centroid diagnostics from a stripped light curve. The original Kepler notebook
and script are preserved, but the current model is intended for TESS two-minute
PDCSAP observations.

FEATURES in transit_features.py defines the exact input schema. TIC ID, sector,
TOI disposition, catalog period/epoch, and train/test membership are metadata,
never model inputs. Catalog ephemerides are used only to label recovered signals.
All model imputation, transforms, and scaling are fitted within training folds.

Predicted labels mean 1=planet-like signal, 0=false-positive-like signal. A calibrated
probability refers to the sampled catalog classes and is not confirmation that a
planet exists. The interface withholds probability display for unsupported cases
and does not force a binary decision on weak detections.

## Reproduction

The included independent test has been opened once and is locked against reuse.
Use verify_project.py and smoke_test_app.py to verify the saved artifact. Further
model development requires a new protocol and additional untouched test stars.
The historical three-sector development code, model, reports, and data tables are
preserved under history/2026-11-07_v3. The superseded expanded run is retained under
data/v4 and history/v4_pre_epoch_fix. The final corrected protocol is data/v5.
