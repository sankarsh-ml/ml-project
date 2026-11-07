"""One brightness-only Random Forest, calibrated using training stars only."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from transit_features import FEATURES

MODEL_PATH = Path(__file__).resolve().parent / "planet_model.joblib"


def feature_matrix(frame):
    """Identifiers, catalog labels and catalog ephemerides are never inputs."""
    x = frame[FEATURES].astype(float)
    base = x[["period_days", "duration_days", "depth", "flux_std", "duty_cycle"]]
    if not np.isfinite(base).all().all() or (base <= 0).any().any():
        raise ValueError("BLS measurements must be finite and positive.")
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Infinite feature values are invalid.")
    if (x.duration_days >= x.period_days).any() or not np.allclose(x.duty_cycle, x.duration_days/x.period_days):
        raise ValueError("Invalid duration/period ratio.")
    return x


def grouped_folds(x, y, groups, n_splits=5):
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=714).split(x, y, groups))
    for fit, valid in folds:
        if set(y[fit]) != {0, 1} or set(y[valid]) != {0, 1}:
            raise ValueError("Each fold needs both classes.")
        assert not set(groups[fit]) & set(groups[valid])
    return folds


def calibration_summary(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    bins = np.minimum((probability*5).astype(int), 4)
    ece = sum(np.mean(bins == b)*abs(np.mean(y[bins == b])-np.mean(probability[bins == b]))
              for b in range(5) if np.any(bins == b))
    brier, prior = brier_score_loss(y, probability), float(np.var(y))
    return {"ece_5_bins": float(ece), "brier_score": float(brier), "prior_brier": prior,
            "probability_reliable": bool(len(y) >= 100 and ece <= .12 and brier < prior)}


def metrics(y, probability):
    labels = (np.asarray(probability) >= .5).astype(int)
    return {"accuracy": float(accuracy_score(y, labels)),
            "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
            "metrics": classification_report(y, labels, output_dict=True, zero_division=0),
            "confusion_matrix_labels_0_1": confusion_matrix(y, labels, labels=[0, 1]).tolist()}


def train(frame):
    x = feature_matrix(frame)
    y, groups = frame.label.to_numpy(int), frame.tic_id.astype(str).to_numpy()
    if pd.Series(groups).duplicated().any():
        raise ValueError("Use one observation per TIC.")
    forest = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", random_state=714, n_jobs=1))
    grid = GridSearchCV(forest, {"randomforestclassifier__min_samples_leaf": [2, 5, 10],
        "randomforestclassifier__max_features": [.5, 1.]}, scoring="balanced_accuracy",
        cv=grouped_folds(x, y, groups), n_jobs=4, error_score="raise")
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        grid.fit(x, y)
    # Out-of-fold calibration check; hyperparameter selection used these training
    # data too, so this is a development estimate, not the independent score.
    probability = np.empty(len(y))
    for fit, valid in grouped_folds(x, y, groups, 3):
        calibrated = CalibratedClassifierCV(clone(grid.best_estimator_), method="sigmoid",
            cv=grouped_folds(x.iloc[fit], y[fit], groups[fit], 3), n_jobs=1)
        calibrated.fit(x.iloc[fit], y[fit])
        probability[valid] = calibrated.predict_proba(x.iloc[valid])[:, 1]
    model = CalibratedClassifierCV(clone(grid.best_estimator_), method="sigmoid",
        cv=grouped_folds(x, y, groups), n_jobs=1)
    model.fit(x, y)
    from tess_pipeline import CONFIG
    calibration = calibration_summary(y, probability)
    artifact = {"model": model, "model_name": "Random Forest", "features": FEATURES,
        "pipeline": CONFIG, "threshold": .5, "probability_reliable": calibration["probability_reliable"],
        "feature_min": x.min().to_dict(), "feature_max": x.max().to_dict()}
    report = {"selected_model": "Random Forest", "best_parameters": grid.best_params_,
        "training_cv_balanced_accuracy": float(grid.best_score_), "training_oof": metrics(y, probability),
        "calibration": calibration, "features": FEATURES, "threshold": .5,
        "selection_rule": "Six Random Forest configurations; five grouped training folds; maximize balanced accuracy. Fixed threshold 0.5. No test-based selection."}
    return artifact, report


def predict(frame, model_path=MODEL_PATH):
    artifact = joblib.load(model_path)
    if artifact["features"] != FEATURES:
        raise ValueError("Saved model feature schema differs from this code.")
    result = frame.copy()
    probability = artifact["model"].predict_proba(feature_matrix(frame))[:, 1]
    result["planet_candidate_probability"] = probability
    result["predicted_label"] = (probability >= .5).astype(int)
    result["probability_reliable"] = artifact["probability_reliable"]
    return result
