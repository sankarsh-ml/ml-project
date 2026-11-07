"""Final candidate classifier; input is BLS features, not raw flux samples.

Train: python svm_classifier.py train labeled_features.csv svm_model.joblib
Predict: python svm_classifier.py predict new_features.csv svm_model.joblib predictions.csv
Labels: 1 = vetted planet/planet candidate, 0 = vetted false positive/alarm.
Do not label uncatalogued stars as negative merely because they lack a TOI.
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, brier_score_loss
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC


FEATURES = ["period_days", "duration_days", "depth", "flux_std", "duty_cycle",
            "transit_snr", "odd_even_significance", "secondary_ratio", "shape_ratio",
            "depth_scatter_ratio", "red_noise_ratio"]


def bls_features(flat_lc, bls):
    """Extract the same features for training and inference from Lightkurve objects.

    Flux must be normalized near 1. Period and duration must be in days.
    This uses the strongest peak; call again with bls2 and masked_lc for signal 2.
    """
    def scalar(value):
        return float(np.asarray(getattr(value, "value", value)))

    period = scalar(bls.period_at_max_power)
    duration = scalar(bls.duration_at_max_power)
    flux = np.asarray(flat_lc.flux.value, dtype=float)
    result = {
        "period_days": period,
        "duration_days": duration,
        "depth": scalar(bls.depth_at_max_power),
        "flux_std": float(np.nanstd(flux)),
        "duty_cycle": duration / period,
    }
    from tess_pipeline import vetting_features
    result.update(vetting_features(np.asarray(flat_lc.time.value), flux, period,
                  scalar(bls.transit_time_at_max_power), duration, result["depth"]))
    return result


def feature_matrix(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    x = frame[FEATURES].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Features must be finite; inspect failed BLS searches.")
    if (x <= 0).any().any() or (x.duration_days >= x.period_days).any():
        raise ValueError("Features must be positive, with duration < period.")
    if not np.allclose(x.duty_cycle, x.duration_days / x.period_days):
        raise ValueError("duty_cycle must equal duration_days / period_days.")
    return x


def grouped_folds(x, y, groups, n_splits):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(x, y, groups))
    for train, test in splits:
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            raise ValueError("Each fold needs both classes. Supply more labeled stars.")
    return splits


def train(frame, model_path):
    x = feature_matrix(frame)
    if not {"label", "tic_id"}.issubset(frame.columns):
        raise ValueError("Training requires label and tic_id columns.")
    if frame[["label", "tic_id"]].isna().any().any():
        raise ValueError("Labels and TIC identifiers cannot be missing.")
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 and 1.")
    y = frame.label.to_numpy(dtype=int)
    groups = frame.tic_id.astype(str).to_numpy()
    # Fix the holdout before tuning. No star appears in both partitions.
    train_idx, test_idx = grouped_folds(x, y, groups, 5)[0]
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    folds = grouped_folds(x_train, y_train, groups[train_idx], 3)
    # Physical scales span orders of magnitude; all validated features are positive.
    pipeline = make_pipeline(FunctionTransformer(np.log10, feature_names_out="one-to-one"),
                             StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    search = GridSearchCV(
        pipeline, {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]},
        scoring="balanced_accuracy", cv=folds, n_jobs=-1, error_score="raise",
    )
    search.fit(x_train, y_train)
    # Learn the decision threshold using training-only out-of-fold probabilities.
    # Inner calibration folds are recomputed for each outer training partition.
    oof = np.empty(len(y_train))
    for fit_idx, validation_idx in folds:
        inner_folds = grouped_folds(x_train.iloc[fit_idx], y_train[fit_idx],
                                    groups[train_idx][fit_idx], 3)
        candidate = CalibratedClassifierCV(search.best_estimator_, method="sigmoid", cv=inner_folds)
        candidate.fit(x_train.iloc[fit_idx], y_train[fit_idx])
        oof[validation_idx] = candidate.predict_proba(x_train.iloc[validation_idx])[:, 1]
    thresholds = np.unique(np.r_[0.5, np.linspace(0.05, 0.95, 181)])
    scores = [balanced_accuracy_score(y_train, oof >= value) for value in thresholds]
    threshold = float(max(zip(scores, thresholds), key=lambda pair: (pair[0], -abs(pair[1] - 0.5)))[1])
    # Explicit grouped folds also prevent star leakage during calibration.
    model = CalibratedClassifierCV(search.best_estimator_, method="sigmoid", cv=folds)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    prediction = (probability >= threshold).astype(int)
    report = {
        "best_parameters": search.best_params_,
        "selection_metric": "training grouped CV balanced accuracy",
        "training_cv_balanced_accuracy": float(search.best_score_),
        "threshold": threshold,
        "threshold_training_oof_balanced_accuracy": float(max(scores)),
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "metrics": classification_report(y_test, prediction, output_dict=True, zero_division=0),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        "average_precision": float(average_precision_score(y_test, probability)),
        "brier_score": float(brier_score_loss(y_test, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "majority_baseline_accuracy": float(np.mean(y_test == np.bincount(y_train).argmax())),
        "evaluation_status": "Development holdout: excluded from fitting and threshold selection; initial baseline results were inspected during development.",
        "train_tic_ids": sorted(set(groups[train_idx])),
        "test_tic_ids": sorted(set(groups[test_idx])),
        "test_predictions": [
            {"tic_id": str(tic), "label": int(label), "prediction": int(pred),
             "probability": float(prob),
             **({"sector": int(frame.iloc[test_idx].iloc[i]["sector"])} if "sector" in frame else {})}
            for i, (tic, label, pred, prob) in enumerate(zip(groups[test_idx], y_test, prediction, probability))
        ],
    }
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold,
                 "feature_min": x_train.min().to_dict(),
                 "feature_max": x_train.max().to_dict()}, model_path)
    print(json.dumps(report, indent=2))
    return report


def predict(frame, model_path):
    # Only load model files you created or trust: joblib uses pickle.
    artifact = joblib.load(model_path)
    if artifact["features"] != FEATURES:
        raise ValueError("Saved model feature schema does not match this module.")
    model = artifact["model"]
    probability = model.predict_proba(feature_matrix(frame))[:, list(model.classes_).index(1)]
    result = frame.copy()
    result["planet_candidate_probability"] = probability
    result["predicted_label"] = (probability >= artifact["threshold"]).astype(int)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("features_csv")
    parser.add_argument("model_path")
    parser.add_argument("output_csv", nargs="?")
    args = parser.parse_args()
    if args.mode == "predict" and not args.output_csv:
        parser.error("predict requires output_csv")
    data = pd.read_csv(args.features_csv, dtype={"tic_id": str})
    if args.mode == "train":
        train(data, args.model_path)
    else:
        predict(data, args.model_path).to_csv(args.output_csv, index=False)
