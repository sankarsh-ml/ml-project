"""Training-only model comparison, grouped calibration, and inference."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, brier_score_loss, log_loss, roc_auc_score, average_precision_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from transit_features import FEATURES, vetting_features, stellar_features
from experiment import RUN_DIR


def feature_matrix(frame):
    missing = set(FEATURES)-set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    x = frame[FEATURES].astype(float)
    base = ["period_days", "duration_days", "depth", "flux_std", "duty_cycle"]
    if not np.isfinite(x[base]).all().all() or (x[base] <= 0).any().any():
        raise ValueError("Basic BLS features must be finite and positive.")
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Infinite feature values are invalid; missing metadata may use NaN.")
    if (x.duration_days >= x.period_days).any() or not np.allclose(x.duty_cycle, x.duration_days/x.period_days):
        raise ValueError("Invalid duration/period ratio.")
    return x


def physics_transform(x):
    """A fixed physical transform, learned scales/imputation remain inside CV."""
    a = np.asarray(x, dtype=float).copy()
    signed = {"secondary_significance", "minimum_event_snr", "median_event_snr", "noise_skew", "noise_kurtosis", "local_depth_snr"}
    for i, name in enumerate(FEATURES):
        a[:, i] = np.arcsinh(a[:, i]) if name in signed else np.log10(np.maximum(a[:, i], 1e-6))
    return a


def grouped_folds(x, y, groups, n_splits=5):
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=714).split(x, y, groups))
    for fit, validation in folds:
        if len(np.unique(y[fit])) != 2 or len(np.unique(y[validation])) != 2:
            raise ValueError("Each grouped fold needs both classes; expand labeled stars.")
        assert not set(groups[fit]) & set(groups[validation])
    return folds


def metrics(y, probability, threshold):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    report = classification_report(y, prediction, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "metrics": report, "confusion_matrix_labels_0_1": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            "average_precision": float(average_precision_score(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1]))}


def calibration_summary(y, probability):
    bins = np.minimum((np.asarray(probability)*5).astype(int), 4)
    ece = sum(np.mean(bins == b)*abs(float(np.mean(np.asarray(y)[bins == b]))-
              float(np.mean(np.asarray(probability)[bins == b]))) for b in range(5) if np.any(bins == b))
    base = float(np.mean((np.asarray(y)-np.mean(y))**2))
    brier = float(brier_score_loss(y, probability))
    return {"ece_5_bins": float(ece), "brier_score": brier, "prior_baseline_brier": base,
            "probability_reliable": bool(len(y) >= 100 and ece <= .12 and brier < base),
            "note": "Estimated calibration for the sampled catalog population, not a physical planet-confirmation probability."}


def threshold_from_training(y, probability):
    thresholds = np.linspace(.1, .9, 161)
    scores = np.array([balanced_accuracy_score(y, probability >= t) for t in thresholds])
    # Select a stable near-optimal threshold rather than an isolated sample-level maximum.
    eligible = thresholds[scores >= scores.max()-.01]
    return float(eligible[np.argmin(np.abs(eligible-.5))])


def calibrated_oof(estimator, x, y, groups):
    out = np.empty(len(y))
    for fit, validation in grouped_folds(x, y, groups, 3):
        inner = grouped_folds(x.iloc[fit], y[fit], groups[fit], 3)
        model = CalibratedClassifierCV(clone(estimator), method="sigmoid", cv=inner, ensemble=True, n_jobs=1)
        model.fit(x.iloc[fit], y[fit])
        out[validation] = model.predict_proba(x.iloc[validation])[:, 1]
    return out


def scaled(estimator):
    return Pipeline([("physical", FunctionTransformer(physics_transform)),
                     ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                     ("scale", StandardScaler()), ("classifier", estimator)])


def trees(estimator):
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                     ("classifier", estimator)])


def _train_impl(frame, model_path):
    frame = frame.sort_values("tic_id").reset_index(drop=True)
    x = feature_matrix(frame)
    y, groups = frame.label.to_numpy(int), frame.tic_id.astype(str).to_numpy()
    if set(y) != {0, 1}:
        raise ValueError("Both vetted classes are required.")
    if pd.Series(groups).duplicated().any():
        raise ValueError("Use one observation per TIC in the expanded benchmark.")
    root = Path(__file__).resolve().parent
    split_path = RUN_DIR / "split_plan.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        forbidden = set(split["test_tic_ids"]+split["quarantined_development_tic_ids"])
        if set(groups) & forbidden:
            raise ValueError("Independent or quarantined development TIC present in training data.")
    folds = grouped_folds(x, y, groups, 5)
    families = {
        "SVM": (scaled(SVC(kernel="rbf", class_weight="balanced")),
            {"classifier__C": [.1, 1, 10, 100, 1000], "classifier__gamma": ["scale", .003, .01, .03, .1]}),
        "Random Forest": (trees(RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", random_state=714, n_jobs=1)),
            {"classifier__min_samples_leaf": [2, 5, 10], "classifier__max_features": [.5, 1.0]}),
        "Extra Trees": (trees(ExtraTreesClassifier(n_estimators=350, class_weight="balanced", random_state=714, n_jobs=1)),
            {"classifier__min_samples_leaf": [2, 5, 10], "classifier__max_features": [.5, 1.0]}),
        "Gradient Boosting": (trees(HistGradientBoostingClassifier(max_iter=180, learning_rate=.06, class_weight="balanced", random_state=714)),
            {"classifier__max_leaf_nodes": [7, 15], "classifier__min_samples_leaf": [8, 20], "classifier__l2_regularization": [1., 10.]}),
        "Logistic baseline": (scaled(LogisticRegression(class_weight="balanced", max_iter=3000, random_state=714)),
            {"classifier__C": [.01, .1, 1, 10]}),
    }
    comparisons, fitted, oof_scores = [], {}, {}
    models_folder = root / "models"
    models_folder.mkdir(exist_ok=True)
    for name, (estimator, grid) in families.items():
        print(f"Training-only search: {name}", flush=True)
        search = GridSearchCV(estimator, grid, cv=folds, scoring="balanced_accuracy", n_jobs=4, error_score="raise")
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            search.fit(x, y)
        best = search.best_estimator_
        probability = calibrated_oof(best, x, y, groups)
        threshold = threshold_from_training(y, probability)
        summary = metrics(y, probability, threshold)
        std = float(search.cv_results_["std_test_score"][search.best_index_])
        summary.update({"model": name, "parameters": search.best_params_,
            "grid_cv_balanced_accuracy": float(search.best_score_), "grid_cv_std": std,
            "threshold": threshold, "calibration": calibration_summary(y, probability),
            "selection_score": summary["balanced_accuracy"]-.25*std,
            "minimum_class_recall": min(summary["metrics"][str(label)]["recall"] for label in [0, 1]),
            "scope": "Training-only selection estimates; hyperparameters and threshold were selected on these training data. Final test is unopened."})
        comparisons.append(summary)
        fitted[name] = best
        oof_scores[name] = probability
        pd.DataFrame({"tic_id": groups, "label": y, "probability": probability}).to_csv(
            models_folder / f"{name.lower().replace(' ', '_')}_training_oof.csv", index=False)
        print(f"{name}: CV balanced={summary['grid_cv_balanced_accuracy']:.3f}, calibrated training OOF balanced={summary['balanced_accuracy']:.3f}, min recall={summary['minimum_class_recall']:.3f}", flush=True)
    viable = [row for row in comparisons if row["minimum_class_recall"] >= .6]
    best = max(viable or comparisons, key=lambda row: row["selection_score"])
    svm = next(row for row in comparisons if row["model"] == "SVM")
    if svm["minimum_class_recall"] >= .6 and svm["selection_score"] >= best["selection_score"]-.01:
        best = svm
    cv = grouped_folds(x, y, groups, 5)
    # Save an SVM candidate even when a tree model is more reliable in training CV.
    for selected in [svm] + ([] if best is svm else [best]):
        name = selected["model"]
        calibrated = CalibratedClassifierCV(clone(fitted[name]), method="sigmoid", cv=cv, ensemble=True, n_jobs=1)
        calibrated.fit(x, y)
        from tess_pipeline import CONFIG
        artifact = {"model": calibrated, "model_name": name, "features": FEATURES, "pipeline": CONFIG,
                    "threshold": selected["threshold"], "calibration": selected["calibration"],
                    "probability_reliable": selected["calibration"]["probability_reliable"],
                    "training_tic_ids": sorted(groups), "feature_min": x.min().to_dict(), "feature_max": x.max().to_dict()}
        joblib.dump(artifact, models_folder / ("svm_candidate.joblib" if name == "SVM" else "tree_candidate.joblib"))
        if name == best["model"]:
            joblib.dump(artifact, model_path)
            joblib.dump(artifact, models_folder / "best_model.joblib")
    report = {"selected_model": best["model"], "selection_rule": "Maximize training-only calibrated OOF balanced accuracy minus 0.25 CV standard deviation, require >=0.6 recall in both classes where feasible. Prefer SVM only within 0.01 of best selection score.",
              "model_comparison": comparisons, "threshold": best["threshold"], "calibration": best["calibration"],
              "probability_reliable": best["calibration"]["probability_reliable"],
              "best_parameters": best["parameters"], "training_cv_balanced_accuracy": best["grid_cv_balanced_accuracy"],
              "training_class_counts": {str(label): int((y == label).sum()) for label in [0, 1]},
              "features": FEATURES, "test_status": "UNOPENED"}
    (models_folder / "training_model_comparison.json").write_text(json.dumps(report, indent=2))
    return report


def train(frame, model_path):
    if (RUN_DIR / "evaluation_lock.json").exists():
        raise RuntimeError("This protocol's independent test is already open. Use a new untouched-data protocol before further model development.")
    with threadpool_limits(limits=1):
        return _train_impl(frame, model_path)


def predict(frame, model_path):
    artifact = joblib.load(model_path)
    if artifact["features"] != FEATURES:
        raise ValueError("Saved model feature schema is incompatible. Retrain with this pipeline.")
    probability = artifact["model"].predict_proba(feature_matrix(frame))[:, 1]
    result = frame.copy()
    result["planet_candidate_probability"] = probability
    result["predicted_label"] = (probability >= artifact["threshold"]).astype(int)
    result["probability_reliable"] = artifact.get("probability_reliable", False)
    return result


def bls_features(flat_lc, bls):
    """Compatibility extractor. The full FITS pipeline adds centroid metadata."""
    def scalar(value):
        return float(np.asarray(getattr(value, "value", value)))
    p, d, depth, epoch = map(scalar, [bls.period_at_max_power, bls.duration_at_max_power,
                                    bls.depth_at_max_power, bls.transit_time_at_max_power])
    flux = np.asarray(flat_lc.flux.value)
    result = {name: np.nan for name in FEATURES}
    result.update(period_days=p, duration_days=d, depth=depth, flux_std=float(np.std(flux)), duty_cycle=d/p)
    result.update(vetting_features(np.asarray(flat_lc.time.value), flux, p, epoch, d, depth))
    result.update(stellar_features(flat_lc.meta, p, d, depth))
    return result
