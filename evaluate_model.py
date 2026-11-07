"""One-time independent evaluation; never selects or changes a classifier."""
from datetime import datetime, timezone
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from tess_pipeline import ROOT, CONFIG
from model_selection import predict, metrics, calibration_summary
from experiment import RUN_DIR

V4 = RUN_DIR


def wilson(correct, total):
    z = 1.95996398454
    p = correct/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [float(center-radius), float(center+radius)]


def evaluate_once():
    frozen = json.loads((V4 / "frozen_model.json").read_text())
    source = V4 / "frozen_model.joblib"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen["model_sha256"]
    assert frozen["pipeline"] == CONFIG
    assert hashlib.sha256((V4 / "split_plan.json").read_bytes()).hexdigest() == frozen["split_plan_sha256"]
    current_code = hashlib.sha256((ROOT / "search_pipeline.py").read_bytes()+(ROOT / "transit_features.py").read_bytes()).hexdigest()
    assert current_code == frozen["processing_code_sha256"], "Processing changed after model freeze"
    assert hashlib.sha256((ROOT / "training_workflow.py").read_bytes()).hexdigest() == frozen["association_code_sha256"], "Association changed after model freeze"
    report = json.loads((V4 / "frozen_training_report.json").read_text())
    plan = json.loads((V4 / "split_plan.json").read_text())
    test = pd.read_csv(V4 / "independent_test_labeled.csv", dtype={"tic_id": str})
    assert test.tic_id.is_unique
    assert set(test.tic_id) <= set(plan["test_tic_ids"])
    assert not set(test.tic_id) & set(plan["legacy_seen_tic_ids"])
    assert not set(test.tic_id) & set(report["train_tic_ids"])
    assert not set(report["train_tic_ids"]) & set(plan["quarantined_development_tic_ids"])
    if set(test.label) != {0, 1}:
        raise RuntimeError("Both recovered test classes are needed for independent evaluation.")
    lock = {"opened_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": frozen["model_sha256"],
            "test_features_sha256": hashlib.sha256((V4 / "independent_test_labeled.csv").read_bytes()).hexdigest(),
            "status": "opened"}
    # Exclusive creation prevents silently rerunning model selection on opened data.
    with (V4 / "evaluation_lock.json").open("x") as handle:
        json.dump(lock, handle, indent=2)
    artifact = joblib.load(source)
    result = predict(test, source)
    probability = result.planet_candidate_probability.to_numpy()
    y = test.label.to_numpy(int)
    measured = metrics(y, probability, artifact["threshold"])
    calibration = calibration_summary(y, probability)
    # Display policy only; independent data never change model/threshold selection.
    probability_reliable = bool(artifact.get("probability_reliable", False) and len(y) >= 50 and
        calibration["ece_5_bins"] <= .15 and calibration["brier_score"] < calibration["prior_baseline_brier"])
    calibration["standalone_min100_sample_check"] = calibration.pop("probability_reliable")
    calibration["joint_probability_display_supported"] = probability_reliable
    calibration["display_policy"] = "Training OOF: n>=100, ECE<=0.12, Brier beats prior. Independent corroboration: n>=50, ECE<=0.15, Brier beats prior. This changes display metadata only, never model/threshold selection."
    predictions = [{"tic_id": row.tic_id, "sector": int(row.sector), "label": int(row.label),
                    "prediction": int(row.predicted_label), "probability": float(row.planet_candidate_probability),
                    "disposition": row.disposition, "source_file": row.source_file}
                   for row in result.itertuples()]
    report.update(measured)
    report.update({"test_status": "OPENED_ONCE", "test_rows": len(test), "test_tic_ids": sorted(test.tic_id),
        "test_predictions": predictions, "test_class_counts": {str(label): int((y == label).sum()) for label in [0, 1]},
        "test_accuracy_wilson_95": wilson(int((result.predicted_label.to_numpy() == y).sum()), len(y)),
        "independent_calibration": calibration, "probability_reliable": probability_reliable,
        "independent_evaluation": {**lock, "frozen_utc": frozen["frozen_utc"],
            "scope": "One observation per new TIC, from an assignment fixed before extraction. Only catalog-matched signals passing the predeclared detection gate are classified."},
        "evaluation_status": "Independent star-disjoint test opened once after model and threshold were frozen. No legacy development-test star entered fitting or selection.",
        "accuracy_target_met": bool(measured["accuracy"] >= .7),
        "both_classes_identified": bool(all(measured["metrics"][str(label)]["recall"] > 0 for label in [0, 1]))})
    observation_plan = pd.read_csv(V4 / "observation_plan.csv", dtype={"tic_id": str})
    recovery = {}
    for partition in ["training", "independent_test"]:
        extracted = pd.read_csv(V4 / f"{partition}_extracted.csv", dtype={"tic_id": str})
        accepted = extracted[extracted.catalog_match & extracted.detection_accepted]
        recovery[partition] = {"assigned": int((observation_plan.partition == partition).sum()),
            "analyzed": len(extracted), "catalog_matched_and_accepted": len(accepted),
            "accepted_class_counts": {str(label): int((accepted.label == label).sum()) for label in [0, 1]}}
    report["data_recovery"] = recovery
    report["total_assigned_stars"] = len(observation_plan)
    report["total_labeled_stars"] = report["train_rows"]+len(test)
    majority = int(max(report["training_class_counts"], key=lambda label: report["training_class_counts"][label]))
    report["majority_baseline_accuracy"] = float(np.mean(y == majority))
    report["test_classification_coverage"] = len(test)/recovery["independent_test"]["assigned"]
    report["correct_classification_fraction_of_all_reserved"] = int((result.predicted_label.to_numpy() == y).sum())/recovery["independent_test"]["assigned"]
    report["catalog_snapshot_sha256"] = plan["catalog_sha256"]
    (ROOT / "training_report.json").write_text(json.dumps(report, indent=2))
    artifact.update({"report": report, "probability_reliable": probability_reliable,
                     "frozen_model_sha256": frozen["model_sha256"]})
    joblib.dump(artifact, ROOT / "svm_model.joblib")
    joblib.dump(artifact, ROOT / "models/best_model.joblib")
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    result.to_csv(output / "independent_test_predictions.csv", index=False)
    lock["status"] = "completed"
    (V4 / "evaluation_lock.json").write_text(json.dumps(lock, indent=2))
    print(json.dumps({key: report[key] for key in ["selected_model", "accuracy", "balanced_accuracy",
        "metrics", "confusion_matrix_labels_0_1", "train_rows", "test_rows", "probability_reliable", "accuracy_target_met"]}, indent=2))


if __name__ == "__main__":
    evaluate_once()
