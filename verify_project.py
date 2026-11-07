"""Check the saved real-data model and export its held-out evaluation plot."""
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from tess_pipeline import ROOT, CONFIG
from svm_classifier import predict
from experiment import RUN_DIR


def main():
    report = json.loads((ROOT / "training_report.json").read_text())
    artifact = joblib.load(ROOT / "svm_model.joblib")
    assert artifact["pipeline"] == CONFIG, "Pipeline mismatch"
    assert not set(report["train_tic_ids"]) & set(report["test_tic_ids"]), "Star leakage"
    rows = pd.read_csv(RUN_DIR / "independent_test_labeled.csv", dtype={"tic_id": str})
    plan = json.loads((RUN_DIR / "split_plan.json").read_text())
    frozen = json.loads((RUN_DIR / "frozen_model.json").read_text())
    assert not set(report["test_tic_ids"]) & set(plan["legacy_seen_tic_ids"])
    assert not set(report["train_tic_ids"]) & set(plan["quarantined_development_tic_ids"])
    assert hashlib.sha256((RUN_DIR / "frozen_model.joblib").read_bytes()).hexdigest() == frozen["model_sha256"]
    assert hashlib.sha256((RUN_DIR / "independent_test_labeled.csv").read_bytes()).hexdigest() == report["independent_evaluation"]["test_features_sha256"]
    test = pd.DataFrame(report["test_predictions"])
    aligned = test.merge(rows, on=["tic_id", "sector"], suffixes=("_report", ""), validate="one_to_one")
    assert len(aligned) == report["test_rows"], "Missing held-out rows"
    result = predict(aligned, ROOT / "svm_model.joblib")
    np.testing.assert_allclose(result.planet_candidate_probability, aligned.probability, atol=1e-12)
    np.testing.assert_array_equal(result.predicted_label, aligned.prediction)
    from svm_classifier import predict as infer
    frozen_result = infer(aligned, RUN_DIR / "frozen_model.joblib")
    np.testing.assert_allclose(result.planet_candidate_probability, frozen_result.planet_candidate_probability, atol=1e-12)
    assert aligned.catalog_match.all()
    for filename in aligned.source_file:
        with (ROOT / "data/lightcurves" / filename).open("rb") as handle:
            assert handle.read(6) == b"SIMPLE", "Missing real FITS data"
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    test.to_csv(output / "held_out_predictions.csv", index=False)
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots(1, 2)
    ConfusionMatrixDisplay.from_predictions(aligned.label, aligned.prediction,
        labels=[0, 1], display_labels=["False positive", "Planet"], ax=axes[0], colorbar=False)
    PrecisionRecallDisplay.from_predictions(aligned.label, aligned.probability, ax=axes[1])
    axes[0].set_title(f"Independent new stars (n={len(aligned)})")
    axes[1].set_title(f"{report['selected_model']} · untouched test opened once")
    figure.suptitle("Real TESS · recovered catalog signals only")
    figure.savefig(output / "evaluation.png", dpi=160)
    print("PASS: real FITS provenance, frozen model/test hashes, new disjoint stars, no legacy holdout fitting, and prediction reproducibility.")
    print("Saved outputs/evaluation.png and outputs/held_out_predictions.csv")


if __name__ == "__main__":
    main()
