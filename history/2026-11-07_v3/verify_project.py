"""Check the saved real-data model and export its held-out evaluation plot."""
import json
import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from tess_pipeline import ROOT, CONFIG
from svm_classifier import predict


def main():
    report = json.loads((ROOT / "training_report.json").read_text())
    artifact = joblib.load(ROOT / "svm_model.joblib")
    assert artifact["pipeline"] == CONFIG, "Pipeline mismatch"
    assert not set(report["train_tic_ids"]) & set(report["test_tic_ids"]), "Star leakage"
    rows = pd.read_csv(ROOT / "data/labeled_features.csv", dtype={"tic_id": str})
    test = pd.DataFrame(report["test_predictions"])
    aligned = test.merge(rows, on=["tic_id", "sector"], suffixes=("_report", ""), validate="one_to_one")
    assert len(aligned) == report["test_rows"], "Missing held-out rows"
    result = predict(aligned, ROOT / "svm_model.joblib")
    np.testing.assert_allclose(result.planet_candidate_probability, aligned.probability, atol=1e-12)
    np.testing.assert_array_equal(result.predicted_label, aligned.prediction)
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
    axes[0].set_title(f"Held-out observations (n={len(aligned)})")
    axes[1].set_title(f"{len(report['test_tic_ids'])} distinct held-out stars")
    figure.suptitle("Real TESS pilot · recovered catalog signals only")
    figure.savefig(output / "evaluation.png", dpi=160)
    print("PASS: real FITS provenance, disjoint stars, pipeline compatibility, saved-model prediction reproducibility.")
    print("Saved outputs/evaluation.png and outputs/held_out_predictions.csv")


if __name__ == "__main__":
    main()
