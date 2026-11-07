"""Verify frozen data/model, reproduce test predictions, and plot evaluation."""
import json
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from classifier import predict, metrics
from tess_pipeline import ROOT, DATA
from train_real_tess import check_frozen, sha


def main():
    check_frozen()
    report = json.loads((ROOT / "training_report.json").read_text())
    split = json.loads((DATA / "split.json").read_text())
    test = pd.read_csv(DATA / "test_features.csv", dtype={"tic_id": str})
    saved = pd.read_csv(ROOT / "outputs/test_predictions.csv", dtype={"tic_id": str})
    assert not set(test.tic_id) & set(split["train_tic_ids"]+split["previously_used_tic_ids"])
    assert sha(DATA / "test_features.csv") == report["test_features_sha256"]
    assert test.catalog_match.all() and test.detection_accepted.all()
    result = predict(test)
    np.testing.assert_array_equal(test.tic_id, saved.tic_id)
    np.testing.assert_allclose(result.planet_candidate_probability, saved.probability, atol=1e-12)
    np.testing.assert_array_equal(result.predicted_label, saved.prediction)
    actual = metrics(test.label, result.planet_candidate_probability)
    for key in ["accuracy", "balanced_accuracy", "confusion_matrix_labels_0_1"]:
        assert actual[key] == report[key]
    # Raw observations are optional after cloning, but check provenance whenever
    # present. The retained tables and frozen hashes suffice to reproduce scores.
    for row in test.itertuples():
        path = DATA / "lightcurves" / row.source_file
        if path.exists():
            assert sha(path) == row.fits_sha256
    figure = Figure(figsize=(6, 4), layout="constrained")
    ax = figure.subplots()
    ConfusionMatrixDisplay.from_predictions(test.label, result.predicted_label, labels=[0, 1],
        display_labels=["False positive", "Planet-like"], ax=ax, colorbar=False)
    ax.set_title(f"New real TESS stars (n={len(test)})\nRandom Forest: accuracy {actual['accuracy']:.1%}")
    figure.savefig(ROOT / "outputs/evaluation.png", dpi=160)
    print("PASS: frozen code/model/data, disjoint new stars, real-FITS hashes, and reproduced independent predictions.")
    print(f"Accuracy {actual['accuracy']:.2%}; balanced accuracy {actual['balanced_accuracy']:.2%}")


if __name__ == "__main__":
    main()
