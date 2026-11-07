"""Rebuild real TESS features, fit a Random Forest, then evaluate once."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from tess_pipeline import ROOT, DATA, CONFIG, download_tic, analyze
from search_pipeline import detection_accepted
from transit_features import FEATURES
from classifier import train, metrics, calibration_summary, MODEL_PATH


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def science_hashes():
    return {name: sha(ROOT / name) for name in ["classifier.py", "search_pipeline.py",
        "transit_features.py", "tess_pipeline.py", "train_real_tess.py"]}


def match_ephemeris(features, catalog):
    """Associate blind BLS with a label, allowing half/double P.

    Propagate the catalog epoch with catalog P; noisy fitted P must not
    accumulate phase error over years. Catalog values never enter the model.
    """
    period, cp = features["period_days"], float(catalog["pl_orbper"])
    ratio = period/cp
    alias = min([.5, 1., 2.], key=lambda value: abs(ratio/value-1))
    association_period = cp*min(alias, 1.)
    delta = (features["epoch_btjd"]+2457000-float(catalog["pl_tranmid"])
             +association_period/2) % association_period-association_period/2
    def uncertainty(name):
        value = catalog.get(name, 0)
        return abs(float(value)) if pd.notna(value) else 0.
    cycles = abs((features["epoch_btjd"]+2457000-float(catalog["pl_tranmid"]))/cp)
    propagated = np.hypot(uncertainty("pl_tranmiderr1"), cycles*uncertainty("pl_orbpererr1"))
    tolerance = min(.15*period, max(.02, 1.5*float(catalog["pl_trandurh"])/24)
                    +min(3*propagated, .1*period))
    period_tolerance = min(.01, max(.0015, 3*uncertainty("pl_orbpererr1")/cp))
    return bool(abs(ratio/alias-1) < period_tolerance and abs(delta) < tolerance)


def process_observation(row):
    path = download_tic(row["tic_id"], row["sector"], row["filename"])
    features = analyze(path, n_bootstraps=0)["features"]
    features.update(label=int(row["label"]), disposition=row["tfopwg_disp"],
                    catalog_match=match_ephemeris(features, row),
                    detection_accepted=detection_accepted(features), fits_sha256=sha(path))
    return features


def extract(partition, workers=4):
    plan = pd.read_csv(DATA / "observation_plan.csv", dtype={"tic_id": str})
    selected = plan[plan.partition == partition]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_observation, row): row for row in selected.to_dict("records")}
        for i, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                feature = future.result()
                rows.append(feature)
                status = "accepted" if feature["catalog_match"] and feature["detection_accepted"] else "unmatched/weak"
            except Exception as error:
                errors.append({"tic_id": row["tic_id"], "error": str(error)})
                status = str(error)
            print(f"{i}/{len(selected)} TIC {row['tic_id']}: {status}", flush=True)
            pd.DataFrame(rows).to_csv(DATA / f"{partition}_extracted.csv", index=False)
            pd.DataFrame(errors, columns=["tic_id", "error"]).to_csv(DATA / f"{partition}_errors.csv", index=False)
    frame = pd.DataFrame(rows).sort_values("tic_id")
    accepted = frame[frame.catalog_match & frame.detection_accepted].copy()
    accepted.to_csv(DATA / f"{partition}_features.csv", index=False)
    return accepted


def fit():
    if (DATA / "evaluation_lock.json").exists():
        raise RuntimeError("This independent test is already open. Reserve additional untouched TICs before further training.")
    split = json.loads((DATA / "split.json").read_text())
    frame = pd.read_csv(DATA / "training_features.csv", dtype={"tic_id": str})
    if set(frame.tic_id) != set(split["train_tic_ids"]):
        raise ValueError("Training TICs differ from the reserved split.")
    if set(frame.tic_id) & set(split["test_tic_ids"]):
        raise ValueError("Training/test TIC overlap.")
    print(f"Fitting Random Forest on {len(frame)} real TESS stars...", flush=True)
    artifact, report = train(frame)
    joblib.dump(artifact, MODEL_PATH, compress=3)
    report.update(created_utc=datetime.now(timezone.utc).isoformat(), pipeline=CONFIG,
        train_rows=len(frame), train_tic_ids=sorted(frame.tic_id),
        training_class_counts=frame.label.value_counts().sort_index().to_dict(),
        test_status="UNOPENED", previous_model_result=split["previous_model_result"],
        limitations="Catalog-selected, ephemeris-recovered CP/KP versus FP signals. Not survey completeness, pure-noise performance, or planet confirmation. Probabilities apply to this sampled population.")
    (ROOT / "training_report.json").write_text(json.dumps(report, indent=2))
    frozen = {"model_sha256": sha(MODEL_PATH), "training_sha256": sha(DATA / "training_features.csv"),
        "split_sha256": sha(DATA / "split.json"), "plan_sha256": sha(DATA / "observation_plan.csv"),
        "catalog_sha256": sha(DATA / "toi_catalog_full.csv"), "code": science_hashes()}
    (DATA / "frozen.json").write_text(json.dumps(frozen, indent=2))
    print("Frozen before opening test. Training CV balanced accuracy:", report["training_cv_balanced_accuracy"], flush=True)


def check_frozen():
    frozen = json.loads((DATA / "frozen.json").read_text())
    checks = {"model_sha256": MODEL_PATH, "training_sha256": DATA / "training_features.csv",
        "split_sha256": DATA / "split.json", "plan_sha256": DATA / "observation_plan.csv",
        "catalog_sha256": DATA / "toi_catalog_full.csv"}
    for key, path in checks.items():
        if frozen[key] != sha(path):
            raise ValueError(f"Frozen data/model changed: {path.name}")
    if frozen["code"] != science_hashes():
        raise ValueError("Scientific code changed after the model was frozen.")
    return frozen


def evaluate(workers=4):
    check_frozen()
    split = json.loads((DATA / "split.json").read_text())
    test_ids = set(split["test_tic_ids"])
    if test_ids & set(split["previously_used_tic_ids"]+split["train_tic_ids"]):
        raise ValueError("Test includes previously used stars.")
    with (DATA / "evaluation_lock.json").open("x") as handle:
        json.dump({"opened_utc": datetime.now(timezone.utc).isoformat()}, handle)
    test = extract("test", workers)
    check_frozen()
    artifact = joblib.load(MODEL_PATH)
    probability = artifact["model"].predict_proba(test[FEATURES])[:, 1]
    report = json.loads((ROOT / "training_report.json").read_text())
    report.update(metrics(test.label.to_numpy(int), probability))
    cal = calibration_summary(test.label.to_numpy(int), probability)
    report.update(test_rows=len(test), test_tic_ids=sorted(test.tic_id),
        test_class_counts=test.label.value_counts().sort_index().to_dict(),
        matched_rows=report["train_rows"]+len(test), total_assigned_stars=len(split["train_tic_ids"])+len(test_ids),
        test_reserved_rows=len(test_ids), test_classification_coverage=len(test)/len(test_ids),
        correct_fraction_of_reserved=float(np.sum((probability >= .5) == test.label))/len(test_ids),
        majority_baseline_accuracy=float(test.label.value_counts().max()/len(test)),
        test_status="EVALUATED_ONCE", evaluation_status="New TICs reserved before fitting; frozen model evaluated once.",
        independent_calibration=cal, test_features_sha256=sha(DATA / "test_features.csv"),
        probability_reliable=bool(report["calibration"]["probability_reliable"] and cal["probability_reliable"]),
        accuracy_target_met=report["accuracy"] >= .7 and report["balanced_accuracy"] >= .7)
    correct = int(np.sum((probability >= .5) == test.label))
    n, z = len(test), 1.96
    center = (correct/n+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt((correct/n)*(1-correct/n)/n+z*z/(4*n*n))/(1+z*z/n)
    report["accuracy_wilson_95"] = [center-half, center+half]
    predictions = test[["tic_id", "sector", "label"]].copy()
    predictions["probability"] = probability
    predictions["prediction"] = (probability >= .5).astype(int)
    (ROOT / "outputs").mkdir(exist_ok=True)
    predictions.to_csv(ROOT / "outputs/test_predictions.csv", index=False)
    (ROOT / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ["accuracy", "balanced_accuracy", "test_rows",
        "test_class_counts", "confusion_matrix_labels_0_1", "accuracy_target_met"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["fit", "evaluate", "extract-training"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "fit":
        fit()
    elif args.stage == "evaluate":
        evaluate(args.workers)
    else:
        if (DATA / "frozen.json").exists():
            raise RuntimeError("Frozen training table is immutable; rebuild in a new experiment directory.")
        extract("training", args.workers)


if __name__ == "__main__":
    main()
