"""Real-data collection, training freeze, and one-time independent evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tess_pipeline import ROOT, DATA, CONFIG, download_tic, analyze
from experiment import RUN_DIR

V4 = RUN_DIR


def match_ephemeris(features, catalog):
    """Catalog association only; catalog ephemerides never become features."""
    period, cp = features["period_days"], float(catalog["pl_orbper"])
    ratio = period/cp
    alias = min([.5, 1., 2.], key=lambda value: abs(ratio/value-1))
    relative_error = abs(ratio/alias-1)
    # Use the catalog period for long-baseline projection, not the noisy fitted
    # period. A double-P detection may select either primary-transit parity.
    association_period = cp*min(alias, 1.)
    delta = (features["epoch_btjd"]+2457000-float(catalog["pl_tranmid"])+association_period/2) % association_period-association_period/2
    def uncertainty(name):
        value = catalog.get(name, 0)
        return abs(float(value)) if pd.notna(value) else 0.
    cycles = abs((features["epoch_btjd"]+2457000-float(catalog["pl_tranmid"]))/cp)
    propagated = np.hypot(uncertainty("pl_tranmiderr1"), cycles*uncertainty("pl_orbpererr1"))
    tolerance = min(.15*period, max(.02, 1.5*float(catalog["pl_trandurh"])/24)+min(3*propagated, .1*period))
    period_tolerance = min(.01, max(.0015, 3*uncertainty("pl_orbpererr1")/cp))
    return {"catalog_match": bool(relative_error < period_tolerance and abs(delta) < tolerance),
            "catalog_period_days": cp, "period_alias": alias, "relative_period_error": relative_error,
            "epoch_error_days": abs(delta), "epoch_tolerance_days": tolerance}


def process_observation(row):
    path = download_tic(row["tic_id"], row["sector"], row["filename"])
    folder = V4 / "cache"
    folder.mkdir(exist_ok=True)
    cache = folder / f"{Path(row['filename']).stem}.json"
    try:
        saved = json.loads(cache.read_text()) if cache.exists() else {}
    except json.JSONDecodeError:
        saved = {}
    if saved.get("pipeline") == CONFIG:
        features = saved["features"]
    else:
        result = analyze(path)
        features = result["features"]
        record = {"pipeline": CONFIG, "features": features, "candidate_peaks": result["candidate_peaks"],
                  "fits_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(cache)
    features = features.copy()
    features.update(match_ephemeris(features, row))
    features.update({"label": int(row["label"]), "disposition": row["tfopwg_disp"],
                     "toi": row["toi"], "partition": row["partition"]})
    features["detection_accepted"] = bool(features["bls_snr"] >= 6 and features["pink_snr"] >= 3 and
        features["observed_transits"] >= 2 and features["event_fraction"] >= .4)
    return features


def extract(partition, workers=4):
    plan = pd.read_csv(V4 / "observation_plan.csv", dtype={"tic_id": str})
    selected = plan[plan.partition == partition]
    rows, errors = [], []
    print(f"Extracting {len(selected)} {partition} stars, pipeline v{CONFIG['version']}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(process_observation, row): row for row in selected.to_dict("records")}
        for index, future in enumerate(as_completed(pending), 1):
            row = pending[future]
            try:
                result = future.result()
                rows.append(result)
                status = "accepted" if result["catalog_match"] and result["detection_accepted"] else "unmatched/weak"
            except Exception as error:
                errors.append({"tic_id": row["tic_id"], "sector": row["sector"], "error": str(error)})
                status = f"failed: {error}"
            print(f"{index}/{len(selected)} TIC {row['tic_id']} sector {row['sector']}: {status}", flush=True)
            pd.DataFrame(rows).to_csv(V4 / f"{partition}_extracted.csv", index=False)
            pd.DataFrame(errors, columns=["tic_id", "sector", "error"]).to_csv(V4 / f"{partition}_errors.csv", index=False)
    frame = pd.DataFrame(rows).sort_values("tic_id")
    accepted = frame[frame.catalog_match & frame.detection_accepted].copy()
    accepted.to_csv(V4 / f"{partition}_labeled.csv", index=False)
    print(f"Saved {len(accepted)} matched, accepted {partition} stars", flush=True)
    return accepted


def fit():
    from svm_classifier import train
    plan = json.loads((V4 / "split_plan.json").read_text())
    frame = pd.read_csv(V4 / "training_labeled.csv", dtype={"tic_id": str})
    forbidden = set(plan["test_tic_ids"]+plan["quarantined_development_tic_ids"])
    assert not set(frame.tic_id) & forbidden
    assert frame.tic_id.is_unique
    if (V4 / "evaluation_lock.json").exists():
        raise RuntimeError("Independent evaluation already opened. Create a new protocol/data split for further development.")
    report = train(frame, ROOT / "svm_model.joblib")
    report.update({"pipeline": CONFIG, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "v5", "epoch_association_method": "Catalog period for epoch projection; detected period only for relative-period agreement and alias identification.",
        "dataset": "TESS SPOC two-minute PDCSAP; CP/KP vs vetted TOI FP", "sectors": plan["sectors"], "sector": 1,
        "unique_stars": len(frame), "matched_rows": len(frame), "train_rows": len(frame),
        "train_tic_ids": sorted(frame.tic_id), "quarantined_development_tic_ids": plan["quarantined_development_tic_ids"],
        "split_plan_sha256": hashlib.sha256((V4 / "split_plan.json").read_bytes()).hexdigest(),
        "historical_evaluations": [json.loads((ROOT / name).read_text()) for name in
            ["history/2026-11-07_v3/training_report.json", "outputs/initial_baseline_report.json", "outputs/five_feature_development_report.json"]],
        "limitations": "Catalog-selected, ephemeris-recovered signals only. Not blind-survey completeness, false-alarm rate, or confirmation. Probabilities refer to the sampled CP/KP-vs-FP population."})
    report["superseded_independent_run"] = {
        "reason": "Final training-period audit found that epoch projection used the fitted period rather than the catalog period. The v4 test was opened and is quarantined; its score is preserved but superseded. v5 uses a freshly assigned disjoint test set and corrected catalog association.",
        "report": json.loads((ROOT / "history/v4_pre_epoch_fix/training_report.json").read_text())}
    (V4 / "frozen_training_report.json").write_text(json.dumps(report, indent=2))
    shutil.copy2(ROOT / "svm_model.joblib", V4 / "frozen_model.joblib")
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
              "model_sha256": hashlib.sha256((V4 / "frozen_model.joblib").read_bytes()).hexdigest(),
              "training_features_sha256": hashlib.sha256((V4 / "training_labeled.csv").read_bytes()).hexdigest(),
              "processing_code_sha256": hashlib.sha256((ROOT / "search_pipeline.py").read_bytes()+(ROOT / "transit_features.py").read_bytes()).hexdigest(),
              "association_code_sha256": hashlib.sha256((ROOT / "training_workflow.py").read_bytes()).hexdigest(),
              "split_plan_sha256": report["split_plan_sha256"], "pipeline": CONFIG}
    (V4 / "frozen_model.json").write_text(json.dumps(frozen, indent=2))
    print("Model frozen. Independent test may now be extracted and evaluated once.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["extract", "fit", "evaluate"], required=True)
    parser.add_argument("--partition", choices=["training", "independent_test"], default="training")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "extract":
        if args.partition == "independent_test" and not (V4 / "frozen_model.json").exists():
            parser.error("Freeze the training-selected model before opening independent test observations.")
        extract(args.partition, args.workers)
    elif args.stage == "fit":
        fit()
    else:
        from evaluate_model import evaluate_once
        evaluate_once()
