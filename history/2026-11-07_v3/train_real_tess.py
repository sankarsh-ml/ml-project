"""Download real TESS observations, match TOI labels, and train a pilot SVM.

python train_real_tess.py --sectors 1 2 3
Downloads eligible labeled observations (not 2,000 unlabeled stars).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import hashlib
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
import lightkurve

from tess_pipeline import DATA, ROOT, CONFIG, manifest, download_tic, analyze
from svm_classifier import train

QUERY = ("select toi,tid,tfopwg_disp,pl_orbper,pl_tranmid,pl_trandurh,ra,dec "
         "from toi where tfopwg_disp in ('CP','KP','FP') and pl_orbper between 0.6 and 10")


def process(row_data):
    row = argparse.Namespace(**row_data)
    path = download_tic(row.tic_id, row.sector, row.filename)
    cache = DATA / f"{Path(row.filename).stem}_features.json"
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    if saved.get("pipeline") == CONFIG:
        f = saved["features"]
    else:
        # Reuse a blind BLS detection when only flux-vetting features changed.
        old_search = {k: v for k, v in saved.get("pipeline", {}).items() if k not in ["version", "feature_set"]}
        new_search = {k: v for k, v in CONFIG.items() if k not in ["version", "feature_set"]}
        cached_detection = saved.get("features") if old_search == new_search else None
        f = analyze(path, cached_detection=cached_detection)["features"]
        cache.write_text(json.dumps({"pipeline": CONFIG, "features": f}, indent=2))
    # Compare blind detection against catalog, allowing common period aliases.
    ratio = f["period_days"] / row.pl_orbper
    alias = min([0.5, 1.0, 2.0], key=lambda value: abs(ratio / value - 1))
    period_error = abs(ratio / alias - 1)
    p = f["period_days"]
    delta = (f["epoch_btjd"] + 2457000 - row.pl_tranmid + p/2) % p - p/2
    tolerance = max(0.1, 2 * row.pl_trandurh / 24)
    matched = period_error < 0.01 and abs(delta) < min(tolerance, 0.15*p)
    f.update({"toi": row.toi, "disposition": row.tfopwg_disp,
              "label": int(row.tfopwg_disp in ["CP", "KP"]),
              "catalog_period_days": row.pl_orbper, "period_alias": alias,
              "relative_period_error": period_error, "epoch_error_days": abs(delta),
              "catalog_match": bool(matched)})
    return f


def main(sectors):
    DATA.mkdir(exist_ok=True)
    catalog_path = DATA / "toi_catalog.csv"
    if not catalog_path.exists():
        response = requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                                params={"query": QUERY, "format": "csv"}, timeout=(10, 90))
        response.raise_for_status()
        catalog_path.write_text(response.text, encoding="utf-8")
    catalog = pd.read_csv(catalog_path)
    # Only one eligible catalog signal per TIC to avoid ambiguous associations.
    catalog = catalog[~catalog.tid.duplicated(keep=False)].copy()
    catalog = catalog.dropna(subset=["pl_orbper", "pl_tranmid", "pl_trandurh"])
    candidates = pd.concat([
        catalog.merge(manifest(sector), left_on="tid", right_on="tic_id").assign(sector=sector)
        for sector in sectors], ignore_index=True)
    candidates.to_csv(DATA / "training_candidates.csv", index=False)
    print(f"Found {len(candidates)} eligible catalog observations in sectors {sectors}.", flush=True)


    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(process, row._asdict()): row for row in candidates.itertuples()}
        for index, future in enumerate(as_completed(pending), 1):
            row = pending[future]
            try:
                result = future.result()
                rows.append(result)
                print(f"{index}/{len(candidates)} TIC {row.tic_id}: {row.tfopwg_disp}, match={result['catalog_match']}", flush=True)
            except Exception as error:
                errors.append({"tic_id": row.tic_id, "sector": row.sector, "error": str(error)})
                print(f"{index}/{len(candidates)} TIC {row.tic_id}: {error}", flush=True)
            pd.DataFrame(rows).to_csv(DATA / "all_extracted_features.csv", index=False)
            pd.DataFrame(errors, columns=["tic_id", "sector", "error"]).to_csv(DATA / "download_errors.csv", index=False)
    all_rows = pd.DataFrame(rows)
    if all_rows.empty:
        raise RuntimeError("No real observations could be analyzed; see data/download_errors.csv")
    labeled = all_rows[all_rows.catalog_match].copy()
    labeled.to_csv(DATA / "labeled_features.csv", index=False)
    print("Matched labels:", labeled.label.value_counts().to_dict(), flush=True)
    if any(labeled[labeled.label == label].tic_id.nunique() < 10 for label in [0, 1]):
        raise RuntimeError("Need at least 10 matched stars in each class. Try another sector.")
    report = train(labeled, ROOT / "svm_model.joblib")
    report.update({"dataset": "Real TESS SPOC 120-second PDCSAP light curves",
                   "sector": sectors[0], "sectors": sectors, "catalog_targets": len(candidates),
                   "downloaded_analyzed": len(all_rows), "matched_rows": len(labeled),
                   "downloaded_files": int(sum((DATA / "lightcurves" / name).exists() for name in candidates.filename)),
                   "rejected_or_failed_observations": len(errors),
                   "class_counts": {str(k): int(v) for k,v in labeled.label.value_counts().items()},
                   "unique_stars": int(labeled.tic_id.nunique()),
                   "unique_stars_per_class": {str(label): int(labeled[labeled.label == label].tic_id.nunique()) for label in [0, 1]},
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "catalog_source": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                   "catalog_query": QUERY,
                   "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                   "software": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                                "numpy": np.__version__, "lightkurve": lightkurve.__version__},
                   "pipeline": CONFIG,
                   "limitations": "Small, catalog-selected pilot; only recovered, ephemeris-matched signals. Not a blind survey accuracy estimate. CP/KP vs FP, no pure-noise training class."})
    (ROOT / "training_report.json").write_text(json.dumps(report, indent=2))
    artifact = joblib.load(ROOT / "svm_model.joblib")
    artifact.update({"pipeline": CONFIG, "report": report})
    joblib.dump(artifact, ROOT / "svm_model.joblib")
    print("Saved real-data model and training_report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sector", type=int, default=None, help="Use one sector instead of --sectors")
    parser.add_argument("--sectors", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    main([args.sector] if args.sector is not None else list(dict.fromkeys(args.sectors)))
