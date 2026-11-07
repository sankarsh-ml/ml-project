"""Freeze train/test TIC assignments before downloading or fitting new data."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from tess_pipeline import ROOT, DATA, manifest


def prepare(sectors=range(1, 56), test_per_class=70, train_per_class=300):
    folder = DATA / "v4"
    folder.mkdir(exist_ok=True)
    if (folder / "split_plan.json").exists():
        print("Using immutable existing split plan.")
        return pd.read_csv(folder / "observation_plan.csv", dtype={"tic_id": str})
    original = ROOT / "history/2026-11-07_v3"
    previous = pd.read_csv(original / "data/training_candidates.csv")
    previous_ids = set(previous.tic_id.astype(str))
    old_report = json.loads((original / "training_report.json").read_text())
    quarantined = set(old_report["test_tic_ids"])
    catalog_path = DATA / "toi_catalog_full.csv"
    catalog = pd.read_csv(catalog_path)
    eligible = catalog[catalog.tfopwg_disp.isin(["CP", "KP", "FP"]) & catalog.pl_orbper.between(.6, 10)].copy()
    eligible = eligible[~eligible.tid.duplicated(keep=False)].dropna(subset=["pl_tranmid", "pl_trandurh"])
    eligible["tic_id"] = eligible.tid.astype("int64").astype(str)
    eligible["label"] = eligible.tfopwg_disp.isin(["CP", "KP"]).astype(int)
    matches = []
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(manifest, sector): sector for sector in sectors}
        for future in as_completed(futures):
            sector = futures[future]
            table = future.result()
            table["tic_id"] = table.tic_id.astype(str)
            found = eligible.merge(table, on="tic_id").assign(sector=sector)
            matches.append(found)
    available = pd.concat(matches, ignore_index=True).sort_values(["tic_id", "sector"])
    available.to_csv(folder / "available_observations.csv", index=False)
    stars = available.drop_duplicates("tic_id")
    rng = np.random.default_rng(20261107)
    test_ids, train_ids = [], []
    for label in [0, 1]:
        pool = stars[(stars.label == label) & ~stars.tic_id.isin(quarantined)].copy()
        novel = pool[~pool.tic_id.isin(previous_ids)].tic_id.to_numpy()
        rng.shuffle(novel)
        # Reserve new stars before any new light-curve extraction. Class labels
        # are used solely to stratify assignment, not to inspect outcomes.
        count = min(test_per_class, max(20, len(novel)//4))
        chosen_test = list(novel[:count])
        test_ids.extend(chosen_test)
        train_pool = pool[~pool.tic_id.isin(chosen_test)].tic_id.to_numpy()
        rng.shuffle(train_pool)
        train_ids.extend(list(train_pool[:train_per_class]))
    selected = stars[stars.tic_id.isin(train_ids+test_ids)].copy()
    selected["partition"] = np.where(selected.tic_id.isin(test_ids), "independent_test", "training")
    selected = selected.sort_values(["partition", "label", "tic_id"]).reset_index(drop=True)
    selected.to_csv(folder / "observation_plan.csv", index=False)
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": 20261107,
            "sectors": list(sectors), "train_tic_ids": sorted(train_ids),
            "test_tic_ids": sorted(test_ids), "legacy_seen_tic_ids": sorted(previous_ids),
            "quarantined_development_tic_ids": sorted(quarantined),
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "protocol": "Class-stratified TIC split fixed before new feature extraction. Legacy development test IDs excluded from all fitting/CV. Independent test excludes every previous target. Evaluate frozen model once; do not reselect based on test metrics."}
    (folder / "split_plan.json").write_text(json.dumps(plan, indent=2))
    print(selected.groupby(["partition", "label"]).size().to_string())
    return selected


if __name__ == "__main__":
    prepare()
