"""Fresh test reservation after correcting long-baseline epoch association."""
from datetime import datetime, timezone
import json
import shutil
import numpy as np
import pandas as pd
from experiment import ROOT, RUN_DIR
from training_workflow import match_ephemeris


def main():
    RUN_DIR.mkdir(exist_ok=True)
    if (RUN_DIR / "split_plan.json").exists():
        print("Immutable follow-up split already exists.")
        return
    current_report = ROOT / "training_report.json"
    if current_report.exists() and json.loads(current_report.read_text()).get("protocol_version") == "v5":
        raise RuntimeError("v5 was already evaluated. Choose a new protocol and additional untouched targets; do not recreate the opened split.")
    previous = ROOT / "data/v4"
    old_plan = json.loads((previous / "split_plan.json").read_text())
    observation = pd.read_csv(previous / "observation_plan.csv", dtype={"tic_id": str})
    available = pd.read_csv(previous / "available_observations.csv", dtype={"tic_id": str})
    used = set(old_plan["legacy_seen_tic_ids"]) | set(observation.tic_id)
    unused = available[~available.tic_id.isin(used)].sort_values(["tic_id", "sector"]).drop_duplicates("tic_id")
    rng = np.random.default_rng(20261108)
    reserved = []
    for label in [0, 1]:
        pool = unused[unused.label == label]
        ids = pool.tic_id.to_numpy().copy()
        rng.shuffle(ids)
        reserved.append(pool[pool.tic_id.isin(ids[:100])])
    test = pd.concat(reserved).assign(partition="independent_test")
    training = observation[observation.partition == "training"]
    pd.concat([training, test], ignore_index=True).sort_values(["partition", "label", "tic_id"]).to_csv(RUN_DIR / "observation_plan.csv", index=False)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": 20261108,
        "sectors": old_plan["sectors"], "train_tic_ids": sorted(training.tic_id), "test_tic_ids": sorted(test.tic_id),
        "legacy_seen_tic_ids": sorted(used),
        "quarantined_development_tic_ids": sorted(set(old_plan["quarantined_development_tic_ids"]) | set(old_plan["test_tic_ids"])),
        "catalog_sha256": old_plan["catalog_sha256"],
        "protocol": "v5 projects catalog epochs using catalog P, not noisy fitted P. Reuse only original 600 training targets; quarantine all v4 test IDs. Reserve 100 new TICs per class before extraction, freeze the selected model, then evaluate once."}
    (RUN_DIR / "split_plan.json").write_text(json.dumps(metadata, indent=2))
    extracted = pd.read_csv(previous / "training_extracted.csv", dtype={"tic_id": str})
    catalog = {row["tic_id"]: row for row in training.to_dict("records")}
    corrected = []
    for row in extracted.to_dict("records"):
        row.update(match_ephemeris(row, catalog[row["tic_id"]]))
        corrected.append(row)
    frame = pd.DataFrame(corrected).sort_values("tic_id")
    frame.to_csv(RUN_DIR / "training_extracted.csv", index=False)
    accepted = frame[frame.catalog_match & frame.detection_accepted]
    accepted.to_csv(RUN_DIR / "training_labeled.csv", index=False)
    shutil.copy2(previous / "training_errors.csv", RUN_DIR / "training_errors.csv")
    print("Corrected training labels:", accepted.label.value_counts().to_dict())
    print("Fresh test reservation:", test.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
