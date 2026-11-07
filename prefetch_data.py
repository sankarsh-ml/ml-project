"""Download assigned public FITS files without extracting test features."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tess_pipeline import DATA, download_tic
from experiment import RUN_DIR


def main():
    plan = pd.read_csv(RUN_DIR / "observation_plan.csv", dtype={"tic_id": str})
    rows = plan.to_dict("records")
    with ThreadPoolExecutor(8) as pool:
        futures = {pool.submit(download_tic, row["tic_id"], row["sector"], row["filename"]): row for row in rows}
        failures = []
        for count, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append({"tic_id": row["tic_id"], "error": str(error)})
            if count % 20 == 0:
                print(f"Downloaded/checked {count}/{len(rows)}, failures {len(failures)}", flush=True)
    pd.DataFrame(failures, columns=["tic_id", "error"]).to_csv(RUN_DIR / "prefetch_errors.csv", index=False)


if __name__ == "__main__":
    main()
