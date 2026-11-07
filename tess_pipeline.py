"""Shared, blind TESS search for training and the desktop application."""
from pathlib import Path
import re
import uuid

import pandas as pd
import requests

from search_pipeline import CONFIG, analyze, vetting_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"



def manifest(sector=1):
    sector = int(sector)
    if sector < 1:
        raise ValueError("Sector must be a positive integer.")
    DATA.mkdir(exist_ok=True)
    path = DATA / f"sector{sector}_download.sh"
    if not path.exists():
        response = requests.get(
            f"https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{sector}_lc.sh",
            timeout=(10, 90))
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    # Parse filenames only. Never execute a downloaded shell script.
    matches = re.findall(r"(tess\d+-s\d+-(\d{16})-\d+-s_lc\.fits)", path.read_text())
    result = pd.DataFrame(sorted(set(matches)), columns=["filename", "tic_id"])
    if result.empty:
        raise ValueError("No two-minute light curve filenames found for this sector.")
    result["tic_id"] = result.tic_id.astype("int64")
    return result.drop_duplicates("tic_id")


def download_tic(tic_id, sector=1, filename=None):
    tic = int(str(tic_id).upper().replace("TIC", "").strip())
    if tic <= 0:
        raise ValueError("Enter a positive numeric TIC ID.")
    if filename is None:
        cached = sorted((DATA / "lightcurves").glob(f"tess*-s{int(sector):04d}-{tic:016d}-*_lc.fits"))
        if cached:
            return cached[-1]
        found = manifest(sector)
        found = found[found.tic_id == tic]
        if found.empty:
            raise ValueError(f"TIC {tic} has no two-minute SPOC light curve in sector {sector}. Try another sector or a local FITS file.")
        filename = found.iloc[0].filename
    folder = DATA / "lightcurves"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    if not path.exists():
        digits = f"{tic:016d}"
        pieces = "/".join(digits[i:i+4] for i in range(0, 16, 4))
        relative = f"tid/s{int(sector):04d}/{pieces}/{filename}"
        # Official public TESS mirror; fall back to the archive if unavailable.
        urls = [f"https://stpubdata.s3.amazonaws.com/tess/public/{relative}",
                f"https://archive.stsci.edu/missions/tess/{relative}"]
        for index, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=(10, 90))
                response.raise_for_status()
                break
            except requests.RequestException:
                if index == len(urls) - 1:
                    raise
        if not response.content.startswith(b"SIMPLE"):
            raise ValueError("Archive returned something other than a FITS light curve.")
        temporary = folder / f"{filename}.{uuid.uuid4().hex}.part"
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return path

