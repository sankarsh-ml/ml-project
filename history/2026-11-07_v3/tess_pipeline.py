"""Shared, blind TESS search for training and the desktop application."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests
import lightkurve as lk
import astropy.units as u

from svm_classifier import bls_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CONFIG = {"version": 3, "feature_set": "transit_vetting_v1", "cadence_seconds": 120, "window_length": 1501,
          "min_period": 0.6, "max_period": 10.0, "period_grid": 6000,
          "search_bin_minutes": 10, "refinement_points": 400,
          "durations_days": [0.025, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25, 0.32]}


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
        temporary = path.with_suffix(".part")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return path


def vetting_features(time, flux, period, epoch, duration, depth):
    """Flux-only diagnostics; no catalog data enters the feature calculation."""
    phase = (time - epoch + period/2) % period - period/2
    inside = np.abs(phase) < duration/2
    secondary = np.abs(np.abs(phase) - period/2) < duration/2
    outside = ~(inside | secondary)
    if outside.sum() < 20:
        raise ValueError("Insufficient out-of-transit samples for vetting; the BLS duty cycle is too large.")
    baseline = float(np.median(flux[outside]))
    scatter = max(1.4826 * float(np.median(np.abs(flux[outside] - baseline))), 1e-12)
    cycle = np.round((time - epoch) / period).astype(int)
    odd = inside & (cycle % 2 == 1)
    even = inside & (cycle % 2 == 0)
    odd_even = 0.0
    if odd.sum() >= 3 and even.sum() >= 3:
        error = 1.2533 * scatter * np.sqrt(1/odd.sum() + 1/even.sum())
        odd_even = abs(float(np.median(flux[odd]) - np.median(flux[even]))) / error
    secondary_depth = max(0.0, baseline - float(np.median(flux[secondary]))) if secondary.any() else 0.0
    core = np.abs(phase) < duration * 0.2
    edge = (np.abs(phase) > duration * 0.3) & inside
    core_depth = max(0.0, baseline - float(np.median(flux[core]))) if core.any() else depth
    edge_depth = max(0.0, baseline - float(np.median(flux[edge]))) if edge.any() else depth
    event_depths = [baseline - float(np.median(flux[inside & (cycle == event)]))
                    for event in np.unique(cycle[inside])
                    if np.sum(inside & (cycle == event)) >= 3]
    bin_ids = np.floor((time[outside] - time.min()) / (2/24)).astype(int)
    bins = pd.DataFrame({"bin": bin_ids, "flux": flux[outside]}).groupby("bin").flux.agg(["mean", "count"])
    bins = bins[bins["count"] >= 10]
    red_noise = float(np.std(bins["mean"])) * np.sqrt(float(np.median(bins["count"]))) / scatter if len(bins) > 2 else 1.0
    values = {"transit_snr": depth * np.sqrt(inside.sum()) / scatter,
              "odd_even_significance": odd_even,
              "secondary_ratio": secondary_depth / max(depth, scatter / np.sqrt(max(inside.sum(), 1))),
              "shape_ratio": core_depth / max(edge_depth, scatter / np.sqrt(max(edge.sum(), 1))),
              "depth_scatter_ratio": float(np.std(event_depths)) / max(depth, 1e-12) if len(event_depths) > 1 else 1.0,
              "red_noise_ratio": red_noise}
    # Log transforms need a finite positive floor for zero-valued diagnostics.
    return {name: max(float(value), 1e-6) for name, value in values.items()}


def analyze(path, progress=lambda message: None, cached_detection=None):
    progress("Reading PDCSAP flux and quality flags...")
    raw = lk.read(str(path), flux_column="pdcsap_flux", quality_bitmask="default")
    if raw.meta.get("MISSION", raw.meta.get("TELESCOP", "")) != "TESS":
        raise ValueError("Choose a TESS SPOC light curve FITS file.")
    raw = raw.remove_nans().normalize()
    cadence = float(np.nanmedian(np.diff(raw.time.value))) * 86400
    if not 110 <= cadence <= 130:
        raise ValueError("This model requires TESS two-minute cadence data.")
    clean = raw.remove_outliers(sigma_upper=4, sigma_lower=float("inf"))
    if len(clean) < 1000:
        raise ValueError("Too few usable samples for this search.")
    flat = clean.flatten(window_length=CONFIG["window_length"]).remove_nans()
    baseline = float(np.ptp(flat.time.value))
    upper = min(CONFIG["max_period"], baseline / 2)
    if upper <= CONFIG["min_period"]:
        raise ValueError("Observation baseline is too short.")
    if cached_detection is None:
        periods = np.geomspace(CONFIG["min_period"], upper, CONFIG["period_grid"])
        progress("Searching repeating transit signals with BLS...")
        search_lc = flat.bin(time_bin_size=CONFIG["search_bin_minutes"] * u.min).remove_nans()
        coarse = search_lc.to_periodogram(method="bls", period=periods,
                                          duration=CONFIG["durations_days"])
        best = float(coarse.period_at_max_power.value)
        fine_periods = np.linspace(max(CONFIG["min_period"], best * 0.997),
                                   min(upper, best * 1.003), CONFIG["refinement_points"])
        progress("Refining the strongest peak on the two-minute data...")
        bls = flat.to_periodogram(method="bls", period=fine_periods,
                                  duration=CONFIG["durations_days"])
        features = bls_features(flat, bls)
        t0 = float(bls.transit_time_at_max_power.value)
        period_grid, power = np.asarray(coarse.period.value), np.asarray(coarse.power.value)
    else:
        features = {name: cached_detection[name] for name in ["period_days", "duration_days", "depth", "flux_std", "duty_cycle"]}
        t0 = cached_detection["epoch_btjd"]
        period_grid, power = np.array([]), np.array([])
    p = features["period_days"]
    duration = features["duration_days"]
    t = np.asarray(flat.time.value)
    phase = (t - t0 + p/2) % p - p/2
    inside = np.abs(phase) < duration / 2
    events = np.unique(np.round((t[inside] - t0) / p).astype(int))
    features.update({"tic_id": str(raw.meta.get("TICID", "unknown")),
                     "sector": int(raw.meta.get("SECTOR", 0)),
                     "epoch_btjd": t0, "observed_transits": len(events),
                     "source_file": Path(path).name})
    features.update(vetting_features(t, np.asarray(flat.flux.value), p, t0, duration, features["depth"]))
    if len(events) < 2:
        raise ValueError("Fewer than two sampled transit windows; no reliable periodic classification.")
    return {"features": features, "time": t, "flux": np.asarray(flat.flux.value),
            "phase": phase, "period_grid": period_grid, "power": power}
