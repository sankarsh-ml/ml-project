"""Blind, transit-preserving TESS preprocessing and multi-peak BLS search."""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter, find_peaks

from transit_features import mad, phase_time, vetting_features, centroid_features, stellar_features

CONFIG = {"version": 4, "feature_set": "flux_centroid_stellar_v1", "cadence_seconds": 120,
          "window_days": 2.0, "min_period": .6, "max_period": 10.0,
          "frequency_grid": 20000, "search_bin_minutes": 8, "peak_count": 4,
          "max_duty_cycle": .15, "mask_width": 1.7,
          "durations_days": [.025, .04, .06, .09, .13, .18, .25, .35]}


def detection_accepted(features):
    return bool(features["bls_snr"] >= 6 and features["pink_snr"] >= 3 and
                features["observed_transits"] >= 2 and features["event_fraction"] >= .4)


def read_curve(path):
    with fits.open(path, memmap=False) as hdus:
        header = dict(hdus[0].header)
        header.update(dict(hdus[1].header))
        if header.get("TELESCOP") != "TESS":
            raise ValueError("Choose a TESS SPOC light curve FITS file.")
        table = hdus[1].data
        # Official Lightkurve default TESS quality bit mask (17087).
        quality = (np.asarray(table["QUALITY"], dtype=np.int64) & 17087) == 0
        time = np.asarray(table["TIME"], dtype=float)
        flux = np.asarray(table["PDCSAP_FLUX"], dtype=float)
        error = np.asarray(table["PDCSAP_FLUX_ERR"], dtype=float)
        valid = quality & np.isfinite(time) & np.isfinite(flux) & (flux > 0)
        if valid.sum() < 1000:
            raise ValueError("Too few usable quality-filtered samples.")
        cadence = float(np.nanmedian(np.diff(time[valid])))*86400
        if not 110 <= cadence <= 130:
            raise ValueError("This model requires TESS two-minute cadence data.")
        centroids = []
        for axis in [1, 2]:
            name, correction = f"MOM_CENTR{axis}", f"POS_CORR{axis}"
            values = np.asarray(table[name], dtype=float) if name in table.names else np.full(len(time), np.nan)
            if correction in table.names:
                values = values - np.asarray(table[correction], dtype=float)
            centroids.append(values[valid])
        scale = np.median(flux[valid])
        return time[valid], flux[valid]/scale, error[valid]/scale, centroids, header


def detrend(time, flux, exclude=None):
    """Iteratively exclude dips/spikes from a broad Savitzky-Golay trend."""
    excluded = np.zeros(len(time), dtype=bool) if exclude is None else np.asarray(exclude)
    trend = np.ones(len(time))
    cuts = np.r_[0, np.where(np.diff(time) > .3)[0]+1, len(time)]
    cadence = np.median(np.diff(time))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        t, y = time[lo:hi], flux[lo:hi]
        allowed = ~excluded[lo:hi]
        if allowed.sum() < 5:
            trend[lo:hi] = np.median(y)
            continue
        width = min(int(CONFIG["window_days"]/cadence)//2*2+1, (len(y)-1)//2*2+1)
        if width < 7:
            trend[lo:hi] = np.median(y[allowed])
            continue
        fit = allowed.copy()
        for _ in range(3):
            filled = np.interp(t, t[fit], y[fit])
            smooth = savgol_filter(filled, width, 2, mode="interp")
            residual = y-smooth
            sigma = max(mad(residual[allowed]), 1e-8)
            new_fit = allowed & (residual > -2.5*sigma) & (residual < 5*sigma)
            if new_fit.sum() < 5:
                break
            fit = new_fit
        trend[lo:hi] = smooth
    if not np.isfinite(trend).all() or np.any(trend <= 0):
        raise ValueError("Invalid brightness trend; inspect this observation.")
    return flux/trend, trend


def bin_curve(time, flux, error, minutes):
    ids = np.floor((time-time.min())/(minutes/1440)).astype(int)
    _, inverse = np.unique(ids, return_inverse=True)
    count = np.bincount(inverse)
    weights = 1/np.maximum(error, 1e-10)**2
    total = np.bincount(inverse, weights=weights)
    return (np.bincount(inverse, weights=time)/count,
            np.bincount(inverse, weights=weights*flux)/total,
            np.sqrt(1/total))


def search(time, flux, error, progress):
    upper = min(CONFIG["max_period"], float(np.ptp(time))/2)
    if upper <= CONFIG["min_period"]:
        raise ValueError("Observation baseline is too short.")
    bt, by, be = bin_curve(time, flux, error, CONFIG["search_bin_minutes"])
    model = BoxLeastSquares(bt, by, be)
    periods = 1/np.linspace(1/upper, 1/CONFIG["min_period"], CONFIG["frequency_grid"])
    result = model.power(periods, CONFIG["durations_days"], objective="snr", oversample=5)
    valid = result.duration/result.period <= CONFIG["max_duty_cycle"]
    scores = np.where(valid, result.power, -np.inf)
    peaks, _ = find_peaks(scores)
    peaks = np.r_[peaks, np.argmax(scores)]
    chosen = []
    for index in peaks[np.argsort(scores[peaks])[::-1]]:
        if not np.isfinite(scores[index]):
            continue
        p = float(result.period[index])
        if all(abs(p/old-1) > .01 for old in chosen):
            chosen.append(p)
        if len(chosen) >= CONFIG["peak_count"]:
            break
    if not chosen:
        raise ValueError("No BLS peak with a physically plausible duty cycle.")
    trials = list(chosen)
    for p in chosen[:2]:
        for alias in [p/2, p*2]:
            if CONFIG["min_period"] <= alias <= upper and all(abs(alias/x-1) > .005 for x in trials):
                trials.append(alias)
    progress("Refining competing BLS peaks and period harmonics...")
    refined = []
    for p in trials:
        grid = np.linspace(max(CONFIG["min_period"], p*.994), min(upper, p*1.006), 100)
        durations = np.asarray(CONFIG["durations_days"])
        durations = durations[durations <= p*CONFIG["max_duty_cycle"]]
        if not len(durations):
            continue
        power = model.power(grid, durations, objective="snr", oversample=10)
        i = int(np.argmax(power.power))
        refined.append({"period": float(power.period[i]), "epoch": float(power.transit_time[i]),
                        "duration": float(power.duration[i]), "snr": float(power.depth_snr[i])})
    refined.sort(key=lambda row: row["snr"], reverse=True)
    return refined, result


def analyze(path, progress=lambda message: None, cached_detection=None):
    # cached_detection is accepted for API compatibility but never reuses v3 fits.
    progress("Reading quality-filtered PDCSAP flux and centroid motion...")
    time, raw, error, centroids, header = read_curve(path)
    flat, trend = detrend(time, raw)
    noise = max(mad(np.diff(flat))/np.sqrt(2), 1e-8)
    error = np.where(np.isfinite(error) & (error > 0), error, noise)
    # Protect downward dips; remove isolated upper excursions relative to trend.
    use = (flat-1) < 8*max(mad(flat), noise)
    time, raw, error = time[use], raw[use], error[use]
    centroids = [value[use] for value in centroids]
    flat = flat[use]
    error = error/np.maximum(trend[use], 1e-8)
    progress("Searching periodic dips on a uniform-frequency BLS grid...")
    candidates, coarse = search(time, flat, error, progress)
    best = candidates[0]
    phase = phase_time(time, best["period"], best["epoch"])
    mask = ((np.abs(phase) < best["duration"]*CONFIG["mask_width"]/2) |
            (np.abs(np.abs(phase)-best["period"]/2) < best["duration"]*CONFIG["mask_width"]/2))
    progress("Re-detrending with transit windows masked...")
    flat, trend = detrend(time, raw, mask)
    corrected_error = error  # trend is close to one; weights remain per-cadence.
    upper = min(CONFIG["max_period"], np.ptp(time)/2)
    grid = np.linspace(max(CONFIG["min_period"], best["period"]*.998),
                       min(upper, best["period"]*1.002), 180)
    durations = np.unique(np.clip(np.asarray(CONFIG["durations_days"] + [best["duration"]]), .015,
                                   min(.4, best["period"]*CONFIG["max_duty_cycle"])))
    model = BoxLeastSquares(time, flat, corrected_error)
    refined = model.power(grid, durations, objective="snr", oversample=10)
    i = int(np.argmax(refined.power))
    period, duration, epoch, depth = (float(refined[key][i]) for key in ["period", "duration", "transit_time", "depth"])
    phase = phase_time(time, period, epoch)
    features = {"period_days": period, "duration_days": duration, "depth": depth,
                "flux_std": float(np.std(flat)), "duty_cycle": duration/period,
                "bls_snr": float(refined.depth_snr[i]),
                "sde": (best["snr"]-float(np.median(coarse.power)))/max(mad(coarse.power), 1e-8),
                "peak_ratio": best["snr"]/max(candidates[1]["snr"], 1e-8) if len(candidates) > 1 else 1.0}
    features.update(vetting_features(time, flat, period, epoch, duration, depth))
    features.update(centroid_features(time, centroids, phase, duration))
    features.update(stellar_features(header, period, duration, depth))
    if features["observed_transits"] < 2:
        raise ValueError("Fewer than two sampled transit windows.")
    features.update({"tic_id": str(header["TICID"]), "sector": int(header["SECTOR"]),
                     "epoch_btjd": epoch, "source_file": Path(path).name})
    order = np.argsort(coarse.period)
    return {"features": features, "time": time, "flux": flat, "phase": phase,
            "period_grid": np.asarray(coarse.period)[order], "power": np.asarray(coarse.power)[order],
            "candidate_peaks": candidates, "trend": trend}
