"""Label-free transit and centroid diagnostics shared by fitting and inference."""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.ndimage import median_filter

FEATURES = [
    "period_days", "duration_days", "depth", "flux_std", "duty_cycle",
    "transit_snr", "odd_even_significance", "secondary_ratio", "shape_ratio",
    "depth_scatter_ratio", "red_noise_ratio", "secondary_significance",
    "observed_transits", "minimum_event_snr", "median_event_snr", "event_fraction",
    "event_depth_chi2", "pink_snr", "bls_snr", "sde", "peak_ratio",
    "noise_mad", "noise_skew", "noise_kurtosis", "variability_ratio",
    "centroid_significance", "centroid_shift", "crowding", "flux_fraction",
    "stellar_radius", "stellar_teff", "stellar_logg", "tess_magnitude",
    "implied_radius_rjup", "duration_density_ratio", "ingress_fraction",
    "positive_excursion_ratio", "local_depth_snr",
]


def mad(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(1.4826 * np.median(np.abs(values - np.median(values)))) if len(values) else np.nan


def phase_time(time, period, epoch):
    return (np.asarray(time) - epoch + period/2) % period - period/2


def vetting_features(time, flux, period, epoch, duration, depth):
    time, flux = np.asarray(time), np.asarray(flux)
    phase = phase_time(time, period, epoch)
    inside = np.abs(phase) < duration/2
    secondary = np.abs(np.abs(phase) - period/2) < duration/2
    outside = ~(inside | secondary)
    if outside.sum() < 20 or inside.sum() < 3:
        raise ValueError("Insufficient in/out-of-transit samples for vetting.")
    baseline = float(np.median(flux[outside]))
    scatter = max(mad(flux[outside]), 1e-12)
    cycle = np.round((time - epoch) / period).astype(int)
    odd, even = inside & (cycle % 2 == 1), inside & (cycle % 2 == 0)
    odd_even = 0.0
    if odd.sum() >= 3 and even.sum() >= 3:
        error = 1.2533 * scatter * np.sqrt(1/odd.sum() + 1/even.sum())
        odd_even = abs(float(np.median(flux[odd]) - np.median(flux[even]))) / error
    secondary_depth = baseline - float(np.median(flux[secondary])) if secondary.any() else 0.0
    core = np.abs(phase) < duration * 0.2
    edge = (np.abs(phase) > duration * 0.3) & inside
    core_depth = max(0.0, baseline - float(np.median(flux[core]))) if core.any() else depth
    edge_depth = max(0.0, baseline - float(np.median(flux[edge]))) if edge.any() else depth
    depths, errors, snrs = [], [], []
    for event in np.unique(cycle[inside]):
        selected = inside & (cycle == event)
        local = (cycle == event) & (np.abs(phase) > duration) & (np.abs(phase) < 3*duration)
        if selected.sum() < 3:
            continue
        local_base = float(np.median(flux[local])) if local.sum() >= 5 else baseline
        error = 1.2533 * scatter / np.sqrt(selected.sum())
        event_depth = local_base - float(np.median(flux[selected]))
        depths.append(event_depth)
        errors.append(error)
        snrs.append(event_depth/error)
    bins = pd.DataFrame({"bin": np.floor((time[outside] - time.min()) / (2/24)).astype(int),
                         "flux": flux[outside]}).groupby("bin").flux.agg(["mean", "count"])
    bins = bins[bins["count"] >= 5]
    red_noise = float(np.std(bins["mean"])) * np.sqrt(float(np.median(bins["count"]))) / scatter if len(bins) > 2 else 1.0
    # Profile fractions provide a bounded shape diagnostic, unlike an unstable
    # center/edge ratio when noisy ingress samples happen to have zero depth.
    folded_bins = np.linspace(-duration, duration, 31)
    profile = []
    centers = []
    for left, right in zip(folded_bins[:-1], folded_bins[1:]):
        selected = (phase >= left) & (phase < right)
        if selected.sum() >= 2:
            centers.append((left + right)/2)
            profile.append(baseline - np.median(flux[selected]))
    widths = [abs(center) for center, value in zip(centers, profile) if value > 0.5*max(core_depth, 1e-12)]
    half_width = max(widths, default=duration/4)
    depth_error = scatter / np.sqrt(max(inside.sum(), 1))
    depths, errors = np.asarray(depths), np.asarray(errors)
    snr = depth / max(depth_error, 1e-12)
    values = {
        "transit_snr": snr,
        "odd_even_significance": odd_even,
        "secondary_ratio": max(0, secondary_depth) / max(depth, depth_error),
        "secondary_significance": secondary_depth*np.sqrt(max(secondary.sum(), 1))/scatter,
        "shape_ratio": core_depth / max(edge_depth, scatter/np.sqrt(max(edge.sum(), 1))),
        "depth_scatter_ratio": float(np.std(depths))/max(depth, 1e-12) if len(depths) > 1 else 1.0,
        "red_noise_ratio": red_noise,
        "observed_transits": len(depths),
        "minimum_event_snr": min(snrs, default=0),
        "median_event_snr": float(np.median(snrs)) if snrs else 0,
        "event_fraction": float(np.mean(np.asarray(snrs) > 2)) if snrs else 0,
        "event_depth_chi2": float(np.mean(((depths-np.median(depths))/errors)**2)) if len(depths) > 1 else 1,
        "pink_snr": snr/max(red_noise, 1),
        "noise_mad": scatter,
        "noise_skew": float(skew(flux[outside])),
        "noise_kurtosis": float(kurtosis(flux[outside])),
        "variability_ratio": float(np.std(flux[outside]))/scatter,
        "ingress_fraction": float(np.clip(1-2*half_width/duration, 0, 1)),
        "positive_excursion_ratio": max(0, float(np.quantile(flux[outside], .995))-baseline)/max(depth, depth_error),
        "local_depth_snr": float(np.median(snrs))*np.sqrt(len(snrs)) if snrs else 0,
    }
    for key in ["transit_snr", "odd_even_significance", "secondary_ratio", "shape_ratio",
                "depth_scatter_ratio", "red_noise_ratio"]:
        values[key] = max(float(values[key]), 1e-6)
    return values


def centroid_features(time, centroids, phase, duration):
    inside = np.abs(phase) < duration/2
    nearby = (np.abs(phase) > duration) & (np.abs(phase) < 3*duration)
    shifts, significances = [], []
    for values in centroids:
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 100 or (valid & inside).sum() < 3 or (valid & nearby).sum() < 10:
            continue
        # A broad running median removes slow pointing drift, while preserving
        # shifts on transit timescales. POS_CORR was already removed by the reader.
        filled = np.interp(time, time[valid], values[valid])
        cadence = np.median(np.diff(time))
        size = min(1501, max(31, int(1/cadence)//2*2+1))
        trend = median_filter(filled[::5], size=max(7, size//5), mode="nearest")
        residual = values - np.interp(time, time[::5], trend)
        noise = max(mad(residual[valid & nearby]), 1e-12)
        ni, no = (valid & inside).sum(), (valid & nearby).sum()
        shift = float(np.median(residual[valid & inside]) - np.median(residual[valid & nearby]))
        shifts.append(shift)
        significances.append(shift/(1.2533*noise*np.sqrt(1/ni+1/no)))
    return {"centroid_shift": float(np.linalg.norm(shifts)) if shifts else np.nan,
            "centroid_significance": float(np.linalg.norm(significances)) if significances else np.nan}


def stellar_features(headers, period, duration, depth):
    def value(key, lower=None, upper=None):
        try:
            result = float(headers.get(key, np.nan))
        except (ValueError, TypeError):
            return np.nan
        return result if np.isfinite(result) and (lower is None or result > lower) and (upper is None or result < upper) else np.nan
    radius = value("RADIUS", 0, 100)
    logg = value("LOGG", 0, 6)
    ratio = np.nan
    if np.isfinite(radius) and np.isfinite(logg):
        gravity = 10**logg / 100  # cgs to SI
        r_m = radius * 695700000
        a_over_r = (gravity * (period*86400)**2 / (4*np.pi**2*r_m))**(1/3)
        central_duration = period/np.pi*np.arcsin(min(.99, 1/max(a_over_r, 1)))
        ratio = duration/central_duration
    return {"stellar_radius": radius, "stellar_teff": value("TEFF", 2000, 15000),
            "stellar_logg": logg, "tess_magnitude": value("TESSMAG", 0, 25),
            "crowding": value("CROWDSAP", 0, 1.001), "flux_fraction": value("FLFRCSAP", 0, 1.001),
            "implied_radius_rjup": radius*9.731*np.sqrt(max(depth, 0)),
            "duration_density_ratio": ratio}
