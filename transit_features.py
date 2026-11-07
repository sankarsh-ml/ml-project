"""Brightness-only transit measurements used by the Random Forest."""
import numpy as np
import pandas as pd

FEATURES = ['period_days', 'duration_days', 'depth', 'flux_std', 'duty_cycle', 'transit_snr', 'odd_even_significance', 'secondary_ratio', 'shape_ratio', 'depth_scatter_ratio', 'red_noise_ratio', 'observed_transits', 'pink_snr', 'bls_snr', 'sde', 'event_fraction']


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
    depths, snrs = [], []
    for event in np.unique(cycle[inside]):
        selected = inside & (cycle == event)
        local = (cycle == event) & (np.abs(phase) > duration) & (np.abs(phase) < 3*duration)
        if selected.sum() < 3:
            continue
        local_base = float(np.median(flux[local])) if local.sum() >= 5 else baseline
        error = 1.2533 * scatter / np.sqrt(selected.sum())
        event_depth = local_base - float(np.median(flux[selected]))
        depths.append(event_depth)
        snrs.append(event_depth/error)
    bins = pd.DataFrame({"bin": np.floor((time[outside] - time.min()) / (2/24)).astype(int),
                         "flux": flux[outside]}).groupby("bin").flux.agg(["mean", "count"])
    bins = bins[bins["count"] >= 5]
    red_noise = float(np.std(bins["mean"])) * np.sqrt(float(np.median(bins["count"]))) / scatter if len(bins) > 2 else 1.0
    depth_error = scatter / np.sqrt(max(inside.sum(), 1))
    depths = np.asarray(depths)
    snr = depth / max(depth_error, 1e-12)
    values = {
        "transit_snr": snr,
        "odd_even_significance": odd_even,
        "secondary_ratio": max(0, secondary_depth) / max(depth, depth_error),
        "shape_ratio": core_depth / max(edge_depth, scatter/np.sqrt(max(edge.sum(), 1))),
        "depth_scatter_ratio": float(np.std(depths))/max(depth, 1e-12) if len(depths) > 1 else 1.0,
        "red_noise_ratio": red_noise,
        "observed_transits": len(depths),
        "event_fraction": float(np.mean(np.asarray(snrs) > 2)) if snrs else 0,
        "pink_snr": snr/max(red_noise, 1),

    }
    for key in ["transit_snr", "odd_even_significance", "secondary_ratio", "shape_ratio",
                "depth_scatter_ratio", "red_noise_ratio"]:
        values[key] = max(float(values[key]), 1e-6)
    return values

