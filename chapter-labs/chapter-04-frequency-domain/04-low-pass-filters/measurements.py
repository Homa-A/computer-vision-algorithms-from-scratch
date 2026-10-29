"""Explicit full-precision metrics used by the deterministic experiments."""
import numpy as np


def step_metrics(profile, edge_index, *, half_window=48):
    """Metrics for a rising unit step at samples edge_index-1/edge_index.

    Use only the stated central window, excluding the periodic boundary step.
    Overshoot/undershoot are positive excursions beyond [0,1]. The 10–90%
    width uses linearly interpolated rising crossings nearest edge_index-.5.
    """
    raw = np.asarray(profile)
    if raw.ndim != 1 or np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError("Profile must be a finite real vector.")
    if (not isinstance(edge_index, (int, np.integer)) or isinstance(edge_index, (bool, np.bool_))
            or not isinstance(half_window, (int, np.integer)) or isinstance(half_window, (bool, np.bool_))
            or half_window < 2 or edge_index-half_window < 0 or edge_index+half_window > raw.size):
        raise ValueError("Edge and half-window must define an interior interval.")
    start, stop = edge_index-half_window, edge_index+half_window
    values = raw[start:stop].astype(float)
    positions = np.arange(start, stop, dtype=float)

    def crossing(level):
        index = np.flatnonzero((values[:-1] <= level) & (values[1:] > level))
        if index.size == 0:
            raise ValueError("Profile has no required rising level crossing.")
        candidates = positions[index]+(level-values[index])/(values[index+1]-values[index])
        return float(candidates[np.argmin(abs(candidates-(edge_index-.5)))])

    return dict(overshoot=max(0., float(values.max()-1)), undershoot=max(0., float(-values.min())),
                width_10_90_pixels=crossing(.9)-crossing(.1),
                maximum_slope=float(np.max(np.abs(np.diff(values)))), half_window_pixels=int(half_window))


def image_metrics(reference, filtered, high_frequency_radius=.18):
    """MSE, periodic gradient energy and unnormalized spectral energy fraction."""
    from low_pass import radial_distance

    reference, filtered = np.asarray(reference), np.asarray(filtered)
    if reference.shape != filtered.shape or reference.ndim != 2:
        raise ValueError("Metric arrays must have the same 2D shape.")
    if not np.isfinite(reference).all() or not np.isfinite(filtered).all():
        raise ValueError("Metric arrays must be finite.")
    power = abs(np.fft.fft2(filtered))**2
    total = float(power.sum())
    gradient = sum(np.mean(abs(np.roll(filtered, -1, axis=axis)-filtered)**2) for axis in (0, 1))
    return dict(mse_to_clean=float(np.mean(abs(filtered-reference)**2)),
                periodic_gradient_energy=float(gradient),
                high_frequency_energy_fraction=float(power[radial_distance(filtered.shape) > high_frequency_radius].sum()/total) if total else 0.)
