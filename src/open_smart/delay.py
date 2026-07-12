from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class DelayEstimate:
    samples: int
    seconds: float
    confidence: float


def estimate_delay(reference: np.ndarray, measurement: np.ndarray, sample_rate: int, max_delay_ms: float = 200.0) -> DelayEstimate:
    """Estimate measurement delay relative to reference using normalized cross-correlation."""

    ref = np.asarray(reference, dtype=np.float64)
    meas = np.asarray(measurement, dtype=np.float64)
    if ref.ndim != 1 or meas.ndim != 1:
        raise ValueError("reference and measurement must be 1-D arrays")
    count = min(ref.size, meas.size)
    if count < 2:
        return DelayEstimate(samples=0, seconds=0.0, confidence=0.0)

    ref = ref[-count:] - np.mean(ref[-count:])
    meas = meas[-count:] - np.mean(meas[-count:])
    ref_energy = np.linalg.norm(ref)
    meas_energy = np.linalg.norm(meas)
    if ref_energy == 0.0 or meas_energy == 0.0:
        return DelayEstimate(samples=0, seconds=0.0, confidence=0.0)

    corr = signal.correlate(meas, ref, mode="full", method="fft")
    lags = signal.correlation_lags(meas.size, ref.size, mode="full")
    max_lag = int(sample_rate * max_delay_ms / 1000.0)
    mask = np.abs(lags) <= max_lag
    if not np.any(mask):
        return DelayEstimate(samples=0, seconds=0.0, confidence=0.0)

    limited_corr = corr[mask]
    limited_lags = lags[mask]
    peak_index = int(np.argmax(np.abs(limited_corr)))
    peak = float(np.abs(limited_corr[peak_index]))
    confidence = peak / float(ref_energy * meas_energy)
    samples = int(limited_lags[peak_index])
    return DelayEstimate(samples=samples, seconds=samples / float(sample_rate), confidence=confidence)


def align_measurement(reference: np.ndarray, measurement: np.ndarray, delay_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Trim reference and measurement into an aligned pair using a positive measurement delay."""

    ref = np.asarray(reference)
    meas = np.asarray(measurement)
    count = min(ref.size, meas.size)
    ref = ref[-count:]
    meas = meas[-count:]

    if delay_samples > 0:
        return ref[:-delay_samples], meas[delay_samples:]
    if delay_samples < 0:
        lead = abs(delay_samples)
        return ref[lead:], meas[:-lead]
    return ref, meas
