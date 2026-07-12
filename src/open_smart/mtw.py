from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class TimeWindow:
    name: str
    fft_size: int
    frequency_min: float
    frequency_max: float


@dataclass(frozen=True)
class MtwSpectrum:
    frequency: np.ndarray
    reference: np.ndarray
    measurement: np.ndarray
    windows: tuple[TimeWindow, ...]


DEFAULT_WINDOWS: tuple[TimeWindow, ...] = (
    TimeWindow("LF 1.37s", 65_536, 20.0, 250.0),
    TimeWindow("MF 341ms", 16_384, 250.0, 2_000.0),
    TimeWindow("HF 85ms", 4_096, 2_000.0, 24_000.0),
)


def _windowed_rfft(samples: np.ndarray, fft_size: int) -> np.ndarray:
    if samples.size < fft_size:
        padded = np.zeros(fft_size, dtype=np.float64)
        padded[-samples.size :] = samples
        samples = padded
    else:
        samples = samples[-fft_size:]
    window = signal.windows.hann(fft_size, sym=False)
    return np.fft.rfft(samples * window)


def compute_mtw_spectrum(
    reference: np.ndarray,
    measurement: np.ndarray,
    sample_rate: int,
    windows: tuple[TimeWindow, ...] = DEFAULT_WINDOWS,
) -> MtwSpectrum:
    """Compute a composite MTW spectrum from aligned time-domain channels."""

    ref = np.asarray(reference, dtype=np.float64)
    meas = np.asarray(measurement, dtype=np.float64)
    if ref.ndim != 1 or meas.ndim != 1:
        raise ValueError("reference and measurement must be 1-D arrays")
    if not windows:
        raise ValueError("at least one time window is required")

    base_size = max(window.fft_size for window in windows)
    frequency = np.fft.rfftfreq(base_size, d=1.0 / sample_rate)
    ref_out = np.zeros(frequency.shape, dtype=np.complex128)
    meas_out = np.zeros(frequency.shape, dtype=np.complex128)

    for window in windows:
        local_freq = np.fft.rfftfreq(window.fft_size, d=1.0 / sample_rate)
        ref_fft = _windowed_rfft(ref, window.fft_size)
        meas_fft = _windowed_rfft(meas, window.fft_size)
        mask = (frequency >= window.frequency_min) & (frequency < window.frequency_max)
        if not np.any(mask):
            continue
        ref_out[mask] = np.interp(frequency[mask], local_freq, ref_fft.real) + 1j * np.interp(
            frequency[mask], local_freq, ref_fft.imag
        )
        meas_out[mask] = np.interp(frequency[mask], local_freq, meas_fft.real) + 1j * np.interp(
            frequency[mask], local_freq, meas_fft.imag
        )

    return MtwSpectrum(frequency=frequency, reference=ref_out, measurement=meas_out, windows=windows)
