from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class SplMetrics:
    rms_dbfs: float
    a_weighted_dbfs: float
    a_weighted_spl: float
    peak_dbfs: float


@dataclass(frozen=True)
class Rt60Metrics:
    rt20: float | None
    rt30: float | None
    decay_range_db: float
    quality: str

    @property
    def best(self) -> float | None:
        return self.rt30 if self.rt30 is not None else self.rt20


def rms_dbfs(samples: np.ndarray, epsilon: float = 1e-12) -> float:
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return -240.0
    rms = np.sqrt(np.mean(data * data))
    return float(20.0 * np.log10(max(rms, epsilon)))


def peak_dbfs(samples: np.ndarray, epsilon: float = 1e-12) -> float:
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return -240.0
    peak = np.max(np.abs(data))
    return float(20.0 * np.log10(max(peak, epsilon)))


def a_weighting_sos(sample_rate: int) -> np.ndarray:
    """Return a digital A-weighting filter using IEC/CD 1672 analog poles."""

    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12_194.217
    a1000 = 1.9997

    nums = [(2.0 * np.pi * f4) ** 2 * 10.0 ** (a1000 / 20.0), 0.0, 0.0, 0.0, 0.0]
    dens = np.polymul([1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2], [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1.0, 2.0 * np.pi * f3]), [1.0, 2.0 * np.pi * f2])
    z, p, k = signal.bilinear_zpk(*signal.tf2zpk(nums, dens), fs=sample_rate)
    return signal.zpk2sos(z, p, k)


def compute_spl_metrics(samples: np.ndarray, sample_rate: int, spl_offset_db: float = 0.0) -> SplMetrics:
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return SplMetrics(-240.0, -240.0, -240.0 + spl_offset_db, -240.0)

    weighted = signal.sosfilt(a_weighting_sos(sample_rate), data)
    weighted_dbfs = rms_dbfs(weighted)
    return SplMetrics(
        rms_dbfs=rms_dbfs(data),
        a_weighted_dbfs=weighted_dbfs,
        a_weighted_spl=weighted_dbfs + spl_offset_db,
        peak_dbfs=peak_dbfs(data),
    )


def estimate_rt60(samples: np.ndarray, sample_rate: int) -> Rt60Metrics:
    """Estimate RT60 from a decay segment using Schroeder reverse integration."""

    data = np.asarray(samples, dtype=np.float64)
    if data.size < sample_rate // 4:
        return Rt60Metrics(None, None, 0.0, "Need more audio")

    tail_count = max(min(data.size // 10, sample_rate // 2), 1)
    data = data - np.median(data[-tail_count:])
    peak = int(np.argmax(np.abs(data)))
    decay = data[peak:]
    if decay.size < sample_rate // 5:
        return Rt60Metrics(None, None, 0.0, "Decay too short")

    energy = decay * decay
    if np.max(energy) <= 1e-14:
        return Rt60Metrics(None, None, 0.0, "Signal too quiet")

    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10.0 * np.log10(np.maximum(edc / np.max(edc), 1e-14))
    times = np.arange(edc_db.size) / float(sample_rate)
    usable_floor = float(np.percentile(edc_db[-max(sample_rate // 10, 1) :], 95))
    decay_range = abs(usable_floor)

    def fit_decay(low_db: float, high_db: float, multiplier: float) -> float | None:
        mask = (edc_db <= low_db) & (edc_db >= high_db)
        if np.count_nonzero(mask) < 12:
            return None
        slope, _ = np.polyfit(times[mask], edc_db[mask], 1)
        if slope >= -1e-9:
            return None
        return float(multiplier / abs(slope))

    rt20 = fit_decay(-5.0, -25.0, 60.0)
    rt30 = fit_decay(-5.0, -35.0, 60.0)
    if rt30 is not None:
        quality = "RT30"
    elif rt20 is not None:
        quality = "RT20"
    else:
        quality = "No clear decay"
    return Rt60Metrics(rt20=rt20, rt30=rt30, decay_range_db=decay_range, quality=quality)


def channel_similarity(reference: np.ndarray, measurement: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    meas = np.asarray(measurement, dtype=np.float64)
    count = min(ref.size, meas.size)
    if count < 2:
        return 0.0
    ref = ref[-count:] - np.mean(ref[-count:])
    meas = meas[-count:] - np.mean(meas[-count:])
    denom = np.linalg.norm(ref) * np.linalg.norm(meas)
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(ref, meas) / denom, -1.0, 1.0))
