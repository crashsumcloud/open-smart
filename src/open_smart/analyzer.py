from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustics import Rt60Metrics, SplMetrics, channel_similarity, compute_spl_metrics, estimate_rt60
from .delay import DelayEstimate, align_measurement, estimate_delay
from .mtw import DEFAULT_WINDOWS, TimeWindow, compute_mtw_spectrum
from .transfer import TransferFunction, compute_transfer


@dataclass(frozen=True)
class AnalyzerConfig:
    sample_rate: int = 48_000
    analysis_seconds: float = 1.5
    max_delay_ms: float = 200.0
    spl_offset_db: float = 0.0
    windows: tuple[TimeWindow, ...] = DEFAULT_WINDOWS


@dataclass(frozen=True)
class AnalyzerResult:
    delay: DelayEstimate
    transfer: TransferFunction
    reference_rms_db: float
    measurement_rms_db: float
    measurement_spl: SplMetrics
    rt60: Rt60Metrics
    channel_similarity: float


def _rms_db(samples: np.ndarray, epsilon: float = 1e-12) -> float:
    rms = np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)) if samples.size else 0.0
    return float(20.0 * np.log10(max(rms, epsilon)))


class TransferAnalyzer:
    """Coordinates delay alignment, MTW FFT, and transfer-function calculation."""

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config

    @property
    def required_frames(self) -> int:
        return max(int(self.config.analysis_seconds * self.config.sample_rate), max(w.fft_size for w in self.config.windows))

    def analyze(self, frames: np.ndarray) -> AnalyzerResult | None:
        if frames.ndim != 2 or frames.shape[1] < 2:
            raise ValueError("frames must contain at least two channels")
        if frames.shape[0] < 2:
            return None

        reference = frames[:, 0]
        measurement = frames[:, 1]
        similarity = channel_similarity(reference, measurement)
        delay = estimate_delay(reference, measurement, self.config.sample_rate, self.config.max_delay_ms)
        aligned_ref, aligned_meas = align_measurement(reference, measurement, delay.samples)
        if aligned_ref.size < 2 or aligned_meas.size < 2:
            return None

        spectrum = compute_mtw_spectrum(aligned_ref, aligned_meas, self.config.sample_rate, self.config.windows)
        transfer = compute_transfer(spectrum)
        return AnalyzerResult(
            delay=delay,
            transfer=transfer,
            reference_rms_db=_rms_db(reference),
            measurement_rms_db=_rms_db(measurement),
            measurement_spl=compute_spl_metrics(measurement, self.config.sample_rate, self.config.spl_offset_db),
            rt60=estimate_rt60(measurement, self.config.sample_rate),
            channel_similarity=similarity,
        )
