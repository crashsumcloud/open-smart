from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mtw import MtwSpectrum


@dataclass(frozen=True)
class TransferFunction:
    frequency: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    coherence: np.ndarray
    complex_response: np.ndarray


def compute_transfer(spectrum: MtwSpectrum, epsilon: float = 1e-12) -> TransferFunction:
    """Compute H1-style complex transfer function from reference to measurement."""

    auto_ref = np.abs(spectrum.reference) ** 2
    auto_meas = np.abs(spectrum.measurement) ** 2
    cross = spectrum.measurement * np.conj(spectrum.reference)
    complex_response = cross / np.maximum(auto_ref, epsilon)

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(complex_response), epsilon))
    phase_deg = np.rad2deg(np.unwrap(np.angle(complex_response)))
    coherence = (np.abs(cross) ** 2) / np.maximum(auto_ref * auto_meas, epsilon)
    coherence = np.clip(coherence.real, 0.0, 1.0)

    return TransferFunction(
        frequency=spectrum.frequency,
        magnitude_db=magnitude_db,
        phase_deg=phase_deg,
        coherence=coherence,
        complex_response=complex_response,
    )


class ExponentialAverager:
    """Stateful exponential average for complex spectra."""

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value: np.ndarray | None = None

    def update(self, value: np.ndarray) -> np.ndarray:
        incoming = np.asarray(value)
        if self._value is None or self._value.shape != incoming.shape:
            self._value = incoming.copy()
        else:
            self._value = self.alpha * incoming + (1.0 - self.alpha) * self._value
        return self._value.copy()

    def reset(self) -> None:
        self._value = None
