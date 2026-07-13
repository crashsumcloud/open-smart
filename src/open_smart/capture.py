from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import re

import numpy as np

from .acoustics import Rt60Metrics, SplMetrics
from .transfer import TransferFunction


@dataclass(frozen=True)
class MeasurementPaths:
    summary: Path
    data: Path | None = None


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("_") or "measurement"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def measurement_dir(root: Path | str = "measurements") -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_noise_floor(name: str, spl: SplMetrics, duration_seconds: float, sample_rate: int, root: Path | str = "measurements") -> MeasurementPaths:
    base = measurement_dir(root) / f"{timestamp()}_{safe_name(name)}_noise_floor.csv"
    with base.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit"])
        writer.writerow(["duration", f"{duration_seconds:.3f}", "s"])
        writer.writerow(["sample_rate", sample_rate, "Hz"])
        writer.writerow(["a_weighted_spl", f"{spl.a_weighted_spl:.3f}", "dBA"])
        writer.writerow(["a_weighted_dbfs", f"{spl.a_weighted_dbfs:.3f}", "dBFS(A)"])
        writer.writerow(["rms_dbfs", f"{spl.rms_dbfs:.3f}", "dBFS"])
        writer.writerow(["peak_dbfs", f"{spl.peak_dbfs:.3f}", "dBFS"])
    return MeasurementPaths(summary=base)


def save_rt60(name: str, rt60: Rt60Metrics, spl: SplMetrics, root: Path | str = "measurements") -> MeasurementPaths:
    base = measurement_dir(root) / f"{timestamp()}_{safe_name(name)}_rt60.csv"
    with base.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit"])
        writer.writerow(["rt60_best", "" if rt60.best is None else f"{rt60.best:.3f}", "s"])
        writer.writerow(["rt20", "" if rt60.rt20 is None else f"{rt60.rt20:.3f}", "s"])
        writer.writerow(["rt30", "" if rt60.rt30 is None else f"{rt60.rt30:.3f}", "s"])
        writer.writerow(["decay_range", f"{rt60.decay_range_db:.3f}", "dB"])
        writer.writerow(["quality", rt60.quality, ""])
        writer.writerow(["measurement_spl_a", f"{spl.a_weighted_spl:.3f}", "dBA"])
        writer.writerow(["measurement_peak", f"{spl.peak_dbfs:.3f}", "dBFS"])
    return MeasurementPaths(summary=base)


def save_transfer(name: str, transfer: TransferFunction, root: Path | str = "measurements") -> MeasurementPaths:
    base = measurement_dir(root) / f"{timestamp()}_{safe_name(name)}_transfer.csv"
    with base.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "magnitude_db", "phase_deg", "coherence"])
        finite = np.isfinite(transfer.frequency) & np.isfinite(transfer.magnitude_db) & np.isfinite(transfer.phase_deg) & np.isfinite(transfer.coherence)
        for frequency, magnitude, phase, coherence in zip(
            transfer.frequency[finite],
            transfer.magnitude_db[finite],
            transfer.phase_deg[finite],
            transfer.coherence[finite],
        ):
            if frequency >= 20.0:
                writer.writerow([f"{frequency:.3f}", f"{magnitude:.3f}", f"{phase:.3f}", f"{coherence:.6f}"])
    return MeasurementPaths(summary=base)
