import csv

import numpy as np

from open_smart.acoustics import Rt60Metrics, SplMetrics
from open_smart.capture import save_noise_floor, save_rt60
from open_smart.signals import SignalConfig, SignalGenerator


def test_signal_generator_outputs_stereo_block() -> None:
    generator = SignalGenerator(SignalConfig(signal_type="sine", amplitude_dbfs=-12.0, sine_frequency=1000.0))

    block = generator.next_block(512)

    assert block.shape == (512, 2)
    assert block.dtype == np.float32
    assert np.max(np.abs(block)) <= 10.0 ** (-12.0 / 20.0) + 1e-6


def test_save_noise_floor_csv(tmp_path) -> None:
    metrics = SplMetrics(rms_dbfs=-50.0, a_weighted_dbfs=-55.0, a_weighted_spl=42.0, peak_dbfs=-12.0)

    paths = save_noise_floor("Client Room", metrics, 10.0, 48_000, root=tmp_path)

    with paths.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert ["a_weighted_spl", "42.000", "dBA"] in rows


def test_save_rt60_csv(tmp_path) -> None:
    metrics = SplMetrics(rms_dbfs=-30.0, a_weighted_dbfs=-35.0, a_weighted_spl=62.0, peak_dbfs=-6.0)
    rt60 = Rt60Metrics(rt20=0.6, rt30=0.65, decay_range_db=38.0, quality="RT30")

    paths = save_rt60("Main Hall", rt60, metrics, root=tmp_path)

    text = paths.summary.read_text(encoding="utf-8")
    assert "rt60_best,0.650,s" in text
    assert "quality,RT30," in text
