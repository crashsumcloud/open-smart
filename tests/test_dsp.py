import numpy as np

from open_smart.analyzer import AnalyzerConfig, TransferAnalyzer
from open_smart.delay import estimate_delay
from open_smart.mtw import TimeWindow, compute_mtw_spectrum
from open_smart.transfer import compute_transfer


def test_delay_estimator_detects_positive_measurement_delay() -> None:
    sample_rate = 48_000
    rng = np.random.default_rng(7)
    reference = rng.normal(0.0, 0.1, 8192)
    delay = 137
    measurement = np.concatenate((np.zeros(delay), reference[:-delay]))

    estimate = estimate_delay(reference, measurement, sample_rate)

    assert estimate.samples == delay
    assert estimate.confidence > 0.75


def test_transfer_detects_gain_relationship() -> None:
    sample_rate = 48_000
    t = np.arange(8192) / sample_rate
    reference = np.sin(2.0 * np.pi * 1000.0 * t)
    measurement = 2.0 * reference
    spectrum = compute_mtw_spectrum(
        reference,
        measurement,
        sample_rate,
        windows=(TimeWindow("test", 8192, 20.0, 20_000.0),),
    )

    transfer = compute_transfer(spectrum)
    index = int(np.argmin(np.abs(transfer.frequency - 1000.0)))

    assert np.isclose(transfer.magnitude_db[index], 6.0206, atol=0.2)
    assert transfer.coherence[index] > 0.99


def test_analyzer_returns_result_for_two_channel_frames() -> None:
    sample_rate = 48_000
    t = np.arange(8192) / sample_rate
    reference = np.sin(2.0 * np.pi * 500.0 * t)
    measurement = 0.5 * reference
    frames = np.column_stack((reference, measurement)).astype(np.float32)
    analyzer = TransferAnalyzer(
        AnalyzerConfig(
            sample_rate=sample_rate,
            analysis_seconds=0.1,
            windows=(TimeWindow("test", 4096, 20.0, 20_000.0),),
        )
    )

    result = analyzer.analyze(frames)

    assert result is not None
    assert result.delay.samples == 0
    assert result.transfer.frequency.size > 0
