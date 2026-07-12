import numpy as np

from open_smart.acoustics import channel_similarity, compute_spl_metrics, estimate_rt60
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
    assert result.measurement_spl.a_weighted_dbfs < 0.0


def test_a_weighted_spl_uses_calibration_offset() -> None:
    sample_rate = 48_000
    t = np.arange(sample_rate) / sample_rate
    tone = 0.1 * np.sin(2.0 * np.pi * 1000.0 * t)

    metrics = compute_spl_metrics(tone, sample_rate, spl_offset_db=120.0)

    assert np.isclose(metrics.a_weighted_spl, metrics.a_weighted_dbfs + 120.0)
    assert metrics.peak_dbfs < 0.0


def test_rt60_estimates_exponential_decay() -> None:
    sample_rate = 48_000
    rt60 = 0.8
    duration = 2.0
    t = np.arange(int(duration * sample_rate)) / sample_rate
    decay_db = -60.0 * t / rt60
    impulse_decay = 10.0 ** (decay_db / 20.0)

    estimate = estimate_rt60(impulse_decay, sample_rate)

    assert estimate.best is not None
    assert np.isclose(estimate.best, rt60, atol=0.08)


def test_channel_similarity_flags_duplicate_channels() -> None:
    rng = np.random.default_rng(11)
    samples = rng.normal(0.0, 0.1, 4096)

    assert channel_similarity(samples, samples.copy()) > 0.99
