from __future__ import annotations

import argparse
import sys

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .analyzer import AnalyzerConfig, TransferAnalyzer
from .audio_io import AudioConfig, AudioDeviceError, AudioInputEngine, format_input_devices, validate_input_settings


class AnalyzerWindow(QtWidgets.QMainWindow):
    def __init__(self, audio: AudioInputEngine, analyzer: TransferAnalyzer) -> None:
        super().__init__()
        self.audio = audio
        self.analyzer = analyzer
        self.setWindowTitle("Open-Smaart Transfer Analyzer")
        self.resize(1280, 820)

        pg.setConfigOptions(antialias=False, background="#111317", foreground="#d7dde8")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        self.status = QtWidgets.QLabel("Starting audio stream...")
        self.status.setMinimumHeight(26)
        layout.addWidget(self.status)
        self.metrics = QtWidgets.QLabel("SPL(A): -- | RT60: -- | Setup: waiting for signal")
        self.metrics.setMinimumHeight(26)
        layout.addWidget(self.metrics)

        grid = pg.GraphicsLayoutWidget()
        layout.addWidget(grid, stretch=1)

        self.mag_plot = grid.addPlot(row=0, col=0, title="Magnitude")
        self.mag_plot.setLabel("left", "dB")
        self.mag_plot.setLabel("bottom", "Hz")
        self.mag_plot.setLogMode(x=True, y=False)
        self.mag_plot.showGrid(x=True, y=True, alpha=0.25)
        self.mag_curve = self.mag_plot.plot(pen=pg.mkPen("#5ec8e5", width=2))

        self.phase_plot = grid.addPlot(row=1, col=0, title="Phase")
        self.phase_plot.setLabel("left", "Degrees")
        self.phase_plot.setLabel("bottom", "Hz")
        self.phase_plot.setLogMode(x=True, y=False)
        self.phase_plot.showGrid(x=True, y=True, alpha=0.25)
        self.phase_curve = self.phase_plot.plot(pen=pg.mkPen("#f5b041", width=2))

        self.coherence_plot = grid.addPlot(row=2, col=0, title="Coherence")
        self.coherence_plot.setLabel("left", "0-1")
        self.coherence_plot.setLabel("bottom", "Hz")
        self.coherence_plot.setLogMode(x=True, y=False)
        self.coherence_plot.setYRange(0, 1)
        self.coherence_plot.showGrid(x=True, y=True, alpha=0.25)
        self.coherence_curve = self.coherence_plot.plot(pen=pg.mkPen("#58d68d", width=2))

        self.setCentralWidget(central)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_plots)

    def _format_tuning_hint(self, result: object) -> str:
        if result.channel_similarity > 0.98 and abs(result.delay.samples) < 2:
            return "Setup warning: reference and mic are nearly identical. Use a real two-channel interface with loopback on input 1 and mic on input 2."

        frequency = result.transfer.frequency
        magnitude = result.transfer.magnitude_db
        coherence = result.transfer.coherence
        valid = (frequency >= 80.0) & (frequency <= 12_000.0) & (coherence >= 0.65) & np.isfinite(magnitude)
        if np.count_nonzero(valid) < 20:
            return "Tuning hint: coherence is low. Raise pink-noise level, reduce background noise, or verify loopback routing."

        low = (frequency >= 80.0) & (frequency <= 250.0) & valid
        mid = (frequency >= 500.0) & (frequency <= 2_000.0) & valid
        high = (frequency >= 4_000.0) & (frequency <= 10_000.0) & valid
        if not (np.any(low) and np.any(mid) and np.any(high)):
            return "Tuning hint: need more coherent bandwidth before making DSP moves."

        low_avg = float(np.median(magnitude[low]))
        mid_avg = float(np.median(magnitude[mid]))
        high_avg = float(np.median(magnitude[high]))
        if low_avg - mid_avg > 6.0:
            return "Tuning hint: low end is hot vs mids. Consider a gentle low-shelf cut or sub/room mode treatment before narrow EQ."
        if mid_avg - low_avg > 6.0:
            return "Tuning hint: low end is light vs mids. Check polarity/crossover/sub level before boosting EQ."
        if high_avg - mid_avg > 6.0:
            return "Tuning hint: top end is hot. Consider a gentle high-shelf cut, but confirm mic calibration first."
        if mid_avg - high_avg > 6.0:
            return "Tuning hint: top end is down. Check mic position/aim and speaker HF trim before boosting."
        return "Tuning hint: broadband balance is close. Look for narrow peaks with high coherence before applying DSP."

    def start(self) -> None:
        self.audio.start()
        self.timer.start()

    def closeEvent(self, event: object) -> None:
        self.timer.stop()
        self.audio.stop()
        super().closeEvent(event)

    def update_plots(self) -> None:
        snapshot = self.audio.ring.latest(self.analyzer.required_frames)
        if snapshot.frames.shape[0] < self.analyzer.required_frames // 4:
            self.status.setText(f"Collecting audio: {snapshot.frames.shape[0]} frames buffered")
            return

        result = self.analyzer.analyze(snapshot.frames)
        if result is None:
            return

        frequency = result.transfer.frequency
        valid = frequency >= 20.0
        frequency = frequency[valid]
        magnitude = result.transfer.magnitude_db[valid]
        phase = ((result.transfer.phase_deg[valid] + 180.0) % 360.0) - 180.0
        coherence = result.transfer.coherence[valid]

        finite = np.isfinite(magnitude) & np.isfinite(phase) & np.isfinite(coherence)
        self.mag_curve.setData(frequency[finite], magnitude[finite])
        self.phase_curve.setData(frequency[finite], phase[finite])
        self.coherence_curve.setData(frequency[finite], coherence[finite])

        self.status.setText(
            "Delay: "
            f"{result.delay.samples:+d} samples / {result.delay.seconds * 1000.0:+.2f} ms "
            f"(confidence {result.delay.confidence:.2f}) | "
            f"Ref {result.reference_rms_db:.1f} dBFS | Mic {result.measurement_rms_db:.1f} dBFS | "
            f"Overflows {snapshot.overflow_count}"
        )
        rt60 = result.rt60.best
        rt60_text = "--" if rt60 is None else f"{rt60:.2f} s ({result.rt60.quality})"
        spl_text = f"{result.measurement_spl.a_weighted_spl:.1f} dBA"
        if abs(self.analyzer.config.spl_offset_db) < 1e-9:
            spl_text += " uncal"
        self.metrics.setText(
            f"SPL(A): {spl_text} | Peak: {result.measurement_spl.peak_dbfs:.1f} dBFS | "
            f"RT60: {rt60_text} | Similarity: {result.channel_similarity:.3f} | "
            f"{self._format_tuning_hint(result)}"
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-Smaart real-time transfer analyzer")
    parser.add_argument("--device", help="Input device index or name")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument(
        "--spl-offset-db",
        type=float,
        default=0.0,
        help="Calibration offset added to A-weighted dBFS. Example: if a 94 dB calibrator reads -32 dBFS(A), use 126.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--check-audio", action="store_true", help="Validate the selected input device and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_devices:
        print(format_input_devices(sample_rate=args.sample_rate))
        return 0

    device: int | str | None
    if args.device is None:
        device = None
    else:
        try:
            device = int(args.device)
        except ValueError:
            device = args.device

    audio_config = AudioConfig(sample_rate=args.sample_rate, block_size=args.block_size, device=device)
    if args.check_audio:
        try:
            validate_input_settings(audio_config)
        except AudioDeviceError as exc:
            print(f"Audio device check failed: {exc}", file=sys.stderr)
            print("\nAvailable input devices:", file=sys.stderr)
            print(format_input_devices(sample_rate=args.sample_rate), file=sys.stderr)
            return 2
        print("Audio device check passed.")
        return 0

    analyzer = TransferAnalyzer(AnalyzerConfig(sample_rate=args.sample_rate, spl_offset_db=args.spl_offset_db))
    audio = AudioInputEngine(audio_config)

    app = QtWidgets.QApplication(sys.argv[:1])
    window = AnalyzerWindow(audio, analyzer)
    window.show()
    try:
        window.start()
    except AudioDeviceError as exc:
        QtWidgets.QMessageBox.critical(
            window,
            "Audio Device Error",
            f"{exc}\n\nRun `python -m open_smart.app --list-devices`, then launch with `--device INDEX`.",
        )
        audio.stop()
        return 2
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
