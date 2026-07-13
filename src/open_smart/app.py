from __future__ import annotations

import argparse
import sys

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .acoustics import compute_spl_metrics, estimate_rt60
from .analyzer import AnalyzerConfig, TransferAnalyzer
from .audio_io import AudioConfig, AudioDeviceError, AudioInputEngine, format_input_devices, validate_input_settings
from .capture import save_noise_floor, save_rt60, save_transfer
from .signals import AudioOutputEngine, SignalConfig, SignalGenerator


class AnalyzerWindow(QtWidgets.QMainWindow):
    def __init__(self, audio: AudioInputEngine, analyzer: TransferAnalyzer, output: AudioOutputEngine | None = None) -> None:
        super().__init__()
        self.audio = audio
        self.analyzer = analyzer
        self.output = output
        self.last_result = None
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

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.measurement_name = QtWidgets.QLineEdit("Room")
        self.measurement_name.setPlaceholderText("Measurement name")
        controls.addWidget(self.measurement_name, stretch=2)

        self.signal_type = QtWidgets.QComboBox()
        self.signal_type.addItems(["pink", "white", "sine", "sweep"])
        self.signal_type.currentTextChanged.connect(self._set_signal_type)
        controls.addWidget(self.signal_type)

        self.level = QtWidgets.QSpinBox()
        self.level.setRange(-60, -3)
        self.level.setValue(-18)
        self.level.setSuffix(" dBFS")
        self.level.valueChanged.connect(self._set_signal_level)
        controls.addWidget(self.level)

        self.frequency = QtWidgets.QSpinBox()
        self.frequency.setRange(20, 20_000)
        self.frequency.setValue(1_000)
        self.frequency.setSuffix(" Hz")
        self.frequency.valueChanged.connect(self._set_signal_frequency)
        controls.addWidget(self.frequency)

        self.signal_button = QtWidgets.QPushButton("Start Signal")
        self.signal_button.clicked.connect(self.toggle_signal)
        controls.addWidget(self.signal_button)

        self.noise_button = QtWidgets.QPushButton("Save Noise Floor")
        self.noise_button.clicked.connect(self.save_noise_snapshot)
        controls.addWidget(self.noise_button)

        self.rt60_button = QtWidgets.QPushButton("Capture RT60")
        self.rt60_button.clicked.connect(self.capture_rt60)
        controls.addWidget(self.rt60_button)

        self.transfer_button = QtWidgets.QPushButton("Save Transfer")
        self.transfer_button.clicked.connect(self.save_transfer_snapshot)
        controls.addWidget(self.transfer_button)

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
        if self.output is not None:
            self.output.stop()
        super().closeEvent(event)

    def _set_signal_type(self, signal_type: str) -> None:
        if self.output is not None:
            self.output.generator.set_signal_type(signal_type)

    def _set_signal_level(self, value: int) -> None:
        if self.output is not None:
            self.output.generator.set_amplitude_dbfs(float(value))

    def _set_signal_frequency(self, value: int) -> None:
        if self.output is not None:
            self.output.generator.set_sine_frequency(float(value))

    def toggle_signal(self) -> None:
        if self.output is None:
            self.status.setText("No output device selected. Relaunch with --output-device INDEX to generate test signals.")
            return
        if self.output.running:
            self.output.stop()
            self.signal_button.setText("Start Signal")
        else:
            self.output.start()
            self.signal_button.setText("Stop Signal")

    def _measurement_name(self) -> str:
        return self.measurement_name.text().strip() or "Room"

    def _mic_samples(self, seconds: float) -> np.ndarray:
        frame_count = int(seconds * self.analyzer.config.sample_rate)
        snapshot = self.audio.ring.latest(frame_count)
        if snapshot.frames.shape[0] == 0:
            return np.empty(0, dtype=np.float32)
        channel = 1 if snapshot.frames.shape[1] > 1 else 0
        return snapshot.frames[:, channel]

    def save_noise_snapshot(self) -> None:
        samples = self._mic_samples(10.0)
        spl = compute_spl_metrics(samples, self.analyzer.config.sample_rate, self.analyzer.config.spl_offset_db)
        paths = save_noise_floor(self._measurement_name(), spl, samples.size / self.analyzer.config.sample_rate, self.analyzer.config.sample_rate)
        self.status.setText(f"Noise floor saved: {paths.summary}")

    def capture_rt60(self) -> None:
        if self.output is not None and self.output.running:
            self.output.stop()
            self.signal_button.setText("Start Signal")
        samples = self._mic_samples(8.0)
        rt60 = estimate_rt60(samples, self.analyzer.config.sample_rate)
        spl = compute_spl_metrics(samples, self.analyzer.config.sample_rate, self.analyzer.config.spl_offset_db)
        paths = save_rt60(self._measurement_name(), rt60, spl)
        rt60_text = "--" if rt60.best is None else f"{rt60.best:.2f} s"
        self.status.setText(f"RT60 saved: {rt60_text} ({rt60.quality}) -> {paths.summary}")

    def save_transfer_snapshot(self) -> None:
        if self.last_result is None:
            self.status.setText("No transfer result available yet.")
            return
        paths = save_transfer(self._measurement_name(), self.last_result.transfer)
        self.status.setText(f"Transfer snapshot saved: {paths.summary}")

    def update_plots(self) -> None:
        snapshot = self.audio.ring.latest(self.analyzer.required_frames)
        if snapshot.frames.shape[0] < self.analyzer.required_frames // 4:
            self.status.setText(f"Collecting audio: {snapshot.frames.shape[0]} frames buffered")
            return

        result = self.analyzer.analyze(snapshot.frames)
        if result is None:
            return
        self.last_result = result

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
    parser.add_argument("--output-device", help="Output device index or name for generated test signals")
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
    output_device: int | str | None
    if args.output_device is None:
        output_device = None
    else:
        try:
            output_device = int(args.output_device)
        except ValueError:
            output_device = args.output_device

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
    output = None
    if output_device is not None:
        output = AudioOutputEngine(
            SignalGenerator(
                SignalConfig(
                    sample_rate=args.sample_rate,
                    block_size=args.block_size,
                    device=output_device,
                )
            )
        )

    app = QtWidgets.QApplication(sys.argv[:1])
    window = AnalyzerWindow(audio, analyzer, output)
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
