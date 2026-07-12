from __future__ import annotations

import argparse
import sys

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .analyzer import AnalyzerConfig, TransferAnalyzer
from .audio_io import AudioConfig, AudioInputEngine, list_input_devices


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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-Smaart real-time transfer analyzer")
    parser.add_argument("--device", help="Input device index or name")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_devices:
        for device in list_input_devices():
            print(
                f"{device['index']}: {device['name']} "
                f"({device['max_input_channels']} in, default {device['default_samplerate']} Hz)"
            )
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
    analyzer = TransferAnalyzer(AnalyzerConfig(sample_rate=args.sample_rate))
    audio = AudioInputEngine(audio_config)

    app = QtWidgets.QApplication(sys.argv[:1])
    window = AnalyzerWindow(audio, analyzer)
    window.show()
    window.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
