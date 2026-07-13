from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np
from scipy import signal
import sounddevice as sd


@dataclass
class SignalConfig:
    sample_rate: int = 48_000
    channels: int = 2
    block_size: int = 512
    device: int | str | None = None
    signal_type: str = "pink"
    amplitude_dbfs: float = -18.0
    sine_frequency: float = 1_000.0


class SignalGenerator:
    def __init__(self, config: SignalConfig) -> None:
        self.config = config
        self._phase = 0.0
        self._rng = np.random.default_rng()
        self._pink_state = np.zeros(7, dtype=np.float64)
        self._lock = threading.Lock()

    @property
    def amplitude(self) -> float:
        return float(10.0 ** (self.config.amplitude_dbfs / 20.0))

    def set_signal_type(self, signal_type: str) -> None:
        with self._lock:
            self.config.signal_type = signal_type

    def set_amplitude_dbfs(self, amplitude_dbfs: float) -> None:
        with self._lock:
            self.config.amplitude_dbfs = amplitude_dbfs

    def set_sine_frequency(self, frequency: float) -> None:
        with self._lock:
            self.config.sine_frequency = frequency

    def next_block(self, frame_count: int) -> np.ndarray:
        with self._lock:
            signal_type = self.config.signal_type
            amplitude = self.amplitude
            frequency = self.config.sine_frequency
            sample_rate = self.config.sample_rate

        if signal_type == "sine":
            t = (np.arange(frame_count, dtype=np.float64) + self._phase) / sample_rate
            mono = np.sin(2.0 * np.pi * frequency * t)
            self._phase = (self._phase + frame_count) % sample_rate
        elif signal_type == "white":
            mono = self._rng.normal(0.0, 0.33, frame_count)
        elif signal_type == "sweep":
            t = (np.arange(frame_count, dtype=np.float64) + self._phase) / sample_rate
            local = np.mod(t, 8.0)
            mono = signal.chirp(local, f0=20.0, f1=20_000.0, t1=8.0, method="logarithmic")
            self._phase = (self._phase + frame_count) % int(sample_rate * 8.0)
        else:
            mono = self._pink_noise(frame_count)

        block = np.repeat((mono * amplitude)[:, np.newaxis], self.config.channels, axis=1)
        return np.clip(block, -1.0, 1.0).astype(np.float32)

    def _pink_noise(self, frame_count: int) -> np.ndarray:
        white = self._rng.normal(0.0, 0.33, frame_count)
        out = np.empty(frame_count, dtype=np.float64)
        b = self._pink_state
        for i, sample in enumerate(white):
            b[0] = 0.99886 * b[0] + sample * 0.0555179
            b[1] = 0.99332 * b[1] + sample * 0.0750759
            b[2] = 0.96900 * b[2] + sample * 0.1538520
            b[3] = 0.86650 * b[3] + sample * 0.3104856
            b[4] = 0.55000 * b[4] + sample * 0.5329522
            b[5] = -0.7616 * b[5] - sample * 0.0168980
            out[i] = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + sample * 0.5362
            b[6] = sample * 0.115926
        return out * 0.11


class AudioOutputEngine:
    def __init__(self, generator: SignalGenerator) -> None:
        self.generator = generator
        self._stream: sd.OutputStream | None = None

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        del time_info, status
        outdata[:, :] = self.generator.next_block(frames)

    def start(self) -> None:
        if self._stream is not None:
            return
        config = self.generator.config
        self._stream = sd.OutputStream(
            samplerate=config.sample_rate,
            blocksize=config.block_size,
            device=config.device,
            channels=config.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None
