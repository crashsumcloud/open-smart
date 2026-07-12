from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd

from .ring_buffer import AudioRingBuffer


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 48_000
    channels: int = 2
    block_size: int = 512
    device: int | str | None = None
    buffer_seconds: float = 4.0


StatusCallback = Callable[[sd.CallbackFlags], None]


class AudioInputEngine:
    """Single-device, multi-channel input stream feeding an AudioRingBuffer."""

    def __init__(self, config: AudioConfig, status_callback: StatusCallback | None = None) -> None:
        self.config = config
        capacity = int(config.sample_rate * config.buffer_seconds)
        self.ring = AudioRingBuffer(capacity_frames=capacity, channels=config.channels)
        self._status_callback = status_callback
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        del frames, time_info
        if status and self._status_callback is not None:
            self._status_callback(status)
        self.ring.write(indata.copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            device=self.config.device,
            channels=self.config.channels,
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

    def __enter__(self) -> AudioInputEngine:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


def list_input_devices() -> list[dict[str, object]]:
    devices = sd.query_devices()
    result: list[dict[str, object]] = []
    for index, device in enumerate(devices):
        max_inputs = int(device.get("max_input_channels", 0))
        if max_inputs > 0:
            result.append(
                {
                    "index": index,
                    "name": device.get("name", ""),
                    "max_input_channels": max_inputs,
                    "default_samplerate": device.get("default_samplerate", 0),
                }
            )
    return result
