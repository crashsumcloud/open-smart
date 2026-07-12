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


class AudioDeviceError(RuntimeError):
    """Raised when an input device cannot satisfy the analyzer requirements."""


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
        validate_input_settings(self.config)
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


def format_input_devices(required_channels: int = 2, sample_rate: int = 48_000) -> str:
    lines = []
    for device in list_input_devices():
        marker = "*" if int(device["max_input_channels"]) >= required_channels else " "
        lines.append(
            f"{marker} {device['index']}: {device['name']} "
            f"({device['max_input_channels']} in, default {device['default_samplerate']} Hz)"
        )
    heading = f"* = has at least {required_channels} input channels. Requested sample rate: {sample_rate} Hz."
    return heading + "\n" + "\n".join(lines)


def validate_input_settings(config: AudioConfig) -> None:
    if config.device is None:
        raise AudioDeviceError(
            "No input device selected. Run `python -m open_smart.app --list-devices`, "
            "then launch with `--device INDEX` for a two-channel input."
        )

    try:
        device_info = sd.query_devices(config.device, kind="input")
    except Exception as exc:
        raise AudioDeviceError(f"Could not query input device {config.device!r}: {exc}") from exc

    max_inputs = int(device_info.get("max_input_channels", 0))
    if max_inputs < config.channels:
        name = device_info.get("name", config.device)
        raise AudioDeviceError(
            f"Input device {name!r} has {max_inputs} input channel(s), "
            f"but Open-Smaart needs {config.channels} synchronous input channels."
        )

    try:
        sd.check_input_settings(
            device=config.device,
            channels=config.channels,
            samplerate=config.sample_rate,
            dtype="float32",
        )
    except Exception as exc:
        name = device_info.get("name", config.device)
        raise AudioDeviceError(
            f"Input device {name!r} cannot open {config.channels} channel(s) "
            f"at {config.sample_rate} Hz: {exc}"
        ) from exc
