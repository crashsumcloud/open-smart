from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np


@dataclass(frozen=True)
class RingSnapshot:
    """Contiguous snapshot returned from the audio handoff buffer."""

    frames: np.ndarray
    sample_index: int
    overflow_count: int


class AudioRingBuffer:
    """Bounded multi-channel ring buffer for callback-to-UI audio handoff."""

    def __init__(self, capacity_frames: int, channels: int, dtype: np.dtype = np.float32) -> None:
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.capacity_frames = int(capacity_frames)
        self.channels = int(channels)
        self._buffer = np.zeros((self.capacity_frames, self.channels), dtype=dtype)
        self._write_index = 0
        self._frames_written = 0
        self._overflow_count = 0
        self._lock = threading.Lock()

    @property
    def frames_written(self) -> int:
        with self._lock:
            return self._frames_written

    @property
    def overflow_count(self) -> int:
        with self._lock:
            return self._overflow_count

    def write(self, frames: np.ndarray) -> None:
        data = np.asarray(frames)
        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError(f"frames must have shape (n, {self.channels})")
        if data.shape[0] == 0:
            return

        with self._lock:
            if data.shape[0] >= self.capacity_frames:
                self._buffer[:, :] = data[-self.capacity_frames :]
                self._write_index = 0
                self._frames_written += data.shape[0]
                self._overflow_count += data.shape[0] - self.capacity_frames
                return

            end = self._write_index + data.shape[0]
            if end <= self.capacity_frames:
                self._buffer[self._write_index : end, :] = data
            else:
                split = self.capacity_frames - self._write_index
                self._buffer[self._write_index :, :] = data[:split]
                self._buffer[: end % self.capacity_frames, :] = data[split:]

            unread_before = min(self._frames_written, self.capacity_frames)
            unread_after = unread_before + data.shape[0]
            if unread_after > self.capacity_frames:
                self._overflow_count += unread_after - self.capacity_frames

            self._write_index = end % self.capacity_frames
            self._frames_written += data.shape[0]

    def latest(self, frame_count: int) -> RingSnapshot:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")

        with self._lock:
            available = min(self._frames_written, self.capacity_frames)
            count = min(int(frame_count), available)
            start = (self._write_index - count) % self.capacity_frames
            if count == 0:
                frames = np.empty((0, self.channels), dtype=self._buffer.dtype)
            elif start + count <= self.capacity_frames:
                frames = self._buffer[start : start + count].copy()
            else:
                split = self.capacity_frames - start
                frames = np.vstack((self._buffer[start:], self._buffer[: count - split])).copy()

            sample_index = self._frames_written - count
            return RingSnapshot(frames=frames, sample_index=sample_index, overflow_count=self._overflow_count)
