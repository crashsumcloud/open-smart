import numpy as np

from open_smart.ring_buffer import AudioRingBuffer


def test_latest_returns_contiguous_wrapped_frames() -> None:
    ring = AudioRingBuffer(capacity_frames=5, channels=2)
    ring.write(np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32))
    ring.write(np.array([[4, 40], [5, 50], [6, 60], [7, 70]], dtype=np.float32))

    snapshot = ring.latest(5)

    assert snapshot.frames.tolist() == [[3, 30], [4, 40], [5, 50], [6, 60], [7, 70]]
    assert snapshot.sample_index == 2
    assert snapshot.overflow_count == 2


def test_large_write_keeps_tail() -> None:
    ring = AudioRingBuffer(capacity_frames=3, channels=1)
    ring.write(np.arange(5, dtype=np.float32).reshape(-1, 1))

    snapshot = ring.latest(3)

    assert snapshot.frames[:, 0].tolist() == [2.0, 3.0, 4.0]
    assert snapshot.overflow_count == 2
