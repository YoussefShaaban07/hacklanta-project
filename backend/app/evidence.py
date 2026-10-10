"""Evidence: rolling frame buffer, snapshots and short clips.

Owner: A. Evidence is what turns "the model said so" into something a judge
can check. Two rules shape this module:

  * Never block capture. Clip encoding happens off the capture thread; the
    snapshot is written immediately and the clip path is attached later.
  * Use real capture timestamps for the clip timeline. The buffer stores the
    monotonic capture time of every frame, so a clip spanning a trigger is
    built from when frames were actually taken, not from when encoding ran.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedFrame:
    """A JPEG-compressed frame plus the timebase needed to place it in a clip."""

    frame_id: int
    jpeg: bytes
    captured_monotonic: float
    captured_at_utc: datetime


class RollingBuffer:
    """Bounded, thread-safe ring of recent frames.

    Frames are held JPEG-compressed: 20 seconds at 5 fps of raw 720p would be
    roughly 275 MB, while the same span as JPEG is a few megabytes.
    """

    def __init__(self, seconds: float = 20.0, fps: float = 5.0) -> None:
        self.seconds = seconds
        self.fps = fps
        self._frames: deque[BufferedFrame] = deque(maxlen=max(1, int(seconds * fps)))
        self._lock = threading.Lock()

    def add(self, frame: BufferedFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def latest(self) -> Optional[BufferedFrame]:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def window(self, center_monotonic: float, before: float, after: float) -> list[BufferedFrame]:
        """Frames captured within [center-before, center+after].

        Called after the trigger, so the "after" half fills in as capture
        continues; the caller waits for the window to close before encoding.
        """
        lo = center_monotonic - before
        hi = center_monotonic + after
        with self._lock:
            return [f for f in self._frames if lo <= f.captured_monotonic <= hi]

    def has_covered(self, center_monotonic: float, after: float) -> bool:
        """True once a frame later than ``center + after`` exists."""
        with self._lock:
            if not self._frames:
                return False
            return self._frames[-1].captured_monotonic >= center_monotonic + after


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("JPEG decoding failed")
    return frame


class EvidenceStore:
    """Writes snapshots and clips under a single evidence directory."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.dir / name

    def write_snapshot(self, event_id: str, jpeg: bytes) -> str:
        """Persist the trigger image. Returns a path for the database."""
        path = self._path(f"{event_id}.jpg")
        path.write_bytes(jpeg)
        return str(path)

    def write_region_overlay(
        self, event_id: str, jpeg: bytes, roi_pixels: tuple[int, int, int, int]
    ) -> str:
        """Save the region outline *separately* from the original image.

        The guide asks for the outline to be retained apart from the untouched
        frame, so evidence is never a drawing over the only copy of the truth.
        """
        frame = decode_jpeg(jpeg)
        x, y, w, h = roi_pixels
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 215, 255), 2)
        path = self._path(f"{event_id}_region.jpg")
        path.write_bytes(encode_jpeg(frame))
        return str(path)

    def write_clip(
        self,
        event_id: str,
        frames: Sequence[BufferedFrame],
        *,
        fallback_fps: float = 5.0,
    ) -> Optional[str]:
        """Encode buffered frames into an MP4 using their real timestamps.

        Returns None if there is nothing to encode or the encoder is
        unavailable; the caller keeps the snapshot and discloses the gap rather
        than pretending a clip exists.
        """
        if not frames:
            log.warning("no buffered frames for event %s; clip skipped", event_id)
            return None

        fps = self._measured_fps(frames, fallback_fps)
        first = decode_jpeg(frames[0].jpeg)
        height, width = first.shape[:2]

        path = self._path(f"{event_id}.mp4")
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            log.error("video encoder unavailable; keeping snapshot only")
            return None

        try:
            for buffered in frames:
                frame = decode_jpeg(buffered.jpeg)
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()

        if not path.exists() or path.stat().st_size == 0:
            log.error("clip for %s was empty after encoding", event_id)
            return None
        return str(path)

    @staticmethod
    def _measured_fps(frames: Sequence[BufferedFrame], fallback: float) -> float:
        """Derive playback rate from actual capture times.

        Using the configured sample interval instead would make a clip play at
        the wrong speed whenever inference ran slower than planned.
        """
        if len(frames) < 2:
            return fallback
        span = frames[-1].captured_monotonic - frames[0].captured_monotonic
        if span <= 0:
            return fallback
        return max(1.0, min(30.0, (len(frames) - 1) / span))

    def storage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("*") if p.is_file())

    def prune(self, keep_bytes: int, referenced: set[str]) -> list[str]:
        """Delete oldest unreferenced evidence to stay under a storage cap.

        Referenced files are never removed, however old: an event row pointing
        at a missing file is worse than being over the cap.
        """
        removed: list[str] = []
        files = sorted(
            (p for p in self.dir.glob("*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= keep_bytes:
                break
            if str(path) in referenced:
                continue
            total -= path.stat().st_size
            path.unlink()
            removed.append(str(path))
        return removed
