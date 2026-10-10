"""Frame capture.

Owner: A. Capture runs on its own thread and never shares a clock or a lock
with HTTP handling. The rest of the app reads one thread-safe "latest frame"
slot; nothing queues frames, so slow inference drops stale work instead of
building a backlog.

Sources are pluggable so the same pipeline runs from a USB webcam on the demo
laptop, from a recorded MP4 for repeatable testing, or from a synthetic
generator in CI where no camera exists.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Protocol

import cv2
import numpy as np

from .evidence import BufferedFrame, encode_jpeg
from .schemas import CameraHealth, SourceKind, utcnow

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """One captured frame with both timebases attached at capture time."""

    frame_id: int
    image: np.ndarray
    captured_at_utc: datetime
    captured_monotonic: float

    def to_buffered(self, quality: int = 80) -> BufferedFrame:
        return BufferedFrame(
            frame_id=self.frame_id,
            jpeg=encode_jpeg(self.image, quality),
            captured_monotonic=self.captured_monotonic,
            captured_at_utc=self.captured_at_utc,
        )


class FrameSource(Protocol):
    def open(self) -> bool: ...
    def read(self) -> Optional[np.ndarray]: ...
    def close(self) -> None: ...
    def describe(self) -> str: ...


class UsbSource:
    """A locally attached camera via OpenCV.

    Untested in the development container, which has no /dev/video*. Verify on
    the demo laptop before relying on it.
    """

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            log.error("camera index %s would not open", self.index)
            return False
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        # A failed read is a disconnection signal, not a reason to reuse the
        # previous frame: serving a cached frame as current would let the
        # engine confirm a condition from an image that no longer exists.
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        return f"usb:{self.index}"


class FileSource:
    """A recorded video, replayed. Label this as replay in the UI, never live."""

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self.path = Path(path)
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if not self.path.exists():
            log.error("video file not found: %s", self.path)
            return False
        self._cap = cv2.VideoCapture(str(self.path))
        return bool(self._cap.isOpened())

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        return f"file:{self.path.name}"


class SyntheticSource:
    """Frames from a callable. Used in tests and for a camera-free demo."""

    def __init__(self, generator: Callable[[int], Optional[np.ndarray]]) -> None:
        self.generator = generator
        self._n = 0

    def open(self) -> bool:
        return True

    def read(self) -> Optional[np.ndarray]:
        frame = self.generator(self._n)
        self._n += 1
        return frame

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return "synthetic"


def build_source(settings) -> FrameSource:
    kind = settings.camera_source_kind
    if kind == SourceKind.USB:
        return UsbSource(settings.camera_index)
    if kind == SourceKind.FILE:
        return FileSource(settings.camera_file_path)
    if kind == SourceKind.RTSP:
        # Stretch input. Treated as a file-style capture; not a promise that
        # every camera brand works.
        return FileSource(settings.camera_file_path, loop=False)
    from .demo import synthetic_tray_frame

    return SyntheticSource(synthetic_tray_frame)


class CaptureWorker:
    """Reads frames on a background thread into a latest-frame slot."""

    def __init__(
        self,
        source: FrameSource,
        *,
        fps: float = 10.0,
        stale_after_seconds: float = 6.0,
        on_frame: Optional[Callable[[Frame], None]] = None,
    ) -> None:
        self.source = source
        self.interval = 1.0 / max(0.1, fps)
        self.stale_after = stale_after_seconds
        self.on_frame = on_frame

        self._latest: Optional[Frame] = None
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0
        self._consecutive_failures = 0
        self._health = CameraHealth.DISCONNECTED

    # -- lifecycle --------------------------------------------------------

    def start(self) -> bool:
        if not self.source.open():
            self._health = CameraHealth.DISCONNECTED
            return False
        self._health = CameraHealth.OK
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="capture", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.source.close()
        self._health = CameraHealth.DISCONNECTED

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                image = self.source.read()
            except Exception:
                log.exception("capture read raised")
                image = None

            if image is None:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    with self._lock:
                        self._health = CameraHealth.DISCONNECTED
            else:
                self._consecutive_failures = 0
                frame = Frame(
                    frame_id=self._frame_id,
                    image=image,
                    captured_at_utc=utcnow(),
                    captured_monotonic=time.monotonic(),
                )
                self._frame_id += 1
                try:
                    jpeg = encode_jpeg(image, 80)
                except Exception:
                    jpeg = None
                with self._lock:
                    self._latest = frame
                    self._latest_jpeg = jpeg
                    self._health = CameraHealth.OK
                if self.on_frame is not None:
                    try:
                        self.on_frame(frame)
                    except Exception:
                        log.exception("on_frame callback raised")

            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.interval - elapsed))

    # -- accessors --------------------------------------------------------

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def health(self) -> CameraHealth:
        """Health, recomputed against frame age on every read.

        A source that stops delivering without erroring (a frozen stream) looks
        healthy from the read loop, so staleness is judged by age here.
        """
        with self._lock:
            health, latest = self._health, self._latest
        if health == CameraHealth.DISCONNECTED or latest is None:
            return CameraHealth.DISCONNECTED
        age = time.monotonic() - latest.captured_monotonic
        if age > self.stale_after:
            return CameraHealth.STALE
        return CameraHealth.OK

    def status(self) -> dict:
        latest = self.latest()
        return {
            "source": self.source.describe(),
            "health": self.health().value,
            "frame_id": latest.frame_id if latest else None,
            "last_frame_at": latest.captured_at_utc.isoformat() if latest else None,
            "age_seconds": (
                round(time.monotonic() - latest.captured_monotonic, 2)
                if latest else None
            ),
        }


def crop_region(image: np.ndarray, roi, *, longest_side: int = 640) -> np.ndarray:
    """Crop the configured region, preserving aspect ratio.

    Resized only enough to keep detail legible; small labels need resolution,
    so the crop is never enlarged beyond its native size.
    """
    height, width = image.shape[:2]
    x, y, w, h = roi.to_pixels(width, height)
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return image

    longest = max(crop.shape[:2])
    if longest > longest_side:
        scale = longest_side / longest
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return crop
