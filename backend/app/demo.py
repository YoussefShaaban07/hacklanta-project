"""Synthetic demo scene: a supply tray that empties and is restocked.

This exists so the whole pipeline is runnable and testable with no camera and
no API key. It is explicitly a stand-in, and the UI labels any non-USB source
as replay rather than live.

The scene is drawn so the heuristic vision provider can read it honestly:
  * occupied -> bottles with hard outlines, producing high edge density
  * empty    -> a shallow gradient, low edge density but enough intensity
                spread to avoid the "lens covered" unknown branch
  * covered  -> near-black, which must come back as unknown
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FPS = 10.0

#: One full cycle of the demo, in seconds.
PHASE_STOCKED = (0.0, 15.0)      # tray has bottles      -> not_met
PHASE_EMPTY = (15.0, 45.0)       # tray is empty         -> met (fires ~25s)
PHASE_COVERED = (45.0, 52.0)     # lens obscured         -> unknown
PHASE_RESTOCKED = (52.0, 70.0)   # bottles back          -> not_met, closes
CYCLE_SECONDS = 70.0


def _phase(t: float) -> str:
    t = t % CYCLE_SECONDS
    if PHASE_STOCKED[0] <= t < PHASE_STOCKED[1]:
        return "stocked"
    if PHASE_EMPTY[0] <= t < PHASE_EMPTY[1]:
        return "empty"
    if PHASE_COVERED[0] <= t < PHASE_COVERED[1]:
        return "covered"
    return "restocked"


def _base_scene() -> np.ndarray:
    """Room background with a workbench. Constant across phases."""
    scene = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    # Wall gradient.
    for y in range(HEIGHT):
        scene[y, :] = (70 + y // 12, 72 + y // 12, 78 + y // 12)
    # Bench.
    cv2.rectangle(scene, (0, 330), (WIDTH, HEIGHT), (96, 104, 120), -1)
    cv2.line(scene, (0, 330), (WIDTH, 330), (60, 66, 80), 2)
    return scene


def _draw_tray(scene: np.ndarray) -> tuple[int, int, int, int]:
    """Draw the tray and return its pixel box."""
    # Sits inside the default ROI (x=0.1, y=0.2, w=0.5, h=0.4).
    x, y, w, h = 70, 100, 300, 190
    cv2.rectangle(scene, (x, y), (x + w, y + h), (138, 142, 150), -1)
    # Shallow gradient across the tray floor: enough variation that an empty
    # tray is not mistaken for a covered lens, but no hard edges.
    floor = scene[y + 8:y + h - 8, x + 8:x + w - 8]
    rows = floor.shape[0]
    for row in range(rows):
        shade = 150 + int(22 * (row / max(1, rows - 1)))
        floor[row, :] = (shade, shade - 4, shade - 10)
    cv2.rectangle(scene, (x, y), (x + w, y + h), (92, 96, 104), 3)
    return x, y, w, h


def _draw_bottles(scene: np.ndarray, box: tuple[int, int, int, int], count: int = 5) -> None:
    x, y, w, h = box
    slot = w // (count + 1)
    for i in range(1, count + 1):
        cx = x + slot * i
        top = y + 30
        bottom = y + h - 25
        cv2.rectangle(scene, (cx - 18, top), (cx + 18, bottom), (58, 120, 196), -1)
        cv2.rectangle(scene, (cx - 18, top), (cx + 18, bottom), (24, 48, 92), 2)
        cv2.rectangle(scene, (cx - 8, top - 16), (cx + 8, top), (40, 90, 150), -1)
        cv2.rectangle(scene, (cx - 8, top - 16), (cx + 8, top), (20, 40, 80), 2)
        cv2.line(scene, (cx - 18, top + 34), (cx + 18, top + 34), (225, 232, 240), 2)


def synthetic_tray_frame(frame_number: int, fps: float = FPS) -> Optional[np.ndarray]:
    """Render the scene at the given frame number."""
    t = frame_number / fps
    phase = _phase(t)

    scene = _base_scene()
    box = _draw_tray(scene)

    if phase in ("stocked", "restocked"):
        _draw_bottles(scene, box)
    elif phase == "covered":
        # Lens obscured: near-black everywhere, which must read as unknown.
        scene = (scene * 0.06).astype(np.uint8)

    label = {
        "stocked": "TRAY STOCKED",
        "empty": "TRAY EMPTY",
        "covered": "LENS COVERED",
        "restocked": "TRAY RESTOCKED",
    }[phase]
    # Drawn outside the ROI so it never influences the judged crop.
    cv2.putText(scene, label, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (235, 240, 248), 2)
    cv2.putText(scene, f"t={t % CYCLE_SECONDS:5.1f}s  SYNTHETIC DEMO",
                (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 198, 210), 1)
    return scene


def write_demo_video(path: str | Path, seconds: float = CYCLE_SECONDS,
                     fps: float = FPS) -> Path:
    """Render one full cycle to an MP4 for repeatable file-source testing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError("could not open video writer")
    try:
        for n in range(int(seconds * fps)):
            writer.write(synthetic_tray_frame(n, fps))
    finally:
        writer.release()
    return path


if __name__ == "__main__":  # pragma: no cover
    out = write_demo_video(Path(__file__).resolve().parents[2] / "fixtures" / "demo_tray.mp4")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
