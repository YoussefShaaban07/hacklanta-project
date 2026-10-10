"""Configuration loaded from the environment.

Owner: C. One place that reads ``.env`` so no module reaches for os.environ on
its own and no secret is ever written anywhere but ``.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .actions import DestinationAllowlist, WebhookDestination
from .schemas import SourceKind

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    # vision
    gemini_api_key: Optional[str] = None
    vision_model: str = "gemini-2.0-flash"
    #: auto -> Gemini when a key exists, otherwise the heuristic provider.
    vision_provider: str = "auto"

    # camera
    camera_source_kind: SourceKind = SourceKind.SYNTHETIC
    camera_index: int = 0
    camera_file_path: str = ""
    sample_interval_seconds: float = 2.0
    preview_fps: float = 10.0
    #: Frames with no read for this long mark the camera stale.
    stale_after_seconds: float = 6.0

    # storage
    database_path: Path = REPO_ROOT / "data" / "camera_agent.db"
    evidence_dir: Path = REPO_ROOT / "data" / "evidence"
    evidence_cap_mb: int = 2048

    # actions
    webhook_timeout_seconds: float = 5.0
    allowlist: DestinationAllowlist = field(default_factory=DestinationAllowlist)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def effective_vision_provider(self) -> str:
        if self.vision_provider != "auto":
            return self.vision_provider
        return "gemini" if self.has_gemini else "heuristic"


def parse_destinations(raw: str) -> DestinationAllowlist:
    """Parse ``id|url,id|url`` into an allowlist.

    URLs only ever come from here, never from an instruction or model output.
    """
    allowlist = DestinationAllowlist()
    for chunk in filter(None, (c.strip() for c in raw.split(","))):
        if "|" not in chunk:
            continue
        dest_id, _, url = chunk.partition("|")
        dest_id, url = dest_id.strip(), url.strip()
        if dest_id and url:
            allowlist.add(WebhookDestination(id=dest_id, url=url, label=dest_id))
    return allowlist


def load_settings(env_file: Optional[Path] = None) -> Settings:
    load_dotenv(env_file or REPO_ROOT / ".env", override=False)

    kind_raw = (os.getenv("CAMERA_SOURCE_KIND") or "synthetic").strip().lower()
    try:
        kind = SourceKind(kind_raw)
    except ValueError:
        kind = SourceKind.SYNTHETIC

    settings = Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        vision_model=os.getenv("VISION_MODEL") or "gemini-2.0-flash",
        vision_provider=(os.getenv("VISION_PROVIDER") or "auto").strip().lower(),
        camera_source_kind=kind,
        camera_index=_int("CAMERA_INDEX", 0),
        camera_file_path=os.getenv("CAMERA_FILE_PATH") or "",
        sample_interval_seconds=_float("SAMPLE_INTERVAL_SECONDS", 2.0),
        preview_fps=_float("PREVIEW_FPS", 10.0),
        stale_after_seconds=_float("STALE_AFTER_SECONDS", 6.0),
        database_path=Path(os.getenv("DATABASE_PATH") or REPO_ROOT / "data" / "camera_agent.db"),
        evidence_dir=Path(os.getenv("EVIDENCE_DIR") or REPO_ROOT / "data" / "evidence"),
        evidence_cap_mb=_int("EVIDENCE_CAP_MB", 2048),
        webhook_timeout_seconds=_float("WEBHOOK_TIMEOUT_SECONDS", 5.0),
        allowlist=parse_destinations(
            os.getenv("WEBHOOK_DESTINATIONS")
            or "test_receiver|http://127.0.0.1:9000/hook"
        ),
    )
    # A placeholder left in .env is not a key.
    if settings.gemini_api_key in {"your-key-here", "changeme", ""}:
        settings.gemini_api_key = None
    return settings
