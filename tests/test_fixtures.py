"""Fixtures must stay loadable by the models they were generated from.

Four people build against fixtures/ while the real workers are unfinished. If
someone renames a field without regenerating them, this fails at commit time
rather than at integration time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import (
    ActionRecord,
    ConditionType,
    Episode,
    Event,
    Observation,
    Task,
    Workflow,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

MODELS = {
    "workflow.json": Workflow,
    "observation.json": Observation,
    "event.json": Event,
    "episode.json": Episode,
    "action.json": ActionRecord,
    "task.json": Task,
}


@pytest.mark.parametrize("filename,model", sorted(MODELS.items()))
def test_fixture_round_trips(filename, model):
    raw = json.loads((FIXTURES / filename).read_text())
    parsed = model.model_validate(raw)
    assert json.loads(parsed.model_dump_json()) == raw


def test_workflow_fixture_matches_the_guide_example():
    wf = Workflow.model_validate_json((FIXTURES / "workflow.json").read_text())
    assert wf.condition_type == ConditionType.REGION_EMPTY
    assert wf.duration_seconds == 10.0
    assert wf.min_samples == 3
    assert wf.action.title == "Restock the supply tray"


def test_every_fixture_has_a_model():
    on_disk = {p.name for p in FIXTURES.glob("*.json")}
    assert on_disk == set(MODELS), "add new fixtures to MODELS so they are checked"


class TestSchemaGuards:
    def test_roi_must_stay_inside_the_frame(self):
        with pytest.raises(ValueError, match="right edge"):
            Workflow.model_validate(
                {
                    **json.loads((FIXTURES / "workflow.json").read_text()),
                    "roi": {"x": 0.8, "y": 0.1, "w": 0.5, "h": 0.2},
                }
            )

    def test_unvalidated_visual_condition_cannot_be_enabled(self):
        base = json.loads((FIXTURES / "workflow.json").read_text())
        with pytest.raises(ValueError, match="preview"):
            Workflow.model_validate(
                {
                    **base,
                    "condition_type": "visual_condition",
                    "enabled": True,
                    "preview_validated": False,
                }
            )

    def test_validated_visual_condition_can_be_enabled(self):
        base = json.loads((FIXTURES / "workflow.json").read_text())
        wf = Workflow.model_validate(
            {
                **base,
                "condition_type": "visual_condition",
                "enabled": True,
                "preview_validated": True,
            }
        )
        assert wf.enabled is True

    def test_roi_pixel_conversion_clamps(self):
        wf = Workflow.model_validate_json((FIXTURES / "workflow.json").read_text())
        x, y, w, h = wf.roi.to_pixels(1280, 720)
        assert (x, y) == (128, 144)
        assert x + w <= 1280 and y + h <= 720

    def test_unknown_fields_are_rejected(self):
        base = json.loads((FIXTURES / "workflow.json").read_text())
        with pytest.raises(ValueError):
            Workflow.model_validate({**base, "sneaky_extra": True})
