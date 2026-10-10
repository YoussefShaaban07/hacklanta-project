"""Instruction compiler tests.

The compiler's job is to be conservative: produce a validated rule, ask when
something is missing, and refuse clearly when the request is out of scope. It
must never invent a capability or auto-activate.
"""

from __future__ import annotations

import pytest

from app.compiler import build_preview_text, compile_instruction
from app.schemas import ConditionType, CreateTaskAction, ROI, WebhookAction

DEMO = "If the supply tray stays empty for 10 seconds, create a restocking task."


class TestDemoInstruction:
    def test_compiles_the_guide_example(self):
        result = compile_instruction(DEMO)
        assert result.ok
        draft = result.draft
        assert draft is not None
        assert draft.condition_type == ConditionType.REGION_EMPTY
        assert draft.duration_seconds == 10.0
        assert isinstance(draft.action, CreateTaskAction)
        assert "restock" in draft.action.title.lower()

    def test_never_auto_activates(self):
        result = compile_instruction(DEMO)
        assert result.draft is not None
        assert result.draft.enabled is False, "activation is always an explicit step"

    def test_preview_text_is_readable(self):
        result = compile_instruction(DEMO)
        text = result.preview_text
        assert "10 seconds" in text
        assert "Reset after 3 clear observations" in text
        assert result.draft.action.title in text


class TestDurations:
    @pytest.mark.parametrize(
        "phrase,expected",
        [
            ("for 5 seconds", 5.0),
            ("for 30 secs", 30.0),
            ("for 2 minutes", 120.0),
            ("for 1 min", 60.0),
            ("for 45s", 45.0),
        ],
    )
    def test_parses_durations(self, phrase, expected):
        result = compile_instruction(f"If the tray is empty {phrase}, create a task.")
        assert result.ok, result.questions
        assert result.draft.duration_seconds == expected

    def test_missing_duration_is_a_question_not_a_guess(self):
        result = compile_instruction("If the tray is empty, create a task.")
        assert not result.ok
        assert result.draft is None
        assert any("how long" in q.lower() for q in result.questions)


class TestConditionClassification:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("If the tray is empty for 10 seconds, create a task.",
             ConditionType.REGION_EMPTY),
            ("If a person appears for 5 seconds, create a task.",
             ConditionType.PERSON_PRESENT),
            ("If a pallet is present for 20 seconds, create a task.",
             ConditionType.OBJECT_PRESENT),
            ("If the warning light is blinking red for 10 seconds, create a task.",
             ConditionType.VISUAL_CONDITION),
        ],
    )
    def test_classifies(self, text, expected):
        result = compile_instruction(text)
        assert result.ok, result.questions
        assert result.draft.condition_type == expected

    def test_custom_condition_requires_preview(self):
        result = compile_instruction(
            "If the warning light is blinking red for 10 seconds, create a task."
        )
        assert result.requires_preview is True
        # And the schema refuses to enable it until the preview is confirmed.
        with pytest.raises(ValueError, match="preview"):
            result.draft.model_copy(update={"enabled": True}).model_validate(
                result.draft.model_copy(update={"enabled": True}).model_dump()
            )


class TestActions:
    def test_webhook_action(self):
        result = compile_instruction(
            "If the tray is empty for 10 seconds, send a webhook."
        )
        assert result.ok
        assert isinstance(result.draft.action, WebhookAction)

    def test_action_uses_allowlist_id_not_a_url(self):
        result = compile_instruction(
            "If the tray is empty for 10 seconds, post to https://evil.example/x"
        )
        if result.ok:
            action = result.draft.action
            assert isinstance(action, WebhookAction)
            assert "evil.example" not in action.destination_id
            assert "http" not in action.destination_id

    def test_missing_action_is_a_question(self):
        result = compile_instruction("If the tray is empty for 10 seconds.")
        assert not result.ok
        assert any("what should happen" in q.lower() for q in result.questions)


class TestUnsupported:
    @pytest.mark.parametrize(
        "text",
        [
            "Tell me who walked past the camera for 10 seconds, create a task.",
            "Count how many boxes are on the shelf for 10 seconds, create a task.",
            "Track the forklift across multiple cameras for 10 seconds, create a task.",
            "If the tray is empty for 10 seconds, email me.",
            "If the temperature rises for 10 seconds, create a task.",
        ],
    )
    def test_explains_rather_than_inventing(self, text):
        result = compile_instruction(text)
        assert not result.ok
        assert result.unsupported_reason
        assert result.draft is None

    def test_empty_instruction(self):
        result = compile_instruction("   ")
        assert not result.ok
        assert result.questions


class TestRoi:
    def test_uses_supplied_region(self):
        roi = ROI(x=0.25, y=0.25, w=0.4, h=0.3)
        result = compile_instruction(DEMO, roi=roi)
        assert result.draft.roi == roi

    def test_defaults_when_none_supplied(self):
        result = compile_instruction(DEMO)
        assert 0 <= result.draft.roi.x <= 1
        assert result.draft.roi.x + result.draft.roi.w <= 1.0


class TestPreviewText:
    def test_webhook_preview_names_destination(self):
        result = compile_instruction(
            "If the tray is empty for 10 seconds, send a webhook."
        )
        assert "webhook" in build_preview_text(result.draft).lower()
