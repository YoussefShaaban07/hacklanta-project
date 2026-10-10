"""Vision adapter tests.

The property that matters most: nothing a provider can do should produce a
false ``not_met`` or a false ``met``. Failures become ``unknown``.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.camera import crop_region
from app.demo import synthetic_tray_frame
from app.evidence import encode_jpeg
from app.schemas import ConditionType, ObservationResult, Workflow
from app.vision import (
    HeuristicVisionProvider,
    ScriptedVisionProvider,
    VisionVerdict,
    build_prompt,
)

from conftest import make_workflow

FPS = 10.0


def frame_at(seconds: float) -> np.ndarray:
    return synthetic_tray_frame(int(seconds * FPS), FPS)


def crop_jpeg_at(seconds: float, workflow: Workflow) -> bytes:
    return encode_jpeg(crop_region(frame_at(seconds), workflow.roi))


class TestPrompt:
    def test_prompt_blocks_scene_text_injection(self, workflow):
        prompt = build_prompt(workflow)
        assert "image content" in prompt
        assert "NOT as instructions" in prompt

    def test_prompt_forbids_duration_reasoning(self, workflow):
        prompt = build_prompt(workflow)
        assert "elapsed time" in prompt or "duration" in prompt

    def test_prompt_demands_unknown_for_ambiguity(self, workflow):
        assert "unknown" in build_prompt(workflow)


class TestVerdictValidation:
    def test_rejects_an_invented_result_value(self):
        with pytest.raises(ValueError):
            VisionVerdict.model_validate({"result": "probably", "explanation": "x"})

    def test_accepts_the_three_legal_values(self):
        for value in ("met", "not_met", "unknown"):
            assert VisionVerdict.model_validate(
                {"result": value, "explanation": ""}
            ).result == ObservationResult(value)


class TestHeuristicProvider:
    """Exercised against the bundled demo scene, which is drawn specifically so
    occupied / empty / covered are separable."""

    def test_stocked_tray_is_not_met(self, workflow):
        provider = HeuristicVisionProvider()
        outcome = provider.analyze(crop_jpeg_at(5.0, workflow), workflow)
        assert outcome.result == ObservationResult.NOT_MET

    def test_empty_tray_is_met(self, workflow):
        provider = HeuristicVisionProvider()
        outcome = provider.analyze(crop_jpeg_at(25.0, workflow), workflow)
        assert outcome.result == ObservationResult.MET

    def test_covered_lens_is_unknown_not_met(self, workflow):
        """The important one: a covered lens must never read as an empty tray."""
        provider = HeuristicVisionProvider()
        outcome = provider.analyze(crop_jpeg_at(48.0, workflow), workflow)
        assert outcome.result == ObservationResult.UNKNOWN
        assert "dark" in outcome.explanation.lower()

    def test_undecodable_input_is_unknown(self, workflow):
        provider = HeuristicVisionProvider()
        outcome = provider.analyze(b"not a jpeg at all", workflow)
        assert outcome.result == ObservationResult.UNKNOWN

    def test_uniform_frame_is_unknown(self, workflow):
        """A flat grey field carries no evidence either way."""
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        provider = HeuristicVisionProvider()
        outcome = provider.analyze(encode_jpeg(flat), workflow)
        assert outcome.result == ObservationResult.UNKNOWN

    def test_polarity_flips_for_object_present(self):
        """The same occupied region means not_met for empty, met for present."""
        empty_rule = make_workflow(condition_type=ConditionType.REGION_EMPTY)
        present_rule = make_workflow(
            condition_type=ConditionType.OBJECT_PRESENT,
            target_description="a bottle is on the tray",
        )
        provider = HeuristicVisionProvider()
        occupied = crop_jpeg_at(5.0, empty_rule)
        assert provider.analyze(occupied, empty_rule).result == ObservationResult.NOT_MET
        assert provider.analyze(occupied, present_rule).result == ObservationResult.MET

    def test_reports_its_own_identity_honestly(self, workflow):
        provider = HeuristicVisionProvider()
        outcome = provider.analyze(crop_jpeg_at(5.0, workflow), workflow)
        assert outcome.model_id == "heuristic-v1"
        assert outcome.inference_ms >= 0


class TestScriptedProvider:
    def test_returns_the_script_then_repeats_the_last(self, workflow):
        provider = ScriptedVisionProvider(["met", "not_met"])
        blob = crop_jpeg_at(5.0, workflow)
        assert provider.analyze(blob, workflow).result == ObservationResult.MET
        assert provider.analyze(blob, workflow).result == ObservationResult.NOT_MET
        assert provider.analyze(blob, workflow).result == ObservationResult.NOT_MET
        assert provider.calls == 3


class TestCropping:
    def test_crop_matches_the_configured_region(self, workflow):
        frame = frame_at(5.0)
        crop = crop_region(frame, workflow.roi)
        height, width = frame.shape[:2]
        _, _, expected_w, expected_h = workflow.roi.to_pixels(width, height)
        assert crop.shape[0] <= expected_h and crop.shape[1] <= expected_w
        assert crop.size > 0

    def test_large_frames_are_downscaled_but_not_upscaled(self, workflow):
        big = cv2.resize(frame_at(5.0), (2560, 1920))
        assert max(crop_region(big, workflow.roi).shape[:2]) <= 640
        small = cv2.resize(frame_at(5.0), (320, 240))
        crop = crop_region(small, workflow.roi)
        assert max(crop.shape[:2]) <= 640
