"""Plain English -> a supported, validated workflow.

Owner: B. Three hard rules from the build plan:

  * Never generate executable code. The output is a :class:`Workflow`, which
    is a Pydantic model with a closed set of condition types and actions.
  * Ask for a missing target or duration rather than inventing one.
  * Explain an unsupported request instead of pretending to support it.

The parser is deterministic and needs no API key. That is a deliberate
choice: rule compilation is a small, closed classification problem, and a
deterministic parser is auditable, instant, free, and cannot hallucinate a
capability that does not exist. An optional model pass can refine the target
phrasing, but it can never widen the action set, because everything it
produces is re-validated against the same schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schemas import (
    ConditionType,
    CreateTaskAction,
    ROI,
    WebhookAction,
    Workflow,
)

DEFAULT_ROI = ROI(x=0.1, y=0.2, w=0.5, h=0.4)

_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\b", re.IGNORECASE
)
_UNIT_SECONDS = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
                 "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60}

_EMPTY_HINTS = ("empty", "runs out", "run out", "out of", "nothing", "no ",
                "bare", "depleted", "vacant", "unstocked")
_PERSON_HINTS = ("person", "people", "someone", "somebody", "worker",
                 "customer", "human", "anyone")
_PRESENT_HINTS = ("appears", "is present", "shows up", "arrives", "is placed",
                  "is visible", "there is", "detected")

_TASK_HINTS = ("task", "ticket", "work order", "to-do", "todo", "job",
               "reminder", "restock")
_WEBHOOK_HINTS = ("webhook", "notify", "notification", "post to", "send to",
                  "alert", "ping", "call the")

#: Things people reasonably ask for that this prototype cannot do. Each maps to
#: an explanation rather than a silent approximation.
_UNSUPPORTED = [
    (("who ", "identify", "recognise", "recognize", "face", "name of the"),
     "This prototype does not identify people. It can detect that a person is "
     "visible, but not who they are."),
    (("count how many", "how many times", "tally", "number of items"),
     "Counting objects is not a supported condition. Supported conditions are "
     "region_empty, object_present, person_present, and a user-defined visible "
     "condition evaluated as true/false."),
    (("follow", "track across", "across cameras", "multiple cameras"),
     "Multi-camera tracking is not supported. One workflow watches one region "
     "of one camera."),
    (("email", "text me", "sms", "slack", "whatsapp"),
     "Only two actions are supported: create an internal task, or deliver a "
     "webhook to an allowlisted destination."),
    (("temperature", "sound", "noise", "smell", "weight"),
     "Only visible conditions in the camera region can be evaluated."),
]


@dataclass
class CompileResult:
    """Outcome of compiling one instruction."""

    ok: bool
    draft: Optional[Workflow] = None
    #: Questions the UI must ask before the rule can be activated.
    questions: list[str] = field(default_factory=list)
    #: Set when the request asks for something out of scope.
    unsupported_reason: Optional[str] = None
    preview_text: str = ""
    #: True when the condition needs a positive/negative preview before enabling.
    requires_preview: bool = False


def _parse_duration(text: str) -> Optional[float]:
    match = _DURATION_RE.search(text)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2).lower()
    return value * _UNIT_SECONDS.get(unit, 1)


def _classify_condition(text: str) -> ConditionType:
    lowered = text.lower()
    if any(h in lowered for h in _PERSON_HINTS):
        return ConditionType.PERSON_PRESENT
    if any(h in lowered for h in _EMPTY_HINTS):
        return ConditionType.REGION_EMPTY
    if any(h in lowered for h in _PRESENT_HINTS):
        return ConditionType.OBJECT_PRESENT
    return ConditionType.VISUAL_CONDITION


def _extract_target(text: str) -> str:
    """Pull the subject clause out of an 'if X then Y' instruction."""
    working = text.strip().rstrip(".")
    lowered = working.lower()

    if lowered.startswith("if "):
        working = working[3:]
        lowered = working.lower()

    # Cut at the action clause.
    for marker in (", create", " then create", ", make", " then make",
                   ", send", " then send", ", notify", " then notify",
                   ", post", " then post", ", alert", " then alert",
                   ", open", " then open", ","):
        idx = lowered.find(marker)
        if idx > 0:
            working = working[:idx]
            break

    working = re.sub(r"\s+for\s+\d+(?:\.\d+)?\s*\w+\s*$", "", working, flags=re.I)
    working = re.sub(r"^(the|a|an)\s+", "", working.strip(), flags=re.I)
    return working.strip() or ""


def _extract_action(text: str) -> tuple[Optional[CreateTaskAction | WebhookAction], Optional[str]]:
    """Return (action, question). Exactly one is non-None."""
    lowered = text.lower()
    tail = ""
    for marker in ("create", "make", "open", "send", "notify", "post", "alert"):
        idx = lowered.find(marker)
        if idx >= 0:
            tail = text[idx:]
            break

    if any(h in lowered for h in _WEBHOOK_HINTS):
        return WebhookAction(destination_id="test_receiver", title=_title_from(tail)), None
    if any(h in lowered for h in _TASK_HINTS):
        return CreateTaskAction(title=_title_from(tail)), None
    return None, (
        "What should happen when the condition is confirmed? "
        "Supported: create an internal task, or deliver a webhook."
    )


def _title_from(tail: str) -> str:
    cleaned = re.sub(r"^\s*(create|make|open|send|notify|post|alert)\s+", "",
                     tail.strip(), flags=re.I)
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.I)
    cleaned = cleaned.rstrip(". ").strip()
    if not cleaned:
        return "Camera event"
    cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned[:200]


def build_preview_text(workflow: Workflow) -> str:
    """Readable summary shown before the user clicks Activate."""
    if workflow.condition_type == ConditionType.REGION_EMPTY:
        condition = f"Monitor this region for: {workflow.target_description}"
    elif workflow.condition_type == ConditionType.PERSON_PRESENT:
        condition = f"Monitor this region for a person: {workflow.target_description}"
    elif workflow.condition_type == ConditionType.OBJECT_PRESENT:
        condition = f"Monitor this region for: {workflow.target_description}"
    else:
        condition = f"Monitor this region for: {workflow.target_description}"

    if isinstance(workflow.action, CreateTaskAction):
        action = f'create one task "{workflow.action.title}"'
    else:
        action = f'deliver one webhook to "{workflow.action.destination_id}"'

    return (
        f"{condition}. After repeated positive observations spanning "
        f"{workflow.duration_seconds:g} seconds (at least {workflow.min_samples} "
        f"samples), {action}. Reset after {workflow.clear_samples} clear "
        f"observations."
    )


def compile_instruction(
    instruction: str,
    *,
    camera_id: str = "cam_01",
    roi: Optional[ROI] = None,
    workflow_id: Optional[str] = None,
    version: int = 1,
) -> CompileResult:
    """Compile one plain-English instruction into a validated draft workflow.

    The draft is always returned disabled. Activation is a separate, explicit
    step in the UI.
    """
    text = (instruction or "").strip()
    if not text:
        return CompileResult(
            ok=False,
            questions=["What should the camera watch for?"],
        )

    lowered = text.lower()
    for hints, reason in _UNSUPPORTED:
        if any(h in lowered for h in hints):
            return CompileResult(ok=False, unsupported_reason=reason)

    questions: list[str] = []

    target = _extract_target(text)
    if not target or len(target) < 3:
        questions.append(
            "What exactly should the camera look for in the region? "
            "For example: 'the supply tray contains no bottles'."
        )

    duration = _parse_duration(text)
    if duration is None:
        questions.append(
            "How long must the condition hold before acting? For example: "
            "'for 10 seconds'."
        )

    action, action_question = _extract_action(text)
    if action_question:
        questions.append(action_question)

    if questions:
        return CompileResult(ok=False, questions=questions)

    condition_type = _classify_condition(text)
    # Confirmation defaults scale with the duration so a short rule is not
    # impossible to satisfy and a long one is not confirmed by two samples.
    min_samples = max(2, min(10, int(duration // 3) or 2))

    draft_kwargs = dict(
        version=version,
        camera_id=camera_id,
        condition_type=condition_type,
        target_description=target[:500],
        roi=roi or DEFAULT_ROI,
        duration_seconds=float(duration),
        min_samples=min_samples,
        max_gap_seconds=5.0,
        clear_samples=3,
        action=action,
        enabled=False,  # never auto-activate
    )
    if workflow_id:
        draft_kwargs["id"] = workflow_id

    draft = Workflow(**draft_kwargs)
    return CompileResult(
        ok=True,
        draft=draft,
        preview_text=build_preview_text(draft),
        requires_preview=condition_type == ConditionType.VISUAL_CONDITION,
    )
