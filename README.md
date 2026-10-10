# hacklanta-project

A camera-to-workflow prototype. Point a camera at something, describe a condition
in plain English, review the compiled rule, and get a task or webhook with
evidence when the condition actually holds.

Built from the four-person 48-hour build plan. Inspired by OpenVector's public
product flow; this is an independent implementation, not their architecture.

> **Status: backend spine.** The state machine, storage, actions and evidence
> are built and tested. The vision adapter, instruction compiler, capture
> worker, HTTP API and UI are **not built yet** — see
> [What is not built](#what-is-not-built). Nothing in this repo has been run
> against a real camera or a real model.

---

## What works today

```
observation ──> engine (pure) ──> runtime ──> SQLite (atomic)
                                      │
                                      ├──> task  / webhook  (at-least-once)
                                      └──> snapshot + clip evidence
```

| Component | File | State |
|---|---|---|
| Shared contracts | `backend/app/schemas.py` | Frozen, fixtures generated from them |
| Event state machine | `backend/app/engine.py` | Complete, pure, 34 tests |
| Storage | `backend/app/db.py` | Complete, WAL + serialized writes |
| Actions (task + webhook) | `backend/app/actions.py` | Complete, allowlist + bounded retry |
| Evidence (buffer/snapshot/clip) | `backend/app/evidence.py` | Complete, real MP4 encoding |
| Runtime wiring | `backend/app/runtime.py` | Complete |

**79 tests, all passing, in ~1.4 seconds.** No test touches a camera, a model,
or the network.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

---

## The parts worth reviewing

### The engine is pure, so the hard behaviour is cheap to test

`transition(episode, observation, workflow) -> (episode, effect)` has no
clock, no I/O and no database. Every row of the plan's state table and every
row of its acceptance table is reachable from synthetic timestamped
observations. A ten-second confirmation, or a five-minute one, runs in
microseconds.

This is why `test_engine.py` can assert things that are otherwise expensive to
check, such as *a condition held for five minutes fires exactly once at every
sampling interval from 0.5s to 4.9s*.

### Duration uses a monotonic clock, display uses UTC

Every observation carries both `captured_monotonic` and `captured_at_utc`. All
timer arithmetic uses the former, so an NTP correction or a DST shift can never
retroactively satisfy a ten-second rule. Stored records keep the latter.

### Duplicate prevention is a database constraint, not dispatcher discipline

`UNIQUE(workflow_id, episode_id)` on events and `UNIQUE(event_id, type)` on
actions. The event and its pending action commit in **one transaction**, so a
crash between them cannot leave an event with no scheduled action. A retry
after a crash returns `False` from `commit_event_with_action` — the normal
outcome, not an error.

Each confirmation mints a fresh `episode_id`, so re-arming after a genuine
clear produces a genuinely distinct event.

### Delivery is at-least-once, and says so

If a receiver accepts a POST and the response is lost, the retry delivers
again. Every payload carries a stable `idempotency_key` derived only from
`(event_id, action_type)`, and the receiver is expected to deduplicate on it.
Retries are bounded (1s, 2s, 4s) and a 4xx other than 408/429 is not retried at
all. A failed delivery is recorded as failed and surfaced, never swallowed.

### The model picks a destination id, never a URL

`WebhookAction.destination_id` is resolved against a backend allowlist. Scene
text, user instructions and model output cannot steer an outbound request to an
arbitrary host. `test_actions.py` asserts that an unknown destination fails
*without any HTTP request being attempted*.

### Unknown is not a negative

An obscured lens, a timeout, or malformed JSON all arrive as `unknown`.
- While `CANDIDATE`: resets the streak, so intermittent obstruction can never
  complete a timer.
- While `ACTIVE`: holds the episode, marks it `condition_unverified`, resets the
  clear streak, and does **not** re-arm. Losing sight of something is not
  evidence that it ended.

---

## What is not built

Honest list. Do not demo from this repo yet.

| Piece | Owner in the plan | Status |
|---|---|---|
| `vision.py` — Gemini adapter | B | **Not written.** No model call has ever run. |
| `compiler.py` — instruction → rule | B | **Not written.** |
| `camera.py` — capture worker | A | **Not written.** |
| `main.py` — FastAPI routes | C | **Not written.** The API contract is agreed but unimplemented. |
| `frontend/` — four screens | D | **Empty scaffold only.** |
| Calibration against real images | B | **Not started.** |
| Acceptance runs (10 positives / 10 negatives) | all | **Not started.** |

The `enabled` guard on `visual_condition` requires `preview_validated`, but
nothing sets that flag yet, because the preview flow lives in the unbuilt UI.

---

## Environment notes

This was developed in a container with **no camera** and **no `GEMINI_API_KEY`**.
Consequences you should know about:

- The USB capture path is written against `cv2.VideoCapture` but has **never
  been executed**. Test it on the demo laptop early — the plan's hour-4 gate
  exists for exactly this reason.
- Clip encoding uses OpenCV's bundled encoder (`mp4v`), which works without a
  system `ffmpeg`. `test_integration.py` writes a real MP4 and reads it back.
- `opencv-python-headless` is pinned rather than `opencv-python`, because the
  GUI build needs `libGL` that a server image does not have. Preview is served
  as JPEG over HTTP, so nothing needs `cv2.imshow`.

## Layout

```
camera-agent/
  backend/app/
    schemas.py     # shared contracts (C)
    db.py          # SQLite, serialized writes (C)
    engine.py      # pure event state machine (C)
    actions.py     # tasks + webhooks, allowlist, retry (C)
    evidence.py    # rolling buffer, snapshots, clips (A)
    runtime.py     # wires the above together (C)
  fixtures/        # generated from the schemas; see scripts/gen_fixtures.py
  tests/           # 79 tests
  scripts/
  data/            # gitignored: database + evidence
```

Regenerate fixtures after any schema change:

```bash
python scripts/gen_fixtures.py && python -m pytest tests/test_fixtures.py -q
```

## Configuration

Copy `.env.example` to `.env`. It is gitignored; never commit a real key.
Pin `VISION_MODEL` to the exact id you calibrate against and do not change it
afterwards — thresholds tuned on one model do not transfer to another.
