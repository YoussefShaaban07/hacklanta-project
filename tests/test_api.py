"""HTTP API tests.

These run the real application through its lifespan: a real capture worker on
the synthetic source, a real database in tmp_path, and the heuristic provider.
Only outbound webhooks are absent, because no rule here uses one.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Set before the app's lifespan runs. load_settings uses override=False, so
    # these win over anything in a developer's local .env.
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("CAMERA_SOURCE_KIND", "synthetic")
    monkeypatch.setenv("SAMPLE_INTERVAL_SECONDS", "0.3")
    monkeypatch.setenv("PREVIEW_FPS", "20")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("WEBHOOK_DESTINATIONS", "test_receiver|http://127.0.0.1:9/hook")

    from app.main import app

    with TestClient(app) as test_client:
        # The capture worker starts on its own thread, so the first frame lands
        # shortly after startup. Wait for it rather than racing it: a test that
        # sometimes sees "no frame available" tells us nothing.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if test_client.get("/api/cameras/cam_01/snapshot").status_code == 200:
                break
            time.sleep(0.05)
        else:
            pytest.fail("capture worker produced no frame within 10s")
        yield test_client


DEMO = "If the supply tray stays empty for 10 seconds, create a restocking task."


class TestHealth:
    def test_health_reports_provider_honestly(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        # No key configured, so the API must not imply a real model is in use.
        assert body["using_real_model"] is False
        assert body["vision_model"] == "heuristic-v1"

    def test_cameras_label_non_usb_sources_as_not_live(self, client):
        camera = client.get("/api/cameras").json()[0]
        assert camera["source_kind"] == "synthetic"
        assert camera["is_live"] is False

    def test_snapshot_returns_a_jpeg(self, client):
        response = client.get("/api/cameras/cam_01/snapshot")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content[:2] == b"\xff\xd8"  # JPEG SOI

    def test_crop_preview_returns_the_configured_region(self, client):
        response = client.post(
            "/api/cameras/cam_01/crop-preview",
            json={"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4},
        )
        assert response.status_code == 200
        assert response.content[:2] == b"\xff\xd8"

    def test_crop_preview_rejects_an_out_of_bounds_region(self, client):
        response = client.post(
            "/api/cameras/cam_01/crop-preview",
            json={"x": 0.8, "y": 0.1, "w": 0.5, "h": 0.2},
        )
        assert response.status_code == 422


class TestCompile:
    def test_compiles_the_demo_instruction(self, client):
        body = client.post("/api/workflows/compile", json={"instruction": DEMO}).json()
        assert body["ok"] is True
        assert body["draft"]["condition_type"] == "region_empty"
        assert body["draft"]["enabled"] is False
        assert "10 seconds" in body["preview_text"]

    def test_unsupported_request_is_explained(self, client):
        body = client.post("/api/workflows/compile", json={
            "instruction": "Tell me who walked past for 10 seconds, create a task."
        }).json()
        assert body["ok"] is False
        assert body["unsupported_reason"]
        assert body["draft"] is None

    def test_missing_duration_asks_a_question(self, client):
        body = client.post("/api/workflows/compile", json={
            "instruction": "If the tray is empty, create a task."
        }).json()
        assert body["ok"] is False
        assert body["questions"]


def make_workflow_payload(**overrides):
    payload = {
        "id": "wf_api", "version": 1, "camera_id": "cam_01",
        "condition_type": "region_empty",
        "target_description": "supply tray contains no bottles",
        "roi": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4},
        "duration_seconds": 10.0, "min_samples": 3,
        "max_gap_seconds": 5.0, "clear_samples": 3,
        "action": {"type": "create_task", "title": "Restock the supply tray"},
        "enabled": False,
    }
    payload.update(overrides)
    return {"workflow": payload}


class TestWorkflowLifecycle:
    def test_save_and_list(self, client):
        saved = client.post("/api/workflows", json=make_workflow_payload()).json()
        assert saved["id"] == "wf_api"
        assert saved["worker_running"] is False  # saved paused

        listed = client.get("/api/workflows").json()
        assert [w["id"] for w in listed] == ["wf_api"]

    def test_enabling_starts_a_worker(self, client):
        """Regression: the save route must run on the event loop, or
        asyncio.create_task has no loop and the worker silently never starts."""
        saved = client.post(
            "/api/workflows", json=make_workflow_payload(enabled=True)
        ).json()
        assert saved["enabled"] is True
        assert saved["worker_running"] is True

        status = client.get("/api/workflows/wf_api/status").json()
        assert status["worker_running"] is True
        assert status["display_state"] in {"monitoring", "starting", "confirming",
                                           "uncertain", "event active"}

    def test_paused_workflow_never_claims_to_be_monitoring(self, client):
        client.post("/api/workflows", json=make_workflow_payload())
        status = client.get("/api/workflows/wf_api/status").json()
        assert status["display_state"] == "paused"

    def test_rule_edit_bumps_the_version(self, client):
        client.post("/api/workflows", json=make_workflow_payload(enabled=True))
        patched = client.patch(
            "/api/workflows/wf_api", json={"duration_seconds": 25.0}
        ).json()
        assert patched["version"] == 2
        assert patched["version_bumped"] is True
        assert patched["duration_seconds"] == 25.0

    def test_enable_toggle_does_not_bump_the_version(self, client):
        """Pausing is not a rule change, so in-flight results stay valid."""
        client.post("/api/workflows", json=make_workflow_payload(enabled=True))
        patched = client.patch("/api/workflows/wf_api", json={"enabled": False}).json()
        assert patched["version"] == 1
        assert patched["version_bumped"] is False
        assert client.get("/api/workflows/wf_api/status").json()["display_state"] == "paused"

    def test_invalid_edit_is_rejected_not_persisted(self, client):
        """model_copy skips validators, so the route must re-validate."""
        client.post("/api/workflows", json=make_workflow_payload())
        response = client.patch(
            "/api/workflows/wf_api", json={"roi": {"x": 0.9, "y": 0.1, "w": 0.5, "h": 0.2}}
        )
        assert response.status_code in (400, 422, 500)
        stored = client.get("/api/workflows").json()[0]
        assert stored["roi"]["x"] == 0.1, "the bad edit must not have been stored"

    def test_patch_unknown_workflow_is_404(self, client):
        assert client.patch("/api/workflows/nope", json={"enabled": True}).status_code == 404

    def test_empty_patch_is_rejected(self, client):
        client.post("/api/workflows", json=make_workflow_payload())
        assert client.patch("/api/workflows/wf_api", json={}).status_code == 400


class TestEventsAndTasks:
    def test_empty_collections(self, client):
        assert client.get("/api/events").json() == []
        assert client.get("/api/tasks").json() == []

    def test_unknown_event_is_404(self, client):
        assert client.get("/api/events/evt_nope").status_code == 404

    def test_retry_unknown_action_is_404(self, client):
        assert client.post("/api/actions/act_nope/retry").status_code == 404


class TestEvidenceServing:
    def test_missing_evidence_is_404(self, client):
        assert client.get("/api/evidence/nope.jpg").status_code == 404

    @pytest.mark.parametrize("attack", [
        "../../../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "subdir/../../../etc/passwd",
    ])
    def test_path_traversal_is_refused(self, client, attack):
        response = client.get(f"/api/evidence/{attack}")
        assert response.status_code in (404, 400)
        assert b"root:" not in response.content


class TestHistoryEndpoint:
    def test_supported_question(self, client):
        body = client.post("/api/history/query", json={
            "question": "How many events occurred today?"
        }).json()
        assert body["supported"] is True
        assert "0" in body["answer"]
        assert body["scope_note"]

    def test_unsupported_question_offers_suggestions(self, client):
        body = client.post("/api/history/query", json={
            "question": "Who was here yesterday?"
        }).json()
        assert body["supported"] is False
        assert body["suggestions"]

    def test_no_records_is_distinguished_from_nothing_happened(self, client):
        body = client.post("/api/history/query", json={
            "question": "When did this rule last trigger?"
        }).json()
        assert "no matching recorded events" in body["answer"].lower()
