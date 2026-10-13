"""Day 26's two read-only routes, against real Postgres and the real app:
``GET /api/sessions`` (the dashboard's history) and ``GET /api/setup-options``
(what the setup wizard may offer)."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from app.bank.taxonomy import load_taxonomy
from app.blueprint.spec import LEVELS, MODES

SPEC = {
    "role": "Backend engineer",
    "level": "senior",
    "mode": "screening",
    "duration_minutes": 30,
    "topics": [{"topic_key": "dsa"}, {"topic_key": "databases", "jd_weight": 0.5}],
}


async def _register(client, email):
    r = await client.post(
        "/api/auth/register", json={"email": email, "password": "correct horse 9"}
    )
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest_asyncio.fixture
async def headers(client):
    return await _register(client, "dash@example.com")


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_requires_auth(client):
    assert (await client.get("/api/sessions")).status_code == 401


@pytest.mark.asyncio
async def test_empty_history_is_an_empty_list(client, headers):
    r = await client.get("/api/sessions", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


@pytest.mark.asyncio
async def test_lists_own_sessions_newest_first_without_internals(client, headers):
    first = (await client.post("/api/sessions", json=SPEC, headers=headers)).json()
    second = (
        await client.post("/api/sessions", json={**SPEC, "role": "Data engineer"}, headers=headers)
    ).json()

    r = await client.get("/api/sessions", headers=headers)
    assert r.status_code == 200
    rows = r.json()["sessions"]
    assert [row["session_id"] for row in rows] == [second["session_id"], first["session_id"]]
    row = rows[1]
    assert row["state"] == "CREATED"
    assert row["role"] == "Backend engineer"
    assert row["level"] == "senior" and row["mode"] == "screening"
    assert row["topic_keys"] == ["dsa", "databases"]
    assert row["report_status"] is None and row["ended_at"] is None
    # The dashboard row carries no selection internals and no grades.
    for leaked in ("blueprint", "target_theta", "difficulty_window", "score", "theta"):
        assert leaked not in row


@pytest.mark.asyncio
async def test_never_lists_another_users_sessions(client, headers):
    await client.post("/api/sessions", json=SPEC, headers=headers)
    other = await _register(client, "other@example.com")
    r = await client.get("/api/sessions", headers=other)
    assert r.json() == {"sessions": []}


@pytest.mark.asyncio
async def test_limit_is_bounded(client, headers):
    for _ in range(3):
        await client.post("/api/sessions", json=SPEC, headers=headers)
    r = await client.get("/api/sessions", params={"limit": 2}, headers=headers)
    assert len(r.json()["sessions"]) == 2
    assert (
        await client.get("/api/sessions", params={"limit": 0}, headers=headers)
    ).status_code == 422
    assert (
        await client.get("/api/sessions", params={"limit": 101}, headers=headers)
    ).status_code == 422


@pytest.mark.asyncio
async def test_existing_session_routes_still_resolve(client, headers):
    """Adding ``GET ""`` must not shadow the ``/{session_id}/...`` routes."""
    sid = (await client.post("/api/sessions", json=SPEC, headers=headers)).json()["session_id"]
    r = await client.get(f"/api/sessions/{sid}/report", headers=headers)
    assert r.status_code == 409  # not finished - the Day-25 route, unchanged
    r = await client.get(f"/api/sessions/{uuid.uuid4()}/report", headers=headers)
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/setup-options
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_options_require_auth(client):
    assert (await client.get("/api/setup-options")).status_code == 401


@pytest.mark.asyncio
async def test_setup_options_mirror_taxonomy_and_spec(client, headers):
    r = await client.get("/api/setup-options", headers=headers)
    assert r.status_code == 200
    body = r.json()
    tax = load_taxonomy()
    assert {t["key"] for t in body["topics"]} == {n.key for n in tax.topics()}
    assert {s["key"] for t in body["topics"] for s in t["subtopics"]} == {
        n.key for n in tax.subtopics()
    }
    assert body["levels"] == list(LEVELS) and body["modes"] == list(MODES)
    assert body["default_level"] in body["levels"] and body["default_mode"] in body["modes"]
    assert (
        body["min_duration_minutes"]
        <= body["default_duration_minutes"]
        <= body["max_duration_minutes"]
    )
    # Nothing about how selection works.
    assert "target_theta" not in r.text and "seconds_per_item" not in r.text


@pytest.mark.asyncio
async def test_every_offered_topic_creates_a_session(client, headers):
    """What the wizard offers, the builder accepts."""
    body = (await client.get("/api/setup-options", headers=headers)).json()
    for topic in body["topics"]:
        spec = {"role": "Engineer", "topics": [{"topic_key": topic["key"]}]}
        r = await client.post("/api/sessions", json=spec, headers=headers)
        assert r.status_code == 201, (topic["key"], r.text)
