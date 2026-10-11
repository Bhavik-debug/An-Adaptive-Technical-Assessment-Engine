"""Report statistics (pure) and the narrative boundary."""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import ValidationError

from app.blueprint import build_blueprint
from app.interview import EventType, StoredEvent
from app.reporting.narrative import (
    STATS_CLOSE,
    STATS_OPEN,
    NarrativeRejected,
    ReportNarrative,
    check_numbers,
    narrative_inputs,
    narrative_view,
    template_narrative,
)
from app.reporting.stats import INSUFFICIENT_RD, QuestionInfo, ReportNotReady, aggregate, display
from tests.unit.blueprint.conftest import BACKEND_SPEC

E = EventType
SID = uuid.UUID("00000000-0000-4000-8000-000000000025")
BP = build_blueprint(BACKEND_SPEC)
Q = {
    "q1": QuestionInfo("dsa", "arrays"),
    "q2": QuestionInfo("databases", "indexing"),
}


def log(*, end="report", rd_arrays=0.6, rd_index=1.1, priors=None, injected_role=None):
    t1, t2, t3 = (str(uuid.uuid4()) for _ in range(3))
    steps = [
        (E.SESSION_CREATED, {}),
        (
            E.PREPARATION_STARTED,
            {"priors": priors or {}, "prior_source": "resume_profile" if priors else "none"},
        ),
        (E.QUESTION_ASKED, {"question_id": "q1", "text": "Q1?"}),
        (E.ANSWER_SUBMITTED, {"turn_id": t1}),
        (
            E.ANSWER_GRADED,
            {
                "turn_id": t1,
                "score": 0.2,
                "delta_theta": -0.3,
                "ability": {
                    "subtopic_key": "arrays",
                    "theta": -0.6,
                    "rd": rd_arrays,
                    "n_observations": 1,
                },
            },
        ),
        (E.FOLLOW_UP_STARTED, {"reasons": ["high_uncertainty"]}),
        (E.QUESTION_ASKED, {"is_follow_up": True, "text": "Probe?"}),
        (E.ANSWER_SUBMITTED, {"turn_id": t2}),
        (E.ANSWER_GRADED, {"turn_id": t2, "score": 0.9}),
        (E.SELECTION_STARTED, {}),
        (E.QUESTION_ASKED, {"question_id": "q2", "text": "Q2?"}),
        (E.ANSWER_SUBMITTED, {"turn_id": t3}),
        (
            E.ANSWER_GRADED,
            {
                "turn_id": t3,
                "score": 0.95,
                "delta_theta": 0.4,
                "ability": {
                    "subtopic_key": "indexing",
                    "theta": 0.7,
                    "rd": rd_index,
                    "n_observations": 1,
                },
            },
        ),
    ]
    if end == "report":
        steps.append((E.REPORTING_STARTED, {"reasons": ["item_budget_reached"]}))
    return [StoredEvent(seq=i, type=str(t), payload=p) for i, (t, p) in enumerate(steps)]


def stats(**kw):
    return aggregate(events=log(**kw), session_id=SID, blueprint=BP, questions=Q)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("theta", "shown"), [(-2.0, 10), (-1.0, 25), (0.0, 50), (1.0, 75), (2.0, 90)]
)
def test_display_matches_the_plans_table(theta, shown):
    assert display(theta) == shown


def test_refuses_before_reporting():
    with pytest.raises(ReportNotReady):
        aggregate(events=log(end="none"), session_id=SID, blueprint=BP, questions=Q)


def test_questions_follow_ups_and_scores_come_from_the_log():
    s = stats()
    assert s["counts"] == {
        "primary_questions": 2,
        "follow_ups_asked": 1,
        "follow_ups_decided": 1,
        "follow_ups_abandoned": 0,
        "answers_graded": 3,
        "grading_failures": 0,
        "item_budget": BP.item_budget,
    }
    q1, q2 = s["questions"]
    assert (q1["question_id"], q1["topic_key"], q1["score"]) == ("q1", "dsa", 0.2)
    assert q1["follow_ups"] == [{"score": 0.9}]
    assert (q2["score"], q2["follow_ups"]) == (0.95, [])
    assert s["stop_reasons"] == ["item_budget_reached"]


def test_observed_ability_carries_the_plans_interval_and_threshold():
    s = stats()
    arrays = next(x for x in s["subtopics"] if x["subtopic_key"] == "arrays")
    o = arrays["observed"]
    assert o["display"] == display(-0.6)
    assert o["interval"] == [display(-0.6 - 1.96 * 0.6), display(-0.6 + 1.96 * 0.6)]
    assert o["insufficient_evidence"] is False
    indexing = next(x for x in s["subtopics"] if x["subtopic_key"] == "indexing")
    assert indexing["observed"]["insufficient_evidence"] is True  # RD 1.1 > 0.8


def test_gaps_strengths_and_needs_more_follow_plan_9_6():
    s = stats(rd_arrays=0.6, rd_index=0.5)
    assert s["gaps"] == ["arrays"]  # low theta AND low RD
    assert s["strengths"] == ["indexing"]
    assert "arrays" not in s["needs_more_assessment"]
    assert "graphs" in s["needs_more_assessment"]  # never asked
    s2 = stats(rd_arrays=0.81)  # RD just above the 0.8 line
    assert s2["gaps"] == [] and "arrays" in s2["needs_more_assessment"]
    assert INSUFFICIENT_RD == 0.8


def test_priors_are_reported_apart_from_observations():
    s = stats(priors={"graphs": {"theta": 0.4, "rd": 1.0}})
    graphs = next(x for x in s["subtopics"] if x["subtopic_key"] == "graphs")
    assert graphs["observed"] is None  # a claim is not a result
    assert graphs["prior"] == {"theta": 0.4, "rd": 1.0, "source": "resume_profile"}
    assert s["priors"]["source"] == "resume_profile"
    assert "graphs" in s["needs_more_assessment"] and "graphs" not in s["strengths"]


def test_topic_level_uses_only_observed_subtopics():
    s = stats()
    dsa = next(t for t in s["topics"] if t["topic_key"] == "dsa")
    assert dsa["observed"]["theta"] == -0.6 and dsa["questions_asked"] == 1
    caching = next(t for t in s["topics"] if t["topic_key"] == "caching")
    assert caching["observed"] is None and caching["questions_asked"] == 0


def test_deterministic():
    events = log()
    a = aggregate(events=events, session_id=SID, blueprint=BP, questions=Q)
    b = aggregate(events=events, session_id=SID, blueprint=BP, questions=Q)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# Narrative boundary
# ---------------------------------------------------------------------------


def test_the_model_sees_keys_and_numbers_only():
    view = json.dumps(narrative_view(stats()))
    assert "Q1?" not in view and "Probe?" not in view and "theta" not in view
    assert "insufficient evidence" in view


def test_inputs_are_fenced_and_redacted():
    s = stats()
    s["role"] = "x"  # role is not in the view at all
    blob = narrative_inputs(s)["stats"]
    assert STATS_OPEN not in blob and STATS_CLOSE not in blob


def test_a_narrative_with_only_known_numbers_passes():
    s = stats()
    n = ReportNarrative(
        summary=f"You answered {s['counts']['primary_questions']} questions in this session."
    )
    assert check_numbers(n, s) is n


@pytest.mark.parametrize(
    "text", ["Your overall score is 87 out of 100.", "Theta is 0.75 here, roughly."]
)
def test_an_invented_number_is_rejected(text):
    with pytest.raises(NarrativeRejected):
        check_numbers(ReportNarrative(summary=text * 2), stats())


@pytest.mark.parametrize(
    "bad",
    [
        {"summary": "too short"},
        {"summary": "x" * 1501},
        {"summary": "a" * 50, "strengths": ["s"] * 6},
        {"summary": "a" * 50, "score": 10},
    ],
)
def test_the_narrative_schema_is_bounded(bad):
    with pytest.raises(ValidationError):
        ReportNarrative.model_validate(bad)


def test_template_is_deterministic_and_number_safe():
    s = stats(rd_arrays=0.6, rd_index=0.5)
    t = template_narrative(s)
    assert t == template_narrative(s)
    assert "arrays" in " ".join(t.growth_areas) and "indexing" in " ".join(t.strengths)


async def test_narrate_with_model_rejects_an_invented_number(env, monkeypatch):
    """The wiring, not just the function: a model narrative claiming a score the
    stats do not contain is rejected inside narrate_with_model."""
    from app.config import get_settings
    from app.llm import client
    from app.reporting import jobs
    from tests.unit.llm.conftest import FakeProvider, MemoryCache, json_result, make_router

    env()
    provider = FakeProvider(
        "fake", [json_result({"summary": "Overall you scored 87 out of 100 in this assessment."})]
    )
    router = make_router(provider, max_attempts_per_provider=1)

    async def call(task, inputs, schema, **kw):
        return await client.call_structured(
            task, inputs, schema, router=router, cache=MemoryCache(), settings=get_settings()
        )

    monkeypatch.setattr(jobs, "call_structured", call)
    job = jobs.ReportJob(report_id=uuid.uuid4(), session_id=SID, user_id=uuid.uuid4(), attempt=1)
    with pytest.raises(NarrativeRejected):
        await jobs.narrate_with_model(stats(), job)
    sent = " ".join(m.content for m in provider.calls[0].messages)
    assert STATS_OPEN in sent and "Q1?" not in sent
