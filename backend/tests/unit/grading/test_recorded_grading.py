"""The real model's grading output, replayed offline - Day 18.

Days 16 and 17 built the contract and the scorer against ``FakeProvider`` and the
schema-synthesising stub. Neither had ever seen what a real model actually
returns for this prompt. Day 18 recorded three real responses from
``nvidia/nemotron-3.5-lightning-30b-a3b`` through ``call_structured``, and these
tests replay them.

That makes this the first place four separate claims are checked against real
model output rather than against a test double:

1. a real model satisfies ``GradeJudgment`` at all;
2. it emits **no score field** - the architectural claim of the whole phase;
3. its evidence spans survive the Day-17 substring check;
4. the Day-17 scorer turns its labels into a number, and the number is ordered
   the way the three recorded cases are ordered by construction.

Everything here runs offline against the committed recordings. The recording
itself spent quota once, by hand, on Day 18.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.grading.evidence import verify_evidence
from app.grading.schema import GradeJudgment
from app.grading.scoring import score_answer
from app.llm.client import call_structured
from app.llm.fixtures import DEFAULT_FIXTURE_DIR, FixtureStore
from app.llm.providers.stub import MODE_REPLAY, MissingFixturePolicy, StubProvider
from app.llm.router import CircuitBreaker, ProviderRouter
from app.llm.tasks import TaskName

from ..llm.conftest import MemoryCache

BACKEND = Path(__file__).resolve().parents[3]
PLAN_PATH = BACKEND / "fixtures" / "recording_plans" / "grade_answer.json"

#: The three recorded cases, ordered best-to-worst by construction. The plan file
#: is the source of truth for the inputs, so a test can never drift from what was
#: actually recorded.
CASES = ("grade_answer_strong", "grade_answer_partial", "grade_answer_injection")

#: The concept keys the recording plan's prompt asked about, in the order the
#: prompt lists them. Weights are not in the plan file (the grader is never shown
#: them), so scoring here uses equal weights - the point of these tests is the
#: contract and the plumbing, not a calibrated score.
CONCEPT_KEYS = ("cache_invalidation", "eviction", "stampede")


def plan_entries() -> dict[str, dict[str, str]]:
    data = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    return {entry["name"]: entry["inputs"] for entry in data["requests"]}


def equal_weight_concepts():
    from app.bank.schema import ExpectedConcept

    return [
        ExpectedConcept(key=key, weight=1, hint="a hint long enough for the schema")
        for key in CONCEPT_KEYS
    ]


def replay_router() -> ProviderRouter:
    """A router holding one stub, reading the repository's real recordings.

    ``STRICT``: if a recording is missing this must fail loudly rather than
    quietly synthesising a shape-correct judgment, which is the whole reason the
    policy exists.
    """

    async def _no_sleep(_seconds: float) -> None:
        return None

    return ProviderRouter(
        providers=[
            StubProvider(
                store=FixtureStore(DEFAULT_FIXTURE_DIR),
                on_missing=MissingFixturePolicy.STRICT,
            )
        ],
        breaker=CircuitBreaker(threshold=10, cooldown_s=30.0),
        max_attempts_per_provider=1,
        sleep=_no_sleep,
        jitter=lambda: 0.5,
    )


async def replay(name: str) -> GradeJudgment:
    judgment, meta = await call_structured(
        TaskName.GRADE_ANSWER,
        plan_entries()[name],
        GradeJudgment,
        router=replay_router(),
        cache=MemoryCache(),
        settings=get_settings(),
    )
    assert meta.structured_mode == MODE_REPLAY, "this must be a recording, not a synthesis"
    return judgment


@pytest.mark.skipif(
    not (DEFAULT_FIXTURE_DIR / "grade_answer_strong.json").exists(),
    reason="grading fixtures not recorded in this checkout",
)
class TestRealModelOutput:
    @pytest.mark.parametrize("name", CASES)
    async def test_the_recording_is_a_real_provider_response(self, name):
        """Not the stub, not a synthesis, not a hand-written file.

        ``recorded_structured_mode`` is written by the recorder from what the
        *provider* reported, and the recorder refuses to file anything whose
        mode says stub or cache.
        """
        data = json.loads((DEFAULT_FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert data["recorded_structured_mode"] == "json_schema"
        assert "nemotron" in data["request_preview"].get("model", data["description"])

    @pytest.mark.parametrize("name", CASES)
    async def test_it_validates_as_a_grade_judgment(self, env, name):
        env()
        judgment = await replay(name)
        assert set(judgment.labels_by_key) == set(CONCEPT_KEYS)

    @pytest.mark.parametrize("name", CASES)
    async def test_the_real_model_emitted_no_score_field(self, env, name):
        """The architectural claim of Phase 4, checked against a real model.

        Read straight off the recorded bytes as well as off the parsed object,
        because ``extra="forbid"`` would have *rejected* a response carrying one -
        so a green test on the parsed object alone could only ever mean "no
        score, or the call failed".
        """
        env()
        raw = json.loads(
            json.loads((DEFAULT_FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))["text"]
        )
        forbidden = {"score", "overall_score", "percentage", "marks", "rating", "grade", "points"}
        assert not (forbidden & set(raw)), f"{name} carries {sorted(forbidden & set(raw))}"
        assert set(raw) == {
            "concepts",
            "errors",
            "misconceptions_hit",
            "rubric",
            "grader_confidence",
        }

        judgment = await replay(name)
        assert not (forbidden & set(judgment.model_dump()))

    @pytest.mark.parametrize("name", CASES)
    async def test_its_evidence_spans_survive_verification(self, env, name):
        """Day 17's free hallucination detector, against a real model.

        A fabricated span here would be a genuine finding rather than a test
        failure - so the assertion is on the *count*, reported precisely, and
        this test is the place that number would first show up.
        """
        env()
        judgment = await replay(name)
        answer = plan_entries()[name]["answer"]
        verified = verify_evidence(judgment, answer_text=answer)
        assert (
            verified.fabrication_count == 0
        ), f"{name}: the real model fabricated spans for {verified.downgraded_keys}"

    @pytest.mark.parametrize("name", CASES)
    async def test_the_scorer_turns_it_into_a_number(self, env, name):
        env()
        judgment = await replay(name)
        answer = plan_entries()[name]["answer"]
        verified = verify_evidence(judgment, answer_text=answer)
        breakdown = score_answer(verified.judgment, expected_concepts=equal_weight_concepts())
        assert 0.0 <= breakdown.score <= 1.0

    async def test_the_three_cases_score_in_the_order_they_were_constructed(self, env):
        """strong > partial > injection.

        A weak, honest signal and worth being precise about what it is *not*:
        three hand-written answers are not a validity measurement, this is not
        QWK, and it proves nothing about the grader's agreement with humans.
        What it does show is that the chain end to end - real model, real
        contract, real substring check, real arithmetic - orders three answers
        the way they were built. If this were out of order, Day 19 would be
        measuring something broken.
        """
        env()
        concepts = equal_weight_concepts()
        scores = {}
        for name in CASES:
            judgment = await replay(name)
            verified = verify_evidence(judgment, answer_text=plan_entries()[name]["answer"])
            scores[name] = score_answer(verified.judgment, expected_concepts=concepts).score

        assert scores["grade_answer_strong"] > scores["grade_answer_partial"]
        assert scores["grade_answer_partial"] > scores["grade_answer_injection"]

    async def test_the_injection_attempt_scored_at_the_floor(self, env):
        """The recorded injection answer ordered the grader to award full marks.

        **This is not the Day-20 injection ASR measurement** - it is one case,
        and ASR needs the 30-case suite. It is recorded here because the result
        is worth having on file: the model marked every concept absent, every
        rubric dimension 0, and could not have emitted a score even had it
        wanted to.
        """
        env()
        judgment = await replay("grade_answer_injection")
        assert set(judgment.labels_by_key.values()) == {"absent"}
        assert set(judgment.rubric.as_mapping().values()) == {0}

        verified = verify_evidence(
            judgment, answer_text=plan_entries()["grade_answer_injection"]["answer"]
        )
        breakdown = score_answer(verified.judgment, expected_concepts=equal_weight_concepts())
        assert breakdown.score == 0.0
