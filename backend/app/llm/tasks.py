"""The task routing table - plan section 13.6, in code.

Every LLM call in this project names a *task*, never a model.  The task decides
the capability tier, the sampling settings, the output ceiling and whether the
model is allowed to think first.  Concentrating those choices here means:

* "which model grades answers?" has exactly one answer, in one file;
* changing the grading temperature is a one-line diff with a test around it;
* a caller cannot accidentally grade at temperature 0.9.

Tasks are declared as soon as the plan names them.  A task without a prompt
template yet raises ``PromptNotRegisteredError`` when called - see ``prompts.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.llm.types import ModelTier


class TaskName(StrEnum):
    """Every distinct job we ask a model to do, across all phases."""

    #: Day 3 only: the smallest possible round-trip that exercises the whole
    #: chokepoint end to end.  Infrastructure, not interview logic.
    CONNECTIVITY_PROBE = "connectivity_probe"

    # Declared now, prompts land with their phase.
    RESUME_EXTRACTION = "resume_extraction"
    JD_EXTRACTION = "jd_extraction"
    QUESTION_RENDER = "question_render"
    GRADE_ANSWER = "grade_answer"
    GRADE_RECHECK = "grade_recheck"
    FOLLOW_UP_PROBE = "follow_up_probe"
    FINAL_REPORT = "final_report"
    DEEP_DIVE_AGENT = "deep_dive_agent"

    #: Day 18, plan section 12.2 step 2. Writes controlled synthetic answers for
    #: the grader's evaluation set, given the concept keys to demonstrate and the
    #: ones to omit. **Offline tooling, not interview machinery**: it is invoked
    #: by `scripts/build_grading_evalset.py` and by nothing on a turn path. It is
    #: in this table because every LLM call in this project names a task, and an
    #: eval-set builder that routed itself would be the first exception.
    EVAL_ANSWER_SYNTHESIS = "eval_answer_synthesis"

    #: Day 19, plan section 12.2 step 3. Assigns the provisional 0-4 reference
    #: label the validity harness measures the grader against. **Offline
    #: tooling**, invoked by `scripts/build_reference_labels.py` and by nothing
    #: on a turn path.
    #:
    #: Three tasks rather than one because they are three different information
    #: sets, and a task is how this project records what a call was shown. The
    #: blind pass sees the question and the answer; the concept-aware pass also
    #: sees the question's expected-concept list; the adjudicator sees two
    #: rationales. Routing them as one task would make a label's provenance
    #: unrecoverable from its trace.
    EVAL_REFERENCE_LABEL = "eval_reference_label"
    EVAL_REFERENCE_LABEL_CONCEPTS = "eval_reference_label_concepts"
    EVAL_REFERENCE_LABEL_ADJUDICATE = "eval_reference_label_adjudicate"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """How one task is sampled and constrained."""

    tier: ModelTier
    #: 0.0 where we want the same input judged the same way twice; higher only
    #: where variety in wording is the point.
    temperature: float
    max_output_tokens: int
    #: Nucleus sampling. Irrelevant at temperature 0 (there is nothing to
    #: sample), so deterministic tasks leave it at 1.0 rather than pretending.
    top_p: float = 1.0
    #: Let the model produce hidden reasoning before the answer.  Off by default:
    #: reasoning is billed as output tokens and adds seconds, and it buys
    #: nothing on schema-constrained extraction.  See docs in ``prompts.py``.
    reasoning: bool = False
    reasoning_budget_tokens: int | None = None
    #: Caching a non-deterministic task would defeat the reason it is
    #: non-deterministic, so ``None`` means "cache iff temperature == 0".
    cacheable: bool | None = None

    @property
    def is_cacheable(self) -> bool:
        if self.cacheable is not None:
            return self.cacheable
        return self.temperature == 0.0


#: Plan section 13.6.  Grading gets the quality budget; everything else is cheap.
TASK_SPECS: dict[TaskName, TaskSpec] = {
    TaskName.CONNECTIVITY_PROBE: TaskSpec(
        tier=ModelTier.SMALL_FAST,
        temperature=0.0,
        max_output_tokens=256,
        # Never cached: the entire point is to prove the network path works,
        # and a cache hit would prove nothing.
        cacheable=False,
    ),
    TaskName.RESUME_EXTRACTION: TaskSpec(
        tier=ModelTier.SMALL_FAST, temperature=0.0, max_output_tokens=2048
    ),
    TaskName.JD_EXTRACTION: TaskSpec(
        tier=ModelTier.SMALL_FAST, temperature=0.0, max_output_tokens=1536
    ),
    TaskName.QUESTION_RENDER: TaskSpec(
        tier=ModelTier.SMALL_FAST, temperature=0.4, top_p=0.95, max_output_tokens=512
    ),
    TaskName.GRADE_ANSWER: TaskSpec(tier=ModelTier.MID, temperature=0.0, max_output_tokens=1024),
    TaskName.GRADE_RECHECK: TaskSpec(
        # The confidence gate samples this one three times and compares, so it
        # must vary - caching it would return the same answer three times and
        # report perfect agreement.
        tier=ModelTier.MID,
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=1024,
        cacheable=False,
    ),
    TaskName.FOLLOW_UP_PROBE: TaskSpec(
        tier=ModelTier.SMALL_FAST, temperature=0.5, top_p=0.95, max_output_tokens=384
    ),
    TaskName.FINAL_REPORT: TaskSpec(
        tier=ModelTier.MID, temperature=0.3, top_p=0.95, max_output_tokens=4096
    ),
    TaskName.EVAL_ANSWER_SYNTHESIS: TaskSpec(
        # Temperature 0: a dataset that regenerates differently every run is not
        # a dataset, it is a sample. Determinism here also makes the build
        # cacheable, so a re-run after a crash costs nothing, and it means the
        # committed file can be regenerated and diffed rather than only trusted.
        # The variety this set needs comes from 24 questions x 5 briefs, which is
        # variety by construction rather than by sampling.
        tier=ModelTier.SMALL_FAST,
        temperature=0.0,
        max_output_tokens=1024,
    ),
    # The reference-labelling passes. Temperature 0 on all three: a yardstick
    # that gives a different reading each time it is applied is not a yardstick,
    # and determinism is what makes the pass resumable after a rate limit and
    # re-derivable from the committed artefact rather than only trusted.
    #
    # MID tier, matching GRADE_ANSWER. Labelling is the harder of the two jobs -
    # it is holistic judgement with no concept checklist to lean on - so a
    # cheaper labeller would make the measured QWK a statement about the
    # labeller's ceiling rather than about the grader.
    TaskName.EVAL_REFERENCE_LABEL: TaskSpec(
        tier=ModelTier.MID, temperature=0.0, max_output_tokens=1024
    ),
    TaskName.EVAL_REFERENCE_LABEL_CONCEPTS: TaskSpec(
        tier=ModelTier.MID, temperature=0.0, max_output_tokens=1024
    ),
    TaskName.EVAL_REFERENCE_LABEL_ADJUDICATE: TaskSpec(
        tier=ModelTier.MID, temperature=0.0, max_output_tokens=1024
    ),
    TaskName.DEEP_DIVE_AGENT: TaskSpec(
        # The one place multi-step reasoning genuinely earns its latency.
        tier=ModelTier.MID,
        temperature=0.2,
        max_output_tokens=2048,
        reasoning=True,
        reasoning_budget_tokens=4096,
    ),
}


def get_task_spec(task: TaskName) -> TaskSpec:
    """The routing entry for ``task``. Every task in the enum has one."""
    try:
        return TASK_SPECS[task]
    except KeyError as exc:  # pragma: no cover - guarded by test_every_task_has_a_spec
        raise KeyError(f"no routing entry for task {task!r}") from exc
