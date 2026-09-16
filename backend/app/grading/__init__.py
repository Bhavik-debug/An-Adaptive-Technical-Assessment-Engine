"""The grader - plan sections 7 and 13.2.

``grading/`` owns rubrics, concept matching, score arithmetic and the confidence
gate, and must not know about session state or retrieval (plan section 13.2).
As of Day 17 it owns all four.

The design in one line, from plan section 7.2::

    answer + expected_concepts
        -> ONE classification call
        -> validated object
        -> code computes the score

and the property that makes it defensible: **the model never emits a number that
becomes a score.**  ``GradeJudgment`` has no score field, forbids extra keys, and
is the only shape ``judge_answer`` can return.

The four passes (plan section 7.2)
----------------------------------

* ``rubric.py``   - the anchored 0-4 levels: a descriptor and an example for each
  of four dimensions, folded into the prompt so the fingerprint covers them.
* ``schema.py``   - ``GradeJudgment``, the Pass-B output contract.
* ``prompt.py``   - grading prompt v1, registered for ``GRADE_ANSWER`` and, at
  identical text, for ``GRADE_RECHECK``.
* ``grader.py``   - **Pass A** (assemble) and **Pass B** (one ``call_structured``
  call), plus the relational check that a judgment is about the question asked.
* ``evidence.py`` - span verification: a fabricated quotation is downgraded to
  ``absent`` and logged. Pure.
* ``scoring.py``  - **Pass C**, the arithmetic. Pure, and the only place a score
  is ever computed.
* ``pipeline.py`` - **Pass D**, the confidence gate, and ``grade_answer()``: the
  one function that runs all four.

Two entry points, and the difference matters::

    judge_answer(...)  -> (GradeJudgment, CallMeta)   one call, no score
    grade_answer(...)  -> GradeResult                 the whole chain, with a score

``judge_answer`` is Pass B on its own and stays public because the Day-19
validity harness wants to collect raw judgments without paying for a gate.
``grade_answer`` is what an interview turn calls.

Evaluation tooling (Day 18)
---------------------------

* ``evalset.py``   - the 120-record contract, the stratified question sample, the
  randomised labelling order, and every whole-dataset validation rule. Pure.
* ``answergen.py`` - controlled synthetic answers for that set, generated with a
  code-decided concept split.

Both are **offline tooling, not interview machinery**: nothing on a turn path
imports them, and they import nothing from ``scoring.py``, ``pipeline.py``,
``evidence.py`` or ``grader.py``. That last part is load-bearing and tested - a
ground truth computed from the thing being measured measures nothing.

Importing this package registers every grading prompt, including the answer
generator's. That is the mechanism by which ``llm/`` stays free of domain text
(see ``app.llm.prompts.register_prompt``), and registering all of them here
rather than on first use keeps the prompt registry's contents a fact about the
code rather than about which module happened to be imported first.
"""

from app.grading.answergen import (
    ANSWER_SYNTHESIS_PROMPT_V1,
    SyntheticAnswer,
    generate_answer,
    split_concepts,
)
from app.grading.evalset import (
    LABEL_ANCHORS,
    QUALITY_LEVELS,
    TARGET_ANSWERS,
    TARGET_QUESTIONS,
    EvalAnswer,
    labelling_order,
    stratified_questions,
    validate_dataset,
)
from app.grading.evidence import (
    EvidenceCheck,
    EvidenceMatch,
    VerifiedJudgment,
    verify_evidence,
    verify_span,
)
from app.grading.grader import (
    GradingContractError,
    GradingInputError,
    build_grading_inputs,
    check_judgment,
    graded_answer_text,
    judge_answer,
)
from app.grading.pipeline import (
    ConfidenceGateOutcome,
    GradeResult,
    gate_triggers,
    grade_answer,
)
from app.grading.prompt import (
    GRADING_PROMPT_V1,
    GRADING_RECHECK_PROMPT_V1,
    isolate_candidate_text,
)
from app.grading.rubric import RUBRIC_DIMENSIONS, RUBRIC_KEYS, render_rubric
from app.grading.schema import (
    AnswerError,
    ConceptJudgment,
    ConceptLabel,
    ErrorSeverity,
    GradeJudgment,
    RubricScores,
)
from app.grading.scoring import ScoreBreakdown, ScoringError, score_answer

__all__ = [
    "ANSWER_SYNTHESIS_PROMPT_V1",
    "GRADING_PROMPT_V1",
    "GRADING_RECHECK_PROMPT_V1",
    "LABEL_ANCHORS",
    "QUALITY_LEVELS",
    "RUBRIC_DIMENSIONS",
    "RUBRIC_KEYS",
    "TARGET_ANSWERS",
    "TARGET_QUESTIONS",
    "AnswerError",
    "ConceptJudgment",
    "ConceptLabel",
    "ConfidenceGateOutcome",
    "ErrorSeverity",
    "EvalAnswer",
    "EvidenceCheck",
    "EvidenceMatch",
    "GradeJudgment",
    "GradeResult",
    "GradingContractError",
    "GradingInputError",
    "RubricScores",
    "ScoreBreakdown",
    "SyntheticAnswer",
    "ScoringError",
    "VerifiedJudgment",
    "build_grading_inputs",
    "check_judgment",
    "gate_triggers",
    "generate_answer",
    "grade_answer",
    "graded_answer_text",
    "isolate_candidate_text",
    "judge_answer",
    "labelling_order",
    "render_rubric",
    "score_answer",
    "split_concepts",
    "stratified_questions",
    "validate_dataset",
    "verify_evidence",
    "verify_span",
]
