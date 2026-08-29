"""The routing table and the prompt registry.

These are configuration expressed as code, so the tests are mostly invariants:
things that must stay true as tasks and prompts are added in later phases.
"""

from __future__ import annotations

import pytest

from app.llm.errors import PromptNotRegisteredError
from app.llm.prompts import PROMPTS, PromptTemplate, get_prompt
from app.llm.tasks import TASK_SPECS, TaskName, get_task_spec
from app.llm.types import ModelTier, Role

# --- the routing table -----------------------------------------------------


def test_every_declared_task_has_a_routing_entry():
    """A task without a spec would fail at call time, not at import time."""
    assert set(TASK_SPECS) == set(TaskName)


def test_grading_is_deterministic():
    """The same answer must be graded the same way twice - plan section 5.4."""
    assert get_task_spec(TaskName.GRADE_ANSWER).temperature == 0.0


def test_grading_gets_the_quality_tier():
    assert get_task_spec(TaskName.GRADE_ANSWER).tier is ModelTier.MID


def test_wording_tasks_are_allowed_to_vary():
    assert get_task_spec(TaskName.QUESTION_RENDER).temperature > 0
    assert get_task_spec(TaskName.FOLLOW_UP_PROBE).temperature > 0


def test_extraction_tasks_are_cheap_and_deterministic():
    for task in (TaskName.RESUME_EXTRACTION, TaskName.JD_EXTRACTION):
        spec = get_task_spec(task)
        assert spec.tier is ModelTier.SMALL_FAST
        assert spec.temperature == 0.0


def test_deterministic_tasks_are_cacheable_by_default():
    assert get_task_spec(TaskName.RESUME_EXTRACTION).is_cacheable is True


def test_sampled_tasks_are_not_cacheable():
    assert get_task_spec(TaskName.QUESTION_RENDER).is_cacheable is False


def test_the_confidence_gate_regrade_is_never_cached():
    """It is sampled three times and compared; a cache would fake agreement."""
    spec = get_task_spec(TaskName.GRADE_RECHECK)
    assert spec.temperature > 0
    assert spec.is_cacheable is False


def test_the_probe_is_never_cached():
    assert get_task_spec(TaskName.CONNECTIVITY_PROBE).is_cacheable is False


def test_only_the_agent_task_asks_for_reasoning_today():
    reasoning_tasks = {task for task, spec in TASK_SPECS.items() if spec.reasoning}
    assert reasoning_tasks == {TaskName.DEEP_DIVE_AGENT}


def test_every_task_bounds_its_output():
    for task, spec in TASK_SPECS.items():
        assert spec.max_output_tokens > 0, task
        assert 0.0 <= spec.temperature <= 1.0, task
        assert 0.0 < spec.top_p <= 1.0, task


# --- prompts ---------------------------------------------------------------


def test_the_probe_prompt_is_registered():
    prompt = get_prompt(TaskName.CONNECTIVITY_PROBE)
    assert prompt.version == "v1"


def test_a_task_whose_phase_has_not_arrived_says_so():
    for task in TaskName:
        if task in PROMPTS:
            continue
        with pytest.raises(PromptNotRegisteredError, match=task.value):
            get_prompt(task)


def test_rendering_produces_a_system_message_then_a_user_message():
    messages = get_prompt(TaskName.CONNECTIVITY_PROBE).render({"token": "abc123"})
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    assert "abc123" in messages[1].content


def test_a_missing_input_is_named():
    with pytest.raises(KeyError, match="token"):
        get_prompt(TaskName.CONNECTIVITY_PROBE).render({})


def test_braces_in_an_input_are_not_interpreted():
    """The reason templates use ``$name`` rather than ``{name}``.

    Prompts in this project carry JSON examples; ``str.format`` would try to
    substitute into every brace in them.
    """
    template = PromptTemplate(
        task=TaskName.CONNECTIVITY_PROBE,
        version="v-test",
        system='Example output: {"ok": true}',
        user="Answer: $answer",
        required_inputs=("answer",),
    )
    messages = template.render({"answer": '{"nested": {"json": 1}}'})
    assert '{"ok": true}' in messages[0].content
    assert '{"nested": {"json": 1}}' in messages[1].content


def test_the_fingerprint_changes_when_the_text_changes():
    original = get_prompt(TaskName.CONNECTIVITY_PROBE)
    edited = PromptTemplate(
        task=original.task,
        version=original.version,  # version deliberately not bumped
        system=original.system + " Be brief.",
        user=original.user,
        required_inputs=original.required_inputs,
    )
    assert edited.fingerprint != original.fingerprint
