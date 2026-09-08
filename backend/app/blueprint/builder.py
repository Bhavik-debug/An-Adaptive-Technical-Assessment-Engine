"""``AssessmentSpec -> Blueprint``.  The arrow in plan section 3's Day 15.

Four derivations, in order, and nothing else:

===========================  =============================================
what                         from
===========================  =============================================
time budget                  ``duration_minutes * 60``
item budget                  ``time_budget / MODE_SECONDS_PER_ITEM[mode]``
topic quotas                 ``allocate_quotas(item_budget, jd_weights, ...)``
cold-start theta             ``LEVEL_TARGET_THETA[level]``
===========================  =============================================

Then :class:`~app.blueprint.blueprint.Blueprint` validates the result, and the
builder does not catch what it raises: a blueprint that cannot be executed must
not be returned, and the caller's next line is a database write.

**No selection happens here.**  The builder never looks at a question, never
scores anything, and never touches an ability estimate.  It divides a budget and
checks the division is possible.
"""

from __future__ import annotations

from app.bank.taxonomy import Taxonomy
from app.blueprint.blueprint import Blueprint, TopicPlan
from app.blueprint.quotas import allocate_quotas
from app.blueprint.spec import (
    MAX_ITEM_BUDGET,
    MIN_ITEM_BUDGET,
    AssessmentSpec,
    BlueprintError,
)


def derive_item_budget(spec: AssessmentSpec) -> int:
    """How many questions fit in the clock, at this mode's pace.

    Floor division, not rounding: an assessment that budgets for a question it
    cannot finish has promised the candidate something the clock will take away,
    and Day 12's time constraint would drop the item anyway - after it had
    already been counted in a quota.

    An explicit ``spec.item_budget`` wins, and the blueprint's own validation
    then checks the clock can hold it.  That ordering is deliberate: a caller
    who states a question count is making a claim, and the right response is to
    check it rather than to quietly overrule it.
    """
    if spec.item_budget is not None:
        return spec.item_budget
    budget = int(spec.time_budget_s // spec.seconds_per_item)
    if budget < MIN_ITEM_BUDGET:
        raise BlueprintError(
            f"a {spec.duration_minutes}-minute {spec.mode} assessment fits {budget} "
            f"question(s) at {spec.seconds_per_item} s each, below the minimum of "
            f"{MIN_ITEM_BUDGET}; lengthen it or choose a faster mode"
        )
    return min(budget, MAX_ITEM_BUDGET)


def check_against_taxonomy(spec: AssessmentSpec, taxonomy: Taxonomy) -> None:
    """Every key in the spec must exist, at the right level, under the right parent.

    Optional, because the blueprint layer is usable without loading the bank -
    the simulation builds blueprints over a synthetic taxonomy that is not in
    ``data/question-bank/``, and requiring the file there would couple an
    experiment to a dataset it does not use.  But when a real taxonomy *is*
    available, a typo in a topic key should fail here rather than as an empty
    candidate pool at question one.
    """
    for request in spec.topics:
        node = taxonomy.get(request.topic_key)
        if node is None:
            raise BlueprintError(f"topic {request.topic_key!r} is not in the taxonomy")
        if node.level != "topic":
            raise BlueprintError(
                f"{request.topic_key!r} is a {node.level} in the taxonomy, not a topic; "
                "quotas are stated at topic level"
            )
        for subtopic in request.subtopic_keys:
            if not taxonomy.is_subtopic_of(subtopic, request.topic_key):
                child = taxonomy.get(subtopic)
                if child is None:
                    raise BlueprintError(f"subtopic {subtopic!r} is not in the taxonomy")
                raise BlueprintError(
                    f"{subtopic!r} is not a subtopic of {request.topic_key!r} "
                    f"(its parent is {child.parent_key!r})"
                )


def resolve_subtopics(spec: AssessmentSpec, taxonomy: Taxonomy) -> dict[str, tuple[str, ...]]:
    """Fill in the subtopics of any topic that did not list them.

    A blueprint needs ``parent_of`` for Day 11's ``roll_up`` and Day 12's
    precision stopping rule, and a topic with no subtopics contributes nothing
    to either.  So when the caller has a taxonomy and left the list empty, the
    honest reading of "cover databases" is "cover all of databases" - and
    filling it in is better than silently producing a blueprint whose precision
    rule can never fire.

    A caller who genuinely wants a subset says so, and their list is kept.
    """
    resolved: dict[str, tuple[str, ...]] = {}
    for request in spec.topics:
        if request.subtopic_keys:
            resolved[request.topic_key] = request.subtopic_keys
            continue
        children = tuple(sorted(node.key for node in taxonomy.children_of(request.topic_key)))
        if not children:
            raise BlueprintError(f"topic {request.topic_key!r} has no subtopics in the taxonomy")
        resolved[request.topic_key] = children
    return resolved


def build_blueprint(spec: AssessmentSpec, *, taxonomy: Taxonomy | None = None) -> Blueprint:
    """Derive the executable contract, or raise :class:`BlueprintError`.

    Deterministic: the same spec always produces an equal blueprint, with topics
    in the spec's own order and quotas from a name-tie-broken division.  Asserted
    by test, because a session's blueprint is written to a database once and
    compared against later.

    The spec is never mutated - it is frozen, and every derived structure here is
    built fresh.
    """
    item_budget = derive_item_budget(spec)

    if taxonomy is not None:
        check_against_taxonomy(spec, taxonomy)
        subtopics = resolve_subtopics(spec, taxonomy)
    else:
        subtopics = {r.topic_key: r.subtopic_keys for r in spec.topics}

    quotas = allocate_quotas(
        item_budget,
        {request.topic_key: request.jd_weight for request in spec.topics},
        minimums={
            request.topic_key: max(request.min_items, 1 if request.required else 0)
            for request in spec.topics
        },
        maximums={request.topic_key: request.max_items for request in spec.topics},
    )

    allocated = sum(quotas.values())
    if allocated < item_budget:
        # Every topic hit its cap before the budget was spent.  Shrinking the
        # budget silently would change what the candidate was told; refusing is
        # the only option that keeps `item_budget` meaning what it says.
        raise BlueprintError(
            f"the per-topic maximums total {allocated} items but the budget is "
            f"{item_budget}; raise a maximum, add a topic, or shorten the assessment"
        )

    return Blueprint(
        role=spec.role.strip(),
        level=spec.level,
        mode=spec.mode,
        item_budget=item_budget,
        time_budget_s=spec.time_budget_s,
        topics=tuple(
            TopicPlan(
                topic_key=request.topic_key,
                quota=quotas[request.topic_key],
                jd_weight=request.jd_weight,
                required=request.required,
                subtopic_keys=tuple(subtopics[request.topic_key]),
            )
            for request in spec.topics
        ),
        target_theta=spec.target_theta,
        notes=spec.notes,
    )


__all__ = [
    "build_blueprint",
    "check_against_taxonomy",
    "derive_item_budget",
    "resolve_subtopics",
]
