"""The anchored rubric - plan section 7.2, and section 7.4 technique 4.

Plan section 7.4 ranks "anchored rubric levels" fourth among the reliability
techniques, and states the rule it exists to enforce:

    Every 0-4 level has a written descriptor and one example answer in the
    prompt.  Never "rate depth 1-10."

**Why anchors, and why 0-4 rather than 0-10.**  An unanchored scale is a
different scale on every call: "rate depth out of 10" clusters at 7-8 because
the model has no shared referent for what a 4 or a 9 would look like, so it
retreats to the safe middle (plan section 7.1 lists that clustering first among
the reasons the obvious design fails).  A descriptor plus a worked example
replaces the model's private notion of "depth" with a *classification against
text it can read* - the same move plan section 7.4 technique 1 makes for
concepts, applied to the qualities a concept checklist cannot see.  Five levels
rather than eleven because an author can genuinely write five distinguishable
descriptors for one dimension and cannot write eleven, and a level nobody can
describe is a level nobody can apply twice the same way.

**What these four dimensions are for.**  They are deliberately *not* a second
opinion about correctness - that is what ``expected_concepts`` measures, and
plan section 7.2 gives concept coverage the majority weight for exactly that
reason.  These catch what a checklist cannot: a candidate who names every
expected concept in a disorganised, unqualified, generic ramble should not score
the same as one who explains them coherently.

**Where the text ends up, and why that matters.**  ``render_rubric()`` is folded
into the *system* message of the grading prompt at import time rather than
passed in as a template input.  ``PromptTemplate.fingerprint`` hashes the
template text, so a rubric embedded in the template is covered by the
fingerprint - and editing a descriptor here therefore changes the prompt
fingerprint, the response-cache key and the fixture key.  A rubric passed in as
a *variable* would be invisible to all three, and a quality regression caused by
a reworded anchor would be unattributable, which is the one failure the
fingerprint exists to prevent.

This module owns rubric text and nothing else.  It imports nothing from ``app``,
which is what keeps it a leaf: the grading package may depend on the LLM
chokepoint, never the other way round (plan section 13.2).
"""

from __future__ import annotations

from dataclasses import dataclass

#: The scale, stated once. Both ends are inclusive and both are reachable: 0 is
#: a real classification ("none of this is present"), not an error code.
MIN_LEVEL = 0
MAX_LEVEL = 4


@dataclass(frozen=True, slots=True)
class RubricLevel:
    """One anchored point on one dimension.

    ``descriptor`` says what the level *means*; ``example`` shows one answer
    fragment that sits at it.  Both are required, because a descriptor without
    an example is an adjective and an example without a descriptor is an
    anecdote - it is the pair that pins the level down.
    """

    level: int
    descriptor: str
    example: str


@dataclass(frozen=True, slots=True)
class RubricDimension:
    """One of the four things the rubric asks about.

    ``key`` is also the field name in :class:`app.grading.schema.RubricScores`,
    so the prompt and the output contract cannot drift apart without a test
    noticing.  ``question`` is the single question the dimension answers,
    written so that a grader who read nothing else would still know what is
    being judged.
    """

    key: str
    question: str
    levels: tuple[RubricLevel, ...]


_STRUCTURE = RubricDimension(
    key="structure",
    question="Is the answer organised around the problem, or is it a pile of points?",
    levels=(
        RubricLevel(
            level=0,
            descriptor=(
                "No discernible organisation: one undifferentiated statement, or a "
                "restatement of the question."
            ),
            example='"Caching is important for performance at scale."',
        ),
        RubricLevel(
            level=1,
            descriptor=(
                "A single relevant point, asserted with no framing and nothing built "
                "on top of it."
            ),
            example='"You would put Redis in front of the database."',
        ),
        RubricLevel(
            level=2,
            descriptor=(
                "Several relevant points, listed rather than ordered. No stated "
                "approach ties them together."
            ),
            example='"Use Redis. Set a TTL. Stale data is a risk. LRU is an option."',
        ),
        RubricLevel(
            level=3,
            descriptor=(
                "A stated approach that the points hang off, with a recognisable "
                "beginning and end."
            ),
            example=(
                '"I would start with read-through caching, then decide invalidation, '
                'then eviction. On reads, ..."'
            ),
        ),
        RubricLevel(
            level=4,
            descriptor=(
                "Organised around the problem's own structure, and the ordering is "
                "justified rather than merely present."
            ),
            example=(
                '"Two questions decide this: what invalidates an entry, and what '
                "happens on a miss storm. Invalidation first, because the eviction "
                'policy depends on the answer. ..."'
            ),
        ),
    ),
)

_TRADEOFF_AWARENESS = RubricDimension(
    key="tradeoff_awareness",
    question="Does the answer acknowledge what its own choices cost?",
    levels=(
        RubricLevel(
            level=0,
            descriptor=(
                "No cost, limit or downside is acknowledged anywhere. Choices are "
                "presented as facts."
            ),
            example='"We use write-through caching."',
        ),
        RubricLevel(
            level=1,
            descriptor=(
                "A downside is named in the abstract, unattached to any choice this "
                "answer actually made."
            ),
            example='"Caches can serve stale data."',
        ),
        RubricLevel(
            level=2,
            descriptor="A downside of one specific choice this answer made is named.",
            example='"Write-through costs latency on every write."',
        ),
        RubricLevel(
            level=3,
            descriptor=(
                "At least two options are compared on a named axis, and one is chosen "
                "with a reason."
            ),
            example=(
                '"Write-behind is faster but loses the window on a crash; write-through '
                'is slower and safe. I would take write-through here."'
            ),
        ),
        RubricLevel(
            level=4,
            descriptor=(
                "Options are compared and the answer states the condition under which "
                "its own choice would flip."
            ),
            example=(
                '"Write-through below roughly 10k writes/s. Above that the write '
                "latency dominates and I would move to write-behind and accept the "
                'window."'
            ),
        ),
    ),
)

_SPECIFICITY = RubricDimension(
    key="specificity",
    question=(
        "Is the answer about this problem, or about the topic in general? Length is "
        "never specificity: a long answer that stays generic sits at 0."
    ),
    levels=(
        RubricLevel(
            level=0,
            descriptor=(
                "Generic statements that would be equally true of any system. Padding, "
                "repetition and restated question text count as nothing."
            ),
            example=(
                '"You should follow best practices, monitor everything and design for ' 'scale."'
            ),
        ),
        RubricLevel(
            level=1,
            descriptor="Correct terminology, but no mechanism, quantity or named technology.",
            example='"We would handle cache invalidation and eviction properly."',
        ),
        RubricLevel(
            level=2,
            descriptor=(
                "Concrete mechanisms or technologies are named, without any quantity, "
                "bound or failure mode."
            ),
            example='"Redis with an LRU eviction policy and a TTL on each key."',
        ),
        RubricLevel(
            level=3,
            descriptor=(
                "Mechanisms are named and at least one concrete quantity, bound or "
                "failure mode is attached to them."
            ),
            example=(
                '"A 60-second TTL, LRU at a 4 GB cap; on a cold start every key misses ' 'at once."'
            ),
        ),
        RubricLevel(
            level=4,
            descriptor=(
                "Mechanisms, quantities, and the consequence worked through to "
                "something that would actually be observed."
            ),
            example=(
                '"A 60-second TTL at 10k reads/s means about 600k origin reads per '
                'expiry wave, so I would jitter the TTL by 10% to spread it."'
            ),
        ),
    ),
)

_COMMUNICATION = RubricDimension(
    key="communication",
    question="Could a colleague act on this answer without asking what it meant?",
    levels=(
        RubricLevel(
            level=0,
            descriptor="Incoherent, self-contradictory, or impossible to follow.",
            example='"Cache the database so the cache is the database, unless not."',
        ),
        RubricLevel(
            level=1,
            descriptor=(
                "Followable only by guessing what was meant; key referents are missing "
                "or ambiguous."
            ),
            example='"You just put the thing in front of it and it handles the rest."',
        ),
        RubricLevel(
            level=2,
            descriptor=(
                "Understandable, with some ambiguity or jargon used without being " "pinned down."
            ),
            example='"We front it with a cache layer and keep it eventually consistent."',
        ),
        RubricLevel(
            level=3,
            descriptor=(
                "Clear and unambiguous throughout; terms are used correctly and defined "
                "where they carry weight."
            ),
            example=(
                '"By eventually consistent I mean a reader may see a value up to one TTL '
                'old, never an older one."'
            ),
        ),
        RubricLevel(
            level=4,
            descriptor=(
                "Clear, and calibrated: it marks what it is unsure about instead of "
                "asserting everything at one confidence."
            ),
            example=(
                '"I am confident about the invalidation path. I am less sure whether the '
                'stampede lock should be per-key or per-shard - I would measure."'
            ),
        ),
    ),
)

#: The four dimensions of plan section 7.2, in the order the prompt shows them.
#: The order is fixed rather than incidental: it is part of the prompt text and
#: therefore part of the prompt fingerprint.
RUBRIC_DIMENSIONS: tuple[RubricDimension, ...] = (
    _STRUCTURE,
    _TRADEOFF_AWARENESS,
    _SPECIFICITY,
    _COMMUNICATION,
)

#: The contract's field names, in prompt order. Used by the schema's own test to
#: assert the two never drift apart.
RUBRIC_KEYS: tuple[str, ...] = tuple(dimension.key for dimension in RUBRIC_DIMENSIONS)


def render_rubric(dimensions: tuple[RubricDimension, ...] = RUBRIC_DIMENSIONS) -> str:
    """The rubric as prompt text: every dimension, every level, both anchors.

    Plain text rather than JSON on purpose. This block is read by the model as
    *instructions*, and JSON in this prompt is reserved for the *output*
    contract, so the two can never be mistaken for one another.
    """
    blocks: list[str] = []
    for dimension in dimensions:
        lines = [f"### {dimension.key}", dimension.question, ""]
        for level in dimension.levels:
            lines.append(f"- {level.level} - {level.descriptor}")
            lines.append(f"  Example at this level: {level.example}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = [
    "MAX_LEVEL",
    "MIN_LEVEL",
    "RUBRIC_DIMENSIONS",
    "RUBRIC_KEYS",
    "RubricDimension",
    "RubricLevel",
    "render_rubric",
]
