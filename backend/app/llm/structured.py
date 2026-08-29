"""Turning a pydantic model into a contract the model must satisfy.

Three jobs live here, in the order they happen:

1. **Describe** - reduce a pydantic class to a JSON Schema the provider can be
   asked to enforce, and hardened so "close enough" is not accepted.
2. **Extract** - pull a JSON object out of whatever the model actually sent,
   because even a well-behaved model sometimes wraps it in a code fence.
3. **Validate** - run it through pydantic.  This is the only gate; a provider
   claiming it enforced the schema is not evidence that it did.

That third point is the load-bearing one.  Provider-side constrained decoding is
an optimisation - it makes the first attempt succeed more often.  Validation is
the guarantee.  The system's correctness must not depend on a vendor flag.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.types import ChatMessage, JsonSchemaSpec, Role

#: How much of a bad response we quote back to the model on a repair attempt.
#: Enough to be recognisable, bounded so a runaway generation cannot double the
#: size of the retry prompt.
_MAX_ECHOED_OUTPUT_CHARS = 2000


def _harden(node: Any) -> Any:
    """Make every object in a JSON Schema closed and fully required.

    Providers that support strict structured output require this shape; more
    importantly, an open schema lets a model return the four fields we asked
    for *plus* six it invented, and "extra keys are fine" is how a typo in a
    field name silently becomes a missing value.
    """
    if isinstance(node, list):
        return [_harden(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {key: _harden(value) for key, value in node.items()}
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        properties: dict[str, Any] = out["properties"]
        out["additionalProperties"] = False
        # Optional fields are expressed as nullable rather than absent: a model
        # that may omit a key produces two different shapes for one meaning.
        out["required"] = list(properties)
    return out


def schema_spec(model: type[BaseModel]) -> JsonSchemaSpec:
    """Derive the provider-facing schema for a pydantic model."""
    raw = model.model_json_schema()
    hardened: dict[str, Any] = _harden(raw)
    canonical = json.dumps(hardened, sort_keys=True, separators=(",", ":"))
    return JsonSchemaSpec(
        name=model.__name__,
        schema=hardened,
        fingerprint=hashlib.sha256(canonical.encode()).hexdigest()[:16],
    )


def schema_instruction(spec: JsonSchemaSpec) -> str:
    """The schema, restated in the prompt.

    Belt and braces.  Even when the provider enforces the schema at decode time
    we still say it in words, because the instruction is what makes the model
    *choose* sensible values, while the enforcement only constrains the shape.
    """
    return (
        "Respond with a single JSON object and nothing else - no prose, no "
        "markdown fences, no explanation before or after. It must validate "
        f"against this JSON Schema:\n\n{json.dumps(spec.schema, indent=2)}"
    )


class JsonExtractionError(ValueError):
    """No JSON value could be found in the model's output."""


def extract_json(text: str) -> Any:
    """Recover the JSON value from a model response.

    Handles, in order: a clean object; a fenced ```json block; an object with
    prose either side of it.  Anything else is an extraction failure, which the
    caller turns into a repair attempt.
    """
    candidate = text.strip()
    if not candidate:
        raise JsonExtractionError("the model returned an empty response")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    if "```" in candidate:
        fenced = _strip_code_fence(candidate)
        if fenced is not None:
            try:
                return json.loads(fenced)
            except json.JSONDecodeError:
                candidate = fenced

    value = _first_json_value(candidate)
    if value is None:
        raise JsonExtractionError("no JSON object or array found in the response")
    return value


def _strip_code_fence(text: str) -> str | None:
    start = text.find("```")
    if start == -1:
        return None
    # Skip the fence and any language tag on the same line.
    after = text[start + 3 :]
    newline = after.find("\n")
    if newline == -1:
        return None
    body = after[newline + 1 :]
    end = body.find("```")
    return body[:end].strip() if end != -1 else body.strip()


def _first_json_value(text: str) -> Any | None:
    """Scan for the first position where a complete JSON value starts."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        return value
    return None


def format_validation_error(exc: ValidationError) -> str:
    """A short, model-readable summary of what was wrong.

    This string goes back into the retry prompt, so it is written for the model,
    not for a log reader: field path, then what was expected.
    """
    parts = []
    for err in exc.errors()[:8]:
        location = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def repair_messages(
    *, raw_output: str, problem: str, spec: JsonSchemaSpec
) -> tuple[ChatMessage, ...]:
    """The two messages appended for a schema-repair retry.

    We show the model its own output and name the defect, rather than simply
    re-asking.  Re-asking an identical prompt usually reproduces an identical
    mistake; showing the mistake is what makes the second attempt different.
    """
    echoed = raw_output[:_MAX_ECHOED_OUTPUT_CHARS]
    if len(raw_output) > _MAX_ECHOED_OUTPUT_CHARS:
        echoed += "\n... [truncated]"
    return (
        ChatMessage(role=Role.ASSISTANT, content=echoed),
        ChatMessage(
            role=Role.USER,
            content=(
                f"That response was rejected: {problem}\n\n"
                "Return the corrected JSON object only. No prose, no markdown "
                f"fences.\n\n{json.dumps(spec.schema, indent=2)}"
            ),
        ),
    )
