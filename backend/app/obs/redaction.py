"""Keeping secrets and candidate data out of the logs.

Observability and confidentiality pull in opposite directions: the more a log
line says, the easier the incident is to diagnose and the worse the leak is when
the log is shipped somewhere.  This module is where that tension is resolved,
once, so that no call site has to remember.

Three independent mechanisms, because each catches what the others miss:

1. **Key-name rules.**  A field called ``password``, ``authorization`` or
   ``refresh_token`` is replaced by its name alone, whatever it contains.
2. **Value-shape rules.**  A string that *looks* like a credential - a JWT, an
   ``nvapi-`` key, a ``Bearer`` header, an argon2 hash - is masked wherever it
   appears, including in the middle of a free-text message.
3. **Known literals.**  At boot the redactor is told this process's actual
   secrets (``SECRET_KEY``, ``NVIDIA_API_KEY``).  Any log line containing one
   of those exact strings has it removed.  This is the backstop that catches a
   secret logged under a name nobody thought of.

Candidate email addresses are masked too.  Plan section 14.1 requires a
redaction pass before anything reaches an LLM; a log file is the same class of
exposure with a longer retention, so the same rule applies here.

The redactor is deliberately conservative about what it *replaces* rather than
what it *drops*: a masked value still shows that a field was present, which is
usually the diagnostic fact you needed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"
EMAIL_MASK = "[EMAIL]"

#: Substrings that make a field name sensitive regardless of its value.
SENSITIVE_KEY_PARTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "cookie",
    "credential",
    "private_key",
    "session_key",
    "hash",
)

#: Field names that contain one of the parts above but are not sensitive.
#: A name ending in ``tokens`` is a *count* - ``input_tokens``,
#: ``llm.output_tokens`` - and those counts are the cost model.  Redacting them
#: would delete exactly the numbers plan section 14.2 calls non-negotiable.
SENSITIVE_KEY_EXCEPTIONS: frozenset[str] = frozenset({"hash_algorithm", "token_type"})
_NON_SENSITIVE_SUFFIXES: tuple[str, ...] = ("tokens",)

#: Shapes that are credentials wherever they appear.  Order matters: the broad
#: ``Bearer <...>`` rule runs before the JWT rule so a bearer header is masked
#: as a whole rather than leaving the scheme and eating the token.
_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # `Authorization: Bearer eyJ...`
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"), f"Bearer {REDACTED}"),
    # NVIDIA build keys.
    (re.compile(r"\bnvapi-[A-Za-z0-9_\-]{8,}"), REDACTED),
    # OpenAI-style keys, and anything else that copied the convention.
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), REDACTED),
    # A JWT: three base64url segments, the first of which decodes to `{"`.
    (re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}"), REDACTED),
    # An argon2 password hash, in any of its variants.
    (re.compile(r"\$argon2[a-z0-9]*\$[^\s\"']+"), REDACTED),
    # Candidate PII. Last, so a credential containing an @ is caught first.
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), EMAIL_MASK),
)

#: Below this length, an "exact secret" match would fire on ordinary words.
_MIN_LITERAL_SECRET_LEN = 8

#: How far into nested structures to walk before giving up. Log payloads in this
#: project are flat; the bound just stops a cyclic structure from hanging a log
#: call, which would be an outage caused by observability.
_MAX_DEPTH = 4


def is_sensitive_key(key: str) -> bool:
    """True when a field's *name* alone is enough to redact it."""
    lowered = key.lower()
    if lowered in SENSITIVE_KEY_EXCEPTIONS or lowered.endswith(_NON_SENSITIVE_SUFFIXES):
        return False
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


class Redactor:
    """Applies the three mechanisms above. Cheap enough to run on every record."""

    def __init__(self, literal_secrets: Iterable[str] = ()) -> None:
        self._literals: list[str] = []
        for secret in literal_secrets:
            self.add_secret(secret)

    def add_secret(self, value: str | None) -> None:
        """Register a literal this process must never print.

        Longest first, so a secret that contains another one is masked whole.
        """
        if not value or len(value) < _MIN_LITERAL_SECRET_LEN:
            return
        if value in self._literals:
            return
        self._literals.append(value)
        self._literals.sort(key=len, reverse=True)

    def text(self, value: str) -> str:
        """Mask every credential shape and known literal inside a string."""
        for literal in self._literals:
            if literal in value:
                value = value.replace(literal, REDACTED)
        for pattern, replacement in _VALUE_PATTERNS:
            value = pattern.sub(replacement, value)
        return value

    def value(self, key: str, value: Any, *, _depth: int = 0) -> Any:
        """Redact one field, by name and then by content."""
        if is_sensitive_key(key):
            return REDACTED
        return self.any(value, _depth=_depth)

    def any(self, value: Any, *, _depth: int = 0) -> Any:
        """Redact a value of unknown shape."""
        if _depth > _MAX_DEPTH:
            return REDACTED
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return self.mapping(value, _depth=_depth + 1)
        if isinstance(value, list | tuple | set):
            return [self.any(item, _depth=_depth + 1) for item in value]
        return value

    def mapping(self, data: Mapping[str, Any], *, _depth: int = 0) -> dict[str, Any]:
        return {str(k): self.value(str(k), v, _depth=_depth) for k, v in data.items()}

    def args(self, args: Any) -> Any:
        """Redact the ``%``-style arguments of a log record.

        ``logging`` accepts either a tuple of positionals or a single mapping;
        both are handled so that ``log.info("hello %s", email)`` is as safe as
        the f-string version.
        """
        if isinstance(args, Mapping):
            return self.mapping(args)
        if isinstance(args, Sequence) and not isinstance(args, str | bytes):
            return tuple(self.any(item) for item in args)
        if args is None:
            return None
        return self.any(args)
