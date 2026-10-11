"""PII redaction for text on its way to a language model (Day 24).

Plan section 14.3: "redaction pass (``[NAME]``/``[EMAIL]``/``[PHONE]``) before
anything reaches an LLM" - mandatory rather than optional because free model
tiers may use requests for training.

**What this detects** (deterministic regular expressions, no model):

* ``EMAIL``        ``local@domain.tld``
* ``PHONE``        international numbers with a ``+`` country code; North-American
                   ``(555) 123-4567`` / ``555.123.4567`` shapes; ten-digit Indian
                   mobiles (``98765 43210``); and any 10-15 digit number directly
                   after a label such as ``Phone:``, ``Tel``, ``Mobile``
* ``PROFILE_URL``  links to personal profiles: LinkedIn, GitHub, GitLab, X/Twitter,
                   Facebook, Instagram, Medium ``@`` pages, Stack Overflow users
* ``URL``          any ``http(s)://`` or ``www.`` link (documents only - a resume's
                   project links identify the person)
* ``NAME``         **resumes only, heuristic**: the first line when it looks like a
                   name (2-4 capitalised words, no digits, no role word such as
                   "Engineer"), and every later occurrence of that exact string

**What it does not guarantee - read this before relying on it:**

* Names anywhere other than a resume's heading (in prose, in a signature, a
  surname on its own) are not found.  General name detection needs a trained
  model and is wrong often enough to damage technical text.
* Postal addresses, dates of birth, national ID numbers, and phone numbers in
  shapes not listed above are not detected.
* It is pattern-based, so obfuscations (``name [at] domain [dot] com``) pass.

So the guarantee is narrow and testable: *the listed shapes are replaced*.  It
is a reduction of exposure, not anonymisation.

**What it is careful not to touch:** IP addresses, version numbers, year ranges,
latencies, ports, Big-O, and epoch timestamps - each pinned by a test - because
redacting technical content would change what a grader or extractor sees.

Redaction and prompt-injection defence are separate: a redacted resume can
still carry "ignore all instructions".  That is handled by fencing the text as
data (``app.grading.prompt.isolate_candidate_text``), not here.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum


class PiiKind(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    PROFILE_URL = "profile_url"
    URL = "url"
    NAME = "name"


MASKS: dict[PiiKind, str] = {
    PiiKind.EMAIL: "[EMAIL]",
    PiiKind.PHONE: "[PHONE]",
    PiiKind.PROFILE_URL: "[URL]",
    PiiKind.URL: "[URL]",
    PiiKind.NAME: "[NAME]",
}

#: For resumes: everything.
DOCUMENT_KINDS = frozenset(PiiKind)
#: For job descriptions: no name heuristic.  A JD's first line is a job title
#: ("Backend Engineer"), which the heuristic would destroy - found by test.
JD_KINDS = DOCUMENT_KINDS - {PiiKind.NAME}
#: For interview answers on their way to the grader or the probe writer:
#: contact details only.  General URLs and the name heuristic stay off, because
#: an answer about an API may legitimately contain ``https://api.example.com``,
#: and an answer has no heading.
ANSWER_KINDS = frozenset({PiiKind.EMAIL, PiiKind.PHONE, PiiKind.PROFILE_URL})

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

_PROFILE_URL = re.compile(
    r"(?i)\b(?:https?://)?(?:www\.)?"
    r"(?:linkedin\.com/(?:in|pub)/|github\.com/|gitlab\.com/|(?:twitter|x)\.com/|"
    r"facebook\.com/|instagram\.com/|medium\.com/@|stackoverflow\.com/users/)"
    r"[^\s,;)\]>\"']+"
)
_URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s,;)\]>\"']+")

# Phone shapes.  Each candidate is accepted only if it has 10-15 digits.
_PHONE_PATTERNS = (
    # +CC then digit groups: +1 (555) 123-4567, +91 98765 43210, +44 20 7946 0958
    re.compile(r"(?<![\w+])\+\d[\d\s().\-]{8,22}\d(?![\w])"),
    # North America without a country code: (555) 123-4567, 555-123-4567, 555.123.4567
    re.compile(r"(?<![\w.\-])\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?![\w.\-])"),
    # Indian mobile: 98765 43210 / 98765-43210 / 9876543210
    re.compile(r"(?<![\w.+\-])[6-9]\d{4}[\s\-]?\d{5}(?![\w.\-])"),
)
_LABELLED_PHONE = re.compile(
    r"(?i)\b(?:phone|tel|telephone|mobile|cell|ph|whatsapp)\b\s*(?:no\.?|number|#)?\s*[:.\-]?\s*"
    r"(\+?\d[\d\s().\-]{7,22}\d)"
)

_NAME_TOKEN = re.compile(r"^(?:[A-Z][a-z'\-]+|[A-Z]{2,}|[A-Z]\.)$")
#: First lines that look like two capitalised words but are not names.
_NOT_A_NAME = frozenset(
    {"curriculum vitae", "resume", "résumé", "professional summary", "contact information"}
)
#: A heading containing one of these is a job title, not a name.
_ROLE_WORDS = frozenset(
    {
        "engineer",
        "developer",
        "manager",
        "scientist",
        "analyst",
        "architect",
        "intern",
        "designer",
        "consultant",
        "lead",
        "senior",
        "junior",
        "principal",
        "staff",
        "administrator",
        "programmer",
        "specialist",
        "director",
        "officer",
        "head",
        "backend",
        "frontend",
        "fullstack",
        "software",
        "data",
        "devops",
        "sre",
    }
)


def _digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


@dataclass(frozen=True, slots=True)
class Redaction:
    text: str
    #: How many of each kind were replaced.  Counts only - never the values.
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _heading_name(text: str) -> str | None:
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.lower() in _NOT_A_NAME or len(candidate) > 60:
            return None
        tokens = candidate.split()
        if any(t.lower().strip(".,") in _ROLE_WORDS for t in tokens):
            return None
        if 2 <= len(tokens) <= 4 and all(_NAME_TOKEN.match(t) for t in tokens):
            return candidate
        return None
    return None


def redact(text: str, kinds: Iterable[PiiKind] = DOCUMENT_KINDS) -> Redaction:
    """Replace every detected identifier of the requested kinds.  Pure."""
    wanted = frozenset(kinds)
    counts: Counter[str] = Counter()
    out = text

    def sub(pattern: re.Pattern[str], kind: PiiKind, s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            counts[kind.value] += 1
            return MASKS[kind]

        return pattern.sub(repl, s)

    if PiiKind.NAME in wanted:
        name = _heading_name(out)
        if name:
            out, n = re.subn(re.escape(name), MASKS[PiiKind.NAME], out, flags=re.IGNORECASE)
            counts[PiiKind.NAME.value] += n
    if PiiKind.EMAIL in wanted:
        out = sub(_EMAIL, PiiKind.EMAIL, out)
    if PiiKind.PROFILE_URL in wanted:
        out = sub(_PROFILE_URL, PiiKind.PROFILE_URL, out)
    if PiiKind.URL in wanted:
        out = sub(_URL, PiiKind.URL, out)
    if PiiKind.PHONE in wanted:

        def labelled(m: re.Match[str]) -> str:
            number = m.group(1)
            if not 10 <= _digits(number) <= 15:
                return m.group(0)
            counts[PiiKind.PHONE.value] += 1
            return m.group(0)[: m.start(1) - m.start(0)] + MASKS[PiiKind.PHONE]

        out = _LABELLED_PHONE.sub(labelled, out)
        for pattern in _PHONE_PATTERNS:

            def bare(m: re.Match[str]) -> str:
                if not 10 <= _digits(m.group(0)) <= 15:
                    return m.group(0)
                counts[PiiKind.PHONE.value] += 1
                return MASKS[PiiKind.PHONE]

            out = pattern.sub(bare, out)
    return Redaction(out, dict(sorted(counts.items())))


__all__ = [
    "ANSWER_KINDS",
    "DOCUMENT_KINDS",
    "JD_KINDS",
    "MASKS",
    "PiiKind",
    "Redaction",
    "redact",
]
