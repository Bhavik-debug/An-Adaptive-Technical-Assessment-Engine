"""PII redaction: what it replaces, and what technical text it must leave alone."""

from __future__ import annotations

import pytest

from app.privacy import ANSWER_KINDS, DOCUMENT_KINDS, MASKS, PiiKind, redact


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("priya.sharma@gmail.com", "email"),
        ("first.last+tag@sub.example.co.uk", "email"),
        ("+1 (555) 123-4567", "phone"),
        ("+91 98765 43210", "phone"),
        ("+44 20 7946 0958", "phone"),
        ("(415) 555-0100", "phone"),
        ("555.123.4567", "phone"),
        ("98765-43210", "phone"),
        ("9876543210", "phone"),
        ("linkedin.com/in/priya-s", "profile_url"),
        ("https://github.com/priyas", "profile_url"),
        ("https://priya.dev/portfolio", "url"),
        ("www.priya-sharma.me", "url"),
    ],
)
def test_each_supported_shape_is_replaced(raw, kind):
    out = redact(f"Contact: {raw} today.")
    assert raw not in out.text
    assert out.counts == {kind: 1}
    assert MASKS[PiiKind(kind)] in out.text


@pytest.mark.parametrize("label", ["Phone:", "Tel", "Mobile -", "Cell #", "WhatsApp:"])
def test_labelled_numbers(label):
    out = redact(f"{label} 020 7946 0958 2")
    assert out.counts.get("phone") == 1 and "7946" not in out.text


def test_multiple_identifiers_in_one_document():
    doc = (
        "Priya Sharma\n"
        "priya@x.io | backup: p.sharma@y.org | +91 98765 43210 | (415) 555-0100\n"
        "linkedin.com/in/priya | https://priya.dev\n"
        "Priya Sharma built a cache in front of Postgres."
    )
    out = redact(doc)
    assert out.counts == {"email": 2, "name": 2, "phone": 2, "profile_url": 1, "url": 1}
    for secret in ("Priya", "priya@x.io", "p.sharma", "98765", "555-0100", "priya.dev"):
        assert secret not in out.text
    assert "built a cache in front of Postgres" in out.text


@pytest.mark.parametrize(
    "technical",
    [
        "Redis 7.2.4 on 10.0.0.1 and 192.168.100.123, port 5432",
        "2019-2021 at Acme; 2019 - 2023 freelance",
        "O(n log n) sort, p99 of 250 ms at 10,000 RPS",
        "epoch 1696000000 and HTTP 404, Python 3.11.4",
        "a 1234567890-byte blob and 1 2 3 4 5 6 7 8 9 0",
        "use x-request-id: 4f2c and SHA 3a1b9e0",
    ],
)
def test_technical_text_is_untouched(technical):
    out = redact(technical)
    assert out.text == technical and out.counts == {}


def test_answers_keep_api_urls_but_lose_contact_details():
    answer = "POST to https://api.example.com/v1/orders; mail me at a@b.io or github.com/me"
    out = redact(answer, ANSWER_KINDS)
    assert "https://api.example.com/v1/orders" in out.text
    assert "a@b.io" not in out.text and "github.com/me" not in out.text


def test_answers_have_no_name_heuristic():
    out = redact("Consistent Hashing\nspreads keys across nodes", ANSWER_KINDS)
    assert out.text.startswith("Consistent Hashing")


@pytest.mark.parametrize(
    "first_line",
    ["Curriculum Vitae", "Senior backend engineer with 6 years", "SKILLS: Python, Go", ""],
)
def test_the_name_heuristic_is_conservative(first_line):
    out = redact(f"{first_line}\nWorked on payments.", DOCUMENT_KINDS)
    assert "name" not in out.counts


@pytest.mark.parametrize("raw", ["", "   ", "\x00\x01", "🙂 ünïcødé ✓", "@@@ +++ ((( )))"])
def test_malformed_and_odd_inputs_do_not_break(raw):
    out = redact(raw)
    assert isinstance(out.text, str)


def test_deterministic_and_idempotent():
    doc = "Ann Lee\nann@lee.io +1 555 123 4567"
    first = redact(doc)
    assert redact(doc) == first
    assert redact(first.text).text == first.text  # masks are not themselves PII


def test_counts_never_contain_values():
    out = redact("x@y.io and +1 555 123 4567")
    assert set(out.counts) <= {k.value for k in PiiKind}
    assert all(isinstance(v, int) for v in out.counts.values())


@pytest.mark.parametrize(
    "heading", ["Backend Engineer", "Senior Software Developer", "Data Scientist", "Staff SRE"]
)
def test_a_job_title_heading_is_not_a_name(heading):
    """Regression (found on Day 24): "Backend Engineer" was redacted as [NAME]."""
    out = redact(f"{heading}\nBuilt payment systems.")
    assert out.text.startswith(heading) and "name" not in out.counts


def test_job_descriptions_never_use_the_name_heuristic():
    from app.privacy import JD_KINDS

    out = redact("Platform Lead Wanted\nMust know Kafka. Contact jobs@acme.io", JD_KINDS)
    assert out.text.startswith("Platform Lead Wanted")
    assert out.counts == {"email": 1}
