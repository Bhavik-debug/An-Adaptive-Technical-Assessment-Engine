"""Turning an uploaded resume into plain text - safely.

Plan section 14.3, "Insecure upload: polyglot PDF, zip bomb, 2 GB file" ->
"5 MB cap, **magic-byte sniffing** (not extension), parse in a worker with 20 s
timeout + memory cap, filenames regenerated as UUIDs, never render user files".

Supported formats, and why only these two:

* **PDF** - recognised by its ``%PDF-`` signature in the first bytes, never by a
  filename or ``Content-Type`` (both are client-controlled).
* **Plain UTF-8 text** - the plan's own cut-line ("paste your skills").
  Accepted only if it decodes as strict UTF-8 and is not binary in disguise.

DOCX, RTF and images are refused: DOCX is a ZIP container (the zip-bomb class
the plan names), and image-only PDFs would need OCR, which this project does
not do - a scanned resume is reported as "no extractable text", not guessed at.

**The parser never runs in the API process.**  MuPDF is C code parsing
attacker-supplied bytes; a crash or a pathological file must not take the API
down or stall it.  So the PDF is handed to a fresh ``python -I`` child process
over stdin, which:

* has a wall-clock timeout (``RESUME_PARSE_TIMEOUT_S``, 20 s) and is killed on it;
* sets an address-space limit on itself (``RESUME_PARSE_MEMORY_MB``) where the
  OS supports ``RLIMIT_AS`` (Linux/macOS; **not Windows** - documented);
* refuses encrypted PDFs and more than ``RESUME_MAX_PAGES`` pages;
* stops collecting text at ``RESUME_MAX_TEXT_CHARS``.

Nothing is written to disk: the bytes go from the request body to the child's
stdin, and only text comes back.  There is therefore no filename to sanitise
and no temporary file to clean up.  The original file is not stored at all.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess  # noqa: S404 - a fixed argv, no shell, our own interpreter
import sys
from dataclasses import dataclass
from typing import Literal

from app.config import Settings

PDF_MAGIC = b"%PDF-"

#: Signatures of formats refused outright, so a renamed file gets "unsupported
#: type" rather than an accidental text decode: ZIP (DOCX, XLSX and zip bombs),
#: legacy Office (OLE2), PNG, JPEG, GIF, RTF, gzip.
_REFUSED_MAGIC = (
    b"PK\x03\x04",
    b"\xd0\xcf\x11\xe0",
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF8",
    b"{\\rtf",
    b"\x1f\x8b",
)

DocumentKind = Literal["pdf", "text"]


class DocumentRejected(ValueError):
    """A file that cannot be accepted.  ``code`` is the stable API error code;
    the message is safe to show (it never quotes the file)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    kind: DocumentKind
    text: str
    pages: int | None
    truncated: bool


def sniff(data: bytes) -> DocumentKind:
    """``pdf`` or ``text`` from the bytes themselves, or :class:`DocumentRejected`."""
    if not data:
        raise DocumentRejected("empty_file", "The uploaded file is empty.")
    if data[:1024].lstrip(b"\x00\t\r\n ").startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(_REFUSED_MAGIC):
        raise DocumentRejected(
            "unsupported_type", "Only PDF and plain UTF-8 text resumes are supported."
        )
    if b"\x00" in data:
        raise DocumentRejected(
            "unsupported_type", "Only PDF and plain UTF-8 text resumes are supported."
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DocumentRejected(
            "unsupported_type", "Only PDF and plain UTF-8 text resumes are supported."
        ) from None
    controls = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    if controls > max(8, len(text) // 100):
        raise DocumentRejected(
            "unsupported_type", "Only PDF and plain UTF-8 text resumes are supported."
        )
    return "text"


def normalise(text: str) -> str:
    """Strip control characters and collapse blank-line runs.  Deterministic."""
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# The child process.  Kept as source text so it can run under ``python -I``
# (isolated: no PYTHONPATH, no user site, no current directory on sys.path),
# which means it imports nothing of ours - only the standard library and
# pymupdf from the virtualenv's site-packages.
_CHILD = r"""
import json, sys
limit_mb, max_pages, max_chars = (int(a) for a in sys.argv[1:4])
try:
    import resource
    cap = limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
except Exception:
    pass
def done(**kw):
    sys.stdout.write(json.dumps(kw)); sys.stdout.flush(); sys.exit(0)
try:
    import pymupdf
except Exception:
    done(error="parser_unavailable")
data = sys.stdin.buffer.read()
try:
    doc = pymupdf.open(stream=data, filetype="pdf")
except Exception:
    done(error="malformed_pdf")
try:
    if doc.needs_pass or doc.is_encrypted:
        done(error="encrypted_pdf")
    if doc.page_count > max_pages:
        done(error="too_many_pages", pages=doc.page_count)
    parts, total, truncated = [], 0, False
    for page in doc:
        chunk = page.get_text("text")
        if total + len(chunk) > max_chars:
            parts.append(chunk[: max_chars - total]); truncated = True; break
        parts.append(chunk); total += len(chunk)
    done(text="\n".join(parts), pages=doc.page_count, truncated=truncated)
except Exception:
    done(error="malformed_pdf")
"""

_CHILD_ERRORS = {
    "malformed_pdf": "The PDF could not be read. It may be damaged.",
    "encrypted_pdf": "Password-protected PDFs are not supported.",
    "too_many_pages": "The PDF has too many pages.",
    "parser_unavailable": "PDF parsing is unavailable on this server.",
}


def _run_child(data: bytes, settings: Settings) -> dict[str, object]:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "-I",
                "-c",
                _CHILD,
                str(settings.resume_parse_memory_mb),
                str(settings.resume_max_pages),
                str(settings.resume_max_text_chars),
            ],
            input=data,
            capture_output=True,
            timeout=settings.resume_parse_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills the child before raising.
        raise DocumentRejected("parse_timeout", "The PDF took too long to read.") from None
    try:
        result = json.loads(completed.stdout.decode("utf-8") or "{}")
    except (UnicodeDecodeError, ValueError):
        result = {}
    if not isinstance(result, dict) or not result:
        # Crashed, or was killed by the memory limit.
        raise DocumentRejected("malformed_pdf", _CHILD_ERRORS["malformed_pdf"])
    return result


async def extract_text(data: bytes, settings: Settings) -> ExtractedDocument:
    """Sniff, bound and extract.  Raises :class:`DocumentRejected`."""
    if len(data) > settings.resume_max_bytes:
        raise DocumentRejected("file_too_large", "The file is larger than the upload limit.")
    kind = sniff(data)
    if kind == "text":
        raw = data.decode("utf-8-sig")
        truncated = len(raw) > settings.resume_max_text_chars
        text = normalise(raw[: settings.resume_max_text_chars])
        pages = None
    else:
        result = await asyncio.to_thread(_run_child, data, settings)
        error = result.get("error")
        if isinstance(error, str):
            raise DocumentRejected(error, _CHILD_ERRORS.get(error, "The PDF could not be read."))
        text = normalise(str(result.get("text", "")))
        raw_pages = result.get("pages")
        pages = raw_pages if isinstance(raw_pages, int) else None
        truncated = bool(result.get("truncated"))
    if len(text) < 20:
        raise DocumentRejected(
            "no_text",
            "No text could be extracted. Scanned (image-only) resumes are not supported.",
        )
    return ExtractedDocument(kind=kind, text=text, pages=pages, truncated=truncated)


__all__ = [
    "PDF_MAGIC",
    "DocumentRejected",
    "ExtractedDocument",
    "extract_text",
    "normalise",
    "sniff",
]
