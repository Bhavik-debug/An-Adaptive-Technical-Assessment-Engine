"""Where the dataset lives.

One module so that no other file has to count ``parents[...]`` levels, and so a
test can point the loader somewhere else without monkeypatching a constant that
was baked into an import.
"""

from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent

# Plan section 13.9 fixes this location: `data/question-bank/*.jsonl`.
BANK_DIR = REPO_ROOT / "data" / "question-bank"
TAXONOMY_PATH = BANK_DIR / "taxonomy.json"

__all__ = ["BACKEND_DIR", "BANK_DIR", "REPO_ROOT", "TAXONOMY_PATH"]
