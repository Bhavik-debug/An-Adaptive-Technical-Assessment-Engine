"""The question bank: the git-versioned dataset and the code that guards it.

Plan section 6 - the bank is a *dataset artefact*, not a table someone types
rows into.  It lives in ``data/question-bank/*.jsonl``, is reviewed in pull
requests, and is loaded into Postgres by an ingest step that is a projection of
the files rather than a second source of truth.

Three modules, three jobs:

* ``taxonomy`` - the controlled vocabulary of domain -> topic -> subtopic.
* ``schema``   - what one item must look like, as pydantic.
* ``loader``   - reading the files, and every check that spans more than one
  item (unique ids, near-duplicates, concept-key typos).

``ingest`` sits on top and is the only part that touches a database.
"""

from app.bank.loader import BankReport, load_bank, validate_bank
from app.bank.schema import BankItem, ExpectedConcept, ReviewStatus
from app.bank.taxonomy import Taxonomy, TaxonomyError, TopicNode, load_taxonomy

__all__ = [
    "BankItem",
    "BankReport",
    "ExpectedConcept",
    "ReviewStatus",
    "Taxonomy",
    "TaxonomyError",
    "TopicNode",
    "load_bank",
    "load_taxonomy",
    "validate_bank",
]
