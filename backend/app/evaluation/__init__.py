"""Retrieval evaluation (Phase 2, Day 10) — how *well* does retrieval work?

Days 8 and 9 established that the retrieval pipeline runs. This package answers
the different, harder question: is it any good, and which parts of it are
actually earning their cost?

* ``metrics``  - Recall@K, MRR, nDCG@K. Pure functions, no database, no models.
* ``dataset``  - the labelled queries and their known-relevant question ids.
* ``runner``   - runs the evaluation set through each retrieval mode.
* ``report``   - formats an ``EvalReport`` into tables.

``docs/evaluation.md`` explains ground truth, each metric and the ablation from
first principles.

**This package measures; it never tunes.** Nothing here feeds a threshold, a
weight or a K value back into the retrieval system. An evaluation set that the
system has been fitted to has stopped being a measurement.

Like ``app.retrieval``, this ``__init__`` re-exports nothing, so importing one
module never drags in the rest. Import from the module that owns the name::

    from app.evaluation.metrics import ndcg_at_k
    from app.evaluation.runner import run_evaluation
"""
