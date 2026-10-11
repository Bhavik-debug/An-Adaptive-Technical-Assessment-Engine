"""Resume and job-description ingestion (Day 24).  See ``docs/ingestion.md``.

::

    files.py       sniff the bytes, extract text in an isolated child process
    extraction.py  prompts, output contracts, verification against the text,
                   topic weights computed in code
    profile.py     the candidate-confirmed profile contract
    jobs.py        extraction as durable, fenced, retried background jobs

Plan section 13.2 names this module (``ingestion/``: resume/JD parsers).  PII
redaction lives in ``app.privacy`` because the interview's LLM boundary uses it
too.
"""
