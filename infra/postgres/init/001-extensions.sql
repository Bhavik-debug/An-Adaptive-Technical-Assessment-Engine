-- Runs once, on an empty data directory, as the superuser.
-- Extensions are a database capability rather than application schema, so they
-- live here instead of in Alembic (which is forward-only and non-superuser).
CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: question embeddings (Phase 2)
CREATE EXTENSION IF NOT EXISTS citext;   -- case-insensitive emails (Phase 1, Day 2)
