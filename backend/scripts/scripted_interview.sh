#!/usr/bin/env sh
# Phase 5 exit gate 1: "A shell script drives a complete 10-question interview
# through the public API and prints a report."
#
# Prerequisites (local only - see docs/reporting.md):
#   1. docker compose up -d postgres redis
#   2. cd backend && alembic upgrade head && python scripts/ingest_question_bank.py
#   3. start the API with the offline scripted model, e.g.
#        LLM_PROVIDER_ORDER=stub LLM_STUB_ON_MISSING=scripted \
#        uvicorn app.main:create_app --factory --port 8000
#
# Usage:  sh scripts/scripted_interview.sh [--base-url http://127.0.0.1:8000]
# Exit status 0 only if 10 primary questions were asked, the report is ready,
# and the completed session refused a further answer.
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
exec "$PYTHON" scripts/scripted_interview.py "$@"
