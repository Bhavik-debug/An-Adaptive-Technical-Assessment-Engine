"""Compute and commit the grading-regression baseline - Day 20.

    python scripts/update_grading_baseline.py              # show what it would write
    python scripts/update_grading_baseline.py --write --notes "..."

**No model calls.** The baseline is computed from the committed grading runs and
the committed reference labels, both of which are already on disk. That is the
point: the number in ``evals/baselines/grading_qwk.json`` must be reproducible
from the repository by anyone, at any time, with no provider credential.

``--notes`` is required to write, and the schema enforces a minimum length. A
baseline that moved without a recorded reason is a number nobody can defend, and
moving the baseline is exactly the action a reviewer needs to be able to
challenge in a diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.grading.evalset import load_answers  # noqa: E402
from app.grading.reflabel import (  # noqa: E402
    REFERENCE_RUBRIC_VERSION,
    ReferencePass,
    labels_by_answer,
    load_reference_labels,
)
from app.grading.regression import GradingBaseline, load_baseline  # noqa: E402
from app.grading.runs import load_runs, select  # noqa: E402
from app.grading.validity import quadratic_weighted_kappa, score_to_band  # noqa: E402

DATASET_DIR = BACKEND.parent / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"
REFERENCE_PATH = DATASET_DIR / "reference_labels.jsonl"
RUNS_PATH = BACKEND.parent / "evals" / "reports" / "grading" / "runs.jsonl"
BASELINE_PATH = BACKEND.parent / "evals" / "baselines" / "grading_qwk.json"

REFERENCE_PASS: ReferencePass = "blind"


def measure(prompt_version: str = "v1") -> tuple[float, int, str, str, str]:
    """QWK of the committed runs against the committed reference. Deterministic.

    Returns ``(qwk, n_items, grader_model, prompt_version, prompt_fingerprint)``.
    """
    labels = labels_by_answer(load_reference_labels(REFERENCE_PATH), pass_name=REFERENCE_PASS)
    runs = select(load_runs(RUNS_PATH), prompt_version=prompt_version, variant="base", run_index=0)
    paired = sorted(set(labels) & set(runs))
    if not paired:
        raise SystemExit("no answers have both a grading run and a reference label")
    qwk = quadratic_weighted_kappa(
        [labels[aid].score for aid in paired],
        [score_to_band(runs[aid].score) for aid in paired],
    )
    models = sorted({runs[aid].model for aid in paired})
    fingerprints = sorted({runs[aid].prompt_fingerprint for aid in paired})
    if len(models) != 1 or len(fingerprints) != 1:
        raise SystemExit(
            f"the runs are not homogeneous: models={models}, fingerprints={fingerprints}. "
            "A baseline has to describe one configuration."
        )
    return qwk, len(paired), models[0], prompt_version, fingerprints[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the baseline file")
    parser.add_argument("--notes", default="", help="why this baseline is what it is")
    parser.add_argument("--prompt", default="v1", help="which grading prompt to baseline")
    args = parser.parse_args()

    answers = load_answers(ANSWERS_PATH)
    labels = labels_by_answer(load_reference_labels(REFERENCE_PATH), pass_name=REFERENCE_PASS)
    sources = sorted({label.label_source for label in labels.values()})
    if len(sources) != 1:
        raise SystemExit(f"reference labels have mixed provenance: {sources}")
    label_source = sources[0]

    qwk, n_items, model, prompt_version, fingerprint = measure(args.prompt)
    existing = load_baseline(BASELINE_PATH) if BASELINE_PATH.exists() else None

    print(f"  answers in dataset  {len(answers)}")
    print(f"  reference labels    {len(labels)} ({REFERENCE_PASS}, {REFERENCE_RUBRIC_VERSION})")
    print(f"  label provenance    {label_source}")
    print(f"  paired items        {n_items}")
    print(f"  measured QWK        {qwk:+.6f}")
    print(f"  grader              {model} {prompt_version}/{fingerprint}")
    if existing:
        print(f"  existing baseline   v{existing.baseline_version} qwk={existing.qwk:+.6f}")
        print(f"  change              {qwk - existing.qwk:+.6f}")

    if not args.write:
        print("\n  dry run: pass --write --notes '...' to commit this baseline.")
        return 0
    if len(args.notes.strip()) < 20:
        print("\n  --notes is required and must say something (20 characters or more).")
        return 1

    baseline = GradingBaseline(
        baseline_version=(existing.baseline_version + 1) if existing else 1,
        qwk=qwk,
        n_items=n_items,
        reference_artifact="evals/datasets/grading/reference_labels.jsonl",
        reference_pass=REFERENCE_PASS,
        reference_rubric_version=REFERENCE_RUBRIC_VERSION,
        reference_label_source=label_source,
        prompt_version=prompt_version,
        prompt_fingerprint=fingerprint,
        grader_model=model,
        notes=args.notes.strip(),
    )
    # No timestamp in the file. The baseline is a *value* that must be
    # byte-reproducible from the repository, and a generation time would make
    # every re-run a diff while changing nothing that matters. The commit
    # carries the date.
    payload = baseline.model_dump(mode="json")

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n  wrote {BASELINE_PATH} (v{baseline.baseline_version})")
    if baseline.is_provisional:
        print("  NOTE: provisional - measured against MODEL labels, not human ones.")
        print("  It gates regression. It is not Phase-4 acceptance evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
