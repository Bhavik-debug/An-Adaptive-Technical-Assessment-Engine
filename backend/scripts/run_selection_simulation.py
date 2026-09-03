"""Run the Day 13 simulation and print the CAT-vs-random-vs-fixed tables.

    cd backend
    python scripts/run_selection_simulation.py                  # the plan's experiment
    python scripts/run_selection_simulation.py --ablation       # plus the weight ablation
    python scripts/run_selection_simulation.py --extended       # plus the 60-item run
    python scripts/run_selection_simulation.py --write-report   # all three, into evals/reports/
    python scripts/run_selection_simulation.py --seed 7         # a different world
    python scripts/run_selection_simulation.py --candidates 20  # a quick look

**This measures; it never tunes.**  No number produced here is fed back into the
selector.  The production weights are read from ``app.selection`` and the
ablation varies a *copy*; nothing in this script can change what the application
does, which is the point - an experiment that edits the system it is measuring
has stopped measuring anything.

**Zero API tokens, no database, no network.**  Plan section 8.6 promises this
and it is literally true: the whole run is arithmetic over generated data.

**Reproducibility.**  Everything derives from ``--seed``.  Two runs with the
same seed produce byte-identical tables; the configuration block printed above
them records what to pass to get them back.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

import dataclasses  # noqa: E402

from app.evaluation.dataset import REPORTS_DIR  # noqa: E402
from app.simulation.config import EXTENDED_CONFIG, MAIN_CONFIG, ExperimentConfig  # noqa: E402
from app.simulation.environment import build_environment  # noqa: E402
from app.simulation.report import (  # noqa: E402
    format_seconds,
    markdown_report,
    text_report,
)
from app.simulation.runner import run_ablation, run_experiment  # noqa: E402

REPORT_PATH = REPORTS_DIR / "selection_simulation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=MAIN_CONFIG.seed)
    parser.add_argument(
        "--candidates",
        type=int,
        default=MAIN_CONFIG.candidate_count,
        help="how many synthetic candidates (default: the plan's 200)",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=MAIN_CONFIG.item_budget,
        help="item budget per session (default: the plan's 20)",
    )
    parser.add_argument("--ablation", action="store_true", help="also run the weight ablation")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="also run the longer-budget experiment, where RD criteria start to fire",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=f"run everything and write {REPORT_PATH.relative_to(REPORTS_DIR.parents[1])}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    want_all = args.write_report
    config = dataclasses.replace(
        MAIN_CONFIG,
        seed=args.seed,
        candidate_count=args.candidates,
        item_budget=args.items,
    )

    started = time.perf_counter()
    environment = build_environment(config)
    print(
        f"environment: {len(environment.bank)} items, "
        f"{len(environment.candidates)} candidates, seed {config.seed}",
        file=sys.stderr,
    )

    result = run_experiment(config, environment=environment)
    print(f"main run: {format_seconds(time.perf_counter() - started)}", file=sys.stderr)

    ablation = None
    if args.ablation or want_all:
        mark = time.perf_counter()
        ablation = run_ablation(config, environment=environment)
        print(f"ablation: {format_seconds(time.perf_counter() - mark)}", file=sys.stderr)

    extended = None
    if args.extended or want_all:
        mark = time.perf_counter()
        extended_config: ExperimentConfig = dataclasses.replace(
            EXTENDED_CONFIG, seed=args.seed, candidate_count=args.candidates
        )
        extended = run_experiment(extended_config)
        print(f"extended run: {format_seconds(time.perf_counter() - mark)}", file=sys.stderr)

    print(text_report(result, ablation))

    if want_all:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            markdown_report(result, ablation, extended=extended), encoding="utf-8"
        )
        print(f"\nwrote {REPORT_PATH}", file=sys.stderr)

    print(f"total: {format_seconds(time.perf_counter() - started)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
