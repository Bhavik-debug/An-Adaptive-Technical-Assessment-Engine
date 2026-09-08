"""Day 14: draw how the ability estimate converges, and write the numbers beside it.

    cd backend
    python scripts/run_convergence_analysis.py                  # tables on stdout
    python scripts/run_convergence_analysis.py --write-report   # + figures into evals/reports/
    python scripts/run_convergence_analysis.py --extended       # the 60-item budget instead
    python scripts/run_convergence_analysis.py --seed 7         # a different world
    python scripts/run_convergence_analysis.py --candidates 20  # a quick look

**This re-runs Day 13's experiment; it does not re-implement it.**  The session
loop, the ability update, the selector and the stopping rule all come from the
packages that ship them, through :func:`app.simulation.runner.run_experiment`.
Day 14 added an observation record to that loop and nothing else - Day 13's own
report regenerates byte-identically, which is the check that the analysis did
not perturb the thing being analysed.

**This measures; it never tunes.**  No number produced here is fed back into the
selector, and nothing in this script can change what the application does.

**Zero API tokens, no database, no network.**  Arithmetic over generated data,
about ninety seconds for the plan's 200 candidates.

**Reproducibility.**  Everything derives from ``--seed``.  Two runs with the
same seed produce byte-identical tables *and* byte-identical SVG files; the
charts are emitted as text by :mod:`app.simulation.charts` precisely so that
this is true and so that a review can read the diff.
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
from app.simulation.config import EXTENDED_CONFIG, MAIN_CONFIG  # noqa: E402
from app.simulation.environment import build_environment  # noqa: E402
from app.simulation.figures import build_figures, write_figures  # noqa: E402
from app.simulation.report import (  # noqa: E402
    convergence_report,
    convergence_table,
    coverage_error_table,
    format_seconds,
    stability_table,
)
from app.simulation.runner import run_experiment  # noqa: E402

# The committed artefacts are UTF-8 and use the right characters for what they
# mean - a real "less than or equal", a real theta.  A Windows console defaults
# to cp1252 and would raise on the first one, so the streams are widened here
# rather than the typography being narrowed there.  `errors="replace"` keeps a
# console that genuinely cannot render a glyph from killing the run.
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPORT_PATH = REPORTS_DIR / "convergence.md"
#: Figures live in their own directory beside the report that embeds them, so a
#: regenerated run replaces files in place rather than accumulating them.
FIGURE_DIR = REPORTS_DIR / "convergence"


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
        default=None,
        help="item budget per session (default: the plan's 20, or 60 with --extended)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="analyse the 60-item run instead of the plan's 20-item one",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=f"write {REPORT_PATH.name} and its figures into {FIGURE_DIR.name}/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = EXTENDED_CONFIG if args.extended else MAIN_CONFIG
    config = dataclasses.replace(
        base,
        seed=args.seed,
        candidate_count=args.candidates,
        item_budget=base.item_budget if args.items is None else args.items,
    )

    started = time.perf_counter()
    environment = build_environment(config)
    print(
        f"environment: {len(environment.bank)} items, "
        f"{len(environment.candidates)} candidates, seed {config.seed}, "
        f"budget {config.item_budget}",
        file=sys.stderr,
    )
    result = run_experiment(config, environment=environment)
    print(f"simulation: {format_seconds(time.perf_counter() - started)}", file=sys.stderr)

    mark = time.perf_counter()
    figures = build_figures(result)
    print(
        f"figures: {len(figures)} in {format_seconds(time.perf_counter() - mark)}",
        file=sys.stderr,
    )

    print("CONVERGENCE")
    print(convergence_table(result))
    print()
    print("COVERAGE AGAINST FINAL ERROR")
    print(coverage_error_table(result))
    print()
    print("STABILITY")
    print(stability_table(result))
    print()
    print("FIGURES")
    for figure in figures:
        print(f"  {figure.filename:34s} {figure.title}")

    if args.write_report:
        written = write_figures(figures, FIGURE_DIR)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            convergence_report(result, figures, figure_dir=FIGURE_DIR.name),
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nwrote {REPORT_PATH} and {len(written)} figures", file=sys.stderr)

    print(f"total: {format_seconds(time.perf_counter() - started)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
