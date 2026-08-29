"""The CI pipeline, checked from inside the test suite it runs.

CI cannot be run locally, and Days 1-29 are local-only, so nothing here proves
GitHub Actions is green.  What it *does* prove is the class of mistake that
would otherwise be discovered only after a push: a workflow that no longer runs
one of the gates, or one that starts asking for a provider credential.

The gate list comes straight from the plan: section 3 (Phase 1, Day 4) says
"GitHub Actions CI: lint -> mypy -> pytest", and section 14.3 adds ``gitleaks``
under Secrets.  These assertions are that sentence, executable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    assert WORKFLOW.exists(), f"no CI workflow at {WORKFLOW}"
    parsed = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


@pytest.fixture(scope="module")
def all_run_steps(workflow: dict[str, Any]) -> str:
    """Every shell command in the workflow, concatenated."""
    commands: list[str] = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "run" in step:
                commands.append(str(step["run"]))
    return "\n".join(commands)


class TestItIsValidAndTriggered:
    def test_the_workflow_parses(self, workflow):
        assert workflow["name"]
        assert workflow["jobs"]

    def test_it_runs_on_pushes_and_pull_requests(self, workflow):
        """The Phase 1 exit gate is "CI is green on a PR"."""
        # PyYAML resolves an unquoted `on:` key to the boolean True - the
        # YAML 1.1 "Norway problem". Accept either spelling rather than
        # quoting the key in the workflow, where `on:` is the idiom.
        triggers = workflow.get("on", workflow.get(True))
        assert triggers is not None, "the workflow declares no triggers"
        assert "pull_request" in triggers
        assert "push" in triggers

    def test_every_job_has_a_timeout(self, workflow):
        """A hung job holds a runner until GitHub's six-hour default expires."""
        for name, job in workflow["jobs"].items():
            assert job.get("timeout-minutes"), f"job {name} has no timeout"


class TestTheQualityGates:
    """Plan section 3, Day 4: lint -> mypy -> pytest. Plus formatting and secrets."""

    @pytest.mark.parametrize(
        ("gate", "command"),
        [
            ("lint", "ruff check ."),
            ("format", "ruff format --check ."),
            ("types", "mypy --strict app"),
            ("tests", "pytest"),
            ("secrets", "gitleaks"),
        ],
    )
    def test_the_gate_runs(self, all_run_steps, gate, command):
        assert command in all_run_steps, f"CI does not run the {gate} gate ({command!r})"

    def test_the_static_job_runs_lint_before_types(self, workflow):
        """Cheapest check first, so the log reads top to bottom."""
        steps = [s.get("run", "") for s in workflow["jobs"]["static"]["steps"]]
        joined = "\n".join(steps)
        assert joined.index("ruff check") < joined.index("mypy --strict")


class TestItDoesNotDependOnAnyoneMachine:
    def test_the_test_job_provides_postgres_and_redis_itself(self, workflow):
        """Integration tests run against real containers, not mocks.

        A mock cannot tell you that an Alembic migration fails to apply.
        """
        services = workflow["jobs"]["tests"]["services"]
        assert "pgvector" in services["postgres"]["image"]
        assert "redis" in services["redis"]["image"]

    def test_a_missing_service_fails_the_run_rather_than_skipping_it(self, workflow):
        """Integration tests skip when the stack is down - green on a laptop.

        In CI the stack is guaranteed, so a skip means something broke. Without
        this flag, "CI is green" could quietly mean "CI ran 37 fewer tests".
        """
        assert workflow["jobs"]["tests"]["env"]["REQUIRE_INTEGRATION"] == "1"

    def test_service_images_are_pinned_to_a_major_version(self, workflow):
        for service in workflow["jobs"]["tests"]["services"].values():
            assert ":" in service["image"], f"{service['image']} has no tag"
            assert not service["image"].endswith(":latest")


class TestNoSecretsAndNoPaidCalls:
    def test_ci_never_asks_for_a_provider_credential(self, workflow):
        """The suite must run free, offline, on somebody else's fork.

        `pytest` is configured with `-m "not smoke"`, and the unit tests supply
        a placeholder key from a fixture - so there is nothing to inject.
        """
        # Comments are stripped: the workflow explains at length *why* there is
        # no provider key, and that explanation must not fail its own test.
        effective = "\n".join(
            line
            for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("#")
        )
        assert "NVIDIA_API_KEY" not in effective
        # `${{ secrets.X }}` is how a GitHub secret is injected. There are none.
        assert "secrets." not in effective

    def test_the_secret_scanner_redacts_its_findings(self, all_run_steps):
        """Otherwise the CI log itself becomes the leak."""
        assert "--redact" in all_run_steps

    def test_the_env_file_is_never_tracked(self, all_run_steps):
        assert "git ls-files --error-unmatch .env" in all_run_steps

    def test_the_gitleaks_allowlist_exists_and_only_allows_placeholders(self):
        config = REPO_ROOT / ".gitleaks.toml"
        assert config.exists()
        text = config.read_text(encoding="utf-8")
        assert "useDefault = true" in text
        assert ".env.example" in text

    def test_the_allowlist_does_not_exempt_the_whole_test_suite(self):
        """A test file is exactly where someone pastes a real key while debugging.

        The observability tests are exempt because credential *shapes* are their
        subject; `backend/tests/` as a whole must never be.
        """
        text = (REPO_ROOT / ".gitleaks.toml").read_text(encoding="utf-8")
        assert "'''backend/tests/'''" not in text
        assert "backend/tests/unit/obs/" in text

    def test_the_workflow_only_reads_the_repository(self, workflow):
        """Least privilege: nothing here writes anything anywhere."""
        assert workflow["permissions"] == {"contents": "read"}


class TestNothingIsDeployed:
    """Days 1-29 are local-only; the first public deploy is Day 30."""

    @pytest.mark.parametrize("forbidden", ["ssh", "docker push", "ghcr.io/", "deploy"])
    def test_no_deployment_step(self, all_run_steps, forbidden):
        assert forbidden not in all_run_steps.lower().replace("ghcr.io/gitleaks/gitleaks", "")
