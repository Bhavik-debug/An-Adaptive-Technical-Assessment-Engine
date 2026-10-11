"""Run the API as a real operating-system process, for the Day-25 gate tests.

``python -m uvicorn app.main:create_app --factory`` in a subprocess, pointed at
the integration database and Redis, with the scripted offline model.  This is
the deployment shape - a separate process with its own in-process worker - not
an ASGI app inside the test's event loop.

``kill()`` is ``Popen.kill()``: ``TerminateProcess`` on Windows, ``SIGKILL`` on
POSIX.  Neither lets the process run its shutdown hooks, so it is a genuine
crash, not a graceful stop.
"""

from __future__ import annotations

import os
import socket
import subprocess  # noqa: S404 - our own interpreter, fixed argv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import Settings
from tests.conftest import VALID_ENV

BACKEND_DIR = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class ApiProcess:
    proc: subprocess.Popen[bytes]
    base_url: str
    log_path: Path
    env: dict[str, str] = field(default_factory=dict)

    def kill(self) -> int:
        """Hard kill: no shutdown hooks, no lease release, no flush."""
        self.proc.kill()
        return self.proc.wait(timeout=30)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


def start_api(database_url: str, redis_url: str, log_dir: Path, **overrides: str) -> ApiProcess:
    port = free_port()
    env = {k: v for k, v in os.environ.items() if k.lower() not in Settings.model_fields}
    env.update(VALID_ENV)
    env.update(
        {
            "DATABASE_URL": database_url,
            "REDIS_URL": redis_url,
            "GRADING_WORKER_ENABLED": "true",
            "GRADING_POLL_INTERVAL_S": "0.1",
            "SSE_POLL_INTERVAL_S": "0.1",
            "LLM_PROVIDER_ORDER": "stub",
            "LLM_STUB_ON_MISSING": "scripted",
            "LLM_CACHE_ENABLED": "false",
            "OTEL_ENABLED": "false",
        }
    )
    env.update(overrides)
    log_path = log_dir / f"api-{port}.log"
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=BACKEND_DIR,
        env=env,
        stdout=log_path.open("wb"),
        stderr=subprocess.STDOUT,
    )
    api = ApiProcess(proc=proc, base_url=f"http://127.0.0.1:{port}", log_path=log_path, env=env)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"API process exited early:\n{api.log_tail()}")
        try:
            if httpx.get(f"{api.base_url}/healthz", timeout=1).status_code == 200:
                return api
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    api.stop()
    raise RuntimeError(f"API process did not become healthy:\n{api.log_tail()}")
