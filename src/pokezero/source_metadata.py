"""Best-effort source provenance helpers for run artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess


def collect_source_metadata(cwd: Path | None = None) -> dict[str, object]:
    """Return git source metadata without making callers depend on git."""

    source_cwd = Path.cwd() if cwd is None else cwd
    try:
        repo_root = _git_output(source_cwd, "rev-parse", "--show-toplevel")
        head = _git_output(source_cwd, "rev-parse", "HEAD")
        branch = _git_output(source_cwd, "branch", "--show-current")
        status = _git_output(source_cwd, "status", "--porcelain")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "repo_root": None,
            "branch": None,
            "head": None,
            "dirty": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "repo_root": repo_root,
        "branch": branch or None,
        "head": head,
        "dirty": bool(status.strip()),
    }


def _git_output(cwd: Path, *args: str) -> str:
    process = subprocess.Popen(
        ("git", *args),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed with exit code {process.returncode}")
    return stdout.strip()
