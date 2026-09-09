#!/usr/bin/env python3
"""Launch MCTS-vs-MCTS with a create-only, atomic terminal receipt.

The scorer writes immutable game, receipt, summary, readout, and COMPLETE
artifacts.  This wrapper supplies the missing process-level handoff: it keeps
the runner's combined output in the same durable directory and emits one
parseable terminal record only after the child exits.  An existing log or
terminal record is a refusal, never an overwrite.

Runner arguments follow ``--``.  The wrapper owns ``--out-dir`` so the
directory named by the terminal receipt cannot diverge from the scorer's
directory.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_TERMINAL_SCHEMA_VERSION = "pokezero.mcts-h2h-runner-terminal.v1"
DEFAULT_LOG_NAME = "runner-mcts-h2h.log"
TERMINAL_NAME = "runner-terminal.json"


class LauncherError(RuntimeError):
    """A refusal before, or failure while, creating a durable handoff."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically create one JSON file without replacing an existing artifact."""

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise LauncherError(
                f"refusing to replace existing terminal receipt {path}."
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument(
        "--runner-script", type=Path, default=REPO_ROOT / "scripts" / "mcts_mcts_h2h.py"
    )
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    for argument in raw_args:
        option = argument.split("=", 1)[0]
        if option == "--runner-log":
            parser.error(
                f"runner logging is fixed to --out-dir/{DEFAULT_LOG_NAME}; "
                "a custom path could collide with scorer artifacts."
            )
    args = parser.parse_args(raw_args)
    if args.runner_args[:1] == ["--"]:
        args.runner_args = args.runner_args[1:]
    for argument in args.runner_args:
        option = argument.split("=", 1)[0]
        if (
            option != "--"
            and option.startswith("--")
            and "--out-dir".startswith(option)
        ):
            parser.error(
                "--out-dir and its argparse abbreviations belong to the durable launcher, "
                "not the wrapped runner."
            )
    return args


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out_dir = args.out_dir.expanduser().resolve()
    if not out_dir.is_dir():
        raise LauncherError(
            f"--out-dir must name an existing durable directory: {out_dir}"
        )
    terminal = out_dir / TERMINAL_NAME
    if terminal.exists():
        raise LauncherError(
            f"refusing to replace existing terminal receipt {terminal}."
        )
    runner_log = out_dir / DEFAULT_LOG_NAME
    if runner_log.exists():
        raise LauncherError(f"refusing to replace existing runner log {runner_log}.")
    runner_script = args.runner_script.expanduser().resolve()
    if not runner_script.is_file():
        raise LauncherError(f"--runner-script does not name a file: {runner_script}")
    return out_dir, runner_log, terminal


def run(args: argparse.Namespace) -> int:
    out_dir, runner_log, terminal = _paths(args)
    runner_script = args.runner_script.expanduser().resolve()
    command = [
        str(args.runner_python),
        str(runner_script),
        "--out-dir",
        str(out_dir),
        *args.runner_args,
    ]

    launcher_error: str | None = None
    with runner_log.open("xb") as log_handle:
        try:
            completed = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            exit_code = int(completed.returncode)
        except OSError as error:
            exit_code = 127
            launcher_error = f"{type(error).__name__}: {error}"
            log_handle.write((launcher_error + "\n").encode("utf-8", errors="replace"))
        log_handle.flush()
        os.fsync(log_handle.fileno())

    payload: dict[str, object] = {
        "schema_version": RUNNER_TERMINAL_SCHEMA_VERSION,
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exit_code": exit_code,
        "status": "COMPLETE" if exit_code == 0 else "FAILED",
        "runner_log": str(runner_log),
        "runner_log_sha256": _sha256_file(runner_log),
        "runner_log_bytes": runner_log.stat().st_size,
    }
    if launcher_error is not None:
        payload["launcher_error"] = launcher_error
    _write_immutable_json(terminal, payload)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(_parse_args(argv))
    except LauncherError as error:
        print(f"MCTS-vs-MCTS launcher REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
