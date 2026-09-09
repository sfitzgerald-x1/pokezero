#!/usr/bin/env python3
"""Launch MCTS-vs-MCTS with a create-only, atomic terminal receipt.

The scorer writes immutable game, receipt, summary, readout, and COMPLETE
artifacts. This wrapper supplies the missing process-level handoff: each
invocation creates an immutable attempt receipt and log beneath the durable
directory, then emits one parseable terminal record only after the child exits.
A reused attempt or an existing terminal record is a refusal, never an
overwrite. A fresh attempt id can resume incomplete scorer work after an
interruption that occurred before terminal creation.

Runner arguments follow ``--``.  The wrapper owns ``--out-dir`` so the
directory named by the terminal receipt cannot diverge from the scorer's
directory.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterator, Sequence
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_TERMINAL_SCHEMA_VERSION = "pokezero.mcts-h2h-runner-terminal.v1"
ATTEMPT_SCHEMA_VERSION = "pokezero.mcts-h2h-launcher-attempt.v1"
ATTEMPT_DIRECTORY_NAME = "launcher-attempts"
TERMINAL_NAME = "runner-terminal.json"
WRITER_LOCK_NAME = "runner-writer.lock"


class LauncherError(RuntimeError):
    """A refusal before, or failure while, creating a durable handoff."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


@contextlib.contextmanager
def _writer_lock(out_dir: Path) -> Iterator[tuple[int, Path]]:
    """Exclusively guard one scorer root for the runner's entire lifetime.

    ``flock`` is associated with the open file description. Passing its file
    descriptor to the child means a hard-killed launcher cannot release the
    lease while its scorer remains alive. The empty lock file is intentionally
    retained as a durable, non-overwritten coordination artifact; only its
    advisory lock state is transient.
    """

    lock_path = out_dir / WRITER_LOCK_NAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LauncherError(
                f"another MCTS-vs-MCTS scorer still holds {lock_path}."
            ) from error
        except OSError as error:
            raise LauncherError(
                f"could not acquire scorer-root writer lock {lock_path}: {error}"
            ) from error
        yield descriptor, lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument(
        "--runner-script", type=Path, default=REPO_ROOT / "scripts" / "mcts_mcts_h2h.py"
    )
    parser.add_argument(
        "--attempt-id",
        help="optional create-only attempt identifier; defaults to a new UUID",
    )
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    for argument in raw_args:
        option = argument.split("=", 1)[0]
        if option == "--runner-log":
            parser.error(
                f"runner logging is owned under --out-dir/{ATTEMPT_DIRECTORY_NAME}; "
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


def _attempt_id(value: str | None) -> str:
    attempt_id = value or uuid.uuid4().hex
    if not attempt_id or len(attempt_id) > 64:
        raise LauncherError("--attempt-id must contain 1 to 64 safe characters.")
    if any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
        for character in attempt_id
    ):
        raise LauncherError(
            "--attempt-id must use lowercase letters, digits, '.', '_', or '-'."
        )
    return attempt_id


def _out_dir(args: argparse.Namespace) -> Path:
    out_dir = args.out_dir.expanduser().resolve()
    if not out_dir.is_dir():
        raise LauncherError(
            f"--out-dir must name an existing durable directory: {out_dir}"
        )
    return out_dir


def _paths(args: argparse.Namespace, out_dir: Path) -> tuple[str, Path, Path, Path]:
    terminal = out_dir / TERMINAL_NAME
    if terminal.exists():
        raise LauncherError(
            f"refusing to replace existing terminal receipt {terminal}."
        )
    runner_script = args.runner_script.expanduser().resolve()
    if not runner_script.is_file():
        raise LauncherError(f"--runner-script does not name a file: {runner_script}")
    attempt_id = _attempt_id(args.attempt_id)
    attempt_dir = out_dir / ATTEMPT_DIRECTORY_NAME
    if attempt_dir.exists() and not attempt_dir.is_dir():
        raise LauncherError(f"attempt directory is not a directory: {attempt_dir}")
    attempt_dir.mkdir(exist_ok=True)
    attempt_receipt = attempt_dir / f"{attempt_id}.json"
    runner_log = attempt_dir / f"{attempt_id}.log"
    for artifact in (attempt_receipt, runner_log):
        if artifact.exists():
            raise LauncherError(
                f"refusing to replace existing attempt artifact {artifact}."
            )
    return attempt_id, attempt_receipt, runner_log, terminal


def run(args: argparse.Namespace) -> int:
    out_dir = _out_dir(args)
    terminal = out_dir / TERMINAL_NAME
    if terminal.exists():
        raise LauncherError(
            f"refusing to replace existing terminal receipt {terminal}."
        )

    with _writer_lock(out_dir) as (writer_lock_descriptor, writer_lock_path):
        attempt_id, attempt_receipt, runner_log, terminal = _paths(args, out_dir)
        runner_script = args.runner_script.expanduser().resolve()
        command = [
            str(args.runner_python),
            str(runner_script),
            "--out-dir",
            str(out_dir),
            *args.runner_args,
        ]
        _write_immutable_json(
            attempt_receipt,
            {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "started_at_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "runner_command_sha256": _sha256_json(command),
                "runner_script": str(runner_script),
                "runner_script_sha256": _sha256_file(runner_script),
                "writer_lock": str(writer_lock_path),
            },
        )

        launcher_error: str | None = None
        with runner_log.open("xb") as log_handle:
            try:
                completed = subprocess.run(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    pass_fds=(writer_lock_descriptor,),
                )
                exit_code = int(completed.returncode)
            except OSError as error:
                exit_code = 127
                launcher_error = f"{type(error).__name__}: {error}"
                log_handle.write(
                    (launcher_error + "\n").encode("utf-8", errors="replace")
                )
            log_handle.flush()
            os.fsync(log_handle.fileno())

        payload: dict[str, object] = {
            "schema_version": RUNNER_TERMINAL_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "attempt_receipt": str(attempt_receipt),
            "attempt_receipt_sha256": _sha256_file(attempt_receipt),
            "completed_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "exit_code": exit_code,
            "status": "COMPLETE" if exit_code == 0 else "FAILED",
            "runner_log": str(runner_log),
            "runner_log_sha256": _sha256_file(runner_log),
            "runner_log_bytes": runner_log.stat().st_size,
            "writer_lock": str(writer_lock_path),
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
