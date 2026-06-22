"""Command-line entry point for normalized replay import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .replay_import import import_replay_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.replay_import_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_command = subparsers.add_parser(
        "import",
        help="Convert normalized replay JSON files into rollout JSONL.",
    )
    import_command.add_argument("--input", type=Path, nargs="+", required=True, help="Normalized replay JSON file(s).")
    import_command.add_argument("--output", type=Path, required=True, help="Rollout JSONL output path.")
    import_command.add_argument("--append", action="store_true", help="Append to an existing rollout JSONL file.")
    import_command.add_argument("--json", action="store_true", help="Print import summary as JSON.")
    import_command.set_defaults(func=_import)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _import(args: argparse.Namespace) -> int:
    result = import_replay_files(args.input, output_path=args.output, append=args.append)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"records_written: {result.records_written}")
        print(f"output: {result.output_path}")
        print(f"elapsed_seconds: {result.elapsed_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
