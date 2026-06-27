"""CLI helpers for optional simulation backends."""

from __future__ import annotations

import argparse
import json

from .poke_engine_backend import probe_poke_engine


ENGINE_NOT_READY_EXIT_CODE = 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.engine_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check whether optional poke-engine support is installed.")
    doctor.add_argument("--json", action="store_true", help="Print the probe result as JSON.")
    doctor.set_defaults(func=_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _doctor(args: argparse.Namespace) -> int:
    probe = probe_poke_engine()
    if args.json:
        print(json.dumps(probe.to_dict(), indent=2, sort_keys=True))
    else:
        print(probe.message())
    return 0 if probe.ready else ENGINE_NOT_READY_EXIT_CODE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
