"""CLI helpers for optional simulation backends."""

from __future__ import annotations

import argparse
import json

from .poke_engine_backend import (
    PokeEngineUnavailableError,
    probe_poke_engine,
    run_poke_engine_reversible_smoke,
)


ENGINE_NOT_READY_EXIT_CODE = 3
SMOKE_FAILED_EXIT_CODE = 4


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.engine_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check whether optional poke-engine support is installed.")
    doctor.add_argument("--json", action="store_true", help="Print the probe result as JSON.")
    doctor.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Also run the real reversible apply/reverse smoke check. "
            "Only runs when the API probe is ready; never required for the default doctor."
        ),
    )
    doctor.set_defaults(func=_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _doctor(args: argparse.Namespace) -> int:
    probe = probe_poke_engine()
    payload = probe.to_dict()
    exit_code = 0 if probe.ready else ENGINE_NOT_READY_EXIT_CODE

    smoke_summary: str | None = None
    if args.smoke:
        smoke_info, smoke_summary, smoke_exit_override = _run_smoke(probe.ready)
        payload["smoke"] = smoke_info
        if smoke_exit_override is not None and smoke_exit_override > exit_code:
            exit_code = smoke_exit_override

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(probe.message())
        if smoke_summary is not None:
            print()
            print(smoke_summary)

    return exit_code


def _run_smoke(probe_ready: bool) -> tuple[dict, str, int | None]:
    """Run the reversible smoke if the probe is ready.

    Returns ``(json_info, human_summary, exit_code_override)``. ``exit_code_override``
    is ``None`` when the smoke did not run because the probe was not ready (the
    not-ready probe exit code already covers that case), and a nonzero failure
    code when the smoke ran but did not succeed.
    """

    if not probe_ready:
        reason = "skipped: poke-engine API probe is not ready"
        return ({"ran": False, "reason": reason}, f"reversible smoke {reason}", None)

    try:
        result = run_poke_engine_reversible_smoke()
    except PokeEngineUnavailableError as exc:
        reason = f"error: {exc}"
        return (
            {"ran": True, "succeeded": False, "reason": reason},
            f"reversible smoke {reason}",
            SMOKE_FAILED_EXIT_CODE,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # pragma: no cover - defensive around native extension failures.
        reason = f"error: {type(exc).__name__}: {exc}"
        return (
            {"ran": True, "succeeded": False, "reason": reason},
            f"reversible smoke {reason}",
            SMOKE_FAILED_EXIT_CODE,
        )

    info = {"ran": True, **result.to_dict()}
    override = None if result.succeeded else SMOKE_FAILED_EXIT_CODE
    return (info, result.summary(), override)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
