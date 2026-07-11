"""CLI for normalizing and analyzing deployment-neutral capstone artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capstone_artifacts import (
    analyze_normalized_capstone_pairs,
    load_normalized_pair,
    normalize_controlled_foulplay_artifact,
    normalize_root_puct_play_artifact,
)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _seed_band(value: str) -> tuple[str, int, int]:
    pieces = value.split(":")
    if len(pieces) != 3 or not pieces[0]:
        raise argparse.ArgumentTypeError("seed bands must use NAME:START:COUNT.")
    try:
        start = int(pieces[1])
        count = int(pieces[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed-band START and COUNT must be integers.") from error
    if start < 0 or count <= 0:
        raise argparse.ArgumentTypeError("seed-band START must be non-negative and COUNT positive.")
    return pieces[0], start, count


def _expected_keys(bands: Sequence[tuple[str, int, int]]) -> tuple[tuple[str, int, str], ...]:
    result: list[tuple[str, int, str]] = []
    names: set[str] = set()
    seeds: set[tuple[str, int]] = set()
    for name, start, count in bands:
        if name in names:
            raise ValueError(f"duplicate seed-band name: {name!r}")
        names.add(name)
        for seed in range(start, start + count):
            key = (name, seed)
            if key in seeds:
                raise ValueError(f"duplicate capstone seed in band: {key!r}")
            seeds.add(key)
            result.extend(((name, seed, "p1"), (name, seed, "p2")))
    return tuple(sorted(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and analyze strict PokeZero MCTS capstone artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    root = subparsers.add_parser("normalize-root", help="Normalize one root-puct-play benchmark artifact.")
    root.add_argument("--input", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--opponent", required=True)
    root.add_argument("--arm", required=True)
    root.add_argument("--band", required=True)
    root.add_argument("--seat", choices=("p1", "p2"), required=True)
    root.add_argument("--raw-value-leaves", action="store_true", help="Allow an explicitly raw-leaf diagnostic artifact.")
    root.set_defaults(func=_normalize_root)

    foulplay = subparsers.add_parser("normalize-foulplay", help="Normalize one controlled FoulPlay comparison artifact.")
    foulplay.add_argument("--input", type=Path, required=True)
    foulplay.add_argument("--out", type=Path, required=True)
    foulplay.add_argument("--arm", required=True)
    foulplay.add_argument("--band", required=True)
    foulplay.add_argument("--seat", choices=("p1", "p2"), required=True)
    foulplay.add_argument("--raw-value-leaves", action="store_true", help="Allow an explicitly raw-leaf diagnostic artifact.")
    foulplay.set_defaults(func=_normalize_foulplay)

    analyze = subparsers.add_parser("analyze", help="Analyze normalized artifacts for one external opponent.")
    analyze.add_argument("--artifact", type=Path, action="append", required=True)
    analyze.add_argument("--seed-band", type=_seed_band, action="append", required=True, metavar="NAME:START:COUNT")
    analyze.add_argument("--bootstrap-replicates", type=int, default=10_000)
    analyze.add_argument("--bootstrap-seed", type=int, default=20260710)
    analyze.add_argument("--out", type=Path, required=True)
    analyze.set_defaults(func=_analyze)
    return parser


def _normalize_root(args: argparse.Namespace) -> int:
    pair = normalize_root_puct_play_artifact(
        _read_json(args.input),
        opponent_id=args.opponent,
        arm_id=args.arm,
        band=args.band,
        seat=args.seat,
        uses_value_leaves=not args.raw_value_leaves,
        source_path=args.input,
    )
    _write_json(args.out, pair.to_dict())
    return 0


def _normalize_foulplay(args: argparse.Namespace) -> int:
    pair = normalize_controlled_foulplay_artifact(
        _read_json(args.input),
        arm_id=args.arm,
        band=args.band,
        seat=args.seat,
        uses_value_leaves=not args.raw_value_leaves,
        source_path=args.input,
    )
    _write_json(args.out, pair.to_dict())
    return 0


def _analyze(args: argparse.Namespace) -> int:
    report = analyze_normalized_capstone_pairs(
        (load_normalized_pair(path) for path in args.artifact),
        expected_keys=_expected_keys(args.seed_band),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_json(args.out, report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
