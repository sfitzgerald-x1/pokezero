"""Count-only canonical protocol capture from production-style self-play.

The v3 omission audit needs observed protocol frequencies from the same public
rollout path used by learned-policy self-play.  This module deliberately keeps
only canonical protocol-signature counts and deterministic seed locators: it
does not retain requests, observations, tensors, player names, or raw logs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Mapping, Sequence

from .audit_provenance import public_repo_commit
from .collection import env_config_with_policy_spec_masks, policy_from_spec, policy_spec_with_showdown_root
from .deep_line_audit import PROTOCOL_SIGNATURE_SCHEMA_VERSION, protocol_signature_counts
from .env import PokeZeroEnv
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .policy import Policy
from .randbat import load_gen3_randbat_source_cached
from .rollout import RolloutConfig, RolloutDriver
from .showdown import observation_schema_version_from_choice, observation_spec_for_schema


ROOT = Path(__file__).resolve().parents[2]
SELFPLAY_PROTOCOL_CAPTURE_SCHEMA_VERSION = "pokezero.selfplay-protocol-capture.v2"


@dataclass(frozen=True)
class SelfplayProtocolCaptureResult:
    """Public-safe protocol census collected from one deterministic seed band."""

    protocol_signatures: Mapping[str, int]
    protocol_signature_game_ids: tuple[str, ...]
    completed_games: int
    decision_rounds: int
    capped_games: int

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_signatures": dict(sorted(self.protocol_signatures.items())),
            "protocol_signature_game_ids": list(self.protocol_signature_game_ids),
            "completed_games": self.completed_games,
            "decision_rounds": self.decision_rounds,
            "capped_games": self.capped_games,
        }


def capture_selfplay_protocol_signatures(
    *,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    seed_start: int,
    battle_id_prefix: str = "selfplay-protocol-capture",
) -> SelfplayProtocolCaptureResult:
    """Run a bounded self-play seed band and retain only canonical signatures.

    An environment is reused across the requested games so the local Showdown
    bridge follows the normal warm-rollout path.  Protocol lines are read only
    after a game is terminal; neither lines nor observations enter the result.
    """

    if games <= 0:
        raise ValueError("games must be positive")
    if not battle_id_prefix.strip():
        raise ValueError("battle_id_prefix must be non-empty")
    missing = sorted({"p1", "p2"} - set(policies))
    if missing:
        raise ValueError(f"self-play protocol capture requires p1 and p2 policies; missing {', '.join(missing)}")

    counts: Counter[str] = Counter()
    game_ids: list[str] = []
    decision_rounds = 0
    capped_games = 0
    env = env_factory()
    try:
        driver = RolloutDriver(env=env, policies=policies, config=rollout_config)
        for offset in range(games):
            seed = seed_start + offset
            result = driver.run(seed=seed, battle_id=f"{battle_id_prefix}-{seed}")
            if result.terminal.capped:
                raise ValueError(
                    "self-play protocol capture refuses capped rollouts; "
                    f"seed {seed} reached max_decision_rounds={rollout_config.max_decision_rounds}"
                )
            lines = getattr(env, "protocol_lines", None)
            if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)):
                raise ValueError("self-play protocol capture environment does not expose protocol_lines")
            counts.update(protocol_signature_counts(tuple(str(line) for line in lines)))
            game_ids.append(_seed_locator(seed))
            decision_rounds += result.decision_round_count
            capped_games += int(result.terminal.capped)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return SelfplayProtocolCaptureResult(
        protocol_signatures=dict(counts),
        protocol_signature_game_ids=tuple(game_ids),
        completed_games=games,
        decision_rounds=decision_rounds,
        capped_games=capped_games,
    )


def _seed_locator(seed: int) -> str:
    """Return a deterministic locator without retaining a battle room identifier."""

    return hashlib.sha256(f"selfplay-protocol-capture:{seed}".encode("utf-8")).hexdigest()[:24]


def _policy_spec_sha256(spec: str) -> str:
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()


def _policy_identity(policy: Policy, *, spec: str) -> dict[str, str | None]:
    """Bind a census seat to its loaded policy, not just a mutable path string."""

    weights_sha256 = getattr(policy, "weights_sha256", None)
    if weights_sha256 is not None and not isinstance(weights_sha256, str):
        raise ValueError("loaded self-play policy has an invalid weights_sha256 provenance field")
    return {
        "policy_id": str(policy.policy_id),
        "policy_spec_sha256": _policy_spec_sha256(spec),
        "weights_sha256": weights_sha256,
    }


def _redacted_command_arguments(command_arguments: Sequence[str]) -> list[str]:
    """Keep a reproducible command shape without persisting private policy specs."""

    policy_flags = frozenset({"--p1-policy", "--p2-policy"})
    redacted: list[str] = []
    index = 0
    while index < len(command_arguments):
        argument = command_arguments[index]
        if argument in policy_flags:
            if index + 1 >= len(command_arguments):
                raise ValueError(f"{argument} is missing its policy specification")
            redacted.extend((argument, f"sha256:{_policy_spec_sha256(command_arguments[index + 1])}"))
            index += 2
            continue
        flag, separator, value = argument.partition("=")
        if separator and flag in policy_flags:
            redacted.append(f"{flag}=sha256:{_policy_spec_sha256(value)}")
        else:
            redacted.append(argument)
        index += 1
    return redacted


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _capture_provenance(
    *,
    source_hash: str,
    command_arguments: Sequence[str],
    seed_start: int,
    games: int,
    capture_label: str,
    max_decision_rounds: int,
    p1_policy_identity: Mapping[str, str | None],
    p2_policy_identity: Mapping[str, str | None],
) -> dict[str, object]:
    return {
        "schema_version": "pokezero.protocol-capture-provenance.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "public_repo_commit": public_repo_commit(ROOT),
        "showdown_source_hash": source_hash,
        "observation_schema": "pokezero.observation.v3",
        "image_digest": os.environ.get("POKEZERO_AUDIT_IMAGE_DIGEST", "local-uncontainerized"),
        "command": [str(Path(__file__).relative_to(ROOT)), *_redacted_command_arguments(command_arguments)],
        "execution_scope": {
            "capture_mode": "selfplay",
            "capture_label": capture_label,
            "seed_range": {"start": seed_start, "end": seed_start + games - 1, "count": games},
            "max_decision_rounds": max_decision_rounds,
            "p1_policy": dict(p1_policy_identity),
            "p2_policy": dict(p2_policy_identity),
        },
    }


def _validate_existing_capture(
    path: Path,
    *,
    expected_provenance: Mapping[str, object],
    expected_games: int,
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing self-play protocol capture {path}: {exc}") from exc
    if payload.get("schema_version") != SELFPLAY_PROTOCOL_CAPTURE_SCHEMA_VERSION:
        raise ValueError("existing self-play protocol capture has an incompatible schema")
    if payload.get("protocol_signature_schema_version") != PROTOCOL_SIGNATURE_SCHEMA_VERSION:
        raise ValueError("existing self-play protocol capture has an incompatible signature schema")
    actual_provenance = payload.get("audit_provenance")
    immutable_fields = ("public_repo_commit", "showdown_source_hash", "observation_schema", "image_digest")
    if not isinstance(actual_provenance, Mapping) or any(
        actual_provenance.get(field) != expected_provenance.get(field) for field in immutable_fields
    ):
        raise ValueError("existing self-play protocol capture has incompatible source, schema, or image provenance")
    expected_scope = expected_provenance["execution_scope"]
    actual_scope = actual_provenance.get("execution_scope")
    if actual_scope != expected_scope:
        raise ValueError("existing self-play protocol capture has an incompatible policy or seed scope")
    counts = payload.get("protocol_signatures")
    game_ids = payload.get("protocol_signature_game_ids")
    capture = payload.get("selfplay_protocol_capture")
    if (
        not isinstance(counts, Mapping)
        or any(not isinstance(key, str) or not isinstance(value, int) or value < 0 for key, value in counts.items())
        or not isinstance(game_ids, list)
        or len(game_ids) != expected_games
        or len(set(game_ids)) != expected_games
        or not isinstance(capture, Mapping)
        or capture.get("completed_games") != expected_games
        or capture.get("capped_games") != 0
    ):
        raise ValueError("existing self-play protocol capture has incomplete count-only evidence")
    return dict(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pokezero.selfplay_protocol_capture",
        description="Capture count-only canonical protocol signatures from a v3 self-play seed band.",
    )
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Atomic count-only summary JSON path.")
    parser.add_argument("--showdown-root", type=Path, required=True, help="Built Pokemon Showdown checkout root.")
    parser.add_argument("--p1-policy", required=True, help="Production self-play policy specification for p1.")
    parser.add_argument("--p2-policy", required=True, help="Production self-play policy specification for p2.")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--max-decision-rounds", type=int, default=250)
    parser.add_argument("--node-binary", default="node")
    parser.add_argument("--capture-label", required=True, help="Stable private capture-pool label.")
    parser.add_argument(
        "--observation-schema",
        choices=("v3",),
        required=True,
        help="The omission audit is v3-only; schema fallback is forbidden.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    command_arguments = list(argv) if argv is not None else list(sys.argv[1:])
    args = build_arg_parser().parse_args(command_arguments)
    try:
        return _run(args, command_arguments)
    except ValueError as exc:
        print(f"selfplay_protocol_capture: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, command_arguments: Sequence[str]) -> int:
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.max_decision_rounds <= 0:
        raise ValueError("--max-decision-rounds must be positive")
    if not args.capture_label.strip():
        raise ValueError("--capture-label must be non-empty")
    observation_schema = observation_schema_version_from_choice(args.observation_schema)
    if observation_schema is None:
        raise AssertionError("self-play protocol capture requires an explicit v3 schema")
    source = load_gen3_randbat_source_cached(args.showdown_root)
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
        observation_spec=observation_spec_for_schema(observation_schema),
    )
    env_config = env_config_with_policy_spec_masks(
        env_config,
        (args.p1_policy, args.p2_policy),
        context="self-play protocol capture",
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    policies = {
        "p1": policy_from_spec(policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)),
        "p2": policy_from_spec(policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)),
    }
    provenance = _capture_provenance(
        source_hash=source.metadata.source_hash,
        command_arguments=command_arguments,
        seed_start=args.seed_start,
        games=args.games,
        capture_label=args.capture_label,
        max_decision_rounds=args.max_decision_rounds,
        p1_policy_identity=_policy_identity(policies["p1"], spec=args.p1_policy),
        p2_policy_identity=_policy_identity(policies["p2"], spec=args.p2_policy),
    )
    if args.out.exists():
        payload = _validate_existing_capture(args.out, expected_provenance=provenance, expected_games=args.games)
        print(
            "selfplay_protocol_capture: reused "
            f"{payload['selfplay_protocol_capture']['completed_games']} games from {args.out}"
        )
        return 0
    result = capture_selfplay_protocol_signatures(
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
        rollout_config=RolloutConfig(max_decision_rounds=args.max_decision_rounds),
        seed_start=args.seed_start,
    )
    payload: dict[str, object] = {
        "schema_version": SELFPLAY_PROTOCOL_CAPTURE_SCHEMA_VERSION,
        "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
        **result.to_dict(),
        "selfplay_protocol_capture": {
            "capture_label": args.capture_label,
            "format_id": "gen3randombattle",
            "completed_games": result.completed_games,
            "decision_rounds": result.decision_rounds,
            "capped_games": result.capped_games,
            "p1_policy_id": policies["p1"].policy_id,
            "p2_policy_id": policies["p2"].policy_id,
        },
        "audit_provenance": provenance,
    }
    _write_json_atomic(args.out, payload)
    print(
        "selfplay_protocol_capture: "
        f"captured {result.completed_games} games, {result.decision_rounds} decisions, "
        f"{len(result.protocol_signatures)} canonical signatures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
