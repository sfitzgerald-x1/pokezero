"""Run the bounded v3 silent engine-mutation audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.golden_corpus_scenarios import ScriptedPreferencePolicy, interaction_registry_specs, scenario_specs  # noqa: E402
from pokezero.golden_corpus_scenarios import _scenario_override  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: E402
from pokezero.randbat_vocab import gen3_category_vocabulary  # noqa: E402
from pokezero.showdown import observation_schema_version_from_choice, observation_spec_for_schema  # noqa: E402
from pokezero.silent_mutation_audit import SilentMutationAuditReport  # noqa: E402


def _current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _first_legal(observation, rng: random.Random) -> int | None:
    legal = [index for index, allowed in enumerate(observation.legal_action_mask) if allowed]
    return rng.choice(legal) if legal else None


def _record_step(env: LocalShowdownEnv, actions: dict[str, int], report: SilentMutationAuditReport, game_id: str) -> None:
    before = env.snapshot()
    env.step(actions)
    report.record_transition(before, env.snapshot(), game_id=game_id)


def _run_game(env: LocalShowdownEnv, *, seed: int, report: SilentMutationAuditReport, max_rounds: int) -> None:
    env.reset(seed=seed)
    rng = random.Random(seed)
    game_id = f"random-seed-{seed}"
    for _ in range(max_rounds):
        if env.terminal() is not None:
            return
        requested = env.requested_players()
        if not requested:
            return
        actions = {player: _first_legal(env.observe(player), rng) for player in requested}
        if any(action is None for action in actions.values()):
            return
        _record_step(env, {player: int(action) for player, action in actions.items()}, report, game_id)


def _run_scenario(env: LocalShowdownEnv, spec, report: SilentMutationAuditReport) -> None:
    env.reset_with_start_override(seed=spec.seed, start_override=_scenario_override(spec))
    policies = {"p1": ScriptedPreferencePolicy(spec.p1_prefs), "p2": ScriptedPreferencePolicy(spec.p2_prefs)}
    for _ in range(spec.max_decision_rounds):
        if env.terminal() is not None:
            return
        requested = env.requested_players()
        if not requested:
            return
        actions = {
            player: policies[player].select_action(env.observe(player), rng=random.Random(spec.seed)).action_index
            for player in requested
        }
        _record_step(env, actions, report, f"scenario-{spec.name}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", type=Path, default=None)
    parser.add_argument("--observation-schema", choices=("v3",), required=True)
    parser.add_argument("--random-games", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--scenarios", action="store_true")
    parser.add_argument("--interaction-registry", action="store_true")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.random_games < 0 or args.seed_start < 0 or args.max_rounds < 1:
        parser.error("--random-games and --seed-start must be non-negative; --max-rounds must be positive")

    schema = observation_schema_version_from_choice(args.observation_schema)
    if schema is None:
        raise AssertionError("silent-mutation audit requires v3")
    base = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
    root = base.resolved_showdown_root()
    config = LocalShowdownConfig(
        showdown_root=root,
        set_belief_source=True,
        observation_spec=observation_spec_for_schema(schema),
        category_vocab=gen3_category_vocabulary(root, include_turn_merged=True),
    )
    report = SilentMutationAuditReport()
    scenarios = {spec.name: spec for spec in scenario_specs()}
    if args.interaction_registry:
        scenarios.update({spec.name: spec for spec in interaction_registry_specs()})
    names = set(args.scenario)
    if args.scenarios:
        names.update(scenarios)
    elif args.interaction_registry:
        names.update(spec.name for spec in interaction_registry_specs())
    unknown = sorted(names - scenarios.keys())
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(unknown)}")

    env = LocalShowdownEnv(config)
    try:
        for seed in range(args.seed_start, args.seed_start + args.random_games):
            _run_game(env, seed=seed, report=report, max_rounds=args.max_rounds)
        for name in sorted(names):
            _run_scenario(env, scenarios[name], report)
    finally:
        env.close()

    source = load_gen3_randbat_source_cached(root)
    payload = report.to_json_dict()
    payload["audit_provenance"] = {
        "schema_version": "pokezero.silent-mutation-audit-provenance.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "public_repo_commit": _current_commit(),
        "showdown_source_hash": source.metadata.source_hash,
        "observation_schema": schema,
        "image_digest": os.environ.get("POKEZERO_AUDIT_IMAGE_DIGEST", "local-uncontainerized"),
        "command": ["silent_mutation_audit.py", *sys.argv[1:]],
    }
    _write_json_atomic(args.json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
