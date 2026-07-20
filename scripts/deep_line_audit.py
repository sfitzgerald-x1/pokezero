"""Run the read-only deep-line observation audit.

The command drives both full random-battle games and deterministic scenario
chains.  It writes a machine-readable report; a confirmed mismatch is a finding
to triage, never an automatic encoder edit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.deep_line_audit import (  # noqa: E402
    DeepLineAuditReport,
    audit_live_decision,
    audit_perspective_pair,
    audit_protocol_cut_fixture,
    census_protocol_cooccurrences,
    protocol_cut_fixtures,
)
from pokezero.golden_corpus_scenarios import (  # noqa: E402
    ScriptedPreferencePolicy,
    interaction_registry_specs,
    scenario_specs,
)
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.observation import ObservationFeatureMasks  # noqa: E402
from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: E402
from pokezero.showdown import observation_schema_version_from_choice, observation_spec_for_schema  # noqa: E402


def _current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
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


def _audit_game(env: LocalShowdownEnv, *, seed: int, report: DeepLineAuditReport, max_rounds: int) -> None:
    env.reset(seed=seed)
    report.begin_game(f"random-seed-{seed}")
    rng = random.Random(seed)
    for _ in range(max_rounds):
        if env.terminal() is not None:
            break
        requested = env.requested_players()
        if not requested:
            break
        observations = {}
        for player in requested:
            observations[player] = env.observe(player)
            # The audit function re-observes to exercise the live encoder path.
            audit_live_decision(env, player, report=report)
        if set(requested) == {"p1", "p2"}:
            audit_perspective_pair(env, report=report)
        actions = {player: _first_legal(observations[player], rng) for player in requested}
        if any(action is None for action in actions.values()):
            break
        env.step({player: int(action) for player, action in actions.items()})
    census_protocol_cooccurrences(env.snapshot().protocol_lines, report=report)


def _audit_scenario(env: LocalShowdownEnv, spec, report: DeepLineAuditReport) -> None:
    from pokezero.golden_corpus_scenarios import _scenario_override

    env.reset_with_start_override(seed=spec.seed, start_override=_scenario_override(spec))
    report.begin_game(f"scenario-{spec.name}")
    policies = {"p1": ScriptedPreferencePolicy(spec.p1_prefs), "p2": ScriptedPreferencePolicy(spec.p2_prefs)}
    for _ in range(spec.max_decision_rounds):
        if env.terminal() is not None:
            break
        requested = env.requested_players()
        if not requested:
            break
        actions = {}
        for player in requested:
            audit_live_decision(env, player, report=report)
            actions[player] = policies[player].select_action(env.observe(player), rng=random.Random(spec.seed)).action_index
        if set(requested) == {"p1", "p2"}:
            audit_perspective_pair(env, report=report)
        env.step(actions)
    census_protocol_cooccurrences(env.snapshot().protocol_lines, report=report)


def _available_scenarios(*, include_interaction_registry: bool) -> dict[str, object]:
    """Return the explicitly selectable scenario fixtures for this audit run."""

    scenarios = {spec.name: spec for spec in scenario_specs()}
    if include_interaction_registry:
        duplicate_names = scenarios.keys() & {spec.name for spec in interaction_registry_specs()}
        if duplicate_names:
            raise AssertionError(f"scenario registry contains duplicate names: {sorted(duplicate_names)}")
        scenarios.update({spec.name: spec for spec in interaction_registry_specs()})
    return scenarios


def _requested_scenario_names(
    *,
    named_scenarios: Iterable[str],
    include_all_scenarios: bool,
    include_interaction_registry: bool,
) -> tuple[dict[str, object], set[str]]:
    """Resolve CLI selectors while keeping the party registry explicitly opt-in."""

    selected = _available_scenarios(include_interaction_registry=include_interaction_registry)
    names = set(named_scenarios)
    if include_all_scenarios:
        names.update(selected)
    elif include_interaction_registry:
        names.update(spec.name for spec in interaction_registry_specs())
    return selected, names


def main(argv: Iterable[str] | None = None) -> int:
    command_arguments = list(argv) if argv is not None else list(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", type=Path, default=None)
    parser.add_argument(
        "--observation-schema",
        choices=("v3",),
        required=True,
        help="Audit v3 observations only; default-schema fallback is forbidden.",
    )
    parser.add_argument("--random-games", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=250)
    parser.add_argument("--scenarios", action="store_true", help="Audit all deterministic chain scenarios.")
    parser.add_argument(
        "--interaction-registry",
        action="store_true",
        help="Audit the curated party/silent-noop interaction registry in addition to selected scenarios.",
    )
    parser.add_argument("--protocol-fixtures", action="store_true", help="Audit minimal public protocol cuts.")
    parser.add_argument("--scenario", action="append", default=[], help="Audit one named scenario (repeatable).")
    parser.add_argument(
        "--suppress-kind",
        action="append",
        default=[],
        help="Record but do not fail on a known finding kind (repeatable).",
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(command_arguments)
    if args.random_games < 0 or args.max_rounds < 1 or args.seed_start < 0:
        parser.error("--random-games and --seed-start must be non-negative; --max-rounds must be positive")

    observation_schema = observation_schema_version_from_choice(args.observation_schema)
    if observation_schema is None:  # Defensive: parser requires a concrete v3 choice.
        raise AssertionError("deep-line audit requires an explicit observation schema")
    masks = ObservationFeatureMasks(tier2_residuals=False, tier2_investment=False)
    config = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True, feature_masks=masks)
    from pokezero.randbat_vocab import gen3_category_vocabulary

    config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        set_belief_source=True,
        feature_masks=masks,
        observation_spec=observation_spec_for_schema(observation_schema),
        category_vocab=gen3_category_vocabulary(
            config.resolved_showdown_root(), include_turn_merged=True
        ),
    )
    report = DeepLineAuditReport(suppressed_kinds=frozenset(args.suppress_kind))
    env = LocalShowdownEnv(config)
    try:
        for seed in range(args.seed_start, args.seed_start + args.random_games):
            _audit_game(env, seed=seed, report=report, max_rounds=args.max_rounds)
        selected, names = _requested_scenario_names(
            named_scenarios=args.scenario,
            include_all_scenarios=args.scenarios,
            include_interaction_registry=args.interaction_registry,
        )
        unknown = sorted(names - set(selected))
        if unknown:
            parser.error(f"unknown scenario(s): {', '.join(unknown)}")
        for name in sorted(names):
            _audit_scenario(env, selected[name], report)
        if args.protocol_fixtures:
            for fixture in protocol_cut_fixtures():
                audit_protocol_cut_fixture(fixture, report=report)
    finally:
        env.close()

    source = load_gen3_randbat_source_cached(config.resolved_showdown_root())
    payload = report.to_json_dict()
    payload["audit_provenance"] = {
        "schema_version": "pokezero.deep-line-audit-provenance.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "public_repo_commit": _current_commit(),
        "showdown_source_hash": source.metadata.source_hash,
        "observation_schema": observation_schema,
        "image_digest": os.environ.get("POKEZERO_AUDIT_IMAGE_DIGEST", "local-uncontainerized"),
        "command": [str(Path(__file__).relative_to(ROOT)), *command_arguments],
    }
    _write_json_atomic(args.json, payload)
    print(
        f"deep-line audit: decisions={report.decisions_checked} turn20+={report.turn_20_plus_decisions} "
        f"findings={len(report.findings)}"
    )
    return 1 if report.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
