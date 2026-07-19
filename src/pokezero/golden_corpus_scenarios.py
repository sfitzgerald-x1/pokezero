"""Curated edge-case scenario games for the golden corpus (track B + fallback audit).

Random-seed games only exercise edge cases by luck. This module scripts
`gen3customgame` games (via ``BattleStartOverride``) that deterministically
reach the positions the belief/mask edge-case matrix cares about — Truant
loafing phases, Transform, Encore locks, Hyper Beam recharge, Baton Pass
boundaries, Wish, sand + Shedinja, RestTalk, screens, toxic stalls — and
captures them through the exact same corpus machinery as the random games
(same schema; scenario identity carried in ``battle_id``, no schema change).

Two consumers:
- the encoder bit-exactness gate (scenario rows join the corpus surface);
- the fallback-detection sweep (`run_scenario_fallback_sweep`), which drives
  the engine-search policy over every scenario game and asserts each decision
  either searched or fell back with a KNOWN taxonomy reason — no crashes, no
  unmapped choices.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .env import BattleStartOverride
from .dex import normalize_id
from .golden_corpus import (
    GoldenGame,
    GoldenGameRecord,
    _CapturingPolicy,
    _decision_row_from_context,
    _true_teams_from_bridge_snapshot,
    write_golden_corpus,
)
from .golden_corpus_fold import FoldSurfaceRecorder, build_fold_rows
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .showdown import OBSERVATION_SCHEMA_VERSION_V2_2
from .policy import PolicyContext, PolicyDecision, legal_action_indices
from .rollout import RolloutConfig, continue_rollout_from_current_state
from .showdown_fixture import FixturePokemon, pack_team

_EVS = {s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")}


def _mon(species: str, moves: Sequence[str], *, ability: str, item: str = "Leftovers",
         level: int = 80, ivs: Mapping[str, int] | None = None) -> FixturePokemon:
    return FixturePokemon(species=species, moves=tuple(moves), ability=ability,
                          item=item, level=level, evs=dict(_EVS), ivs=ivs)


@dataclass(frozen=True)
class ScriptedPreferencePolicy:
    """Deterministic policy: per decision, pick the first legal candidate whose
    move id matches the preference list for that turn (cycled); otherwise the
    first legal move, otherwise the first legal action. Consumes no RNG."""

    preferences: Sequence[Sequence[str]]
    policy_id: str = "scripted-preference"
    _turn: list = field(default_factory=lambda: [0])

    def reset(self) -> None:
        self._turn[0] = 0

    def select_action(self, observation, *, rng: random.Random) -> PolicyDecision:
        index = self._pick(observation)
        return PolicyDecision(action_index=index, policy_id=self.policy_id)

    def select_action_with_context(self, context: PolicyContext, *, rng: random.Random) -> PolicyDecision:
        return self.select_action(context.observation, rng=rng)

    def _pick(self, observation) -> int:
        candidates = observation.metadata.get("action_candidates") or ()
        legal = [c for c in candidates if isinstance(c, Mapping) and c.get("legal")]
        wanted = self.preferences[self._turn[0] % len(self.preferences)] if self.preferences else ()
        self._turn[0] += 1
        for want in wanted:
            for candidate in legal:
                if candidate.get("kind") == "move" and normalize_id(str(want)) == normalize_id(
                    str(candidate.get("move_id") or "")
                ):
                    return int(candidate["action_index"])
                if candidate.get("kind") == "switch" and str(want).startswith("switch:"):
                    pokemon = candidate.get("pokemon") or {}
                    if normalize_id(want.split(":", 1)[1]) == normalize_id(str(pokemon.get("species") or "")):
                        return int(candidate["action_index"])
        moves = [c for c in legal if c.get("kind") == "move"]
        pick = moves[0] if moves else (legal[0] if legal else None)
        if pick is None:
            mask = observation.legal_action_mask
            return legal_action_indices(mask)[0]
        return int(pick["action_index"])


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    p1_team: Sequence[FixturePokemon]
    p2_team: Sequence[FixturePokemon]
    p1_prefs: Sequence[Sequence[str]]
    p2_prefs: Sequence[Sequence[str]]
    seed: int = 91000
    max_decision_rounds: int = 24


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    swampert = _mon("Swampert", ("Earthquake", "Ice Beam", "Protect", "Toxic"), ability="Torrent", level=84)
    blissey = _mon("Blissey", ("Seismic Toss", "Soft-Boiled", "Toxic", "Ice Beam"), ability="Natural Cure")
    celebi = _mon("Celebi", ("Calm Mind", "Baton Pass", "Psychic", "Recover"), ability="Natural Cure")

    return (
        ScenarioSpec(
            "truant_slaking",
            (_mon("Slaking", ("Double-Edge", "Earthquake", "Shadow Ball", "Return"),
                  ability="Truant", item="Choice Band", level=78), swampert),
            (blissey, swampert),
            p1_prefs=(("doubleedge",),),
            p2_prefs=(("softboiled", "seismictoss"),),
        ),
        ScenarioSpec(
            "ditto_transform",
            (_mon("Ditto", ("Transform",), ability="Limber", item="Quick Claw", level=100), swampert),
            (_mon("Snorlax", ("Body Slam", "Curse", "Rest", "Shadow Ball"), ability="Immunity"), swampert),
            p1_prefs=(("transform",), ("bodyslam",)),
            p2_prefs=(("curse",), ("bodyslam",)),
        ),
        ScenarioSpec(
            "encore_wobbuffet",
            (_mon("Wobbuffet", ("Encore", "Counter", "Mirror Coat", "Safeguard"), ability="Shadow Tag"), swampert),
            (celebi, swampert),
            p1_prefs=(("safeguard",), ("encore",), ("counter",)),
            p2_prefs=(("calmmind",), ("calmmind",), ("psychic",)),
        ),
        ScenarioSpec(
            "hyperbeam_recharge",
            (_mon("Slaking", ("Hyper Beam", "Earthquake", "Return", "Shadow Ball"),
                  ability="Truant", item="Choice Band", level=78), swampert),
            (blissey, swampert),
            p1_prefs=(("hyperbeam",),),
            p2_prefs=(("softboiled", "seismictoss"),),
        ),
        ScenarioSpec(
            "baton_pass_boundary",
            (celebi, _mon("Snorlax", ("Body Slam", "Curse", "Rest", "Shadow Ball"), ability="Immunity"), swampert),
            (blissey, swampert),
            p1_prefs=(("calmmind",), ("batonpass",), ("bodyslam",)),
            p2_prefs=(("seismictoss",),),
        ),
        ScenarioSpec(
            "wish_boundary",
            (_mon("Vaporeon", ("Wish", "Surf", "Protect", "Ice Beam"), ability="Water Absorb"), swampert),
            (blissey, swampert),
            p1_prefs=(("wish",), ("protect",), ("surf",)),
            p2_prefs=(("seismictoss",),),
        ),
        ScenarioSpec(
            "sand_shedinja",
            (_mon("Tyranitar", ("Rock Slide", "Earthquake", "Crunch", "Pursuit"), ability="Sand Stream"), swampert),
            (_mon("Shedinja", ("Shadow Ball", "Silver Wind", "Agility", "Baton Pass"),
                  ability="Wonder Guard", item="Lum Berry", level=100), blissey),
            p1_prefs=(("crunch",),),
            p2_prefs=(("agility",), ("shadowball",)),
        ),
        ScenarioSpec(
            "resttalk_snorlax",
            (_mon("Snorlax", ("Rest", "Sleep Talk", "Body Slam", "Curse"), ability="Immunity"), swampert),
            (blissey, swampert),
            p1_prefs=(("curse",), ("rest",), ("sleeptalk",), ("sleeptalk",)),
            p2_prefs=(("seismictoss",),),
        ),
        ScenarioSpec(
            "screens_jirachi",
            (_mon("Jirachi", ("Reflect", "Light Screen", "Psychic", "Wish"), ability="Serene Grace"), swampert),
            (_mon("Snorlax", ("Body Slam", "Curse", "Rest", "Shadow Ball"), ability="Immunity"), swampert),
            p1_prefs=(("reflect",), ("lightscreen",), ("psychic",)),
            p2_prefs=(("bodyslam",),),
        ),
        ScenarioSpec(
            "toxic_stall",
            (swampert, blissey),
            (_mon("Starmie", ("Surf", "Recover", "Psychic", "Rapid Spin"), ability="Natural Cure"), blissey),
            p1_prefs=(("toxic",), ("protect",), ("earthquake",)),
            p2_prefs=(("recover", "surf"),),
        ),
    )


def _scenario_override(spec: ScenarioSpec) -> BattleStartOverride:
    return BattleStartOverride(player_teams={
        "p1": pack_team(tuple(spec.p1_team)),
        "p2": pack_team(tuple(spec.p2_team)),
    })


def play_scenario_games(
    *,
    showdown_root: Path | str | None = None,
    specs: Sequence[ScenarioSpec] | None = None,
    belief_set_source: bool | None = None,
) -> tuple[list[GoldenGame], str | None]:
    """Play every scenario and return corpus games (same capture machinery)."""

    config = LocalShowdownConfig(showdown_root=showdown_root, set_belief_source=belief_set_source)
    env = LocalShowdownEnv(config)
    turn_merged_active = config.observation_spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_2
    games: list[GoldenGame] = []
    try:
        belief_hash = env.belief_set_source_hash
        for spec in specs if specs is not None else scenario_specs():
            override = _scenario_override(spec)
            battle_id = f"golden-scenario-{spec.name}-{spec.seed}"
            env.reset_with_start_override(seed=spec.seed, start_override=override)
            true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
            captures: list[tuple[PolicyContext, PolicyDecision]] = []
            recorder = FoldSurfaceRecorder(env)

            def _sink(context: PolicyContext, decision: PolicyDecision) -> None:
                captures.append((context, decision))
                recorder.record(context.player_id)

            policies = {
                "p1": _CapturingPolicy(ScriptedPreferencePolicy(spec.p1_prefs), _sink),
                "p2": _CapturingPolicy(ScriptedPreferencePolicy(spec.p2_prefs), _sink),
            }
            result = continue_rollout_from_current_state(
                env=env,
                policies=policies,
                config=RolloutConfig(
                    max_decision_rounds=spec.max_decision_rounds,
                    format_id=override.format_id,
                    hide_opponent_legal_action_masks=True,
                ),
                seed=spec.seed,
                battle_id=battle_id,
                reset_policies=True,
            )
            rows = tuple(
                _decision_row_from_context(context, decision, battle_seed=spec.seed)
                for context, decision in captures
            )
            fold_rows = build_fold_rows(
                replays=[context.public_materialization_state.replay for context, _ in captures],
                surfaces=recorder.surfaces,
                turn_merged_active=turn_merged_active,
            )
            record = GoldenGameRecord(
                battle_seed=spec.seed,
                battle_id=battle_id,
                format_id=override.format_id,
                policy_ids={"p1": policies["p1"].policy_id, "p2": policies["p2"].policy_id},
                true_teams=true_teams,
                terminal={
                    "winner": result.terminal.winner,
                    "turn_count": result.terminal.turn_count,
                    "capped": result.terminal.capped,
                },
            )
            games.append(GoldenGame(record=record, rows=rows, fold_rows=fold_rows))
    finally:
        env.close()
    return games, belief_hash


KNOWN_FALLBACK_REASONS = frozenset({
    "no_public_state", "no_worlds_constructed", "choices_unmapped",
})


def run_scenario_fallback_sweep(
    *,
    showdown_root: Path | str | None = None,
    specs: Sequence[ScenarioSpec] | None = None,
    worlds: int = 2,
    search_time_ms: int = 25,
) -> dict[str, Any]:
    """Drive the engine-search policy over every scenario game.

    Asserts nothing itself; returns per-scenario stats for callers/tests to
    assert on: every decision searched or fell back with a KNOWN reason,
    zero unmapped choices, world-failure taxonomy observed per scenario.
    """

    from .dex import load_showdown_dex
    from .engine_search import EngineMctsConfig, EngineMctsPolicy
    from .local_showdown import DEFAULT_SHOWDOWN_ROOT
    from .randbat import Gen3RandbatSource
    from .rollout import RolloutDriver

    root = showdown_root or DEFAULT_SHOWDOWN_ROOT
    dex = load_showdown_dex(root)
    set_source = Gen3RandbatSource.from_showdown_root(root)
    report: dict[str, Any] = {}
    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=root))
    try:
        for spec in specs if specs is not None else scenario_specs():
            override = _scenario_override(spec)
            policy = EngineMctsPolicy(
                dex=dex, set_source=set_source,
                config=EngineMctsConfig(worlds=worlds, search_time_ms=search_time_ms),
                fixed_override=override,
            )
            env.reset_with_start_override(seed=spec.seed, start_override=override)
            continue_rollout_from_current_state(
                env=env,
                policies={"p1": policy, "p2": ScriptedPreferencePolicy(spec.p2_prefs)},
                config=RolloutConfig(
                    max_decision_rounds=spec.max_decision_rounds,
                    format_id=override.format_id,
                    hide_opponent_legal_action_masks=True,
                ),
                seed=spec.seed,
                battle_id=f"sweep-{spec.name}",
                reset_policies=True,
            )
            report[spec.name] = policy.stats.to_dict()
    finally:
        env.close()
    return report


def generate_scenario_corpus(
    *,
    out_dir: Path,
    showdown_root: Path | str | None = None,
    belief_set_source: bool | None = None,
) -> dict[str, Any]:
    """Write the scenario games as a standalone corpus directory."""

    games, belief_hash = play_scenario_games(
        showdown_root=showdown_root, belief_set_source=belief_set_source
    )
    config = LocalShowdownConfig(showdown_root=showdown_root)
    spec = config.observation_spec
    masks = config.feature_masks
    from .actions import ACTION_COUNT

    header = {
        "generator": {
            "scenario_suite": [s.name for s in scenario_specs()],
            "hide_opponent_legal_action_masks": True,
        },
        "observation": {
            "schema_version": spec.schema_version,
            "token_count": spec.token_count,
            "categorical_feature_count": spec.categorical_feature_count,
            "numeric_feature_count": spec.numeric_feature_count,
            "action_count": ACTION_COUNT,
            "feature_masks": {
                "stats_block": masks.stats_block,
                "exact_state": masks.exact_state,
                "transition_token_budget": masks.transition_token_budget,
                "tier2_residuals": masks.tier2_residuals,
                "tier2_investment": masks.tier2_investment,
            },
        },
        "belief_set_source_hash": belief_hash,
    }
    return write_golden_corpus(out_dir, header=header, games=games)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the edge-case scenario corpus")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--belief-set-source",
        choices=("env", "on", "off"),
        default="env",
        help="Candidate-set belief source: pin on/off, or defer to POKEZERO_BELIEF_SET_SOURCE (default).",
    )
    args = parser.parse_args(argv)
    belief_set_source = {"env": None, "on": True, "off": False}[args.belief_set_source]
    manifest = generate_scenario_corpus(
        out_dir=Path(args.out),
        showdown_root=args.showdown_root,
        belief_set_source=belief_set_source,
    )
    counts = manifest.get("counts", {})
    print(
        f"scenario corpus written: {counts.get('decisions', '?')} rows "
        f"({counts.get('fold_rows', 0)} fold rows) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
