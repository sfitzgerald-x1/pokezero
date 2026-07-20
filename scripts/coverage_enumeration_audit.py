"""Run deterministic breadth coverage over the Gen 3 randbat observation surface.

The driver deliberately uses ``gen3customgame`` only to materialize explicit,
source-derived 1v1 teams.  ``observation_format_id=gen3randombattle`` keeps the
production candidate-set belief source active, so both self and opponent token
surfaces are audited against the same closed randbat universe used in training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.coverage_enumeration_audit import (  # noqa: E402
    CoverageGame,
    CoveragePlan,
    CoverageSelection,
    build_coverage_plan,
    normalize_coverage_move,
)
from pokezero.deep_line_audit import (  # noqa: E402
    AuditFinding,
    DeepLineAuditReport,
    audit_live_decision,
    audit_perspective_pair,
    census_protocol_cooccurrences,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.dex import normalize_id  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: E402
from pokezero.randbat_vocab import gen3_category_vocabulary, gen3_randbat_entities  # noqa: E402
from pokezero.showdown import (  # noqa: E402
    ACTION_CANDIDATE_TOKEN_OFFSET,
    BELIEF_ABILITY_BUCKET_COUNT,
    BELIEF_ITEM_BUCKET_COUNT,
    BELIEF_MOVE_BUCKET_COUNT,
    CATEGORY_BELIEF_ABILITY_OFFSET,
    CATEGORY_BELIEF_ITEM_OFFSET,
    CATEGORY_BELIEF_MOVE_OFFSET,
    CATEGORY_PRIMARY,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    SELF_POKEMON_TOKEN_OFFSET,
)
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402


def _parse_shard(value: str) -> tuple[int, int]:
    try:
        index_text, count_text = value.split("/", 1)
        index, count = int(index_text), int(count_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--shard must be zero-based INDEX/COUNT, for example 0/8") from exc
    if count < 1 or index < 0 or index >= count:
        raise argparse.ArgumentTypeError("--shard index must satisfy 0 <= INDEX < COUNT")
    return index, count


def _passes(value: str) -> tuple[str, ...]:
    return {"A": ("A",), "B": ("B",), "both": ("A", "B")}[value]


def _record(
    report: DeepLineAuditReport,
    *,
    kind: str,
    player_id: str,
    turn: int,
    column: str,
    expected: Any,
    actual: Any,
    detail: str,
) -> None:
    report.add(
        AuditFinding(
            kind=kind,
            player_id=player_id,
            turn=turn,
            column=column,
            expected=expected,
            actual=actual,
            detail=detail,
        )
    )


def _category_values(observation, token_index: int, start: int, count: int) -> set[int]:
    return {int(value) for value in observation.categorical_ids[token_index][start : start + count] if value}


def _audit_source_selection(
    env: LocalShowdownEnv,
    *,
    player_id: str,
    self_selection: CoverageSelection,
    opponent_selection: CoverageSelection,
    report: DeepLineAuditReport,
) -> None:
    """Assert source atoms survive on the real self and opponent token surfaces."""

    observation = audit_live_decision(env, player_id, report=report)
    state = env._state_for_player(player_id)
    turn = state.turn_number
    vocab = env.config.category_vocab
    if vocab is None:  # LocalShowdownConfig in this driver always receives one.
        raise AssertionError("coverage audit requires an explicit category vocabulary")

    # The entire source universe must have dedicated rows.  This catches a source/vocabulary
    # mismatch before it can hide behind the OOV safety bucket.
    expected_self_categories = (
        f"belief:possible_ability:{self_selection.ability_id}",
        f"belief:possible_item:{self_selection.item_id}",
    )
    expected_opponent_categories = (
        f"belief:possible_ability:{opponent_selection.ability_id}",
        f"belief:possible_item:{opponent_selection.item_id}",
        *(f"belief:possible_move:{normalize_coverage_move(move)}" for move in opponent_selection.moves),
    )
    for category in (*expected_self_categories, *expected_opponent_categories):
        if not vocab.is_enumerated(category):
            _record(
                report,
                kind="coverage_source_atom_oov",
                player_id=player_id,
                turn=turn,
                column="category_vocabulary",
                expected="enumerated closed-universe row",
                actual=category,
                detail="a source-derived randbat atom must not enter the OOV bucket",
            )

    # In a one-mon fixture the first self/opponent Pokemon rows are the active mons.
    self_token = SELF_POKEMON_TOKEN_OFFSET
    opponent_token = OPPONENT_POKEMON_TOKEN_OFFSET
    actual_self_abilities = _category_values(
        observation, self_token, CATEGORY_BELIEF_ABILITY_OFFSET, BELIEF_ABILITY_BUCKET_COUNT
    )
    actual_self_items = _category_values(
        observation, self_token, CATEGORY_BELIEF_ITEM_OFFSET, BELIEF_ITEM_BUCKET_COUNT
    )
    self_ability_row = vocab.encode(expected_self_categories[0])
    self_item_row = vocab.encode(expected_self_categories[1])
    if self_ability_row not in actual_self_abilities:
        _record(
            report,
            kind="coverage_self_ability_missing",
            player_id=player_id,
            turn=turn,
            column="self[0].belief_ability_bucket",
            expected=self_ability_row,
            actual=sorted(actual_self_abilities),
            detail="the source variant's self-known ability must be encoded",
        )
    if self_item_row not in actual_self_items:
        _record(
            report,
            kind="coverage_self_item_missing",
            player_id=player_id,
            turn=turn,
            column="self[0].belief_item_bucket",
            expected=self_item_row,
            actual=sorted(actual_self_items),
            detail="the source variant's self-known item must be encoded",
        )

    opponent_abilities = _category_values(
        observation, opponent_token, CATEGORY_BELIEF_ABILITY_OFFSET, BELIEF_ABILITY_BUCKET_COUNT
    )
    opponent_items = _category_values(
        observation, opponent_token, CATEGORY_BELIEF_ITEM_OFFSET, BELIEF_ITEM_BUCKET_COUNT
    )
    opponent_moves = _category_values(
        observation, opponent_token, CATEGORY_BELIEF_MOVE_OFFSET, BELIEF_MOVE_BUCKET_COUNT
    )
    belief = state.belief_view.opponent_by_species().get(opponent_selection.species_id)
    native_ability_id = opponent_selection.ability_id
    public_ability_id = (
        normalize_id(belief.revealed_ability)
        if belief is not None and belief.revealed_ability
        else native_ability_id
    )
    expected_ability_row = vocab.encode(f"belief:possible_ability:{public_ability_id}")
    expected_item_row = vocab.encode(expected_opponent_categories[1])
    if expected_ability_row not in opponent_abilities:
        _record(
            report,
            kind="coverage_opponent_public_ability_missing",
            player_id=player_id,
            turn=turn,
            column="opponent[0].belief_ability_bucket",
            expected=expected_ability_row,
            actual=sorted(opponent_abilities),
            detail="the currently public ability fact must be encoded in the opponent ability bucket",
        )
    if expected_item_row not in opponent_items:
        _record(
            report,
            kind="coverage_opponent_item_missing",
            player_id=player_id,
            turn=turn,
            column="opponent[0].belief_item_bucket",
            expected=expected_item_row,
            actual=sorted(opponent_items),
            detail="the true source variant item must remain publicly possible before reveal",
        )
    for move, category in zip(opponent_selection.moves, expected_opponent_categories[2:]):
        expected_move_row = vocab.encode(category)
        if expected_move_row not in opponent_moves:
            _record(
                report,
                kind="coverage_opponent_move_missing",
                player_id=player_id,
                turn=turn,
                column="opponent[0].belief_move_bucket",
                expected=expected_move_row,
                actual=sorted(opponent_moves),
                detail=(
                    f"the true source variant move {normalize_coverage_move(move)!r} must remain "
                    "publicly possible before reveal"
                ),
            )

    if belief is None:
        _record(
            report,
            kind="coverage_opponent_belief_missing",
            player_id=player_id,
            turn=turn,
            column="belief_view.opponent_pokemon",
            expected=opponent_selection.species_id,
            actual=None,
            detail="a visible one-mon opponent must have a source-backed belief record",
        )
    else:
        possible_ability_ids = {normalize_id(ability) for ability in belief.possible_abilities}
        if native_ability_id not in possible_ability_ids:
            _record(
                report,
                kind="coverage_source_ability_missing",
                player_id=player_id,
                turn=turn,
                column="belief_view.possible_abilities",
                expected=native_ability_id,
                actual=sorted(possible_ability_ids),
                detail="the source variant's native ability must remain in its candidate universe",
            )
        candidate_ids = {
            str(candidate.get("variant_id") or "")
            for candidate in belief.candidate_variants
            if isinstance(candidate, Mapping)
        }
        source = getattr(env, "_belief_set_source", None)
        universe = source.universe_for(opponent_selection.species) if source is not None else None
        valid_candidate_ids = {variant.variant_id for variant in getattr(universe, "variants", ())}
        report.randbat_candidate_variants_checked += len(candidate_ids)
        unexpected_candidate_ids = sorted(candidate_ids - valid_candidate_ids)
        if unexpected_candidate_ids:
            _record(
                report,
                kind="coverage_candidate_membership",
                player_id=player_id,
                turn=turn,
                column="belief_view.candidate_variants",
                expected="member of configured source universe",
                actual=unexpected_candidate_ids,
                detail="each public candidate must belong to the configured Gen 3 randbat source",
            )
        if opponent_selection.variant_id not in candidate_ids:
            _record(
                report,
                kind="coverage_true_variant_missing",
                player_id=player_id,
                turn=turn,
                column="belief_view.candidate_variants",
                expected=opponent_selection.variant_id,
                actual=sorted(candidate_ids),
                detail="a valid source-derived fixture must remain in the opponent candidate universe",
            )

    _audit_action_token_identity(env, observation, player_id=player_id, report=report, turn=turn)


def _audit_action_token_identity(
    env: LocalShowdownEnv,
    observation,
    *,
    player_id: str,
    report: DeepLineAuditReport,
    turn: int,
) -> None:
    """Compare request-derived action IDs to their encoded action-token identities."""

    snapshot = env.snapshot()
    request = snapshot.latest_requests.get(player_id)
    active = (
        request.get("active", [None])[0]
        if isinstance(request, Mapping) and isinstance(request.get("active"), list) and request.get("active")
        else None
    )
    moves = active.get("moves") if isinstance(active, Mapping) else None
    vocab = env.config.category_vocab
    if not isinstance(moves, list) or vocab is None:
        return
    for move_index, move in enumerate(moves[:4]):
        if not isinstance(move, Mapping):
            continue
        raw_name = str(move.get("id") or move.get("move") or "")
        expected_id = normalize_coverage_move(raw_name)
        if expected_id.startswith("hiddenpower"):
            expected_id = "hiddenpower"
        expected = vocab.encode(f"move:{expected_id}")
        actual = int(observation.categorical_ids[ACTION_CANDIDATE_TOKEN_OFFSET + move_index][CATEGORY_PRIMARY])
        if expected != actual:
            _record(
                report,
                kind="coverage_action_move_identity",
                player_id=player_id,
                turn=turn,
                column=f"action[{move_index}].primary",
                expected=expected,
                actual=actual,
                detail="action-token move identity must equal the raw Showdown request move id",
            )


def _first_legal(observation) -> int | None:
    return next((index for index, allowed in enumerate(observation.legal_action_mask) if allowed), None)


class _MoveUseLaneError(RuntimeError):
    """Preserve already-submitted moves if a bounded depth fixture fails."""

    def __init__(self, cause: Exception, move_use: Mapping[str, list[str]]) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.move_use = {player_id: list(moves) for player_id, moves in move_use.items()}


def _audit_true_variant_survival(
    env: LocalShowdownEnv,
    *,
    player_id: str,
    opponent_selection: CoverageSelection,
    report: DeepLineAuditReport,
) -> None:
    """Ensure public move evidence never prunes the fixture's true source tuple."""

    state = env._state_for_player(player_id)
    turn = state.turn_number
    belief = state.belief_view.opponent_by_species().get(opponent_selection.species_id)
    if belief is None:
        _record(
            report,
            kind="coverage_opponent_belief_missing",
            player_id=player_id,
            turn=turn,
            column="belief_view.opponent_pokemon",
            expected=opponent_selection.species_id,
            actual=None,
            detail="a visible fixture opponent must retain a source-backed belief record",
        )
        return
    candidate_ids = {
        str(candidate.get("variant_id") or "")
        for candidate in belief.candidate_variants
        if isinstance(candidate, Mapping)
    }
    if opponent_selection.variant_id not in candidate_ids:
        _record(
            report,
            kind="coverage_true_variant_pruned_after_action",
            player_id=player_id,
            turn=turn,
            column="belief_view.candidate_variants",
            expected=opponent_selection.variant_id,
            actual=sorted(candidate_ids),
            detail="public action evidence must not eliminate the true source-derived fixture variant",
        )


def _audit_depth_boundary(
    env: LocalShowdownEnv,
    *,
    game: CoverageGame,
    report: DeepLineAuditReport,
) -> None:
    """Run the dynamic oracle suite at a post-action decision boundary."""

    selections = {"p1": game.p1, "p2": game.p2}
    requested = env.requested_players()
    for player_id in requested:
        observation = audit_live_decision(env, player_id, report=report)
        _audit_action_token_identity(
            env,
            observation,
            player_id=player_id,
            report=report,
            turn=env._state_for_player(player_id).turn_number,
        )
        opponent_id = "p2" if player_id == "p1" else "p1"
        _audit_true_variant_survival(
            env,
            player_id=player_id,
            opponent_selection=selections[opponent_id],
            report=report,
        )
    if requested:
        audit_perspective_pair(env, report=report)


def _run_move_use_lane(
    env: LocalShowdownEnv,
    *,
    game: CoverageGame,
    report: DeepLineAuditReport,
    max_rounds: int,
    depth_rounds: int = 0,
) -> dict[str, list[str]]:
    """Best-effort move-reveal lane with an optional post-action oracle pass."""

    selections = {"p1": game.p1, "p2": game.p2}
    next_move_index = {"p1": 0, "p2": 0}
    used: dict[str, list[str]] = {"p1": [], "p2": []}
    try:
        for _ in range(max_rounds):
            if env.terminal() is not None:
                break
            requested = env.requested_players()
            if not requested:
                break
            actions: dict[str, int] = {}
            for player_id in requested:
                observation = env.observe(player_id)
                desired = next_move_index[player_id]
                if desired < len(selections[player_id].moves) and observation.legal_action_mask[desired]:
                    actions[player_id] = desired
                    used[player_id].append(normalize_coverage_move(selections[player_id].moves[desired]))
                    next_move_index[player_id] += 1
                else:
                    legal = _first_legal(observation)
                    if legal is None:
                        return used
                    actions[player_id] = legal
            protocol_start = len(env.protocol_lines) if depth_rounds else 0
            env.step(actions)
            if depth_rounds:
                # The v3 silent-noop sweep owns the exhaustive static inventory.
                # This live census records which protocol tags the exact-variant
                # fixtures actually exercised, without treating absence as proof
                # that a reachable event is harmless or unreachable.
                census_protocol_cooccurrences(env.protocol_lines[protocol_start:], report=report)
            if depth_rounds and env.terminal() is None:
                _audit_depth_boundary(env, game=game, report=report)
    except Exception as exc:
        raise _MoveUseLaneError(exc, used) from exc
    return used


def _run_game(
    env: LocalShowdownEnv,
    *,
    game: CoverageGame,
    report: DeepLineAuditReport,
    source,
    use_moves: bool,
    max_move_rounds: int,
    depth_rounds: int,
) -> dict[str, list[str]]:
    env.reset_with_start_override(seed=game.seed, start_override=game.start_override())
    report.begin_game(game.game_id)
    report.record_randbat_source(source)
    for player_id, own, opponent in (("p1", game.p1, game.p2), ("p2", game.p2, game.p1)):
        _audit_source_selection(
            env,
            player_id=player_id,
            self_selection=own,
            opponent_selection=opponent,
            report=report,
        )
        report.record_observed_randbat_team(
            ({"species": own.species, "ability": own.ability, "item": own.item, "moves": own.moves},),
            source=source,
        )
    audit_perspective_pair(env, report=report)
    return _run_move_use_lane(
        env,
        game=game,
        report=report,
        max_rounds=max_move_rounds,
        depth_rounds=depth_rounds,
    ) if use_moves else {"p1": [], "p2": []}


def _write_failure_artifact(
    failure_dir: Path,
    *,
    env: LocalShowdownEnv,
    game: CoverageGame,
    findings: Iterable[AuditFinding],
    move_use: Mapping[str, list[str]],
    exception: Exception | None = None,
) -> str:
    """Persist a compact repro only for a fixture with a real audit failure."""

    failure_dir.mkdir(parents=True, exist_ok=True)
    terminal = env.terminal()
    try:
        turn = int(env.snapshot().replay.turn_number)
    except Exception:
        turn = None
    payload = {
        "schema_version": "coverage-depth-failure-v1",
        "game": game.to_json_dict(),
        "turn": turn,
        "terminal": (
            {"winner": terminal.winner, "turn_count": terminal.turn_count, "capped": terminal.capped}
            if terminal is not None
            else None
        ),
        "move_use": {player_id: list(moves) for player_id, moves in sorted(move_use.items())},
        "findings": [finding.to_json_dict() for finding in findings],
        "exception": (
            {"type": type(exception).__name__, "message": str(exception)}
            if exception is not None
            else None
        ),
        "protocol_lines": list(env.protocol_lines),
    }
    path = failure_dir / f"{game.game_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _first_variant_with_move(source, move_id: str):
    return next(
        variant
        for universe in source.universes.values()
        for variant in universe.variants
        if move_id in variant.moves
    )


def _source_fixture(variant, *, moves: tuple[str, ...] | None = None, **overrides: Any) -> FixturePokemon:
    fields = {
        "species": variant.species,
        "moves": moves or tuple(variant.moves),
        "ability": variant.ability,
        "item": variant.item,
        "level": variant.level,
    }
    fields.update(overrides)
    return FixturePokemon(**fields)


def _universal_override(p1: FixturePokemon, p2: FixturePokemon) -> BattleStartOverride:
    return BattleStartOverride(
        player_teams={"p1": pack_team((p1,)), "p2": pack_team((p2,))},
        observation_format_id="gen3randombattle",
    )


def _action_primary(env: LocalShowdownEnv, player_id: str, action_index: int = 0) -> int:
    observation = env.observe(player_id)
    return int(observation.categorical_ids[ACTION_CANDIDATE_TOKEN_OFFSET + action_index][CATEGORY_PRIMARY])


def _run_universal_move_lane(
    env: LocalShowdownEnv,
    *,
    source,
    report: DeepLineAuditReport,
    seed_start: int,
) -> dict[str, Any]:
    """Mechanically reach vocabulary-only Struggle, Recharge, and generic Hidden Power."""

    vocab = env.config.category_vocab
    if vocab is None:
        raise AssertionError("coverage audit requires an explicit category vocabulary")
    wobbuffet = source.universe_for("wobbuffet")
    if wobbuffet is None or not wobbuffet.variants:
        raise ValueError("universal mini-lane requires a Wobbuffet source carrier")
    wob_variant = wobbuffet.variants[0]
    results: dict[str, Any] = {}

    # Mean Look has the minimum reachable PP (8), is non-damaging after its first use, and Counter
    # cannot damage Ghost-type Misdreavus. This gives a deterministic path to a real Struggle request.
    mean_look = _first_variant_with_move(source, "meanlook")
    env.reset_with_start_override(
        seed=seed_start,
        start_override=_universal_override(
            _source_fixture(mean_look, moves=("meanlook",)),
            _source_fixture(wob_variant),
        ),
    )
    report.begin_game("universal-struggle")
    for _ in range(8):
        env.step({"p1": 0, "p2": 0})
    struggle_before = len(env.protocol_lines)
    struggle_row = _action_primary(env, "p1")
    expected_struggle = vocab.encode("move:struggle")
    env.step({"p1": 0, "p2": 0})
    struggle_lines = env.protocol_lines[struggle_before:]
    struggle_ok = struggle_row == expected_struggle and any("|move|p1a:" in line and "|Struggle|" in line for line in struggle_lines)
    results["struggle"] = {
        "action_row": struggle_row,
        "expected_row": expected_struggle,
        "protocol_move_observed": any("|move|p1a:" in line and "|Struggle|" in line for line in struggle_lines),
        "complete": struggle_ok,
    }
    if not struggle_ok:
        _record(
            report,
            kind="coverage_universal_struggle_missing",
            player_id="p1",
            turn=env.snapshot().replay.turn_number,
            column="action[0].primary",
            expected=expected_struggle,
            actual=struggle_row,
            detail="PP exhaustion must surface a real Struggle request and protocol move",
        )

    # Slaking is the sole reachable Hyper Beam carrier. A max-HP/Def Wobbuffet survives one beam;
    # Destiny Bond is selected instead of Counter so the next request is a real Recharge action.
    hyper_beam = _first_variant_with_move(source, "hyperbeam")
    recharge_wobbuffet = _source_fixture(
        wob_variant,
        level=100,
        evs={"hp": 252, "def": 252},
    )
    env.reset_with_start_override(
        seed=seed_start + 1,
        start_override=_universal_override(
            _source_fixture(hyper_beam, moves=("hyperbeam",)),
            recharge_wobbuffet,
        ),
    )
    report.begin_game("universal-recharge")
    env.step({"p1": 0, "p2": 1})
    recharge_before = len(env.protocol_lines)
    recharge_row = _action_primary(env, "p1")
    expected_recharge = vocab.encode("move:recharge")
    env.step({"p1": 0, "p2": 0})
    recharge_lines = env.protocol_lines[recharge_before:]
    recharge_ok = recharge_row == expected_recharge and any("|cant|p1a:" in line and "|recharge" in line for line in recharge_lines)
    results["recharge"] = {
        "action_row": recharge_row,
        "expected_row": expected_recharge,
        "protocol_cant_observed": any("|cant|p1a:" in line and "|recharge" in line for line in recharge_lines),
        "complete": recharge_ok,
    }
    if not recharge_ok:
        _record(
            report,
            kind="coverage_universal_recharge_missing",
            player_id="p1",
            turn=env.snapshot().replay.turn_number,
            column="action[0].primary",
            expected=expected_recharge,
            actual=recharge_row,
            detail="a charged Hyper Beam turn must surface the generic Recharge action",
        )

    typed_hidden_power = next(
        variant
        for universe in source.universes.values()
        for variant in universe.variants
        if any(move.startswith("hiddenpower") for move in variant.moves)
    )
    hidden_power_index = next(
        index for index, move in enumerate(typed_hidden_power.moves) if move.startswith("hiddenpower")
    )
    env.reset_with_start_override(
        seed=seed_start + 2,
        start_override=_universal_override(_source_fixture(typed_hidden_power), _source_fixture(wob_variant)),
    )
    report.begin_game("universal-hiddenpower")
    hidden_power_row = _action_primary(env, "p1", hidden_power_index)
    expected_hidden_power = vocab.encode("move:hiddenpower")
    hidden_power_ok = hidden_power_row == expected_hidden_power
    results["hiddenpower"] = {
        "source_move": typed_hidden_power.moves[hidden_power_index],
        "action_row": hidden_power_row,
        "expected_row": expected_hidden_power,
        "complete": hidden_power_ok,
    }
    if not hidden_power_ok:
        _record(
            report,
            kind="coverage_universal_hiddenpower_missing",
            player_id="p1",
            turn=env.snapshot().replay.turn_number,
            column=f"action[{hidden_power_index}].primary",
            expected=expected_hidden_power,
            actual=hidden_power_row,
            detail="a typed source Hidden Power must use the checkpoint-stable generic action token",
        )
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, required=True, help="Write oracle findings JSON here.")
    parser.add_argument("--coverage-json", type=Path, required=True, help="Write coverage ledger JSON here.")
    parser.add_argument("--pass", dest="coverage_pass", choices=("A", "B", "both"), default="both")
    parser.add_argument("--gap-fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exact-variants",
        action="store_true",
        help="Audit every exact source variant in paired 1v1 fixtures, not only atom coverage.",
    )
    parser.add_argument(
        "--depth-rounds",
        type=int,
        default=0,
        help="After each scripted action, audit this many rounds per exact-variant fixture (default: 0).",
    )
    parser.add_argument(
        "--failure-dir",
        type=Path,
        default=None,
        help="Write protocol reproductions only for depth fixtures with findings or execution errors.",
    )
    parser.add_argument("--use-moves", action="store_true", help="Run the optional best-effort move-reveal lane.")
    parser.add_argument("--universal-lane", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-move-rounds", type=int, default=8)
    parser.add_argument("--shard", type=_parse_shard, default=(0, 1))
    parser.add_argument("--seed-start", type=int, default=9_300_000)
    parser.add_argument("--max-games", type=int, default=None, help="Testing-only cap; makes the execution ledger partial.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_move_rounds < 1:
        parser.error("--max-move-rounds must be positive")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive when provided")
    if args.depth_rounds < 0:
        parser.error("--depth-rounds must be non-negative")
    if args.exact_variants and args.coverage_pass != "both":
        parser.error("--exact-variants covers every reachable ability and requires --pass both")
    if args.depth_rounds and not args.exact_variants:
        parser.error("--depth-rounds requires --exact-variants")
    if args.depth_rounds and args.failure_dir is None:
        parser.error("--depth-rounds requires --failure-dir so only failing fixtures retain traces")
    if args.failure_dir is not None and not args.depth_rounds:
        parser.error("--failure-dir is only supported with --depth-rounds")

    preliminary = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
    root = preliminary.resolved_showdown_root()
    source = load_gen3_randbat_source_cached(root)
    entities = gen3_randbat_entities(root)
    plan = build_coverage_plan(
        source,
        source_species=entities["species"],
        source_moves=entities["moves"],
        source_items=entities["items"],
        passes=_passes(args.coverage_pass),
        gap_fill=args.gap_fill,
        exact_variants=args.exact_variants,
        seed_start=args.seed_start,
    )
    shard_index, shard_count = args.shard
    selected_games = plan.games_for_shard(shard_index=shard_index, shard_count=shard_count)
    if args.max_games is not None:
        selected_games = selected_games[: args.max_games]

    config = LocalShowdownConfig(
        showdown_root=root,
        set_belief_source=True,
        category_vocab=gen3_category_vocabulary(root, include_turn_merged=True),
    )
    report = DeepLineAuditReport()
    completed: list[CoverageGame] = []
    move_use: dict[str, dict[str, list[str]]] = {}
    universal_moves: dict[str, Any] = {}
    failure_artifacts: list[str] = []
    env = LocalShowdownEnv(config)
    try:
        for game in selected_games:
            findings_before = len(report.findings)
            game_exception: Exception | None = None
            game_move_use: dict[str, list[str]] = {"p1": [], "p2": []}
            try:
                game_move_use = _run_game(
                    env,
                    game=game,
                    report=report,
                    source=source,
                    use_moves=args.use_moves or bool(args.depth_rounds),
                    max_move_rounds=args.depth_rounds or args.max_move_rounds,
                    depth_rounds=args.depth_rounds,
                )
                move_use[game.game_id] = game_move_use
                completed.append(game)
            except Exception as exc:  # Preserve partial evidence for a fixture execution failure.
                if isinstance(exc, _MoveUseLaneError):
                    game_exception = exc.cause
                    game_move_use = exc.move_use
                else:
                    game_exception = exc
                move_use[game.game_id] = game_move_use
                _record(
                    report,
                    kind="coverage_fixture_execution",
                    player_id="system",
                    turn=0,
                    column="fixture",
                    expected="successful deterministic custom-game materialization",
                    actual=type(game_exception).__name__,
                    detail=str(game_exception),
                )
            if args.failure_dir is not None and len(report.findings) > findings_before:
                try:
                    failure_artifacts.append(
                        _write_failure_artifact(
                            args.failure_dir,
                            env=env,
                            game=game,
                            findings=report.findings[findings_before:],
                            move_use=game_move_use,
                            exception=game_exception,
                        )
                    )
                except Exception as artifact_exc:
                    _record(
                        report,
                        kind="coverage_failure_artifact_write",
                        player_id="system",
                        turn=0,
                        column="failure_artifact",
                        expected="writable failure artifact",
                        actual=type(artifact_exc).__name__,
                        detail=str(artifact_exc),
                    )
            if game_exception is not None and not args.depth_rounds:
                break
        if args.universal_lane:
            universal_moves = _run_universal_move_lane(
                env,
                source=source,
                report=report,
                seed_start=args.seed_start + len(plan.games) + 1,
            )
    finally:
        env.close()

    execution_ledger = plan.coverage_ledger(completed)
    coverage_payload = {
        **execution_ledger,
        "execution": {
            "shard": {"index": shard_index, "count": shard_count},
            "selected_game_ids": [game.game_id for game in selected_games],
            "completed_game_ids": [game.game_id for game in completed],
            "move_use": move_use,
            "universal_moves": universal_moves,
            "failure_artifacts": failure_artifacts,
            "depth_rounds": args.depth_rounds,
        },
        "planned_full_coverage": plan.coverage_ledger(),
    }
    audit_payload = report.to_json_dict()
    audit_payload["coverage_execution"] = {
        "source_hash": source.metadata.source_hash,
        "completed_games": len(completed),
        "selected_games": len(selected_games),
        "planned_games": len(plan.games),
        "coverage_complete": execution_ledger["complete"],
        "depth_rounds": args.depth_rounds,
        "failure_artifact_count": len(failure_artifacts),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.coverage_json.write_text(json.dumps(coverage_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "coverage enumeration audit: "
        f"games={len(completed)}/{len(selected_games)} findings={len(report.findings)} "
        f"coverage_complete={execution_ledger['complete']}"
    )
    return 1 if report.findings or len(completed) != len(selected_games) else 0


if __name__ == "__main__":
    raise SystemExit(main())
