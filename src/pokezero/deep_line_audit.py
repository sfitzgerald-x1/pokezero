"""Read-only deep-line audit helpers for the production observation encoder.

The ordinary encoder tests prove isolated mechanics.  This module drives live
Showdown battles across multiple turns and checks the public observation surface
against two independent sources:

* the bridge's serialized simulator state for mechanical facts that are public
  at the decision boundary; and
* a fresh, batch-built replay/belief fold for incremental-state agreement.

It intentionally does not assert belief candidate identities against the hidden
simulator state. Those values are epistemic by design; the audit checks their
structural invariants, monotonicity, and membership in the configured public
Gen 3 randbat source as public evidence accumulates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Mapping, Sequence

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .category_vocab import CategoryVocabulary
from .dex import load_showdown_dex_cached
from .local_showdown import LocalShowdownEnv, LocalShowdownSnapshot
from .observation import PokeZeroObservationV0, TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
from .randbat import canonical_gen3_randbat_species_id
from .randbat_vocab import gen3_category_vocabulary
from .belief import PublicBattleBeliefEngine
from .showdown import (
    CATEGORY_PRIMARY,
    CATEGORY_SECONDARY,
    CATEGORY_TYPE_1,
    CATEGORY_TYPE_2,
    CATEGORY_BELIEF_ABILITY_OFFSET,
    CATEGORY_BELIEF_ITEM_OFFSET,
    ACTION_CANDIDATE_TOKEN_OFFSET,
    FIELD_TOKEN_OFFSET,
    NUMERIC_ACTIVE,
    NUMERIC_BOOST_ATK,
    NUMERIC_BOOST_DEF,
    NUMERIC_BOOST_SPA,
    NUMERIC_BOOST_SPD,
    NUMERIC_BOOST_SPE,
    NUMERIC_CANDIDATE_SET_COUNT,
    NUMERIC_HP_FRACTION,
    NUMERIC_LEGAL,
    NUMERIC_LEVEL,
    NUMERIC_PRESENT,
    NUMERIC_MOVE_PP_FRACTION,
    NUMERIC_TIER2_CB_PINNED,
    NUMERIC_TIER2_INVESTMENT_PINNED,
    NUMERIC_TT_CB_BIT,
    NUMERIC_TT_INVESTMENT_BIT,
    NUMERIC_TT_RESIDUAL,
    NUMERIC_TT_RESIDUAL_VALID,
    NUMERIC_TM2_CB_BIT,
    NUMERIC_TM2_INVESTMENT,
    NUMERIC_TM2_RESIDUAL,
    NUMERIC_TM2_RESIDUAL_VALID,
    NUMERIC_OPP_HAZARDS,
    NUMERIC_OPP_LIGHT_SCREEN_TURNS,
    NUMERIC_OPP_MIST_TURNS,
    NUMERIC_OPP_REFLECT_TURNS,
    NUMERIC_OPP_SAFEGUARD_TURNS,
    NUMERIC_OPP_SCREENS,
    NUMERIC_REVEALED_ABILITY,
    NUMERIC_REVEALED_ITEM,
    NUMERIC_SELF_HAZARDS,
    NUMERIC_SELF_LIGHT_SCREEN_TURNS,
    NUMERIC_SELF_MIST_TURNS,
    NUMERIC_SELF_REFLECT_TURNS,
    NUMERIC_SELF_SAFEGUARD_TURNS,
    NUMERIC_SELF_SCREENS,
    NUMERIC_TOXIC_STAGE,
    NUMERIC_TURN_COUNT,
    NUMERIC_WEATHER_PERMANENT,
    NUMERIC_WEATHER_TURNS,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    SELF_POKEMON_TOKEN_OFFSET,
    _normalize_identifier,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
)


_BOOST_SLOTS = (
    ("atk", NUMERIC_BOOST_ATK),
    ("def", NUMERIC_BOOST_DEF),
    ("spa", NUMERIC_BOOST_SPA),
    ("spd", NUMERIC_BOOST_SPD),
    ("spe", NUMERIC_BOOST_SPE),
)
_PUBLIC_UNIT_INTERVAL_SLOTS = frozenset(
    {
        NUMERIC_HP_FRACTION,
        NUMERIC_ACTIVE,
        NUMERIC_LEGAL,
        NUMERIC_PRESENT,
        NUMERIC_TOXIC_STAGE,
    }
)
_RANDBAT_COMPONENT_KINDS = ("species", "variant", "move", "ability", "item")
_FORECAST_FORM_TYPE_FALLBACK = {
    "castformsunny": ("Fire",),
    "castformrainy": ("Water",),
    "castformsnowy": ("Ice",),
    "castform": ("Normal",),
}
# These fields are intentionally assessed at the first live observation after a
# strike. A fresh batch fold sees the final boundary's strictly larger evidence
# set, so equality is not a valid invariant for the corresponding history and
# current-state projection columns.
_LIVE_INCREMENTAL_ANNOTATION_COLUMNS = frozenset(
    {
        NUMERIC_TIER2_CB_PINNED,
        NUMERIC_TIER2_INVESTMENT_PINNED,
        NUMERIC_TT_RESIDUAL,
        NUMERIC_TT_RESIDUAL_VALID,
        NUMERIC_TT_CB_BIT,
        NUMERIC_TT_INVESTMENT_BIT,
        NUMERIC_TM2_RESIDUAL,
        NUMERIC_TM2_RESIDUAL_VALID,
        NUMERIC_TM2_CB_BIT,
        NUMERIC_TM2_INVESTMENT,
    }
)


def _normalize_randbat_component_move(value: str) -> str:
    """Normalize request-side dynamic-power spellings to the source move id."""

    move = _normalize_identifier(value)
    if move.startswith("return"):
        return "return"
    if move.startswith("frustration"):
        return "frustration"
    return move


@dataclass(frozen=True)
class AuditFinding:
    """One reproducible encoder inconsistency or invariant failure."""

    kind: str
    player_id: str
    turn: int
    column: str
    expected: Any
    actual: Any
    detail: str
    game_id: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "player_id": self.player_id,
            "turn": self.turn,
            "column": self.column,
            "expected": self.expected,
            "actual": self.actual,
            "detail": self.detail,
            "game_id": self.game_id,
        }


@dataclass(frozen=True)
class ProtocolCutFixture:
    """Minimal public protocol sequence for a stateful encoder regression."""

    name: str
    purpose: str
    lines: tuple[str, ...]


_PROTOCOL_CUT_FIXTURES = (
    ProtocolCutFixture(
        name="cureteam_benched_toxic",
        purpose="Heal Bell/Aromatherapy must clear a living benched ally.",
        lines=(
            "|start",
            "|switch|p1a: Vigoroth|Vigoroth, L88|300/300",
            "|switch|p2a: Snorlax|Snorlax, L80|300/300",
            "|turn|1",
            "|-status|p1a: Vigoroth|tox",
            "|turn|2",
            "|switch|p1a: Vileplume|Vileplume, L80|300/300",
            "|move|p1a: Vileplume|Aromatherapy|p1a: Vileplume",
            "|-cureteam|p1",
            "|turn|3",
        ),
    ),
    ProtocolCutFixture(
        name="forecast_formechange",
        purpose="A public Forecast form must replace active species/type identity.",
        lines=(
            "|start",
            "|switch|p1a: Castform|Castform, L80|200/200",
            "|switch|p2a: Starmie|Starmie, L80|200/200",
            "|turn|1",
            "|-formechange|p1a: Castform|Castform-Rainy|[msg]",
        ),
    ),
    ProtocolCutFixture(
        name="color_change_typechange",
        purpose="A public Color Change type override must persist until switch-out.",
        lines=(
            "|start",
            "|switch|p1a: Kecleon|Kecleon, L80|200/200",
            "|switch|p2a: Starmie|Starmie, L80|200/200",
            "|turn|1",
            "|-start|p1a: Kecleon|typechange|Ice|[from] ability: Color Change",
        ),
    ),
    ProtocolCutFixture(
        name="leech_seed_pending_snapshot",
        purpose="A snapshot between move declaration and Leech Seed start preserves source side.",
        lines=(
            "|start",
            "|switch|p1a: Meganium|Meganium, L80|300/300",
            "|switch|p2a: Milotic|Milotic, L80|300/300",
            "|turn|1",
            "|move|p1a: Meganium|Leech Seed|p2a: Milotic",
            "|-start|p2a: Milotic|move: Leech Seed|[of] p1a: Meganium",
        ),
    ),
    ProtocolCutFixture(
        name="intimidate_switch_in",
        purpose="A switch-in ability plus its target boost change stays ordered in the fold.",
        lines=(
            "|start",
            "|switch|p1a: Arbok|Arbok, L85|250/250",
            "|switch|p2a: Snorlax|Snorlax, L80|300/300",
            "|-ability|p1a: Arbok|Intimidate|boost",
            "|-unboost|p2a: Snorlax|atk|1",
            "|turn|1",
        ),
    ),
    ProtocolCutFixture(
        name="sand_stream_switch_in",
        purpose="Ability weather emitted at switch-in remains visible at the first request boundary.",
        lines=(
            "|start",
            "|switch|p1a: Tyranitar|Tyranitar, L80|300/300",
            "|switch|p2a: Gengar|Gengar, L80|250/250",
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
            "|turn|1",
        ),
    ),
    ProtocolCutFixture(
        name="baton_pass_switch_boundary",
        purpose="A Baton Pass declaration followed by replacement remains an ordered public chain.",
        lines=(
            "|start",
            "|switch|p1a: Celebi|Celebi, L80|300/300",
            "|switch|p2a: Snorlax|Snorlax, L80|300/300",
            "|turn|1",
            "|move|p1a: Celebi|Calm Mind|p1a: Celebi",
            "|-boost|p1a: Celebi|spa|1",
            "|-boost|p1a: Celebi|spd|1",
            "|move|p1a: Celebi|Baton Pass|p1a: Celebi",
            "|switch|p1a: Vaporeon|Vaporeon, L80|300/300",
            "|turn|2",
        ),
    ),
)


def protocol_cut_fixtures() -> tuple[ProtocolCutFixture, ...]:
    """Return minimal public protocol cuts for confirmed deep-line regressions."""

    return _PROTOCOL_CUT_FIXTURES


@dataclass
class DeepLineAuditReport:
    """Aggregate result for one or more complete games/scenario chains."""

    games_audited: int = 0
    decisions_checked: int = 0
    turn_20_plus_decisions: int = 0
    findings: list[AuditFinding] = field(default_factory=list)
    protocol_cooccurrences: Counter[tuple[str, ...]] = field(default_factory=Counter)
    protocol_ordered_pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    protocol_ordered_triples: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    protocol_events: Counter[str] = field(default_factory=Counter)
    # Top-level tags identify broad protocol coverage; this companion census
    # preserves the subtype/reason carried by omission-prone tags such as
    # -activate, cant, and -fail.
    protocol_signatures: Counter[str] = field(default_factory=Counter)
    candidate_count_history: dict[tuple[str, str], int] = field(default_factory=dict)
    current_game_id: str | None = None
    suppressed_kinds: frozenset[str] = frozenset()
    suppressed_findings: Counter[str] = field(default_factory=Counter)
    # Lane 5 is source-backed rather than omniscient: beliefs are allowed to be
    # uncertain, but every emitted candidate must belong to the exact randbat
    # universe used by the belief engine.  Keep catalog and observed components
    # separate so a sampled long-game run cannot be mistaken for exhaustive
    # universe coverage.
    randbat_source_metadata: dict[str, Any] | None = None
    randbat_catalog_components: dict[str, set[str]] = field(
        default_factory=lambda: {kind: set() for kind in _RANDBAT_COMPONENT_KINDS}
    )
    randbat_observed_components: dict[str, set[str]] = field(
        default_factory=lambda: {kind: set() for kind in _RANDBAT_COMPONENT_KINDS}
    )
    randbat_candidate_variants_checked: int = 0

    def add(self, finding: AuditFinding) -> None:
        if finding.kind in self.suppressed_kinds:
            self.suppressed_findings[finding.kind] += 1
            return
        if finding.game_id is None and self.current_game_id is not None:
            finding = replace(finding, game_id=self.current_game_id)
        self.findings.append(finding)

    def begin_game(self, game_id: str) -> None:
        """Start a new battle without carrying belief evidence across seeds."""

        self.games_audited += 1
        self.candidate_count_history.clear()
        self.current_game_id = game_id

    @property
    def ok(self) -> bool:
        return not self.findings

    def merge(self, other: "DeepLineAuditReport") -> None:
        self.games_audited += other.games_audited
        self.decisions_checked += other.decisions_checked
        self.turn_20_plus_decisions += other.turn_20_plus_decisions
        self.findings.extend(other.findings)
        self.protocol_cooccurrences.update(other.protocol_cooccurrences)
        self.protocol_ordered_pairs.update(other.protocol_ordered_pairs)
        self.protocol_ordered_triples.update(other.protocol_ordered_triples)
        self.protocol_events.update(other.protocol_events)
        self.protocol_signatures.update(other.protocol_signatures)
        self.suppressed_findings.update(other.suppressed_findings)
        if self.randbat_source_metadata is None:
            self.randbat_source_metadata = other.randbat_source_metadata
        elif other.randbat_source_metadata is not None:
            if self.randbat_source_metadata != other.randbat_source_metadata:
                raise ValueError("Cannot merge reports from different randbat source versions.")
        for kind in _RANDBAT_COMPONENT_KINDS:
            self.randbat_catalog_components[kind].update(other.randbat_catalog_components[kind])
            self.randbat_observed_components[kind].update(other.randbat_observed_components[kind])
        self.randbat_candidate_variants_checked += other.randbat_candidate_variants_checked

    def record_randbat_source(self, source: Any) -> None:
        """Register the immutable universe used to evaluate belief candidates."""

        metadata = getattr(source, "metadata", None)
        source_metadata = {
            "format_id": str(getattr(metadata, "format_id", "gen3randombattle")),
            "generation": int(getattr(metadata, "generation", 3)),
            "source_hash": str(getattr(metadata, "source_hash", "")),
        }
        if self.randbat_source_metadata is not None:
            if self.randbat_source_metadata != source_metadata:
                raise ValueError("A single audit report cannot mix randbat source versions.")
            return
        self.randbat_source_metadata = source_metadata
        universes = getattr(source, "universes", {})
        if not isinstance(universes, Mapping):
            return
        for universe in universes.values():
            species = canonical_gen3_randbat_species_id(
                _normalize_identifier(str(getattr(universe, "species", "")))
            )
            if species:
                self.randbat_catalog_components["species"].add(species)
            for variant in getattr(universe, "variants", ()):
                variant_id = str(getattr(variant, "variant_id", ""))
                if variant_id:
                    self.randbat_catalog_components["variant"].add(variant_id)
                ability = _normalize_identifier(str(getattr(variant, "ability", "")))
                item = _normalize_identifier(str(getattr(variant, "item", "")))
                if ability:
                    self.randbat_catalog_components["ability"].add(ability)
                if item:
                    self.randbat_catalog_components["item"].add(item)
                self.randbat_catalog_components["move"].update(
                    move
                    for raw_move in getattr(variant, "moves", ())
                    if (move := _normalize_randbat_component_move(str(raw_move)))
                )

    def record_observed_randbat_team(
        self,
        team: Sequence[Mapping[str, Any]],
        *,
        source: Any,
    ) -> None:
        """Record components disclosed by one player's complete self request."""

        for pokemon in team:
            species = canonical_gen3_randbat_species_id(
                _normalize_identifier(str(pokemon.get("species") or ""))
            )
            ability = _normalize_identifier(str(pokemon.get("ability") or ""))
            item = _normalize_identifier(str(pokemon.get("item") or ""))
            if species:
                self.randbat_observed_components["species"].add(species)
            if ability:
                self.randbat_observed_components["ability"].add(ability)
            if item:
                self.randbat_observed_components["item"].add(item)
            moves = pokemon.get("moves")
            observed_moves: set[str] = set()
            if isinstance(moves, Sequence) and not isinstance(moves, (str, bytes)):
                observed_moves.update(
                    move for raw_move in moves if (move := _normalize_randbat_component_move(str(raw_move)))
                )
            self.randbat_observed_components["move"].update(observed_moves)
            universe = source.universe_for(species) if species else None
            for variant in getattr(universe, "variants", ()):
                variant_moves = {
                    _normalize_randbat_component_move(str(raw_move)) for raw_move in variant.moves
                }
                if (
                    observed_moves == variant_moves
                    and ability == _normalize_identifier(str(variant.ability))
                    and item == _normalize_identifier(str(variant.item))
                ):
                    self.randbat_observed_components["variant"].add(str(variant.variant_id))

    def to_json_dict(self) -> dict[str, Any]:
        source_coverage = None
        if self.randbat_source_metadata is not None:
            source_coverage = {
                "source_metadata": dict(self.randbat_source_metadata),
                "catalog_component_counts": {
                    kind: len(self.randbat_catalog_components[kind]) for kind in _RANDBAT_COMPONENT_KINDS
                },
                "observed_component_counts": {
                    kind: len(self.randbat_observed_components[kind]) for kind in _RANDBAT_COMPONENT_KINDS
                },
                "unobserved_component_counts": {
                    kind: len(self.randbat_catalog_components[kind] - self.randbat_observed_components[kind])
                    for kind in _RANDBAT_COMPONENT_KINDS
                },
                "candidate_variants_checked": self.randbat_candidate_variants_checked,
            }
        return {
            "schema_version": "pokezero.deep-line-audit.v1",
            "games_audited": self.games_audited,
            "decisions_checked": self.decisions_checked,
            "turn_20_plus_decisions": self.turn_20_plus_decisions,
            "finding_count": len(self.findings),
            "findings": [finding.to_json_dict() for finding in self.findings],
            "suppressed_finding_counts": dict(sorted(self.suppressed_findings.items())),
            "protocol_events": dict(sorted(self.protocol_events.items())),
            "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
            "protocol_signatures": dict(sorted(self.protocol_signatures.items())),
            "protocol_cooccurrences": [
                {"events": list(events), "count": count}
                for events, count in sorted(
                    self.protocol_cooccurrences.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "protocol_ordered_pairs": [
                {"events": list(events), "count": count}
                for events, count in sorted(
                    self.protocol_ordered_pairs.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "protocol_ordered_triples": [
                {"events": list(events), "count": count}
                for events, count in sorted(
                    self.protocol_ordered_triples.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "randbat_source_coverage": source_coverage,
        }


def audit_live_decision(
    env: LocalShowdownEnv,
    player_id: str,
    *,
    report: DeepLineAuditReport,
    check_snapshot_roundtrip: bool = True,
    check_incremental_batch: bool = True,
) -> PokeZeroObservationV0:
    """Audit one live decision boundary without changing the battle's trajectory.

    The bridge serialization is intentionally used only as an omniscient oracle.
    It never feeds values back into the encoder under test.
    """

    observation = attach_audit_vocabulary(env.observe(player_id), _category_vocab_for_env(env))
    snapshot = env.snapshot()
    turn = int(snapshot.replay.turn_number)
    report.decisions_checked += 1
    if turn >= 20:
        report.turn_20_plus_decisions += 1

    check_randbat_candidates = snapshot.format_id == "gen3randombattle"
    if check_randbat_candidates:
        _audit_randbat_source_coverage(env, observation, player_id, turn, report)
    _audit_numeric_invariants(
        observation,
        player_id,
        turn,
        report,
        check_randbat_candidates=check_randbat_candidates,
    )
    _audit_self_known_facts(observation, player_id, turn, report)
    _audit_request_action_surface(snapshot, player_id, observation, turn, report)
    _audit_public_transform_identity(observation, player_id, turn, report)
    _audit_snapshot_public_surface(env, snapshot, player_id, observation, report)
    if check_randbat_candidates:
        _audit_candidate_monotonicity(observation, player_id, turn, report)
    if check_incremental_batch:
        _audit_incremental_vs_batch(env, snapshot, player_id, observation, turn, report)
    if check_snapshot_roundtrip:
        _audit_snapshot_roundtrip(env, snapshot, player_id, observation, turn, report)
    return observation


def _audit_randbat_source_coverage(
    env: LocalShowdownEnv,
    observation: PokeZeroObservationV0,
    player_id: str,
    turn: int,
    report: DeepLineAuditReport,
) -> None:
    """Validate public belief candidates against the configured randbat universe.

    The check deliberately never compares candidates to simulator-private truth:
    a public belief can legitimately retain many variants.  It does establish
    that the candidates emitted by the production fold are members of the same
    fixed source universe, and it records exactly how much of that universe a
    sampled battle shard happened to disclose through self requests.
    """

    source = getattr(env, "_belief_set_source", None)
    if source is None:
        return
    report.record_randbat_source(source)

    team = observation.metadata.get("self_team")
    if isinstance(team, Sequence) and not isinstance(team, (str, bytes)):
        report.record_observed_randbat_team(
            tuple(pokemon for pokemon in team if isinstance(pokemon, Mapping)),
            source=source,
        )

    belief_view = observation.metadata.get("belief_view")
    if not isinstance(belief_view, Mapping):
        return
    opponent = belief_view.get("opponent_pokemon")
    if not isinstance(opponent, Sequence) or isinstance(opponent, (str, bytes)):
        return
    for belief in opponent:
        if not isinstance(belief, Mapping):
            continue
        species = str(belief.get("species") or "")
        universe = source.universe_for(species)
        candidates = belief.get("candidate_variants")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            continue
        actual_count = belief.get("candidate_set_count")
        if isinstance(actual_count, int):
            _compare(
                report,
                kind="source_candidate_count",
                player_id=player_id,
                turn=turn,
                column=f"belief:{species}.candidate_set_count",
                expected=len(candidates),
                actual=actual_count,
                detail="the encoded candidate count must equal the public source-variant list length",
            )
        valid_variant_ids = {
            str(variant.variant_id) for variant in getattr(universe, "variants", ())
        }
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            report.randbat_candidate_variants_checked += 1
            variant_id = str(candidate.get("variant_id") or "")
            if variant_id not in valid_variant_ids:
                report.add(
                    AuditFinding(
                        kind="source_candidate_membership",
                        player_id=player_id,
                        turn=turn,
                        column=f"belief:{species}.variant_id",
                        expected="member of configured species universe",
                        actual=variant_id,
                        detail="a public belief candidate must come from the configured Gen 3 randbat source",
                    )
                )


def audit_perspective_pair(
    env: LocalShowdownEnv,
    *,
    report: DeepLineAuditReport,
) -> None:
    """Check player-relative public active/field facts mirror between p1 and p2.

    This intentionally compares only facts both seats can see.  Candidate-set,
    item, and exact-stat fields remain player-relative and are not symmetry
    assertions.
    """

    vocab = _category_vocab_for_env(env)
    observations = {
        player: attach_audit_vocabulary(env.observe(player), vocab) for player in ("p1", "p2")
    }
    turn = int(env.snapshot().replay.turn_number)
    p1, p2 = observations["p1"], observations["p2"]
    pairs = (
        (
            _active_pokemon_token(p1, SELF_POKEMON_TOKEN_OFFSET),
            _active_pokemon_token(p2, OPPONENT_POKEMON_TOKEN_OFFSET),
            "p1-self/p2-opponent",
        ),
        (
            _active_pokemon_token(p1, OPPONENT_POKEMON_TOKEN_OFFSET),
            _active_pokemon_token(p2, SELF_POKEMON_TOKEN_OFFSET),
            "p1-opponent/p2-self",
        ),
    )
    for p1_token, p2_token, label in pairs:
        if p1_token is None or p2_token is None:
            continue
        for slot in (NUMERIC_HP_FRACTION, NUMERIC_ACTIVE, NUMERIC_PRESENT, NUMERIC_LEGAL):
            _compare(
                report,
                kind="perspective_symmetry",
                player_id="p1",
                turn=turn,
                column=f"{label}:numeric[{slot}]",
                expected=_numeric(p1, p1_token, slot),
                actual=_numeric(p2, p2_token, slot),
                detail="public active Pokemon values must mirror across player-relative views",
            )
        for slot in (CATEGORY_PRIMARY, CATEGORY_SECONDARY):
            if slot == CATEGORY_SECONDARY and (
                _numeric(p1, p1_token, NUMERIC_HP_FRACTION) <= 0.0
                or _numeric(p2, p2_token, NUMERIC_HP_FRACTION) <= 0.0
            ):
                # A fainted mon's last nonvolatile status is historical evidence,
                # not a current battle fact. Different player-relative folds may
                # preserve or clear it, so it is not a meaningful symmetry gate.
                continue
            _compare(
                report,
                kind="perspective_symmetry",
                player_id="p1",
                turn=turn,
                column=f"{label}:category[{slot}]",
                expected=_categorical(p1, p1_token, slot),
                actual=_categorical(p2, p2_token, slot),
                detail="public active Pokemon categories must mirror across player-relative views",
            )


def audit_protocol_cut_fixture(fixture: ProtocolCutFixture, *, report: DeepLineAuditReport) -> None:
    """Check parser/belief state at a minimal public protocol cut.

    These cases complement live simulator games for event boundaries that are
    difficult to force from a random battle. They are intentionally protocol
    only: the fixture never supplies private request data.
    """

    report.begin_game(f"protocol-{fixture.name}")
    replay = parse_showdown_replay(
        fixture.lines,
        battle_id=f"battle-gen3randombattle-audit-{fixture.name}",
    )
    census_protocol_cooccurrences(fixture.lines, report=report)
    turn = int(replay.turn_number)
    if fixture.name == "cureteam_benched_toxic":
        belief_snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        for side in ("p1", "p2"):
            for pokemon in replay.public_revealed.get(side, ()):
                if _condition_is_fainted(pokemon.condition):
                    continue
                _compare(
                    report,
                    kind="protocol_cureteam_public_status",
                    player_id=side,
                    turn=turn,
                    column=f"{side}:{pokemon.species}.condition_status",
                    expected="none",
                    actual=_condition_status(pokemon.condition),
                    detail="-cureteam must clear the public condition status of every living ally",
                )
            for pokemon in belief_snapshot.side(side):
                if _condition_is_fainted(pokemon.condition):
                    continue
                _compare(
                    report,
                    kind="protocol_cureteam_belief_status",
                    player_id=side,
                    turn=turn,
                    column=f"{side}:{pokemon.species}.status",
                    expected=None,
                    actual=pokemon.status,
                    detail="-cureteam must clear the public belief status of every living ally",
                )
    elif fixture.name == "forecast_formechange":
        line = next(line for line in fixture.lines if line.startswith("|-formechange|"))
        parts = line.split("|")
        target = parts[2]
        slot = target[:2]
        active = replay.public_active.get(slot)
        _compare(
            report,
            kind="protocol_formechange_base_identity",
            player_id=slot,
            turn=turn,
            column=f"{slot}.active_species",
            expected=_normalize_identifier(target.split(": ", 1)[-1]),
            actual=_normalize_identifier(active.species) if active is not None else None,
            detail="Forecast retains Castform's base species while its live type changes",
        )
        _compare(
            report,
            kind="protocol_formechange_live_type",
            player_id=slot,
            turn=turn,
            column=f"{slot}.live_type_override",
            expected=f"forme:{parts[3].strip()}",
            actual=replay.live_type_override.get(slot),
            detail="a public -formechange must retain the form as a live type source",
        )


def census_protocol_cooccurrences(
    lines: Sequence[str],
    *,
    report: DeepLineAuditReport,
) -> None:
    """Count set and ordered protocol event co-occurrences per turn boundary."""

    current: list[str] = []
    for line in lines:
        parts = line.split("|")
        event = parts[1] if len(parts) > 1 else ""
        if not event:
            continue
        if event == "turn":
            _record_turn_cooccurrence(current, report)
            current = []
            continue
        if event not in {"request", "upkeep", "t:", "player"}:
            current.append(event)
            report.protocol_signatures[_canonical_protocol_signature(parts)] += 1
    _record_turn_cooccurrence(current, report)


_SIGNATURE_PAYLOAD_EVENTS = frozenset(
    {
        "move",
        "cant",
        "-activate",
        "-fail",
        "-start",
        "-end",
        "-singleturn",
        "-fieldactivate",
        "-mustrecharge",
        "-notarget",
        "-sethp",
        "-weather",
        "-status",
        "-curestatus",
    }
)

# This pins the spelling of ``protocol_signatures`` separately from the
# surrounding deep-line report. Consumers reject older spellings rather than
# treating legacy ``move:`` or ``ability:`` prefixes as new omissions.
PROTOCOL_SIGNATURE_SCHEMA_VERSION = "pokezero.protocol-signature-census.v2"

_EFFECT_IDENTIFIER_EVENTS = frozenset(
    {
        "-activate",
        "-end",
        "-fail",
        "-fieldactivate",
        "-singleturn",
        "-start",
    }
)


def _canonical_effect_identifier(value: str) -> str:
    """Normalize a protocol effect name without preserving transport prefixes.

    Showdown spells the same effect as either ``Protect`` or ``move: Protect``
    depending on the event family. The census must treat those as one semantic
    signature so a handler can be matched with its redundant announcement.
    """

    prefix, separator, remainder = value.partition(":")
    if separator and prefix.strip().lower() in {"ability", "item", "move"}:
        value = remainder
    return _normalize_identifier(value)


def _canonical_protocol_signature(parts: Sequence[str]) -> str:
    """Return a stable tag-plus-meaningful-payload protocol census key.

    The static omission inventory distinguishes subtypes such as ``cant``
    reasons and ``-activate`` identifiers.  Counting only ``parts[1]`` here
    would make their real occurrence frequency invisible.  Names, slots,
    targets, and ``[from]`` annotations stay out of the key because they do
    not identify the protocol semantic being consumed.
    """

    event = parts[1] if len(parts) > 1 else ""
    if not event or event not in _SIGNATURE_PAYLOAD_EVENTS:
        return event
    # Most protocol events place their semantic identifier after the target.
    # Weather and field activation are field-scoped, carrying the effect as
    # their first payload. Perish Song uses ``-fieldactivate`` before its
    # per-Pokemon counter starts.
    payload_index = 2 if event in {"-fieldactivate", "-weather"} else 3
    if len(parts) <= payload_index:
        return event
    payload = parts[payload_index].strip()
    if not payload or payload.startswith("["):
        return event
    identifier = (
        _canonical_effect_identifier(payload)
        if event in _EFFECT_IDENTIFIER_EVENTS
        else _normalize_identifier(payload)
    )
    return f"{event}:{identifier}" if identifier else event


def _record_turn_cooccurrence(events: Sequence[str], report: DeepLineAuditReport) -> None:
    if not events:
        return
    unique = tuple(sorted(set(events)))
    report.protocol_cooccurrences[unique] += 1
    report.protocol_ordered_pairs.update(zip(events, events[1:]))
    report.protocol_ordered_triples.update(zip(events, events[1:], events[2:]))
    report.protocol_events.update(events)


def _audit_numeric_invariants(
    observation: PokeZeroObservationV0,
    player_id: str,
    turn: int,
    report: DeepLineAuditReport,
    *,
    check_randbat_candidates: bool,
) -> None:
    for token_index, row in enumerate(observation.numeric_features):
        for slot, value in enumerate(row):
            numeric = float(value)
            if not isfinite(numeric):
                report.add(
                    AuditFinding(
                        kind="non_finite_numeric",
                        player_id=player_id,
                        turn=turn,
                        column=f"token[{token_index}].numeric[{slot}]",
                        expected="finite",
                        actual=numeric,
                        detail="all production numeric features must be finite",
                    )
                )
            elif slot in _PUBLIC_UNIT_INTERVAL_SLOTS and not 0.0 <= numeric <= 1.0:
                report.add(
                    AuditFinding(
                        kind="out_of_bounds",
                        player_id=player_id,
                        turn=turn,
                        column=f"token[{token_index}].numeric[{slot}]",
                        expected="[0, 1]",
                        actual=numeric,
                        detail="normalized public feature escaped its declared interval",
                    )
                )
            elif slot in {NUMERIC_BOOST_ATK, NUMERIC_BOOST_DEF, NUMERIC_BOOST_SPA, NUMERIC_BOOST_SPD, NUMERIC_BOOST_SPE} and not -1.0 <= numeric <= 1.0:
                report.add(
                    AuditFinding(
                        kind="out_of_bounds",
                        player_id=player_id,
                        turn=turn,
                        column=f"token[{token_index}].numeric[{slot}]",
                        expected="[-1, 1]",
                        actual=numeric,
                        detail="normalized boost stage escaped its declared interval",
                    )
                )

    for token_index in range(OPPONENT_POKEMON_TOKEN_OFFSET, OPPONENT_POKEMON_TOKEN_OFFSET + 6):
        if _numeric(observation, token_index, NUMERIC_PRESENT) <= 0.0:
            continue
        candidate_count = _numeric(observation, token_index, NUMERIC_CANDIDATE_SET_COUNT)
        if check_randbat_candidates and candidate_count < 1.0:
            report.add(
                AuditFinding(
                    kind="empty_revealed_candidate_set",
                    player_id=player_id,
                    turn=turn,
                    column=f"token[{token_index}].numeric[{NUMERIC_CANDIDATE_SET_COUNT}]",
                    expected=">= 1 for a visible opponent Pokemon",
                    actual=candidate_count,
                    detail="a visible opponent must retain at least one candidate set",
                )
            )
        # A Toxic counter is allowed to be zero on the decision immediately after a
        # poisoned Pokemon switches in. Gen 3 resets its escalation stage on the
        # switch and increments it only at the next residual; a status-only
        # invariant cannot distinguish that valid boundary from a lost counter.


def _audit_public_transform_identity(
    observation: PokeZeroObservationV0,
    player_id: str,
    turn: int,
    report: DeepLineAuditReport,
) -> None:
    """Assert that a public Transform changes the active battle identity for both seats."""

    vocab = _category_vocab_for_observation(observation)
    team = observation.metadata.get("self_team")
    beliefs = observation.metadata.get("belief_view")
    if not isinstance(team, Sequence) or isinstance(team, (str, bytes)):
        return
    if not isinstance(beliefs, Mapping):
        return
    transformed_by_species = {
        _normalize_identifier(str(belief.get("species") or "")): str(belief.get("transform_species") or "")
        for belief in beliefs.get("self_pokemon", ())
        if isinstance(belief, Mapping) and belief.get("transformed") and belief.get("transform_species")
    }
    for slot_index, pokemon in enumerate(team[:6]):
        if not isinstance(pokemon, Mapping):
            continue
        original_species = _normalize_identifier(str(pokemon.get("species") or ""))
        copied_species = transformed_by_species.get(original_species)
        token_index = SELF_POKEMON_TOKEN_OFFSET + slot_index
        if not copied_species or _numeric(observation, token_index, NUMERIC_ACTIVE) <= 0.0:
            continue
        _compare(
            report,
            kind="self_transform_identity",
            player_id=player_id,
            turn=turn,
            column=f"token[{token_index}].species",
            expected=vocab.encode(f"species:{copied_species}"),
            actual=_categorical(observation, token_index, CATEGORY_PRIMARY),
            detail="a transformed self Pokemon must encode its public copied battle identity",
        )


def _audit_self_known_facts(
    observation: PokeZeroObservationV0,
    player_id: str,
    turn: int,
    report: DeepLineAuditReport,
) -> None:
    """Verify that self-request item and ability facts reach self Pokemon tokens."""

    team = observation.metadata.get("self_team")
    if not isinstance(team, Sequence) or isinstance(team, (str, bytes)):
        return
    vocab = _category_vocab_for_observation(observation)
    for slot_index, pokemon in enumerate(team[:6]):
        if not isinstance(pokemon, Mapping):
            continue
        token_index = SELF_POKEMON_TOKEN_OFFSET + slot_index
        ability = _normalize_identifier(str(pokemon.get("ability") or ""))
        item = _normalize_identifier(str(pokemon.get("item") or ""))
        if ability:
            _compare(
                report,
                kind="self_known_ability",
                player_id=player_id,
                turn=turn,
                column=f"token[{token_index}].ability",
                expected=vocab.encode(f"belief:possible_ability:{ability}"),
                actual=_categorical(observation, token_index, CATEGORY_BELIEF_ABILITY_OFFSET),
                detail="the self request's known ability must reach the self Pokemon token",
            )
            _compare(
                report,
                kind="self_known_ability",
                player_id=player_id,
                turn=turn,
                column=f"token[{token_index}].numeric[{NUMERIC_REVEALED_ABILITY}]",
                expected=1.0,
                actual=_numeric(observation, token_index, NUMERIC_REVEALED_ABILITY),
                detail="a self-request ability is always known, not merely a candidate",
            )
        if item:
            _compare(
                report,
                kind="self_known_item",
                player_id=player_id,
                turn=turn,
                column=f"token[{token_index}].item",
                expected=vocab.encode(f"belief:possible_item:{item}"),
                actual=_categorical(observation, token_index, CATEGORY_BELIEF_ITEM_OFFSET),
                detail="the self request's current held item must reach the self Pokemon token",
            )
            _compare(
                report,
                kind="self_known_item",
                player_id=player_id,
                turn=turn,
                column=f"token[{token_index}].numeric[{NUMERIC_REVEALED_ITEM}]",
                expected=1.0,
                actual=_numeric(observation, token_index, NUMERIC_REVEALED_ITEM),
                detail="a self-request item is always known while it is held",
            )


def _audit_request_action_surface(
    snapshot: LocalShowdownSnapshot,
    player_id: str,
    observation: PokeZeroObservationV0,
    turn: int,
    report: DeepLineAuditReport,
) -> None:
    """Compare action tokens against the raw Showdown request, not encoder helpers.

    The production action mask and action-token encoder intentionally share some
    request parsing helpers. Rebuilding the nine candidate slots here keeps the
    audit independent: a common parsing mistake must disagree with the bridge
    payload rather than merely agreeing with itself.
    """

    request = snapshot.latest_requests.get(player_id)
    if not isinstance(request, Mapping):
        return
    expected_mask = _raw_request_action_mask(request)
    for action_index, expected_legal in enumerate(expected_mask):
        token_index = ACTION_CANDIDATE_TOKEN_OFFSET + action_index
        _compare(
            report,
            kind="request_action_oracle",
            player_id=player_id,
            turn=turn,
            column=f"legal_action_mask[{action_index}]",
            expected=expected_legal,
            actual=bool(observation.legal_action_mask[action_index]),
            detail="the legal action mask must be reconstructed from the raw request",
        )
        _compare(
            report,
            kind="request_action_oracle",
            player_id=player_id,
            turn=turn,
            column=f"action_token[{action_index}].legal",
            expected=1.0 if expected_legal else 0.0,
            actual=_numeric(observation, token_index, NUMERIC_LEGAL),
            detail="each action token's legal bit must match the raw request",
        )

    active_rows = request.get("active")
    active = (
        active_rows[0]
        if isinstance(active_rows, list) and active_rows and isinstance(active_rows[0], Mapping)
        else None
    )
    moves = active.get("moves") if isinstance(active, Mapping) else None
    if not isinstance(moves, list):
        return
    for move_index, move in enumerate(moves[:MOVE_ACTION_COUNT]):
        if not isinstance(move, Mapping):
            continue
        pp, max_pp = move.get("pp"), move.get("maxpp")
        expected_pp_fraction = (
            max(0.0, min(1.0, float(pp) / float(max_pp)))
            if isinstance(pp, (int, float))
            and isinstance(max_pp, (int, float))
            and max_pp
            else 1.0
        )
        _compare(
            report,
            kind="request_action_oracle",
            player_id=player_id,
            turn=turn,
            column=f"action_token[{move_index}].pp_fraction",
            expected=expected_pp_fraction,
            actual=_numeric(
                observation,
                ACTION_CANDIDATE_TOKEN_OFFSET + move_index,
                NUMERIC_MOVE_PP_FRACTION,
            ),
            detail="move PP fraction must match the raw active-request move slot",
        )


def _raw_request_action_mask(request: Mapping[str, Any]) -> tuple[bool, ...]:
    """Independently derive the action convention from raw Showdown request fields."""

    mask = [False] * ACTION_COUNT
    if request.get("wait"):
        return tuple(mask)

    force_switch = request.get("forceSwitch")
    forced = isinstance(force_switch, list) and any(bool(value) for value in force_switch)
    active_rows = request.get("active")
    active = (
        active_rows[0]
        if isinstance(active_rows, list) and active_rows and isinstance(active_rows[0], Mapping)
        else None
    )
    if not forced and isinstance(active, Mapping):
        moves = active.get("moves")
        if isinstance(moves, list):
            for move_index, move in enumerate(moves[:MOVE_ACTION_COUNT]):
                if isinstance(move, Mapping) and not bool(move.get("disabled")):
                    mask[move_index] = True

    switch_allowed = forced or (
        isinstance(active, Mapping)
        and not bool(active.get("trapped"))
        and not bool(active.get("maybeTrapped"))
    )
    side = request.get("side")
    team = side.get("pokemon") if isinstance(side, Mapping) else None
    if not switch_allowed or not isinstance(team, list):
        return tuple(mask)
    active_team_index = next(
        (
            index
            for index, candidate in enumerate(team)
            if isinstance(candidate, Mapping) and bool(candidate.get("active"))
        ),
        None,
    )
    if active_team_index is None:
        return tuple(mask)
    switch_slot = 0
    for team_index, candidate in enumerate(team):
        if team_index == active_team_index:
            continue
        if switch_slot >= ACTION_COUNT - MOVE_ACTION_COUNT:
            break
        condition = str(candidate.get("condition") or "") if isinstance(candidate, Mapping) else "0 fnt"
        candidate_active = bool(candidate.get("active")) if isinstance(candidate, Mapping) else True
        mask[MOVE_ACTION_COUNT + switch_slot] = not candidate_active and not condition.startswith("0 ")
        switch_slot += 1
    return tuple(mask)


def _audit_snapshot_public_surface(
    env: LocalShowdownEnv,
    snapshot: LocalShowdownSnapshot,
    player_id: str,
    observation: PokeZeroObservationV0,
    report: DeepLineAuditReport,
) -> None:
    battle = _serialized_battle(snapshot)
    if battle is None:
        report.add(
            AuditFinding(
                kind="oracle_gap",
                player_id=player_id,
                turn=int(snapshot.replay.turn_number),
                column="bridge_snapshot.battle",
                expected="serialized battle",
                actual=None,
                detail="bridge snapshot lacked the omniscient oracle surface",
            )
        )
        return
    turn = int(snapshot.replay.turn_number)
    vocab = _category_vocab_for_env(env)
    dex = load_showdown_dex_cached(env.config.resolved_showdown_root())
    _compare(
        report,
        kind="bridge_oracle",
        player_id=player_id,
        turn=turn,
        column="field.turn_count",
        expected=min(1.0, float(battle.get("turn") or 0) / 1000.0),
        actual=_numeric(observation, FIELD_TOKEN_OFFSET, NUMERIC_TURN_COUNT),
        detail="field turn count must match the simulator decision boundary",
    )

    _audit_field_surface(
        observation,
        battle,
        player_id,
        turn,
        vocab,
        report,
    )

    sides = _sides_by_id(battle)
    self_side = sides.get(player_id)
    opponent_side = sides.get("p2" if player_id == "p1" else "p1")
    if self_side is not None:
        _audit_side_tokens(
            observation,
            self_side,
            SELF_POKEMON_TOKEN_OFFSET,
            observation.metadata.get("self_team"),
            player_id,
            turn,
            vocab,
            report,
            live_type_source=snapshot.replay.live_type_override.get(player_id),
            dex=dex,
        )
    if opponent_side is not None:
        _audit_side_tokens(
            observation,
            opponent_side,
            OPPONENT_POKEMON_TOKEN_OFFSET,
            observation.metadata.get("opponent_team"),
            player_id,
            turn,
            vocab,
            report,
            live_type_source=snapshot.replay.live_type_override.get(
                "p2" if player_id == "p1" else "p1"
            ),
            dex=dex,
        )


_TIMED_FIELD_SLOTS = (
    ("reflect", NUMERIC_SELF_REFLECT_TURNS, NUMERIC_OPP_REFLECT_TURNS),
    ("lightscreen", NUMERIC_SELF_LIGHT_SCREEN_TURNS, NUMERIC_OPP_LIGHT_SCREEN_TURNS),
    ("safeguard", NUMERIC_SELF_SAFEGUARD_TURNS, NUMERIC_OPP_SAFEGUARD_TURNS),
    ("mist", NUMERIC_SELF_MIST_TURNS, NUMERIC_OPP_MIST_TURNS),
)


def _audit_field_surface(
    observation: PokeZeroObservationV0,
    battle: Mapping[str, Any],
    player_id: str,
    turn: int,
    vocab: CategoryVocabulary,
    report: DeepLineAuditReport,
) -> None:
    """Compare encoded weather, hazards, and screens to bridge field state."""

    field = battle.get("field") if isinstance(battle.get("field"), Mapping) else {}
    weather = _normalize_identifier(str(field.get("weather") or ""))
    _compare(
        report,
        kind="bridge_field_oracle",
        player_id=player_id,
        turn=turn,
        column="field.weather",
        expected=vocab.encode(f"weather:{weather}") if weather else 0,
        actual=_categorical(observation, FIELD_TOKEN_OFFSET, CATEGORY_SECONDARY),
        detail="weather category must match the omniscient bridge field",
    )
    weather_state = field.get("weatherState") if isinstance(field.get("weatherState"), Mapping) else {}
    duration = weather_state.get("duration")
    # The live bridge serializes ability weather with ``duration: 0`` (rather
    # than omitting duration), while ordinary move weather has a positive
    # countdown. Both absent and non-positive duration therefore mean
    # permanent weather for this raw-state oracle.
    weather_has_countdown = isinstance(duration, (int, float)) and duration > 0
    expected_permanent = 1.0 if weather and not weather_has_countdown else 0.0
    expected_weather_turns = (
        max(0.0, min(1.0, float(duration) / 5.0))
        if weather_has_countdown
        else (1.0 if weather else 0.0)
    )
    _compare(
        report,
        kind="bridge_field_oracle",
        player_id=player_id,
        turn=turn,
        column="field.weather_turns",
        expected=expected_weather_turns,
        actual=_numeric(observation, FIELD_TOKEN_OFFSET, NUMERIC_WEATHER_TURNS),
        detail="weather duration must match the bridge weather state",
    )
    _compare(
        report,
        kind="bridge_field_oracle",
        player_id=player_id,
        turn=turn,
        column="field.weather_permanent",
        expected=expected_permanent,
        actual=_numeric(observation, FIELD_TOKEN_OFFSET, NUMERIC_WEATHER_PERMANENT),
        detail="ability weather must be represented as permanent",
    )

    sides = _sides_by_id(battle)
    self_counts = _raw_side_condition_counts(sides.get(player_id))
    opponent_id = "p2" if player_id == "p1" else "p1"
    opponent_counts = _raw_side_condition_counts(sides.get(opponent_id))
    for label, counts, hazard_slot, screen_slot, timed_slot_index in (
        ("self", self_counts, NUMERIC_SELF_HAZARDS, NUMERIC_SELF_SCREENS, 1),
        ("opponent", opponent_counts, NUMERIC_OPP_HAZARDS, NUMERIC_OPP_SCREENS, 2),
    ):
        expected_hazards = min(1.0, float(counts.get("spikes", 0)) / 3.0)
        expected_screens = min(
            1.0,
            float(sum(1 for condition in ("reflect", "lightscreen") if counts.get(condition))) / 2.0,
        )
        _compare(
            report,
            kind="bridge_field_oracle",
            player_id=player_id,
            turn=turn,
            column=f"field.{label}_hazards",
            expected=expected_hazards,
            actual=_numeric(observation, FIELD_TOKEN_OFFSET, hazard_slot),
            detail="Spikes layers must match the bridge side-condition state",
        )
        _compare(
            report,
            kind="bridge_field_oracle",
            player_id=player_id,
            turn=turn,
            column=f"field.{label}_screens",
            expected=expected_screens,
            actual=_numeric(observation, FIELD_TOKEN_OFFSET, screen_slot),
            detail="active Reflect/Light Screen flags must match bridge side conditions",
        )
        for condition, self_slot, opponent_slot in _TIMED_FIELD_SLOTS:
            value = counts.get(condition, 0)
            expected_duration = max(0.0, min(1.0, float(value) / 5.0)) if value else 0.0
            _compare(
                report,
                kind="bridge_field_oracle",
                player_id=player_id,
                turn=turn,
                column=f"field.{label}_{condition}_turns",
                expected=expected_duration,
                actual=_numeric(
                    observation,
                    FIELD_TOKEN_OFFSET,
                    self_slot if timed_slot_index == 1 else opponent_slot,
                ),
                detail="timed side-condition duration must match the bridge side-condition state",
            )


def _raw_side_condition_counts(side: Mapping[str, Any] | None) -> dict[str, int]:
    """Extract public side-condition layers/durations from a serialized bridge side."""

    conditions = side.get("sideConditions") if isinstance(side, Mapping) else None
    if not isinstance(conditions, Mapping):
        return {}
    counts: dict[str, int] = {}
    for raw_name, raw_state in conditions.items():
        name = _normalize_identifier(str(raw_name))
        if isinstance(raw_state, Mapping):
            value = raw_state.get("layers", raw_state.get("duration", 1))
        else:
            value = raw_state
        if isinstance(value, (int, float)) and value > 0:
            counts[name] = int(value)
    return counts


def _audit_side_tokens(
    observation: PokeZeroObservationV0,
    side: Mapping[str, Any],
    offset: int,
    source_team: Any,
    player_id: str,
    turn: int,
    vocab: CategoryVocabulary,
    report: DeepLineAuditReport,
    *,
    live_type_source: str | None,
    dex: Any,
) -> None:
    raw_pokemon = tuple(item for item in side.get("pokemon", ()) if isinstance(item, Mapping))
    source_rows = source_team if isinstance(source_team, Sequence) and not isinstance(source_team, (str, bytes)) else ()
    used: set[int] = set()
    for token_index in range(offset, offset + 6):
        if _numeric(observation, token_index, NUMERIC_PRESENT) <= 0.0:
            continue
        slot_index = token_index - offset
        source = source_rows[slot_index] if slot_index < len(source_rows) else None
        source_species = (
            _normalize_identifier(str(source.get("species") or ""))
            if isinstance(source, Mapping)
            else None
        )
        source_details = str(source.get("details") or "") if isinstance(source, Mapping) else None
        expected_species_id = _reverse_lookup(vocab, _categorical(observation, token_index, CATEGORY_PRIMARY))
        raw_index, raw = _match_serialized_pokemon(
            raw_pokemon,
            expected_species_id,
            used,
            source_species=source_species,
            source_details=source_details,
            expected_active=_numeric(observation, token_index, NUMERIC_ACTIVE) > 0.0,
        )
        if raw is None:
            continue
        used.add(raw_index)
        hp = _raw_hp_fraction(raw)
        if hp is not None:
            _compare(
                report,
                kind="bridge_oracle",
                player_id=player_id,
                turn=turn,
                column=f"token[{token_index}].hp_fraction",
                expected=hp,
                actual=_numeric(observation, token_index, NUMERIC_HP_FRACTION),
                detail="public HP fraction must match the simulator state",
            )
        active = 1.0 if _raw_or_request_active(raw, source) else 0.0
        _compare(
            report,
            kind="bridge_oracle",
            player_id=player_id,
            turn=turn,
            column=f"token[{token_index}].active",
            expected=active,
            actual=_numeric(observation, token_index, NUMERIC_ACTIVE),
            detail="active flag must match the simulator state",
        )
        fainted = bool(raw.get("fainted")) or (raw.get("hp") == 0)
        _compare(
            report,
            kind="bridge_oracle",
            player_id=player_id,
            turn=turn,
            column=f"token[{token_index}].legal",
            expected=0.0 if fainted else 1.0,
            actual=_numeric(observation, token_index, NUMERIC_LEGAL),
            detail="fainted Pokemon must be non-legal and living Pokemon legal",
        )
        if not fainted:
            _compare(
                report,
                kind="bridge_oracle",
                player_id=player_id,
                turn=turn,
                column=f"token[{token_index}].status",
                expected=vocab.encode(f"status:{_raw_status(raw)}"),
                actual=_categorical(observation, token_index, CATEGORY_SECONDARY),
                detail="public status category must match the simulator state",
            )
        is_forecast_form = bool(
            active and live_type_source and live_type_source.startswith("forme:")
        )
        if not raw.get("transformed"):
            _compare(
                report,
                kind="bridge_oracle",
                player_id=player_id,
                turn=turn,
                column=f"token[{token_index}].species",
                expected=_encode_species_category(
                    vocab, _raw_base_species(raw) if is_forecast_form else _raw_species(raw)
                ),
                actual=_categorical(observation, token_index, CATEGORY_PRIMARY),
                detail="non-transformed species category must match the simulator state",
            )
        if is_forecast_form and active:
            expected_types = _forecast_form_types(dex, live_type_source)
            if expected_types is None:
                report.add(
                    AuditFinding(
                        kind="oracle_gap",
                        player_id=player_id,
                        turn=turn,
                        column=f"token[{token_index}].forecast_type",
                        expected="resolvable Forecast form type",
                        actual=live_type_source,
                        detail="the audit must resolve every public Forecast form before validating it",
                    )
                )
            else:
                _compare(
                    report,
                    kind="bridge_oracle",
                    player_id=player_id,
                    turn=turn,
                    column=f"token[{token_index}].type1",
                    expected=vocab.encode(f"type:{expected_types[0]}"),
                    actual=_categorical(observation, token_index, CATEGORY_TYPE_1),
                    detail="Forecast form must update the active type slot",
                )
                _compare(
                    report,
                    kind="bridge_oracle",
                    player_id=player_id,
                    turn=turn,
                    column=f"token[{token_index}].type2",
                    expected=(
                        vocab.encode(f"type:{expected_types[1]}")
                        if len(expected_types) > 1
                        else vocab.encode("")
                    ),
                    actual=_categorical(observation, token_index, CATEGORY_TYPE_2),
                    detail="Forecast form must update the active type slot",
                )
        level = _raw_level(raw)
        if level is not None:
            _compare(
                report,
                kind="bridge_oracle",
                player_id=player_id,
                turn=turn,
                column=f"token[{token_index}].level",
                expected=min(1.0, float(level) / 100.0),
                actual=_numeric(observation, token_index, NUMERIC_LEVEL),
                detail="level must match the simulator's generated set",
            )
        if raw.get("isActive"):
            boosts = raw.get("boosts") if isinstance(raw.get("boosts"), Mapping) else {}
            for stat, slot in _BOOST_SLOTS:
                _compare(
                    report,
                    kind="bridge_oracle",
                    player_id=player_id,
                    turn=turn,
                    column=f"token[{token_index}].boost.{stat}",
                    expected=max(-1.0, min(1.0, float(boosts.get(stat, 0)) / 6.0)),
                    actual=_numeric(observation, token_index, slot),
                    detail="active boost stage must match the simulator state",
                )


def _audit_candidate_monotonicity(
    observation: PokeZeroObservationV0,
    player_id: str,
    turn: int,
    report: DeepLineAuditReport,
) -> None:
    vocab = _category_vocab_for_observation(observation)
    for token_index in range(OPPONENT_POKEMON_TOKEN_OFFSET, OPPONENT_POKEMON_TOKEN_OFFSET + 6):
        if _numeric(observation, token_index, NUMERIC_PRESENT) <= 0.0:
            continue
        species = _reverse_lookup(vocab, _categorical(observation, token_index, CATEGORY_PRIMARY))
        if not species:
            continue
        key = (player_id, species)
        current = int(_numeric(observation, token_index, NUMERIC_CANDIDATE_SET_COUNT))
        prior = report.candidate_count_history.get(key)
        if prior is not None and current > prior:
            report.add(
                AuditFinding(
                    kind="candidate_count_increased",
                    player_id=player_id,
                    turn=turn,
                    column=f"opponent:{species}.candidate_set_count",
                    expected=f"<= {prior}",
                    actual=current,
                    detail="candidate sets may collapse as public evidence arrives but may not expand",
                )
            )
        report.candidate_count_history[key] = current


def _audit_incremental_vs_batch(
    env: LocalShowdownEnv,
    snapshot: LocalShowdownSnapshot,
    player_id: str,
    live: PokeZeroObservationV0,
    turn: int,
    report: DeepLineAuditReport,
) -> None:
    state = normalize_for_player(
        snapshot.replay,
        player_id=player_id,
        configured_showdown_slot=player_id,
        format_id=snapshot.observation_format_id,
        set_source=env._belief_set_source,
        include_turn_merged=(
            env.config.observation_spec.schema_version
            in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
        ),
    )
    batch = observation_from_player_state(
        state,
        category_vocab=_category_vocab_for_env(env),
        spec=env.config.observation_spec,
        dex=load_showdown_dex_cached(env.config.resolved_showdown_root()),
        feature_masks=env.config.feature_masks,
    )
    _compare_observation_arrays(
        report,
        kind="incremental_vs_batch",
        player_id=player_id,
        turn=turn,
        expected=batch,
        actual=live,
        detail="fresh replay/belief fold must match the incremental encoder outside live-only annotations",
        ignored_numeric_columns=_LIVE_INCREMENTAL_ANNOTATION_COLUMNS,
    )


def _audit_snapshot_roundtrip(
    env: LocalShowdownEnv,
    snapshot: LocalShowdownSnapshot,
    player_id: str,
    before: PokeZeroObservationV0,
    turn: int,
    report: DeepLineAuditReport,
) -> None:
    env.restore(snapshot)
    after = attach_audit_vocabulary(env.observe(player_id), _category_vocab_for_env(env))
    _compare_observation_arrays(
        report,
        kind="snapshot_vs_live",
        player_id=player_id,
        turn=turn,
        expected=before,
        actual=after,
        detail="restoring a live snapshot must preserve the exact observation",
    )


def _compare_observation_arrays(
    report: DeepLineAuditReport,
    *,
    kind: str,
    player_id: str,
    turn: int,
    expected: PokeZeroObservationV0,
    actual: PokeZeroObservationV0,
    detail: str,
    ignored_numeric_columns: frozenset[int] = frozenset(),
) -> None:
    surfaces = (
        ("categorical_ids", expected.categorical_ids, actual.categorical_ids),
        ("numeric_features", expected.numeric_features, actual.numeric_features),
        ("token_type_ids", expected.token_type_ids, actual.token_type_ids),
        ("attention_mask", expected.attention_mask, actual.attention_mask),
        ("legal_action_mask", expected.legal_action_mask, actual.legal_action_mask),
    )
    for name, expected_value, actual_value in surfaces:
        equal = expected_value == actual_value
        if name == "numeric_features" and ignored_numeric_columns:
            equal = _numeric_features_equal_except(
                expected_value, actual_value, ignored_numeric_columns
            )
        if equal:
            continue
        report.add(
            AuditFinding(
                kind=kind,
                player_id=player_id,
                turn=turn,
                column=name,
                expected=expected_value,
                actual=actual_value,
                detail=detail,
            )
        )


def _numeric_features_equal_except(
    expected: Sequence[Sequence[float]],
    actual: Sequence[Sequence[float]],
    ignored_columns: frozenset[int],
) -> bool:
    """Compare numeric arrays while excluding documented live-only annotations."""

    if len(expected) != len(actual):
        return False
    for expected_row, actual_row in zip(expected, actual):
        if len(expected_row) != len(actual_row):
            return False
        if any(
            expected_value != actual_value
            for column, (expected_value, actual_value) in enumerate(zip(expected_row, actual_row))
            if column not in ignored_columns
        ):
            return False
    return True


def _forecast_form_types(dex: Any, live_type_source: str) -> tuple[str, ...] | None:
    """Resolve a public Forecast form independently from the production encoder.

    The Gen 3 surface is deliberately small, but an audit must fail closed when a
    public form cannot be interpreted. The dex is the primary authority; the
    explicit fallback keeps the oracle live if a dex serialization omits Castform's
    weather formes.
    """

    form = live_type_source.removeprefix("forme:").strip()
    if not form:
        return None
    info = dex.species_info(form) if dex is not None else None
    if info is None and dex is not None:
        canonical = canonical_gen3_randbat_species_id(_normalize_identifier(form))
        if canonical and canonical != _normalize_identifier(form):
            info = dex.species_info(canonical)
    if info is not None and info.types:
        return tuple(info.types)
    return _FORECAST_FORM_TYPE_FALLBACK.get(_normalize_identifier(form))


def _serialized_battle(snapshot: LocalShowdownSnapshot) -> Mapping[str, Any] | None:
    battle = snapshot.bridge_snapshot.get("battle")
    return battle if isinstance(battle, Mapping) else None


def _sides_by_id(battle: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(side.get("id")): side
        for side in battle.get("sides", ())
        if isinstance(side, Mapping) and isinstance(side.get("id"), str)
    }


def _raw_species(pokemon: Mapping[str, Any]) -> str:
    raw = str(pokemon.get("species") or "")
    if raw.startswith("[Species:") and raw.endswith("]"):
        raw = raw[len("[Species:") : -1]
    return canonical_gen3_randbat_species_id(_normalize_identifier(raw))


def _condition_status(condition: str | None) -> str:
    values = str(condition or "").split()
    return values[1] if len(values) > 1 and values[1] != "fnt" else "none"


def _condition_is_fainted(condition: str | None) -> bool:
    return "fnt" in str(condition or "").split()


def _raw_base_species(pokemon: Mapping[str, Any]) -> str:
    raw = str(pokemon.get("baseSpecies") or pokemon.get("species") or "")
    if raw.startswith("[Species:") and raw.endswith("]"):
        raw = raw[len("[Species:") : -1]
    return canonical_gen3_randbat_species_id(_normalize_identifier(raw))


def _raw_status(pokemon: Mapping[str, Any]) -> str:
    raw = _normalize_identifier(str(pokemon.get("status") or ""))
    return "none" if not raw or raw == "fnt" else raw


def _raw_or_request_active(raw: Mapping[str, Any], source: Any) -> bool:
    """Preserve the request-visible fainted active during a forced replacement.

    The simulator clears ``isActive`` before it asks the owning player to choose a
    replacement, while that player's request still marks the fainted outgoing
    Pokemon active. The encoder deliberately follows the request at that
    decision boundary; treating the bridge field as the only oracle would flag
    the intended forced-switch representation as an inconsistency.
    """

    if raw.get("isActive"):
        return True
    return bool(
        isinstance(source, Mapping)
        and source.get("active")
        and source.get("fainted")
        and (raw.get("fainted") or raw.get("hp") == 0)
    )


def _raw_hp_fraction(pokemon: Mapping[str, Any]) -> float | None:
    hp, max_hp = pokemon.get("hp"), pokemon.get("maxhp")
    if not isinstance(hp, (int, float)) or not isinstance(max_hp, (int, float)) or max_hp <= 0:
        return None
    return max(0.0, min(1.0, float(hp) / float(max_hp)))


def _raw_level(pokemon: Mapping[str, Any]) -> int | None:
    set_payload = pokemon.get("set")
    if not isinstance(set_payload, Mapping):
        return None
    level = set_payload.get("level")
    return int(level) if isinstance(level, int) and level > 0 else None


def _encode_species_category(vocab: CategoryVocabulary, species_id: str) -> int:
    """Encode a bridge species id using the vocabulary's canonical display spelling.

    Showdown's serialized battle uses compact ids (for example ``deoxysspeed``),
    while the category vocabulary retains selected public punctuation
    (``species:deoxys-speed``).  Match by normalized spelling only for the
    audit oracle; this is not a production species canonicalization path.
    """

    direct = f"species:{species_id}"
    if direct in vocab.tokens:
        return vocab.encode(direct)
    for token in vocab.tokens:
        if not token.startswith("species:"):
            continue
        if _normalize_identifier(token.removeprefix("species:")) == species_id:
            return vocab.encode(token)
    return vocab.encode(direct)


def _match_serialized_pokemon(
    candidates: Sequence[Mapping[str, Any]],
    expected_species_id: str | None,
    used: set[int],
    *,
    source_species: str | None,
    source_details: str | None,
    expected_active: bool,
) -> tuple[int, Mapping[str, Any] | None]:
    expected = expected_species_id.removeprefix("species:") if expected_species_id else ""
    best: tuple[int, Mapping[str, Any]] | None = None
    best_score = -1
    for index, candidate in enumerate(candidates):
        if index in used:
            continue
        base_match = bool(source_species) and _raw_base_species(candidate) == source_species
        battle_match = bool(expected) and _raw_species(candidate) == expected
        if not base_match and not battle_match:
            continue
        score = (8 if base_match else 0) + (4 if battle_match else 0)
        if source_details and str(candidate.get("details") or "") == source_details:
            score += 2
        if bool(candidate.get("isActive")) == expected_active:
            score += 1
        if score > best_score:
            best = (index, candidate)
            best_score = score
    if best is not None:
        return best
    return -1, None


def _category_vocab_for_env(env: LocalShowdownEnv) -> CategoryVocabulary:
    return env.config.category_vocab or gen3_category_vocabulary(
        env.config.resolved_showdown_root(),
        include_turn_merged=(
            env.config.observation_spec.schema_version
            in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
        ),
    )


def _category_vocab_for_observation(observation: PokeZeroObservationV0) -> CategoryVocabulary:
    # This fallback is used only by pure invariant calls.  Live audit calls patch
    # in a cached vocabulary through the encoded row IDs, so it remains cheap.
    vocab = observation.metadata.get("_deep_line_audit_vocab")
    if isinstance(vocab, CategoryVocabulary):
        return vocab
    raise ValueError("deep-line audit requires the category vocabulary in observation metadata")


def attach_audit_vocabulary(
    observation: PokeZeroObservationV0,
    vocab: CategoryVocabulary,
) -> PokeZeroObservationV0:
    """Return an audit-only metadata wrapper without changing encoded tensors."""

    from dataclasses import replace

    return replace(observation, metadata={**dict(observation.metadata), "_deep_line_audit_vocab": vocab})


def _numeric(observation: PokeZeroObservationV0, token: int, slot: int) -> float:
    return float(observation.numeric_features[token][slot])


def _active_pokemon_token(observation: PokeZeroObservationV0, offset: int) -> int | None:
    for token_index in range(offset, offset + 6):
        if _numeric(observation, token_index, NUMERIC_ACTIVE) > 0.0:
            return token_index
    return None


def _categorical(observation: PokeZeroObservationV0, token: int, slot: int) -> int:
    return int(observation.categorical_ids[token][slot])


def _reverse_lookup(vocab: CategoryVocabulary, encoded: int) -> str | None:
    if encoded <= 0 or encoded > len(vocab.tokens):
        return None
    return vocab.tokens[encoded - 1]


def _compare(
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
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            if abs(float(expected) - float(actual)) <= 1e-12:
                return
        except (TypeError, ValueError):
            pass
    elif expected == actual:
        return
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
