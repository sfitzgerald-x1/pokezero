"""Engine-as-environment self-play driver: games advanced by the Rust engine.

:class:`LocalShowdownEnv` plays a game by talking to a Node Showdown process
over a pipe — one boundary round-trip per decision, plus a protocol parse and a
Python observation encode. :class:`EngineEnv` is the same
:class:`~pokezero.env.PokeZeroEnv` contract with Node removed from the
transition loop: the vendored gen3-patched poke-engine advances the battle, and
the crate's own encoder produces the observation. It exists to answer ONE
question — what does a self-play collection loop cost when Showdown is not in
it — so it is deliberately scoped to that measurement.

Where each piece comes from
---------------------------

* **Teams** still come from Showdown. ``generate_scenario_team`` is one bridge
  RPC per seat per GAME, which is amortized noise against a ~60-decision
  episode, and a native gen3 randbats generator would be a large surface with
  its own fidelity risk for zero effect on the number being measured.
* **Transitions** come from ``pokezero_search.env_step``: the engine enumerates
  the chance branches for the joint action, one is sampled with a per-ply seed
  derived from the episode seed, and the realized branch is rendered to
  protocol lines and applied.
* **Observations** come from ``pokezero_search.LeafEncoder.encode_leaf`` — the
  same native encoder the search stack prices leaves with, held at
  ``engine_authoritative=True`` because every encode here is a real decision
  point, not a search leaf over a reconstructed world.
* **Legality** comes from the engine's own option surface, which is also what
  the encoder writes into the action block, so the mask an agent sees and the
  actions the env accepts are derived from one source.

The information-state discipline
--------------------------------

The env holds both teams in full; the observation must not. The native encoder
recomputes the state-dependent columns from the engine state but carries the
*identity* columns (which opponent mons exist, their moves/ability/item)
verbatim from the root row inputs — so those inputs are exactly where leakage
would happen, and exactly where it is prevented. :class:`_PublicLedger` replays
the same synthesized protocol lines that drive the fold and admits an opponent
mon (or one of its moves, its ability, its item) into the row inputs only once
a line has publicly revealed it. An unrevealed opponent mon is absent from
``opponent_team`` entirely, which is what Showdown's own path produces.

Root semantics
--------------

The encoder is always rooted at the GAME-START state with the full accumulated
line history replayed on top. That is not an accident of convenience: the
encoder's delta families (PP ledger, toxic stage, sleep counters) are defined
as "root value + engine delta", and its evolve-on-change rules treat the root
strings as authoritative until the engine moves that mon. Anchoring the root at
turn 1 — where every one of those values is trivially known — makes both rules
exact by construction. Only the public-knowledge sets grow, and those are
precisely the columns the encoder never evolves on its own.

Known fidelity residuals (smoke-grade)
--------------------------------------

* **Belief candidate sets are degenerate.** The Showdown collector runs
  :class:`~pokezero.belief.PublicBattleBeliefEngine` over parsed public events,
  which narrows a randbats candidate universe and drives the uncertainty /
  possible-move / possible-item columns. This env tracks *revealed facts* only;
  candidate sets are empty and opponent ``uncertainty`` is pinned at 1.0. Those
  columns therefore carry a different distribution than a Showdown-collected
  shard. Shapes, dtypes and field names are unaffected — this is train/serve
  skew, not a schema difference, and it is the first thing to fix before the
  data is used for anything but timing.
* **Forced-replacement residual ordering** depends on the engine patch set; see
  the ``scott/engine-gen3-spikes-residuals`` branch.
* **No endless-battle clause.** Showdown ends a non-progressing game; the
  engine does not, so a stalled position (mutual recovery, random-legal play)
  runs forever. ``EngineEnvConfig.max_plies`` is the backstop and reports
  ``TerminalState(capped=True)``, which the dataset layer already prices via
  ``--capped-terminal-value``. Measured incidence with a random-legal policy:
  2 of 120 seeds exceeded 1500 plies against a median of 80.
* ``public_materialization_state`` is not implemented, so ``engine-mcts:``
  policies would silently lose their search context. The env raises instead.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .actions import ACTION_COUNT
from .env import BattleFormat, PlayerId, StepResult, TerminalState
from .observation import (
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    ObservationFeatureMasks,
    ObservationPerspective,
    ObservationSpec,
    PokeZeroObservationV0,
)

# Little-endian dtypes of the five encoder buffers, in the crate's own order.
# Sourced from the golden corpus so a dtype drift breaks one constant, not two.
from .golden_corpus import GOLDEN_ARRAY_FIELDS

_ARRAY_DTYPES: Mapping[str, str] = {name: dtype for name, dtype, _ in GOLDEN_ARRAY_FIELDS}
_ARRAY_NAMES: tuple[str, ...] = tuple(name for name, _, _ in GOLDEN_ARRAY_FIELDS)

_PLAYERS: tuple[PlayerId, PlayerId] = ("p1", "p2")
# Gen 3 randbats sets are always neutral-natured; the stat replication below
# assumes it, so a non-neutral set must fail loudly rather than encode wrong
# stats. (Showdown's gen3 generator emits one of these two.)
_NEUTRAL_NATURES = frozenset({"", "serious", "hardy", "docile", "bashful", "quirky"})


class EngineEnvError(RuntimeError):
    """The engine environment could not advance or observe a battle."""


class EngineEnvUnsupportedError(EngineEnvError):
    """A capability the Showdown env has that this env deliberately lacks."""


def _normalize(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _opponent_of(player: PlayerId) -> PlayerId:
    return "p2" if player == "p1" else "p1"


# Domain separators, so two different uses of the same episode seed cannot
# collide. Without them the p1 team draw and the first chance outcome would
# share a seed.
_SEED_DOMAIN_PLY = 0x0000000000000000
_SEED_DOMAIN_TEAM = 0x9E3779B97F4A7C15


def _derived_seed(episode_seed: int, index: int, domain: int = _SEED_DOMAIN_PLY) -> int:
    """A stable derived seed. SplitMix64 finalizer over (seed, index, domain).

    Cheap enough to call every ply and avalanche-y enough that consecutive
    indices do not correlate — the trajectory must be reproducible from the
    episode seed, not merely deterministic.
    """
    x = (int(episode_seed) * 0x9E3779B97F4A7C15 + int(index) + 1 + domain) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class EngineEnvTimings:
    """Wall-clock split of the env's own work, in seconds.

    The point of this env is a number, so the number is instrumented rather
    than inferred. ``encode`` covers the native encode AND the row-inputs
    rebuild it needs; ``step`` is the engine transition; ``action_map`` is
    translating a chosen action index back into an engine option;
    ``materialize`` is converting the encoder's buffers into the nested
    sequences downstream expects; ``teams`` is
    the once-per-game Showdown bridge round-trip; ``ledger`` is the public-info
    replay. Policy forward time is NOT here — it belongs to the caller.
    """

    teams: float = 0.0
    step: float = 0.0
    encode: float = 0.0
    ledger: float = 0.0
    action_map: float = 0.0
    materialize: float = 0.0
    encode_calls: int = 0
    step_calls: int = 0
    games: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "teams_s": self.teams,
            "step_s": self.step,
            "encode_s": self.encode,
            "ledger_s": self.ledger,
            "action_map_s": self.action_map,
            "materialize_s": self.materialize,
            "encode_calls": float(self.encode_calls),
            "step_calls": float(self.step_calls),
            "games": float(self.games),
        }


# ---------------------------------------------------------------------------
# Public-information ledger
# ---------------------------------------------------------------------------


@dataclass
class _MonReveal:
    """What the PUBLIC transcript has disclosed about one Pokemon."""

    revealed: bool = False
    moves: list[str] = field(default_factory=list)
    ability: str | None = None
    item: str | None = None

    def add_move(self, move_id: str) -> bool:
        if not move_id or move_id == "struggle" or move_id in self.moves:
            return False
        self.moves.append(move_id)
        return True


class _PublicLedger:
    """Reveal state for both seats, replayed from synthesized protocol lines.

    Only the handful of line kinds that disclose *identity* are read. Anything
    state-shaped (damage, boosts, status, hazards) is deliberately ignored:
    the encoder recomputes all of that from the engine state, and duplicating
    it here would create a second, divergent source of truth.

    ``version`` increments on every disclosure so callers can cache anything
    derived from the ledger and rebuild only when it actually moved.
    """

    def __init__(self, party_species: Mapping[str, Sequence[str]]) -> None:
        self._mons: dict[str, dict[str, _MonReveal]] = {
            player: {_normalize(species): _MonReveal() for species in party_species[player]}
            for player in _PLAYERS
        }
        self.version = 0

    def reveal_lead(self, player: PlayerId, species: str) -> None:
        """Both leads are public before the first decision (Showdown's ``|switch|``)."""
        self._touch(player, _normalize(species))

    def _entry(self, player: PlayerId, species_key: str) -> _MonReveal | None:
        return self._mons.get(player, {}).get(species_key)

    def _touch(self, player: PlayerId, species_key: str) -> None:
        entry = self._entry(player, species_key)
        if entry is not None and not entry.revealed:
            entry.revealed = True
            self.version += 1

    def revealed_species(self, player: PlayerId) -> tuple[str, ...]:
        return tuple(key for key, entry in self._mons[player].items() if entry.revealed)

    def facts(self, player: PlayerId, species: str) -> _MonReveal | None:
        return self._entry(player, _normalize(species))

    def is_revealed(self, player: PlayerId, species: str) -> bool:
        entry = self._entry(player, _normalize(species))
        return bool(entry and entry.revealed)

    def ingest(self, lines: Iterable[str]) -> None:
        for line in lines:
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            # parts[0] is the empty string before the leading pipe.
            if len(parts) < 3:
                continue
            kind = parts[1]
            if kind in {"switch", "drag", "replace"}:
                self._ingest_ident(parts[2])
            elif kind == "move":
                self._ingest_move(parts)
            elif kind in {"-ability", "-item", "-enditem"}:
                self._ingest_trait(kind, parts)

    def _split_ident(self, ident: str) -> tuple[PlayerId, str] | None:
        # "p1a: Swampert" / "p1: Swampert"
        prefix, _, name = str(ident).partition(":")
        player = prefix.strip()[:2]
        if player not in _PLAYERS:
            return None
        return player, _normalize(name)

    def _ingest_ident(self, ident: str) -> None:
        split = self._split_ident(ident)
        if split is not None:
            self._touch(*split)

    def _ingest_move(self, parts: list[str]) -> None:
        split = self._split_ident(parts[2])
        if split is None or len(parts) < 4:
            return
        player, species_key = split
        self._touch(player, species_key)
        # A called or locked continuation is not a PP-charging reveal of a
        # move slot; the parser's own rule (see LeafMeta's PP replay).
        tail = "|".join(parts[4:])
        if "[from]" in tail:
            return
        entry = self._entry(player, species_key)
        if entry is not None and entry.add_move(_normalize(parts[3])):
            self.version += 1

    def _ingest_trait(self, kind: str, parts: list[str]) -> None:
        split = self._split_ident(parts[2])
        if split is None or len(parts) < 4:
            return
        player, species_key = split
        entry = self._entry(player, species_key)
        if entry is None:
            return
        value = _normalize(parts[3])
        if not value:
            return
        if kind == "-ability":
            if entry.ability != value:
                entry.ability = value
                self.version += 1
        elif entry.item != value:
            # Both |-item| (revealed, e.g. Trick/Knock Off) and |-enditem|
            # (consumed) disclose WHICH item it was. Currency (still held vs
            # already gone) is a belief-layer concern this smoke env does not
            # model; see the module docstring.
            entry.item = value
            self.version += 1


# ---------------------------------------------------------------------------
# Team materialization
# ---------------------------------------------------------------------------


def _gen3_stat(base: int, iv: int, ev: int, level: int) -> int:
    return int(int(2 * base + iv + int(ev / 4)) * level / 100 + 5)


def _gen3_hp_stat(base: int, iv: int, ev: int, level: int) -> int:
    return int(int(2 * base + iv + int(ev / 4) + 100) * level / 100 + 10)


_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


@dataclass(frozen=True)
class _GeneratedMon:
    """One generated set, resolved against the dex into everything both the
    engine constructor and the row-inputs metadata need.

    THREE species spellings, deliberately: ``species`` is the generator's
    display name (``"Unown-K"``) and is what the protocol renderer and the
    observation metadata must use; ``species_key`` is its normalized form
    (``"unownk"``) and is the ledger / md-team matching key; ``engine_id``
    collapses cosmetic formes to what the dex and the engine actually know
    (``"unown"``). Mixing them up is the `self_world_mismatch` class of bug
    documented in :mod:`pokezero.engine_world`, whose ``_engine_species_id``
    defines the collapse.
    """

    species: str
    species_key: str
    engine_id: str
    level: int
    gender: str
    ability: str
    item: str
    #: Showdown move ids, generator order — what the protocol, the ledger and
    #: the observation metadata use (``hiddenpowerfighting``).
    moves: tuple[str, ...]
    #: The same moves as poke-engine spells them, index-aligned with `moves`.
    #: Hidden Power is typed+BP on the engine side (``hiddenpowerfighting70``);
    #: handing the engine the Showdown spelling yields an EMPTY move slot, and
    #: a mon whose only move is Hidden Power then has no legal action at all.
    engine_moves: tuple[str, ...]
    move_pp: Mapping[str, int]
    types: tuple[str, ...]
    stats: Mapping[str, int]
    weight_kg: float

    @property
    def details(self) -> str:
        """Showdown's ``details`` string: species, level (omitted at 100), gender."""
        level = "" if self.level == 100 else f", L{self.level}"
        gender = f", {self.gender}" if self.gender in {"M", "F"} else ""
        return f"{self.species}{level}{gender}"

    @property
    def max_hp(self) -> int:
        return int(self.stats["hp"])


def _engine_move_id(move_id: str, ivs: Mapping[str, int], species: str) -> str:
    """One Showdown move id as poke-engine spells it.

    Only Hidden Power differs today: the engine keys it by type AND base
    power, both of which are IV-derived. `engine_world.hidden_power_engine_id`
    is the single derivation (it also fails closed when the set's declared
    type disagrees with its IVs), reused rather than re-implemented.
    """
    if not move_id.startswith("hiddenpower"):
        return move_id
    from .engine_world import EngineWorldUnsupported, hidden_power_engine_id  # noqa: PLC0415

    try:
        return hidden_power_engine_id(move_id, ivs)
    except EngineWorldUnsupported as exc:
        raise EngineEnvError(f"{species}: {move_id} is inconsistent with its IVs ({exc})") from exc


def _generated_mon(row: Mapping[str, Any], dex: Any) -> _GeneratedMon:
    from .engine_world import _engine_species_id  # noqa: PLC0415 — one collapse rule, not two

    species = str(row.get("species") or "")
    engine_id = _engine_species_id(_normalize(species))
    info = dex.species_info(engine_id)
    if info is None:
        raise EngineEnvError(
            f"generated species {species!r} (engine id {engine_id!r}) is not in the gen3 dex"
        )
    nature = _normalize(row.get("nature"))
    if nature not in _NEUTRAL_NATURES:
        # Fail closed: the stat replication below is the neutral-nature formula.
        raise EngineEnvError(
            f"generated {species} carries non-neutral nature {row.get('nature')!r}; "
            "the gen3 stat replication in engine_env assumes neutral natures"
        )
    level = int(row.get("level") or 100)
    evs = {key: int((row.get("evs") or {}).get(key, 0)) for key in _STAT_KEYS}
    ivs = {key: int((row.get("ivs") or {}).get(key, 31)) for key in _STAT_KEYS}
    base = info.base_stats
    stats = {
        key: _gen3_stat(int(base.get(key, 0)), ivs[key], evs[key], level)
        for key in _STAT_KEYS
        if key != "hp"
    }
    base_hp = int(base.get("hp", 0))
    # Shedinja is the only base-1-HP species; the engine pins its max HP to 1.
    stats["hp"] = 1 if base_hp == 1 else _gen3_hp_stat(base_hp, ivs["hp"], evs["hp"], level)

    moves = tuple(_normalize(move) for move in (row.get("moves") or ()))
    move_pp: dict[str, int] = {}
    engine_moves: list[str] = []
    for move in moves:
        move_info = dex.move_info(move)
        move_pp[move] = int(getattr(move_info, "max_pp", 0) or 0) or 32
        engine_moves.append(_engine_move_id(move, ivs, species))
    return _GeneratedMon(
        species=species,
        species_key=_normalize(species),
        engine_id=engine_id,
        level=level,
        gender=str(row.get("gender") or ""),
        ability=_normalize(row.get("ability")),
        item=_normalize(row.get("item")),
        moves=moves,
        engine_moves=tuple(engine_moves),
        move_pp=move_pp,
        types=tuple(str(t).lower() for t in info.types),
        stats=stats,
        weight_kg=float(getattr(info, "weight_kg", 0.0) or 0.0),
    )


def _battle_spec(p1: Sequence[_GeneratedMon], p2: Sequence[_GeneratedMon]) -> Any:
    from .poke_engine_adapter import BattleSpec, MoveSpec, PokemonSpec, SideSpec

    def side(party: Sequence[_GeneratedMon]) -> Any:
        return SideSpec(
            pokemon=tuple(
                PokemonSpec(
                    # Engine-facing: cosmetic formes collapsed.
                    id=mon.engine_id,
                    level=mon.level,
                    types=mon.types,
                    hp=mon.max_hp,
                    maxhp=mon.max_hp,
                    attack=int(mon.stats["atk"]),
                    defense=int(mon.stats["def"]),
                    special_attack=int(mon.stats["spa"]),
                    special_defense=int(mon.stats["spd"]),
                    speed=int(mon.stats["spe"]),
                    moves=tuple(
                        MoveSpec(id=engine_move, pp=int(mon.move_pp.get(move, 32)))
                        for move, engine_move in zip(mon.moves, mon.engine_moves)
                    ),
                    ability=mon.ability or None,
                    item=mon.item or None,
                    gender=mon.gender or None,
                    weight_kg=mon.weight_kg or None,
                )
                for mon in party
            ),
            active_index=0,
        )

    return BattleSpec(side_one=side(p1), side_two=side(p2))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineEnvConfig:
    """Construction options for :class:`EngineEnv`."""

    showdown_root: Path | None = None
    node_binary: str = "node"
    encoder_tables: Path | None = None
    observation_spec: ObservationSpec | None = None
    feature_masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS
    # Damage-roll branching in the transition. True matches what the engine's
    # own MCTS does near the root and is the faithful chance surface; False is
    # a cheaper coarse mode kept only for benchmarking the sampler's cost.
    branch_on_damage: bool = True
    # Hard stop on engine plies per game. The rollout driver enforces the real
    # decision cap; this only stops a pathological non-terminating loop.
    max_plies: int = 2000

    def resolved_showdown_root(self) -> Path:
        from .local_showdown import LocalShowdownConfig

        return LocalShowdownConfig(showdown_root=self.showdown_root).resolved_showdown_root()


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


class EngineEnv:
    """A :class:`~pokezero.env.PokeZeroEnv` whose transitions are the engine's.

    Not thread-safe; use one instance per worker (which is what
    :class:`~pokezero.collection.ReusableEnvPool` does).
    """

    def __init__(self, config: EngineEnvConfig | None = None) -> None:
        self.config = config or EngineEnvConfig()
        self.timings = EngineEnvTimings()

        import pokezero_search  # noqa: PLC0415 — native module, imported lazily

        self._native = pokezero_search
        self._dex = None
        self._tables_json: str | None = None
        self._team_env: Any = None

        self._spec: ObservationSpec | None = None
        self._battle_id = ""
        self._root_state_str = ""
        self._seed = 0
        self._format_id = "gen3randombattle"
        self._state_str = ""
        self._turn = 1
        self._ply = 0
        self._terminal: TerminalState | None = None
        self._party: dict[PlayerId, tuple[_GeneratedMon, ...]] = {}
        self._lines: list[str] = []
        self._folds: dict[PlayerId, Any] = {}
        self._ledger: _PublicLedger | None = None
        self._encoders: dict[PlayerId, Any] = {}
        self._encoder_version = -1
        # Survives reset(): the parsed tables inside it are the expensive part
        # and are battle-independent.
        self._encoder_template: Any = None
        self._observations: dict[PlayerId, PokeZeroObservationV0] = {}
        self._options: dict[str, Any] = {}

    # -- lazily-built shared resources ------------------------------------

    def _dex_cached(self) -> Any:
        if self._dex is None:
            from .dex import load_showdown_dex_cached  # noqa: PLC0415

            self._dex = load_showdown_dex_cached(self.config.resolved_showdown_root())
        return self._dex

    def _observation_spec(self) -> ObservationSpec:
        if self._spec is None:
            self._spec = self.config.observation_spec or _default_observation_spec()
        return self._spec

    def _tables(self) -> str:
        """The encoder-tables JSON, with this env's feature masks latched in.

        The crate reads ``transition_token_budget`` (and the other ablation
        switches) out of ``layout.default_feature_masks``, so the masks travel
        with the tables rather than as a separate encode argument. k=0 is set
        here.
        """
        if self._tables_json is None:
            tables = _load_encoder_tables(
                self.config.encoder_tables,
                self.config.showdown_root,
                self._observation_spec().schema_version,
            )
            payload = json.loads(tables)
            masks = self.config.feature_masks
            payload.setdefault("layout", {})["default_feature_masks"] = {
                "opponent_tendency_stats_block": bool(masks.opponent_tendency_stats_block),
                "exact_state": bool(masks.exact_state),
                "transition_token_budget": int(masks.transition_token_budget),
                "tier2_residuals": bool(masks.tier2_residuals),
                "tier2_investment": bool(masks.tier2_investment),
            }
            spec = self._observation_spec()
            payload["layout"]["schema_version"] = spec.schema_version
            self._tables_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return self._tables_json

    def _team_source(self) -> Any:
        if self._team_env is None:
            from .local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: PLC0415

            self._team_env = LocalShowdownEnv(
                LocalShowdownConfig(
                    showdown_root=self.config.showdown_root,
                    node_binary=self.config.node_binary,
                )
            )
        return self._team_env

    # -- PokeZeroEnv ------------------------------------------------------

    def reset(self, *, seed: int, format_id: BattleFormat = "gen3randombattle") -> None:
        started = time.perf_counter()
        source = self._team_source()
        # Two disjoint team seeds per episode. The bridge derives its own
        # sub-seed per call, so distinct ints are all that is required.
        rows = {
            player: source.generate_scenario_team(
                seed=_derived_seed(seed, index, _SEED_DOMAIN_TEAM) % (2**31)
            )
            for index, player in enumerate(_PLAYERS)
        }
        self.timings.teams += time.perf_counter() - started
        self.timings.games += 1

        dex = self._dex_cached()
        self._party = {
            player: tuple(_generated_mon(row, dex) for row in rows[player]) for player in _PLAYERS
        }

        from .poke_engine_adapter import build_poke_engine_state  # noqa: PLC0415

        state = build_poke_engine_state(_battle_spec(self._party["p1"], self._party["p2"]))
        self._state_str = state.to_string()
        self._root_state_str = self._state_str

        self._battle_id = f"engine-{format_id}-{seed}"
        self._seed = int(seed)
        self._format_id = str(format_id)
        self._turn = 1
        self._ply = 0
        self._terminal = None
        self._lines = []
        self._observations = {}
        self._encoders = {}
        self._encoder_version = -1

        self._ledger = _PublicLedger(
            {player: [mon.species_key for mon in self._party[player]] for player in _PLAYERS}
        )
        for player in _PLAYERS:
            # Showdown opens the battle by switching both leads in publicly.
            self._ledger.reveal_lead(player, self._party[player][0].species_key)

        # The crate's fold, not the Python mirror: it is ~9x faster per advance
        # and is the same implementation the search stack chains per branch.
        self._folds = {
            player: self._native.FoldState.initial(perspective_slot=player) for player in _PLAYERS
        }
        self._refresh_options()

    def observe(self, player: PlayerId) -> PokeZeroObservationV0:
        cached = self._observations.get(player)
        if cached is not None:
            return cached
        observation = self._encode(player)
        self._observations[player] = observation
        return observation

    def legal_actions(self, player: PlayerId) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def requested_players(self) -> tuple[PlayerId, ...]:
        if self._terminal is not None:
            return ()
        return tuple(p for p in _PLAYERS if self._options.get(f"{p}_requested"))

    def terminal(self) -> TerminalState | None:
        return self._terminal

    def step(self, actions: Mapping[PlayerId, int]) -> StepResult:
        if self._terminal is not None:
            raise EngineEnvError("step() called on a terminated battle")
        requested = self.requested_players()
        missing = [player for player in requested if player not in actions]
        if missing:
            raise EngineEnvError(f"missing actions for requested players: {', '.join(missing)}")

        choices = {
            player: (self._engine_choice(player, int(actions[player])) if player in requested else "none")
            for player in _PLAYERS
        }

        started = time.perf_counter()
        report = json.loads(
            self._native.env_step(
                self._state_str,
                choices["p1"],
                choices["p2"],
                self._ctx_json(self._turn),
                _derived_seed(self._seed, self._ply),
                self.config.branch_on_damage,
            )
        )
        self.timings.step += time.perf_counter() - started
        self.timings.step_calls += 1

        self._ply += 1
        self._state_str = str(report["post_state"])
        lines = [str(line) for line in report.get("events") or ()]
        self._lines.extend(lines)
        if report.get("turn_completed"):
            self._turn += 1

        ledger_started = time.perf_counter()
        for fold in self._folds.values():
            fold.advance_in_place(lines)
        assert self._ledger is not None
        self._ledger.ingest(lines)
        self.timings.ledger += time.perf_counter() - ledger_started

        self._observations = {}
        self._refresh_options()
        self._update_terminal(float(report.get("battle_over") or 0.0))

        next_requested = self.requested_players()
        observations = (
            {}
            if self._terminal is not None
            else {player: self.observe(player) for player in next_requested}
        )
        return StepResult(
            observations=observations,
            rewards=self._rewards(),
            terminal=self._terminal,
            requested_players=next_requested,
        )

    def close(self) -> None:
        if self._team_env is not None:
            close = getattr(self._team_env, "close", None)
            if callable(close):
                close()
            self._team_env = None

    def public_materialization_state(self, player: PlayerId) -> Any:
        raise EngineEnvUnsupportedError(
            "EngineEnv does not expose public_materialization_state; policies that search "
            "over materialized worlds (engine-mcts:) need the Showdown env."
        )

    # -- internals --------------------------------------------------------

    def _refresh_options(self) -> None:
        self._options = json.loads(self._native.env_options(self._state_str, True))

    def _update_terminal(self, battle_over: float) -> None:
        if self._terminal is not None:
            return
        if battle_over > 0.0:
            self._terminal = TerminalState(winner="p1", turn_count=self._turn)
        elif battle_over < 0.0:
            self._terminal = TerminalState(winner="p2", turn_count=self._turn)
        elif self._ply >= self.config.max_plies:
            self._terminal = TerminalState(winner=None, turn_count=self._turn, capped=True)

    def _rewards(self) -> Mapping[PlayerId, float]:
        terminal = self._terminal
        if terminal is None or terminal.winner is None:
            return {"p1": 0.0, "p2": 0.0}
        return {player: (1.0 if player == terminal.winner else -1.0) for player in _PLAYERS}

    def _ctx_json(self, turn: int) -> str:
        return json.dumps(
            {
                "p1": [mon.species for mon in self._party["p1"]],
                "p2": [mon.species for mon in self._party["p2"]],
                "turn": int(turn),
            },
            sort_keys=True,
        )

    def _encoder(self, player: PlayerId) -> Any:
        """The seat's :class:`LeafEncoder`, rebuilt only when reveals moved it.

        Root inputs are re-derived from the public ledger, so an encoder built
        before a reveal would encode a stale opponent team. Everything else in
        the root inputs is game-start-constant, which is why the rebuild is
        keyed on the ledger version alone.
        """
        assert self._ledger is not None
        if self._encoder_version != self._ledger.version:
            self._encoders = {}
            self._encoder_version = self._ledger.version
        encoder = self._encoders.get(player)
        if encoder is None:
            root_inputs = json.dumps(self._root_inputs(player), sort_keys=True)
            ctx = self._ctx_json(1)
            # Constructing from tables_json re-parses ~475 KB (~3 ms); at one
            # rebuild per reveal per seat that would dominate the whole loop.
            # `rebased` shares the parsed tables, so only the first encoder in
            # the process pays for them. The template is a TABLES HOLDER only —
            # its own root belongs to whichever battle happened to be first, so
            # it is never used to encode.
            if self._encoder_template is None:
                self._encoder_template = self._native.LeafEncoder(
                    self._tables(), root_inputs, ctx, self._root_state_str
                )
            encoder = self._encoder_template.rebased(root_inputs, ctx, self._root_state_str)
            self._encoders[player] = encoder
        return encoder

    def _engine_choice(self, player: PlayerId, action_index: int) -> str:
        """Translate an action index into the engine option string for it."""
        if not 0 <= action_index < ACTION_COUNT:
            raise EngineEnvError(f"action index {action_index} out of range")
        started = time.perf_counter()
        mapping = self._encoder(player).self_action_map(
            self._state_str, self._lines or None, True, True
        )
        self.timings.action_map += time.perf_counter() - started
        for display, index in mapping:
            if index == action_index:
                return display
        legal = sorted({index for _, index in mapping if index is not None})
        raise EngineEnvError(
            f"{player} action {action_index} has no engine option "
            f"(legal action indices: {legal})"
        )

    def _encode(self, player: PlayerId) -> PokeZeroObservationV0:
        started = time.perf_counter()
        encoder = self._encoder(player)
        lines = self._lines or None
        buffers = encoder.encode_leaf(
            self._state_str, self._folds[player], self._turn, lines, True
        )
        # The metadata must describe the SAME row the tensors were built from,
        # so it is read back from the encoder rather than reconstructed here.
        row = json.loads(encoder.leaf_inputs_json(self._state_str, self._turn, lines, True))
        self.timings.encode += time.perf_counter() - started
        self.timings.encode_calls += 1

        spec = self._observation_spec()
        arrays = _arrays_from_buffers(buffers, spec)

        # Downstream (dataset padding, linear features, torch batching) expects
        # the nested-sequence shape the Showdown encoder emits, and
        # `pokezero.padding._shape_of` walks it elementwise — which a numpy
        # array's ambiguous truthiness breaks. `.tolist()` is the C-speed
        # conversion; the Showdown path builds the same structure in pure
        # Python, so this is strictly cheaper than parity. It is nonetheless a
        # real per-decision cost, hence its own timing bucket: closing it means
        # teaching the padding helper about buffers, which is fingerprinted
        # into linear-policy checkpoints and out of scope here.
        materialize_started = time.perf_counter()
        categorical = arrays["categorical_ids"].tolist()
        numeric = arrays["numeric_features"].tolist()
        token_types = arrays["token_type_ids"].tolist()
        attention = arrays["attention_mask"].tolist()
        legal = tuple(bool(value) for value in arrays["legal_action_mask"].tolist())
        self.timings.materialize += time.perf_counter() - materialize_started

        metadata = dict(row.get("observation_metadata") or {})
        metadata["battle_id"] = self._battle_id
        metadata["player_id"] = player
        _refresh_derived_hp(metadata)
        observation = PokeZeroObservationV0(
            categorical_ids=categorical,
            numeric_features=numeric,
            token_type_ids=token_types,
            attention_mask=attention,
            legal_action_mask=legal,
            perspective=ObservationPerspective.from_showdown_slot(player, player),
            metadata=metadata,
            schema_version=spec.schema_version,
        )
        observation.validate(spec)
        return observation

    # -- root row inputs ---------------------------------------------------

    def _root_inputs(self, player: PlayerId) -> dict[str, Any]:
        """The seat's row-inputs surface, anchored at the game-start state.

        Everything state-shaped here is a turn-1 value that the native encoder
        will recompute or delta from; the only fields that carry real
        information are the identity sets, which are ledger-gated.
        """
        assert self._ledger is not None
        opponent = _opponent_of(player)
        self_party = self._party[player]
        opponent_party = [
            mon for mon in self._party[opponent] if self._ledger.is_revealed(opponent, mon.species_key)
        ]

        metadata: dict[str, Any] = {
            "battle_id": self._battle_id,
            "player_id": player,
            "showdown_slot": player,
            "opponent_showdown_slot": opponent,
            "request_kind": "move",
            "turn_number": 1,
            "weather": None,
            "weather_turns_remaining": 0,
            "weather_permanent": False,
            "self_side_conditions": [],
            "opponent_side_conditions": [],
            "self_side_condition_counts": {},
            "opponent_side_condition_counts": {},
            "self_active_boosts": {},
            "opponent_active_boosts": {},
            "self_active_volatiles": [],
            "opponent_active_volatiles": [],
            "self_future_sight_turns": 0,
            "opponent_future_sight_turns": 0,
            "self_toxic_stage": 0,
            "opponent_toxic_stage": 0,
            "self_sleep_clause_used": False,
            "opponent_sleep_clause_used": False,
            "self_sleep_clause_blocks": False,
            "opponent_sleep_clause_blocks": False,
            "self_wish_pending": False,
            "opponent_wish_pending": False,
            "self_wish_turns": 0,
            "opponent_wish_turns": 0,
            "self_stall_counter": 0,
            "opponent_stall_counter": 0,
            "self_confusion_elapsed": 0,
            "opponent_confusion_elapsed": 0,
            "self_encore_elapsed": 0,
            "opponent_encore_elapsed": 0,
            "self_wrap_trap_elapsed": 0,
            "opponent_wrap_trap_elapsed": 0,
            "self_meanlook_trap": False,
            "opponent_meanlook_trap": False,
            "self_team": [_mon_metadata(mon, player, active=index == 0, own=True)
                          for index, mon in enumerate(self_party)],
            "opponent_team": [
                _mon_metadata(mon, opponent, active=mon is self._party[opponent][0], own=False)
                for mon in opponent_party
            ],
            # Rebuilt from the engine option surface on every encode.
            "action_candidates": [],
            "recent_public_events": [],
            "transition_token_count": self._observation_spec().transition_token_count,
            "belief_view": {
                "self_slot": player,
                "opponent_slot": opponent,
                "self_pokemon": [
                    _belief_entry(mon, active=index == 0, own=True, facts=None)
                    for index, mon in enumerate(self_party)
                ],
                "opponent_pokemon": [
                    _belief_entry(
                        mon,
                        active=mon is self._party[opponent][0],
                        own=False,
                        facts=self._ledger.facts(opponent, mon.species_key),
                    )
                    for mon in opponent_party
                ],
            },
        }
        active = self_party[0]
        return {
            "battle_id": self._battle_id,
            "battle_seed": self._seed,
            "format_id": self._format_id,
            "player_id": player,
            "observation_schema_version": self._observation_spec().schema_version,
            "observation_metadata": metadata,
            "public_materialization": {
                "turn": 1,
                "selfActiveMoves": [
                    {
                        "id": move,
                        "move": move,
                        "pp": int(active.move_pp.get(move, 32)),
                        "maxpp": int(active.move_pp.get(move, 32)),
                        "disabled": False,
                    }
                    for move in active.moves
                ],
                "sides": {
                    slot: {"sideConditions": {}, "sideConditionSetTurns": {}} for slot in _PLAYERS
                },
            },
        }


def _refresh_derived_hp(metadata: dict[str, Any]) -> None:
    """Recompute ``hp_fraction`` / ``fainted`` from the leaf's ``condition``.

    The native encoder rewrites each mon's ``condition`` from engine state but
    leaves these two DERIVED fields at their root values — it never reads them
    itself, so nothing in the tensor path noticed. The metadata does have a
    consumer, though: ``dataset._visible_team_snapshot`` reads exactly
    ``hp_fraction`` and ``fainted`` to build the potential-shaping terms behind
    ``--hp-delta-return-weight`` / ``--faint-delta-return-weight``. Left stale
    they read "everyone at full HP, nobody fainted" for the whole game, which
    silently zeroes those shaping arms rather than failing — so they are
    re-derived here from the field the encoder DOES maintain.
    """
    for key in ("self_team", "opponent_team"):
        for entry in metadata.get(key) or ():
            if not isinstance(entry, dict):
                continue
            hp_fraction, fainted = _condition_hp(entry.get("condition"))
            entry["hp_fraction"] = hp_fraction
            entry["fainted"] = fainted


def _condition_hp(condition: Any) -> tuple[float | None, bool]:
    """Parse a Showdown condition string: ``"202/232 brn"`` / ``"0 fnt"``."""
    parts = str(condition or "").split()
    if not parts:
        return None, False
    if "fnt" in parts:
        return 0.0, True
    numerator, _, denominator = parts[0].partition("/")
    try:
        current = float(numerator)
        maximum = float(denominator)
    except ValueError:
        return None, False
    if maximum <= 0:
        return None, False
    return current / maximum, current <= 0


def _mon_metadata(mon: _GeneratedMon, slot: PlayerId, *, active: bool, own: bool) -> dict[str, Any]:
    """One ``self_team`` / ``opponent_team`` row at game-start values.

    ``own`` gates the two fields that would leak: exact computed ``stats`` and
    the full move list are the player's own request data. For the opponent the
    encoder falls back to expected stats (the same fallback the Showdown path
    uses, where opponent ``stats`` is ``None`` by construction), and the move
    list is whatever the public transcript has shown.
    """
    return {
        "ident": f"{slot}: {mon.species}",
        "showdown_slot": slot,
        "species": mon.species,
        "condition": f"{mon.max_hp}/{mon.max_hp}",
        "hp_fraction": 1.0,
        "status": "",
        "fainted": False,
        "active": bool(active),
        "details": mon.details,
        "moves": list(mon.moves) if own else [],
        "ability": mon.ability if own else None,
        "item": mon.item if own else None,
        "stats": dict(mon.stats) if own else None,
        "live_type_source": None,
    }


def _belief_entry(
    mon: _GeneratedMon, *, active: bool, own: bool, facts: _MonReveal | None
) -> dict[str, Any]:
    """One belief-ledger row.

    Own mons are fully known. Opponent mons carry only what the public
    transcript disclosed; the candidate-set machinery the Showdown collector
    runs is absent here, so ``possible_*`` stay empty and ``uncertainty`` is
    pinned at maximum. See the module docstring — this is the headline
    fidelity residual, and it is a value skew, never a shape change.
    """
    revealed_moves = list(mon.moves) if own else list(facts.moves if facts else ())
    return {
        "showdown_slot": None,
        "species": mon.species,
        "condition": f"{mon.max_hp}/{mon.max_hp}",
        "status": "",
        "active": bool(active),
        "revealed_moves": revealed_moves,
        "revealed_ability": (mon.ability if own else (facts.ability if facts else None)) or None,
        "revealed_item": (mon.item if own else (facts.item if facts else None)) or None,
        "ruled_out_abilities": [],
        "ruled_out_items": [],
        "possible_abilities": [],
        "possible_items": [],
        "possible_moves": [],
        "candidate_variants": [],
        "candidate_set_count": 1 if own else 0,
        "uncertainty": 0.0 if own else 1.0,
        "transformed": False,
        "transform_species": None,
        "move_uses": [],
        "sleep_turns": 0,
        "rest_sleep": False,
        "sleep_skipped_turns": 0,
        "turns_active": 1 if active else 0,
        "item_mutated": False,
        "item_removed": False,
        "current_public_item": (mon.item if own else (facts.item if facts else None)) or None,
    }


# ---------------------------------------------------------------------------
# Encoder plumbing
# ---------------------------------------------------------------------------


def _arrays_from_buffers(buffers: Mapping[str, Any], spec: ObservationSpec) -> dict[str, Any]:
    import numpy  # noqa: PLC0415

    shapes = {
        "categorical_ids": (spec.token_count, spec.categorical_feature_count),
        "numeric_features": (spec.token_count, spec.numeric_feature_count),
        "token_type_ids": (spec.token_count,),
        "attention_mask": (spec.token_count,),
        "legal_action_mask": (ACTION_COUNT,),
    }
    arrays: dict[str, Any] = {}
    for name in _ARRAY_NAMES:
        buffer = buffers.get(name)
        if buffer is None:
            raise EngineEnvError(f"native encoder returned no buffer for {name}")
        # A read-only view over crate-owned bytes. Callers must materialize
        # (`.tolist()`) rather than retain it — the buffer is not theirs.
        arrays[name] = numpy.frombuffer(buffer, dtype=_ARRAY_DTYPES[name]).reshape(shapes[name])
    return arrays


def _default_observation_spec() -> ObservationSpec:
    from .showdown import DEFAULT_REPLAY_OBSERVATION_SPEC  # noqa: PLC0415

    return DEFAULT_REPLAY_OBSERVATION_SPEC


# The schemas `scripts/export_encoder_tables.py --observation-schema` accepts. Kept beside the
# loader so a new schema is one edit, and so an unsupported one is a named failure rather than a
# silent fallback to whichever layout the else-branch happened to name.
_EXPORTABLE_TABLE_SCHEMAS = frozenset({"v2.2", "v3", "v4"})


def encoder_tables_schema(schema_version: str) -> str:
    """The exporter's short layout name for an observation schema version.

    A pure function, extracted so the test can exercise THIS rather than restate it. The first
    version of that test re-derived the mapping inside the test body, which would have passed with
    production still broken -- the exact vacuity this codebase has been finding all week.
    """
    prefix = "pokezero.observation."
    text = str(schema_version)
    schema = text[len(prefix):] if text.startswith(prefix) else ""
    if schema not in _EXPORTABLE_TABLE_SCHEMAS:
        raise ValueError(
            f"no encoder-tables layout for observation schema {schema_version!r}; "
            f"scripts/export_encoder_tables.py supports {sorted(_EXPORTABLE_TABLE_SCHEMAS)}"
        )
    return schema


def _load_encoder_tables(
    path: Path | None, showdown_root: Path | None, schema_version: str
) -> str:
    """Read the encoder-tables artifact, building it on demand when absent.

    ``scripts/export_encoder_tables.py`` is the single sanctioned producer;
    calling it here keeps the env from growing a second, drifting copy of the
    dex/vocab export. The generated artifact is cached per schema under
    ``corpus/`` (gitignored build output, not a source file).
    """
    if path is not None:
        return Path(path).read_text(encoding="utf-8")

    from .local_showdown import LocalShowdownConfig  # noqa: PLC0415

    root = LocalShowdownConfig(showdown_root=showdown_root).resolved_showdown_root()
    repo = Path(__file__).resolve().parents[2]
    # Schema versions look like "pokezero.observation.v2.2"; the exporter takes the short form.
    #
    # Derived from the version string, not enumerated. The previous form was
    # `"v3" if ...endswith(".v3") else "v2.2"`, written when v2.2 and v3 were "the two current
    # layouts" -- so v4 fell through the else and silently loaded V2.2 TABLES, which is a wrong
    # answer rather than an error: the layouts disagree on width (132 vs 155) and on which columns
    # exist at all, so the first encode raises deep in the encoder about a missing column instead
    # of here about an unsupported schema.
    #
    # `export_encoder_tables.py` already accepts v4 (`choices=("v2.2", "v3", "v4")`); only this
    # mapping had to be told. Unknown schemas now fail HERE, named, rather than falling back to a
    # layout that happens to parse.
    schema = encoder_tables_schema(schema_version)
    cache = repo / "corpus" / f"encoder_tables_{schema}.json"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    script = repo / "scripts" / "export_encoder_tables.py"
    if not script.exists():
        raise EngineEnvError(
            f"no encoder tables at {cache} and no exporter at {script}; "
            "pass EngineEnvConfig(encoder_tables=...)"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--showdown-root",
            str(root),
            "--observation-schema",
            schema,
            "--out",
            str(cache),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EngineEnvError(
            f"export_encoder_tables.py failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return cache.read_text(encoding="utf-8")



__all__ = [
    "EngineEnv",
    "EngineEnvConfig",
    "EngineEnvError",
    "EngineEnvTimings",
    "EngineEnvUnsupportedError",
]
