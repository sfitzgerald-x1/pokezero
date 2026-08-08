"""Resolve a recorded fallback address into a concrete, runnable replay spec.

``fallback_addresses`` reads ``{battle_id, round, seat, reason}`` tuples out of the
shards. That tuple is an *address*, not a replay: ``battle_id`` is
``f"{prefix}-{seed}"`` (``foulplay_bridge.py:2699``) and carries the seed and
nothing else. Depth, sims, batch, worlds, the checkpoint, the opponent and the
opponent's own seed all live in the shard's config fields, and they differ per
writer. This module joins the two halves into a :class:`ReplaySpec` -- everything
needed to stand one battle back up -- and, just as importantly, states what the
shard does **not** pin.

It is deliberately pure: no battle is run here, nothing is imported from the
search stack, and every function takes parsed JSON. That is what makes the
resolution testable without a checkpoint, a Showdown build or a GPU.

Two things this module exists to stop being assumed.

**1. The battle-id grammar is per-harness, and the naive parse is wrong.**
Four writers mint four different ids::

    battle-gen3randombattle-controlled-8220001   foulplay bridge   seed is last
    accept-search-600004-p1                      mcts_acceptance   seed is 2nd-last
    hcgrid-hc-d4-600000                          hc_depth_grid     seed is last
    k0grid-600003                                k0_grid_h2h       seed is last

``rsplit("-", 1)[-1]`` returns ``"p1"`` for the acceptance form. The grammar is
therefore looked up by prefix, and the prefix is cross-checked against the
document's own schema: a mismatch is a resolution failure, not a warning, because
the whole point of the locator is that the shard identifies the battle.

**2. Replay fidelity is a property of the WRITER, not of the address.** Two
independent axes, and no writer scores well on both:

* *Battle reconstruction* -- can the game state at ``round`` be rebuilt? Only if
  every actor is pinned. The three self-play writers drive
  ``RolloutDriver``/``run_rollout_record_on_env`` in-process with both policies
  seeded off the battle seed, so yes. The foul-play writer does not: the external
  opponent searches on a **wall-clock budget**
  (``--search-time-ms 1000``, ``foulplay_bridge.py:2650``; consumed by
  ``foul-play`` at ``fp/search/main.py:56`` as
  ``monte_carlo_tree_search(state, search_time_ms, threads=threads)``) and then
  draws its move with ``random.choices`` weighted by the resulting visit shares
  (``fp/search/main.py:46``). Its RNG *is* pinned -- the schedule is recorded and
  injected as ``POKEZERO_FOULPLAY_RANDOM_SEED`` (``foulplay_bridge.py:2611``) --
  but the weights it draws against are machine-speed dependent, so the draw is
  not. One divergent opponent move and every later round is a different battle.
* *Decision RNG* -- can the policy's own ``random.Random`` for that one decision
  be rebuilt? Under the bridge, yes and trivially: it is reseeded per decision
  from ``f"{seed}:{player_id}:{decision_round_index}"``
  (``foulplay_bridge.py:3541``), which is exactly the address. Under
  ``RolloutDriver`` it is a per-battle-per-seat stream created once
  (``rollout.py:414-426``) and advanced by every preceding decision, with a
  data-dependent number of draws (the world-sampling retry loop,
  ``engine_search.py:1050-1098``), so decision *N* is reachable only by replaying
  ``0..N-1``.

The two axes are exactly inverted between the two families. Eras 61-64 -- the
entire 1,136-address corpus -- came from the foul-play writer, i.e. from the side
whose battle cannot be rebuilt from the address alone. Saying so is the point:
:attr:`ReplaySpec.fidelity` carries the verdict and
:attr:`ReplaySpec.fidelity_notes` carries the evidence, so a driver can refuse,
and a report cannot quietly claim an exact replay it did not perform.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .fallback_addresses import (
    FallbackAddress,
    _iter_shard_paths,  # noqa: PLC2701 -- see _shard_documents
    iter_shard_addresses,
)

__all__ = [
    "BattleIdGrammar",
    "ReplaySpec",
    "SpecResolution",
    "UnresolvedAddress",
    "GRAMMARS",
    "battle_id_grammar",
    "parse_battle_id",
    "resolve_address",
    "resolve_corpus",
]


# --- harness identity -------------------------------------------------------
#
# Named for the CODE PATH that has to be stood back up, not for the script that
# happened to invoke it: `mcts_acceptance_h2h` and `k0_grid_h2h` are two scripts
# over one harness (`RolloutDriver` with `leaf_eval="model"`), and a replay driver
# cares about the former distinction only for locating config fields.
HARNESS_FOULPLAY_BRIDGE = "foulplay-bridge"
HARNESS_ROLLOUT_ACCEPTANCE = "rollout-acceptance"
HARNESS_ROLLOUT_HC_GRID = "rollout-hc-grid"
HARNESS_ROLLOUT_K0_GRID = "rollout-k0-grid"

# --- fidelity verdicts ------------------------------------------------------

#: Every input to the target decision is reconstructible from this spec.
FIDELITY_EXACT = "exact"
#: The battle cannot be rebuilt: an actor in it is not pinned by anything the
#: shard records. Re-running the seed produces *a* battle, not *the* battle.
FIDELITY_OPPONENT_UNPINNED = "opponent-unpinned"
#: The writer is not recognised, so nothing is claimed either way.
FIDELITY_UNKNOWN = "unknown"


@dataclass(frozen=True)
class BattleIdGrammar:
    """How one harness mints ``battle_id``, and how to read a seed back out.

    ``seed_field_from_end`` is 1-based from the right: the acceptance harness
    appends the seat *after* the seed (``accept-{arm}-{seed}-{seat}``,
    ``scripts/mcts_acceptance_h2h.py:439``) and is the reason this is a field
    rather than a hard-coded ``[-1]``.
    """

    harness: str
    prefix: str
    seed_field_from_end: int = 1

    def parse(self, battle_id: str) -> int | None:
        parts = battle_id.split("-")
        if len(parts) < self.seed_field_from_end:
            return None
        token = parts[-self.seed_field_from_end]
        # `int()` accepts "+7", " 7" and unicode digits; a battle id field is a
        # decimal literal or it is not a seed.
        if not token.isascii() or not token.isdigit():
            return None
        return int(token)


# Ordered longest-prefix-first so a future `k0grid-x-` cannot be shadowed by
# `k0grid-`.
GRAMMARS: tuple[BattleIdGrammar, ...] = (
    # foulplay_bridge.py:105 DEFAULT_BATTLE_ID_PREFIX + :2699
    BattleIdGrammar(HARNESS_FOULPLAY_BRIDGE, "battle-gen3randombattle-controlled-"),
    # scripts/mcts_acceptance_h2h.py:439  f"accept-{arm}-{seed}-{seat}"
    BattleIdGrammar(HARNESS_ROLLOUT_ACCEPTANCE, "accept-", seed_field_from_end=2),
    # scripts/hc_depth_grid.py:243  f"hcgrid-{cell}-{seed}"
    BattleIdGrammar(HARNESS_ROLLOUT_HC_GRID, "hcgrid-"),
    # scripts/k0_grid_h2h.py:192  f"k0grid-{seed}"
    BattleIdGrammar(HARNESS_ROLLOUT_K0_GRID, "k0grid-"),
)


def battle_id_grammar(battle_id: str) -> BattleIdGrammar | None:
    """Return the grammar whose prefix ``battle_id`` carries, longest first."""
    matches = [g for g in GRAMMARS if battle_id.startswith(g.prefix)]
    if not matches:
        return None
    return max(matches, key=lambda g: len(g.prefix))


def parse_battle_id(battle_id: str) -> tuple[str, int] | None:
    """Return ``(harness, seed)`` for a recognised battle id, else ``None``."""
    grammar = battle_id_grammar(battle_id)
    if grammar is None:
        return None
    seed = grammar.parse(battle_id)
    if seed is None:
        return None
    return grammar.harness, seed


# --- shard shape detection --------------------------------------------------

_SCHEMA_FOULPLAY_SIDECAR = "pokezero.controlled-foulplay-benchmark.v1"
_SCHEMA_FOULPLAY_PAIRED = "pokezero.foulplay-paired-shard.v1"
_SCHEMA_ACCEPTANCE = "pokezero.mcts-acceptance-shard.v1"


def _document_harness(document: Mapping[str, Any]) -> str | None:
    """Identify the writer from the document itself, independently of the id.

    Independence is the whole value: the id says which harness *minted* the
    battle and the document says which harness *wrote the shard*, and
    :func:`resolve_address` refuses when they disagree. A resolver that trusted
    only one of them would silently apply the wrong config-field layout to a
    misfiled shard.
    """
    schema = document.get("schema_version")
    if schema == _SCHEMA_FOULPLAY_SIDECAR or schema == _SCHEMA_FOULPLAY_PAIRED:
        return HARNESS_FOULPLAY_BRIDGE
    if schema == _SCHEMA_ACCEPTANCE:
        return HARNESS_ROLLOUT_ACCEPTANCE
    # The two grid writers emit no schema_version. `cell` + `raw_spec` is
    # hc_depth_grid's signature (scripts/hc_depth_grid.py:283-291); k0_grid_h2h
    # writes `config` + `per_game` and no `cell` (scripts/k0_grid_h2h.py:219-241).
    if "cell" in document and "raw_spec" in document:
        return HARNESS_ROLLOUT_HC_GRID
    if "config" in document and "per_game" in document and "cell" not in document:
        return HARNESS_ROLLOUT_K0_GRID
    return None


# --- the spec ---------------------------------------------------------------


@dataclass(frozen=True)
class ReplaySpec:
    """One recorded fallback decision, expressed as something runnable.

    Every field is either read off the shard or derived from a cited line of the
    producer. Anything the shard does not pin is ``None`` and is named in
    :attr:`missing`, never defaulted -- a default here is a silent claim that the
    replay matched the recording.
    """

    # --- the address ---
    battle_id: str
    round: int
    seat: str
    reason: str
    key: str
    #: Shard label, from `FallbackAddress.source`. Part of the locator: a
    #: depth/arm grid reuses one `seed_start` across shards, so the address alone
    #: collides across genuinely different search configurations.
    source: str

    # --- battle identity ---
    harness: str
    seed: int

    # --- what to run ---
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    format_id: str | None = None
    policy_mode: str | None = None
    leaf_eval: str | None = None
    engine_depth: int | None = None
    engine_sims: int | None = None
    engine_batch: int | None = None
    engine_worlds: int | None = None
    engine_c_puct: float | None = None
    opponent_priors: bool | None = None
    max_decision_rounds: int | None = None
    belief_set_source: bool | None = None
    device: str | None = None

    # --- who it was played against ---
    opponent_policy_id: str | None = None
    #: The external opponent's per-game RNG seed, taken from the recorded
    #: schedule rather than assumed equal to the battle seed: a run that set
    #: `foulplay_random_seed=F` with `seed_start=S` played battle seed N against
    #: opponent seed `F + (N - S)` (`foulplay_bridge.py:2375`).
    opponent_random_seed: int | None = None
    opponent_search_time_ms: int | None = None

    # --- how the decision's RNG was made ---
    #: `"per-decision"` (bridge; reseeded from the address itself) or
    #: `"per-battle-stream"` (RolloutDriver; only reachable by replaying).
    rng_regime: str = ""
    #: The literal seed argument to `random.Random` for THIS decision, when the
    #: regime makes one constructible. `foulplay_bridge.py:3541`.
    decision_rng_seed: str | None = None

    # --- what is and is not claimable ---
    fidelity: str = FIDELITY_UNKNOWN
    fidelity_notes: tuple[str, ...] = ()
    #: Names of spec fields the shard did not record. A driver must decide what
    #: to do about each; it must not discover them by getting `None`.
    missing: tuple[str, ...] = ()

    @property
    def locator(self) -> tuple[str, str, int, str]:
        return (self.source, self.battle_id, self.round, self.seat)

    @property
    def replayable_exactly(self) -> bool:
        return self.fidelity == FIDELITY_EXACT

    def to_dict(self) -> dict[str, Any]:
        return {
            "battle_id": self.battle_id,
            "round": self.round,
            "seat": self.seat,
            "reason": self.reason,
            "key": self.key,
            "source": self.source,
            "harness": self.harness,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "format_id": self.format_id,
            "policy_mode": self.policy_mode,
            "leaf_eval": self.leaf_eval,
            "engine_depth": self.engine_depth,
            "engine_sims": self.engine_sims,
            "engine_batch": self.engine_batch,
            "engine_worlds": self.engine_worlds,
            "engine_c_puct": self.engine_c_puct,
            "opponent_priors": self.opponent_priors,
            "max_decision_rounds": self.max_decision_rounds,
            "belief_set_source": self.belief_set_source,
            "device": self.device,
            "opponent_policy_id": self.opponent_policy_id,
            "opponent_random_seed": self.opponent_random_seed,
            "opponent_search_time_ms": self.opponent_search_time_ms,
            "rng_regime": self.rng_regime,
            "decision_rng_seed": self.decision_rng_seed,
            "fidelity": self.fidelity,
            "fidelity_notes": list(self.fidelity_notes),
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class UnresolvedAddress:
    """An address that could not be turned into a spec, and why.

    Kept as a value rather than raised or skipped: the count of unresolved
    addresses is the coverage number of any corpus gate, and a resolver that
    silently drops them reports 100% coverage of whatever it happened to
    understand.
    """

    address: FallbackAddress
    problem: str


SpecResolution = ReplaySpec | UnresolvedAddress


# --- field extraction, per writer ------------------------------------------


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _opponent_seed_from_schedule(
    document: Mapping[str, Any], seed: int
) -> tuple[int | None, str | None]:
    """Read the external opponent's seed for THIS battle out of the schedule.

    Returns ``(seed, problem)``. The schedule is indexed by game offset, and the
    offset is ``battle_seed - seed_start`` because the bridge walks
    ``seed = config.seed_start + offset`` (``foulplay_bridge.py:1721-1722``) while
    handing game ``offset`` the schedule's ``offset``-th entry (``:1728``).
    Recomputing it as ``seed`` instead of reading the list is the specific error
    the ``foulplay_random_seed`` field exists to make detectable.
    """
    schedule = _mapping(document.get("foulplay_random_seed_schedule"))
    seeds = schedule.get("seeds")
    seed_start = _as_int(document.get("seed_start"))
    if not isinstance(seeds, list) or seed_start is None:
        return None, "foulplay_random_seed_schedule.seeds"
    offset = seed - seed_start
    if offset < 0 or offset >= len(seeds):
        return None, (
            f"battle seed {seed} is outside this shard's band "
            f"[{seed_start}, {seed_start + len(seeds)})"
        )
    return _as_int(seeds[offset]), None


def _foulplay_fields(
    document: Mapping[str, Any], address: FallbackAddress, seed: int
) -> tuple[dict[str, Any], list[str], str | None]:
    """Config for a foul-play bridge battle.

    Only the bridge's own summary (the ``-p1.json`` / ``-p2.json`` sidecar) is
    accepted. The merged paired shard carries the addresses too -- the reader
    finds them under ``per_seat[seat].policy_stats`` -- but it records
    depth/sims/batch/worlds ONLY inside the ``config_id`` string, records no
    ``format_id``, no ``foulplay_random_seed_schedule`` and no
    ``max_decision_rounds``. Parsing a spec out of ``"d4-s1024-b64-w4@k1"`` would
    be a guess dressed as a reading, and the sidecar sits next to it on disk.
    """
    if document.get("schema_version") == _SCHEMA_FOULPLAY_PAIRED:
        # Name the file to point at instead of just refusing. The driver writes
        # it as `out.parent / f"{out.stem}-{seat}.json"`
        # (scripts/foulplay_paired_eval.py:158-160), so it is derivable from the
        # shard label the address already carries.
        source = Path(address.source)
        sidecar = str(source.with_name(f"{source.stem}-{address.seat}.json"))
        return (
            {},
            [],
            "paired shard carries addresses but not a full config; resolve "
            f"against the per-seat sidecar {sidecar} instead",
        )

    engine = _mapping(document.get("engine_mcts"))
    root_puct = _mapping(document.get("root_puct"))
    opponent_seed, schedule_problem = _opponent_seed_from_schedule(document, seed)
    # A seed outside the shard's own band is not a missing field, it is the wrong
    # shard: the config here describes a different set of battles.
    if schedule_problem is not None and schedule_problem.startswith("battle seed "):
        return {}, [], schedule_problem

    # The sidecar is written per --pokezero-player invocation, so it describes
    # ONE seat. An address filed under the other seat did not come from this
    # document and its config must not be read off it.
    recorded_seat = _as_str(document.get("pokezero_player"))
    if recorded_seat is not None and recorded_seat != address.seat:
        return (
            {},
            [],
            f"address seat {address.seat!r} is not this sidecar's seat {recorded_seat!r}",
        )

    fields: dict[str, Any] = {
        "checkpoint": _as_str(document.get("checkpoint")),
        "checkpoint_sha256": _as_str(document.get("checkpoint_sha256")),
        "format_id": _as_str(document.get("format_id")),
        "policy_mode": _as_str(document.get("policy_mode")),
        # foulplay_bridge.py:2474-2487 hardcodes the model leaf for engine-mcts.
        "leaf_eval": "model" if document.get("policy_mode") == "engine-mcts" else None,
        "engine_depth": _as_int(engine.get("depth")),
        "engine_sims": _as_int(engine.get("sims")),
        "engine_batch": _as_int(engine.get("batch")),
        "engine_worlds": _as_int(engine.get("worlds")),
        "opponent_priors": _as_bool(engine.get("opponent_priors")),
        "max_decision_rounds": _as_int(document.get("max_decision_rounds")),
        "belief_set_source": _as_bool(document.get("belief_set_source")),
        "opponent_policy_id": _as_str(document.get("opponent_policy_id")),
        "opponent_random_seed": opponent_seed,
        "opponent_search_time_ms": _as_int(root_puct.get("foulplay_search_time_ms")),
        "rng_regime": "per-decision",
        # foulplay_bridge.py:3541 -- verbatim, including the str seed argument.
        "decision_rng_seed": f"{seed}:{address.seat}:{address.round}",
    }
    missing = ["foulplay_random_seed_schedule.seeds"] if schedule_problem else []
    # `--device` is a bridge CLI flag that never reaches the summary payload.
    missing.append("device")
    return fields, missing, None


def _hc_grid_fields(
    document: Mapping[str, Any], address: FallbackAddress, seed: int
) -> tuple[dict[str, Any], list[str], str | None]:
    """Config for an ``hc_depth_grid`` battle (scripts/hc_depth_grid.py:282-300)."""
    raw_spec = _as_str(document.get("raw_spec")) or ""
    device = None
    if "device=" in raw_spec:
        device = raw_spec.rsplit("device=", 1)[1].split("&", 1)[0] or None
    fields: dict[str, Any] = {
        "checkpoint": _as_str(document.get("checkpoint")),
        "policy_mode": "engine-mcts",
        # scripts/hc_depth_grid.py:220 -- the handcrafted leaf, so the checkpoint
        # is the OPPONENT's and no model/tables artifact is involved.
        "leaf_eval": "hp_fraction_crate",
        "engine_depth": _as_int(document.get("depth")),
        "engine_sims": _as_int(document.get("sims")),
        "engine_worlds": _as_int(document.get("worlds")),
        "engine_c_puct": _as_float(document.get("c_puct")),
        "device": device,
        "opponent_policy_id": raw_spec or None,
        "rng_regime": "per-battle-stream",
        "decision_rng_seed": None,
    }
    # scripts/hc_depth_grid.py:235 -- seat is seed parity, so the shard's own
    # rule and the recorded address must agree. They are two independent
    # statements about the same battle; a disagreement means the address was
    # filed against a different run.
    expected_seat = "p1" if seed % 2 == 0 else "p2"
    if address.seat != expected_seat:
        return (
            {},
            [],
            f"seat {address.seat!r} contradicts hc_depth_grid's seed-parity rule "
            f"(seed {seed} -> {expected_seat})",
        )
    missing = ["checkpoint_sha256", "format_id", "max_decision_rounds"]
    return fields, missing, None


def _k0_grid_fields(
    document: Mapping[str, Any], address: FallbackAddress, seed: int
) -> tuple[dict[str, Any], list[str], str | None]:
    """Config for a ``k0_grid_h2h`` battle (scripts/k0_grid_h2h.py:219-241).

    Reachable only if that writer is fixed: it never calls ``stats.to_dict()``,
    so its shards carry no ``fallback_samples`` at all. Kept because the layout
    is knowable now and the producer gap is tracked separately -- and because a
    resolver that quietly failed on it would look identical to one that had
    never been asked.
    """
    config = _as_str(document.get("config")) or ""
    batch = None
    for part in config.split("-"):
        if part.startswith("b") and part[1:].isdigit():
            batch = int(part[1:])
    fields: dict[str, Any] = {
        "checkpoint": _as_str(document.get("checkpoint")),
        "checkpoint_sha256": _as_str(document.get("checkpoint_sha256")),
        "format_id": "gen3randombattle",  # scripts/k0_grid_h2h.py:190, hardcoded
        "policy_mode": "engine-mcts",
        "leaf_eval": "model",  # scripts/k0_grid_h2h.py:154-168
        "engine_depth": _as_int(document.get("depth")),
        "engine_sims": _as_int(document.get("sims")),
        "engine_batch": batch,
        "engine_worlds": _as_int(document.get("worlds")),
        "rng_regime": "per-battle-stream",
        "decision_rng_seed": None,
    }
    expected_seat = "p1" if seed % 2 == 0 else "p2"  # scripts/k0_grid_h2h.py:179
    if address.seat != expected_seat:
        return (
            {},
            [],
            f"seat {address.seat!r} contradicts k0_grid_h2h's seed-parity rule "
            f"(seed {seed} -> {expected_seat})",
        )
    return fields, ["device", "engine_c_puct", "max_decision_rounds"], None


def _acceptance_fields(
    document: Mapping[str, Any], address: FallbackAddress, seed: int
) -> tuple[dict[str, Any], list[str], str | None]:
    """Config for an ``mcts_acceptance_h2h`` battle (:252-288).

    Depth/sims/batch/worlds exist only inside ``config_id``
    (``f"d{depth}-s{sims}-b{batch}-w{worlds}"``, :334). Unlike the paired shard's
    ``config_id`` this one is the *whole* record -- there is no sidecar to defer
    to -- so it is parsed, strictly: an unparseable component yields ``None`` and
    a named entry in ``missing`` rather than a partial guess.
    """
    config_id = _as_str(document.get("config_id")) or ""
    parsed: dict[str, int] = {}
    for part in config_id.split("@", 1)[0].split("-"):
        if len(part) > 1 and part[0] in "dsbw" and part[1:].isdigit():
            parsed[part[0]] = int(part[1:])
    fields: dict[str, Any] = {
        "checkpoint": _as_str(document.get("checkpoint")),
        "format_id": "gen3randombattle",  # scripts/mcts_acceptance_h2h.py:435
        "policy_mode": "engine-mcts",
        "leaf_eval": "model",  # scripts/mcts_acceptance_h2h.py:79-107
        "engine_depth": parsed.get("d"),
        "engine_sims": parsed.get("s"),
        "engine_batch": parsed.get("b"),
        "engine_worlds": parsed.get("w"),
        "rng_regime": "per-battle-stream",
        "decision_rng_seed": None,
    }
    missing = [
        name
        for letter, name in (
            ("d", "engine_depth"),
            ("s", "engine_sims"),
            ("b", "engine_batch"),
            ("w", "engine_worlds"),
        )
        if letter not in parsed
    ]
    missing += ["checkpoint_sha256", "device", "max_decision_rounds"]
    return fields, missing, None


_FIELD_READERS = {
    HARNESS_FOULPLAY_BRIDGE: _foulplay_fields,
    HARNESS_ROLLOUT_ACCEPTANCE: _acceptance_fields,
    HARNESS_ROLLOUT_HC_GRID: _hc_grid_fields,
    HARNESS_ROLLOUT_K0_GRID: _k0_grid_fields,
}


# --- fidelity ---------------------------------------------------------------

_SELF_PLAY_HARNESSES = frozenset(
    {HARNESS_ROLLOUT_ACCEPTANCE, HARNESS_ROLLOUT_HC_GRID, HARNESS_ROLLOUT_K0_GRID}
)


def _fidelity(harness: str, fields: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Verdict plus the evidence for it. See the module docstring."""
    if harness in _SELF_PLAY_HARNESSES:
        return FIDELITY_EXACT, (
            "both policies are pokezero, driven in-process off the battle seed "
            "(rollout.py:414-426), so the trajectory is reconstructible",
            "the decision RNG is a per-battle-per-seat stream advanced by every "
            "preceding decision, so round N is reachable only by replaying 0..N-1",
        )
    if harness == HARNESS_FOULPLAY_BRIDGE:
        notes = [
            "the external opponent searches on a wall-clock budget "
            "(foul-play fp/search/main.py:56) and samples its move with "
            "random.choices over the resulting visit shares (:46), so its move "
            "is not reproducible even though its RNG seed is pinned",
            "one divergent opponent move makes every later round a different "
            "battle, so the recorded round is not addressable by re-running the seed",
            "the decision RNG itself IS exactly reconstructible: "
            "random.Random(decision_rng_seed) (foulplay_bridge.py:3541)",
        ]
        if fields.get("opponent_random_seed") is None:
            notes.append(
                "the opponent's own seed is not recorded in this shard either"
            )
        return FIDELITY_OPPONENT_UNPINNED, tuple(notes)
    return FIDELITY_UNKNOWN, ("writer not recognised",)


# --- resolution -------------------------------------------------------------


def resolve_address(
    address: FallbackAddress, document: Mapping[str, Any]
) -> SpecResolution:
    """Join one address to the config of the shard that recorded it."""
    parsed = parse_battle_id(address.battle_id)
    if parsed is None:
        return UnresolvedAddress(
            address, f"unrecognised battle id grammar: {address.battle_id!r}"
        )
    id_harness, seed = parsed

    doc_harness = _document_harness(document)
    if doc_harness is None:
        return UnresolvedAddress(
            address, "shard does not identify its writer (no schema_version, no "
            "recognised key signature)",
        )
    if doc_harness != id_harness:
        # Never resolve through a disagreement. The two are independent readings
        # of the same fact and the config-field layout depends on which is right.
        return UnresolvedAddress(
            address,
            f"battle id names harness {id_harness!r} but the shard is {doc_harness!r}",
        )

    fields, missing, problem = _FIELD_READERS[doc_harness](document, address, seed)
    if problem is not None:
        return UnresolvedAddress(address, problem)

    fidelity, notes = _fidelity(doc_harness, fields)
    # `decision_rng_seed` is absent under the per-battle-stream regime by
    # construction, not by omission -- `rng_regime` already says so, and listing
    # it as missing would read as a shard defect the producer could fix.
    still_missing = tuple(
        sorted(
            {
                *missing,
                *(
                    name
                    for name, value in fields.items()
                    if value is None and name != "decision_rng_seed"
                ),
            }
        )
    )
    return ReplaySpec(
        battle_id=address.battle_id,
        round=address.round,
        seat=address.seat,
        reason=address.reason,
        key=address.key,
        source=address.source,
        harness=doc_harness,
        seed=seed,
        fidelity=fidelity,
        fidelity_notes=notes,
        missing=still_missing,
        **{k: v for k, v in fields.items() if v is not None},
    )


def _shard_documents(paths: Sequence[Path]) -> Iterator[tuple[str, Any]]:
    """Yield ``(label, document)`` for each shard, once.

    Path iteration is delegated to ``fallback_addresses._iter_shard_paths``
    rather than rewritten. That helper de-duplicates by *resolved* path, and its
    docstring records the defect that motivated it: overlapping arguments
    (``runs/ runs/a.json``) otherwise read one shard twice, which the
    per-document de-duplication one level down cannot see. Re-implementing the
    walk here would re-import that bug.
    """
    for path, label in _iter_shard_paths(paths):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(document, Mapping):
            yield label, document


def resolve_corpus(paths: Sequence[Path]) -> list[SpecResolution]:
    """Resolve every address reachable from ``paths`` into a spec or a failure."""
    resolutions: list[SpecResolution] = []
    for label, document in _shard_documents(paths):
        for address in iter_shard_addresses(document, source=label):
            resolutions.append(resolve_address(address, document))
    return resolutions


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pokezero.fallback_replay_spec",
        description=(
            "Resolve recorded fallback addresses into runnable replay specs. "
            "Reads shards; runs nothing."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Shard JSON files, or directories searched recursively for *.json.",
    )
    parser.add_argument(
        "--json-out", type=Path, help="Write the resolved specs to this path as JSON."
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Print only specs whose battle is exactly reconstructible.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    absent = [str(path) for path in args.paths if not path.exists()]
    if absent:
        print(f"path does not exist: {', '.join(absent)}")
        return 2

    resolutions = resolve_corpus(args.paths)
    specs = [r for r in resolutions if isinstance(r, ReplaySpec)]
    failures = [r for r in resolutions if isinstance(r, UnresolvedAddress)]
    if not resolutions:
        print("no fallback addresses found")
        return 1

    print(f"addresses: {len(resolutions)}   resolved: {len(specs)}   "
          f"unresolved: {len(failures)}")

    by_fidelity: Counter[str] = Counter(spec.fidelity for spec in specs)
    for verdict, count in sorted(by_fidelity.items()):
        print(f"  fidelity {verdict:20s} {count}")
    # Coverage, not a headline: a corpus gate that replays only the exactly
    # replayable share must say what share that is.
    if specs:
        exact = by_fidelity.get(FIDELITY_EXACT, 0)
        print(f"  exactly reconstructible: {exact}/{len(specs)}")

    shown = [s for s in specs if not args.exact_only or s.replayable_exactly]
    for spec in sorted(shown, key=lambda s: s.locator):
        print(
            f"\n{spec.harness}  seed={spec.seed} round={spec.round} seat={spec.seat}"
            f"  [{spec.fidelity}]\n  reason={spec.reason}\n  source={spec.source}"
        )
        if spec.missing:
            print(f"  NOT RECORDED BY THIS SHARD: {', '.join(spec.missing)}")

    if failures:
        print(f"\nUNRESOLVED ({len(failures)}):")
        for failure in failures:
            print(f"  {failure.address.battle_id} r{failure.address.round} "
                  f"{failure.address.seat}: {failure.problem}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "specs": [spec.to_dict() for spec in specs],
            "unresolved": [
                {
                    "battle_id": f.address.battle_id,
                    "round": f.address.round,
                    "seat": f.address.seat,
                    "key": f.address.key,
                    "source": f.address.source,
                    "problem": f.problem,
                }
                for f in failures
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {len(specs)} specs to {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
