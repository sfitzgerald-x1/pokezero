#!/usr/bin/env python3
"""V4 — full-game Python/Rust differential over the BOUNDARY encode surface.

WHAT THIS TESTS, stated plainly (the previous name of this file, "replay parity gate
(accumulator drift)", claimed something it structurally cannot do -- see NOT TESTED below):

For every decision point of a full self-play game, this takes the sanctioned per-row input
surface (``observation_metadata`` + ``public_materialization``, composed with production's own
``_public_materialization_payload`` + ``_json_safe`` helpers, exactly as ``engine_search.py``
composes the crate's input) and byte-compares:

    Python  ``backends.PythonReferenceBackend``  -> ``observation_from_player_state``
    native  ``backends.RustBackend``             -> ``pokezero_search.encode_decision``

over all five observation arrays. This is the BOUNDARY entry point (``encoder.rs`` docstring:
"The boundary-only entry point reproduces the sanctioned per-row surface"). What it genuinely
gates is the two sides' INDEPENDENT re-derivations from live, evolving game state: species and
forme resolution, base/expected stat arithmetic, belief-bag composition and bucketing,
categorical vocabulary mapping, mask construction, timed-condition arithmetic. That surface is
re-implemented twice, so it can drift twice -- and does; see the divergence report in the
accompanying commit.

NOT TESTED -- and no Python/Rust differential can test it, which is why this file no longer
claims to. The v4 k0 pack's B-columns (hazard credit, items-removed credit, switch propensity)
and the last-damage ledgers are ACCUMULATORS, but they are accumulated in exactly ONE place:
the Python protocol tracker. ``showdown.field_credit_values`` derives them once and publishes
the settled scalars on ``observation_metadata``; ``encoder.rs`` (~1245) reads those published
scalars verbatim ("derives these ONCE in Python and publishes the settled numbers on the
observation metadata, so this side reads them"), and ``backends.state_from_row_inputs`` reads
the same keys verbatim on the Python side. There is no second implementation, so there is no
cross-language accumulator DRIFT to detect: for those columns this differential is
copy-vs-copy of one scalar and would agree even if the accumulation were completely wrong.

The native LEAF lane does not accumulate them either, by design and already documented:
``scripts/leaf_vs_reality.py::V4_ROOT_FROZEN_PACK_COLUMNS`` (~165-212) classifies fifteen v4
pack columns as ROOT-FROZEN -- ``leaf.rs::leaf_row`` rewrites ~22 metadata keys from evolved
engine state and deliberately rewrites NONE of these, because re-deriving the gen3 hazard
grounding rule in Rust "is the duplication this design deliberately refuses". Grepping
``leaf.rs`` for ``credit|hazard|last_damage|switched_vs`` returns zero hits, which is that
design, not a defect. ``leaf_vs_reality.py`` is the harness that covers the leaf lane and it
correctly reports these as an accepted residual class rather than pretending to test them.

So the accumulation itself is single-implementation and must be gated against the SIM, not
against Python -- plan §3's first standard ("differentials compare against the engine or the
sim, never Python-vs-Python"). ``scripts/oracle_differential.py`` is that lane. What this file
adds on that axis is narrow and honest: it asserts the published accumulators actually MOVE
across a game (``pack_columns_varied``), so a tracker that silently froze fails here even
though both encoders would still agree byte-for-byte.

Exit criterion (plan §1, V4): byte-identical across >=200 full games (~20k states) with every
required v4 pack column reached. All three are ENFORCED by the verdict -- ``--games 1`` cannot
pass.

Per plan §3 the wheel must be rebuilt and reinstalled before results are read. The summary
records the sha256 of the loaded extension module, so the artifact says WHICH binary produced
the verdict rather than a constant venv path.

Usage:
    uv run python scripts/v4_boundary_encode_differential.py --games 200 --seed 3 \
        --showdown-root ~/workspace/pokerena/vendor/pokemon-showdown \
        --out runs/v4-boundary-encode-2026-08-04
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy  # noqa: E402
import pokezero_search  # noqa: E402

import engine_build_fingerprint as fingerprint  # noqa: E402
import golden_encoder_backends as backends  # noqa: E402
import export_encoder_tables as exporter  # noqa: E402
from pokezero.golden_corpus import _json_safe  # noqa: E402
from pokezero.local_showdown import (  # noqa: E402
    LocalShowdownConfig,
    LocalShowdownEnv,
    _public_materialization_payload,
)
from pokezero.observation import (  # noqa: E402
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    OBSERVATION_SCHEMA_VERSION_V4,
)
from pokezero.showdown import MOVE_ACTION_COUNT, observation_spec_for_schema  # noqa: E402

PLAYERS = ("p1", "p2")

# The v4 k0 pack's numeric columns, i.e. the ones this schema ADDS. Reachability is asserted
# PER COLUMN (finding B2: an OR over the set passed with one column lit, and
# NUMERIC_SELF_ITEMS_REMOVED_CREDIT reaches only ~3% of states, so the OR was carried by the
# common columns while the rare one went untested). This mirrors what
# tests/test_rust_encoder_v4.py:193-208 already asserts for its single fixture -- a full-game
# gate must not be WEAKER than the fixture test it claims to improve on.
#
# NUMERIC_OPP_HAZARD_EXPECTED is measured and REPORTED but not required: it is the opponent
# arm of the hazard-expectation pair and is unreachable from the public surface in self-play
# sampling (measured 0/40850 states, see --require-columns to demand it). Requiring a column
# that cannot be reached would make the gate unpassable; omitting it silently is what B2 was
# about, so it is named here instead.
V4_PACK_NUMERIC_COLUMNS = (
    "NUMERIC_SELF_HAZARD_CREDIT",
    "NUMERIC_OPP_HAZARD_CREDIT",
    "NUMERIC_SELF_HAZARD_EXPECTED",
    "NUMERIC_OPP_HAZARD_EXPECTED",
    "NUMERIC_SELF_ITEMS_REMOVED_CREDIT",
    "NUMERIC_OPP_ITEMS_REMOVED_CREDIT",
    "NUMERIC_MON_SWITCHED_VS_ACTIVE",
    "NUMERIC_MON_STAYED_VS_ACTIVE",
    "NUMERIC_LAST_DAMAGE_DEALT",
    "NUMERIC_LAST_DAMAGE_TAKEN",
    "NUMERIC_TRUANT_LOAF",
    "NUMERIC_CHOICE_LOCKED",
    "NUMERIC_ITEM_SWAPPED",
)
DEFAULT_REQUIRED_COLUMNS = tuple(
    name for name in V4_PACK_NUMERIC_COLUMNS if name != "NUMERIC_OPP_HAZARD_EXPECTED"
)

# The published accumulator scalars, read VERBATIM by both sides (so byte-parity over them is
# vacuous -- see the module docstring). What is NOT vacuous is whether they move at all, which
# is the one accumulation property this harness can honestly assert.
ACCUMULATOR_METADATA_KEYS = (
    "self_hazard_damage_suffered",
    "opponent_hazard_damage_suffered",
    "self_items_removed",
    "opponent_items_removed",
    "self_last_damage_dealt",
    "self_last_damage_taken",
    "opponent_last_damage_dealt",
    "opponent_last_damage_taken",
)

# Exit-criterion minimums (plan §1, V4). Enforced by the verdict, not merely documented.
DEFAULT_MIN_GAMES = 200
DEFAULT_MIN_STATES = 20_000

# How many DISTINCT mismatch signatures to keep an example of. A head cap over the mismatch
# list (the previous `mismatches[:40]`) is a FIFO slice, so every reported row comes from the
# first offending game and the report describes one game's failure mode as if it were the
# run's -- finding B1, which is what made the PR's own diagnosis wrong. Keying by signature
# means a second, different divergence in game 180 still appears.
MAX_SIGNATURE_EXAMPLES = 64


def _v4_header(spec) -> dict[str, Any]:
    """A minimal v4 corpus header, enough for ``observation_contract_from_header``."""
    masks = DEFAULT_OBSERVATION_FEATURE_MASKS
    return {
        "observation": {
            "schema_version": OBSERVATION_SCHEMA_VERSION_V4,
            "token_count": spec.token_count,
            "categorical_feature_count": spec.categorical_feature_count,
            "numeric_feature_count": spec.numeric_feature_count,
            # Stamped from the SAME defaults the env below is constructed with, so the two cannot
            # disagree about the contract under test.
            "feature_masks": {
                "stats_block": masks.opponent_tendency_stats_block,
                "exact_state": masks.exact_state,
                "transition_token_budget": masks.transition_token_budget,
                "tier2_residuals": masks.tier2_residuals,
                "tier2_investment": masks.tier2_investment,
            },
        }
    }


def native_build_identity() -> dict[str, Any]:
    """WHICH binary is loaded, in a form that distinguishes one wheel from another.

    Finding B4: the previous artifact read ``pokezero_search.ENGINE_BUILD_FINGERPRINT``, an
    attribute that does not exist, so ``native_fingerprint`` was always null and
    ``native_module`` was a constant venv path -- identical for a fresh wheel and a
    three-day-stale one, which is exactly the failure plan §3 exists to prevent. ``lib.rs``
    compiles in only ``__version__`` (the hand-bumped Cargo version), ``ENGINE_FEATURES`` and
    ``MODEL_FEATURE_ENABLED``, none of which move when the crate is edited -- so build
    identity has to come from the artifact itself. The extension sha256 does move: it is the
    compiled ``.so``.
    """
    identity: dict[str, Any] = {
        "package": getattr(pokezero_search, "__file__", None),
        "version": getattr(pokezero_search, "__version__", None),
        "engine_features": getattr(pokezero_search, "ENGINE_FEATURES", None),
        "model_feature_enabled": getattr(pokezero_search, "MODEL_FEATURE_ENABLED", None),
    }
    inner = getattr(pokezero_search, "pokezero_search", None)
    extension = Path(getattr(inner, "__file__", "") or "")
    if extension.is_file():
        stat = extension.stat()
        identity["extension"] = str(extension)
        identity["extension_sha256"] = hashlib.sha256(extension.read_bytes()).hexdigest()
        identity["extension_bytes"] = int(stat.st_size)
        identity["extension_mtime"] = _dt.datetime.fromtimestamp(
            stat.st_mtime, _dt.timezone.utc
        ).isoformat()
    else:  # pragma: no cover - a wheel without its extension cannot encode anything
        identity["extension"] = None
        identity["extension_sha256"] = None
    # The crate SOURCE fingerprint, so a stale binary is detectable as well as identifiable:
    # source moved + extension sha256 unchanged == the wheel was not rebuilt.
    try:
        identity["crate_source_fingerprint"] = fingerprint.compute_fingerprint()["fingerprint"]
    except Exception as error:  # pragma: no cover - source tree may be absent in a container
        identity["crate_source_fingerprint"] = f"unavailable: {type(error).__name__}: {error}"
    stamp = Path(sys.prefix) / ".engine-build-fingerprint.json"
    if stamp.is_file():
        try:
            identity["build_stamp"] = json.loads(stamp.read_text())
        except Exception as error:  # pragma: no cover
            identity["build_stamp"] = f"unreadable: {type(error).__name__}: {error}"
    return identity


def _row_inputs(env: LocalShowdownEnv, player: str, observation, *, seed: int) -> dict[str, Any]:
    """The crate's sanctioned input surface for one live decision point.

    Built with production's own helpers rather than re-derived here: `engine_search.py` composes the
    identical dict for the native policy, so a divergence found by this gate is a divergence the
    native leaf would really see.
    """
    state = env.public_materialization_state(player)
    return {
        "battle_id": "v4-boundary-encode-differential",
        "battle_seed": int(seed),
        "format_id": "gen3randombattle",
        "player_id": player,
        "observation_schema_version": observation.schema_version,
        "observation_metadata": _json_safe(
            dict(observation.metadata), context="observation_metadata"
        ),
        "public_materialization": _json_safe(
            _public_materialization_payload(state), context="public_materialization"
        ),
    }


class _MismatchLedger:
    """Per-column aggregation over the WHOLE run, with attribution for every array kind.

    Finding B1 had two halves. The head cap is fixed by aggregating counts over all states and
    keying examples by signature. The attribution gap is fixed by resolving differing indices
    to column NAMES for the categorical array too, not just the numeric one -- without that, a
    categorical divergence is reported as bare "categorical_ids differ" and the reader has to
    guess which feature moved. Guessing is what produced the wrong "the species categorical
    differs" claim: all 28 cosmetic ``species:unown*`` labels alias to a single vocabulary id,
    so the species cell is IDENTICAL; the column that actually diverges is CATEGORY_TYPE_1.
    """

    def __init__(self, *, numeric_columns: dict[str, int], categorical_columns: dict[str, int]):
        self._numeric_by_index = {int(v): k for k, v in numeric_columns.items()}
        self._categorical_by_index = {int(v): k for k, v in categorical_columns.items()}
        # states (not arrays, not rows) in which this column differed, over the whole run
        self.column_states: Counter = Counter()
        self.array_states: Counter = Counter()
        self.games: Counter = Counter()
        self.seat_kind: Counter = Counter()
        self.mismatched_states = 0
        self.mismatched_arrays = 0
        self._examples: dict[frozenset, dict[str, Any]] = {}

    def _columns_for(self, array: str, left: Any, right: Any) -> list[str]:
        lhs = numpy.asarray(left)
        rhs = numpy.asarray(right)
        if lhs.shape != rhs.shape:
            return [f"{array}:SHAPE({lhs.shape}!={rhs.shape})"]
        bad = numpy.argwhere(lhs != rhs)
        if not len(bad):  # pragma: no cover - bytes differed so values must
            return [f"{array}:BYTES_ONLY"]
        if lhs.ndim == 2:
            by_index = (
                self._categorical_by_index if array == "categorical_ids" else self._numeric_by_index
            )
            # Only 16 of the 41 categorical indices are NAMED in the exported layout -- the
            # belief-bag interior slots are positional. An unnamed index is labelled as such
            # rather than printed as a bare number, so nobody reads "categorical_ids:12" as a
            # column called 12. The species cell is CATEGORY_PRIMARY (index 0).
            return sorted(
                {
                    f"{array}:{by_index[int(col)]}"
                    if int(col) in by_index
                    else f"{array}:UNNAMED_INDEX_{int(col)}"
                    for _row, col in bad
                },
                key=str,
            )
        # 1-D arrays (token_type_ids, attention_mask, legal_action_mask): the index IS the
        # meaning, so attribute to the index. Capped only in the example, never in the counts.
        indices = sorted({int(i) for (i,) in bad})
        return [f"{array}:INDEX_{index}" for index in indices]

    def record(
        self,
        *,
        game: int,
        step: int,
        player: str,
        requested_seat: bool,
        differing: list[tuple[str, Any, Any]],
    ) -> None:
        self.mismatched_states += 1
        self.mismatched_arrays += len(differing)
        self.games[game] += 1
        self.seat_kind["requested" if requested_seat else "non_requested"] += 1
        columns: list[str] = []
        for array, left, right in differing:
            self.array_states[array] += 1
            columns.extend(self._columns_for(array, left, right))
        for key in set(columns):
            self.column_states[key] += 1
        signature = frozenset(columns)
        if signature not in self._examples and len(self._examples) < MAX_SIGNATURE_EXAMPLES:
            first = None
            for array, left, right in differing:
                lhs = numpy.asarray(left)
                rhs = numpy.asarray(right)
                if lhs.shape != rhs.shape:
                    continue
                bad = numpy.argwhere(lhs != rhs)
                if not len(bad):
                    continue
                index = tuple(int(v) for v in bad[0])
                first = {
                    "array": array,
                    "index": list(index),
                    "python": lhs[bad[0][0]].tolist()
                    if lhs.ndim == 1
                    else float(lhs[index[0], index[1]])
                    if array != "categorical_ids"
                    else int(lhs[index[0], index[1]]),
                    "rust": rhs[bad[0][0]].tolist()
                    if rhs.ndim == 1
                    else float(rhs[index[0], index[1]])
                    if array != "categorical_ids"
                    else int(rhs[index[0], index[1]]),
                }
                break
            self._examples[signature] = {
                "game": game,
                "step": step,
                "player": player,
                "requested_seat": requested_seat,
                "columns": sorted(signature, key=str),
                "first": first,
            }

    def payload(self) -> dict[str, Any]:
        return {
            "mismatched_states": self.mismatched_states,
            "mismatched_arrays": self.mismatched_arrays,
            # Over the WHOLE run, not a head slice: this is the corrected divergence report.
            "column_mismatch_states": dict(sorted(self.column_states.items())),
            "array_mismatch_states": dict(sorted(self.array_states.items())),
            "mismatch_games": sorted(self.games),
            "mismatch_states_by_seat_kind": dict(sorted(self.seat_kind.items())),
            "distinct_signatures": len(self._examples),
            "signature_examples": [
                self._examples[key] for key in sorted(self._examples, key=lambda s: sorted(s))
            ],
        }


def run_gate(
    *,
    showdown_root: Path,
    games: int = 200,
    seed: int = 3,
    max_steps: int = 400,
    move_bias: float = 0.75,
    seats: str = "both",
    min_games: int = DEFAULT_MIN_GAMES,
    min_states: int = DEFAULT_MIN_STATES,
    require_columns: tuple[str, ...] = DEFAULT_REQUIRED_COLUMNS,
) -> dict[str, Any]:
    if seats not in {"both", "requested"}:
        raise ValueError("seats must be 'both' or 'requested'")
    spec = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V4)
    header = _v4_header(spec)
    tables = exporter.build_tables(
        str(showdown_root), observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4
    )
    tables_json = json.dumps(tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    numeric_columns = {str(k): int(v) for k, v in tables["layout"]["numeric_columns"].items()}
    categorical_columns = {
        str(k): int(v) for k, v in tables["layout"]["categorical_columns"].items()
    }
    unknown = [name for name in require_columns if name not in numeric_columns]
    if unknown:
        raise ValueError(f"required columns absent from the v4 layout: {unknown}")

    python_backend = backends.PythonReferenceBackend(showdown_root=showdown_root, header=header)
    rust_backend = backends.RustBackend(tables_json=tables_json, header=header)

    env = LocalShowdownEnv(
        LocalShowdownConfig(
            showdown_root=str(showdown_root),
            observation_spec=spec,
            feature_masks=DEFAULT_OBSERVATION_FEATURE_MASKS,
            set_belief_source=True,
        )
    )

    counts: Counter = Counter(
        {
            "games": 0,
            "states": 0,
            "arrays_compared": 0,
            "states_requested_seat": 0,
            "states_non_requested_seat": 0,
            "truncated_games": 0,
        }
    )
    ledger = _MismatchLedger(
        numeric_columns=numeric_columns, categorical_columns=categorical_columns
    )
    # Per-column reachability over the whole run (B2), on the PYTHON side -- the reference.
    column_reach: Counter = Counter({name: 0 for name in V4_PACK_NUMERIC_COLUMNS})
    # Vacuity: numeric columns that were nonzero on NEITHER side in ANY state. Byte-parity over
    # those is vacuous and the summary must say so rather than let the reader assume 97 columns
    # were exercised. `state_from_row_inputs` rebuilds the transition/tendency/turn-merged
    # history EMPTY by design (its docstring says so), so those columns can never light up here.
    numeric_nonzero: set[str] = set()
    # Did the published accumulators actually MOVE? (the one accumulation claim in reach)
    accumulator_values: dict[str, set[float]] = {key: set() for key in ACCUMULATOR_METADATA_KEYS}

    try:
        for game in range(games):
            rng = random.Random(seed * 1_000_003 + game)
            env.reset(seed=seed + game)
            counts["games"] += 1
            steps = 0
            while steps < max_steps and env.terminal() is None:
                requested = env.requested_players()
                if not requested:
                    break
                # One observe() per (step, seat); the previous version called env.observe() three
                # to four times per step, re-deriving the whole observation to read a mask.
                observations = {player: env.observe(player) for player in PLAYERS}
                # Both seats by default: an accumulator that drifts only on the non-acting seat
                # is still a wrong input at the next search. The non-requested seat IS a shape
                # production never builds, so mismatches are counted BY SEAT KIND (see
                # `mismatch_states_by_seat_kind`) and `--seats requested` restricts to the
                # production-reachable shape when a finding needs to be attributed.
                compared = PLAYERS if seats == "both" else tuple(requested)
                for player in compared:
                    observation = observations[player]
                    row_inputs = _row_inputs(env, player, observation, seed=seed + game)
                    want = python_backend.encode(row_inputs)
                    got = rust_backend.encode(row_inputs)
                    counts["states"] += 1
                    is_requested = player in requested
                    counts[
                        "states_requested_seat" if is_requested else "states_non_requested_seat"
                    ] += 1

                    reference_numeric = numpy.asarray(want["numeric_features"])
                    native_numeric = numpy.asarray(got["numeric_features"])
                    for name in V4_PACK_NUMERIC_COLUMNS:
                        index = numeric_columns.get(name)
                        if index is not None and reference_numeric[:, index].any():
                            column_reach[name] += 1
                    for name, index in numeric_columns.items():
                        if name not in numeric_nonzero and (
                            reference_numeric[:, index].any() or native_numeric[:, index].any()
                        ):
                            numeric_nonzero.add(name)
                    metadata = row_inputs["observation_metadata"]
                    for key in ACCUMULATOR_METADATA_KEYS:
                        value = metadata.get(key)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            accumulator_values[key].add(float(value))

                    differing: list[tuple[str, Any, Any]] = []
                    for name in backends.ARRAY_NAMES:
                        counts["arrays_compared"] += 1
                        left = numpy.ascontiguousarray(want[name])
                        right = numpy.ascontiguousarray(got[name])
                        if left.tobytes() != right.tobytes():
                            differing.append((name, left, right))
                    if differing:
                        ledger.record(
                            game=game,
                            step=steps,
                            player=player,
                            requested_seat=is_requested,
                            differing=differing,
                        )
                actions = {}
                for player in requested:
                    mask = observations[player].legal_action_mask
                    legal = [index for index, allowed in enumerate(mask) if allowed]
                    if not legal:
                        break
                    moves = [index for index in legal if index < MOVE_ACTION_COUNT]
                    if moves and rng.random() < move_bias:
                        actions[player] = rng.choice(moves)
                    else:
                        actions[player] = rng.choice(legal)
                if len(actions) != len(requested):
                    counts["truncated_games"] += 1
                    break
                env.step(actions)
                steps += 1
    finally:
        env.close()

    unreached = [name for name in require_columns if column_reach[name] <= 0]
    varied = sorted(key for key, seen in accumulator_values.items() if len(seen) > 1)
    unvaried = sorted(key for key, seen in accumulator_values.items() if len(seen) <= 1)
    never_nonzero = sorted(set(numeric_columns) - numeric_nonzero)

    # The verdict enforces the plan's exit criterion instead of restating it in a docstring
    # (finding B2: `--games 1 --seed 41 --max-steps 2` printed PASS with four states, because
    # the only precondition was "at least one state lit at least one of nine columns").
    failures: list[str] = []
    if ledger.mismatched_states:
        failures.append(f"{ledger.mismatched_states} mismatched states")
    if counts["games"] < min_games:
        failures.append(f"games {counts['games']} < required {min_games}")
    if counts["states"] < min_states:
        failures.append(f"states {counts['states']} < required {min_states}")
    if unreached:
        failures.append(f"v4 pack columns never reached: {unreached}")
    if not varied:
        failures.append("no published accumulator scalar took more than one value")

    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "verdict": "FAIL" if failures else "PASS",
        "failures": failures,
        # WHICH binary produced this verdict (plan §3: a stale wheel's results were read as real
        # twice this generation).
        "native": native_build_identity(),
        "args": {
            "games": games,
            "seed": seed,
            "max_steps": max_steps,
            "move_bias": move_bias,
            "seats": seats,
            "min_games": min_games,
            "min_states": min_states,
            "require_columns": list(require_columns),
        },
        "counts": dict(counts),
        "reachability": {
            "v4_pack_states_reached": dict(sorted(column_reach.items())),
            "required_columns_unreached": unreached,
            "accumulator_scalars_varied": varied,
            "accumulator_scalars_constant": unvaried,
        },
        "vacuity": {
            # Named so nobody reads "97 numeric columns byte-identical" as 97 columns tested.
            "numeric_columns_never_nonzero": never_nonzero,
            "numeric_columns_never_nonzero_count": len(never_nonzero),
            "numeric_column_count": len(numeric_columns),
            "note": (
                "byte-parity over these columns is VACUOUS -- neither side ever wrote a nonzero "
                "value. backends.state_from_row_inputs rebuilds transition tokens, tendency "
                "aggregates and turn-merged rows EMPTY (its own docstring), so the history "
                "surface cannot be exercised through this boundary lane; "
                "tests/test_rust_encoder_v3.py drives it via RustFoldBackend.encode_with_fold."
            ),
        },
        "divergence": ledger.payload(),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    native = summary["native"]
    reach = summary["reachability"]
    divergence = summary["divergence"]
    print(
        f"[v4-boundary-encode] {summary['verdict']} games={counts.get('games', 0)} "
        f"states={counts.get('states', 0)} arrays={counts.get('arrays_compared', 0)} "
        f"mismatched_states={divergence['mismatched_states']} "
        f"unreached={reach['required_columns_unreached']} "
        f"accumulators_varied={len(reach['accumulator_scalars_varied'])}"
        f"/{len(reach['accumulator_scalars_varied']) + len(reach['accumulator_scalars_constant'])} "
        f"native_so_sha256={str(native.get('extension_sha256'))[:16]}"
    )
    for reason in summary["failures"]:
        print(f"  FAIL: {reason}")
    if divergence["column_mismatch_states"]:
        print("  diverging columns (mismatched states over the WHOLE run):")
        for column, hits in sorted(
            divergence["column_mismatch_states"].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"    {hits:7d}  {column}")
    vacuous = summary["vacuity"]["numeric_columns_never_nonzero_count"]
    print(
        f"  vacuous numeric columns (never nonzero on either side): {vacuous}"
        f"/{summary['vacuity']['numeric_column_count']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--move-bias", type=float, default=0.75)
    parser.add_argument("--seats", choices=("both", "requested"), default="both")
    parser.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES)
    parser.add_argument("--min-states", type=int, default=DEFAULT_MIN_STATES)
    parser.add_argument(
        "--require-columns",
        default=",".join(DEFAULT_REQUIRED_COLUMNS),
        help="comma-separated v4 pack columns that must each be reached at least once",
    )
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = run_gate(
        showdown_root=args.showdown_root,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        move_bias=args.move_bias,
        seats=args.seats,
        min_games=args.min_games,
        min_states=args.min_states,
        require_columns=tuple(
            name.strip() for name in args.require_columns.split(",") if name.strip()
        ),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v4-boundary-encode-differential.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _print_summary(summary)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
