#!/usr/bin/env python3
"""Produce a uniform-rollout leaf artifact for the canonical vhprobe bank.

This is deliberately only the *pricing* half of the rollout-leaf arbiter.
The canonical bank records the two policy-selected actions and their
policy-continuation labels, but not the private successor states needed by the
native engine.  An authorised replay job must materialise those states first;
this program accepts that immutable state corpus, proves it has exactly the
bank's A/B keys, prices those states through the public native uniform-rollout
surface, and immediately invokes the pinned deploy-side estimand gate.  A
zero exit is not enough: the gate must write a fresh, schema-checked PASS
verdict for all 465 positions / 930 leaves before this command calls its
output validated.

It never reconstructs a state from a seed alone and never substitutes another
checkpoint.  Doing either would make a neat 930-row artifact for a different
question.  In particular, the state corpus must contain the original
checkpoint, belief, Showdown, and engine identities and one fixed opponent
action shared by both sibling arms at each position.

Input ``--leaf-states`` schema::

    {
      "schema": "pokezero.rollout-leaf-states.v1",
      "provenance": {
        "bank_sha256": "...",
        "checkpoint_sha256": "...",
        "belief_set_source_hash": "...",
        "engine_build_fingerprint": "...",
        "showdown_commit": "...",
        "state_builder_source_commit": "...",
        "branch_rule": "policy_top2_fixed_opponent_joint_successor.v1"
      },
      "leaves": [
        {
          "seed": 24010000, "prefix": 4, "seat": "p1", "arm": 0,
          "subject_action": 0, "opponent_action": 3,
          "state": "serialized poke-engine state", "state_sha256": "..."
        }
      ]
    }

The output intentionally does not copy the serialized states.  It carries the
keyed state hashes, native extension digest, writer digest, and the full
terminal/fallback rollout ledger so the deploy validator can join values to the
bank without publishing private state material.

An action that completes the battle has no successor observation to
materialize. Its input row therefore carries an uncapped terminal record plus
the exact side-one value (1, 0, or 0.5) and never enters the native pricer.
That is an exact uniform-continuation value, not a rollout fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATE_SCHEMA = "pokezero.rollout-leaf-states.v1"
OUTPUT_SCHEMA = "pokezero.rollout-leaf-values.v1"
ROW_PRICER_SCHEMA = "pokezero.uniform-rollout-row-prices.v1"
VERDICT_SCHEMA = "pokezero.rollout-leaf-estimand-verdict.v1"
BRANCH_RULE = "policy_top2_fixed_opponent_joint_successor.v1"
CANONICAL_POSITIONS = 465
CANONICAL_LEAVES = CANONICAL_POSITIONS * 2
# This is the SHA256 of the reviewed deploy-side reader at deploy commit
# 4a8208c.  The writer is intentionally coupled to that concrete gate: a
# path supplied by an operator is not evidence that the required reader ran.
REVIEWED_DEPLOY_VALIDATOR_SHA256 = "67e7663aed5d4bb3072cd18375947547d28ff206767cbacf77715ecf7e1b2d50"


class UniformLeafWriterError(ValueError):
    """The supplied corpus cannot support a canonical uniform-leaf artifact."""


@dataclass(frozen=True, order=True)
class LeafKey:
    seed: int
    prefix: int
    seat: str
    arm: int

    def as_dict(self) -> dict[str, int | str]:
        return {"seed": self.seed, "prefix": self.prefix, "seat": self.seat, "arm": self.arm}


@dataclass(frozen=True)
class LeafState:
    key: LeafKey
    subject_action: int
    opponent_action: int
    state: str | None
    state_sha256: str | None
    terminal_value: float | None = None
    terminal_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _terminal_successor(
    row: Mapping[str, Any], *, label: str
) -> tuple[float, str]:
    """Validate an exact post-branch terminal result in side-one coordinates.

    These rows deliberately do not enter the native rollout pricer: uniform
    continuation after a completed battle is already known exactly.  A capped
    trajectory is not a completed battle and must never be smuggled in as a
    terminal value.
    """

    terminal = row.get("terminal")
    if not isinstance(terminal, Mapping):
        raise UniformLeafWriterError(
            f"{label} must supply a non-empty state or an exact terminal successor"
        )
    if set(terminal) != {"winner", "turn_count", "capped"}:
        raise UniformLeafWriterError(
            f"{label}.terminal must contain exactly winner, turn_count, and capped"
        )
    winner = terminal.get("winner")
    if winner not in {"p1", "p2", None}:
        raise UniformLeafWriterError(f"{label}.terminal.winner must be p1, p2, or null")
    turn_count = _int(terminal.get("turn_count"), field=f"{label}.terminal.turn_count")
    if turn_count < 0:
        raise UniformLeafWriterError(f"{label}.terminal.turn_count must be non-negative")
    if terminal.get("capped") is not False:
        raise UniformLeafWriterError(
            f"{label}.terminal must be an uncapped completed battle, not a rollout cap"
        )
    value = 1.0 if winner == "p1" else 0.0 if winner == "p2" else 0.5
    declared_value = _probability(row.get("terminal_value"), field=f"{label}.terminal_value")
    if declared_value != value:
        raise UniformLeafWriterError(
            f"{label}.terminal_value must equal the exact side-one terminal result {value}"
        )
    terminal_sha256 = _hex(row.get("terminal_sha256"), field=f"{label}.terminal_sha256", length=64)
    actual_terminal_sha256 = hashlib.sha256(canonical_json(dict(terminal))).hexdigest()
    if terminal_sha256 != actual_terminal_sha256:
        raise UniformLeafWriterError(
            f"{label}.terminal_sha256 does not match the canonical terminal record"
        )
    return value, terminal_sha256


def _load_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UniformLeafWriterError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UniformLeafWriterError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise UniformLeafWriterError(f"{label} must be a JSON object, not {type(document).__name__}")
    return document


def _int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise UniformLeafWriterError(f"{field} must be an integer, not boolean {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise UniformLeafWriterError(f"{field} must be an integer, got {value!r}") from exc
    if result != value:
        raise UniformLeafWriterError(f"{field} must be an integer, got {value!r}")
    return result


def _hex(value: Any, *, field: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise UniformLeafWriterError(f"{field} must be {length} lowercase hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise UniformLeafWriterError(f"{field} must be {length} lowercase hexadecimal characters") from exc
    if value != value.lower():
        raise UniformLeafWriterError(f"{field} must be {length} lowercase hexadecimal characters")
    return value


def _key(row: Mapping[str, Any], *, label: str, arm: Any | None = None) -> LeafKey:
    seat = row.get("seat")
    if not isinstance(seat, str) or not seat:
        raise UniformLeafWriterError(f"{label}.seat must be a non-empty string")
    return LeafKey(
        seed=_int(row.get("seed"), field=f"{label}.seed"),
        prefix=_int(row.get("prefix"), field=f"{label}.prefix"),
        seat=seat,
        arm=_int(row.get("arm") if arm is None else arm, field=f"{label}.arm"),
    )


def load_banked_keys(path: Path) -> tuple[set[LeafKey], str]:
    """Read only the canonical A/B key set and its one belief-source identity."""

    document = _load_object(path, label="banked pairs artifact")
    pairs = document.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise UniformLeafWriterError("banked pairs artifact must carry a non-empty pairs list")
    keys: set[LeafKey] = set()
    positions: set[tuple[int, int, str]] = set()
    for index, row in enumerate(pairs):
        label = f"banked pairs.pairs[{index}]"
        if not isinstance(row, Mapping):
            raise UniformLeafWriterError(f"{label} must be an object")
        a = _key(row, label=f"{label}.arm_a", arm=row.get("arm_a"))
        b = _key(row, label=f"{label}.arm_b", arm=row.get("arm_b"))
        if a.arm == b.arm:
            raise UniformLeafWriterError(f"{label} names the same arm twice")
        position = (a.seed, a.prefix, a.seat)
        if position in positions:
            raise UniformLeafWriterError(f"banked pairs artifact duplicates position {position!r}")
        positions.add(position)
        keys.update((a, b))
    if len(keys) != 2 * len(positions):
        raise UniformLeafWriterError("banked pairs artifact reuses a leaf key across positions")

    provenance = document.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise UniformLeafWriterError("banked pairs artifact must carry non-empty provenance")
    belief_hashes = {
        _hex(row.get("belief_set_source_hash"), field=f"banked provenance[{index}].belief_set_source_hash", length=16)
        for index, row in enumerate(provenance)
        if isinstance(row, Mapping)
    }
    if len(belief_hashes) != 1:
        raise UniformLeafWriterError(
            "banked pairs artifact must have exactly one belief_set_source_hash across its shards"
        )
    return keys, next(iter(belief_hashes))


def _require_state_provenance(
    provenance: Mapping[str, Any], *, bank_sha256: str, belief_hash: str, checkpoint_sha256: str
) -> dict[str, str]:
    if _hex(provenance.get("bank_sha256"), field="state corpus provenance.bank_sha256", length=64) != bank_sha256:
        raise UniformLeafWriterError("state corpus provenance.bank_sha256 does not match --banked-pairs")
    if _hex(
        provenance.get("checkpoint_sha256"), field="state corpus provenance.checkpoint_sha256", length=64
    ) != checkpoint_sha256:
        raise UniformLeafWriterError("state corpus checkpoint_sha256 does not match --checkpoint-sha256")
    if _hex(
        provenance.get("belief_set_source_hash"), field="state corpus provenance.belief_set_source_hash", length=16
    ) != belief_hash:
        raise UniformLeafWriterError("state corpus belief_set_source_hash does not match the bank")
    if provenance.get("branch_rule") != BRANCH_RULE:
        raise UniformLeafWriterError(
            "state corpus provenance.branch_rule must be " + repr(BRANCH_RULE)
        )
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "belief_set_source_hash": belief_hash,
        "engine_build_fingerprint": _hex(
            provenance.get("engine_build_fingerprint"),
            field="state corpus provenance.engine_build_fingerprint",
            length=64,
        ),
        "showdown_commit": _hex(
            provenance.get("showdown_commit"), field="state corpus provenance.showdown_commit", length=40
        ),
        "state_builder_source_commit": _hex(
            provenance.get("state_builder_source_commit"),
            field="state corpus provenance.state_builder_source_commit",
            length=40,
        ),
    }


def load_leaf_states(
    path: Path, *, expected_keys: set[LeafKey], bank_sha256: str, belief_hash: str, checkpoint_sha256: str
) -> tuple[dict[LeafKey, LeafState], dict[str, str]]:
    """Load states, prove their content hashes, and require the exact bank key set."""

    document = _load_object(path, label="leaf-state corpus")
    if document.get("schema") != STATE_SCHEMA:
        raise UniformLeafWriterError(
            f"leaf-state corpus schema must be {STATE_SCHEMA!r}, got {document.get('schema')!r}"
        )
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise UniformLeafWriterError("leaf-state corpus must carry an object provenance")
    checked_provenance = _require_state_provenance(
        provenance,
        bank_sha256=bank_sha256,
        belief_hash=belief_hash,
        checkpoint_sha256=checkpoint_sha256,
    )
    leaves = document.get("leaves")
    if not isinstance(leaves, list) or not leaves:
        raise UniformLeafWriterError("leaf-state corpus must carry a non-empty leaves list")
    result: dict[LeafKey, LeafState] = {}
    opponent_by_position: dict[tuple[int, int, str], int] = {}
    for index, row in enumerate(leaves):
        label = f"leaf-state corpus.leaves[{index}]"
        if not isinstance(row, Mapping):
            raise UniformLeafWriterError(f"{label} must be an object")
        key = _key(row, label=label)
        if key in result:
            raise UniformLeafWriterError(f"leaf-state corpus duplicates leaf key {key!r}")
        subject_action = _int(row.get("subject_action"), field=f"{label}.subject_action")
        if subject_action != key.arm:
            raise UniformLeafWriterError(f"{label}.subject_action must equal its canonical arm")
        opponent_action = _int(row.get("opponent_action"), field=f"{label}.opponent_action")
        state = row.get("state")
        has_state = isinstance(state, str) and bool(state)
        has_terminal = "terminal" in row
        if has_state == has_terminal:
            raise UniformLeafWriterError(
                f"{label} must supply exactly one of a non-empty state or an exact terminal successor"
            )
        state_sha256: str | None = None
        terminal_value: float | None = None
        terminal_sha256: str | None = None
        if has_state:
            state_sha256 = _hex(row.get("state_sha256"), field=f"{label}.state_sha256", length=64)
            actual_state_sha256 = hashlib.sha256(state.encode("utf-8")).hexdigest()
            if actual_state_sha256 != state_sha256:
                raise UniformLeafWriterError(
                    f"{label}.state_sha256 does not match the serialized state bytes"
                )
            if "terminal_value" in row or "terminal_sha256" in row:
                raise UniformLeafWriterError(f"{label} cannot mix state and terminal successor fields")
        else:
            if state is not None or "state_sha256" in row:
                raise UniformLeafWriterError(f"{label}.state must be absent for an exact terminal successor")
            terminal_value, terminal_sha256 = _terminal_successor(row, label=label)
        position = (key.seed, key.prefix, key.seat)
        existing_opponent = opponent_by_position.setdefault(position, opponent_action)
        if existing_opponent != opponent_action:
            raise UniformLeafWriterError(
                f"{label}.opponent_action differs between the two arms at position {position!r}"
            )
        result[key] = LeafState(
            key,
            subject_action,
            opponent_action,
            state if has_state else None,
            state_sha256,
            terminal_value,
            terminal_sha256,
        )
    missing = sorted(expected_keys - set(result))
    foreign = sorted(set(result) - expected_keys)
    if missing or foreign:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} canonical leaves (first {missing[0]!r})")
        if foreign:
            parts.append(f"contains {len(foreign)} foreign leaves (first {foreign[0]!r})")
        raise UniformLeafWriterError("leaf-state corpus must match the bank key set exactly: " + "; ".join(parts))
    return result, checked_provenance


def ordinal_for_key(key: LeafKey) -> int:
    """Stable independent RNG stream key, refusing collision at the caller."""

    digest = hashlib.sha256(canonical_json(key.as_dict())).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise UniformLeafWriterError(f"{field} must be a finite probability, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise UniformLeafWriterError(f"{field} must be a finite probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise UniformLeafWriterError(f"{field} must be a finite probability in [0, 1]")
    return result


def _native_pricer(
    states: list[str], ordinals: list[int], *, rollouts: int, max_plies: int, seed: int,
    threads: int, branch_on_damage: bool,
) -> tuple[Mapping[str, Any], str]:
    try:
        import pokezero_search
    except ImportError as exc:  # pragma: no cover - exercised by the real collection environment
        raise UniformLeafWriterError("pokezero_search native module is required to price leaf states") from exc
    if not hasattr(pokezero_search, "price_uniform_rollout_rows"):
        raise UniformLeafWriterError(
            "installed pokezero_search lacks price_uniform_rollout_rows; rebuild the current public search seam"
        )
    try:
        report = json.loads(
            pokezero_search.price_uniform_rollout_rows(
                states,
                ordinals,
                rollouts=rollouts,
                rollout_max_plies=max_plies,
                rollout_seed=seed,
                rollout_threads=threads,
                rollout_branch_on_damage=branch_on_damage,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise UniformLeafWriterError(f"native uniform row pricer did not return valid JSON: {exc}") from exc
    if not isinstance(report, Mapping):
        raise UniformLeafWriterError("native uniform row pricer must return an object")
    return report, _native_extension_sha256(pokezero_search)


def _native_extension_sha256(package: Any) -> str:
    """Hash the compiled module across supported maturin import layouts.

    A wheel may expose ``pokezero_search`` as the extension itself, or as a
    Python package whose ``.pokezero_search`` child is the extension.  Never
    hash ``__init__.py`` just because that is the convenient import object.
    """

    package_path = getattr(package, "__file__", None)
    extension = package if isinstance(package_path, str) and package_path.endswith((".so", ".pyd")) else getattr(
        package, "pokezero_search", None
    )
    extension_path = getattr(extension, "__file__", None)
    if not isinstance(extension_path, str) or not extension_path.endswith((".so", ".pyd")):
        raise UniformLeafWriterError(
            "cannot identify the loaded pokezero_search native extension for provenance"
        )
    return sha256_file(Path(extension_path))


def _snapshot_input(path: Path, *, snapshot_dir: Path, label: str, prefix: str) -> tuple[Path, str]:
    """Copy an input once and return the private copy with its exact digest.

    The bank, state corpus, and reader all contribute to one validation
    transaction.  Re-opening one of their public paths after an expensive
    price pass would permit a later reader to see different bytes from the
    provenance and values the writer produced.
    """

    try:
        source = path.read_bytes()
    except OSError as exc:
        raise UniformLeafWriterError(f"could not read {label}: {exc}") from exc
    actual = hashlib.sha256(source).hexdigest()
    descriptor, snapshot_name = tempfile.mkstemp(dir=snapshot_dir, prefix=prefix)
    snapshot = Path(snapshot_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source)
        if sha256_file(snapshot) != actual:
            raise UniformLeafWriterError(f"could not preserve {label} bytes")
    except Exception:
        snapshot.unlink(missing_ok=True)
        raise
    return snapshot, actual


def _snapshot_reviewed_validator(path: Path, *, snapshot_dir: Path) -> tuple[Path, str]:
    """Copy the reviewed reader once, then execute only those verified bytes.

    The external path is an operator input and can change while pricing 930
    leaves.  Hashing it before that work and re-opening it afterwards would
    merely turn a reviewer-approved reader into a pathname race.  Hash the
    exact bytes we copy into a private temporary file instead; the executable
    snapshot is then independent of later source-path replacements.
    """

    snapshot, actual = _snapshot_input(
        path,
        snapshot_dir=snapshot_dir,
        label="--validator",
        prefix="reviewed-deploy-estimand-gate-",
    )
    if actual != REVIEWED_DEPLOY_VALIDATOR_SHA256:
        snapshot.unlink(missing_ok=True)
        raise UniformLeafWriterError(
            "--validator is not the reviewed deploy estimand gate; its SHA256 does not match "
            "the pinned reader identity"
        )
    return snapshot, actual


def _require_passing_verdict(
    path: Path,
    *,
    expected_positions: int,
    expected_leaves: int,
    expected_uniform_leaf_artifact_sha256: str,
) -> None:
    """Require the gate's fresh, complete PASS result rather than its exit code alone."""

    verdict = _load_object(path, label="deploy estimand verdict")
    if verdict.get("schema") != VERDICT_SCHEMA:
        raise UniformLeafWriterError("deploy estimand verdict has an unknown schema")
    if verdict.get("verdict") != "PASS" or verdict.get("failures") != []:
        raise UniformLeafWriterError("deploy estimand gate did not return an unconditional PASS verdict")
    if _int(verdict.get("n_positions"), field="deploy estimand verdict.n_positions") != expected_positions:
        raise UniformLeafWriterError(
            f"deploy estimand verdict covers the wrong position count; expected {expected_positions}"
        )
    if _int(verdict.get("n_leaves"), field="deploy estimand verdict.n_leaves") != expected_leaves:
        raise UniformLeafWriterError(
            f"deploy estimand verdict covers the wrong leaf count; expected {expected_leaves}"
        )
    actual_uniform_sha256 = _hex(
        verdict.get("uniform_leaf_artifact_sha256"),
        field="deploy estimand verdict.uniform_leaf_artifact_sha256",
        length=64,
    )
    if actual_uniform_sha256 != expected_uniform_leaf_artifact_sha256:
        raise UniformLeafWriterError(
            "deploy estimand verdict is not bound to the exact uniform artifact bytes it assessed"
        )


def _run_reviewed_validator(
    validator: Path, *, banked_pairs: Path, uniform_leaves: Path, verdict_out: Path,
    expected_positions: int, expected_leaves: int,
) -> int:
    try:
        completed = subprocess.run(
            [
                sys.executable, str(validator),
                "--banked-pairs", str(banked_pairs),
                "--uniform-leaves", str(uniform_leaves),
                "--out", str(verdict_out),
            ],
            check=False,
        )
    except OSError as exc:
        raise UniformLeafWriterError(f"could not execute the reviewed deploy estimand gate: {exc}") from exc
    if completed.returncode != 0:
        return completed.returncode
    _require_passing_verdict(
        verdict_out,
        expected_positions=expected_positions,
        expected_leaves=expected_leaves,
        expected_uniform_leaf_artifact_sha256=sha256_file(uniform_leaves),
    )
    return 0


def _state_manifest_sha256(rows: Sequence[LeafState]) -> str:
    manifest = [
        (
            {
                **row.key.as_dict(),
                "opponent_action": row.opponent_action,
                "state_sha256": row.state_sha256,
                "value_source": "native_uniform_rollout",
            }
            if row.state is not None
            else {
                **row.key.as_dict(),
                "opponent_action": row.opponent_action,
                "terminal_sha256": row.terminal_sha256,
                "terminal_value": row.terminal_value,
                "value_source": "exact_terminal",
            }
        )
        for row in rows
    ]
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def build_uniform_leaf_document(
    rows: Mapping[LeafKey, LeafState],
    state_provenance: Mapping[str, str],
    *,
    bank_sha256: str,
    state_corpus_sha256: str,
    rollouts: int,
    max_plies: int,
    seed: int,
    threads: int,
    branch_on_damage: bool,
    price: Callable[..., tuple[Mapping[str, Any], str]],
    writer_sha256: str,
    validator_sha256: str,
) -> dict[str, Any]:
    """Price sorted canonical rows and return the validator's input document.

    ``price`` is injected so the shape/provenance contract has stdlib-only
    tests.  The production path is `_native_pricer` above.
    """

    ordered = [rows[key] for key in sorted(rows)]
    native_rows = [row for row in ordered if row.state is not None]
    if not native_rows:
        raise UniformLeafWriterError(
            "canonical leaf-state corpus has no nonterminal leaves to attest the native uniform pricer"
        )
    ordinals = [ordinal_for_key(row.key) for row in native_rows]
    if len(set(ordinals)) != len(ordinals):
        raise UniformLeafWriterError("canonical leaf-key hash collision; refusing shared rollout streams")
    report, extension_sha256 = price(
        [row.state for row in native_rows],
        ordinals,
        rollouts=rollouts,
        max_plies=max_plies,
        seed=seed,
        threads=threads,
        branch_on_damage=branch_on_damage,
    )
    if report.get("schema") != ROW_PRICER_SCHEMA:
        raise UniformLeafWriterError("native row pricer returned an unknown schema")
    if report.get("value_frame") != "side_one_absolute":
        raise UniformLeafWriterError("native row pricer did not declare side_one_absolute values")
    if report.get("rollout_policy") != "uniform":
        raise UniformLeafWriterError("native row pricer did not declare uniform continuation")
    expected_pricer_config = {
        "rollouts": rollouts,
        "rollout_max_plies": max_plies,
        "rollout_seed": seed,
        "rollout_threads": threads,
        "rollout_branch_on_damage": branch_on_damage,
    }
    for field, expected in expected_pricer_config.items():
        if report.get(field) != expected:
            raise UniformLeafWriterError(
                f"native row pricer returned {field}={report.get(field)!r}, expected {expected!r}"
            )
    values = report.get("values")
    if not isinstance(values, list) or len(values) != len(native_rows):
        raise UniformLeafWriterError("native row pricer returned the wrong number of leaf values")
    ledger_fields = (
        "leaves_priced", "rollouts_run", "rollout_plies", "rollout_terminal_hits",
        "rollout_cap_hits", "rollout_dead_ends", "rollout_terminal_fraction",
        "rollout_fallback_fraction", "rollout_mean_plies",
    )
    if any(field not in report for field in ledger_fields):
        raise UniformLeafWriterError("native row pricer omitted terminal/fallback ledger fields")
    if (
        report["leaves_priced"] != len(native_rows)
        or report["rollouts_run"] != len(native_rows) * rollouts
    ):
        raise UniformLeafWriterError("native row pricer ledger does not account for every requested rollout")
    # `rollout_once` has a deliberately visible HP-fraction fallback for the
    # search arm: a cap or an engine dead end must not turn a whole game into
    # a process crash. That rescue is NOT part of this validation estimand,
    # which is the uniform policy's terminal continuation from each canonical
    # successor. Recording a nonzero fallback rate would make the blend
    # auditable, but still lets it be joined and called "uniform" by the
    # deploy reader. Refuse it here, before an artifact exists.
    if (
        report["rollout_terminal_hits"] != report["rollouts_run"]
        or report["rollout_cap_hits"] != 0
        or report["rollout_dead_ends"] != 0
        or report["rollout_terminal_fraction"] != 1.0
        or report["rollout_fallback_fraction"] != 0.0
    ):
        raise UniformLeafWriterError(
            "native row pricer had nonterminal rollouts; cap/dead-end HP-fraction fallbacks "
            "cannot answer the uniform terminal-continuation estimand"
        )
    native_values = {
        row.key: (_probability(value, field=f"native values[{index}]"), ordinal)
        for index, (row, ordinal, value) in enumerate(zip(native_rows, ordinals, values))
    }
    leaves: list[dict[str, Any]] = []
    for row in ordered:
        if row.state is not None:
            value, ordinal = native_values[row.key]
            leaves.append(
                {
                    **row.key.as_dict(),
                    "uniform_value": value,
                    "opponent_action": row.opponent_action,
                    "state_sha256": row.state_sha256,
                    "ordinal": ordinal,
                    "value_source": "native_uniform_rollout",
                }
            )
            continue
        if row.terminal_value is None or row.terminal_sha256 is None:
            raise UniformLeafWriterError("terminal leaf lost its exact successor evidence")
        leaves.append(
            {
                **row.key.as_dict(),
                "uniform_value": row.terminal_value,
                "opponent_action": row.opponent_action,
                "terminal_sha256": row.terminal_sha256,
                "value_source": "exact_terminal",
            }
        )
    return {
        "schema": OUTPUT_SCHEMA,
        "provenance": {
            "rollout_policy": "uniform",
            "value_frame": "side_one_absolute",
            "rollouts": rollouts,
            "rollout_max_plies": max_plies,
            "rollout_seed": seed,
            "rollout_threads": threads,
            "rollout_branch_on_damage": branch_on_damage,
            "native_priced_leaves": len(native_rows),
            "terminal_successor_leaves": len(ordered) - len(native_rows),
            "bank_sha256": bank_sha256,
            "state_corpus_sha256": state_corpus_sha256,
            "state_manifest_sha256": _state_manifest_sha256(ordered),
            "checkpoint_sha256": state_provenance["checkpoint_sha256"],
            "belief_set_source_hash": state_provenance["belief_set_source_hash"],
            "engine_build_fingerprint": state_provenance["engine_build_fingerprint"],
            "showdown_commit": state_provenance["showdown_commit"],
            "state_builder_source_commit": state_provenance["state_builder_source_commit"],
            "branch_rule": BRANCH_RULE,
            "native_extension_sha256": _hex(extension_sha256, field="native extension sha256", length=64),
            "writer_sha256": _hex(writer_sha256, field="writer sha256", length=64),
            "deploy_validator_sha256": _hex(
                validator_sha256, field="deploy validator sha256", length=64
            ),
        },
        "rollout_ledger": {
            **{field: report[field] for field in ledger_fields},
            "native_priced_leaves": len(native_rows),
            "terminal_successor_leaves": len(ordered) - len(native_rows),
            "total_leaves": len(ordered),
        },
        "leaves": leaves,
    }


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise UniformLeafWriterError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        # `replace` would overwrite a file another actor created after our
        # preflight. A same-directory hard link is atomic create-only, so the
        # transaction never destroys an intervening evidence file.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise UniformLeafWriterError(f"refusing to overwrite existing output: {path}") from exc
    except OSError as exc:
        raise UniformLeafWriterError(f"could not publish new output {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banked-pairs", type=Path, required=True)
    parser.add_argument("--leaf-states", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True,
                        help="the immutable checkpoint digest expected for the state corpus")
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--rollout-max-plies", type=int, required=True)
    parser.add_argument("--rollout-seed", type=int, required=True)
    parser.add_argument("--rollout-threads", type=int, required=True)
    parser.add_argument("--rollout-branch-on-damage", choices=("true", "false"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True,
                        help="deploy mcts/validate_rollout_leaf_estimand.py")
    parser.add_argument("--verdict-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checkpoint_sha256 = _hex(args.checkpoint_sha256, field="--checkpoint-sha256", length=64)
        if args.rollouts <= 0 or args.rollout_max_plies <= 0 or args.rollout_threads <= 0:
            raise UniformLeafWriterError("rollouts, rollout-max-plies, and rollout-threads must all be > 0")
        if args.rollout_seed < 0:
            raise UniformLeafWriterError("rollout-seed must be >= 0")
        if args.out.exists() or args.verdict_out.exists():
            raise UniformLeafWriterError("refusing to overwrite --out or --verdict-out")
        if args.out.resolve() == args.verdict_out.resolve():
            raise UniformLeafWriterError("--out and --verdict-out must name different output paths")
        with tempfile.TemporaryDirectory(prefix="pokezero-uniform-leaf-validator-") as directory:
            transaction_dir = Path(directory)
            bank_snapshot, bank_sha256 = _snapshot_input(
                args.banked_pairs,
                snapshot_dir=transaction_dir,
                label="--banked-pairs",
                prefix="banked-pairs-",
            )
            leaf_states_snapshot, leaf_states_sha256 = _snapshot_input(
                args.leaf_states,
                snapshot_dir=transaction_dir,
                label="--leaf-states",
                prefix="leaf-states-",
            )
            validator_snapshot, validator_sha256 = _snapshot_reviewed_validator(
                args.validator, snapshot_dir=transaction_dir
            )
            expected_keys, belief_hash = load_banked_keys(bank_snapshot)
            if len(expected_keys) != CANONICAL_LEAVES:
                raise UniformLeafWriterError(
                    f"banked pairs artifact has {len(expected_keys)} leaves; expected canonical {CANONICAL_LEAVES}"
                )
            rows, state_provenance = load_leaf_states(
                leaf_states_snapshot,
                expected_keys=expected_keys,
                bank_sha256=bank_sha256,
                belief_hash=belief_hash,
                checkpoint_sha256=checkpoint_sha256,
            )
            document = build_uniform_leaf_document(
                rows,
                state_provenance,
                bank_sha256=bank_sha256,
                state_corpus_sha256=leaf_states_sha256,
                rollouts=args.rollouts,
                max_plies=args.rollout_max_plies,
                seed=args.rollout_seed,
                threads=args.rollout_threads,
                branch_on_damage=args.rollout_branch_on_damage == "true",
                price=_native_pricer,
                writer_sha256=sha256_file(Path(__file__).resolve()),
                validator_sha256=validator_sha256,
            )
            document_text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
            uniform_snapshot = transaction_dir / "uniform-leaves.json"
            verdict_snapshot = transaction_dir / "estimand-verdict.json"
            _write_new(uniform_snapshot, document_text)
            try:
                return_code = _run_reviewed_validator(
                    validator_snapshot,
                    banked_pairs=bank_snapshot,
                    uniform_leaves=uniform_snapshot,
                    verdict_out=verdict_snapshot,
                    expected_positions=CANONICAL_POSITIONS,
                    expected_leaves=CANONICAL_LEAVES,
                )
            except UniformLeafWriterError as exc:
                print(f"CANNOT VALIDATE UNIFORM LEAVES: {exc}", file=sys.stderr)
                return 2
            if return_code == 0:
                # The uniform document must appear before its PASS marker.
                # If a competing writer wins either create-only publication,
                # stop rather than leaving a PASS verdict that could be paired
                # with different public artifact bytes.
                _write_new(args.out, document_text)
                _write_new(args.verdict_out, verdict_snapshot.read_text(encoding="utf-8"))
    except UniformLeafWriterError as exc:
        print(f"CANNOT WRITE UNIFORM LEAVES: {exc}", file=sys.stderr)
        return 2
    if return_code == 0:
        print(f"WROTE VALIDATED UNIFORM LEAVES: {args.out}")
    else:
        print(
            f"UNIFORM LEAVES WERE NOT PUBLISHED; THE ESTIMAND GATE EXITED {return_code}: "
            f"private verdict was discarded (requested destination {args.verdict_out})",
            file=sys.stderr,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
