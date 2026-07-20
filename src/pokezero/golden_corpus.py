"""Golden encoder corpus generator (track B of the engine-swap plan).

Plays K local gen3randombattle games and writes, per decision point and per
requested seat, one corpus row carrying:

(a) the GOLDEN production observation exactly as the model consumes it — the
    encoded arrays emitted by the production protocol-stream encoder
    (``LocalShowdownEnv.observe`` -> ``observation_from_player_state``) at
    full precision (numeric features are Python floats, i.e. IEEE float64),
    plus the legal action mask;
(b) the public materialization payload
    (``local_showdown._public_materialization_payload``) verbatim;
(c) both sides' true teams in as-packed-as-capturable form (the generator's
    own ``PokemonSet`` rows from the bridge battle snapshot, plus a packed
    team string built through ``showdown_fixture.pack_team``);
(d) belief-view metadata (carried inside the observation metadata verbatim);
(e) identifiers: battle seed, battle id, decision round index, acting player.

The corpus is the bit-exactness reference that later gates a Rust/engine-side
v2.2 encoder: track B is done when every stored tensor is reproduced
bit-for-bit (docs/test_time_search_plan_v3.md, "The golden corpus").

Capture is strictly additive: rollout/search/env modules are imported, never
modified. Per-decision data comes from a capturing policy wrapper that
implements ``select_action_with_context`` (the documented context-aware hook,
which makes the rollout driver populate ``PolicyContext.observation`` and
``PolicyContext.public_materialization_state``). Per-battle ground truth
comes from one ``LocalShowdownEnv.snapshot()`` taken at the opening request
boundary, before ``rollout.continue_rollout_from_current_state`` drives the
game — the snapshot is an oracle artifact and is stored in the corpus only,
never shown to a policy.

Layout of a corpus directory:

- ``rows.jsonl``     — one header record, then per game one game record
                       followed by its decision records (JSON, sorted keys).
- ``arrays.npz``     — the golden arrays, stacked over all decision rows in
                       row order (``array_row_index`` links a JSONL row to
                       its slice).
- ``fold.jsonl.gz``  — schema v2: the per-row fold surface (fold state at the
                       boundary + inter-decision event slice + annotation
                       overlay + boundary products), gzip JSONL in corpus row
                       order; see :mod:`pokezero.golden_corpus_fold`.
- ``manifest.json``  — schema id + SHA-256 of the files + row counts +
                       array shapes/dtypes.

CLI:

    python -m pokezero.golden_corpus --showdown-root <root> --games K \
        --seed-start N --out <dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterator, Mapping, Sequence

from .actions import ACTION_COUNT
from .golden_corpus_fold import (
    FOLD_RECORD_FIELDS,
    FOLD_ROWS_FILENAME,
    FoldSidecarWriter,
    FoldSurfaceRecorder,
    GoldenFoldRow,
    build_fold_rows,
    fold_record_from_row,
    fold_row_from_record,
    iter_fold_records,
)
from .local_showdown import (
    LocalShowdownConfig,
    LocalShowdownEnv,
    _public_materialization_payload,
)
from .observation import PokeZeroObservationV0
from .policy import PolicyContext, PolicyDecision, RandomLegalPolicy, SimpleLegalPolicy
from .public_decision_corpus import sha256_file
from .rollout import RolloutConfig, continue_rollout_from_current_state
from .showdown import OBSERVATION_SCHEMA_VERSION_V2_2
from .observation import TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
from .showdown_fixture import FixturePokemon, pack_team

GOLDEN_CORPUS_SCHEMA_VERSION = "pokezero.golden-encoder-corpus.v2"

GOLDEN_ROWS_FILENAME = "rows.jsonl"
GOLDEN_ARRAYS_FILENAME = "arrays.npz"
GOLDEN_MANIFEST_FILENAME = "manifest.json"

# Canonical storage order, dtype, and rank for the golden arrays. The per-row
# ``arrays_sha256`` is computed over the concatenation of each field's C-order
# bytes in exactly this order, after casting to exactly these (explicitly
# little-endian) dtypes — so the hash is platform-stable and a future encoder
# can be checked slice-by-slice without loading the JSONL.
GOLDEN_ARRAY_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("categorical_ids", "<i4", 2),
    ("numeric_features", "<f8", 2),
    ("token_type_ids", "<i2", 1),
    ("attention_mask", "|b1", 1),
    ("legal_action_mask", "|b1", 1),
)

GOLDEN_CORPUS_SCHEMA_DESCRIPTION: Mapping[str, Any] = {
    "schema_version": GOLDEN_CORPUS_SCHEMA_VERSION,
    "record_types": ("header", "game", "decision"),
    "decision_fields": (
        "row_sha256",
        "battle_seed",
        "battle_id",
        "format_id",
        "player_id",
        "decision_round_index",
        "requested_players",
        "chosen_action_index",
        "chosen_policy_id",
        "chosen_action_probability",
        "observation",
        "observation_metadata",
        "public_materialization",
    ),
    "game_fields": (
        "battle_seed",
        "battle_id",
        "format_id",
        "policy_ids",
        "true_teams",
        "terminal",
        "decision_row_count",
    ),
    "arrays": {name: dtype for name, dtype, _ in GOLDEN_ARRAY_FIELDS},
    "array_hash": (
        "sha256 over the row's array slices concatenated in GOLDEN_ARRAY_FIELDS "
        "order, C-order little-endian bytes"
    ),
    # Schema v2: the fold surface. One gzip JSONL sidecar (fold.jsonl.gz,
    # mtime=0) holding a fold_header record then one fold record per decision
    # row, in corpus row order. Each fold record carries the serialized
    # incremental fold state at the decision boundary
    # (pokezero.transitions_fold.FoldState.to_payload, production-default tail
    # limits), the |t:|-filtered inter-decision event slice since the previous
    # SAME-SEAT decision boundary, the Tier-2 annotation overlay applied at the
    # boundary, and the boundary products — the row-pair validation surface for
    # candidate advance() implementations (scripts/validate_corpus_v2.py). The
    # sidecar links to rows.jsonl by array_row_index + row_sha256 and is
    # optional at the writer level (synthetic corpora may omit it); the
    # reference corpus always carries it.
    "fold_record_types": ("fold_header", "fold"),
    "fold_fields": FOLD_RECORD_FIELDS,
    "fold_file": FOLD_ROWS_FILENAME,
}


def _canonical_json_bytes(payload: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, no whitespace, ASCII, NaN refused."""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def golden_canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


GOLDEN_CORPUS_SCHEMA_SHA256 = golden_canonical_sha256(GOLDEN_CORPUS_SCHEMA_DESCRIPTION)


def _require_numpy() -> Any:
    try:
        import numpy
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "NumPy is required for the golden corpus. Install with `pip install -e .[neural]`."
        ) from exc
    return numpy


def _json_safe(value: Any, *, context: str) -> Any:
    """Deep-normalize to plain JSON values; loud on anything non-JSON-safe.

    The corpus stores metadata and materialization payloads VERBATIM: tuples
    become lists (a JSON representation change only), but no value is coerced
    or stringified. A non-JSON leaf is a schema gap that must surface here,
    not a silently mangled golden record.
    """

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context}: non-string mapping key {key!r}.")
            normalized[key] = _json_safe(item, context=f"{context}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, context=f"{context}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{context}: non-finite float {value!r}.")
        return value
    raise TypeError(f"{context}: non-JSON-safe value of type {type(value).__name__}.")


# ---------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenObservationArrays:
    """The golden encoded arrays for one decision, in canonical dtypes."""

    categorical_ids: Any
    numeric_features: Any
    token_type_ids: Any
    attention_mask: Any
    legal_action_mask: Any

    @classmethod
    def from_observation(cls, observation: PokeZeroObservationV0) -> "GoldenObservationArrays":
        numpy = _require_numpy()
        categorical = numpy.asarray(observation.categorical_ids)
        numeric = numpy.asarray(observation.numeric_features, dtype="<f8")
        token_type = numpy.asarray(observation.token_type_ids)
        attention = numpy.asarray(observation.attention_mask, dtype="|b1")
        legal = numpy.asarray(observation.legal_action_mask, dtype="|b1")
        if categorical.ndim != 2 or numeric.ndim != 2:
            raise ValueError("golden observation arrays must be 2-D token tables.")
        if token_type.ndim != 1 or attention.ndim != 1 or legal.ndim != 1:
            raise ValueError("golden observation vectors must be 1-D.")
        if not (categorical.shape[0] == numeric.shape[0] == token_type.shape[0] == attention.shape[0]):
            raise ValueError("golden observation arrays disagree on token count.")
        if legal.shape[0] != ACTION_COUNT:
            raise ValueError(f"legal_action_mask must contain {ACTION_COUNT} values.")
        int32 = numpy.iinfo("<i4")
        if categorical.size and (categorical.min() < int32.min or categorical.max() > int32.max):
            raise ValueError("categorical ids exceed the corpus int32 range.")
        int16 = numpy.iinfo("<i2")
        if token_type.size and (token_type.min() < int16.min or token_type.max() > int16.max):
            raise ValueError("token type ids exceed the corpus int16 range.")
        return cls(
            categorical_ids=categorical.astype("<i4"),
            numeric_features=numeric,
            token_type_ids=token_type.astype("<i2"),
            attention_mask=attention,
            legal_action_mask=legal,
        )

    def field_arrays(self) -> tuple[tuple[str, Any], ...]:
        numpy = _require_numpy()
        out = []
        for name, dtype, rank in GOLDEN_ARRAY_FIELDS:
            array = numpy.ascontiguousarray(getattr(self, name), dtype=dtype)
            if array.ndim != rank:
                raise ValueError(f"{name} must have rank {rank}, got {array.ndim}.")
            out.append((name, array))
        return tuple(out)

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for _, array in self.field_arrays():
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True)
class GoldenDecisionRow:
    """One decision point from one seat: golden arrays + verbatim context."""

    battle_seed: int
    battle_id: str
    format_id: str
    player_id: str
    decision_round_index: int
    requested_players: tuple[str, ...]
    observation_schema_version: str
    perspective: Mapping[str, Any] | None
    observation_metadata: Mapping[str, Any]
    public_materialization: Mapping[str, Any]
    chosen_action_index: int
    chosen_policy_id: str
    chosen_action_probability: float | None
    arrays: GoldenObservationArrays

    def __post_init__(self) -> None:
        if self.player_id not in {"p1", "p2"}:
            raise ValueError("player_id must be p1 or p2.")
        if self.decision_round_index < 0:
            raise ValueError("decision_round_index must be non-negative.")
        if not 0 <= self.chosen_action_index < ACTION_COUNT:
            raise ValueError(f"chosen_action_index must be between 0 and {ACTION_COUNT - 1}.")
        if not bool(self.arrays.legal_action_mask[self.chosen_action_index]):
            raise ValueError("chosen_action_index must be legal in the golden legal mask.")

    def to_json_dict(self, *, array_row_index: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "record_type": "decision",
            "schema_version": GOLDEN_CORPUS_SCHEMA_VERSION,
            "battle_seed": self.battle_seed,
            "battle_id": self.battle_id,
            "format_id": self.format_id,
            "player_id": self.player_id,
            "decision_round_index": self.decision_round_index,
            "requested_players": list(self.requested_players),
            "chosen_action_index": self.chosen_action_index,
            "chosen_policy_id": self.chosen_policy_id,
            "chosen_action_probability": self.chosen_action_probability,
            "observation": {
                "schema_version": self.observation_schema_version,
                "perspective": dict(self.perspective) if self.perspective is not None else None,
                "array_row_index": array_row_index,
                "arrays_sha256": self.arrays.sha256(),
                "legal_action_mask": [bool(value) for value in self.arrays.legal_action_mask],
            },
            "observation_metadata": _json_safe(self.observation_metadata, context="observation_metadata"),
            "public_materialization": _json_safe(self.public_materialization, context="public_materialization"),
        }
        payload["row_sha256"] = golden_canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class GoldenGameRecord:
    """Per-battle ground truth: identifiers, policies, true teams, outcome."""

    battle_seed: int
    battle_id: str
    format_id: str
    policy_ids: Mapping[str, str]
    true_teams: Mapping[str, Any]
    terminal: Mapping[str, Any]

    def to_json_dict(self, *, decision_row_count: int) -> dict[str, Any]:
        return {
            "record_type": "game",
            "schema_version": GOLDEN_CORPUS_SCHEMA_VERSION,
            "battle_seed": self.battle_seed,
            "battle_id": self.battle_id,
            "format_id": self.format_id,
            "policy_ids": _json_safe(self.policy_ids, context="policy_ids"),
            "true_teams": _json_safe(self.true_teams, context="true_teams"),
            "terminal": _json_safe(self.terminal, context="terminal"),
            "decision_row_count": decision_row_count,
        }


@dataclass(frozen=True)
class GoldenGame:
    record: GoldenGameRecord
    rows: tuple[GoldenDecisionRow, ...]
    # Schema-v2 fold surface, parallel to ``rows`` (one fold row per decision
    # row) when present. Optional at the writer level so synthetic/array-only
    # corpora stay expressible; the reference corpus always carries it.
    fold_rows: tuple[GoldenFoldRow, ...] = ()

    def __post_init__(self) -> None:
        if self.fold_rows and len(self.fold_rows) != len(self.rows):
            raise ValueError(
                f"game {self.record.battle_id!r} carries {len(self.fold_rows)} fold rows "
                f"for {len(self.rows)} decision rows; the fold surface must be parallel."
            )


@dataclass(frozen=True)
class GoldenCorpus:
    header: Mapping[str, Any]
    manifest: Mapping[str, Any]
    games: tuple[GoldenGame, ...]

    @property
    def decision_rows(self) -> tuple[GoldenDecisionRow, ...]:
        return tuple(row for game in self.games for row in game.rows)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_golden_corpus(
    out_dir: Path,
    *,
    header: Mapping[str, Any],
    games: Sequence[GoldenGame],
) -> dict[str, Any]:
    """Write rows.jsonl + arrays.npz + manifest.json; refuse to overwrite."""

    numpy = _require_numpy()
    if not games:
        raise ValueError("golden corpus requires at least one game.")
    if not any(game.rows for game in games):
        raise ValueError("golden corpus requires at least one decision row.")
    for key in ("record_type", "schema_version", "schema_sha256"):
        if key in header:
            raise ValueError(f"header must not pre-set writer-owned field {key!r}.")
    fold_flags = {bool(game.fold_rows) for game in games if game.rows}
    if fold_flags == {True, False}:
        raise ValueError(
            "either every game carries the fold surface or none does; a mixed corpus "
            "would break per-row fold chain validation."
        )
    fold_present = fold_flags == {True}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / GOLDEN_ROWS_FILENAME
    arrays_path = out_dir / GOLDEN_ARRAYS_FILENAME
    manifest_path = out_dir / GOLDEN_MANIFEST_FILENAME
    fold_path = out_dir / FOLD_ROWS_FILENAME
    checked_paths = [rows_path, arrays_path, manifest_path]
    if fold_present:
        checked_paths.append(fold_path)
    for path in checked_paths:
        if path.exists():
            raise FileExistsError(f"golden corpus file already exists: {path}")

    header_record = {
        "record_type": "header",
        "schema_version": GOLDEN_CORPUS_SCHEMA_VERSION,
        "schema_sha256": GOLDEN_CORPUS_SCHEMA_SHA256,
        **_json_safe(header, context="header"),
    }

    stacked: dict[str, list[Any]] = {name: [] for name, _, _ in GOLDEN_ARRAY_FIELDS}
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    array_row_index = 0
    fold_writer: FoldSidecarWriter | None = None
    try:
        if fold_present:
            fold_writer = FoldSidecarWriter(fold_path, canonical_json_bytes=_canonical_json_bytes)
            fold_writer.write_header(
                {
                    "record_type": "fold_header",
                    "schema_version": GOLDEN_CORPUS_SCHEMA_VERSION,
                    "schema_sha256": GOLDEN_CORPUS_SCHEMA_SHA256,
                    "fold_fields": list(FOLD_RECORD_FIELDS),
                }
            )
        with rows_path.open("x", encoding="utf-8") as handle:

            def _write_line(payload: Mapping[str, Any]) -> None:
                handle.write(_canonical_json_bytes(payload).decode("utf-8"))
                handle.write("\n")

            _write_line(header_record)
            for game in games:
                _write_line(game.record.to_json_dict(decision_row_count=len(game.rows)))
                for row_index, row in enumerate(game.rows):
                    shapes = {name: array.shape for name, array in row.arrays.field_arrays()}
                    if reference_shapes is None:
                        reference_shapes = shapes
                    elif shapes != reference_shapes:
                        raise ValueError(
                            f"decision row {array_row_index} array shapes {shapes} do not match "
                            f"the corpus shapes {reference_shapes}."
                        )
                    for name, array in row.arrays.field_arrays():
                        stacked[name].append(array)
                    row_payload = row.to_json_dict(array_row_index=array_row_index)
                    _write_line(row_payload)
                    if fold_writer is not None:
                        fold_writer.write_record(
                            fold_record_from_row(
                                game.fold_rows[row_index],
                                schema_version=GOLDEN_CORPUS_SCHEMA_VERSION,
                                battle_seed=row.battle_seed,
                                battle_id=row.battle_id,
                                format_id=row.format_id,
                                player_id=row.player_id,
                                decision_round_index=row.decision_round_index,
                                array_row_index=array_row_index,
                                row_sha256=row_payload["row_sha256"],
                                canonical_sha256=golden_canonical_sha256,
                            )
                        )
                    array_row_index += 1
    finally:
        if fold_writer is not None:
            fold_writer.close()

    assert reference_shapes is not None
    arrays = {
        name: numpy.stack(stacked[name], axis=0).astype(dtype)
        for name, dtype, _ in GOLDEN_ARRAY_FIELDS
    }
    with arrays_path.open("xb") as handle:
        numpy.savez_compressed(handle, **arrays)

    counts: dict[str, int] = {"games": len(games), "decisions": array_row_index}
    files: dict[str, Any] = {
        GOLDEN_ROWS_FILENAME: {
            "sha256": sha256_file(rows_path),
            "bytes": rows_path.stat().st_size,
        },
        GOLDEN_ARRAYS_FILENAME: {
            "sha256": sha256_file(arrays_path),
            "bytes": arrays_path.stat().st_size,
        },
    }
    if fold_writer is not None:
        counts["fold_rows"] = fold_writer.record_count
        files[FOLD_ROWS_FILENAME] = {
            "sha256": sha256_file(fold_path),
            "bytes": fold_path.stat().st_size,
            "uncompressed_bytes": fold_writer.uncompressed_bytes,
        }
    manifest = {
        "schema_version": GOLDEN_CORPUS_SCHEMA_VERSION,
        "schema_sha256": GOLDEN_CORPUS_SCHEMA_SHA256,
        "counts": counts,
        "array_dtypes": {name: dtype for name, dtype, _ in GOLDEN_ARRAY_FIELDS},
        "array_shapes": {name: list(shape) for name, shape in reference_shapes.items()},
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


# ---------------------------------------------------------------------------
# Reader + verification
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid golden corpus JSON at line {line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"golden corpus line {line_number} must be a JSON object.")
            yield line_number, payload


def _load_manifest(corpus_dir: Path) -> Mapping[str, Any]:
    manifest_path = Path(corpus_dir) / GOLDEN_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("golden corpus manifest must be a JSON object.")
    if manifest.get("schema_version") != GOLDEN_CORPUS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported golden corpus schema: {manifest.get('schema_version')!r} "
            f"(expected {GOLDEN_CORPUS_SCHEMA_VERSION!r})."
        )
    if manifest.get("schema_sha256") != GOLDEN_CORPUS_SCHEMA_SHA256:
        raise ValueError("golden corpus schema hash does not match this reader's schema.")
    return manifest


def _load_arrays(corpus_dir: Path, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    numpy = _require_numpy()
    arrays_path = Path(corpus_dir) / GOLDEN_ARRAYS_FILENAME
    with numpy.load(arrays_path) as bundle:
        arrays = {name: bundle[name] for name, _, _ in GOLDEN_ARRAY_FIELDS}
    decisions = int(manifest["counts"]["decisions"])
    for name, dtype, _ in GOLDEN_ARRAY_FIELDS:
        array = arrays[name]
        if array.dtype != numpy.dtype(dtype):
            raise ValueError(f"arrays.npz field {name} has dtype {array.dtype}, expected {dtype}.")
        expected_shape = (decisions, *manifest["array_shapes"][name])
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"arrays.npz field {name} has shape {tuple(array.shape)}, expected {expected_shape}."
            )
    return arrays


def _row_from_json(
    payload: Mapping[str, Any],
    *,
    arrays: Mapping[str, Any],
    expected_array_row_index: int,
) -> GoldenDecisionRow:
    observation = payload.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("decision record is missing its observation block.")
    array_row_index = int(observation["array_row_index"])
    if array_row_index != expected_array_row_index:
        raise ValueError(
            f"decision rows out of order: array_row_index {array_row_index}, "
            f"expected {expected_array_row_index}."
        )
    row_arrays = GoldenObservationArrays(
        **{name: arrays[name][array_row_index] for name, _, _ in GOLDEN_ARRAY_FIELDS}
    )
    if row_arrays.sha256() != observation.get("arrays_sha256"):
        raise ValueError(f"decision row {array_row_index}: arrays_sha256 does not match arrays.npz.")
    inline_mask = [bool(value) for value in observation.get("legal_action_mask", ())]
    if inline_mask != [bool(value) for value in row_arrays.legal_action_mask]:
        raise ValueError(f"decision row {array_row_index}: inline legal mask disagrees with arrays.npz.")
    row = GoldenDecisionRow(
        battle_seed=int(payload["battle_seed"]),
        battle_id=str(payload["battle_id"]),
        format_id=str(payload["format_id"]),
        player_id=str(payload["player_id"]),
        decision_round_index=int(payload["decision_round_index"]),
        requested_players=tuple(str(player) for player in payload["requested_players"]),
        observation_schema_version=str(observation["schema_version"]),
        perspective=observation.get("perspective"),
        observation_metadata=payload["observation_metadata"],
        public_materialization=payload["public_materialization"],
        chosen_action_index=int(payload["chosen_action_index"]),
        chosen_policy_id=str(payload["chosen_policy_id"]),
        chosen_action_probability=(
            float(payload["chosen_action_probability"])
            if payload.get("chosen_action_probability") is not None
            else None
        ),
        arrays=row_arrays,
    )
    rebuilt = row.to_json_dict(array_row_index=array_row_index)
    if rebuilt["row_sha256"] != payload.get("row_sha256"):
        raise ValueError(f"decision row {array_row_index}: row_sha256 does not match its payload.")
    return row


def load_golden_corpus(corpus_dir: Path) -> GoldenCorpus:
    """Read and structurally validate a golden corpus (row hashes included).

    The fold sidecar is deliberately NOT loaded here (late-game fold payloads
    are ~226 KB each; loading them all would multiply resident memory by an
    order of magnitude). Stream it with
    :func:`pokezero.golden_corpus_fold.iter_fold_records`;
    :func:`verify_golden_corpus` checks its integrity and links.
    """

    corpus_dir = Path(corpus_dir)
    manifest = _load_manifest(corpus_dir)
    arrays = _load_arrays(corpus_dir, manifest)

    header: Mapping[str, Any] | None = None
    games: list[GoldenGame] = []
    pending_record: GoldenGameRecord | None = None
    pending_expected = 0
    pending_rows: list[GoldenDecisionRow] = []
    next_array_row_index = 0

    def _finish_game() -> None:
        nonlocal pending_record, pending_rows
        if pending_record is None:
            return
        if len(pending_rows) != pending_expected:
            raise ValueError(
                f"game {pending_record.battle_id!r} declares {pending_expected} decision rows "
                f"but carries {len(pending_rows)}."
            )
        games.append(GoldenGame(record=pending_record, rows=tuple(pending_rows)))
        pending_record = None
        pending_rows = []

    for line_number, payload in _iter_jsonl(corpus_dir / GOLDEN_ROWS_FILENAME):
        record_type = payload.get("record_type")
        if record_type == "header":
            if header is not None or games or pending_record is not None:
                raise ValueError("golden corpus header must be the first record.")
            if payload.get("schema_version") != GOLDEN_CORPUS_SCHEMA_VERSION:
                raise ValueError("golden corpus header schema_version mismatch.")
            if payload.get("schema_sha256") != GOLDEN_CORPUS_SCHEMA_SHA256:
                raise ValueError("golden corpus header schema hash mismatch.")
            header = payload
            continue
        if header is None:
            raise ValueError("golden corpus is missing its header record.")
        if record_type == "game":
            _finish_game()
            pending_record = GoldenGameRecord(
                battle_seed=int(payload["battle_seed"]),
                battle_id=str(payload["battle_id"]),
                format_id=str(payload["format_id"]),
                policy_ids=payload["policy_ids"],
                true_teams=payload["true_teams"],
                terminal=payload["terminal"],
            )
            pending_expected = int(payload["decision_row_count"])
            continue
        if record_type == "decision":
            if pending_record is None:
                raise ValueError(f"decision record at line {line_number} precedes any game record.")
            pending_rows.append(
                _row_from_json(payload, arrays=arrays, expected_array_row_index=next_array_row_index)
            )
            next_array_row_index += 1
            continue
        raise ValueError(f"unsupported golden corpus record_type {record_type!r} at line {line_number}.")

    _finish_game()
    if header is None:
        raise ValueError("golden corpus is empty or missing its header.")
    if len(games) != int(manifest["counts"]["games"]):
        raise ValueError("golden corpus game count does not match its manifest.")
    if next_array_row_index != int(manifest["counts"]["decisions"]):
        raise ValueError("golden corpus decision count does not match its manifest.")
    return GoldenCorpus(header=header, manifest=manifest, games=tuple(games))


@dataclass(frozen=True)
class GoldenCorpusVerification:
    games: int
    decisions: int
    array_shapes: Mapping[str, tuple[int, ...]]
    rows_sha256: str
    arrays_sha256: str
    fold_rows: int = 0


def _verify_fold_sidecar(corpus_dir: Path, corpus: GoldenCorpus, manifest: Mapping[str, Any]) -> int:
    """Stream-verify the fold sidecar: links, hashes, chain contiguity."""

    decision_rows = corpus.decision_rows
    expected_index = 0
    chain_expect: dict[tuple[str, str], int] = {}
    for record in iter_fold_records(corpus_dir, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION):
        if expected_index >= len(decision_rows):
            raise ValueError("fold sidecar carries more records than the corpus has decision rows.")
        row = decision_rows[expected_index]
        if int(record["array_row_index"]) != expected_index:
            raise ValueError(
                f"fold record #{expected_index} carries array_row_index "
                f"{record['array_row_index']}; fold records must follow corpus row order."
            )
        rebuilt = row.to_json_dict(array_row_index=expected_index)
        if record["row_sha256"] != rebuilt["row_sha256"]:
            raise ValueError(f"fold record #{expected_index}: row_sha256 does not match its decision row.")
        for key, want in (
            ("battle_seed", row.battle_seed),
            ("battle_id", row.battle_id),
            ("format_id", row.format_id),
            ("player_id", row.player_id),
            ("decision_round_index", row.decision_round_index),
        ):
            if record[key] != want:
                raise ValueError(
                    f"fold record #{expected_index}: {key} {record[key]!r} does not match "
                    f"its decision row ({want!r})."
                )
        for payload_key, hash_key in (("fold_state", "fold_state_sha256"), ("products", "products_sha256")):
            if golden_canonical_sha256(record[payload_key]) != record[hash_key]:
                raise ValueError(f"fold record #{expected_index}: {hash_key} does not match its payload.")
        if record["fold_state"].get("perspective_slot") != row.player_id:
            raise ValueError(
                f"fold record #{expected_index}: fold state perspective does not match its seat."
            )
        for line in record["event_slice"]:
            parts = str(line).split("|")
            if (parts[1] if len(parts) > 1 else "") == "t:":
                raise ValueError(
                    f"fold record #{expected_index}: event_slice contains a |t:| wall-clock line."
                )
        chain_key = (str(record["battle_id"]), str(record["player_id"]))
        expected_chain = chain_expect.get(chain_key, 0)
        if int(record["chain_index"]) != expected_chain:
            raise ValueError(
                f"fold record #{expected_index}: chain_index {record['chain_index']} breaks the "
                f"{chain_key} chain (expected {expected_chain})."
            )
        chain_expect[chain_key] = expected_chain + 1
        expected_index += 1
    if expected_index != len(decision_rows):
        raise ValueError(
            f"fold sidecar carries {expected_index} records for {len(decision_rows)} decision rows."
        )
    if expected_index != int(manifest["counts"].get("fold_rows", -1)):
        raise ValueError("fold sidecar record count does not match its manifest.")
    return expected_index


def verify_golden_corpus(corpus_dir: Path) -> GoldenCorpusVerification:
    """Full verification: manifest file hashes + every row/array checksum, plus
    the fold sidecar's links, payload hashes, and chain contiguity when present."""

    corpus_dir = Path(corpus_dir)
    manifest = _load_manifest(corpus_dir)
    fold_listed = FOLD_ROWS_FILENAME in manifest["files"]
    fold_path = corpus_dir / FOLD_ROWS_FILENAME
    if fold_listed != fold_path.exists():
        raise ValueError(
            f"fold sidecar presence mismatch: manifest lists it: {fold_listed}, "
            f"file exists: {fold_path.exists()}."
        )
    filenames = [GOLDEN_ROWS_FILENAME, GOLDEN_ARRAYS_FILENAME]
    if fold_listed:
        filenames.append(FOLD_ROWS_FILENAME)
    for filename in filenames:
        entry = manifest["files"][filename]
        path = corpus_dir / filename
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"golden corpus file {filename} hash mismatch: manifest {entry['sha256']}, "
                f"actual {actual}."
            )
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"golden corpus file {filename} size does not match its manifest.")
    corpus = load_golden_corpus(corpus_dir)
    fold_rows = _verify_fold_sidecar(corpus_dir, corpus, manifest) if fold_listed else 0
    return GoldenCorpusVerification(
        games=len(corpus.games),
        decisions=len(corpus.decision_rows),
        array_shapes={name: tuple(shape) for name, shape in manifest["array_shapes"].items()},
        rows_sha256=manifest["files"][GOLDEN_ROWS_FILENAME]["sha256"],
        arrays_sha256=manifest["files"][GOLDEN_ARRAYS_FILENAME]["sha256"],
        fold_rows=fold_rows,
    )


def sample_golden_corpus(src_dir: Path, dst_dir: Path, *, max_decisions: int = 5) -> dict[str, Any]:
    """Write a small, fully valid corpus holding the first rows of ``src_dir``.

    Used to commit a tiny regression fixture: the sample keeps original row
    hashes (rows are unchanged) while its own manifest re-describes the
    truncated files. The header records the source rows-file hash.
    """

    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive.")
    corpus = load_golden_corpus(Path(src_dir))
    first_game = corpus.games[0]
    if len(first_game.rows) < max_decisions:
        raise ValueError(
            f"first game has only {len(first_game.rows)} decision rows; cannot sample {max_decisions}."
        )
    header = {
        key: value
        for key, value in corpus.header.items()
        if key not in {"record_type", "schema_version", "schema_sha256"}
    }
    header["sampled_from"] = {
        "rows_sha256": corpus.manifest["files"][GOLDEN_ROWS_FILENAME]["sha256"],
        "games": int(corpus.manifest["counts"]["games"]),
        "decisions": int(corpus.manifest["counts"]["decisions"]),
    }
    fold_rows: tuple[GoldenFoldRow, ...] = ()
    if FOLD_ROWS_FILENAME in corpus.manifest["files"]:
        # Carry the sampled rows' fold surface (sidecar records are in corpus
        # row order; the first game's rows are the first records). Link fields
        # are recomputed by the writer; row hashes are unchanged, so the links
        # stay identical to the source corpus.
        sampled_records = []
        for record in iter_fold_records(
            Path(src_dir), expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
        ):
            if int(record["array_row_index"]) >= max_decisions:
                break
            sampled_records.append(fold_row_from_record(record))
        fold_rows = tuple(sampled_records)
    sampled = GoldenGame(
        record=first_game.record, rows=first_game.rows[:max_decisions], fold_rows=fold_rows
    )
    return write_golden_corpus(Path(dst_dir), header=header, games=[sampled])


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class _CapturingPolicy:
    """Context-aware wrapper: records golden capture, delegates the decision.

    Implementing ``select_action_with_context`` makes the (unmodified) rollout
    driver hand this policy the acting seat's observation AND its
    ``public_materialization_state``. The wrapper consumes no RNG draws and
    returns the inner decision unchanged, so wrapped games are bit-identical
    to unwrapped ones.
    """

    inner: Any
    sink: Callable[[PolicyContext, PolicyDecision], None]
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.policy_id = f"golden-capture[{self.inner.policy_id}]"

    def reset(self) -> None:
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()

    def select_action(self, observation: PokeZeroObservationV0, *, rng: random.Random) -> PolicyDecision:
        return self.inner.select_action(observation, rng=rng)

    def select_action_with_context(self, context: PolicyContext, *, rng: random.Random) -> PolicyDecision:
        contextual = getattr(self.inner, "select_action_with_context", None)
        if callable(contextual):
            decision = contextual(context, rng=rng)
        else:
            decision = self.inner.select_action(context.observation, rng=rng)
        self.sink(context, decision)
        return decision


def _decision_row_from_context(
    context: PolicyContext,
    decision: PolicyDecision,
    *,
    battle_seed: int,
) -> GoldenDecisionRow:
    state = context.public_materialization_state
    if state is None:
        raise ValueError(
            "rollout did not populate public_materialization_state; the capturing policy "
            "must be seen as context-aware by the driver."
        )
    observation = context.observation
    perspective = None
    if observation.perspective is not None:
        perspective = {
            "player_id": observation.perspective.player_id,
            "showdown_slot": observation.perspective.showdown_slot,
            "opponent_showdown_slot": observation.perspective.opponent_showdown_slot,
        }
    return GoldenDecisionRow(
        battle_seed=battle_seed,
        battle_id=context.battle_id,
        format_id=context.format_id,
        player_id=context.player_id,
        decision_round_index=context.decision_round_index,
        requested_players=tuple(context.requested_players),
        observation_schema_version=observation.schema_version,
        perspective=perspective,
        observation_metadata=_json_safe(dict(observation.metadata), context="observation_metadata"),
        public_materialization=_json_safe(
            _public_materialization_payload(state), context="public_materialization"
        ),
        chosen_action_index=decision.action_index,
        chosen_policy_id=decision.policy_id,
        chosen_action_probability=decision.action_probability,
        arrays=GoldenObservationArrays.from_observation(observation),
    )


def _fixture_from_generator_set(pokemon_set: Mapping[str, Any], *, battle_gender: str | None) -> FixturePokemon:
    species = str(pokemon_set.get("species") or pokemon_set.get("name") or "")
    moves = tuple(str(move) for move in pokemon_set.get("moves") or ())
    evs = pokemon_set.get("evs")
    ivs = pokemon_set.get("ivs")
    # The generator leaves gender unset (""); the battle rolls it at start. The
    # packed string pins the battle-actual gender so the packed team replays the
    # same mons; the verbatim `set` payload preserves the generator's own view.
    gender = battle_gender or str(pokemon_set.get("gender") or "") or None
    return FixturePokemon(
        species=species,
        moves=moves,
        ability=str(pokemon_set.get("ability")) if pokemon_set.get("ability") else None,
        item=str(pokemon_set.get("item")) if pokemon_set.get("item") else None,
        level=int(pokemon_set.get("level") or 100),
        nature=str(pokemon_set.get("nature") or ""),
        gender=gender,
        evs={str(k): int(v) for k, v in evs.items()} if isinstance(evs, Mapping) else None,
        ivs={str(k): int(v) for k, v in ivs.items()} if isinstance(ivs, Mapping) else None,
    )


def _true_teams_from_bridge_snapshot(bridge_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Both sides' generator sets (EVs/IVs included) from the opening snapshot."""

    battle = bridge_snapshot.get("battle")
    if not isinstance(battle, Mapping):
        raise ValueError("bridge snapshot is missing its battle payload.")
    sides = battle.get("sides")
    if not isinstance(sides, Sequence) or not sides:
        raise ValueError("bridge snapshot is missing side payloads.")
    teams: dict[str, Any] = {}
    for side in sides:
        if not isinstance(side, Mapping):
            raise ValueError("bridge snapshot side payload must be a mapping.")
        side_id = str(side.get("id") or "")
        if side_id not in {"p1", "p2"}:
            raise ValueError(f"bridge snapshot side id must be p1 or p2; got {side_id!r}.")
        entries: list[dict[str, Any]] = []
        fixtures: list[FixturePokemon] = []
        for mon in side.get("pokemon") or ():
            if not isinstance(mon, Mapping):
                raise ValueError("bridge snapshot pokemon payload must be a mapping.")
            pokemon_set = mon.get("set")
            if not isinstance(pokemon_set, Mapping):
                raise ValueError("bridge snapshot pokemon is missing its generator set.")
            battle_gender = str(mon.get("gender") or "") or None
            entries.append(
                {
                    "set": _json_safe(pokemon_set, context="true_teams.set"),
                    "battle_gender": battle_gender,
                    "details": str(mon.get("details") or "") or None,
                }
            )
            fixtures.append(_fixture_from_generator_set(pokemon_set, battle_gender=battle_gender))
        if not entries:
            raise ValueError(f"bridge snapshot side {side_id} has no pokemon.")
        teams[side_id] = {
            "source": "bridge-snapshot-generator-set",
            "pokemon": entries,
            "packed": pack_team(fixtures),
        }
    if set(teams) != {"p1", "p2"}:
        raise ValueError("bridge snapshot must carry both p1 and p2 sides.")
    return teams


# Deterministic per-game seat policy rotation for v1 corpus diversity.
GOLDEN_POLICY_ROTATION: tuple[tuple[str, str], ...] = (
    ("simple", "simple"),
    ("simple", "random"),
    ("random", "simple"),
)


def _base_policy(name: str) -> Any:
    if name == "simple":
        return SimpleLegalPolicy()
    if name == "random":
        return RandomLegalPolicy()
    raise ValueError(f"unknown golden corpus base policy {name!r}.")


def generate_golden_corpus(
    *,
    out_dir: Path,
    games: int,
    seed_start: int,
    showdown_root: Path | str | None = None,
    format_id: str = "gen3randombattle",
    max_decision_rounds: int = 250,
    belief_set_source: bool | None = None,
) -> dict[str, Any]:
    """Play ``games`` local games and write the golden corpus into ``out_dir``."""

    if games <= 0:
        raise ValueError("games must be positive.")
    config = LocalShowdownConfig(showdown_root=showdown_root, set_belief_source=belief_set_source)
    env = LocalShowdownEnv(config)
    turn_merged_active = (
        config.observation_spec.schema_version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
    )
    collected: list[GoldenGame] = []
    belief_hash: str | None = None
    try:
        belief_hash = env.belief_set_source_hash
        for game_index in range(games):
            seed = seed_start + game_index
            battle_id = f"golden-{format_id}-{seed}"
            env.reset(seed=seed, format_id=format_id)
            # Oracle snapshot at the opening boundary: the only reader of both
            # sides' generator-internal sets. Stored in the corpus, never fed
            # to a policy.
            true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
            captures: list[tuple[PolicyContext, PolicyDecision]] = []
            recorder = FoldSurfaceRecorder(env)

            def _sink(context: PolicyContext, decision: PolicyDecision) -> None:
                captures.append((context, decision))
                recorder.record(context.player_id)

            rotation = GOLDEN_POLICY_ROTATION[game_index % len(GOLDEN_POLICY_ROTATION)]
            policies = {
                "p1": _CapturingPolicy(_base_policy(rotation[0]), _sink),
                "p2": _CapturingPolicy(_base_policy(rotation[1]), _sink),
            }
            result = continue_rollout_from_current_state(
                env=env,
                policies=policies,
                config=RolloutConfig(
                    max_decision_rounds=max_decision_rounds,
                    format_id=format_id,
                    hide_opponent_legal_action_masks=True,
                ),
                seed=seed,
                battle_id=battle_id,
                reset_policies=True,
            )
            rows = tuple(
                _decision_row_from_context(context, decision, battle_seed=seed)
                for context, decision in captures
            )
            fold_rows = build_fold_rows(
                replays=[context.public_materialization_state.replay for context, _ in captures],
                surfaces=recorder.surfaces,
                turn_merged_active=turn_merged_active,
            )
            record = GoldenGameRecord(
                battle_seed=seed,
                battle_id=battle_id,
                format_id=format_id,
                policy_ids={"p1": policies["p1"].policy_id, "p2": policies["p2"].policy_id},
                true_teams=true_teams,
                terminal={
                    "winner": result.terminal.winner,
                    "turn_count": result.terminal.turn_count,
                    "capped": result.terminal.capped,
                },
            )
            collected.append(GoldenGame(record=record, rows=rows, fold_rows=fold_rows))
    finally:
        env.close()

    spec = config.observation_spec
    masks = config.feature_masks
    header = {
        "generator": {
            "games": games,
            "seed_start": seed_start,
            "format_id": format_id,
            "max_decision_rounds": max_decision_rounds,
            "policy_rotation": [list(pair) for pair in GOLDEN_POLICY_ROTATION],
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
        "belief_set_source": {
            "enabled": config.belief_set_source_enabled(),
            "source_hash": belief_hash,
        },
    }
    return write_golden_corpus(Path(out_dir), header=header, games=collected)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pokezero.golden_corpus",
        description="Generate the golden encoder corpus from local gen3randombattle games.",
    )
    parser.add_argument(
        "--showdown-root",
        type=Path,
        default=None,
        help="Built Pokemon Showdown checkout (default: POKEZERO_SHOWDOWN_ROOT or the repo default).",
    )
    parser.add_argument("--games", type=int, required=True, help="Number of games to play.")
    parser.add_argument("--seed-start", type=int, default=0, help="First battle seed (seeds are contiguous).")
    parser.add_argument("--out", type=Path, required=True, help="Output corpus directory.")
    parser.add_argument("--format-id", default="gen3randombattle")
    parser.add_argument("--max-decision-rounds", type=int, default=250)
    parser.add_argument(
        "--belief-set-source",
        choices=("env", "on", "off"),
        default="env",
        help="Candidate-set belief source: pin on/off, or defer to POKEZERO_BELIEF_SET_SOURCE (default).",
    )
    args = parser.parse_args(argv)
    belief_set_source = {"env": None, "on": True, "off": False}[args.belief_set_source]
    manifest = generate_golden_corpus(
        out_dir=args.out,
        games=args.games,
        seed_start=args.seed_start,
        showdown_root=args.showdown_root,
        format_id=args.format_id,
        max_decision_rounds=args.max_decision_rounds,
        belief_set_source=belief_set_source,
    )
    verification = verify_golden_corpus(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "games": verification.games,
                "decisions": verification.decisions,
                "fold_rows": verification.fold_rows,
                "array_shapes": {name: list(shape) for name, shape in verification.array_shapes.items()},
                "rows_sha256": verification.rows_sha256,
                "arrays_sha256": verification.arrays_sha256,
                "manifest_sha256": golden_canonical_sha256(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
