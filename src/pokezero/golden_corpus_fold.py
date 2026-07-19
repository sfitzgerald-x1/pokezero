"""Schema-v2 fold surface of the golden corpus (track B of the engine-swap plan).

Corpus schema v2 (docs/test_time_search_plan_v3.md, "Schema v2") makes the
fold-state ADVANCE — ``advance(fold_state, events) -> (fold_state', products')``,
exactly the operation search executes and the validation target for the Rust
``advance()`` port — checkable row-pair by row-pair over the corpus. Per decision
row, alongside the untouched v1 golden surface, the corpus records:

- the serialized :class:`pokezero.transitions_fold.FoldState` at the decision
  boundary (production-default tail limits — the corpus exercises production
  state, never shrunken knobs);
- the inter-decision raw event slice: the public protocol lines appended since
  the previous SAME-SEAT decision boundary (``|t:|`` wall-clock lines filtered,
  the schema-v2 byte-determinism rule);
- the Tier-2 annotation overlay at the boundary (the live trackers' per-index
  conclusions — the runtime-dependent inputs of :meth:`FoldState.apply_annotations`);
- the boundary products (:class:`pokezero.transitions_fold.FoldProducts`),
  generation-time asserted equal to the production encoder state's surfaces.

The fold rows live in a gzip-compressed sidecar (``fold.jsonl.gz``) next to
``rows.jsonl``: fold payloads are tail-dominated (~226 KB late-game) and would
multiply the uncompressed corpus by an order of magnitude, while their JSON is
~10x gzip-compressible; the golden files stay byte-compatible with v1 tooling.
Records link to the golden rows by ``array_row_index`` + ``row_sha256`` and are
integrity-bound by per-record payload hashes plus the manifest file hash.

The row-pair validation contract (``validate_fold_chains``): for every
per-(game, seat) chain, row 0 must be reproduced from ``FoldState.initial()``
advanced over its slice, and every consecutive same-seat row pair must satisfy
``load(row_n.fold_state).advance(row_{n+1}.event_slice)`` +
``apply_annotations(row_{n+1}.annotation_overlay)`` == row_{n+1}'s recorded
state AND products, byte-exactly on the canonical payloads. The comparison
entry point is a pluggable backend (:class:`PythonReferenceFoldBackend`); the
upcoming Rust ``advance()`` plugs into the same seam
(``scripts/validate_corpus_v2.py --backend ...``, mirroring
``scripts/validate_rust_encoder.py``).

Read-only imports of private helpers from :mod:`pokezero.transitions_fold` are
intentional (the same pattern that module uses toward its oracle modules):
reusing the token payload codecs is what keeps this surface definitionally
aligned with the fold-state serialization rather than a re-implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import io
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Protocol, Sequence

from .transitions import TendencyStats
from .transitions_fold import (
    AnnotationValues,
    FoldProducts,
    FoldState,
    _merged_token_to_payload,
    _transition_token_to_payload,
)

FOLD_ROWS_FILENAME = "fold.jsonl.gz"

FOLD_PRODUCTS_SCHEMA_VERSION = "pokezero.fold-products.v1"

# Fields of the fold record, in documentation order (the canonical JSON writer
# sorts keys; this tuple is the schema surface for the corpus header/notes).
FOLD_RECORD_FIELDS: tuple[str, ...] = (
    "record_type",
    "schema_version",
    "battle_seed",
    "battle_id",
    "format_id",
    "player_id",
    "decision_round_index",
    "chain_index",
    "array_row_index",
    "row_sha256",
    "event_slice",
    "annotation_overlay",
    "fold_state",
    "fold_state_sha256",
    "products",
    "products_sha256",
)

_TENDENCY_STATS_FIELDS: tuple[str, ...] = (
    "perspective_slot",
    "opponent_slot",
    "opponent_switch_count",
    "opponent_decision_opportunities",
    "blocked_on_our_attack_count",
    "pursuit_intercept_predict_count",
    "my_switch_turn_count",
)

_MON_TENDENCY_FIELDS: tuple[str, ...] = (
    "slot",
    "species",
    "switched_out_before_attacking",
    "stayed_and_attacked",
    "turns_active",
)


def _tendency_stats_to_payload(stats: TendencyStats) -> dict[str, Any]:
    payload: dict[str, Any] = {name: getattr(stats, name) for name in _TENDENCY_STATS_FIELDS}
    payload["opponent_mon_tendencies"] = [
        {name: getattr(entry, name) for name in _MON_TENDENCY_FIELDS}
        for entry in stats.opponent_mon_tendencies
    ]
    payload["opponent_weather_reveals"] = [
        {"weather": entry.weather, "from_ability": entry.from_ability}
        for entry in stats.opponent_weather_reveals
    ]
    return payload


def fold_products_to_payload(products: FoldProducts) -> dict[str, Any]:
    """JSON-safe, deterministic export of the boundary products.

    This is the byte-comparison surface of the row-pair validation contract:
    a candidate ``advance()`` implementation must reproduce these bytes (via
    canonical JSON) for every boundary, not just the fold state.
    """

    return {
        "schema": FOLD_PRODUCTS_SCHEMA_VERSION,
        "transition_tokens": [
            _transition_token_to_payload(token) for token in products.transition_tokens
        ],
        "transition_token_total": products.transition_token_total,
        "turn_merged_tokens": [
            _merged_token_to_payload(token) for token in products.turn_merged_tokens
        ],
        "turn_merged_total": products.turn_merged_total,
        "tendency_stats": _tendency_stats_to_payload(products.tendency_stats),
        "cb_pinned_species": sorted(products.cb_pinned_species),
        "investment_pinned": {
            species: code for species, code in sorted(products.investment_pinned.items())
        },
    }


def _is_wall_clock_line(raw_line: str) -> bool:
    parts = raw_line.split("|")
    return (parts[1] if len(parts) > 1 else "") == "t:"


def overlay_to_payload(overlay: Mapping[int, AnnotationValues]) -> dict[str, list]:
    return {
        str(index): [values[0], bool(values[1]), bool(values[2]), float(values[3])]
        for index, values in sorted(overlay.items())
    }


def overlay_from_payload(payload: Mapping[str, Sequence[Any]]) -> dict[int, AnnotationValues]:
    return {
        int(index): (
            values[0] if values[0] is None else float(values[0]),
            bool(values[1]),
            bool(values[2]),
            float(values[3]),
        )
        for index, values in payload.items()
    }


# ---------------------------------------------------------------------------
# Generation-side capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldBoundarySurface:
    """The production encoder state's fold-visible surfaces for one captured
    decision (acting seat, at the boundary): the ANNOTATED per-action stream
    (the Tier-2 substrate), the annotated merged stream, and the tendency
    stats. Captured at sink time because tracker conclusions are as-of-first-
    assessment — they cannot be recomputed after the game."""

    player_id: str
    transition_tokens: tuple
    turn_merged_tokens: tuple
    tendency_stats: TendencyStats


class FoldSurfaceRecorder:
    """Records a :class:`FoldBoundarySurface` per captured decision, in capture
    order (parallel to the golden ``captures`` list). Reads the same
    per-player state derivation the observation encode uses; the extra read is
    deterministic and tracker-idempotent (the closure-probe battery interleaves
    identical calls)."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.surfaces: list[FoldBoundarySurface] = []

    def record(self, player_id: str) -> None:
        state = self._env._state_for_player(player_id)
        self.surfaces.append(
            FoldBoundarySurface(
                player_id=player_id,
                transition_tokens=tuple(state.transition_tokens),
                turn_merged_tokens=tuple(state.turn_merged_tokens),
                tendency_stats=state.tendency_stats,
            )
        )


@dataclass(frozen=True)
class GoldenFoldRow:
    """The schema-v2 fold surface of one decision row (writer input)."""

    player_id: str
    chain_index: int
    event_slice: tuple[str, ...]
    annotation_overlay: Mapping[str, list]
    fold_state: Mapping[str, Any]
    products: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.player_id not in {"p1", "p2"}:
            raise ValueError("fold row player_id must be p1 or p2.")
        if self.chain_index < 0:
            raise ValueError("fold row chain_index must be non-negative.")
        for line in self.event_slice:
            if _is_wall_clock_line(line):
                raise ValueError("fold row event_slice must not contain |t:| wall-clock lines.")


def _production_binding_error(
    what: str, surface: FoldBoundarySurface, boundary_index: int
) -> ValueError:
    return ValueError(
        f"fold products diverged from the production encoder state on {what} "
        f"(seat {surface.player_id}, capture #{boundary_index}); the corpus would "
        "record non-production products — refusing to write."
    )


def build_fold_rows(
    *,
    replays: Sequence[Any],
    surfaces: Sequence[FoldBoundarySurface],
    turn_merged_active: bool = True,
) -> tuple[GoldenFoldRow, ...]:
    """Build per-decision fold rows from one game's captures, in capture order.

    ``replays`` holds each captured decision's public-materialization replay
    snapshot (``context.public_materialization_state.replay``); ``surfaces``
    the parallel :class:`FoldBoundarySurface` list. Per (game, seat) chain the
    fold state is advanced over the inter-boundary slice, the live-tracker
    overlay is applied, and the resulting products are ASSERTED equal to the
    production encoder state's surfaces before anything is recorded.
    """

    if len(replays) != len(surfaces):
        raise ValueError("replays and surfaces must be parallel capture lists.")
    states: dict[str, FoldState] = {}
    prev_line_count: dict[str, int] = {"p1": 0, "p2": 0}
    chain_counters: dict[str, int] = {"p1": 0, "p2": 0}
    game_prev_raw: tuple[str, ...] = ()
    rows: list[GoldenFoldRow] = []
    for boundary_index, (replay, surface) in enumerate(zip(replays, surfaces)):
        seat = surface.player_id
        raw = tuple(event.raw_line for event in replay.public_events)
        if raw[: len(game_prev_raw)] != game_prev_raw:
            raise ValueError(
                f"public event stream is not append-only at capture #{boundary_index}; "
                "cannot slice inter-decision events."
            )
        game_prev_raw = raw
        slice_ = tuple(
            line for line in raw[prev_line_count[seat] :] if not _is_wall_clock_line(line)
        )
        prev_line_count[seat] = len(raw)

        state = states.get(seat) or FoldState.initial(perspective_slot=seat)
        state, _ = state.advance(slice_)
        overlay = {
            index: (token.residual, token.residual_valid, token.cb_bit, token.investment)
            for index, token in enumerate(surface.transition_tokens)
            if token.residual is not None
            or token.residual_valid
            or token.cb_bit
            or token.investment
        }
        if overlay:
            state = state.apply_annotations(overlay)
        states[seat] = state
        products = state.products()

        # Production binding: the recorded products must BE the encoder-visible
        # surfaces, not merely fold-internally consistent.
        if products.transition_token_total != len(surface.transition_tokens):
            raise _production_binding_error("transition_token_total", surface, boundary_index)
        if products.transition_tokens != surface.transition_tokens[-state.action_tail_limit :]:
            raise _production_binding_error("transition_tokens", surface, boundary_index)
        if turn_merged_active:
            if products.turn_merged_total != len(surface.turn_merged_tokens):
                raise _production_binding_error("turn_merged_total", surface, boundary_index)
            if products.turn_merged_tokens != surface.turn_merged_tokens[-state.merged_tail_limit :]:
                raise _production_binding_error("turn_merged_tokens", surface, boundary_index)
        if products.tendency_stats != surface.tendency_stats:
            raise _production_binding_error("tendency_stats", surface, boundary_index)

        rows.append(
            GoldenFoldRow(
                player_id=seat,
                chain_index=chain_counters[seat],
                event_slice=slice_,
                annotation_overlay=overlay_to_payload(overlay),
                fold_state=state.to_payload(),
                products=fold_products_to_payload(products),
            )
        )
        chain_counters[seat] += 1
    return tuple(rows)


# ---------------------------------------------------------------------------
# Sidecar writer / reader
# ---------------------------------------------------------------------------


class FoldSidecarWriter:
    """Streams fold records into ``fold.jsonl.gz`` (gzip, ``mtime=0`` so the
    file is byte-deterministic for identical content). Tracks the naive
    (uncompressed) byte total — the schema-v2 size-engineering headline."""

    def __init__(self, path: Path, *, canonical_json_bytes: Callable[[Any], bytes]) -> None:
        self._canonical = canonical_json_bytes
        self._raw = path.open("xb")
        self._gzip = gzip.GzipFile(filename="", fileobj=self._raw, mode="wb", mtime=0)
        self.uncompressed_bytes = 0
        self.record_count = 0

    def write_header(self, header: Mapping[str, Any]) -> None:
        self._write(header)

    def write_record(self, record: Mapping[str, Any]) -> None:
        self._write(record)
        self.record_count += 1

    def _write(self, payload: Mapping[str, Any]) -> None:
        line = self._canonical(payload) + b"\n"
        self._gzip.write(line)
        self.uncompressed_bytes += len(line)

    def close(self) -> None:
        self._gzip.close()
        self._raw.close()


def fold_record_from_row(
    fold_row: GoldenFoldRow,
    *,
    schema_version: str,
    battle_seed: int,
    battle_id: str,
    format_id: str,
    player_id: str,
    decision_round_index: int,
    array_row_index: int,
    row_sha256: str,
    canonical_sha256: Callable[[Any], str],
) -> dict[str, Any]:
    if fold_row.player_id != player_id:
        raise ValueError(
            f"fold row seat {fold_row.player_id!r} does not match its decision row "
            f"seat {player_id!r} (array row {array_row_index})."
        )
    fold_state = dict(fold_row.fold_state)
    products = dict(fold_row.products)
    return {
        "record_type": "fold",
        "schema_version": schema_version,
        "battle_seed": battle_seed,
        "battle_id": battle_id,
        "format_id": format_id,
        "player_id": player_id,
        "decision_round_index": decision_round_index,
        "chain_index": fold_row.chain_index,
        "array_row_index": array_row_index,
        "row_sha256": row_sha256,
        "event_slice": list(fold_row.event_slice),
        "annotation_overlay": dict(fold_row.annotation_overlay),
        "fold_state": fold_state,
        "fold_state_sha256": canonical_sha256(fold_state),
        "products": products,
        "products_sha256": canonical_sha256(products),
    }


def iter_fold_records(
    corpus_dir: Path, *, expected_schema_version: str
) -> Iterator[Mapping[str, Any]]:
    """Stream fold records (header excluded) from a corpus directory."""

    path = Path(corpus_dir) / FOLD_ROWS_FILENAME
    with gzip.open(path, "rb") as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8")
        for line_number, line in enumerate(text, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid fold record JSON at line {line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"fold record at line {line_number} must be a JSON object.")
            record_type = payload.get("record_type")
            if line_number == 1:
                if record_type != "fold_header":
                    raise ValueError("fold sidecar must start with its fold_header record.")
                if payload.get("schema_version") != expected_schema_version:
                    raise ValueError(
                        f"fold sidecar schema_version {payload.get('schema_version')!r} does not "
                        f"match the corpus schema {expected_schema_version!r}."
                    )
                continue
            if record_type != "fold":
                raise ValueError(f"unsupported fold record_type {record_type!r} at line {line_number}.")
            if payload.get("schema_version") != expected_schema_version:
                raise ValueError(f"fold record schema_version mismatch at line {line_number}.")
            yield payload


def fold_row_from_record(record: Mapping[str, Any]) -> GoldenFoldRow:
    """Reconstruct the writer-input fold row from a sidecar record (link fields
    dropped — they are recomputed on any re-write, e.g. corpus sampling)."""

    return GoldenFoldRow(
        player_id=str(record["player_id"]),
        chain_index=int(record["chain_index"]),
        event_slice=tuple(str(line) for line in record["event_slice"]),
        annotation_overlay={
            str(key): list(value) for key, value in record["annotation_overlay"].items()
        },
        fold_state=record["fold_state"],
        products=record["products"],
    )


# ---------------------------------------------------------------------------
# Row-pair validation (the --backend seam the Rust advance() plugs into)
# ---------------------------------------------------------------------------


class FoldBackend(Protocol):
    """A candidate implementation of the fold-state advance contract.

    ``step`` performs one boundary transition — advance over the event slice,
    then apply the annotation overlay — and returns the resulting state and
    products as canonical payload dicts. A Rust ``advance()`` implements the
    same three methods (payload in, payload out) and drops into
    ``validate_fold_chains`` unchanged.
    """

    name: str

    def start(
        self, *, perspective_slot: str, merged_tail_limit: int, action_tail_limit: int
    ) -> Any: ...

    def load(self, fold_state_payload: Mapping[str, Any]) -> Any: ...

    def step(
        self,
        handle: Any,
        event_slice: Sequence[str],
        annotation_overlay: Mapping[str, Sequence[Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


class PythonReferenceFoldBackend:
    """The production Python fold (:class:`FoldState`) behind the backend seam."""

    name = "python-reference"

    def start(
        self, *, perspective_slot: str, merged_tail_limit: int, action_tail_limit: int
    ) -> FoldState:
        return FoldState.initial(
            perspective_slot=perspective_slot,
            merged_tail_limit=merged_tail_limit,
            action_tail_limit=action_tail_limit,
        )

    def load(self, fold_state_payload: Mapping[str, Any]) -> FoldState:
        return FoldState.from_payload(fold_state_payload)

    def step(
        self,
        handle: FoldState,
        event_slice: Sequence[str],
        annotation_overlay: Mapping[str, Sequence[Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state, _ = handle.advance(event_slice)
        overlay = overlay_from_payload(annotation_overlay)
        if overlay:
            state = state.apply_annotations(overlay)
        return state.to_payload(), fold_products_to_payload(state.products())


@dataclass(frozen=True)
class FoldValidationMismatch:
    battle_id: str
    player_id: str
    chain_index: int
    surface: str  # "fold_state" | "products"
    detail: str


@dataclass
class FoldChainValidationReport:
    backend: str
    games: int = 0
    chains: int = 0
    rows_validated: int = 0
    initial_validations: int = 0
    pair_validations: int = 0
    scenario_rows: int = 0
    random_rows: int = 0
    mismatch_total: int = 0
    mismatches: list[FoldValidationMismatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.mismatch_total == 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "games": self.games,
            "chains": self.chains,
            "rows_validated": self.rows_validated,
            "initial_validations": self.initial_validations,
            "pair_validations": self.pair_validations,
            "scenario_rows": self.scenario_rows,
            "random_rows": self.random_rows,
            "mismatch_total": self.mismatch_total,
            "ok": self.ok,
            "mismatches": [
                {
                    "battle_id": m.battle_id,
                    "player_id": m.player_id,
                    "chain_index": m.chain_index,
                    "surface": m.surface,
                    "detail": m.detail,
                }
                for m in self.mismatches
            ],
        }


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _payload_diff_detail(got: Any, want: Any) -> str:
    """A short, human-oriented locator for the first differing payload region."""

    if isinstance(got, Mapping) and isinstance(want, Mapping):
        keys = sorted(set(got) | set(want))
        differing = [
            key
            for key in keys
            if key not in got
            or key not in want
            or _canonical_bytes(got.get(key)) != _canonical_bytes(want.get(key))
        ]
        return f"differing keys: {differing[:8]}"
    return "payloads differ (non-mapping)"


def validate_fold_chains(
    corpus_dir: Path,
    backend: FoldBackend,
    *,
    expected_schema_version: str,
    max_mismatch_reports: int = 20,
    progress: Callable[[int], None] | None = None,
) -> FoldChainValidationReport:
    """Row-pair validation over every fold chain of a corpus.

    Per (game, seat) chain: chain start is validated from
    ``backend.start(...)`` advanced over row 0's slice; every consecutive pair
    is validated from the RECORDED row-n state (``backend.load``) — each pair
    independently, exactly the transition a search-time advance performs.
    Comparison is canonical-JSON byte equality on both the fold state and the
    products.
    """

    report = FoldChainValidationReport(backend=backend.name)
    seen_battles: set[str] = set()
    previous_by_seat: dict[str, Mapping[str, Any]] = {}
    current_battle: Optional[str] = None
    chains_seen: set[tuple[str, str]] = set()

    for record in iter_fold_records(Path(corpus_dir), expected_schema_version=expected_schema_version):
        battle_id = str(record["battle_id"])
        seat = str(record["player_id"])
        chain_index = int(record["chain_index"])
        if battle_id != current_battle:
            current_battle = battle_id
            previous_by_seat = {}
            seen_battles.add(battle_id)
        if (battle_id, seat) not in chains_seen:
            chains_seen.add((battle_id, seat))
            if chain_index != 0:
                raise ValueError(
                    f"fold chain {battle_id}/{seat} does not start at chain_index 0."
                )

        state_payload = record["fold_state"]
        if state_payload.get("perspective_slot") != seat:
            raise ValueError(
                f"fold record {battle_id}/{seat}#{chain_index} carries a state for "
                f"perspective {state_payload.get('perspective_slot')!r}."
            )

        if chain_index == 0:
            handle = backend.start(
                perspective_slot=seat,
                merged_tail_limit=int(state_payload["merged_tail_limit"]),
                action_tail_limit=int(state_payload["action_tail_limit"]),
            )
            report.initial_validations += 1
        else:
            previous = previous_by_seat.get(seat)
            if previous is None or int(previous["chain_index"]) != chain_index - 1:
                raise ValueError(
                    f"fold chain {battle_id}/{seat} is not contiguous at chain_index {chain_index}."
                )
            handle = backend.load(previous["fold_state"])
            report.pair_validations += 1

        got_state, got_products = backend.step(
            handle, list(record["event_slice"]), record["annotation_overlay"]
        )
        for surface, got, want in (
            ("fold_state", got_state, state_payload),
            ("products", got_products, record["products"]),
        ):
            if _canonical_bytes(got) != _canonical_bytes(want):
                report.mismatch_total += 1
                if len(report.mismatches) < max_mismatch_reports:
                    report.mismatches.append(
                        FoldValidationMismatch(
                            battle_id=battle_id,
                            player_id=seat,
                            chain_index=chain_index,
                            surface=surface,
                            detail=_payload_diff_detail(got, want),
                        )
                    )

        previous_by_seat[seat] = record
        report.rows_validated += 1
        if battle_id.startswith("golden-scenario-"):
            report.scenario_rows += 1
        else:
            report.random_rows += 1
        if progress is not None:
            progress(report.rows_validated)

    report.games = len(seen_battles)
    report.chains = len(chains_seen)
    return report
