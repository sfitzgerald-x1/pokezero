"""Route saved boundaries into two independent projected player histories.

This is orchestration, not a player-wire qualification. Its caller must bind
the projector, the production Python reference driver, the JAX adapters and
the complete archive. References are regenerated in full; the original
trusted reference is retained only as a digest. No simulator is started here.
"""
from dataclasses import dataclass
import json
from types import MappingProxyType

import numpy as np

from .mirror_value_probe import canonical, digest
from .recorded_encoder_corpus import RecordedEncoderCursor
from .recorded_encoder_parity import SHAPES
from .observation import PokeZeroObservationV0

SEATS = ('p1', 'p2')
VISIBILITY = 'SAVED_SOURCE_PROJECTED_PLAYER_REFERENCES_NOT_CAPTURED_WIRE'
SHOWDOWN_REVISION = 'f76228a1354b5d0f307ca2d16101294ad3a2308b'


@dataclass(frozen=True)
class _SeatSnapshot:
    seat: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _BatchSnapshot:
    battle_id: str
    p1: _SeatSnapshot
    p2: _SeatSnapshot
    source_nonrequest_start: int
    source_nonrequest_end: int
    source_nonrequest_prefix_sha256: str
    counts: object
    showdown_revision: str
    scope: str
    struggle_warning_policy: str
    captured_player_wire: bool = False
    owning_stream_bytes_reconstructed: bool = False
    public_wire_qualified: bool = False
    fullencoder: bool = False

    def for_seat(self, seat):
        if seat not in SEATS:
            raise ValueError('literal player seat required')
        return self.p1 if seat == 'p1' else self.p2


def _snapshot_batch(batch, *, original, start):
    """Read owner lines once, before any consumer, and bind source chronology."""
    end = start + len(original.trusted_nonrequest_delta)
    if (batch.battle_id != original.battle_id
            or type(batch.source_nonrequest_start) is not int
            or type(batch.source_nonrequest_end) is not int
            or batch.source_nonrequest_start != start or batch.source_nonrequest_end != end
            or batch.source_nonrequest_prefix_sha256 != original.trusted_nonrequest_prefix_sha256
            or batch.showdown_revision != SHOWDOWN_REVISION
            or any(getattr(batch, key) is not False for key in (
                'captured_player_wire', 'owning_stream_bytes_reconstructed',
                'public_wire_qualified', 'fullencoder'))
            or any(not isinstance(getattr(batch, key), str) or not getattr(batch, key).strip()
                   for key in ('scope', 'struggle_warning_policy'))):
        raise ValueError('projected source identity, chronology and unqualified scope required')
    owners = []
    for seat in SEATS:
        value = batch.for_seat(seat)
        lines = value.lines
        if (value.seat != seat or not isinstance(lines, tuple)
                or any(not isinstance(line, str) or not line.startswith('|')
                       or '\n' in line or '\r' in line
                       or line.split('|')[1] == 'request' for line in lines)):
            raise ValueError('both owning projected nonrequest line tuples required')
        owners.append(_SeatSnapshot(seat, lines))
    counts = MappingProxyType(json.loads(canonical(dict(batch.counts))))
    return _BatchSnapshot(original.battle_id, *owners, start, end,
        original.trusted_nonrequest_prefix_sha256, counts, SHOWDOWN_REVISION,
        batch.scope, batch.struggle_warning_policy)


@dataclass(frozen=True)
class ProjectedEncoderPair:
    decision_id: str
    battle_id: str
    seat: str
    decision_round_index: int
    turn_number: int
    source_prefix_sha256: str
    own_request_json: bytes
    regenerated_reference_json: bytes
    original_reference_sha256: str
    actual: dict
    visibility: str = VISIBILITY
    actual_captured_wire: bool = False
    full_encoder_admitted: bool = False


def _own_request_deltas(row):
    requests = {seat: [] for seat in SEATS}
    private_tail = False
    for line in row['protocol_delta']:
        if line.startswith('|request|'):
            private_tail = True
            packet = json.loads(line[len('|request|'):])
            seat = packet['side']['id']
            if seat not in requests:
                raise ValueError('owning request seat required')
            requests[seat].append(line)
        elif private_tail:
            # The driver receives public lines and owning requests separately.
            # That preserves chronology only for a literal private-request tail.
            raise ValueError('interleaved public/request boundary requires ordered projection')
    return {seat: tuple(lines) for seat, lines in requests.items()}


def _reference(observation, view):
    seat = view.seat
    other = SEATS[1-SEATS.index(seat)]
    perspective = getattr(observation, 'perspective', None)
    metadata = getattr(observation, 'metadata', {})
    if (not isinstance(observation, PokeZeroObservationV0)
            or observation.schema_version != 'pokezero.observation.v4'
            or perspective is None or perspective.player_id != seat
            or perspective.showdown_slot != seat or perspective.opponent_showdown_slot != other
            or metadata.get('battle_id') != view.battle_id
            or metadata.get('player_id') != seat or metadata.get('showdown_slot') != seat
            or metadata.get('opponent_showdown_slot') != other
            or type(metadata.get('turn_number')) is not int
            or metadata['turn_number'] != view.turn_number):
        raise ValueError('regenerated production V4 reference identity differs from owning view')
    fields = {}
    for name, shape in SHAPES.items():
        values = np.asarray(getattr(observation, name))
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('complete finite regenerated reference required: ' + name)
        fields[name] = values.tolist()
    fields['turn_number'] = view.turn_number
    return canonical(fields)


def compare_player_battle(rows, *, battle, stream_id, projector,
                          reference_feed, adapters, on_view):
    """Advance BOTH owners at every boundary, then compare requested views.

    ``reference_feed(batch, own_request_deltas=..., requested_seats=...,
    request_tail_verified=True)``
    must advance two separate production histories, consuming all emitted
    owning requests, including wait and asymmetric updates. It returns only
    the requested owning observations. Each adapter has reset/feed/encode.

    A callback must belong to a private whole-collection accumulator. Any
    exception aborts that collection; prior callback calls or advanced
    adapters cannot be published as a successful partial comparison. Terminal
    records are validated but are never encoded as decision observations.
    """
    if (type(stream_id) is not int or not 0 < stream_id < 2**30
            or set(adapters) != set(SEATS) or adapters['p1'] is adapters['p2']
            or not callable(reference_feed) or not callable(on_view)):
        raise ValueError('two independent adapters and bounded stream ID required')
    cursor = RecordedEncoderCursor(battle_id=battle['battle_id'], seed=battle['seed'])
    for index, seat in enumerate(SEATS):
        adapters[seat].reset(stream_id=2*stream_id+index, battle_id=battle['battle_id'])
    boundaries = views = asymmetric = source_nonrequest_end = 0
    request_counts = {seat: 0 for seat in SEATS}
    for row in rows:
        if not isinstance(row, dict) or cursor.closed:
            raise ValueError('invalid or post-terminal record')
        if row.get('event') == 'end':
            cursor.close(row)
            continue
        original = cursor.advance(row)
        requests = _own_request_deltas(row)
        projected = projector.project_nonrequest_delta(original.trusted_nonrequest_delta)
        batch = _snapshot_batch(projected, original=original, start=source_nonrequest_end)
        seats = tuple(view.seat for view in original.views)
        references = reference_feed(batch, own_request_deltas=requests,
                                    requested_seats=seats, request_tail_verified=True)
        if set(references) != set(seats):
            raise ValueError('exact requested regenerated reference roster required')
        reference_json = {view.seat: _reference(references[view.seat], view)
                          for view in original.views}
        for seat in SEATS:
            lines = batch.for_seat(seat).lines
            adapters[seat].feed(lines)
            request_counts[seat] += len(requests[seat])
        source_nonrequest_end = batch.source_nonrequest_end
        boundaries += 1
        asymmetric += len(seats) != 2
        for view in original.views:
            reference = reference_json[view.seat]
            encoded = adapters[view.seat].encode(json.loads(view.own_request_json),
                                               owner_side=SEATS.index(view.seat))
            if set(encoded) != set(SHAPES):
                raise ValueError('all five actual tensors required')
            actual = {}
            for name, shape in SHAPES.items():
                value = np.array(encoded[name], copy=True)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('complete finite actual tensors required')
                value.setflags(write=False)
                actual[name] = value
            pair = ProjectedEncoderPair(view.decision_id, view.battle_id, view.seat,
                view.decision_round_index, view.turn_number, view.source_prefix_sha256,
                view.own_request_json, reference,
                digest(json.loads(view.own_reference_json)), actual)
            on_view(pair)
            views += 1
    if not cursor.closed:
        raise ValueError('complete terminal record required')
    return dict(battle_id=battle['battle_id'], boundaries=boundaries, views=views,
        asymmetric_boundaries=asymmetric, owning_request_deltas=request_counts,
        source_prefix_sha256=cursor.source_prefix_sha256,
        visibility=VISIBILITY, actual_captured_wire=False,
        full_encoder_admitted=False, terminal_validated_not_encoded=True,
        limitation='Caller attests projection/source/runtime and discards all partial work on failure.')
