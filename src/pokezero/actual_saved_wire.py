"""Offline parity against immutable, actually consumed A2 owning observations.

The mixed transcript is integrity provenance only. Only literal captured owner
lines enter its JAX backend; Python references are never regenerated. Importing
this module starts no simulator, loads no weights, and reads no corpus.
"""
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np

from .mirror_value_probe import CHAMPION_SHA, canonical, cohort, digest
from .projected_encoder_diagnostic import DiagnosticTally, refusal
from .projected_encoder_parity import ProjectedParityTally
from .recorded_encoder_corpus import RecordedEncoderCursor, _Prefix
from .recorded_encoder_inventory import _inside, _json, _sha
from .recorded_encoder_parity import SHAPES, _arrays
from .recorded_mirror_roots import CHAMPION_CONTRACT
from .live_protocol_actor import _actionable, _packet, _terminal

SEATS = ('p1', 'p2')
PRODUCER_SOURCE = '737f4e629995801297187a0367894a6effb1452f'
PRODUCER_CONFIG_SHA = '657d9a2409421855812eb9e2604a212d85ab2b6c0cf73aaec5533c95d7a16b2e'
PRODUCER_PRESPEC_SHA = 'f0a8d1dfd6c8785d35075009a0999b7d6981d4e327b0ddfd526ded74024a066f'
PRODUCER_HASHES_SHA = '8eb4b426201a595c749a779ccd127373e12a80f739281f698c187f4af8e1a59d'
PRODUCER_COHORT_FILE_SHA = '7c7e1d624d2a83ac504c43ede598c4c9fddbc5cccdce06b4a6b0fcb395dd3355'
PRODUCER_ROSTER_DIGEST = 'ab063a77bdb34390b41d0e8b1e16438581ed935666b1e3ff5471343cdd284198'
PRODUCER_FILES = {
    'scripts/run_mirror_value_probe.py': '8611f6f4086104e5dc42ae0025662ac2473565d72a27170784ea2ecf5ab45706',
    'src/pokezero/mirror_value_probe.py': 'c615cd851344982569794b6fc4c27cbe1d7fc407a58d7568362cff280d1f3b76',
    'tests/test_mirror_value_probe.py': 'cc2b98ffd54308d8172ad7e3370fd5370c22f933de08a7d86d05a611f7b0dae5',
}
VISIBILITY = 'ACTUAL_CAPTURED_OWNING_LINES_AND_IMMUTABLE_CONSUMED_PYTHON_TENSORS'
SCOPE = 'literal bridge-delivered owning lines; upstream empty strings stripped; bare | retained'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _lines(value):
    require(type(value) is list and all(type(line) is str and line.startswith('|')
        and '\n' not in line and '\r' not in line for line in value),
        'literal captured line list required')
    return tuple(value)


@dataclass(frozen=True)
class ActualSavedWirePair:
    decision_id: str
    battle_id: str
    seat: str
    decision_round_index: int
    turn_number: int
    source_prefix_sha256: str
    own_request_json: bytes
    own_reference_json: bytes
    original_reference_sha256: str
    actual: dict
    visibility: str = VISIBILITY


def actual_pair_arrays(pair):
    require(type(pair) is ActualSavedWirePair and pair.visibility == VISIBILITY,
            'distinct actual saved-wire pair required')
    require(all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) for v in
        (pair.decision_id, pair.source_prefix_sha256, pair.original_reference_sha256)),
        'actual pair hashes required')
    require(pair.seat in SEATS and type(pair.battle_id) is str and bool(pair.battle_id.strip())
        and '\n' not in pair.battle_id and '\r' not in pair.battle_id
        and type(pair.decision_round_index) is int and pair.decision_round_index >= 0
        and type(pair.turn_number) is int and pair.turn_number >= 0,
        'actual pair identity required')
    for value in (pair.own_request_json, pair.own_reference_json):
        require(type(value) is bytes and canonical(json.loads(value)) == value,
                'immutable canonical reference/request required')
    require(_actionable(_packet('|request|' + pair.own_request_json.decode(), pair.seat)),
            'actionable actual owning request required')
    reference = json.loads(pair.own_reference_json)
    require(set(reference) == set(SHAPES) | {'turn_number'}
        and type(reference['turn_number']) is int and reference['turn_number'] == pair.turn_number
        and digest(reference) == pair.original_reference_sha256,
        'actual consumed reference digest/turn mismatch')
    return _arrays(pair.actual, device=True), _arrays({k: reference[k] for k in SHAPES}, device=False)


class ActualSavedWireTally(ProjectedParityTally):
    """Reuse the all-cell reducer, with an actual-capture envelope and identity."""
    def __init__(self, expected_views):
        super().__init__(expected_views=expected_views, pair_arrays=actual_pair_arrays,
            reference_json=lambda p: p.own_reference_json,
            reference_hash_key='actual_consumed_reference_sha256')

    def report(self):
        result = super().report()
        for key in ('scope', 'struggle_warning_policy', 'status', 'visibility', 'schema'):
            result.pop(key)
        for key in tuple(result):
            if key.startswith('projected_'):
                result[key.removeprefix('projected_')] = result.pop(key)
        result.update(schema='pokezero.B2-actual-saved-wire-metrics.v1',
            status='ACTUAL_CAPTURE_METRICS_REQUIRES_SCIENTIFIC_REVIEW',
            visibility=VISIBILITY, scope=SCOPE, actual_captured_wire=True,
            captured_player_wire=True, original_reference_values_compared=self.rows > 0,
            reference_values_source='actual immutable consumed Python observation tensors',
            owning_stream_bytes_reconstructed=False, raw_bridge_chunk_bytes_captured=False,
            full_encoder_admitted=False, public_wire_qualified=False)
        return result


class ActualSavedWireCursor:
    """Atomically validate both owner journals, mixed provenance, and references."""
    def __init__(self, battle):
        self.battle = battle
        self.source = RecordedEncoderCursor(battle_id=battle['battle_id'], seed=battle['seed'])
        self.prefixes = {s: _Prefix() for s in SEATS}
        self.requests = dict.fromkeys(SEATS)
        self.turns = dict.fromkeys(SEATS, 0)
        self.names = {}
        self.closed = False

    def _owners(self, row, *, end):
        require(type(row.get('owning_wire')) is dict and set(row['owning_wire']) == set(SEATS),
                'both actual owning journals required')
        prefixes, requests, turns, deltas = dict(self.prefixes), dict(self.requests), dict(self.turns), {}
        names = dict(self.names)
        for seat in SEATS:
            wire = row['owning_wire'][seat]
            require(type(wire) is dict and set(wire) == {'start', 'end', 'delta', 'prefix_sha256'},
                    'exact captured owner cursor required')
            lines = _lines(wire['delta'])
            require(type(wire['start']) is int and wire['start'] == prefixes[seat].count
                and type(wire['end']) is int and wire['end'] == wire['start'] + len(lines),
                'owning wire continuity mismatch')
            prefixes[seat] = prefixes[seat].append(lines)
            require(prefixes[seat].sha256() == wire['prefix_sha256'], 'owning prefix hash mismatch')
            terminal_seen = False
            for line in lines:
                require(not terminal_seen or line == '|', 'postterminal captured owner input')
                if _terminal(line):
                    require(end, 'terminal lifecycle cannot enter decision input')
                    terminal_seen = True
                elif line.startswith('|request|'):
                    requests[seat] = _packet(line, seat)
                elif line.startswith('|turn|'):
                    text = line[len('|turn|'):]
                    require(text.isdecimal() and int(text) >= turns[seat], 'owner turn chronology mismatch')
                    turns[seat] = int(text)
                elif line.startswith('|player|'):
                    fields = line.split('|')
                    require(len(fields) >= 4 and fields[2] in SEATS and bool(fields[3]),
                            'player identity required')
                    if fields[2] in names:
                        require(names[fields[2]] == fields[3], 'player name changed')
                    names[fields[2]] = fields[3]
            deltas[seat] = lines
        return prefixes, requests, turns, deltas, names

    def advance(self, row):
        require(not self.closed, 'post-end captured record')
        prefixes, requests, turns, deltas, names = self._owners(row, end=False)
        require(requests == row.get('requests'), 'actual latest owner requests differ from provenance')
        for seat in row.get('requested_players', ()):
            reference = row['observations'][seat]
            require(set(reference) == set(SHAPES) | {'turn_number'}
                and type(reference.get('turn_number')) is int and reference['turn_number'] == turns[seat],
                'actual consumed observation turn mismatch')
            _arrays({k: reference[k] for k in SHAPES}, device=False)
        projection = self.source.advance(row)
        # The mixed latest requests were checked as provenance only. The
        # request used for encoding is copied from this owner's literal
        # captured journal, even though equality was required above.
        projection = replace(projection, views=tuple(replace(view,
            own_request_json=canonical(requests[view.seat])) for view in projection.views))
        self.prefixes, self.requests, self.turns, self.names = prefixes, requests, turns, names
        return projection, deltas

    def close(self, row):
        require(not self.closed and row.get('event') == 'end'
            and row.get('battle_id') == self.battle['battle_id'] and 'actual_terminal' in row,
            'complete captured end identity required')
        prefixes, requests, turns, deltas, names = self._owners(row, end=True)
        source, _, _, _, _ = self.source._delta(row)
        terminal = row['actual_terminal']
        require(terminal == self.battle['actual_terminal'], 'attached actual terminal differs')
        if terminal is not None:
            require(type(terminal) is dict and set(terminal) == {'winner', 'turn_count', 'capped'}
                and type(terminal['capped']) is bool and terminal['winner'] in (None, *SEATS)
                and type(terminal['turn_count']) is int and terminal['turn_count'] >= 0,
                'actual terminal contract required')
        natural = terminal is not None and not terminal['capped']
        endings = [[line for line in lines if _terminal(line)] for lines in
            (row['protocol_delta'], deltas['p1'], deltas['p2'])]
        if natural:
            require(all(len(lines) == 1 for lines in endings), 'natural terminal missing from captured streams')
            expected = '|tie' if terminal['winner'] is None else '|win|' + names.get(terminal['winner'], '')
            require(terminal['winner'] is None or terminal['winner'] in names, 'terminal player name missing')
            require(all(lines[0].rstrip('|') == expected.rstrip('|') for lines in endings),
                    'captured terminal winner mismatch')
            require(all(turn == terminal['turn_count'] for turn in turns.values()),
                    'terminal turn count mismatch')
        else:
            require(not any(endings), 'absent/capped terminal cannot be a captured natural result')
            # The max-round collector end marker is explicit, never a draw.
        self.prefixes, self.requests, self.turns, self.names = prefixes, requests, turns, names
        self.source._source = source
        self.closed = True


class SavedWireJaxAdapter:
    """Exact JaxBackend ingestion, retaining segment offsets for refusal evidence."""
    def __init__(self, forward_adapter, seat):
        require(seat in SEATS, 'literal owner required')
        self.forward_adapter, self.seat = forward_adapter, seat

    def reset(self, *, stream_id, battle_id):
        from .live_protocol_actor import ActorContract, JaxBackend
        self.backend = JaxBackend(ActorContract.champion(), catalog=self.forward_adapter.catalog,
            battle_id=battle_id, owner=self.seat, stream_id=stream_id)
        require(getattr(self.backend.adapter, 'owner_side', None) == SEATS.index(self.seat),
                'reviewed explicit owner HP-provenance encoder fix required')

    def ingest(self, lines):
        # The backend already flushes every own request. These same segments
        # provide precise offsets when a public fold/private parser refuses.
        segments, start = [], 0
        for offset, line in enumerate(lines):
            if line.startswith('|request|'):
                if start < offset:
                    segments.append((start, lines[start:offset]))
                segments.append((offset, (line,)))
                start = offset + 1
        if start < len(lines):
            segments.append((start, lines[start:]))
        for start, segment in segments:
            try:
                self.backend.ingest(segment)
            except Exception as error:
                detail = refusal(error, 'jax_ingest', segment)
                if len(segment) == 1:
                    detail.update(line_kind=segment[0].split('|')[1] or 'SEPARATOR', line_offset=0,
                        line_sha256=hashlib.sha256(segment[0].encode()).hexdigest())
                if detail['line_offset'] is not None:
                    detail['line_offset'] += start
                raise WireEncoderRefusal(detail) from error

    def encode(self, view):
        arrays, _ = self.backend.encode(SimpleNamespace(
            own_request_line='|request|' + view.own_request_json.decode()))
        return arrays

    def forward(self, arrays):
        return self.forward_adapter.forward(arrays)


class WireEncoderRefusal(Exception):
    def __init__(self, detail):
        super().__init__('captured owning encoder refusal')
        self.detail = detail


def _rows(path, expected_sha):
    # Check both sides of streaming consumption. Corrupt or changed artifacts
    # are fatal, even if their already-seen decisions were refused by JAX.
    require(_sha(path) == expected_sha, 'boundary artifact hash mismatch')
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)
    require(_sha(path) == expected_sha, 'boundary artifact changed during read')


def inventory_saved_battle(rows, battle):
    cursor = ActualSavedWireCursor(battle)
    boundaries = views = asymmetric = 0
    identities = []
    for row in rows:
        require(type(row) is dict and not cursor.closed, 'invalid or post-end captured record')
        if row.get('event') == 'end':
            cursor.close(row)
            continue
        projection, _ = cursor.advance(row)
        if boundaries == 0 and 'initial_true_requests' in battle:
            require(cursor.requests == battle['initial_true_requests'],
                    'initial true-set provenance differs from first captured requests')
        boundaries += 1
        views += len(projection.views)
        asymmetric += len(projection.views) != 2
        if len(projection.views) == 2:
            identities.extend(dict(root_id=projection.root_id, seat=v.seat,
                turn_number=v.turn_number, decision_round_index=v.decision_round_index)
                for v in projection.views)
    require(cursor.closed, 'missing captured end record')
    require(identities == battle['identities'] and len(identities) == battle['paired_rows'],
            'actual paired identity roster differs from attached receipt')
    return dict(battle_id=battle['battle_id'], boundaries=boundaries, requested_views=views,
        asymmetric_boundaries=asymmetric, actual_terminal=battle['actual_terminal'],
        owning_prefix_sha256={s: cursor.prefixes[s].sha256() for s in SEATS},
        owning_line_count={s: cursor.prefixes[s].count for s in SEATS})


def compare_saved_battle(rows, *, battle, ordinal, adapters, tally, with_forward):
    cursor = ActualSavedWireCursor(battle)
    taints = dict.fromkeys(SEATS)
    for index, seat in enumerate(SEATS):
        adapters[seat].reset(stream_id=2 * ordinal + index + 1, battle_id=battle['battle_id'])
    for row in rows:
        require(type(row) is dict and not cursor.closed, 'invalid or post-end captured record')
        if row.get('event') == 'end':
            cursor.close(row)
            # Remaining transition and terminal lines validate provenance only.
            # There is no requested decision after this collector end marker.
            continue
        projection, deltas = cursor.advance(row)
        for seat in SEATS:
            if taints[seat] is None:
                try:
                    adapters[seat].ingest(deltas[seat])
                except Exception as error:
                    cause = error.detail if type(error) is WireEncoderRefusal else refusal(error, 'jax_ingest', deltas[seat])
                    taints[seat] = dict(cause, origin_decision_round_index=row['decision_round_index'])
        for view in projection.views:
            tally.visit(view)
            cause = taints[view.seat]
            if cause is not None:
                tally.refuse(view, cause, propagated=cause['origin_decision_round_index'] != view.decision_round_index)
                continue
            try:
                encoded = adapters[view.seat].encode(view)
                actual = {k: np.array(v, copy=True) for k, v in encoded.items()}
                _arrays(actual, device=True)
                for value in actual.values():
                    value.setflags(write=False)
            except Exception as error:
                # Encode may mutate an implementation or fail after a partial
                # update: conservatively taint this owner's remaining prefix.
                cause = dict(refusal(error, 'decision_encode'),
                    origin_decision_round_index=row['decision_round_index'])
                taints[view.seat] = cause
                tally.refuse(view, cause)
                continue
            pair = ActualSavedWirePair(view.decision_id, view.battle_id, view.seat,
                view.decision_round_index, view.turn_number, view.source_prefix_sha256,
                view.own_request_json, view.own_reference_json,
                digest(json.loads(view.own_reference_json)), actual)
            # Envelope and reducer failures are contracts, never encoder refusals.
            actual_pair_arrays(pair)
            tally.pair(pair, adapters[view.seat], with_forward)
    require(cursor.closed, 'missing captured end record')


def _selection(root, selection, expected_selection_sha256, expected_collection_sha256,
               selection_metadata, expected_selection_metadata_sha256):
    spec = _json(selection, expected_selection_sha256)
    require(spec.get('schema') == 'pokezero.B2-actual-owning-wire-selection-prespec.v1'
        and spec.get('producer_source') == PRODUCER_SOURCE
        and spec.get('producer_config_sha256') == PRODUCER_CONFIG_SHA
        and spec.get('producer_prespec_sha256') == PRODUCER_PRESPEC_SHA
        and spec.get('cohort_sha256') == PRODUCER_ROSTER_DIGEST
        and spec.get('selection_is_training_only') is True
        and spec.get('no_actual_tensors_or_outcomes_examined') is True
        and spec.get('minimum_view_gate') == 10000
        and spec.get('selection') == 'Exactly first256 TRAIN cohort battles sorted by numeric index, both seats and every requested decision; no tensor/outcome/mismatch/length-based selection.',
        'exact frozen training-only selection prespec required')
    producer_root = root.parent
    config = _json(_inside(producer_root, 'config.json'), PRODUCER_CONFIG_SHA)
    _json(_inside(producer_root, 'prespec.json'), PRODUCER_PRESPEC_SHA)
    require(_json(_inside(producer_root, 'scientific-hashes.json'), PRODUCER_HASHES_SHA) == PRODUCER_FILES,
            'producer scientific source bindings mismatch')
    for name, expected in PRODUCER_FILES.items():
        require(_sha(_inside(producer_root, 'repo/' + name)) == expected, 'producer source file mismatch')
    source = _json(_inside(root, 'cohort.json'), PRODUCER_COHORT_FILE_SHA)
    roster = cohort(config['root'], config['battles'])
    require(source.get('schema') == 'pokezero.mirror_cohort.v1'
        and source.get('battles') == roster and source.get('roster_sha256') == PRODUCER_ROSTER_DIGEST
        and digest(roster) == PRODUCER_ROSTER_DIGEST and source.get('config') == config
        and source.get('config_sha256') == digest(config), 'exact producer cohort/config mismatch')
    manifest = _json(_inside(root, 'collection.json'), expected_collection_sha256)
    require(manifest.get('schema') == 'pokezero.mirror_collection.v1'
        and manifest.get('status') == 'COMPLETE_FIXED_COHORT'
        and manifest.get('config') == config and manifest.get('config_sha256') == digest(config)
        and config['champion_sha256'] == CHAMPION_SHA
        and config.get('probe_contract') == 'owning-wire-true-moves-champion-linear-v2'
        and manifest.get('cohort_sha256') == PRODUCER_COHORT_FILE_SHA
        and manifest.get('no_refill') is True and manifest.get('sibling_measurement') is False
        and len(manifest.get('battles', ())) == len(roster), 'complete actual producer collection required')
    for row, actual in zip(roster, manifest['battles']):
        require(all(actual.get(k) == v for k, v in row.items()), 'full collection roster mismatch')
    selected = [row for row in roster if row['split'] == 'train'][:256]
    require(len(selected) == 256, 'fixed training roster incomplete; no refill')
    metadata = _json(selection_metadata, expected_selection_metadata_sha256)
    require(metadata.get('schema') == 'pokezero.B2-fixed-actual-owning-selection-metadata.v1'
        and metadata.get('collection_sha256') == expected_collection_sha256
        and metadata.get('cohort_file_sha256') == PRODUCER_COHORT_FILE_SHA
        and metadata.get('canonical_roster_digest') == PRODUCER_ROSTER_DIGEST
        and metadata.get('producer_source_commit') == PRODUCER_SOURCE
        and metadata.get('selected_battles') == 256 and len(metadata.get('selected', ())) == 256
        and metadata.get('no_outcomes_tensors_or_actions_used') is True
        and metadata.get('only_metadata_fields_examined') is True
        and metadata.get('selection') == 'First256TRAINcohortindicesnumeric/bothseats/allrequested; fixedbefore tensors/results'
        and type(metadata.get('expected_views')) is int and metadata['expected_views'] > 0,
        'frozen metadata-only expected roster/count binding required')
    for row, frozen in zip(selected, metadata['selected']):
        require(set(frozen) == {'index', 'battle_id', 'seed', 'split', 'battle_receipt_sha256',
                'source_file_sha256', 'boundaries', 'requested_views'}
            and all(frozen.get(k) == row[k] for k in ('index', 'battle_id', 'seed', 'split'))
            and frozen.get('battle_receipt_sha256') == manifest['battle_receipt_hashes'].get(
                f"battle-{row['index']:04d}/battle.json")
            and type(frozen.get('requested_views')) is int and frozen['requested_views'] > 0
            and type(frozen.get('boundaries')) is int and frozen['boundaries'] > 0,
            'prespecified battle identity/order/count/receipt mismatch')
    require(sum(row['requested_views'] for row in metadata['selected']) == metadata['expected_views'],
            'metadata expected view denominator mismatch')
    return manifest, selected, metadata


def inventory_saved_collection(collection, *, selection, expected_selection_sha256,
                               expected_collection_sha256, selection_metadata,
                               expected_selection_metadata_sha256):
    from pathlib import Path
    root = Path(collection).resolve(strict=True)
    manifest, selected, metadata = _selection(root, Path(selection).resolve(strict=True),
        expected_selection_sha256, expected_collection_sha256,
        Path(selection_metadata).resolve(strict=True), expected_selection_metadata_sha256)
    battles, reports = [], []
    for row in selected:
        directory = f"battle-{row['index']:04d}/"
        receipt = _json(_inside(root, directory + 'battle.json'),
            manifest['battle_receipt_hashes'][directory + 'battle.json'])
        summary = manifest['battles'][row['index']]
        require(receipt.get('schema') == 'pokezero.mirror_battle.v1'
            and all(receipt.get(k) == v for k, v in row.items())
            and receipt.get('contract') == CHAMPION_CONTRACT
            and receipt.get('observation_roots_are_siblings') is False
            and all(receipt.get(k) == v for k, v in summary.items())
            and receipt.get('initial_true_requests_sha256') == digest(receipt.get('initial_true_requests')),
            'actual battle receipt/contract/provenance mismatch')
        initial = receipt['initial_true_requests']
        require(type(initial) is dict and set(initial) == set(SEATS)
            and all(_packet('|request|' + canonical(initial[s]).decode(), s)
                and len(initial[s].get('side', {}).get('pokemon', ())) == 6 for s in SEATS),
            'complete initial true-set provenance required')
        require(type(receipt.get('frozen_model_sha256')) is str
            and re.fullmatch('[0-9a-f]{64}', receipt['frozen_model_sha256']),
            'frozen producer model digest required')
        if battles:
            require(receipt['frozen_model_sha256'] == battles[0]['frozen_model_sha256'], 'producer model drift')
        report = inventory_saved_battle(_rows(_inside(root, directory + 'boundaries.jsonl.gz'),
            receipt['files']['boundaries.jsonl.gz']), receipt)
        frozen = metadata['selected'][len(battles)]
        require(report['requested_views'] == frozen['requested_views']
            and report['boundaries'] == frozen['boundaries']
            and receipt['files']['boundaries.jsonl.gz'] == frozen['source_file_sha256'],
            'actual capture differs from fixed metadata count/file hash')
        battles.append(receipt)
        reports.append(report)
    return dict(root=root, manifest=manifest, selected=selected, battles=battles, reports=reports,
        expected_views=metadata['expected_views'],
        selection_metadata_sha256=expected_selection_metadata_sha256,
        selection_sha256=expected_selection_sha256, collection_sha256=expected_collection_sha256)


def compare_saved_collection(inventory, *, adapters, checkpoint=None, progress=None, with_forward=True):
    require(type(with_forward) is bool and set(adapters) == set(SEATS)
        and adapters['p1'] is not adapters['p2'], 'two independent owner adapters required')
    expected = inventory['expected_views']
    tally = DiagnosticTally(expected, parity=ActualSavedWireTally(expected))
    for ordinal, battle in enumerate(inventory['battles']):
        directory = f"battle-{battle['index']:04d}/"
        compare_saved_battle(_rows(_inside(inventory['root'], directory + 'boundaries.jsonl.gz'),
            battle['files']['boundaries.jsonl.gz']), battle=battle, ordinal=ordinal,
            adapters=adapters, tally=tally, with_forward=with_forward)
        metrics = tally.report(require_complete=False)
        if checkpoint is not None:
            checkpoint(dict(schema='pokezero.B2-actual-saved-wire-checkpoint.v1',
                status='PARTIAL_FIXED_ROSTER_NOT_COMPLETE', battles_processed=ordinal + 1,
                expected_battles=len(inventory['battles']), metrics=metrics,
                selection_sha256=inventory['selection_sha256'],
                selection_metadata_sha256=inventory['selection_metadata_sha256'],
                collection_sha256=inventory['collection_sha256'], full_encoder_admitted=False))
        if progress is not None:
            progress(dict(battles_processed=ordinal + 1, expected_battles=len(inventory['battles']),
                processed_views=metrics['processed_views'], compared_views=metrics['views'],
                refused_input_views=metrics['refused_input_views']))
    require(_sha(_inside(inventory['root'], 'collection.json')) == inventory['collection_sha256'],
            'collection changed during comparison')
    metrics = tally.report()
    return dict(schema='pokezero.B2-actual-saved-wire-comparison.v1',
        status='COMPLETE_FIXED_ROSTER_WITH_REFUSALS', metrics=metrics,
        selection_sha256=inventory['selection_sha256'], collection_sha256=inventory['collection_sha256'],
        selection_metadata_sha256=inventory['selection_metadata_sha256'],
        cohort_file_sha256=PRODUCER_COHORT_FILE_SHA, canonical_roster_digest=PRODUCER_ROSTER_DIGEST,
        selected_battles=inventory['selected'], battles=inventory['reports'],
        producer_source=PRODUCER_SOURCE, producer_scientific_file_sha256=PRODUCER_FILES,
        actual_captured_wire=True, original_reference_values_compared=metrics['views'] > 0,
        raw_bridge_chunk_bytes_captured=False, owning_stream_bytes_reconstructed=False,
        visibility=VISIBILITY, scope=SCOPE, new_games=0, simulator_steps=0,
        full_encoder_admitted=False, qualification='requires independent scientific review and gate accounting',
        capped_or_absent_terminal_battles=sum(r['actual_terminal'] is None
            or r['actual_terminal']['capped'] for r in inventory['reports']))
