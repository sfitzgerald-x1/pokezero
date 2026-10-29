"""Compare the complete fixed collection using two regenerated player histories.

Only JSON protocol/reference records are opened. A private tally is published
only after both passes, all owning views and every compressed descriptor hash
validate. Runtime/source attestation remains the caller's responsibility.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import zlib

import numpy as np

from .mirror_value_probe import digest
from .recorded_encoder_corpus import _Prefix
from .recorded_encoder_inventory import inventory_collection, _inside, _json, _HashedInput, _sha
from .recorded_player_encoder_runner import compare_player_battle, VISIBILITY
from .saved_player_projection import (SavedSourcePlayerProjector, new_player_reference_environments,
    feed_player_reference_boundary, PROJECTION_SCOPE, STRUGGLE_WARNING_POLICY, PINNED_SHOWDOWN_REVISION)
from .projected_encoder_parity import ProjectedParityTally, projected_pair_arrays, _sha as _valid_sha

SEATS = ('p1', 'p2')
FLAGS = ('captured_player_wire', 'owning_stream_bytes_reconstructed', 'public_wire_qualified', 'fullencoder')


class _BattleAudit:
    """Audit the immutable runner snapshot, then require that exact GPU handoff."""
    def __init__(self, battle_id):
        self.battle_id = battle_id
        self.source_end = self.boundaries = 0
        self.source_prefix = None
        self.pending = {}
        self.players = {seat: _Prefix() for seat in SEATS}
        self.requests = {seat: _Prefix() for seat in SEATS}
        self.counts = Counter()
        self.snapshots = _Prefix()

    def reference(self, environments, batch, *, own_request_deltas, requested_seats, request_tail_verified):
        if (self.pending or batch.battle_id != self.battle_id
                or batch.scope != PROJECTION_SCOPE or batch.struggle_warning_policy != STRUGGLE_WARNING_POLICY
                or batch.showdown_revision != PINNED_SHOWDOWN_REVISION
                or any(getattr(batch, flag) is not False for flag in FLAGS)
                or type(batch.source_nonrequest_start) is not int or type(batch.source_nonrequest_end) is not int
                or batch.source_nonrequest_start != self.source_end
                or batch.source_nonrequest_end < self.source_end
                or not _valid_sha(batch.source_nonrequest_prefix_sha256)
                or request_tail_verified is not True or set(own_request_deltas) != set(SEATS)):
            raise ValueError('exact unqualified projected snapshot chronology required')
        counts = dict(batch.counts)
        if any(not isinstance(key, str) or type(value) is not int or value < 0 for key, value in counts.items()):
            raise ValueError('nonnegative source projection counters required')
        lines, requests = {}, {}
        for seat in SEATS:
            projected = batch.for_seat(seat)
            if projected.seat != seat or not isinstance(projected.lines, tuple):
                raise ValueError('immutable owning projection required')
            lines[seat] = projected.lines
            requests[seat] = own_request_deltas[seat]
            if not isinstance(requests[seat], tuple):
                raise ValueError('immutable owning request tails required')
            for line in lines[seat]:
                if (not isinstance(line, str) or not line.startswith('|') or '\n' in line or '\r' in line
                        or line.startswith('|request|')):
                    raise ValueError('invalid projected nonrequest line')
            for line in requests[seat]:
                if (not isinstance(line, str) or not line.startswith('|request|') or '\n' in line or '\r' in line
                        or json.loads(line[len('|request|'):]).get('side', {}).get('id') != seat):
                    raise ValueError('invalid owning request tail')
        result = feed_player_reference_boundary(environments, batch, own_request_deltas=own_request_deltas,
            requested_seats=requested_seats, request_tail_verified=True)
        # The reviewed runner passes this SAME frozen batch to the reference
        # driver and adapters. The wrapper below verifies both owner tuple IDs.
        self.pending = dict(lines)
        for seat in SEATS:
            self.players[seat] = self.players[seat].append(lines[seat])
            self.requests[seat] = self.requests[seat].append(requests[seat])
        self.counts.update(counts)
        self.source_end = batch.source_nonrequest_end
        self.source_prefix = batch.source_nonrequest_prefix_sha256
        self.boundaries += 1
        self.snapshots = self.snapshots.append([dict(source_nonrequest_start=batch.source_nonrequest_start,
            source_nonrequest_end=batch.source_nonrequest_end, source_nonrequest_prefix_sha256=self.source_prefix,
            counts=counts, requested_seats=list(requested_seats),
            owners={seat: dict(projected_lines=len(lines[seat]), projected_delta_sha256=digest(lines[seat]),
                own_request_lines=len(requests[seat]), own_request_delta_sha256=digest(requests[seat])) for seat in SEATS})])
        return result

    def report(self):
        if self.pending:
            raise ValueError('both owning adapter handoffs must complete')
        return dict(boundaries=self.boundaries, source_nonrequest_lines=self.source_end,
            decision_boundaries_only=True, terminal_projected=False,
            source_nonrequest_prefix_sha256=self.source_prefix, snapshot_audit_sha256=self.snapshots.sha256(),
            projection_counts=dict(sorted(self.counts.items())),
            owners={seat: dict(projected_lines=self.players[seat].count,
                projected_prefix_sha256=self.players[seat].sha256(),
                own_request_lines=self.requests[seat].count,
                own_request_prefix_sha256=self.requests[seat].sha256()) for seat in SEATS},
            audit_uses_actual_reference_and_adapter_snapshot=True,
            scope=PROJECTION_SCOPE, struggle_warning_policy=STRUGGLE_WARNING_POLICY,
            showdown_revision=PINNED_SHOWDOWN_REVISION, **dict.fromkeys(FLAGS, False))


class _AuditedAdapter:
    def __init__(self, adapter, seat, audit):
        self.adapter, self.seat, self.audit = adapter, seat, audit

    def reset(self, **kwargs):
        return self.adapter.reset(**kwargs)

    def feed(self, lines):
        if self.seat not in self.audit.pending or lines is not self.audit.pending[self.seat]:
            raise ValueError('adapter must consume the exact audited reference snapshot')
        self.adapter.feed(lines)
        del self.audit.pending[self.seat]

    def encode(self, request, *, owner_side):
        if self.audit.pending:
            raise ValueError('both owners must advance before any encode')
        return self.adapter.encode(request, owner_side=owner_side)


def _close_environments(environments):
    """Attempt both closes even if the first raises; never hide a close failure."""
    errors = []
    for environment in environments.values():
        try:
            environment.close()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup('owning reference environment cleanup failed', errors)


def _forward_input(arrays):
    # Separate immutable copies prevent a callback from altering the tally's
    # pair or the other model input. Metadata/lineage never reaches a model.
    result = {}
    for name, values in arrays.items():
        value = np.array(values, copy=True)
        value.setflags(write=False)
        result[name] = value
    return result


def compare_projected_collection(collection, *, expected_collection_sha256, expected_cohort_sha256,
                                 adapters, reference_config, with_forward):
    """Whole fixed cohort only; any error discards every private accumulated row."""
    if (type(with_forward) is not bool or set(adapters) != set(SEATS)
            or adapters['p1'] is adapters['p2']):
        raise ValueError('two independent adapters and explicit forward mode required')
    root = Path(collection).resolve(strict=True)
    inventory = inventory_collection(root, expected_collection_sha256=expected_collection_sha256,
        expected_cohort_sha256=expected_cohort_sha256)
    manifest = _json(_inside(root, 'collection.json'), expected_collection_sha256)
    tally = ProjectedParityTally(expected_views=inventory['totals']['requested_views'])
    reports, replayed = [], []
    for index, battle in enumerate(manifest['battles']):
        rel = f"battle-{battle['index']:04d}/battle.json"
        receipt = _json(_inside(root, rel), manifest['battle_receipt_hashes'][rel])
        expected = receipt['files']['boundaries.jsonl.gz']
        path = _inside(root, f"battle-{battle['index']:04d}/boundaries.jsonl.gz")
        environments = new_player_reference_environments(reference_config, battle_id=battle['battle_id'])
        try:
            if set(environments) != set(SEATS) or environments['p1'] is environments['p2']:
                raise ValueError('fresh two-owner production references required')
            audit = _BattleAudit(battle['battle_id'])
            def reference_feed(batch, **kwargs):
                return audit.reference(environments, batch, **kwargs)
            def on_view(pair):
                got, ref = projected_pair_arrays(pair)
                kwargs = {}
                if with_forward:
                    adapter = adapters[pair.seat]
                    kwargs = dict(actual_forward=adapter.forward(_forward_input(got)),
                                  reference_forward=adapter.forward(_forward_input(ref)))
                tally.add(pair, **kwargs)
            with path.open('rb') as raw:
                hashed = _HashedInput(raw)
                try:
                    with gzip.GzipFile(fileobj=hashed, mode='rb') as stream:
                        report = compare_player_battle((json.loads(line) for line in stream), battle=battle,
                            stream_id=index+1, projector=SavedSourcePlayerProjector(battle_id=battle['battle_id']),
                            reference_feed=reference_feed, adapters={seat: _AuditedAdapter(adapters[seat], seat, audit)
                                for seat in SEATS}, on_view=on_view)
                except (gzip.BadGzipFile, EOFError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
                    raise ValueError('malformed projected comparison boundary artifact') from None
                for _ in iter(lambda: hashed.read(1024*1024), b''):
                    pass
                if hashed.hasher.hexdigest() != expected:
                    raise ValueError('replayed projected boundary artifact SHA mismatch')
            report['projection_audit'] = audit.report()
            if (report['boundaries'] != audit.boundaries
                    or report['owning_request_deltas'] != {seat: audit.requests[seat].count for seat in SEATS}):
                raise ValueError('projected snapshot and owning request counts differ')
            reports.append(report)
            replayed.append(dict(battle_id=battle['battle_id'], boundary_sha256=expected))
        finally:
            _close_environments(environments)
    if (_sha(_inside(root, 'collection.json')) != expected_collection_sha256
            or _sha(_inside(root, 'cohort.json')) != expected_cohort_sha256):
        raise ValueError('projected comparison metadata changed during execution')
    metrics = tally.report()
    if (len(reports) != inventory['battle_count'] or not metrics['complete_view_count']
            or sum(row['boundaries'] for row in reports) != inventory['totals']['boundaries']
            or sum(row['asymmetric_boundaries'] for row in reports) != inventory['totals']['asymmetric_boundaries']
            or sum(sum(row['owning_request_deltas'].values()) for row in reports)
               != inventory['totals']['discarded_private_request_lines']
            or (with_forward and not metrics['model_forward_complete'])):
        raise ValueError('whole fixed projected source roster was not compared')
    return dict(schema='pokezero.B2-projected-player-collection-comparison.v1',
        status='COMPLETE_FIXED_PROJECTED_PLAYER_METRICS_ONLY', visibility=VISIBILITY,
        scope=PROJECTION_SCOPE, struggle_warning_policy=STRUGGLE_WARNING_POLICY,
        showdown_revision=PINNED_SHOWDOWN_REVISION, inventory=inventory, metrics=metrics,
        battle_execution_sha256=digest(reports), replayed_artifact_sha256=digest(replayed), battles=reports,
        new_games=0, simulator_steps=0, actual_captured_wire=False, original_reference_values_compared=False,
        full_encoder_admitted=False, **dict.fromkeys(FLAGS, False),
        limitation='Source-derived observation projection only. Original references supply lineage only. '
            'Complete runtime/source, vocabulary, checkpoint and actual JAX/FP32 execution attestation remain required.')
