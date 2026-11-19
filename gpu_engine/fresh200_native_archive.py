"""Read Native decision boundaries from a pinned, completed prefix recording."""

import copy
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .fresh200_fixed_prefix import prefix_schedule
from .fresh200_native import require_resolved, step_packet_from_native
from .fresh200_native_consumer import verify_visibility_snapshot, verify_visibility_start
from .fresh200_native_dispatch import (
    OWNERS, DecisionEvidence, _confirmed_choices, _constructor, _decode_rows, _need, _select,
)
from .fresh200_native_phase import PhaseProof, ReplacementCounter, retained_inputs
from .fresh200_native_recording import MAX_BYTES, _bounded_read, _json
from .fresh200_owner_history import OwnerCut
from .fresh200_witness_audit import _same, digest, document


@dataclass(frozen=True)
class RecordedBoundary:
    frame: int
    snapshot: dict
    phase: PhaseProof
    history: dict
    relations: tuple
    owner_cuts: tuple[OwnerCut, OwnerCut]
    replacement_packets: int
    retained_chance: dict


def _artifact(folder, recorded_folder, name, reference):
    _need(reference['path'] == str(recorded_folder / name), 'recording artifact path differs: ' + name)
    data = _bounded_read(folder / name)
    _need(len(data) == reference['bytes'] and digest(data) == reference['sha256'],
          'recording artifact identity differs: ' + name)
    return data


class _ClosedStream:
    def __init__(self, stream, reference, recorded_path):
        _need(reference['path'] == str(recorded_path), 'recording stream path differs: ' + recorded_path.name)
        _need(type(reference['bytes']) is int and reference['bytes'] >= 0, 'invalid recording stream size')
        self.stream = stream
        self.reference = reference
        self.hash = hashlib.sha256()

    def consume(self, extent):
        start, end = extent
        _need(type(start) is int and type(end) is int and start == self.stream.tell()
              and start <= end <= self.reference['bytes'] and end - start <= MAX_BYTES,
              'closed command extents are discontinuous or out of bounds')
        data = self.stream.read(end - start)
        _need(len(data) == end - start, 'recording stream ended within a verified extent')
        self.hash.update(data)
        return _decode_rows(data)

    def finish(self):
        _need(self.stream.tell() == self.reference['bytes'] and not self.stream.read(1)
              and self.hash.hexdigest() == self.reference['sha256'],
              'recording stream has an unaccounted tail or a changed digest')


def load_recorded_boundaries(prefix, root, *, recorded_root, completed_sha256, source_registry_hash,
                             visibility_policy_sha256=None):
    """Extract the admitted PRE boundaries without starting Native or opening expected values.

    The caller supplies the admitted inputs, terminal receipt digest, and the
    producer's output mount for the same durable root. Paths inside the receipt
    are only checked against that binding. Return boundaries after all three
    consumed streams have matched their receipt hashes and EOF.
    """
    root = Path(root)
    recorded_root = Path(recorded_root)
    _need(root.is_absolute() and recorded_root.is_absolute(), 'absolute archive mount binding required')
    recording = root / 'recording'
    recorded_recording = recorded_root / 'recording'
    manifest_wire = _bounded_read(recording / 'completed.json')
    _need(digest(manifest_wire) == completed_sha256, 'completed recording digest differs')
    receipt = document(manifest_wire.decode('utf8'))
    schedule = prefix_schedule(len(prefix.steps))
    binding = document(prefix.binding_wire.decode('utf8'))
    _need(_same(binding['schedule'], [vars(command) for command in schedule]),
          'archive extent differs from the sealed prefix schedule')
    version2 = binding['schema'] == 'pokezero.fresh200.fixed_prefix_inputs.v2'
    _need(not version2 or visibility_policy_sha256 is not None,
          'frozen corpus prefix requires its owner visibility policy')
    records = receipt['commands']
    _need(receipt['schema'] == 'pokezero.fresh200.raw_command_recording.v1'
          and receipt['status'] == 'RAW_CLOSED_ARTIFACT_SET_RECORDED_ONLY'
          and receipt['source_registry_hash'] == source_registry_hash and receipt['failure'] is None
          and receipt.get('owner_visibility_sha256') == visibility_policy_sha256
          and receipt['qualification'] is False and receipt['child_exit_code'] == 0
          and receipt['closure']['status'] == 'RAW_CLOSED_ARTIFACT_SET_VERIFIED_ONLY'
          and len(records) == len(schedule) == receipt['closure']['commands'],
          'source-bound completed Native prefix receipt required')
    if visibility_policy_sha256 is not None:
        rule_wire = _artifact(recording, recorded_recording, 'owner-visibility.json',
                              receipt['admission_files']['owner-visibility.json'])
        rule = document(rule_wire.decode('utf8'))
        expected_version = 'v2' if version2 else 'v1'
        _need(rule['schema'] == 'pokezero.fresh200.owner_visibility.' + expected_version
              and digest(rule_wire) == visibility_policy_sha256
              and rule['fixed_input_binding_sha256'] == digest(prefix.binding_wire)
              and rule['start_wire_sha256'] == digest(prefix.start_wire)
              and rule['root_index'] == prefix.root_index and rule['stream_id'] == prefix.stream_id
              and rule['endpoint_frame'] == len(prefix.steps)
              and rule['reference_owner'] == OWNERS[prefix.owner] and rule['battle_id'] == prefix.battle_id,
              'archived visibility declaration differs from the frozen root')
        if version2:
            address = json.dumps(binding['selected_root'], ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).encode('utf8')
            _need(rule['storage_row'] == prefix.row and rule['selected_address_sha256'] == digest(address)
                  and rule['capture_sha256'] == binding['capture_reference']['sha256']
                  and rule['device_replay_sha256'] == binding['device_reference']['sha256']
                  and rule['ordered_roster_sha256'] == binding['ordered_roster_sha256'],
                  'archived visibility declaration differs from the frozen corpus roster')
    _need(_bounded_read(recording / 'frozen-inputs.json') == prefix.binding_wire
          and _bounded_read(recording / 'literal-teams.json') == prefix.team_wire,
          'recorded prefix binding or literal constructor differs')

    evidence = DecisionEvidence(prefix, source_registry_hash)
    prior_ordinary = None
    ordinary_packet = None
    pending_choice = None
    counter = ReplacementCounter()
    boundaries = []
    streams = {
        'audit': ('witness-audit.jsonl', 'native/witness-audit.jsonl'),
        'events': ('witness-transport-events.jsonl', 'native/witness-transport-events.jsonl'),
        'stdout': ('stdout', 'recording/stdout.bin'),
    }
    with ExitStack() as stack:
        readers = {
            key: _ClosedStream(stack.enter_context((root / relative).open('rb')),
                               receipt['files'][name], recorded_root / relative)
            for key, (name, relative) in streams.items()
        }
        for index, (command, record) in enumerate(zip(schedule, records, strict=True)):
            kind, frame = command.phase, command.frame
            _need(record['index'] == index and record['phase'] == kind
                  and record['command_id'] == f'{prefix.stream_id}/0/{index}'
                  and record['status'] == 'RAW_CLOSED_COMMAND_VERIFIED_ONLY',
                  'archived command differs from the fixed roster')
            folder = recording / f'command-{index}'
            recorded_folder = recorded_recording / folder.name
            stdin = _artifact(folder, recorded_folder, 'stdin.jsonl', record['inputs']['stdin.jsonl'])
            causal = readers['audit'].consume(record['verified_extents']['audit'])
            transport = readers['events'].consume(record['verified_extents']['events'])
            stdout = readers['stdout'].consume(record['verified_extents']['stdout'])
            stdout, events = evidence.ingest_closed(kind, causal, transport, stdout)
            if kind in ('start', 'choices'):
                chance_audit = _artifact(folder, recorded_folder, 'chance-audit.jsonl', record['chance_audit'])
                require_resolved(_decode_rows(chance_audit))
            if kind == 'start':
                if visibility_policy_sha256 is not None:
                    verify_visibility_start(document(stdin.decode('utf8')), transport, visibility_policy_sha256)
                initialization = _artifact(folder, recorded_folder, 'initializing.json', record['inputs']['initializing.json'])
                _need(stdin == prefix.start_wire
                      and _same(document(initialization.decode('utf8')), document(prefix.initial_chance_wire.decode('utf8'))),
                      'archived start inputs differ from the frozen constructor')
            elif kind == 'snapshot':
                _need(stdin == _json(dict(type=kind, battleId=prefix.battle_id,
                                          requestId=f'PRE{frame}', witnessPassive=True)),
                      'archived passive request differs from its PRE address')
                snapshots = [row['snapshot'] for row in stdout if row['type'] == 'snapshot']
                _need(len(snapshots) == 1, 'one original passive snapshot required')
                snapshot = snapshots[0]
                if visibility_policy_sha256 is not None:
                    verify_visibility_snapshot(snapshot)
                if frame == 0:
                    _constructor(snapshot, evidence.teams)
                phase, party, relations = evidence.boundary(snapshot, ordinary=prior_ordinary)
                if pending_choice is not None:
                    nonce, before = pending_choice
                    counter.completed(nonce, before=before, after=phase)
                    pending_choice = None
                boundaries.append(RecordedBoundary(
                    frame, snapshot, phase, copy.deepcopy(evidence.history.snapshot()), tuple(relations),
                    (evidence.owners['p1'].current, evidence.owners['p2'].current),
                    counter.value, retained_inputs(phase, ordinary_packet),
                ))
            elif kind == 'choices':
                step = prefix.steps[frame]
                fixed = _bounded_read(recording / f'fresh-chance-{frame}.json')
                _need(fixed == step.chance_wire, 'archived fresh chance operand identity differs')
                selection = _select(step, phase, party, relations)
                choices = {OWNERS[choice.owner]: choice.command for choice in selection}
                _need(stdin == _json(dict(type=kind, battleId=prefix.battle_id, choices=choices, submissionId=frame)),
                      'archived choices differ from independently bound Native lanes')
                chance = _artifact(folder, recorded_folder, 'chance.json', record['inputs']['chance.json'])
                packet = step_packet_from_native(phase, step.fresh_chance(), ordinary_packet,
                                                 stream=prefix.stream_id, request=frame)
                _need(_same(document(chance.decode('utf8')), packet), 'archived submitted chance packet differs')
                confirmed = _confirmed_choices(step, phase, selection, choices, stdout, events)
                if confirmed is not None:
                    prior_ordinary = confirmed
                    ordinary_packet = packet
                pending_choice = (record['command_id'], phase)
            else:
                _need(stdin == _json(dict(type=kind, battleId=prefix.battle_id)),
                      'archived teardown command differs')
        for reader in readers.values():
            reader.finish()

    _need([boundary.frame for boundary in boundaries] == list(range(len(prefix.steps) + 1))
          and boundaries[-1].phase.requested[prefix.owner] and pending_choice is None,
          'archived PRE boundaries do not cover the frozen owning root')
    return tuple(boundaries)
