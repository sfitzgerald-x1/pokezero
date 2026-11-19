"""Replay the fixed Native prefix from verified requests and causal records."""

from dataclasses import dataclass

from .fresh200_fixed_prefix import FrozenPrefix, prefix_schedule
from .fresh200_native import require_resolved
from .fresh200_native_actions import LockResults, UNKNOWN, native_action_relation
from .fresh200_native_consumer import CommandSpec
from .fresh200_native_identity import PartyMap, to_id
from .fresh200_native_ledger import NativeHistory, fields, items, member, move_id, pokemon
from .fresh200_native_phase import OrdinaryInputs, phase_from_native
from .fresh200_native_recording import MAX_BYTES, _durable, _json
from .fresh200_owner_history import OwnerHistory
from .fresh200_witness_audit import _same, digest, document


OWNERS = ('p1', 'p2')
REQUEST_PREFIX = '|request|'
UNDEFINED = {'$native': 'undefined'}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(path, extent):
    start, end = extent
    _need(0 <= start <= end and end - start <= MAX_BYTES, 'closed command bytes exceed limit')
    with path.open('rb') as stream:
        stream.seek(start)
        data = stream.read(end - start)
    _need(len(data) == end - start, 'closed command extent is incomplete')
    return _decode_rows(data)


def _decode_rows(data):
    _need(not data or data.endswith(b'\n'), 'closed command bytes are incomplete')
    return [document(line.decode('utf8')) for line in data.split(b'\n')[:-1]]


def _literal(projected):
    """Apply the serializer's treatment of undefined, with one data value per token."""
    if projected is None or type(projected) in (str, bool, int, float):
        return projected
    if _same(projected, UNDEFINED):
        return None
    if type(projected) is dict and 'items' in projected:
        return [_literal(value) for value in items(projected)]
    return {key: _literal(value) for key, value in fields(projected).items() if not _same(value, UNDEFINED)}


@dataclass(frozen=True)
class EmittedRequest:
    generation: int
    line_sha256: str
    emit_call_id: int
    request_call_id: int | None
    request_object: int | None
    originally_requested: bool


@dataclass(frozen=True)
class FixedSelection:
    owner: int
    actor: int
    lane: int
    command: str
    target: int | None
    move: str | None
    kind: str


class DecisionEvidence:
    def __init__(self, prefix, registry):
        self.teams = document(prefix.team_wire.decode('utf8'))
        self.species = tuple(tuple(mon['species'] for mon in team) for team in self.teams)
        self.history = NativeHistory(battle_id=prefix.battle_id, source_registry_hash=registry)
        self.owners = {owner: OwnerHistory(owner) for owner in OWNERS}
        self.emissions = {}

    def ingest(self, recorder, record):
        offsets = record['verified_extents']
        causal = _rows(recorder.directory / 'witness-audit.jsonl', offsets['audit'])
        events = _rows(recorder.directory / 'witness-transport-events.jsonl', offsets['events'])
        stdout = _rows(recorder.artifacts / 'stdout.bin', offsets['stdout'])
        return self.ingest_closed(record['phase'], causal, events, stdout)

    def ingest_closed(self, phase, causal, events, stdout):
        if phase == 'start':
            self.history.bind_constructor(causal)
        previous = len(self.history.events)
        self.history.consume(causal)
        for packet in stdout:
            _need(packet['type'] != 'error', 'Native bridge reported a command error')
            if packet['type'] == 'stream' and packet['stream'] in self.owners:
                lines = packet['lines']
                _need(not any(line.startswith('|error|') for line in lines), 'Native refused an owner action')
                self.owners[packet['stream']].append(lines)
        deliveries = [row['evidence'] for row in events if row['operation'] == 'REQUEST_CONSUMED']
        self._bind_deliveries(causal, deliveries)
        return stdout, self.history.events[previous:]

    def _bind_deliveries(self, causal, deliveries):
        if not deliveries:
            return
        enters = {row['frame']['callId']: row for row in causal if row['operation'] == 'CALL_ENTER'}
        returns = {row['frame']['callId']: row for row in causal if row['operation'] == 'CALL_RETURN'}
        builds = [row for row in returns.values() if row['frame']['operation'] == 'Battle.getRequests']
        ordered = {owner: {} for owner in OWNERS}
        for owner, view in self.owners.items():
            for generation, line in enumerate((x for x in view.lines if x.startswith(REQUEST_PREFIX)), 1):
                ordered[owner][generation] = line
        for delivery in deliveries:
            self._bind_delivery(delivery, enters, returns, builds,
                                ordered[delivery['owner']][delivery['generation']])

    def _bind_delivery(self, delivery, enters, returns, builds, line):
        owner = delivery['owner']
        side = OWNERS.index(owner)
        call = delivery['emit_call_id']
        emit = enters[call]
        _need(emit['frame']['operation'] == 'Side.emitRequest' and returns[call]['frame'] == emit['frame']
              and emit['evidence']['subject']['n'] == side and emit['nonce'] == delivery['emit_command_id']
              and digest(line.encode()) == delivery['line_sha256'], 'owner delivery and original emitRequest differ')
        request = document(line[len(REQUEST_PREFIX):])
        actionable = delivery['classification'] == 'actionable'
        source_call = None
        object_token = None
        if actionable:
            eligible = [row for row in builds if row['sequence'] < emit['sequence']]
            _need(eligible, 'owner request has no completed causal Native build')
            latest = max(eligible, key=lambda row: row['sequence'])
            source = items(latest['evidence']['result'])[side]
            _need(_same(_literal(source), request), 'delivered owner request differs from the latest Native build')
            object_token = source['object']
            if request.get('active'):
                request_build = latest['frame']['callId']
                active = items(fields(source)['active'])
                candidates = [
                    row for row in enters.values()
                    if row['frame']['operation'] == 'Pokemon.getMoveRequestData'
                    and row['frame']['parentCall'] == request_build
                    and pokemon(row['evidence']['subject'])[0] == side
                    and row['frame']['callId'] in returns
                    and len(active) == 1
                    and returns[row['frame']['callId']]['evidence']['result']['object'] == active[0]['object']
                ]
                _need(len(candidates) == 1, 'owner request lacks one original active move-request build')
                source_call = candidates[0]['frame']['callId']
        self.emissions[owner] = EmittedRequest(delivery['generation'], delivery['line_sha256'], call,
                                               source_call, object_token, actionable)

    def boundary(self, snapshot, *, ordinary):
        party = PartyMap(snapshot['battle'], self.species)
        phase = phase_from_native(snapshot, party, original_inputs=ordinary, old_stall=self.history.old_stall)
        observed = snapshot['witness']['nativeObservation']['state']
        relations = []
        for side, owner in enumerate(OWNERS):
            request = snapshot['boundaryRequests'].get(owner)
            current = self.owners[owner].current
            evidence = self.emissions.get(owner)
            _need(current is not None and evidence is not None and current.generation == evidence.generation
                  and current.actionable == phase.requested[side]
                  and _same(document(current.request_line[len(REQUEST_PREFIX):]), request),
                  'passive owner snapshot differs from latest owning delivery')
            locks = LockResults()
            if phase.requested[side]:
                cache_request = observed['sides'][side]['fields']['activeRequest']
                _need(evidence.originally_requested and cache_request['object'] == evidence.request_object
                      and _same(_literal(cache_request), request), 'passive snapshot lost the original request object')
                if request.get('active'):
                    locks = self._locks(side, party, observed, evidence)
            relation = native_action_relation(request, party, owner=side, locks=locks)
            _need('hard_slot_fallback' not in relation, 'Native hard-lock slot was not identified')
            relations.append(relation)
        return phase, party, relations

    def _locks(self, side, party, observed, evidence):
        actor = (side, party.active(side))
        proof = self.history.lock_results.get(actor)
        _need(proof is not None and proof['request_call_id'] == evidence.request_call_id,
              'Native locks do not belong to the delivered active request')
        hard, semi = proof['hard'], proof['semi']
        if semi == 'UNOBSERVED':
            semi = UNKNOWN
        slot = UNKNOWN
        if not hard and semi not in (None, UNKNOWN):
            cache = member(observed, actor)
            encore = fields(cache['volatiles']).get('encore')
            _need(type(encore) is dict, 'semi-lock has no observed live condition')
            found = self.history.encore_slots.get((*actor, encore['object']))
            _need(found is not None, 'semi-lock has no current physical slot witness')
            slot = found['physical_slot']
        return LockResults(hard=hard, semi=semi, semi_slot=slot)


def _select(step, phase, party, relations):
    _need(phase.requested == step.requested, 'Native and frozen requested sides differ')
    selected = []
    for side, action in enumerate(step.actions):
        if not step.requested[side]:
            continue
        relation = relations[side]
        command = relation['physical_commands'][action]
        _need(relation['physical_mask'][action] and type(command) is str, 'frozen lane is illegal in Native')
        aliases = [entry for entry in relation['numeric_relations']
                   if entry['command'] == command and action in entry['physical_lanes']]
        _need(len(aliases) == 1, 'frozen lane has no unique numeric choice relation')
        alias = aliases[0]
        kind = alias['kind']
        move = alias.get('move_id', kind if kind in ('recharge', 'solarbeam', 'struggle') else None)
        selected.append(FixedSelection(side, party.active(side), action, command, alias.get('original_target'), move, kind))
    return tuple(selected)


def _accept(selection, events, phase):
    accepted = {}
    for choice in selection:
        side = choice.owner
        found = [event for event in events if event['kind'] == 'COMMITTED_CHOICE' and event['side'] == side]
        _need(len(found) == 1, 'frozen choice did not produce one original Native commit')
        event = found[0]
        action = event['choice']
        if choice.target is not None:
            expected_kind = 'switch' if phase.kind == 'ordinary' else 'instaswitch'
            _need(action['choice'] == expected_kind and pokemon(action['target']) == (side, choice.target)
                  and pokemon(action['pokemon']) == (side, choice.actor),
                  'committed Native switch target differs from frozen lane')
            installed = [
                item for item in events
                if item['kind'] == 'INSTALLED_CHOSEN_SWITCH' and item['side'] == side
                and item['commit_call_id'] == event['call_id'] and item['original_target'] == choice.target
            ]
            _need(expected_kind == 'switch' or len(installed) == 1,
                  'forced replacement lacks its committed Native installation')
            actual_move = None
        else:
            _need(action['choice'] == 'move' and pokemon(action['pokemon']) == (side, choice.actor)
                  and move_id(action['moveid']) == choice.move, 'Native move commit differs from frozen lane')
            actual_move = (choice.actor, choice.move)
            for dispatch in events:
                if dispatch['kind'] != 'MOVE_USED' or dispatch['side'] != side:
                    continue
                _need(dispatch['original_actor'] == choice.actor and dispatch['move_id'] == choice.move,
                      'Native dispatched a different original actor or move')
                if choice.kind not in ('recharge', 'struggle'):
                    _need(dispatch['physical_slot'] == choice.lane, 'Native dispatched a different physical move slot')
        accepted[side] = (actual_move, event['call_id'])
    return accepted


def _confirmed_choices(step, phase, selection, choices, stdout, events):
    actual = [(event['player'], event['choice'], event['submissionId'])
              for event in stdout if event['type'] == 'choice_ack']
    _need(actual == [(owner, value, step.frame) for owner, value in choices.items()],
          'original Native choice acknowledgments differ from frozen submission')
    accepted = _accept(selection, events, phase)
    if phase.kind == 'ordinary':
        return OrdinaryInputs(step.actions, tuple(accepted[side][0] for side in (0, 1)),
                              tuple(accepted[side][1] for side in (0, 1)))
    return None


def _constructor(snapshot, teams):
    party = PartyMap(snapshot['battle'], [[member['species'] for member in side] for side in teams])
    facts = {(entry['side'], entry['original']): entry for entry in snapshot['witness']['elidedProperties']}
    _need(len(facts) == 12, 'complete live constructor scalars required')
    for side, members in enumerate(teams):
        for original, literal in enumerate(members):
            mon = party.mon(side, original)
            actual = mon['set']
            actual_facts = facts[side, original]
            _need(actual_facts['current'] == party.original_to_current[side][original]
                  and actual_facts['level'] == literal['level'] and actual_facts['happiness'] == literal['happiness']
                  and _same(actual['ivs'], literal['ivs']) and _same(actual['evs'], literal['evs'])
                  and to_id(actual['nature']) == to_id(literal['nature'])
                  and mon.get('gender', '') == ('' if literal['gender'] == 'N' else literal['gender']),
                  'Native constructor differs from literal original team')


class FixedPrefixDispatcher:
    """Use only the admitted action/chance rows on a prepared recorder."""

    def __init__(self, prefix, recorder, chance, *, session):
        _need(type(prefix) is FrozenPrefix and type(session) is str and session,
              'one admitted frozen prefix and session required')
        self.schedule = prefix_schedule(len(prefix.steps))
        binding = document(prefix.binding_wire.decode('utf8'))
        _need(_same(binding['schedule'], [vars(command) for command in self.schedule])
              and recorder.max_commands == len(self.schedule), 'recorder extent differs from the sealed prefix schedule')
        self.prefix = prefix
        self.recorder = recorder
        self.chance = chance
        self.session = session
        self.evidence = DecisionEvidence(prefix, recorder.verifier.source_registry_hash)
        self.command_ids = []

    def _submit(self, wire, phase, request, chance_wire=None, chance_nonce=None):
        verifier = self.recorder.verifier
        command_id = f'{self.prefix.stream_id}/0/{len(self.command_ids)}'
        control = _json(dict(
            schema='pokezero.native_witness_control.v1', run=self.session, stream=self.prefix.stream_id,
            reset=0, command=len(self.command_ids), phase=phase, request=request, nonce=command_id,
            chance_nonce=chance_nonce, previous_receipt_hash=verifier.native_cursor.receipt_hash,
        ))
        visibility = verifier.transport_cursor.visibility_policy_sha256
        transport_fields = dict(
            schema='pokezero.native_transport_control.v2' if visibility else 'pokezero.native_transport_control.v1',
            native_nonce=command_id, stdin_sha256=digest(wire),
            command=document(wire.decode('utf8')), previous_completion_hash=verifier.transport_cursor.previous_completion,
        )
        if visibility is not None:
            transport_fields['visibility_policy_sha256'] = visibility
        transport = _json(transport_fields)
        binding = verifier.hook_cursor.binding(verifier.hook_policy)
        spec = CommandSpec(wire, control, transport, self.prefix.battle_id, hook_admission=binding)
        record = self.recorder.record(spec, chance_control_wire=chance_wire)
        self.command_ids.append(command_id)
        return self.evidence.ingest(self.recorder, record)

    def run(self):
        root = self.recorder.artifacts
        _durable(root / 'frozen-inputs.json', self.prefix.binding_wire)
        for step in self.prefix.steps:
            _durable(root / f'fresh-chance-{step.frame}.json', step.chance_wire)
        _durable(root / 'literal-teams.json', self.prefix.team_wire)
        launch_wire = self.chance.initializing.read_bytes()
        _need(_same(document(launch_wire.decode('utf8')), document(self.prefix.initial_chance_wire.decode('utf8'))),
              'Native initialization chance control differs from sealed input')
        prior_ordinary = None
        ordinary_packet = None
        boundary = None
        battle_id = self.prefix.battle_id
        endpoint = len(self.prefix.steps)
        for command in self.schedule:
            phase_name, frame = command.phase, command.frame
            if phase_name == 'start':
                self._submit(self.prefix.start_wire, phase_name, 0, launch_wire)
                require_resolved(self.recorder.archive_chance_before_retirement(retire=self.chance.initialized))
            elif phase_name == 'snapshot':
                request_id = f'PRE{frame}'
                wire = _json(dict(type=phase_name, battleId=battle_id, requestId=request_id, witnessPassive=True))
                stdout, _ = self._submit(wire, phase_name, frame)
                snapshots = [packet['snapshot'] for packet in stdout if packet['type'] == 'snapshot']
                _need(len(snapshots) == 1, 'passive snapshot missing from original bridge output')
                if frame == 0:
                    _constructor(snapshots[0], self.evidence.teams)
                boundary = self.evidence.boundary(snapshots[0], ordinary=prior_ordinary)
                if frame == endpoint:
                    _need(boundary[0].requested[self.prefix.owner], 'frozen endpoint owner is not requested in Native')
            elif phase_name == 'choices':
                step = self.prefix.steps[frame]
                phase, party, relations = boundary
                selection = _select(step, phase, party, relations)
                _durable(root / f'admitted-PRE{frame}.json', _json(dict(
                    phase=phase.record(), relations=relations, selections=[vars(choice) for choice in selection],
                )))
                packet = self.chance.begin_from_native(
                    phase, step.fresh_chance(), ordinary_packet, stream=self.prefix.stream_id, request=frame,
                )
                choices = {OWNERS[choice.owner]: choice.command for choice in selection}
                wire = _json(dict(type=phase_name, battleId=battle_id, choices=choices, submissionId=frame))
                stdout, events = self._submit(wire, phase_name, frame, self.chance.control.read_bytes(), packet['nonce'])
                require_resolved(self.recorder.archive_chance_before_retirement(retire=self.chance.end))
                confirmed = _confirmed_choices(step, phase, selection, choices, stdout, events)
                if confirmed is not None:
                    prior_ordinary = confirmed
                    ordinary_packet = packet
            else:
                self._submit(_json(dict(type=phase_name, battleId=battle_id)), phase_name, endpoint)
        completion = self.recorder.finish(expected_command_ids=self.command_ids)
        _need(len(self.command_ids) == len(self.schedule) and completion['qualification'] is False,
              'complete raw prefix required')
        return completion
