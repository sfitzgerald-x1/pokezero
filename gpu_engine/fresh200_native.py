"""Strict Native control packets from sealed fresh-corpus operands only.

No JAX import, RNG call, constructor, actor or policy occurs here. A suspended
request uses the captured turn's saved operands and the captured fresh entry
ties, matching the transition's selectors. Unknown Native roles remain fatal.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

CHANCE_FIELDS = ('draws', 'phazing', 'secondary', 'sleep_talk', 'entry_ties', 'initial_tie', 'effect')
DRAW_BOUNDS = np.array([2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2])
EFFECT_BOUNDS = np.array([16, 16, 4, 2, 2])
AUDIT_KINDS = frozenset(('COUPLED', 'COUPLED_INITIAL', 'COUPLED_INITIAL_GENDER',
    'INERT_QUICK_CLAW_ROLL', 'INERT_SELF_DROPS', 'INERT_FORCED_FOE', 'INERT_EACH_EVENT_TIE',
    'INERT_DISABLE_MOVE_TIE', 'INERT_MAGNET_PULL_TIE', 'INERT_TRUANT_TIE', 'INERT_RESIDUAL_TIE', 'INERT_ITEM_TIE'))
MAX_AUDIT_BYTES = 16*1024**2


def _operand(value, shape, bounds, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'iu' or np.any(value < 0) or np.any(value >= bounds):
        raise ValueError('invalid sealed operand: '+name)
    return value.tolist()


def initial_packet(raw, initial_tie, *, row):
    gender = np.asarray(raw.gender);ties = np.asarray(initial_tie)
    if gender.ndim != 3 or ties.shape != (gender.shape[0],) or type(row) is not int or not 0 <= row < len(gender):
        raise ValueError('literal constructor row required')
    genders = _operand(gender[row], (2, 6), 3, 'constructor genders')
    tie = _operand(ties[row], (), 2, 'initial tie')
    return dict(genders=genders, initial_tie=tie, allow_inert_gender=False)


def step_packet(state, chance, *, row, stream, request):
    """Convert all seven saved fields; never derive a missing chance value."""
    if getattr(chance, '_fields', None) != CHANCE_FIELDS or any(getattr(chance, name) is None for name in CHANCE_FIELDS):
        raise ValueError('complete sealed seven-field ChanceInputs required')
    if type(stream) is not int or stream <= 0 or type(request) is not int or request < 0:
        raise ValueError('literal stream/request identity required')
    pending = np.asarray(state.pending_residual)
    if (pending.dtype.kind != 'b' or pending.ndim != 1 or type(row) is not int or not 0 <= row < len(pending)):
        raise ValueError('sealed pending-residual selector/row required')
    shapes = dict(draws=(2, 12), phazing=(2,), secondary=(2,), sleep_talk=(2,),
        entry_ties=(2,), initial_tie=(), effect=(2, 5))
    bounds = dict(draws=DRAW_BOUNDS, phazing=60, secondary=100, sleep_talk=60,
        entry_ties=2, initial_tie=2, effect=EFFECT_BOUNDS)
    # Check fresh operands even when the suspended-turn selector reads saved
    # ones. Malformed fields are never accepted merely because unused here.
    fresh = {}
    for name in CHANCE_FIELDS:
        values = np.asarray(getattr(chance, name))
        if values.shape != (len(pending), *shapes[name]):raise ValueError('sealed chance batch shape differs: '+name)
        fresh[name] = _operand(values[row], shapes[name], bounds[name], name)
    resuming = bool(pending[row]);selected = dict(fresh)
    if resuming:
        for name, saved in (('draws', 'residual_draws'), ('phazing', 'phazing_draws'),
                ('secondary', 'secondary_draws'), ('sleep_talk', 'sleep_talk_draws'), ('effect', 'effect_draws')):
            values = np.asarray(getattr(state, saved))
            if values.shape != (len(pending), *shapes[name]):raise ValueError('sealed saved batch shape differs: '+saved)
            selected[name] = _operand(values[row], shapes[name], bounds[name], saved)
    return dict(schema='pokezero.gpu_chance_tape.v6', nonce=f'{stream}:{request}',
        **selected, resuming=resuming)


def require_resolved(events):
    for event in events:
        if event.get('kind') not in AUDIT_KINDS:raise ValueError('unresolved sealed Native chance: '+json.dumps(event, sort_keys=True))
    return events


def step_packet_from_native(phase,fresh,retained_packet,*,stream,request):
    """Select only ORIGINAL INPUT operands using an independent Native phase.

    All seven fresh fields remain checked during a suspended replacement. The
    old five groups come from the sealed ordinary packet, never device State.
    """
    from .fresh200_native_phase import PhaseProof,retained_inputs
    if (not isinstance(phase,PhaseProof) or phase.kind=='terminal' or type(stream) is not int or stream<=0
            or type(request) is not int or request<0 or type(fresh) is not dict or set(fresh)!=set(CHANCE_FIELDS)):
        raise ValueError('Native phase and complete sealed fresh chance row required')
    shapes=dict(draws=(2,12),phazing=(2,),secondary=(2,),sleep_talk=(2,),entry_ties=(2,),initial_tie=(),effect=(2,5))
    bounds=dict(draws=DRAW_BOUNDS,phazing=60,secondary=100,sleep_talk=60,entry_ties=2,initial_tie=2,effect=EFFECT_BOUNDS)
    selected={name:_operand(fresh[name],shapes[name],bounds[name],name) for name in CHANCE_FIELDS}
    if phase.pending_residual:selected.update(retained_inputs(phase,retained_packet))
    return dict(schema='pokezero.gpu_chance_tape.v6',nonce=f'{stream}:{request}',**selected,resuming=phase.pending_residual)


class SealedChanceControl:
    """Own finite replay files; the caller seals audits even on divergence."""
    def __init__(self, folder, config, initial):
        self.folder = Path(folder);self.folder.mkdir(exist_ok=False)
        launcher = self.folder/'bridge.mjs'
        launcher.write_text('await import('+json.dumps(Path(__file__).with_name('fresh200_bridge.mjs').as_uri())+');\n')
        self.config = replace(config, bridge_path=launcher)
        self.control = self.folder/'chance.json';self.audit = self.folder/'chance-audit.jsonl'
        self.initializing = self.folder/'initializing.json'
        if (set(initial) != {'genders', 'initial_tie', 'allow_inert_gender'} or initial['allow_inert_gender'] is not False):
            raise ValueError('complete sealed initialization required')
        _operand(initial['genders'], (2, 6), 3, 'constructor genders')
        _operand(initial['initial_tie'], (), 2, 'initial tie')
        with self.initializing.open('x') as stream:json.dump(initial, stream, allow_nan=False)

    def take_audit(self):
        if not self.audit.exists():return []
        if self.audit.stat().st_size > MAX_AUDIT_BYTES:raise ValueError('sealed Native audit exceeds bound')
        events = [json.loads(line) for line in self.audit.read_text().splitlines()]
        self.audit.unlink()
        return events

    def initialized(self):
        self.initializing.unlink()
        return self.take_audit()

    def begin(self, state, chance, *, row, stream, request):
        if self.initializing.exists():raise ValueError('initialization audit must precede actions')
        packet = step_packet(state, chance, row=row, stream=stream, request=request)
        with self.control.open('x') as output:json.dump(packet, output, allow_nan=False)
        return packet

    def end(self):
        self.control.unlink()
        return self.take_audit()

    def begin_from_native(self,phase,fresh,retained_packet,*,stream,request):
        if self.initializing.exists():raise ValueError('initialization audit must precede actions')
        packet=step_packet_from_native(phase,fresh,retained_packet,stream=stream,request=request)
        with self.control.open('x') as output:json.dump(packet,output,allow_nan=False)
        return packet
