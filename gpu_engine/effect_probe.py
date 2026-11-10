"""Per-move two-engine probe: one authored scenario through JAX and Showdown, same chance tape.

A scenario (JSON, schema SCENARIO_SCHEMA) names two authored teams, a per-request
action script and the outcome axes of one effect. The probe takes the Cartesian
product of the axes; each case is one lane of the JAX engine and one Showdown
f76228a1 battle (gpu_engine/effect_probe.cjs). Every request step sends the
same semantic chance tape (pokezero.gpu_chance_tape.v2) to both engines, then
compares the post-step states on the fields of the fidelity-patch design 5.1.
Fields whose engine leaf does not exist yet (perish_counter and so on) are
always compared, against the engine's neutral value (0, -1 or False), so a
scenario reports MISMATCH until the PR that adds the state; a scenario's
`require_fields` names engine fields that must exist. Unresolved chance roles,
and any unaudited start-of-battle randomness (lead speed ties, gender, Quick
Claw), count as failures. Output: OUT/<move>.json per scenario with a per-case diff list, and
OUT/summary.json.

Pool rows come from gpu_engine/team_rows.cjs (generate_teams.cjs's row builder)
and the engine tables from the real pool metadata, so vocabulary ids match.
All paths are command-line arguments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_SCHEMA = 'pokezero.effect_probe.scenario.v1'
RESULT_SCHEMA = 'pokezero.effect_probe.result.v1'
TAPE_SCHEMA = 'pokezero.gpu_chance_tape.v2'
SLEEP_CLAUSE_FORMAT = 'gen3customgame@@@sleepclausemod'
STATUS = ['', 'par', 'brn', 'psn', 'tox', 'slp', 'frz']
DRAW_BOUNDS = (2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2)
EFFECT_BOUNDS = (16, 16, 4, 2)
# Neutral tape: no thaw, no full paralysis, hits, no crit, the minimum (85%)
# damage roll (draws[5]=0 -> random(16)=15), no secondary, protect succeeds,
# side 0 wins ties. Scenarios override fields.
DEFAULT_ROW = [0, 99, 99, 0, 15, 0, 0, 0, 0, 0, 0, 0]
DEFAULT_TAPE = dict(draws=[list(DEFAULT_ROW), list(DEFAULT_ROW)], secondary=[99, 99], sleep_talk=[0, 0],
                    effect=[[15, 0, 0, 0], [15, 0, 0, 0]])
# Engine-only auxiliary inputs (no Showdown role); kept explicit and in-range.
DEFAULT_AUX = dict(phazing=[0, 0], entry_ties=[0, 0], initial_tie=0)
BOUNDS = dict(draws=DRAW_BOUNDS, effect=EFFECT_BOUNDS, secondary=100, sleep_talk=60, phazing=60, entry_ties=2, initial_tie=2)
DESTINY_BOND_NAME = 'DESTINY_BOND'
# Design 5.1 fields an engine may not have yet: Showdown accessor and the engine's
# neutral value used while the leaf is absent.
OPTIONAL = {
    'perish_counter': (lambda v: v['perishsong'] or 0, 0),
    'yawn_counter': (lambda v: v['yawn'] or 0, 0),
    'partial_trap_counter': (lambda v: (v['partiallytrapped'] or {}).get('duration') or 0, 0),
    'partial_trap_source': (lambda v: -1 if v['partiallytrapped'] is None else v['partiallytrapped']['source'], -1),
    'recharge_counter': (lambda v: v['mustrecharge'] or 0, 0),
}
# require_fields names: the OPTIONAL leaves, item_knocked, destiny_bond (state.DESTINY_BOND)
# and effect (ChanceInputs.effect).
REQUIRABLE = tuple(OPTIONAL) + ('item_knocked', 'destiny_bond', 'effect')


# ---------------------------------------------------------------------------
# Scenario, cases and tapes (pure).

def load_scenario(path):
    scenario = json.loads(Path(path).read_text())
    if scenario.get('schema') != SCENARIO_SCHEMA:raise ValueError(f'{path}: schema {SCENARIO_SCHEMA} required')
    for key in ('move', 'teams', 'script'):
        if key not in scenario:raise ValueError(f'{path}: missing {key}')
    if len(scenario['teams']) != 2 or not all(len(t) == 6 for t in scenario['teams']):
        raise ValueError(f'{path}: two six-member teams required (the engine state is [B,2,6])')
    if not scenario['script'] or any(len(step) != 2 for step in scenario['script']):
        raise ValueError(f'{path}: script steps are [p1 choice, p2 choice]')
    names = [axis['name'] for axis in scenario.get('axes', [])]
    if len(set(names)) != len(names):raise ValueError(f'{path}: duplicate axis names')
    for axis in scenario.get('axes', []):
        if not axis.get('values') or not axis.get('field'):raise ValueError(f'{path}: axis {axis.get("name")} needs field and values')
        for value in axis['values']:_check_value(axis['field'], value)
    for field, value in scenario.get('tape', {}).items():_check_value(field, value)
    for step, fields in scenario.get('tape_steps', {}).items():
        int(step)
        for field, value in fields.items():_check_value(field, value)
    unknown = sorted(set(scenario.get('require_fields', ())) - set(REQUIRABLE))
    if unknown:raise ValueError(f'{path}: unknown require_fields {unknown}; choose from {REQUIRABLE}')
    scenario.setdefault('format', SLEEP_CLAUSE_FORMAT)
    scenario.setdefault('seed', [1, 2, 3, 4])
    scenario.setdefault('axes', [])
    scenario.setdefault('expect', dict(exact=True))
    return scenario


def _parse_field(field):
    parts = field.split('.')
    if parts[0] not in BOUNDS:raise ValueError(f'unknown tape field {field}')
    return parts[0], [int(x) for x in parts[1:]]


def _check_value(field, value):
    name, index = _parse_field(field)
    bound = BOUNDS[name]
    expected = {'draws': 2, 'effect': 2, 'initial_tie': 0}.get(name, 1)
    if len(index) != expected:raise ValueError(f'{field}: {expected} indices required')
    if name in ('draws', 'effect'):
        if index[0] not in (0, 1) or not 0 <= index[1] < len(bound):raise ValueError(f'{field}: out of range')
        bound = bound[index[1]]
    elif index and index[0] not in (0, 1):raise ValueError(f'{field}: side must be 0 or 1')
    if type(value) is not int or not 0 <= value < bound:raise ValueError(f'{field}={value}: outside [0,{bound})')


def enumerate_cases(scenario):
    """Cartesian product of the axes, in axis order; one case when there are none."""
    axes = scenario.get('axes', [])
    cases = []
    for values in itertools.product(*(axis['values'] for axis in axes)):
        assignment = {axis['name']: value for axis, value in zip(axes, values)}
        label = ','.join(f'{k}={v}' for k, v in assignment.items()) or 'base'
        cases.append(dict(case_id=label, assignment=assignment))
    return cases


def _set(tape, field, value):
    name, index = _parse_field(field)
    if not index:tape[name] = value
    elif len(index) == 1:tape[name][index[0]] = value
    else:tape[name][index[0]][index[1]] = value


def step_inputs(scenario, case, step):
    """(Showdown tape body, engine auxiliary inputs) for one ordinary step of one case."""
    tape = copy.deepcopy(DEFAULT_TAPE);aux = copy.deepcopy(DEFAULT_AUX)
    for field, value in scenario.get('tape', {}).items():_set(tape if _parse_field(field)[0] in tape else aux, field, value)
    for field, value in scenario.get('tape_steps', {}).get(str(step), {}).items():
        _set(tape if _parse_field(field)[0] in tape else aux, field, value)
    for axis in scenario.get('axes', []):
        steps = axis.get('steps')
        if steps is None or step in steps:
            _set(tape if _parse_field(axis['field'])[0] in tape else aux, axis['field'], case['assignment'][axis['name']])
    return tape, aux


def engine_action(text, engine_active):
    """Script choice -> engine action. `switch K` names the authored 1-based party position."""
    if not text:return 0
    kind, value = text.split(None, 1)
    if kind == 'move':
        slot = int(value) - 1
        if not 0 <= slot < 4:raise ValueError(f'bad move choice {text}')
        return slot
    if kind == 'switch':
        target = int(value) - 1
        if not 0 <= target < 6 or target == engine_active:return -1   # invalid for the engine
        return 4 + target - (1 if target > engine_active else 0)
    raise ValueError(f'unsupported choice {text}')


# ---------------------------------------------------------------------------
# Comparison (pure: numpy lane views against Showdown snapshots).

def _name(vocab, kind, index):
    index = int(index)
    return '' if index == 0 else vocab[kind][index]


def requested_seats(state):
    hp = np.take_along_axis(np.asarray(state['hp']), np.asarray(state['active'])[..., None], axis=2)[..., 0]
    forced = np.asarray(state['forced_switch']) | (hp <= 0)
    return (~forced.any(axis=1)[:, None] | forced) & ~np.asarray(state['done'])[:, None]


def compare_lane(view, snapshot, vocab, *, mask, destiny_bond=None):
    """Differences between one engine lane (numpy dict) and one Showdown snapshot.

    Every design 5.1 field is compared; an engine leaf that does not exist yet is
    read as its neutral value (OPTIONAL, item_knocked False, Destiny Bond False)."""
    diffs = []
    def check(field, showdown, engine):
        if showdown != engine:diffs.append(dict(field=field, showdown=showdown, engine=engine))
    for side in range(2):
        sd = snapshot['sides'][side];active = int(view['active'][side])
        check(f'{side}:active', sd['active'], active)
        for slot, mon in enumerate(sd['mons']):
            p = f'{side}:{slot}:'
            hp = int(view['hp'][side, slot])
            check(p + 'hp', mon['hp'], hp)
            check(p + 'fainted', mon['fainted'], hp <= 0)
            alive = mon['hp'] > 0 or hp > 0
            if alive:check(p + 'status', mon['status'], STATUS[int(view['status'][side, slot])])
            if alive and mon['status'] == 'slp':
                check(p + 'status_counter', mon['status_time'], int(view['status_counter'][side, slot]))
            check(p + 'item', mon['item'], _name(vocab, 'items', view['item'][side, slot]))
            check(p + 'item_knocked', mon['item_knocked'], bool(view['item_knocked'][side, slot]) if 'item_knocked' in view else False)
            check(p + 'pp', mon['pp'], [int(x) for x in view['pp'][side, slot, :len(mon['pp'])]])
        if sd['active'] != active or sd['active'] is None:continue
        # Active-only fields: Showdown clears volatiles and stages on switch-out.
        mon = sd['mons'][active];v = mon['volatiles'];p = f'{side}:{active}:'
        if mon['hp'] > 0 and mon['status'] == 'tox':
            check(p + 'status_counter', mon['tox_stage'], int(view['status_counter'][side, active]))
        check(p + 'boosts', mon['boosts'], [int(x) for x in view['boosts'][side, active]])
        check(p + 'confusion_counter', v['confusion'] or 0, int(view['confusion_counter'][side, active]))
        check(p + 'choice_move', v['choicelock'] or '', _name(vocab, 'moves', view['choice_move'][side, active]))
        check(p + 'encore_counter', (v['encore'] or {}).get('duration') or 0, int(view['encore_counter'][side, active]))
        check(p + 'encore_move', (v['encore'] or {}).get('move') or '', _name(vocab, 'moves', view['encore_move'][side, active]))
        check(p + 'substitute_hp', v['substitute'] or 0, int(view['substitute_hp'][side, active]))
        for name, (read, neutral) in OPTIONAL.items():
            check(p + name, read(v), int(view[name][side, active]) if name in view else neutral)
        check(p + 'destiny_bond', v['destinybond'],
              bool(int(view['volatile'][side, active]) & destiny_bond) if destiny_bond is not None else False)
    for side in range(2):
        check(f'{side}:mask', snapshot['sides'][side]['mask'], [bool(x) for x in mask[side]])
    done = bool(view['done'])
    check('winner', snapshot['winner'] if snapshot['ended'] else None, int(view['winner']) if done else None)
    return diffs


def field_kind(field):
    return field.rsplit(':', 1)[-1]


def judge(scenario, cases):
    """Did the run meet the scenario's stated expectation?"""
    expect = scenario.get('expect', dict(exact=True))
    if expect.get('exact'):
        return all(c['status'] == 'EXACT' for c in cases), 'every case exact'
    allowed = set(expect.get('mismatch_fields', ()))
    exact_when = expect.get('exact_cases', [])
    reasons = []
    for c in cases:
        must_exact = any(all(c['assignment'].get(k) == v for k, v in rule.items()) for rule in exact_when)
        if must_exact and c['status'] != 'EXACT':reasons.append(f"{c['case_id']}: expected EXACT, got {c['status']}")
        if not must_exact:
            if c['status'] != 'MISMATCH':reasons.append(f"{c['case_id']}: expected MISMATCH, got {c['status']}")
            stray = sorted({field_kind(d['field']) for s in c['steps'] for d in s['diffs']} - allowed)
            if stray:reasons.append(f"{c['case_id']}: unexpected fields {stray}")
    return not reasons, '; '.join(reasons) or 'expected mismatches only'


# ---------------------------------------------------------------------------
# Engines.

class Showdown:
    def __init__(self, node, showdown_root):
        self.process = subprocess.Popen([node, str(ROOT / 'gpu_engine/effect_probe.cjs'), str(showdown_root)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def ask(self, **command):
        self.process.stdin.write(json.dumps(command) + '\n');self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:raise RuntimeError('effect_probe.cjs exited')
        reply = json.loads(line)
        if not reply.pop('ok'):raise RuntimeError(reply['error'])
        return reply

    def close(self):
        self.process.stdin.close();self.process.wait(timeout=30)


def load_pool(path, expected_sha):
    blob = Path(path).read_bytes();sha = hashlib.sha256(blob).hexdigest()
    if expected_sha and sha != expected_sha:raise ValueError(f'pool sha {sha} differs from {expected_sha}')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata_utf8'].tobytes())
    return metadata, sha


class Engine:
    def __init__(self, metadata, lanes):
        import jax
        from gpu_engine import transition, state as state_module
        self.jax, self.transition = jax, transition
        self.table = transition.tables_from_metadata(metadata);self.lanes = lanes
        self.step = jax.jit(lambda s, a, c: transition.step_with_chance(s, a, self.table, c))
        self.mask = jax.jit(lambda s: transition.action_mask(s, self.table))
        self.destiny_bond = getattr(state_module, DESTINY_BOND_NAME, None)
        self.optional = tuple(n for n in OPTIONAL if n in state_module.BattleState._fields)
        self.fields = set(self.optional) | ({'item_knocked'} if 'item_knocked' in state_module.BattleState._fields else set())
        self.fields |= ({'destiny_bond'} if self.destiny_bond is not None else set())
        self.fields |= ({'effect'} if 'effect' in transition.ChanceInputs._fields else set())
        self.has_effect = 'effect' in transition.ChanceInputs._fields

    def start(self, rows):
        from gpu_engine.team_pool import FIELDS
        pool = {}
        for name in FIELDS:
            teams = []
            for team in rows:
                if len(team) != 6:raise ValueError('the engine needs six-member teams')
                teams.append([m[name] for m in team])
            pool[name] = np.asarray(teams)
        state = self.transition.state_from_pool(pool, np.tile([[0, 1]], (self.lanes, 1)), seed=0)
        return self.jax.block_until_ready(self.transition.initialize(state, self.table))

    def advance(self, state, actions, tapes, auxes):
        import jax.numpy as j
        stack = lambda key, dtype=j.int32: j.asarray(np.asarray([t[key] for t in tapes]), dtype)
        fields = dict(draws=stack('draws'), secondary=stack('secondary'), sleep_talk=stack('sleep_talk'),
            phazing=j.asarray(np.asarray([a['phazing'] for a in auxes]), j.int32),
            entry_ties=j.asarray(np.asarray([a['entry_ties'] for a in auxes]), j.int32),
            initial_tie=j.asarray(np.asarray([a['initial_tie'] for a in auxes]), j.int32))
        if self.has_effect:fields['effect'] = stack('effect')
        chance = self.transition.ChanceInputs(**fields)
        result = self.step(state, j.asarray(np.asarray(actions), j.int32), chance)
        return self.jax.block_until_ready(result)

    def views(self, state):
        arrays = {name: np.asarray(getattr(state, name)) for name in state._fields if name != 'keys'}
        mask = np.asarray(self.mask(state)) & requested_seats(arrays)[..., None]
        return arrays, mask


def lane_view(arrays, lane):
    return {name: value[lane] for name, value in arrays.items()}


# ---------------------------------------------------------------------------

def run_scenario(scenario, engine, showdown, metadata):
    missing = sorted(set(scenario.get('require_fields', ())) - engine.fields)
    if missing:
        return dict(schema=RESULT_SCHEMA, move=scenario['move'], format=scenario['format'], cases=0,
            status_counts={'MISSING_ENGINE_FIELDS': 1}, missing_engine_fields=missing, expectation=scenario.get('expect'),
            expectation_met=False, expectation_detail=f'engine lacks required fields {missing}', results=[])
    prepared = showdown.ask(cmd='prepare', teams=scenario['teams'])
    if prepared['vocabulary'] != metadata['vocabulary']:raise ValueError('probe vocabulary differs from the pool metadata')
    cases = enumerate_cases(scenario)
    if len(cases) > engine.lanes:raise ValueError(f"{scenario['move']}: {len(cases)} cases exceed {engine.lanes} lanes")
    vocab = metadata['vocabulary']
    started = showdown.ask(cmd='start', format=scenario['format'], seed=scenario['seed'], cases=len(cases), sets=prepared['sets'])
    state = engine.start(prepared['rows'])
    arrays, mask = engine.views(state)
    records = []
    # Start-of-battle randomness runs on the seeded PRNG but is audited: anything
    # beyond inert gender and the no-Quick-Claw roll (a lead Speed tie, a real
    # Quick Claw, gender that matters) makes the case unresolved.
    initial_random = [e for e in started['audit'] if e['kind'] in ('INITIAL_RANDOM', 'UNRESOLVED')]
    for i, case in enumerate(cases):
        diffs = compare_lane(lane_view(arrays, i), started['snapshots'][i], vocab, mask=mask[i],
            destiny_bond=engine.destiny_bond)
        records.append(dict(case_id=case['case_id'], assignment=case['assignment'], status=None, initial_diffs=diffs,
            initial_audit=started['audit'], steps=[], last_tape=None, last_aux=None, live=True))
        if initial_random:records[-1].update(status='UNRESOLVED_INITIAL_CHANCE', live=False)
        elif diffs:records[-1].update(status='INITIAL_MISMATCH', live=False)
    for step, script in enumerate(scenario['script']):
        if not any(r['live'] for r in records):break
        pending = np.asarray(state.pending_residual)
        tapes, auxes, actions, choices = [], [], [], []
        active = np.asarray(state.active)
        for i, record in enumerate(records):
            body, aux = step_inputs(scenario, cases[i], step)
            if pending[i] and record['last_tape'] is not None:
                body, aux = record['last_tape'], record['last_aux']   # a suspended turn replays its block
            else:
                record['last_tape'], record['last_aux'] = body, aux
            tapes.append(dict(schema=TAPE_SCHEMA, nonce=f"{scenario['move']}:{i}:{step}", **body));auxes.append(aux)
            actions.append([engine_action(script[s], int(active[i, s])) for s in range(2)])
            choices.append(list(script))
        for lane in range(len(cases), engine.lanes):
            tapes.append(tapes[0]);auxes.append(auxes[0]);actions.append(actions[0])
        if any(a < 0 for row in actions[:len(cases)] for a in row):
            for i, row in enumerate(actions[:len(cases)]):
                if records[i]['live'] and min(row) < 0:records[i].update(status='SCRIPT_INVALID_FOR_ENGINE', live=False)
            actions = [[max(0, a) for a in row] for row in actions]
        live = [r['live'] for r in records]
        reply = showdown.ask(cmd='step', tapes=tapes[:len(cases)], choices=choices[:len(cases)], live=live)
        result = engine.advance(state, actions, tapes, auxes);state = result.state
        arrays, mask = engine.views(state);invalid = np.asarray(result.invalid_actions)
        for i, record in enumerate(records):
            if not live[i]:continue
            sd = reply['results'][i]
            unresolved = [e for e in sd['audit'] if e['kind'] == 'UNRESOLVED']
            entry = dict(step=step, choices=choices[i], engine_actions=actions[i], tape=tapes[i],
                coupled=[(e['role'], e['side'], e['raw']) for e in sd['audit'] if e['kind'] == 'COUPLED'],
                inert=[e['kind'] for e in sd['audit'] if e['kind'].startswith('INERT')], diffs=[])
            record['steps'].append(entry)
            if unresolved:
                entry['unresolved'] = unresolved;record.update(status='UNRESOLVED', live=False);continue
            if sd['ignored']:entry['ignored_choices'] = sd['ignored']
            if sd['error']:
                entry['showdown_error'] = sd['error'];record.update(status='SHOWDOWN_ERROR', live=False);continue
            if invalid[i]:
                record.update(status='ENGINE_INVALID_ACTION', live=False);continue
            entry['diffs'] = compare_lane(lane_view(arrays, i), sd['snapshot'], vocab, mask=mask[i],
                destiny_bond=engine.destiny_bond)
            if entry['diffs']:entry['showdown_log_tail'] = sd['snapshot']['log_tail']
            if sd['snapshot']['ended'] or bool(arrays['done'][i]):record['live'] = False
    for record in records:
        if record['status'] is None:
            record['status'] = 'MISMATCH' if any(s['diffs'] for s in record['steps']) else 'EXACT'
        mismatched = [s['step'] for s in record['steps'] if s['diffs']]
        record['first_mismatch_step'] = mismatched[0] if mismatched else None
        record['diff_fields'] = sorted({field_kind(d['field']) for s in record['steps'] for d in s['diffs']})
        for key in ('live', 'last_tape', 'last_aux'):record.pop(key)
    met, reason = judge(scenario, records)
    counts = {}
    for record in records:counts[record['status']] = counts.get(record['status'], 0) + 1
    return dict(schema=RESULT_SCHEMA, move=scenario['move'], format=scenario['format'], cases=len(records),
        status_counts=counts, expectation=scenario.get('expect'), expectation_met=met, expectation_detail=reason,
        engine_fields=sorted(engine.fields), neutral_engine_fields=sorted(set(REQUIRABLE) - engine.fields - {'effect'}),
        require_fields=scenario.get('require_fields', []),
        effect_stream_in_engine=engine.has_effect, sets=prepared['sets'], script=scenario['script'],
        axes=scenario.get('axes', []), results=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scenarios', nargs='+', help='scenario JSON files')
    parser.add_argument('--team-pool', required=True, help='pool .npz whose metadata defines the vocabulary and tables')
    parser.add_argument('--expected-pool-sha', default=None)
    parser.add_argument('--showdown-root', required=True)
    parser.add_argument('--node', default='node')
    parser.add_argument('--out', required=True, help='output directory (must not exist)')
    parser.add_argument('--lanes', type=int, default=16, help='engine batch (one lane per case; padded)')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():raise FileExistsError(f'{out} exists')
    scenarios = [load_scenario(p) for p in args.scenarios]
    moves = [s['move'] for s in scenarios]
    if len(set(moves)) != len(moves):raise ValueError('one scenario per move')
    metadata, pool_sha = load_pool(args.team_pool, args.expected_pool_sha)
    engine = Engine(metadata, args.lanes);showdown = Showdown(args.node, args.showdown_root)
    out.mkdir(parents=True)
    source = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'gpu_engine', 'src'], capture_output=True, text=True).stdout
    summary = dict(schema=RESULT_SCHEMA + '.summary', source_commit=source, source_dirty=bool(dirty.strip()), pool_sha256=pool_sha,
        showdown_root=str(args.showdown_root), scenarios={})
    try:
        for path, scenario in zip(args.scenarios, scenarios):
            result = run_scenario(scenario, engine, showdown, metadata)
            result.update(scenario_path=str(path), source_commit=source, pool_sha256=pool_sha)
            (out / f"{scenario['move']}.json").write_text(json.dumps(result, indent=1, default=str) + '\n')
            summary['scenarios'][scenario['move']] = dict(cases=result['cases'], status_counts=result['status_counts'],
                expectation_met=result['expectation_met'], detail=result['expectation_detail'],
                diff_fields=sorted({f for r in result['results'] for f in r['diff_fields']}))
            print(json.dumps(dict(move=scenario['move'], **summary['scenarios'][scenario['move']])), flush=True)
    finally:
        showdown.close()
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return 0 if all(s['expectation_met'] for s in summary['scenarios'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
