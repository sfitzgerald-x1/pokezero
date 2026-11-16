"""Remote replay of the trap walls' retained roots and planted negatives; no new games or model forwards.

Wall 1 (a request trap the world reproduces): Shadow Tag and maybeTrapped (#1958's r4a3 roots, from their JSON), r5a's ability-
trapped 5708/28 and Spider Web on us (1823/34). Wall 2 (a partial trap's drawn clock): r5a's opponent wrapped by our Shuckle
(2741/2-4: one, two and three ticks), our own wrapped active (497/44), and the opponent wrapped while Encored (494/116). Planted:
a non-trapping foe ability after map_spec (Shadow Tag: the world no longer traps us; maybeTrapped: it need not), our Mean Look
link removed (the request trap is no longer reproduced), and seven more ticks than the lines show (no duration is left).

usage: python tests/phase_d_trap_walls_replay.py --metadata .. --vocabulary .. --showdown-root .. --node-binary ..
       --expected-metadata-sha256 .. --expected-vocabulary-sha256 .. [--baseline]
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from gpu_engine import phase_d_protocol_world as protocol, combat, transition
from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor, load_inputs

FIXTURES = Path(__file__).parent / 'fixtures' / 'phase_d_trap_walls'


def _set(array, index, value):
    """The array with index set: a host-built world's NumPy leaf stays NumPy, a device leaf becomes a device array again."""
    out = np.array(array, copy=True);out[index] = value
    if isinstance(array, np.ndarray):return out
    import jax.numpy as j
    return j.asarray(out)


def plants(name, owner, vocab, calls):
    """The patches a planted fixture applies (only those whose target exists: the base tree has no trap writers)."""
    foe = 1 - owner
    out = []
    if 'planted_non_trapping_foe' in name:
        original = protocol.map_spec
        ability = 'pressure' if name.startswith('shadow') else 'sandveil'
        def plant(*a, **kw):
            result = original(*a, **kw)
            if result.state is None:return result
            state = result.state;active = int(state.active[0, foe]);calls.append(ability)
            return replace(result, state=state._replace(ability=_set(state.ability, (0, foe, active), vocab['abilities'].index(ability))))
        out.append(patch.object(protocol, 'map_spec', side_effect=plant))
    if 'planted_unlinked_trap' in name and hasattr(protocol, '_write_own_meanlook'):
        original = protocol._write_own_meanlook
        def plant(physical, detached, meanlook, owner_):
            physical = original(physical, detached, meanlook, owner_);calls.append('unlinked')
            return physical._replace(trap_source=_set(physical.trap_source, (0, owner_, int(physical.active[0, owner_])), -1))
        out.append(patch.object(protocol, '_write_own_meanlook', side_effect=plant))
    if 'planted_tick_count' in name and hasattr(protocol, '_wrap_facts'):
        original = protocol._wrap_facts
        def plant(lines):
            calls.append('ticks')
            return tuple(None if f is None else dict(f, k=f['k'] + 7, upkeeps=f['upkeeps'] + 7) for f in original(lines))
        out.append(patch.object(protocol, '_wrap_facts', side_effect=plant))
    return out


def world_facts(name, state, owner, vocab, table, diagnostics):
    """The per-world facts a fixture pins: wall 1 the foe's ability and our trap state and switch lanes; wall 2 the holder's
    drawn duration, its trapper and the holder's switch lanes (the drawn count also inside the world's logged support)."""
    foe = 1 - owner;oa, fa = int(state.active[0, owner]), int(state.active[0, foe])
    mask = np.asarray(transition.action_mask(state, table))[0]
    species = lambda side, at:vocab['species'][int(state.species[0, side, at])]
    facts = dict(own=species(owner, oa), foe=species(foe, fa))
    if name.startswith(('shadow', 'maybe', 'ability', 'meanlook')):
        facts.update(foe_ability=vocab['abilities'][int(state.ability[0, foe, fa])], own_trapped_flag=bool(state.trapped[0, owner]),
                     own_trap_source=int(state.trap_source[0, owner, oa]), own_switch_lanes=mask[owner, 4:].tolist())
    if name.startswith('wrap'):
        holder = owner if name.startswith('wrap_own') else foe;at = int(state.active[0, holder])
        row = next(x for x in diagnostics if x['kind'] == 'PUBLIC_WRAP_WORLD_POSTERIOR')
        counter = int(state.partial_trap_counter[0, holder, at])
        if counter not in row['supports'][holder]:raise AssertionError((counter, row))
        facts.update(holder='own' if holder == owner else 'foe', support=list(row['supports'][holder]), counter=counter,
                     source=int(state.partial_trap_source[0, holder, at]), trapper_active=int(state.active[0, 1 - holder]),
                     holder_switch_lanes=mask[holder, 4:].tolist())
    return facts


def run_fixture(constructor, name, data, baseline):
    """(outcome, per-world facts or None, planted calls) of one fixture's request through ProtocolConstructor.prepare."""
    request = data['request']
    if hashlib.sha256(''.join(x + '\n' for x in request['lines']).encode()).hexdigest() != data['prefix_sha256']:
        raise AssertionError('fixture prefix')
    owner = int(request['owner'][1]) - 1;vocab = constructor.metadata['vocabulary'];calls = []
    made = error = None
    with ExitStack() as stack:
        for p in plants(name, owner, vocab, calls):stack.enter_context(p)
        try:_, made, _ = constructor.prepare(request, request['decision_index'])
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:error = type(exc).__name__ + ':' + str(exc)
    worlds = () if made is None else made.worlds
    outcome = dict(built=sum(w.state is not None for w in worlds), error=error, refusals=dict(Counter(r for w in worlds for r in w.refusals)))
    if baseline or not outcome['built']:return outcome, None, calls, made
    table = SimpleNamespace(combat=combat.from_metadata(constructor.metadata))
    per = []
    for w in worlds:
        checks = next(x['checks'] for x in made.mapper_diagnostics if x['world'] == w.index)
        per.append(dict(world=w.index, **world_facts(name, w.state, owner, vocab, table, checks)))
    return outcome, per, calls, made


class TrapWallsReplay(unittest.TestCase):
    constructor = None
    baseline = False
    outcomes = []

    def setUp(self):
        shared = type(self).constructor
        self.constructor = ProtocolConstructor(catalog=shared.catalog, source=shared.source, metadata=shared.metadata,
            owning_config=shared.owning_config, count=16, seed=20261103, functions=shared.functions)

    def replay(self, name):
        data = json.loads((FIXTURES / (name + '.json')).read_text())
        outcome, per, calls, made = run_fixture(self.constructor, name, data, self.baseline)
        self.outcomes.append(dict(fixture=name, **outcome))
        self.assertEqual(outcome, data['base' if self.baseline else 'patched'])
        if self.baseline:return
        if 'planted' in name:self.assertEqual(len(calls), 16 if 'non_trapping' in name or 'unlinked' in name else 1, calls)
        self.assertEqual(len(made.worlds), 16)
        if outcome['built']:self.assertEqual(per, data['checks']['per_world'])

    # Wall 1
    def test_shadow_tag(self):self.replay('shadow_tag_trapped')
    def test_shadow_tag_wrong_ability_refused(self):self.replay('shadow_tag_trapped_planted_non_trapping_foe')
    def test_maybe_trapped(self):self.replay('maybe_trapped_614_0')
    def test_maybe_trapped_non_trapping_ability(self):self.replay('maybe_trapped_614_0_planted_non_trapping_foe')
    def test_ability_trapped_r5a(self):self.replay('ability_trapped_5708_28')
    def test_spider_web_on_us(self):self.replay('meanlook_own_1823_34')
    def test_spider_web_unlinked_refused(self):self.replay('meanlook_own_1823_34_planted_unlinked_trap')
    # Wall 2
    def test_opponent_wrapped_one_tick(self):self.replay('wrap_opp_2741_2')
    def test_opponent_wrapped_two_ticks(self):self.replay('wrap_opp_2741_3')
    def test_opponent_wrapped_three_ticks(self):self.replay('wrap_opp_2741_4')
    def test_opponent_wrapped_seven_more_ticks_refused(self):self.replay('wrap_opp_2741_4_planted_tick_count')
    def test_our_wrapped_active(self):self.replay('wrap_own_497_44')
    def test_opponent_wrapped_and_encored(self):self.replay('wrap_opp_encore_494_116')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'vocabulary', 'showdown-root', 'node-binary', 'expected-metadata-sha256', 'expected-vocabulary-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    catalog, source, metadata, config = load_inputs(args)
    TrapWallsReplay.constructor = ProtocolConstructor(catalog=catalog, source=source, metadata=metadata, owning_config=config, count=16, seed=20261103)
    TrapWallsReplay.baseline = args.baseline
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrapWallsReplay))
    print(json.dumps(dict(passed=result.wasSuccessful(), outcomes=TrapWallsReplay.outcomes), sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
