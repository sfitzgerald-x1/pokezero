import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fresh200_native_conditions_test import ability, living_cache
from fresh200_native_ledger_test import Trace, arr, effect, ledger, mon, obj
from gpu_engine import fresh200_native_conditions as conditions
from gpu_engine.fresh200_native_phase import PhaseProof


def root199_boundary(*, writer=True, same_move=True, preserve_cache=True,
                     competing=False, invalidator=None):
    trace = Trace()
    victim, source, replacement = mon(0, 0), mon(1, 0), mon(1, 2)
    before = living_cache()
    before['fields'] = dict(requestState='move')
    before['sides'][0]['pokemon'][0]['fields'].update(hp=254, fainted=False, faintQueued=False,
        maybeTrapped=False, types=['Steel', 'Rock'], ability='sturdy')
    before['sides'][1]['pokemon'][0]['fields'].update(hp=183, fainted=False, ability='arenatrap')
    before['sides'][1]['pokemon'][2]['fields'].update(hp=310, fainted=False, faintQueued=False, ability='rockhead')

    hidden = copy.deepcopy(before)
    hidden['sides'][0]['pokemon'][0]['fields']['trapped'] = 'hidden'
    trace.enter('Pokemon.tryTrap', victim, (True,), before)
    name = 'arenatrap' if writer else 'magnetpull'
    trace.rows[-1]['evidence']['context'] = dict(effect=ability(name),
        effectState=obj(11000, id=name, target=source))
    trace.leave(True, after=hidden)

    trapped = copy.deepcopy(hidden)
    trapped['sides'][0]['pokemon'][0]['fields']['maybeTrapped'] = True
    if competing:
        trace.enter('Pokemon.tryTrap', victim, (True,), before=trapped)
        trace.rows[-1]['evidence']['context'] = dict(effect=ability('arenatrap'),
            effectState=obj(11000, id='arenatrap', target=source))
        trace.leave(True, after=trapped)
    trace.enter('Battle.commitChoices', before=trapped)
    trace.leave()
    switch = obj(11001, choice='switch', pokemon=source, target=replacement)
    trapped['sides'][1]['fields']['choice'] = obj(11002, actions=arr(11003, switch))
    switched = copy.deepcopy(trapped)
    switched['sides'][1]['fields']['active'] = arr(6001, replacement)
    trace.enter('Side.commitChoices', dict(kind='Side', object=11, n=1), before=trapped)
    trace.leave(after=trapped)
    trace.enter('Battle.runAction', arguments=(switch,), before=trapped)
    trace.enter('BattleActions.switchIn', arguments=(replacement, 0), before=trapped)
    trace.leave(True, after=switched)
    trace.leave()

    move = effect('explosion' if same_move else 'selfdestruct')
    queued = copy.deepcopy(switched)
    queued['sides'][0]['pokemon'][0]['fields'].update(hp=0, faintQueued=True)
    queued['sides'][1]['pokemon'][2]['fields']['hp'] = 183
    trace.enter('BattleActions.runMove', arguments=(move, victim, 0), before=switched)
    trace.enter('Pokemon.faint', victim, (victim, move), before=switched)
    trace.leave(254, after=queued)
    trace.enter('Pokemon.faint', victim, (victim, move), before=queued)
    trace.leave(0, after=queued)
    trace.leave()
    cleared = copy.deepcopy(queued)
    if not preserve_cache:
        cleared['sides'][0]['pokemon'][0]['fields']['trapped'] = False
    fainted = copy.deepcopy(queued)
    fainted['sides'][0]['pokemon'][0]['fields']['fainted'] = True
    trace.enter('Battle.faintMessages', before=queued)
    trace.enter('Pokemon.clearVolatile', victim, (False,), before=queued)
    trace.leave(after=cleared)
    trace.leave(False, after=fainted)
    trace.enter('Battle.faintMessages', before=fainted)
    trace.leave(False, after=fainted)
    if invalidator == 'refresh':
        trace.enter('Battle.endTurn', before=fainted)
        trace.leave(after=fainted)
    elif invalidator == 'commit':
        forced = copy.deepcopy(fainted)
        forced['fields']['requestState'] = 'switch'
        trace.enter('Battle.commitChoices', before=forced)
        trace.leave()
    elif invalidator == 'later-install':
        away = copy.deepcopy(fainted)
        away['sides'][1]['fields']['active'] = arr(6001, source)
        trace.enter('BattleActions.switchIn', arguments=(source, 0), before=fainted)
        trace.leave(True, after=away)
        trace.enter('BattleActions.switchIn', arguments=(replacement, 0), before=away)
        trace.leave(True, after=fainted)
    elif invalidator == 'failed-try-trap':
        trace.enter('Pokemon.tryTrap', victim, (True,), before=fainted)
        trace.leave(False, after=fainted)

    phase = PhaseProof('replacement-before-residual', 1, (1, 0), (True, False), True, -1,
                       (0, 0), (0, 0), len(trace.rows))
    history = ledger().consume(trace.rows).snapshot()
    relations = ({'kind': 'forced-replacement', 'physical_mask': [False] * 4 + [True] * 5},
                 {'kind': 'inactive'})
    return SimpleNamespace(history=history, phase=phase, relations=relations), fainted


def trapped_projection(boundary, observed):
    active = (0, 2)
    party = SimpleNamespace(
        active=lambda side: active[side],
        mon=lambda side, slot: dict(types=['Normal'], ability='rockhead',
                                    hp=observed['sides'][side]['pokemon'][slot]['fields']['hp']),
    )
    layout = conditions._layout()
    values = {name: np.zeros((1, *layout[name]['shape'][1:]), dtype=layout[name]['dtype'])
              for name in conditions.CONDITION_FIELDS}
    for name in ('trap_source', 'partial_trap_source'):
        values[name].fill(-1)
    with patch.object(conditions, 'bound_boundary', return_value=(party, observed)):
        return conditions.project_trapped(boundary, original_species=None, conditions=values)


class RetiredTrapAfterVictimFaintTests(unittest.TestCase):
    def test_exact_departure_then_victim_faint_keeps_forced_switch_lanes(self):
        boundary, observed = root199_boundary()
        self.assertEqual(boundary.history['trap_causes'], [])
        cause, = boundary.history['retired_fainted_trap_proofs'][0]['causes']
        first_faint, = boundary.history['retained_status']
        self.assertEqual(cause['arena_trap_retired_faint']['faint_call_id'], first_faint['call_id'])
        self.assertEqual(trapped_projection(boundary, observed)['trapped'].tolist(), [[False, False]])
        changed = SimpleNamespace(history=boundary.history, phase=boundary.phase, relations=(
            {'kind': 'forced-replacement', 'physical_mask': [False] * 5 + [True] * 4},
            boundary.relations[1],
        ))
        with self.assertRaisesRegex(ValueError, 'disagree with the full Native switch lanes'):
            trapped_projection(changed, observed)

    def test_missing_writer_or_broken_causal_chain_still_refuses(self):
        for options in ({'writer': False}, {'same_move': False}, {'preserve_cache': False},
                        {'competing': True}):
            with self.subTest(options=options):
                boundary, observed = root199_boundary(**options)
                with self.assertRaisesRegex(ValueError, 'unexplained contribution'):
                    trapped_projection(boundary, observed)

    def test_later_native_activity_retires_the_proof(self):
        for invalidator in ('refresh', 'commit', 'later-install', 'failed-try-trap'):
            with self.subTest(invalidator=invalidator):
                boundary, observed = root199_boundary(invalidator=invalidator)
                self.assertEqual(boundary.history['retired_fainted_trap_proofs'], [])
                with self.assertRaisesRegex(ValueError, 'unexplained contribution'):
                    trapped_projection(boundary, observed)

    def test_the_same_cache_does_not_explain_a_live_request_or_queued_move(self):
        boundary, observed = root199_boundary()
        for phase in (replace(boundary.phase, requested=(True, True)),
                      replace(boundary.phase, queued_actor=1, queued_actions=(0, 1))):
            with self.subTest(phase=phase):
                changed = SimpleNamespace(history=boundary.history, phase=phase, relations=boundary.relations)
                with self.assertRaisesRegex(ValueError, 'unexplained contribution'):
                    trapped_projection(changed, observed)
