import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from fresh200_native_conditions_test import ability, living_cache
from fresh200_native_ledger_test import Trace, arr, ledger, mon, obj
from gpu_engine.fresh200_native_conditions import _retired_arena_trap
from gpu_engine.fresh200_native_phase import PhaseProof


def suspended_arena_trap(*, named_source=1, same_victim=True, refreshed=False):
    trace = Trace()
    victim, source, replacement = mon(0, 0), mon(1, 1), mon(1, 5)
    before = living_cache()
    before['fields'] = dict(requestState='move')
    before['sides'][1]['fields']['active'] = arr(6001, source)
    before['sides'][1]['pokemon'][1]['fields']['ability'] = 'arenatrap'
    trapped = copy.deepcopy(before)
    trapped['sides'][0]['pokemon'][0]['fields']['trapped'] = 'hidden'
    trace.enter('Battle.commitChoices', before=before)
    trace.leave()
    trace.enter('Pokemon.tryTrap', victim, (True,), before)
    trace.rows[-1]['evidence']['context'] = dict(effect=ability('arenatrap'),
        effectState=obj(11000, id='arenatrap', target=mon(1, named_source)))
    trace.leave(True, after=trapped)
    trace.enter('Battle.commitChoices', before=trapped)
    trace.leave()

    switch = obj(11001, choice='switch', pokemon=source, target=replacement)
    side = trapped['sides'][1]
    side['fields']['choice'] = obj(11002, actions=arr(11003, switch))
    switched = copy.deepcopy(trapped)
    switched['sides'][1]['fields']['active'] = arr(6001, replacement)
    if not same_victim:
        switched['sides'][0]['fields']['active'] = arr(6000, mon(0, 2))
    trace.enter('Side.commitChoices', dict(kind='Side', object=11, n=1), before=trapped)
    trace.leave(after=trapped)
    trace.enter('Battle.runAction', arguments=(switch,), before=trapped)
    trace.enter('BattleActions.switchIn', arguments=(replacement, 0), before=trapped)
    trace.leave(True, after=switched)
    trace.leave()

    falling = copy.deepcopy(switched)
    falling['sides'][1]['pokemon'][5]['fields'].update(hp=0, status='', fainted=False, faintQueued=False)
    fainted = copy.deepcopy(falling)
    fainted['sides'][1]['pokemon'][5]['fields'].update(fainted=True, faintQueued=True)
    trace.enter('Pokemon.faint', replacement, before=falling)
    trace.leave(0, after=fainted)
    if refreshed:
        trace.enter('Battle.endTurn', before=fainted)
        trace.leave(after=fainted)
    history = ledger().consume(trace.rows).snapshot()
    phase = PhaseProof('replacement-before-residual', 7, (1, 0), (False, True), True, -1,
                       (0, 0), (0, 0), len(trace.rows))
    boundary = SimpleNamespace(history=history, phase=phase,
        relations=({'kind': 'inactive'}, {'kind': 'forced-replacement'}))
    return boundary, fainted


class NativeRetiredArenaTrapTests(unittest.TestCase):
    def test_frozen49_requires_actual_writer_and_unchanged_victim(self):
        boundary, observed = suspended_arena_trap()
        cause, = boundary.history['trap_causes'][0]['causes']
        self.assertEqual(cause['arena_trap_writer']['source'], mon(1, 1))
        self.assertEqual(cause['arena_trap_departure']['replacement'], mon(1, 5))
        self.assertTrue(_retired_arena_trap(cause, boundary, observed, 0, 0))
        for options, absent in ((dict(named_source=2), 'arena_trap_writer'),
                                (dict(same_victim=False), 'arena_trap_departure')):
            with self.subTest(options=options):
                altered, cache = suspended_arena_trap(**options)
                cause, = altered.history['trap_causes'][0]['causes']
                self.assertNotIn(absent, cause)
                self.assertFalse(_retired_arena_trap(cause, altered, cache, 0, 0))
        refreshed, _ = suspended_arena_trap(refreshed=True)
        self.assertEqual(refreshed.history['trap_causes'], [])

    def test_frozen49_refuses_wrong_phase_order_or_live_replacement(self):
        boundary, observed = suspended_arena_trap()
        cause, = boundary.history['trap_causes'][0]['causes']
        for phase in (replace(boundary.phase, pending_residual=False),
                      replace(boundary.phase, queued_actor=0), replace(boundary.phase, requested=(True, False)),
                      replace(boundary.phase, snapshot_sequence=cause['arena_trap_departure']['end'])):
            with self.subTest(phase=phase):
                altered = SimpleNamespace(history=boundary.history, phase=phase, relations=boundary.relations)
                self.assertFalse(_retired_arena_trap(cause, altered, observed, 0, 0))
        observed['sides'][1]['pokemon'][5]['fields']['hp'] = 1
        self.assertFalse(_retired_arena_trap(cause, boundary, observed, 0, 0))


if __name__ == '__main__':
    unittest.main()
