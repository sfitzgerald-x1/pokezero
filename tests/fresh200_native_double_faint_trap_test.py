import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from fresh200_native_conditions_test import ability, living_cache
from fresh200_native_ledger_test import Trace, arr, effect, ledger, mon, obj
from gpu_engine.fresh200_native_conditions import _double_faint_arena_trap
from gpu_engine.fresh200_native_phase import PhaseProof


def explosion_boundary(*, same_move=True, cache_reset=False, switched=False):
    trace = Trace()
    victim, source = mon(1, 3), mon(0, 5)
    before = living_cache()
    before['fields'] = dict(requestState='move')
    before['sides'][1]['fields']['active'] = arr(6001, victim)
    before['sides'][0]['fields']['active'] = arr(6000, source)
    before['sides'][0]['pokemon'][5]['fields']['ability'] = 'arenatrap'
    trapped = copy.deepcopy(before)
    trapped['sides'][1]['pokemon'][3]['fields'].update(trapped='hidden', maybeTrapped=True)
    trace.enter('Battle.commitChoices', before=before)
    trace.leave()
    trace.enter('Pokemon.tryTrap', victim, (True,), before)
    trace.rows[-1]['evidence']['context'] = dict(effect=ability('arenatrap'),
        effectState=obj(11000, id='arenatrap', target=source))
    trace.leave(True, after=trapped)
    trace.enter('Battle.commitChoices', before=trapped)
    trace.leave()
    if switched:
        replacement = mon(0, 4)
        departed = copy.deepcopy(trapped)
        departed['sides'][0]['fields']['active'] = arr(6000, replacement)
        trace.enter('BattleActions.switchIn', arguments=(replacement, 0), before=trapped)
        trace.leave(True, after=departed)
        trace.enter('BattleActions.switchIn', arguments=(source, 0), before=departed)
        trace.leave(True, after=trapped)

    trace.enter('BattleActions.runMove', arguments=(effect('explosion' if same_move else 'selfdestruct'), victim), before=trapped)
    current = trapped
    for actor, hp in ((victim, 45), (source, -418)):
        position = (actor['side'], actor['original'])
        dying = copy.deepcopy(current)
        dying['sides'][position[0]]['pokemon'][position[1]]['fields'].update(hp=hp, fainted=False, faintQueued=False)
        if cache_reset and actor == victim:
            dying['sides'][1]['pokemon'][3]['fields']['trapped'] = False
        queued = copy.deepcopy(dying)
        queued['sides'][position[0]]['pokemon'][position[1]]['fields'].update(hp=0, faintQueued=True)
        queued['sides'][1]['pokemon'][3]['fields']['trapped'] = 'hidden'
        trace.enter('Pokemon.faint', actor, before=dying)
        trace.leave(hp, after=queued)
        current = queued
    trace.leave()
    completed = copy.deepcopy(current)
    for side, index in ((0, 5), (1, 3)):
        completed['sides'][side]['pokemon'][index]['fields'].update(fainted=True, faintQueued=False)
    trace.enter('Battle.faintMessages', before=current)
    for actor in (victim, source):
        trace.enter('Pokemon.clearVolatile', actor, (False,), before=current)
        trace.leave(after=current)
    trace.leave(False, after=completed)
    phase = PhaseProof('replacement-before-residual', 7, (1, 1), (True, True), True, -1,
                       (0, 0), (0, 0), len(trace.rows))
    boundary = SimpleNamespace(history=ledger().consume(trace.rows).snapshot(), phase=phase,
        relations=({'kind': 'forced-replacement'}, {'kind': 'forced-replacement'}))
    return boundary, completed


class DoubleFaintArenaTrapTests(unittest.TestCase):
    def test_frozen68_requires_the_same_explosion_and_no_intervening_switch_or_cache_reset(self):
        for options, permitted in (({}, True), (dict(same_move=False), False),
                                   (dict(cache_reset=True), False), (dict(switched=True), False)):
            with self.subTest(options=options):
                boundary, observed = explosion_boundary(**options)
                self.assertEqual(boundary.history['trap_causes'], [])
                proofs = boundary.history['double_faint_trap_proofs']
                if permitted:
                    cause, = proofs[0]['causes']
                    self.assertNotIn('arena_trap_departure', cause)
                    self.assertTrue(_double_faint_arena_trap(cause, boundary, observed, 1, 3))
                else:
                    self.assertEqual(proofs, [])

    def test_frozen68_proof_only_applies_to_the_closed_both_forced_boundary(self):
        boundary, observed = explosion_boundary()
        cause, = boundary.history['double_faint_trap_proofs'][0]['causes']
        for phase in (replace(boundary.phase, requested=(True, False)), replace(boundary.phase, pending_residual=False),
                      replace(boundary.phase, queued_actions=(0, 3)), replace(boundary.phase, snapshot_sequence=0)):
            with self.subTest(phase=phase):
                altered = SimpleNamespace(history=boundary.history, phase=phase, relations=boundary.relations)
                self.assertFalse(_double_faint_arena_trap(cause, altered, observed, 1, 3))
        observed['sides'][1]['pokemon'][3]['fields']['trapped'] = True
        self.assertFalse(_double_faint_arena_trap(cause, boundary, observed, 1, 3))


if __name__ == '__main__':
    unittest.main()
