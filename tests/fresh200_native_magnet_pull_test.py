import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from fresh200_native_conditions_test import ability, living_cache
from fresh200_native_ledger_test import Trace, arr, ledger, mon, obj
from gpu_engine.fresh200_native_conditions import _fainted_magnet_pull
from gpu_engine.fresh200_native_ledger import MAGNET_PULL_TRAP_CALLBACK
from gpu_engine.fresh200_native_phase import PhaseProof


def source_faint_boundary(*, callback=MAGNET_PULL_TRAP_CALLBACK, write=True,
                          clear_source=True, clear_victim=False, reinstall=False):
    trace = Trace()
    victim, holder = mon(0, 4), mon(1, 4)
    before = living_cache()
    before['fields'] = dict(requestState='move')
    for ref in (victim, holder):
        side = ref['side']
        before['sides'][side]['fields']['active'] = arr(6000 + side, ref)
        before['sides'][side]['pokemon'][4]['fields'].update(fainted=False, faintQueued=False)
    before['sides'][1]['pokemon'][4]['fields']['ability'] = 'magnetpull'
    before['sides'][0]['pokemon'][4]['fields']['trapped'] = False if write else 'hidden'
    trapped = copy.deepcopy(before)
    trapped['sides'][0]['pokemon'][4]['fields']['trapped'] = 'hidden'
    context = dict(effect=ability('magnetpull'), effectState=obj(11000, id='magnetpull', target=holder))

    trace.enter('Battle.commitChoices', before=before)
    trace.leave()
    trace.enter('Battle.runEvent', arguments=('TrapPokemon', victim), before=before)
    trace.enter(callback, arguments=(victim,), before=before)
    trace.rows[-1]['evidence']['context'] = context
    trace.enter('Pokemon.tryTrap', victim, (True,), before)
    trace.rows[-1]['evidence']['context'] = context
    trace.leave(True, after=trapped)
    trace.leave(after=trapped)
    trace.leave(after=trapped)
    trace.enter('Battle.commitChoices', before=trapped)
    trace.leave()

    dying = copy.deepcopy(trapped)
    dying['sides'][0]['pokemon'][4]['fields'].update(hp=205, maybeTrapped=True)
    dying['sides'][1]['pokemon'][4]['fields']['hp'] = -62
    queued = copy.deepcopy(dying)
    queued['sides'][1]['pokemon'][4]['fields'].update(hp=0, faintQueued=True)
    trace.enter('Pokemon.faint', holder, before=dying)
    trace.leave(-62, after=queued)
    completed = copy.deepcopy(queued)
    completed['sides'][1]['pokemon'][4]['fields'].update(fainted=True, faintQueued=False)
    trace.enter('Battle.faintMessages', before=queued)
    if clear_source:
        trace.enter('Pokemon.clearVolatile', holder, (False,), before=queued)
        trace.leave(after=completed)
    if clear_victim:
        trace.enter('Pokemon.clearVolatile', victim, (False,), before=queued)
        trace.leave(after=completed)
    trace.leave(False, after=completed)
    if reinstall:
        replaced = copy.deepcopy(completed)
        replacement = mon(1, 5)
        replaced['sides'][1]['fields']['active'] = arr(6001, replacement)
        trace.enter('BattleActions.switchIn', arguments=(replacement, 0), before=completed)
        trace.leave(True, after=replaced)
        trace.enter('BattleActions.switchIn', arguments=(holder, 0), before=replaced)
        trace.leave(True, after=completed)
    phase = PhaseProof('replacement-before-residual', 4, (1, 1), (False, True), True, -1,
                       (0, 0), (0, 0), len(trace.rows))
    boundary = SimpleNamespace(history=ledger().consume(trace.rows).snapshot(), phase=phase,
        relations=({'kind': 'inactive'}, {'kind': 'forced-replacement'}))
    return boundary, completed


class FaintedMagnetPullTests(unittest.TestCase):
    def test_requires_the_pinned_write_and_completed_same_holder_faint(self):
        for changes in ({}, dict(callback='Callback:' + '0' * 64), dict(write=False),
                        dict(clear_source=False), dict(clear_victim=True), dict(reinstall=True)):
            with self.subTest(changes=changes):
                boundary, observed = source_faint_boundary(**changes)
                causes = [cause for row in boundary.history['trap_causes'] for cause in row['causes']]
                self.assertEqual(any(_fainted_magnet_pull(cause, boundary, observed, 0, 4)
                                     for cause in causes), not changes)

    def test_certificate_is_limited_to_the_witnessed_phase_and_hidden_cache(self):
        boundary, observed = source_faint_boundary()
        cause, = boundary.history['trap_causes'][0]['causes']
        for phase in (replace(boundary.phase, requested=(True, True)),
                      replace(boundary.phase, queued_actions=(1, 0)),
                      replace(boundary.phase, snapshot_sequence=cause['sequence'])):
            with self.subTest(phase=phase):
                altered = SimpleNamespace(history=boundary.history, phase=phase, relations=boundary.relations)
                self.assertFalse(_fainted_magnet_pull(cause, altered, observed, 0, 4))
        observed['sides'][0]['pokemon'][4]['fields']['trapped'] = True
        self.assertFalse(_fainted_magnet_pull(cause, boundary, observed, 0, 4))
