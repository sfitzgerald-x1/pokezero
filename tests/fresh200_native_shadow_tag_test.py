import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from fresh200_native_conditions_test import ability, living_cache
from fresh200_native_ledger_test import Trace, arr, ledger, mon, obj
from gpu_engine.fresh200_native_conditions import _retired_shadow_tag
from gpu_engine.fresh200_native_ledger import SHADOW_TAG_TRAP_CALLBACK
from gpu_engine.fresh200_native_phase import PhaseProof


def retired_shadow_tag_boundary(*, callback=SHADOW_TAG_TRAP_CALLBACK, write=True,
                                clear=True, refresh=False, result=True):
    trace = Trace()
    holder, replacement, victim = mon(0, 3), mon(0, 1), mon(1, 2)
    before = living_cache()
    before['fields'] = dict(requestState='move')
    for ref in (holder, victim):
        side = ref['side']
        before['sides'][side]['fields']['active'] = arr(6000 + side, ref)
        before['sides'][side]['pokemon'][ref['original']]['fields'].update(fainted=False, faintQueued=False)
    before['sides'][0]['pokemon'][3]['fields']['ability'] = 'shadowtag'
    before['sides'][0]['pokemon'][1]['fields'].update(fainted=False, faintQueued=False)
    before['sides'][1]['pokemon'][2]['fields']['trapped'] = not write
    trapped = copy.deepcopy(before)
    trapped['sides'][1]['pokemon'][2]['fields']['trapped'] = True
    context = dict(effect=ability('shadowtag'), effectState=obj(11000, id='shadowtag', target=holder))
    trace.enter('Battle.commitChoices', before=before)
    trace.leave()
    trace.enter('Battle.runEvent', arguments=('TrapPokemon', victim), before=before)
    trace.enter(callback, arguments=(victim,), before=before)
    trace.rows[-1]['evidence']['context'] = context
    trace.leave({'$native': 'undefined'}, after=trapped)
    trace.rows[-1]['evidence']['context'] = context
    trace.leave(after=trapped)
    trace.enter('Battle.commitChoices', before=trapped)
    trace.leave()

    switch = obj(11001, choice='switch', pokemon=holder, target=replacement)
    trapped['sides'][0]['fields']['choice'] = obj(11002, actions=arr(11003, switch))
    switched = copy.deepcopy(trapped)
    switched['sides'][0]['fields']['active'] = arr(6000, replacement)
    trace.enter('Side.commitChoices', dict(kind='Side', object=10, n=0), before=trapped)
    trace.leave(after=trapped)
    trace.enter('Battle.runAction', arguments=(switch,), before=trapped)
    trace.enter('BattleActions.switchIn', arguments=(replacement, 0), before=trapped)
    trace.leave(result, after=switched)
    trace.leave()

    dying = copy.deepcopy(switched)
    dying['sides'][0]['pokemon'][1]['fields']['hp'] = -42
    queued = copy.deepcopy(dying)
    queued['sides'][0]['pokemon'][1]['fields'].update(hp=0, faintQueued=True)
    trace.enter('Pokemon.faint', replacement, before=dying)
    trace.leave(-42, after=queued)
    completed = copy.deepcopy(queued)
    completed['sides'][0]['pokemon'][1]['fields'].update(fainted=True, faintQueued=False)
    trace.enter('Battle.faintMessages', before=queued)
    if clear:
        trace.enter('Pokemon.clearVolatile', replacement, (False,), before=queued)
        trace.leave(after=completed)
    trace.leave(False, after=completed)
    if refresh:
        trace.enter('Battle.endTurn', before=completed)
        trace.leave(after=completed)
    phase = PhaseProof('replacement-before-residual', 6, (1, 1), (True, False), True, -1,
                       (0, 0), (0, 0), len(trace.rows))
    boundary = SimpleNamespace(history=ledger().consume(trace.rows).snapshot(), phase=phase,
        relations=({'kind': 'forced-replacement'}, {'kind': 'inactive'}))
    return boundary, completed


class RetiredShadowTagTests(unittest.TestCase):
    def test_requires_the_direct_write_departure_and_completed_replacement_faint(self):
        for changes in ({}, dict(callback='Callback:' + '0' * 64), dict(write=False),
                        dict(result='pursuitfaint'), dict(clear=False), dict(refresh=True)):
            with self.subTest(changes=changes):
                boundary, observed = retired_shadow_tag_boundary(**changes)
                causes = [cause for row in boundary.history['trap_causes'] for cause in row['causes']]
                self.assertEqual(any(_retired_shadow_tag(cause, boundary, observed, 1, 2)
                                     for cause in causes), not changes)

    def test_rejects_a_new_install_a_hidden_cache_or_an_actionable_victim(self):
        boundary, observed = retired_shadow_tag_boundary()
        cause, = boundary.history['trap_causes'][0]['causes']
        phase = replace(boundary.phase, requested=(True, True))
        altered = SimpleNamespace(history=boundary.history, phase=phase, relations=boundary.relations)
        self.assertFalse(_retired_shadow_tag(cause, altered, observed, 1, 2))

        history = dict(boundary.history, last_active_install=[boundary.phase.snapshot_sequence, -1])
        installed = SimpleNamespace(history=history, phase=boundary.phase, relations=boundary.relations)
        self.assertFalse(_retired_shadow_tag(cause, installed, observed, 1, 2))
        observed['sides'][1]['pokemon'][2]['fields']['trapped'] = 'hidden'
        self.assertFalse(_retired_shadow_tag(cause, boundary, observed, 1, 2))
