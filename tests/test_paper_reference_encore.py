"""Public Encore lock with hidden durations conditioned on survival, not truth."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError, _public_reference_encore
from pokezero.env import BattleStartOverride
from pokezero.showdown_fixture import FixturePokemon, pack_team


class PublicEncoreLedgerTests(unittest.TestCase):
    def state(self, lines, elapsed):
        return SimpleNamespace(observation_format_id='gen3randombattle', self_request={},
            deferred_opponent_action_player=None, replay=SimpleNamespace(volatiles={'p1':['encore']},
                encore_elapsed={'p1':elapsed}, public_events=[SimpleNamespace(raw_line=x) for x in lines]))

    def test_duration_support_conditions_on_residuals_and_application_order(self):
        start=['|switch|p1a: Aipom|Aipom|100/100', '|move|p1a: Aipom|Thunder Wave|p2a: Wobbuffet', '|turn|2']
        tail=['|-start|p1a: Aipom|Encore','|upkeep','|turn|3']
        before=_public_reference_encore(self.state(start+tail,1),'p1')
        after=_public_reference_encore(self.state(start+['|move|p1a: Aipom|Thunder Wave|p2a: Wobbuffet']+tail,1),'p1')
        self.assertEqual(before['remaining_candidates'],[2,3,4,5])
        self.assertEqual(after['remaining_candidates'],[3,4,5,6])
        survived=_public_reference_encore(self.state(start+tail+['|upkeep','|turn|4']*4,5),'p1')
        self.assertEqual(survived['remaining_candidates'],[1])

    def test_truncated_unknown_lock_and_midturn_requests_refuse(self):
        with self.assertRaisesRegex(LocalShowdownError,'locked move'):
            _public_reference_encore(self.state(['|-start|p1a: Aipom|Encore'],0),'p1')
        state=self.state([],0);state.self_request={'forceSwitch':[True]}
        with self.assertRaisesRegex(LocalShowdownError,'post-upkeep'):
            _public_reference_encore(state,'p1')


@requires_showdown()
class PublicEncoreServerTests(unittest.TestCase):
    def test_real_server_support_lock_expiry_and_default_refusal(self):
        cfg=LocalShowdownConfig(showdown_root=showdown_root(),set_belief_source=True)
        override=BattleStartOverride(player_teams={
            'p1':pack_team((FixturePokemon('Aipom',('Thunder Wave','Tackle'),ability='Run Away'),)),
            'p2':pack_team((FixturePokemon('Wobbuffet',('Encore','Splash'),ability='Shadow Tag'),))})
        with LocalShowdownEnv(cfg) as live,LocalShowdownEnv(cfg) as restored:
            live.reset_with_start_override(seed=223,start_override=override)
            live.step({'p1':0,'p2':0})
            for actor in ('p1','p2'):
                state=live.public_materialization_state(actor)
                cert=_public_reference_encore(state,'p1')
                self.assertEqual(cert['move'],'thunderwave')
                self.assertTrue(cert['after_target_acted'])
                with self.assertRaisesRegex(LocalShowdownError,'volatile effect encore'):
                    restored.materialize_public_world(state=state,start_override=override,seed=19)
                for duration in cert['remaining_candidates']:
                    restored.materialize_public_world(state=state,start_override=override,seed=19,
                        reference_encore_durations={'p1':duration})
                    row=restored.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]
                    self.assertEqual(row['volatiles']['encore']['duration'],duration)
                    self.assertEqual(row['volatiles']['encore']['move'],'thunderwave')
                    self.assertEqual(row['lastMove'],'[DataMove:thunderwave]')
                    for _ in range(duration-1):
                        restored.step({'p1':0,'p2':1})
                        self.assertIn('encore',restored.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]['volatiles'])
                    restored.step({'p1':0,'p2':1})
                    self.assertNotIn('encore',restored.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]['volatiles'])
                with self.assertRaisesRegex(LocalShowdownError,'conditioning'):
                    restored.materialize_public_world(state=state,start_override=override,seed=19,
                        reference_encore_durations={'p1':1})


if __name__=='__main__':
    unittest.main()
