"""Induced sleep support uses public history, never the source hidden timer."""
from types import SimpleNamespace
from dataclasses import replace
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
from pokezero.showdown_fixture import FixturePokemon, pack_team
from pokezero.mcts_eval.paper_reference_sleep import induced_sleep_certificates, induced_sleep_support


def state(lines):
    return SimpleNamespace(replay=SimpleNamespace(public_events=[SimpleNamespace(raw_line=x) for x in lines]))


START = ['|move|p2a: Wailord|Spore|p1a: Articuno',
    '|-status|p1a: Articuno|slp|[from] move: Spore', '|turn|2']


class InducedSleepLedgerTests(unittest.TestCase):
    def test_initial_support_survival_and_early_bird(self):
        cert = induced_sleep_certificates(state(START))['p1:articuno']
        self.assertEqual([v['time'] for v in induced_sleep_support(cert, 'Pressure')], [2,3,4,5])
        lines = START + ['|cant|p1a: Articuno|slp', '|turn|3']
        cert = induced_sleep_certificates(state(lines))['p1:articuno']
        self.assertEqual([v['time'] for v in induced_sleep_support(cert, 'Pressure')], [1,2,3,4])
        self.assertEqual([v['startTime'] for v in induced_sleep_support(cert, 'Early Bird')], [3,4,5])

    def test_sleep_talk_switch_refund_does_not_erase_survival_constraint(self):
        lines = START + ['|cant|p1a: Articuno|slp','|move|p1a: Articuno|Sleep Talk|p1a: Articuno',
            '|move|p1a: Articuno|Splash|p1a: Articuno|[from]move: Sleep Talk','|turn|3',
            '|switch|p1a: Skarmory|Skarmory|100/100','|turn|4',
            '|switch|p1a: Articuno|Articuno|100/100 slp','|turn|5']
        cert = induced_sleep_certificates(state(lines))['p1:articuno']
        self.assertEqual((cert['attempts'], cert['refunded'], cert['skipped']), (1,1,0))
        self.assertEqual([v['startTime'] for v in induced_sleep_support(cert,'Early Bird')], [3,4,5])
        self.assertEqual([v['time'] for v in induced_sleep_support(cert,'Early Bird')], [2,3,4])

    def test_unknown_source_unsettled_attempt_and_cure(self):
        with self.assertRaisesRegex(LocalShowdownError,'explicit public'):
            induced_sleep_certificates(state(START[1:]))
        with self.assertRaisesRegex(LocalShowdownError,'settled'):
            induced_sleep_certificates(state(START + ['|cant|p1a: Articuno|slp']))
        self.assertEqual(induced_sleep_certificates(state(START + ['|-curestatus|p1a: Articuno|slp'])), {})


@requires_showdown()
class InducedSleepServerTests(unittest.TestCase):
    def test_both_seats_support_expiry_and_sleep_clause_source(self):
        cfg = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = BattleStartOverride(observation_format_id='gen3randombattle', player_teams={
            'p1': pack_team((FixturePokemon('Articuno',('Splash',), ability='Pressure'),
                FixturePokemon('Skarmory',('Splash',), ability='Keen Eye'))),
            'p2': pack_team((FixturePokemon('Wailord',('Spore','Splash'), ability='Water Veil'),))})
        with LocalShowdownEnv(cfg) as live, LocalShowdownEnv(cfg) as restored:
            live.reset_with_start_override(seed=17,start_override=override)
            live.step({'p1':0,'p2':0})
            for actor in ('p1','p2'):
                # The explicit fixture starts in a packed customgame shell;
                # register the intended random-battle rule boundary explicitly.
                public = replace(live.public_materialization_state(actor), format_id='gen3randombattle')
                cert = induced_sleep_certificates(public)['p1:articuno']
                for draw in induced_sleep_support(cert,'Pressure'):
                    restored.materialize_public_world(state=public,start_override=override,seed=19,
                        reference_rest_sleep=True, reference_induced_sleep={'p1:articuno':draw})
                    sides=restored.snapshot().bridge_snapshot['battle']['sides']
                    timer=sides[0]['pokemon'][0]['statusState']
                    self.assertTrue(timer['source'].startswith('[Pokemon:p2'))
                    self.assertTrue(timer['target'].startswith('[Pokemon:p1'))
                    for _ in range(draw['time']-1):
                        restored.step({'p1':0,'p2':1})
                        self.assertEqual(restored.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]['status'],'slp')
                    restored.step({'p1':0,'p2':1})
                    self.assertEqual(restored.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]['status'],'')
                draw=induced_sleep_support(cert,'Pressure')[0]
                restored.materialize_public_world(state=public,start_override=override,seed=19,
                    reference_rest_sleep=True, reference_induced_sleep={'p1:articuno':draw})
                restored.step({'p1':4,'p2':0})
                sides=restored.snapshot().bridge_snapshot['battle']['sides']
                self.assertEqual(sides[0]['pokemon'][0]['status'],'')  # Sleep Clause blocks a second victim.

    def test_invalid_draw_and_default_lane_refuse(self):
        cfg=LocalShowdownConfig(showdown_root=showdown_root(),set_belief_source=True)
        override=BattleStartOverride(player_teams={
            'p1':pack_team((FixturePokemon('Articuno',('Splash',),ability='Pressure'),)),
            'p2':pack_team((FixturePokemon('Wailord',('Spore',),ability='Water Veil'),))})
        with LocalShowdownEnv(cfg) as live, LocalShowdownEnv(cfg) as restored:
            live.reset_with_start_override(seed=17,start_override=override);live.step({'p1':0,'p2':0})
            public=live.public_materialization_state('p1')
            with self.assertRaisesRegex(LocalShowdownError,'sleep counters'):
                restored.materialize_public_world(state=public,start_override=override,seed=19)
            with self.assertRaisesRegex(LocalShowdownError,'conditioning'):
                restored.materialize_public_world(state=public,start_override=override,seed=19,
                    reference_rest_sleep=True,reference_induced_sleep={'p1:articuno':dict(time=9,startTime=9,skippedTime=0)})


if __name__=='__main__': unittest.main()
