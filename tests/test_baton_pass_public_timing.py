"""Classify only public current-turn action completion, not private commitments."""
from types import SimpleNamespace
import unittest

from pokezero.local_showdown import PublicBattleMaterializationState


class PublicBatonTimingTests(unittest.TestCase):
    def state(self,lines,subject='p2',pending=True,force=True):
        return PublicBattleMaterializationState(player_id=subject,format_id='gen3randombattle',
            observation_format_id='gen3randombattle',belief_engine=None,
            self_request={'forceSwitch':[True]} if force else {'active':[]},
            replay=SimpleNamespace(pending_baton_pass=(subject,) if pending else (),
                public_events=[SimpleNamespace(raw_line=line) for line in lines]))

    def test_original_failed_public_order_has_no_deferred_opponent_action(self):
        state=self.state(['|turn|68',
            '|move|p1a: Mawile|Hidden Power|p2a: Ariados',
            '|-activate|p2a: Ariados|Substitute|[damage]',
            '|move|p2a: Ariados|Baton Pass|p2a: Ariados'])
        self.assertIsNone(state.deferred_opponent_action_player)

    def test_fast_baton_retains_hidden_pending_action_but_not_its_identity(self):
        state=self.state(['|turn|68','|move|p2a: Ariados|Baton Pass|p2a: Ariados'])
        self.assertEqual(state.deferred_opponent_action_player,'p1')
        self.assertEqual(self.state(['|turn|68','|move|p1a: X|Baton Pass|p1a: X'],
            subject='p1').deferred_opponent_action_player,'p2')

    def test_current_switch_cant_and_faint_cancellation_consume_commitment(self):
        for action in ('|switch|p1a: Z|Zapdos|100/100',
                       '|cant|p1a: X|slp', '|faint|p1a: X'):
            with self.subTest(action=action):
                self.assertIsNone(self.state(['|turn|68',action,
                    '|move|p2a: Ariados|Baton Pass|p2a: Ariados']).deferred_opponent_action_player)

    def test_previous_turn_action_never_consumes_current_commitment(self):
        state=self.state(['|turn|67','|move|p1a: X|Surf|p2a: Y','|turn|68',
            '|move|p2a: Y|Baton Pass|p2a: Y'])
        self.assertEqual(state.deferred_opponent_action_player,'p1')

    def test_called_move_without_public_selected_action_is_not_completion_proof(self):
        state=self.state(['|turn|68',
            '|move|p1a: X|Surf|p2a: Y|[from] move: Sleep Talk',
            '|move|p2a: Y|Baton Pass|p2a: Y'])
        self.assertEqual(state.deferred_opponent_action_player,'p1')

    def test_non_baton_force_switch_and_ordinary_requests_have_no_deferred_move(self):
        lines=['|turn|68','|move|p2a: Y|Baton Pass|p2a: Y']
        self.assertIsNone(self.state(lines,pending=False).deferred_opponent_action_player)
        self.assertIsNone(self.state(lines,force=False).deferred_opponent_action_player)


if __name__=='__main__': unittest.main()
