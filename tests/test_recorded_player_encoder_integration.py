"""Synthetic two-player production-reference/JAX integration; no games or models."""
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class RecordedPlayerIntegrationTests(unittest.TestCase):
    def test_real_projector_two_production_histories_and_two_jax_adapters(self):
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from test_recorded_mirror_roots import observation
        from pokezero.local_showdown import LocalShowdownConfig
        from pokezero.mirror_value_probe import digest
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        from pokezero.recorded_encoder_parity import SHAPES
        from pokezero.recorded_player_encoder_runner import compare_player_battle
        from pokezero.saved_player_projection import (RULES, SavedSourcePlayerProjector,
            feed_player_reference_boundary, new_player_reference_environments)
        from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC, NUMERIC_HP_FRACTION

        V4EncoderTests.setUpClass()
        fixture = V4EncoderTests()
        battle = dict(battle_id='actual-player-adapter-authored-fixture', seed=321)
        requests = {seat: fixture.request(index) for index, seat in enumerate(('p1', 'p2'))}
        setup = ['|gen|3', '|gametype|singles', '|tier|[Gen 3] Random Battle',
                 '|player|p1|One|', '|player|p2|Two|', '|teamsize|p1|6', '|teamsize|p2|6',
                 *['|rule|' + rule for rule in sorted(RULES)], '|start',
                 '|switch|p1a: Snorlax|Snorlax, L80|200/247',
                 '|switch|p2a: Absol|Absol, L80|300/301',
                 '|-ability|p2a: Absol|Pressure|[silent]', '|turn|1']
        prefix = []

        def row(index, public, private, active):
            start = len(prefix)
            delta = [*public, *['|request|' + json.dumps(packet) for packet in private]]
            prefix.extend(delta)
            saved = {seat: dict(observation(marker=.123), turn_number=index+1) for seat in active}
            value = dict(schema='pokezero.mirror_boundary.v1', **battle,
                decision_round_index=index, requested_players=list(active),
                protocol_start=start, protocol_end=len(prefix), protocol_delta=delta,
                public_prefix_sha256=digest(prefix), requests=copy.deepcopy(requests),
                observations=saved, actions={seat: 0 for seat in active}, paired=len(active)==2)
            value['root_id'] = digest([battle['battle_id'], index,
                                      value['public_prefix_sha256'], value['requests']])
            return value

        first = row(0, setup, [requests['p1'], requests['p2']], ('p1', 'p2'))
        requests['p1']['side']['pokemon'][1]['condition'] = '101/247'
        requests['p2']['wait'] = True
        second = row(1, ['|turn|2', '|-damage|p1a: Snorlax|101/247'],
                     [requests['p1'], requests['p2']], ('p1',))
        end = dict(event='end', protocol_start=len(prefix), protocol_end=len(prefix)+1,
                   protocol_delta=['|win|One'])
        pairs = []
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            # Production parser/belief/annotation/encoder code runs on authored
            # source/dex fixtures; no Node, simulator, archive or checkpoint.
            stack.enter_context(patch('subprocess.Popen', side_effect=AssertionError('no process')))
            stack.enter_context(patch('subprocess.run', side_effect=AssertionError('no runtime execution')))
            stack.enter_context(patch('pokezero.local_showdown.load_gen3_randbat_source_cached',
                                      return_value=fixture.public.source))
            stack.enter_context(patch('pokezero.local_showdown.load_showdown_dex_cached',
                                      return_value=fixture.catalog.dex))
            stack.enter_context(patch('pokezero.category_vocab.logger.warning'))
            config = LocalShowdownConfig(showdown_root=Path(temporary), set_belief_source=True,
                category_vocab=fixture.vocab, observation_spec=V4_REPLAY_OBSERVATION_SPEC,
                feature_masks=ObservationFeatureMasks(exact_state=True, feature_pack_last_move=True,
                    investment_belief_narrowing=False, item_belief_narrowing=False,
                    opponent_tendency_stats_block=True, tier2_investment=True,
                    tier2_residuals=True, transition_token_budget=0))
            envs = new_player_reference_environments(config, battle_id=battle['battle_id'])
            self.addCleanup(lambda: [env.close() for env in envs.values()])
            adapters = {seat: JaxRecordedAdapter(fixture.catalog) for seat in ('p1', 'p2')}
            report = compare_player_battle(iter([first, second, end]), battle=battle, stream_id=100,
                projector=SavedSourcePlayerProjector(battle_id=battle['battle_id']),
                reference_feed=lambda batch, **kw: feed_player_reference_boundary(envs, batch, **kw),
                adapters=adapters, on_view=pairs.append)
            self.assertEqual((report['boundaries'], report['views']), (2, 3))
            self.assertFalse(report['actual_captured_wire'] or report['full_encoder_admitted'])
            for pair in pairs:
                reference = json.loads(pair.regenerated_reference_json)
                for name in SHAPES:
                    if name == 'numeric_features':
                        np.testing.assert_allclose(pair.actual[name], reference[name], atol=1e-6, rtol=0)
                    else:
                        np.testing.assert_array_equal(pair.actual[name], reference[name])
            # Peer HP is percentage-scaled, while the owning private request
            # still supplies its own exact value; old .123 fixtures are unused.
            self.assertAlmostEqual(json.loads(pairs[0].regenerated_reference_json)
                ['numeric_features'][7][NUMERIC_HP_FRACTION], .99, places=6)
            self.assertEqual(len(envs['p1']._request_history['p1']), 2)
            self.assertEqual(len(envs['p2']._request_history['p2']), 2)
            self.assertEqual(envs['p1']._request_history['p2'], [])
            self.assertEqual(envs['p2']._request_history['p1'], [])
