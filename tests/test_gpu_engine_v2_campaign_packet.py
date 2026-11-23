from pathlib import Path
import unittest

from scripts.bind_v2_campaign_packet import build_packet, rebind_runtime


class CampaignPacketTests(unittest.TestCase):
    def test_runtime_rebind_changes_only_pool_policy_and_source(self):
        previous = {'source_commit': 'a'*40, 'constructor_pool': {'path': '/old', 'sha256': 'b'*64},
                    'gender_policy': {'path': '/old-gender', 'sha256': 'c'*64}, 'versions': {'torch': 'pinned'}}
        pool = {'path': '/new', 'sha256': 'd'*64}
        policy = {'path': '/new-gender', 'sha256': 'e'*64}
        old = {'provenance': {'pool_sha256': 'b'*64, 'node': 'pinned'}, 'species': []}
        new = {'provenance': {'pool_sha256': 'd'*64, 'node': 'pinned'}, 'species': []}
        self.assertEqual(rebind_runtime(previous, old, new, pool, policy, 'f'*40),
                         dict(previous, source_commit='f'*40, constructor_pool=pool, gender_policy=policy))
        with self.assertRaisesRegex(ValueError, 'constructor semantics'):
            rebind_runtime(previous, old, dict(new, species=['changed']), pool, policy, 'f'*40)

    def test_packet_binds_absolute_declaration_paths_and_parent_round(self):
        config = dict(checkout='/checkout', source_commit='a'*40, launcher={}, cpu=32, memory='896Gi',
                      reference_run='/reference', stop_key_reference='existing:reference',
                      previous_runtime={'path': '/previous'}, main={'output': 'v2a-012abc-gpu'},
                      children=[{'parent_games': 13_500_000, 'output': 'v2a-765432-gpu'}])
        declarations = {
            'main': {'path': 'main.json', 'sha256': 'd'*64, 'stops': [1250, 2500, 3750, 5000]},
            'children': [{'parent_games': 13_500_000, 'path': 'fork.json', 'sha256': 'e'*64, 'stop': 75}],
        }
        runtime = {'path': '/new-runtime', 'sha256': 'f'*64}
        packet = build_packet(config, declarations, Path('/bundle'), runtime)
        self.assertEqual(packet['main']['declaration'], {'path': '/bundle/main.json', 'sha256': 'd'*64})
        self.assertEqual(packet['children'][0]['parent_round'], 675)
        self.assertEqual(packet['children'][0]['stop'], 75)
        self.assertEqual(packet['runtime'], runtime)
        with self.assertRaisesRegex(ValueError, 'every declared cooldown'):
            build_packet(dict(config, children=[]), declarations, Path('/bundle'), runtime)
