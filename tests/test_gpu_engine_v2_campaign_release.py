from dataclasses import asdict
import unittest

from pokezero.v2_fused_state import SCHEMA, campaign_batch_addresses, verify_campaign_prefix
from pokezero.v5_phasec import CAMPAIGN_COOLDOWN_GAMES_BY_PARENT, campaign_iterations
from pokezero.v5_run import _sha, manifest_iteration_game_counts
from scripts.prepare_v2_campaign_declarations import build_declarations


class CampaignReleaseDeclarationsTests(unittest.TestCase):
    def test_adopted_prefix_and_cooldowns_share_one_disjoint_roster(self):
        iterations = campaign_iterations(
            'arm-a', 250, stream_start=10_000_000, team_start=0, identity_seed_start=20_000_000,
            constructor_seed_start=30_000_000, sampling_seed_start=40_000_000,
            gender_seed=50_000_000, gender_domain=51_000_000, gender_root_start=52_000_000,
        )
        parent = {
            'run_id': 'arm-a', 'pool_sha256': 'a' * 64, 'train_both_seats': True, 'max_requests': 1024,
            'train': {'batch_size': 2048, 'epochs': 1, 'learning_rate': 1.18e-4},
            'evaluation': {'execution': 'deferred', 'interval_games': 1_000_000},
            'iterations': [[asdict(batch) for batch in round_] for round_ in iterations],
        }
        main, forks, required_team_rows = build_declarations(parent, 'b' * 64)
        verify_campaign_prefix(parent, main, {
            'round_ledger': {'completed_rounds': 100, 'schema': SCHEMA},
            'declaration_sha256': _sha(parent),
        })
        self.assertEqual(sum(manifest_iteration_game_counts(main)), 100_000_000)
        self.assertEqual(required_team_rows, 293_800_000)
        self.assertEqual(set(forks), {13_500_000, *(9_000_000 * index for index in range(1, 10))})

        last_reserved_batch = parent['iterations'][119][-1]
        frontier = tuple(start + size for start, size in campaign_batch_addresses(last_reserved_batch))
        segments = [main['iterations'][100:]]
        for parent_games, child in forks.items():
            expected_games = CAMPAIGN_COOLDOWN_GAMES_BY_PARENT[parent_games]
            self.assertEqual(sum(manifest_iteration_game_counts(child)), expected_games)
            self.assertEqual(child['evaluation']['interval_games'], expected_games)
            self.assertEqual(child['iterations'][0][0]['games']['first_ordinal'], 0)
            segments.append(child['iterations'])
        for rounds in segments:
            for round_ in rounds:
                for batch in round_:
                    addresses = campaign_batch_addresses(batch)
                    self.assertEqual(tuple(start for start, _ in addresses), frontier)
                    frontier = tuple(start + size for start, size in addresses)
