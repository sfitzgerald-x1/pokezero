from copy import deepcopy
import unittest

from gpu_engine.fresh200_native_consumer import verify_visibility_snapshot, verify_visibility_start


class OwnerVisibilityTransportTest(unittest.TestCase):
    def test_start_requires_the_exact_write_bytes_and_pre_player_order(self):
        policy = 'a' * 64
        command = dict(formatid='gen3customgame@@@sleepclausemod', seed='1,2,3,4',
                       players={owner: dict(name=owner, team='literal-' + owner) for owner in ('p1', 'p2')})
        rows = [
            dict(operation='START_WRITE_RETURN', evidence=dict(ordinal=1,
                sha256='6cdf8b445e3d751170ea038b8a022e631db36ded1480cdc6e47e4757318894c9')),
            dict(operation='START_VISIBILITY_APPLIED', evidence=dict(policy_sha256=policy, before_exact=True,
                after_exact=False, before_percentage=False, after_percentage=False, turn=0, installed_players=0)),
            dict(operation='REQUEST_CALL_BEGIN', evidence={}),
            dict(operation='START_WRITE_RETURN', evidence=dict(ordinal=2,
                sha256='16f28c89d5f99dd22ad5410d349ea96e3ce708fbdd11a3643ee6e0a0516cd7c8')),
        ]
        verify_visibility_start(command, rows, policy)
        bad = deepcopy(rows)
        bad[3]['evidence']['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'exact write bytes differ'):
            verify_visibility_start(command, bad, policy)
        with self.assertRaisesRegex(ValueError, 'preceded visibility'):
            verify_visibility_start(command, [rows[0], rows[2], rows[1], rows[3]], policy)
        with self.assertRaisesRegex(ValueError, 'complete Native visibility start'):
            verify_visibility_start(command, rows[:-1], policy)

    def test_snapshot_requires_the_actual_percentage_flags(self):
        snapshot = dict(battle=dict(reportExactHP=False, reportPercentages=True))
        verify_visibility_snapshot(snapshot)
        snapshot['battle']['reportExactHP'] = True
        with self.assertRaisesRegex(ValueError, 'lacks admitted percentage'):
            verify_visibility_snapshot(snapshot)
