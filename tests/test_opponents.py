import unittest

from pokezero.opponents import opponent_pool_policy_specs


class OpponentPoolTest(unittest.TestCase):
    def test_excludes_current_policy_by_default(self) -> None:
        pool = opponent_pool_policy_specs(
            fixed_policy_specs=("simple-legal", "scripted-teacher"),
            checkpoint_history=("neural:/runs/iter-0001.pt",),
            current_policy_spec="neural:/runs/iter-0002.pt",
            max_historical_opponents=3,
        )
        # No mirror by default; current policy is never an opponent.
        self.assertEqual(pool, ("simple-legal", "scripted-teacher", "neural:/runs/iter-0001.pt"))

    def test_mirror_match_appends_current_policy(self) -> None:
        pool = opponent_pool_policy_specs(
            fixed_policy_specs=("simple-legal", "scripted-teacher"),
            checkpoint_history=(),
            current_policy_spec="neural:/runs/iter-0002.pt",
            max_historical_opponents=3,
            include_current_policy=True,
        )
        # current-vs-current self-play is available from the start.
        self.assertEqual(
            pool, ("simple-legal", "scripted-teacher", "neural:/runs/iter-0002.pt")
        )

    def test_mirror_match_does_not_duplicate_existing_identity(self) -> None:
        # The current policy identity is already in the pool (here via a fixed spec, since
        # history always excludes the current identity) -> mirror must not add a duplicate.
        pool = opponent_pool_policy_specs(
            fixed_policy_specs=("simple-legal", "neural:/runs/iter-0002.pt"),
            checkpoint_history=(),
            current_policy_spec="neural:/runs/iter-0002.pt",
            max_historical_opponents=3,
            include_current_policy=True,
        )
        self.assertEqual(pool.count("neural:/runs/iter-0002.pt"), 1)
        self.assertEqual(pool, ("simple-legal", "neural:/runs/iter-0002.pt"))


if __name__ == "__main__":
    unittest.main()
