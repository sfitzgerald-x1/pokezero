import unittest

from pokezero.cli_audit import validate_post_iteration_audit_evaluation_games
from pokezero.run_audit import RunAuditConfig


class PostIterationAuditCliTest(unittest.TestCase):
    def test_validate_post_iteration_audit_evaluation_games_allows_exact_floor(self) -> None:
        validate_post_iteration_audit_evaluation_games(
            RunAuditConfig(min_latest_benchmark_games=12),
            evaluation_games=3,
            minimum_benchmark_matchups=4,
        )

    def test_validate_post_iteration_audit_evaluation_games_rejects_unreachable_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires enough --evaluation-games"):
            validate_post_iteration_audit_evaluation_games(
                RunAuditConfig(min_latest_benchmark_games=13),
                evaluation_games=3,
                minimum_benchmark_matchups=4,
            )

    def test_validate_post_iteration_audit_evaluation_games_skips_optional_benchmark(self) -> None:
        validate_post_iteration_audit_evaluation_games(
            RunAuditConfig(
                min_latest_benchmark_games=100,
                require_benchmark=False,
            ),
            evaluation_games=0,
            minimum_benchmark_matchups=4,
        )


if __name__ == "__main__":
    unittest.main()
