"""Gate on the acceptance shard's ``policy_stats`` block.

The shard report is the only place the reached-depth aggregates reach disk, and
a depth ladder cannot be read without them: ``depth_reached_*`` is the entire
difference between "depth does not help" and "the sims budget never let the
tree reach the cap" (the confound that closed the depth axis once already).

The bug this pins: the report used to serialize the block as
``search.stats.to_payload() if hasattr(search.stats, "to_payload") else {}``.
``EngineMctsStats`` has never had a ``to_payload`` -- its serializer is
``to_dict`` -- so the guard was always false and every shard ever written
recorded ``"policy_stats": {}``. It failed silently, in the one direction that
looks like a clean run.

Two things are therefore pinned separately, because either alone would let the
bug back in:

* the serializer contract on ``EngineMctsStats`` (a rename to ``to_payload``
  would make the report raise, not silently empty -- that is the point);
* the assembled report actually carrying the keys.

Runs on a hand-populated stats object -- no cluster, no checkpoint, no torch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from pokezero.engine_search import EngineMctsStats

REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "mcts_acceptance_h2h_test", REPO_ROOT / "scripts" / "mcts_acceptance_h2h.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_H2H = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_H2H)
build_shard_report = _H2H.build_shard_report

# The aggregates the depth ladder is read from. `depth_reached_mean` is emitted
# only when samples > 0, so it is asserted against a populated stats object.
DEPTH_KEYS = (
    "depth_reached_samples",
    "depth_reached_max",
    "depth_reached_histogram",
)


def populated_stats() -> EngineMctsStats:
    """A stats object shaped like a real shard's: searched decisions, worlds, depths."""
    stats = EngineMctsStats()
    stats.decisions = 40
    stats.searched_decisions = 38
    stats.fallback_decisions = 2
    stats.worlds_attempted = 160
    # A real shard populates this too. Leaving it 0 while `worlds_searched` is
    # 152 makes the derived rates incoherent (100% construction failure next to
    # 152 searched worlds) -- exactly the broken-instrument state the abort-rate
    # work exists to make visible, so the fixture must not model it by accident.
    stats.worlds_constructed = 156
    stats.worlds_searched = 152
    stats.total_iterations = 38 * 1024
    stats.search_wall_seconds = 152.0
    stats.decision_wall_seconds = 160.0
    stats.depth_reached_samples = 152
    stats.depth_reached_sum = 456
    stats.depth_reached_max = 4
    stats.depth_reached_histogram.update({2: 40, 3: 60, 4: 52})
    return stats


def report_for(stats: EngineMctsStats) -> dict:
    return build_shard_report(
        arm="search",
        config_id="d4-s1024-b64-w4",
        checkpoint="/tmp/ckpt.pt",
        engine_fingerprint="f" * 64,
        provenance="a" * 64,
        pair_start=7800000,
        pairs=10,
        stats=stats,
        search_decisions=38,
        search_wall=152.0,
        wall_s=161.0,
        results=[],
        per_game=[],
    )


class SerializerContractTest(unittest.TestCase):
    def test_stats_serializer_is_to_dict_and_there_is_no_to_payload(self) -> None:
        # The precise false assumption behind the bug. If a future rename adds
        # `to_payload`, this fails loudly here rather than in a 400-pair campaign.
        stats = EngineMctsStats()
        self.assertTrue(hasattr(stats, "to_dict"))
        self.assertFalse(
            hasattr(stats, "to_payload"),
            "EngineMctsStats grew a to_payload; the shard report calls to_dict "
            "unguarded -- reconcile the two rather than reinstating a hasattr guard.",
        )


class ShardPolicyStatsTest(unittest.TestCase):
    def test_policy_stats_is_not_empty(self) -> None:
        # The exact regression: the old expression produced {} here.
        report = report_for(populated_stats())
        self.assertNotEqual(report["policy_stats"], {})

    def test_policy_stats_carries_the_reached_depth_aggregates(self) -> None:
        payload = report_for(populated_stats())["policy_stats"]
        for key in DEPTH_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, payload)
        self.assertEqual(payload["depth_reached_samples"], 152)
        self.assertEqual(payload["depth_reached_max"], 4)
        self.assertEqual(payload["depth_reached_histogram"], {"2": 40, "3": 60, "4": 52})
        # 456 / 152 -- the aggregate the ladder's non-starvation rule reads.
        self.assertAlmostEqual(payload["depth_reached_mean"], 3.0)

    def test_policy_stats_carries_the_latency_gate_field(self) -> None:
        # The same silent-{} also dropped the field the 20 s/turn rejection rule
        # is defined on, so it is pinned in the same place.
        payload = report_for(populated_stats())["policy_stats"]
        self.assertIn("search_wall_per_searched_decision", payload)
        self.assertAlmostEqual(payload["search_wall_per_searched_decision"], 4.0)

    def test_depth_mean_absent_but_block_still_populated_when_unsampled(self) -> None:
        # A raw arm never searches, so it has no depth samples. The block must
        # still be a real payload -- absent `depth_reached_mean` is meaningful,
        # an empty dict is not.
        payload = report_for(EngineMctsStats())["policy_stats"]
        self.assertNotEqual(payload, {})
        self.assertNotIn("depth_reached_mean", payload)
        self.assertEqual(payload["depth_reached_samples"], 0)

    def test_control_arm_stub_can_serialize(self) -> None:
        # Regression, found in independent review. The --arm control path
        # replaces search.stats with a local _NoStats stub that has no
        # to_dict, so the unguarded call raised AFTER every game had been
        # played, discarding the whole shard. The stub now declares its own
        # serializer. The sibling test below uses a real EngineMctsStats and
        # therefore did NOT cover this.
        import re

        source = (REPO_ROOT / "scripts" / "mcts_acceptance_h2h.py").read_text(
            encoding="utf-8"
        )
        stub = re.search(r"class _NoStats:.*?(?=\n        search\.stats)", source, re.S)
        self.assertIsNotNone(stub, "control-arm stub not found; did the arm change?")
        namespace: dict = {}
        exec(stub.group(0).replace("\n        ", "\n"), namespace)  # noqa: S102
        stats = namespace["_NoStats"]()
        self.assertEqual(stats.to_dict(), {})
        # And it must survive the real report builder end to end.
        report = report_for(stats)
        self.assertEqual(report["policy_stats"], {})
        self.assertEqual(report["fallback_decisions"], 0)

    def test_report_is_json_serializable(self) -> None:
        # to_dict() returns a Counter-derived histogram; the shard is written
        # with json.dumps, so a non-serializable member would only surface at
        # the very end of a real run.
        import json

        json.dumps(report_for(populated_stats()))


if __name__ == "__main__":
    unittest.main()
