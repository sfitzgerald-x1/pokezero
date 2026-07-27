"""Timing corpus contract: privacy, determinism, coverage, fail-closed reads."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.timing_corpus import (
    DEFAULT_DECISION_COUNT,
    STRATA_AXES,
    CorpusError,
    TimingDecisionRecord,
    boost_bucket,
    build_corpus,
    bucket_counts,
    hp_bucket,
    label_strata,
    phase_bucket,
    read_corpus,
    remaining_bucket,
    select_stratified,
    uncertainty_bucket,
    write_corpus,
)

MASK = (True, True, False, False, False, False, False, False, False)


def _record(index: int, **overrides) -> TimingDecisionRecord:
    values = dict(
        decision_id=f"d{index:04d}",
        battle_id=f"battle-{index}",
        seat="p1" if index % 2 == 0 else "p2",
        turn_index=index % 30,
        team_seed=1000 + index,
        battle_seed=2000 + index,
        bot_rng_seed=3000 + index,
        event_prefix=("|start|", f"|turn|{index % 30}"),
        action_candidates=({"kind": "move", "move_id": "surf", "slot": 1},),
        legal_action_mask=MASK,
        public_belief_inputs={"revealed_moves": ["surf"]},
        strata=label_strata(
            remaining=6 - (index % 6),
            team_hp_fraction=(index % 10) / 10.0,
            boosts={"attack": 1} if index % 3 == 0 else None,
            forced_switch=index % 5 == 0,
            hidden_world_count=index % 4,
            turn_index=index % 30,
        ),
    )
    values.update(overrides)
    return TimingDecisionRecord(**values)


class StrataLabelTest(unittest.TestCase):
    def test_buckets_are_deterministic_and_total(self) -> None:
        self.assertEqual(remaining_bucket(6), "remaining_6")
        self.assertEqual(remaining_bucket(3), "remaining_4")
        self.assertEqual(remaining_bucket(1), "remaining_1")
        self.assertEqual(hp_bucket(0.9), "hp_high")
        self.assertEqual(hp_bucket(0.5), "hp_medium")
        self.assertEqual(hp_bucket(0.1), "hp_low")
        self.assertEqual(boost_bucket(None), "boost_none")
        self.assertEqual(boost_bucket({"attack": 2}), "boost_offensive")
        self.assertEqual(boost_bucket({"defense": 1}), "boost_defensive")
        self.assertEqual(uncertainty_bucket(0), "uncertainty_low")
        self.assertEqual(uncertainty_bucket(3), "uncertainty_high")
        self.assertEqual(phase_bucket(1), "phase_early")
        self.assertEqual(phase_bucket(15), "phase_middle")
        self.assertEqual(phase_bucket(40), "phase_late")

    def test_every_record_is_labeled_on_every_axis(self) -> None:
        strata = set(_record(7).strata)
        for axis in STRATA_AXES:
            self.assertEqual(len(strata.intersection(axis)), 1, f"axis {axis} unlabeled")


class PrivacyContractTest(unittest.TestCase):
    def test_opponent_private_inputs_rejected(self) -> None:
        for leak in ("opponent_request", "hidden_team", "opponent_private"):
            with self.assertRaisesRegex(ValueError, "non-public"):
                _record(1, public_belief_inputs={leak: {"x": 1}})

    def test_action_candidates_required(self) -> None:
        # The precise reason public-decision-corpus.v1 cannot be reused.
        with self.assertRaisesRegex(ValueError, "action_candidates"):
            _record(1, action_candidates=())

    def test_decision_needs_a_legal_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "legal action"):
            _record(1, legal_action_mask=tuple(False for _ in MASK))


class SelectionTest(unittest.TestCase):
    def test_selection_is_deterministic(self) -> None:
        pool = [_record(i) for i in range(400)]
        first = select_stratified(pool, count=64)
        shuffled = list(reversed(pool))
        second = select_stratified(shuffled, count=64)
        self.assertEqual([r.decision_id for r in first], [r.decision_id for r in second])

    def test_selection_has_no_duplicates(self) -> None:
        pool = [_record(i) for i in range(400)]
        chosen = select_stratified(pool, count=256)
        self.assertEqual(len(chosen), 256)
        self.assertEqual(len({r.decision_id for r in chosen}), 256)

    def test_selection_spreads_across_buckets(self) -> None:
        pool = [_record(i) for i in range(400)]
        counts = bucket_counts(select_stratified(pool, count=256))
        populated = {b: c for b, c in counts.items() if c}
        # Round-robin must not collapse onto a couple of buckets.
        self.assertGreaterEqual(len(populated), 12)

    def test_insufficient_pool_is_terminal(self) -> None:
        with self.assertRaisesRegex(CorpusError, "widen the held-out"):
            select_stratified([_record(i) for i in range(10)], count=256)


class RoundTripTest(unittest.TestCase):
    def test_write_read_round_trip_and_hash_gate(self) -> None:
        pool = [_record(i) for i in range(400)]
        manifest, records = build_corpus(
            pool, held_out_seed_start=900_000, held_out_seed_end=901_000, count=DEFAULT_DECISION_COUNT
        )
        self.assertEqual(manifest.decision_count, DEFAULT_DECISION_COUNT)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.jsonl"
            write_corpus(path, manifest, records)
            read_manifest, read_records = read_corpus(path)
            self.assertEqual(read_manifest.corpus_sha256, manifest.corpus_sha256)
            self.assertEqual(
                [r.decision_id for r in read_records], [r.decision_id for r in records]
            )

    def test_mutated_corpus_fails_closed(self) -> None:
        pool = [_record(i) for i in range(400)]
        manifest, records = build_corpus(
            pool, held_out_seed_start=1, held_out_seed_end=2, count=64
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.jsonl"
            write_corpus(path, manifest, records)
            lines = path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(lines[1])
            payload["turn_index"] = 999  # tamper with one decision
            lines[1] = json.dumps(payload, sort_keys=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CorpusError, "content hash"):
                read_corpus(path)

    def test_corpus_hash_moves_with_selection_inputs(self) -> None:
        pool = [_record(i) for i in range(400)]
        a, _ = build_corpus(pool, held_out_seed_start=1, held_out_seed_end=2, count=64)
        b, _ = build_corpus(pool, held_out_seed_start=1, held_out_seed_end=2, count=65)
        self.assertNotEqual(a.corpus_sha256, b.corpus_sha256)

    def test_atomic_write_leaves_no_partial_file(self) -> None:
        pool = [_record(i) for i in range(120)]
        manifest, records = build_corpus(pool, held_out_seed_start=1, held_out_seed_end=2, count=64)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "corpus.jsonl"
            write_corpus(path, manifest, records)
            self.assertTrue(path.is_file())
            self.assertFalse(list(path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
