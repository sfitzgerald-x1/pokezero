from dataclasses import replace
import random
from types import SimpleNamespace
import unittest

from pokezero.policy import PolicyDecision
from pokezero.search_labels import (select_search_label, validate_search_targets,
                                    with_search_labels, search_targets_from_reports)


def context(seed=7, seat="p1", decision=0):
    return SimpleNamespace(seed=seed, player_id=seat, decision_round_index=decision,
        observation=SimpleNamespace(legal_action_mask=(True, True) + (False,) * 7))


class RawActor:
    policy_id = "unchanged-raw"
    checkpoint_path = "raw.pt"
    def __init__(self):
        self.calls = 0
        self.last_decision = None

    def select_action(self, observation, *, rng):
        self.calls += 1
        self.last_decision = PolicyDecision(action_index=rng.randrange(2),
            policy_id=self.policy_id, action_probability=.5, value_estimate=-.3,
            metadata={"raw_sentinel": 19})
        return self.last_decision


class Teacher:
    def __init__(self, metadata=None, failure=None):
        self.metadata = metadata if metadata is not None else {"engine_mcts": {"training_targets": {
            "status": "LABELED", "policy_target": [.25, .75] + [0.] * 7, "value_target": -.4}}}
        self.failure = failure
        self.seen = []

    def select_action_with_context(self, context, *, rng):
        self.seen.append((context.player_id, context.decision_round_index,
                          tuple(rng.random() for _ in range(11))))
        if self.failure:
            raise self.failure
        # This proposed action is deliberately illegal in the raw request.
        # It is discarded; the wrapper must return the raw actor's action.
        return PolicyDecision(action_index=8, policy_id="teacher", metadata=self.metadata)


class NativeSearchTargetTests(unittest.TestCase):
    mask = (True, True) + (False,) * 7

    def world(self, side="side_one", total=2.0, visits=3, entries=None):
        return {"side_key": side, "report": {
            "training_root_stats": {"visits": visits, "side_one_value_total": total},
            side: entries if entries is not None else [
                {"move": "a", "visits": 1}, {"move": "b", "visits": 2}],
            # These deliberately disagree: targets must never use rounded q.
            "root_value": .123456,
        }}

    def targets(self, worlds, mapping=None):
        mapping = mapping or {"a": 0, "b": 1}
        return search_targets_from_reports(worlds, self.mask, mapping.get)

    def test_integer_visits_and_full_value_total_precede_rounded_telemetry(self):
        target = self.targets([self.world()])
        self.assertEqual(target["policy_target"], [1 / 3, 2 / 3] + [0.] * 7)
        self.assertEqual(target["value_target"], 2 * (2 / 3) - 1)

    def test_native_side_key_controls_the_value_frame(self):
        first = self.targets([self.world(side="side_one")])
        second = self.targets([self.world(side="side_two")])
        self.assertAlmostEqual(first["value_target"], -second["value_target"])
        self.assertEqual(first["policy_target"], second["policy_target"])

    def test_aliases_accumulate_without_dropping_visits(self):
        entries = [{"move": "a", "visits": 1}, {"move": "alias_a", "visits": 1},
                   {"move": "b", "visits": 1}]
        target = self.targets([self.world(entries=entries)], {"a": 0, "alias_a": 0, "b": 1})
        self.assertEqual(target["policy_target"], [2 / 3, 1 / 3] + [0.] * 7)

    def test_belief_draws_are_equal_weight_and_collapsed_copies_are_not_counted_twice(self):
        a = self.world(total=3.)
        a["_collapse_multiplicity"] = 2
        b = self.world(total=0.)
        target = self.targets([a, a, b])
        self.assertAlmostEqual(target["value_target"], 1 / 3)

    def test_unmapped_or_illegal_positive_mass_is_not_silently_renormalized(self):
        for mapping in ({"a": 0}, {"a": 0, "b": 8}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.targets([self.world()], mapping)
        zero = self.world(entries=[{"move": "a", "visits": 3}, {"move": "unknown", "visits": 0}])
        self.assertEqual(self.targets([zero])["policy_target"], [1.] + [0.] * 8)

    def test_malformed_or_missing_native_work_cannot_become_a_label(self):
        for world in (self.world(visits=0), self.world(visits=True),
                      self.world(total=float("nan")), self.world(total=4.),
                      self.world(visits=4), self.world(side="p1"),
                      self.world(entries=[{"move": "a", "visits": True}])):
            with self.subTest(world=world), self.assertRaises(ValueError):
                self.targets([world])
        with self.assertRaises(ValueError):
            self.targets([])


class SearchLabelPolicyTests(unittest.TestCase):
    def wrap(self, actor, teacher, fraction=1.):
        return with_search_labels(actor, fraction=fraction, teacher_factory=lambda: teacher,
            teacher_identity={"checkpoint_sha256": "a" * 64, "generation": 25})

    def test_zero_fraction_is_the_identical_actor_without_teacher_construction(self):
        actor = RawActor()
        def forbidden():
            raise AssertionError("disabled teacher was constructed")
        wrapped = with_search_labels(actor, fraction=0., teacher_factory=forbidden,
                                    teacher_identity={})
        self.assertIs(wrapped, actor)

    def test_teacher_never_changes_raw_actions_probabilities_values_or_rng_on_either_seat(self):
        actor, control, teacher = RawActor(), RawActor(), Teacher()
        wrapped = self.wrap(actor, teacher)
        actual_rng, control_rng = random.Random(123), random.Random(123)
        for seat in ("p1", "p2"):
            for decision in range(6):
                c = context(seat=seat, decision=decision)
                expected = control.select_action(c.observation, rng=control_rng)
                actual = wrapped.select_action_with_context(c, rng=actual_rng)
                self.assertEqual(replace(actual, metadata=expected.metadata), expected)
                self.assertEqual(actual.metadata["raw_sentinel"], 19)
                self.assertEqual(actual.metadata["search_label"]["value_target"], -.4)
                self.assertEqual(actual_rng.getstate(), control_rng.getstate())
        self.assertEqual(actor.calls, 12)
        self.assertEqual(len(teacher.seen), 12)
        self.assertEqual(wrapped.policy_id, actor.policy_id)
        self.assertEqual(wrapped.checkpoint_path, actor.checkpoint_path)

    def test_unselected_decision_keeps_the_original_decision_and_metadata_objects(self):
        actor, teacher = RawActor(), Teacher()
        wrapped = self.wrap(actor, teacher, fraction=1e-100)
        actual = wrapped.select_action_with_context(context(), rng=random.Random(3))
        self.assertIs(actual, actor.last_decision)
        self.assertEqual(teacher.seen, [])

    def test_contextual_raw_actor_receives_the_same_context_and_original_rng(self):
        class ContextualActor(RawActor):
            def select_action_with_context(self, context, *, rng):
                self.received_context = context
                self.received_rng = rng
                return super().select_action(context.observation, rng=rng)
        actor = ContextualActor()
        c, rng = context(seat="p2"), random.Random(9)
        actual = self.wrap(actor, Teacher()).select_action_with_context(c, rng=rng)
        self.assertIs(actor.received_context, c)
        self.assertIs(actor.received_rng, rng)
        self.assertEqual(actual.action_index, actor.last_decision.action_index)
        self.assertEqual(actor.calls, 1)

    def test_label_selection_and_teacher_randomness_are_reproducible(self):
        contexts = [context(seed=41, decision=i) for i in range(100)]
        mask = [select_search_label(c, .1) for c in contexts]
        self.assertEqual(mask, [select_search_label(c, .1) for c in contexts])
        self.assertGreater(sum(mask), 0)
        self.assertLess(sum(mask), len(mask))
        one, two = Teacher(), Teacher()
        for teacher in (one, two):
            wrapped = self.wrap(RawActor(), teacher)
            for c in contexts[:3]:
                wrapped.select_action_with_context(c, rng=random.Random(19))
        self.assertEqual(one.seen, two.seen)
        self.assertNotEqual(one.seen[0][2], one.seen[1][2])

    def test_fallback_is_explicitly_unlabeled_never_a_uniform_target(self):
        teacher = Teacher(metadata={"engine_mcts": {"fallback": "crate_search_failed"}})
        actual = self.wrap(RawActor(), teacher).select_action_with_context(context(), rng=random.Random(1))
        label = actual.metadata["search_label"]
        self.assertEqual(label["status"], "UNLABELED")
        self.assertEqual(label["refusal_code"], "fallback.crate_search_failed")
        self.assertNotIn("policy_target", label)
        self.assertNotIn("value_target", label)

    def test_invalid_positive_mass_on_illegal_action_is_not_a_label(self):
        teacher = Teacher(metadata={"engine_mcts": {"training_targets": {
            "status": "LABELED", "policy_target": [0.] * 8 + [1.], "value_target": 0.}}})
        actual = self.wrap(RawActor(), teacher).select_action_with_context(context(), rng=random.Random(1))
        self.assertEqual(actual.metadata["search_label"]["status"], "UNLABELED")

    def test_capture_failure_remains_fatal_but_ordinary_teacher_error_is_recorded(self):
        from pokezero.failure_capture import CaptureFailed
        with self.assertRaises(CaptureFailed):
            self.wrap(RawActor(), Teacher(failure=CaptureFailed("disk failure"))).select_action_with_context(
                context(), rng=random.Random(1))
        actual = self.wrap(RawActor(), Teacher(failure=ValueError("captured refusal"))).select_action_with_context(
            context(), rng=random.Random(1))
        self.assertEqual(actual.metadata["search_label"]["error_type"], "ValueError")

    def test_teacher_configuration_error_is_not_silently_masked(self):
        def missing_runtime():
            raise ValueError("missing native training target capability")
        wrapped = with_search_labels(RawActor(), fraction=1., teacher_factory=missing_runtime,
                                    teacher_identity={})
        with self.assertRaisesRegex(ValueError, "missing native"):
            wrapped.select_action_with_context(context(), rng=random.Random(1))

    def test_invalid_fractions_and_values_refuse(self):
        for fraction in (-1., 1.1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                select_search_label(context(), fraction)
        for value in (-1.1, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_search_targets([1.] + [0.] * 8, value, context().observation.legal_action_mask)

    def test_unseeded_or_unaddressed_context_cannot_silently_label(self):
        for c in (context(seed=None), context(seat="unknown"), context(decision=-1)):
            with self.subTest(context=c), self.assertRaisesRegex(ValueError, "seeded, seat-addressed"):
                select_search_label(c, .1)


if __name__ == "__main__":
    unittest.main()
