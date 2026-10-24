"""Binding fixtures; fake native/model objects do not certify native fidelity."""
import json
import os
import struct
import unittest
from unittest.mock import patch

from pokezero.shallow_exact import ExactSearchRefused
from pokezero.shallow_exact_native import NativeExactBackend, NativePosition, validate_encoder_frame


class Fold:
    def __init__(self, lines=()):
        self.lines = list(lines)
    def to_payload(self):
        return {"lines": self.lines.copy()}
    @classmethod
    def from_payload(cls, payload):
        return cls(payload["lines"])
    def advance_in_place(self, lines):
        self.lines.extend(lines)


class Native:
    FoldState = Fold
    def __init__(self):
        self.calls = []
        self.branches = [{"percentage": 25, "post_state": "a", "events": ["|move|a"],
                          "turn_completed": True, "lossy": [], "attribution_unsafe": False},
                         {"percentage": 75, "post_state": "b", "events": ["|move|b"],
                          "turn_completed": False, "lossy": [], "attribution_unsafe": False}]
        for branch in self.branches:
            branch["post"] = {"p1": {"pokemon": [{"hp": 10}]}, "p2": {"pokemon": [{"hp": 10}]}}
    def env_battle_over(self, state):
        return {"win": 1, "loss": -1}.get(state, 0)
    def env_options(self, state, *, root):
        return json.dumps({"p1": ["surf"], "p2": ["switch blissey"]})
    def branch_events(self, state, row, col, ctx, **kwargs):
        self.calls.append((state, row, col, json.loads(ctx), kwargs))
        return json.dumps({"branches": self.branches})


class Encoder:
    def __init__(self):
        self.calls = []
    def encode_leaf(self, state, fold, turn, lines, authoritative):
        self.calls.append((state, fold.lines.copy(), turn, lines, authoritative))
        return {"categorical_ids": struct.pack("<ii", 3, 5),
                "numeric_features": struct.pack("<dd", 0.5, 0.75),
                "token_type_ids": struct.pack("<h", 2), "attention_mask": b"\x01",
                "legal_action_mask": b"\x01\x00"}


class Model:
    def eval_obs_flat(self, count, *arrays):
        self.last = (count, arrays)
        return [0.25] * count, [], [], 9


class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"POKEZERO_ENUMERATE_ROLLS": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.native, self.encoder, self.model = Native(), Encoder(), Model()
        self.backend = NativeExactBackend(self.native, self.model)
        self.root = NativePosition("root", {"p1": ("starmie",), "p2": ("snorlax", "blissey")},
                                   7, Fold(["|root|event"]), self.encoder, "side_one", ("|old|line",))

    def test_child_histories_folds_and_turns_do_not_alias(self):
        a, b = list(self.backend.outcomes(self.root, "surf", "switch blissey"))
        self.assertEqual((a.probability, b.probability), (0.25, 0.75))
        self.assertEqual((a.state.turn, b.state.turn), (8, 7))
        self.assertEqual(self.root.fold.lines, ["|root|event"])
        self.assertEqual(a.state.fold.lines, ["|root|event", "|move|a"])
        self.assertEqual(b.state.fold.lines, ["|root|event", "|move|b"])
        self.assertEqual(a.state.lines, ("|old|line", "|move|a"))
        a.state.fold.advance_in_place(["mutated"])
        self.assertNotIn("mutated", b.state.fold.lines)
        self.assertEqual(self.native.calls[0][1:3], ("surf", "blissey"))
        self.assertEqual(self.native.calls[0][3]["turn"], 7)
        self.assertTrue({"native_transition_render_serialize", "fold_history_advance"}
                        <= self.backend.stage_seconds.keys())

    def test_value_frame_and_byte_shapes(self):
        from dataclasses import replace
        reflected = replace(self.root, self_side="side_two")
        values = self.backend.evaluate_leaves([self.root, reflected])
        self.assertEqual(values, [0.25, -0.25])
        self.assertEqual(self.model.last, (2, ([3, 5, 3, 5], [0.5, 0.75, 0.5, 0.75],
                                             [2, 2], [True, True], [True, True])))
        self.assertEqual(self.encoder.calls[0],
                         ("root", ["|root|event"], 7, ["|old|line"], False))
        self.assertTrue({"cpu_encode_and_pack", "champion_forward"}
                        <= self.backend.stage_seconds.keys())

    def test_terminal_values_are_side_one_absolute(self):
        from dataclasses import replace
        self.assertEqual(self.backend.terminal_value(replace(self.root, teams_hp=((10,), (0,)))), 1)
        self.assertEqual(self.backend.terminal_value(replace(self.root, teams_hp=((0,), (10,)))), -1)
        self.assertEqual(self.backend.terminal_value(replace(self.root, state_str="loss", teams_hp=((0,), (0,)))), 0)
        with self.assertRaisesRegex(ExactSearchRefused, "attestation"):
            self.backend.terminal_value(replace(self.root, state_str="loss"))
        self.assertIsNone(self.backend.terminal_value(self.root))

    def test_packed_rows_preserve_all_effective_inputs_order_and_value_frame(self):
        from dataclasses import replace
        class PackedModel(Model):
            def eval_obs_packed(self, count, cats, nums, kinds, attention, history, *, byte_order):
                self.packed = (count, cats, nums, kinds, attention, history, byte_order)
                self.effective = (list(struct.unpack("<" + "i" * (len(cats) // 4), cats)),
                    [struct.unpack("<f", struct.pack("<f", value[0]))[0]
                     for value in struct.iter_unpack("<d", nums)],
                    list(struct.unpack("<" + "h" * (len(kinds) // 2), kinds)),
                    [value == 1 for value in attention], [value == 1 for value in history])
                return [0.25] * count, [], [], 9
        model = PackedModel()
        backend = NativeExactBackend(self.native, model, packed_leaves=True,
            observation_spec=dict(window=1, tokens=1, categorical_features=2, numeric_features=2))
        states = [self.root, replace(self.root, self_side="side_two"), self.root]
        self.assertEqual(backend.evaluate_leaves(states), self.backend.evaluate_leaves(states))
        self.assertEqual(model.effective, self.model.last[1])
        self.assertEqual(model.packed[-1], "little")
        self.assertEqual((backend.encode_calls, backend.forward_batches, backend.unique_leaf_evaluations), (3, 1, 3))
        self.assertTrue({"native_leaf_encode", "python_leaf_pack", "cpu_encode_and_pack", "champion_forward"}
                        <= backend.stage_seconds.keys())
        self.assertEqual(len(self.encoder.calls), 6)  # Both paths encode every state/history.

    def test_packed_pilot_refuses_incompatible_api_spec_type_or_width(self):
        class PackedModel(Model):
            def eval_obs_packed(self, *args, **kwargs):
                self.fail = True
                raise AssertionError("bad input must not reach forward")
        spec = dict(window=1, tokens=1, categorical_features=2, numeric_features=2)
        for options in ({"packed_leaves": 1}, {"packed_leaves": True},
                        {"packed_leaves": True, "observation_spec": spec},
                        {"packed_leaves": True, "memoize_leaves": True, "observation_spec": spec}):
            with self.subTest(options=options), self.assertRaises(ExactSearchRefused):
                NativeExactBackend(self.native, self.model, **options)
        model = PackedModel()
        for replacement in (b"short", bytearray(struct.pack("<ii", 3, 5))):
            backend = NativeExactBackend(self.native, model, packed_leaves=True, observation_spec=spec)
            original = self.encoder.encode_leaf
            def wrong(*args):
                return {**original(*args), "categorical_ids": replacement}
            with patch.object(self.encoder, "encode_leaf", wrong):
                with self.assertRaisesRegex(ExactSearchRefused, "bytes type or width"):
                    backend.evaluate_leaves([self.root])
            self.assertFalse(getattr(model, "fail", False))

    def test_fixture_label_cannot_override_actual_encoder_frame(self):
        row = json.dumps({"observation_metadata": {"showdown_slot": "p2"}})
        validate_encoder_frame(row, "side_two")
        with self.assertRaisesRegex(ExactSearchRefused, "showdown_slot"):
            validate_encoder_frame(row, "side_one")

    def test_lossy_unsafe_missing_post_and_large_fan_refuse(self):
        for mutation in ({"lossy": ["unsupported"]}, {"attribution_unsafe": True}, {"post_state": None}):
            with self.subTest(mutation=mutation):
                self.native.branches[0].update(mutation)
                with self.assertRaises(ExactSearchRefused):
                    list(self.backend.outcomes(self.root, "surf", "bodyslam"))
                self.native = Native()
                self.backend = NativeExactBackend(self.native, self.model)
        self.backend.max_outcomes_per_call = 1
        with self.assertRaisesRegex(ExactSearchRefused, "materialization cap"):
            list(self.backend.outcomes(self.root, "surf", "bodyslam"))

    def test_flag_and_window_require_explicit_compatible_worker(self):
        with patch.dict(os.environ, {"POKEZERO_ENUMERATE_ROLLS": "0"}):
            with self.assertRaisesRegex(ExactSearchRefused, "fresh worker"):
                NativeExactBackend(self.native, self.model)
        with self.assertRaisesRegex(ExactSearchRefused, "one-window"):
            NativeExactBackend(self.native, self.model, window=4)

    def test_memoized_nonduplicate_states_history_and_value_frames(self):
        from dataclasses import replace
        class RowEncoder(Encoder):
            def encode_leaf(self, state, fold, turn, lines, authoritative):
                row = super().encode_leaf(state, fold, turn, lines, authoritative)
                value = 0.25 if state == "a" else 0.75
                row["numeric_features"] = struct.pack("<dd", value, turn / 100)
                row["categorical_ids"] = struct.pack("<ii", 3, 5 + len(fold.lines))
                return row
        class ValueModel(Model):
            def __init__(self):
                self.calls = []
            def eval_obs_flat(self, count, *arrays):
                self.calls.append((count, arrays))
                return arrays[1][::2], [], [], 0
        model, encoder = ValueModel(), RowEncoder()
        backend = NativeExactBackend(self.native, model, memoize_leaves=True, model_sha256="a" * 64,
            observation_spec=dict(window=1, tokens=1, categorical_features=2, numeric_features=2))
        first = replace(self.root, state_str="a", encoder=encoder)
        second = replace(first, state_str="b")
        revealed = replace(first, fold=Fold(["|root|event", "|revealed|move"]))
        reflected = replace(first, self_side="side_two")
        backend.begin_decision()
        self.assertEqual(backend.evaluate_leaves([first, second, first, reflected]), [0.25, 0.75, 0.25, -0.25])
        self.assertEqual(backend.evaluate_leaves([second, revealed]), [0.75, 0.25])
        self.assertEqual((backend.encode_calls, backend.unique_leaf_evaluations, backend.leaf_cache_hits,
                          backend.forward_batches), (6, 3, 3, 2))
        self.assertEqual([call[0] for call in model.calls], [2, 1])
        self.assertEqual(len(encoder.calls), 6)  # No state/fold/history construction is skipped.
        backend.begin_decision()
        self.assertEqual(backend.evaluate_leaves([first]), [0.25])
        self.assertEqual((backend.encode_calls, backend.unique_leaf_evaluations, backend.leaf_cache_hits), (1, 1, 0))
        backend.model = ValueModel()
        with self.assertRaisesRegex(ExactSearchRefused, "binding changed"):
            backend.begin_decision()

    def test_memoization_requires_explicit_fixed_spec_and_model(self):
        for options in ({}, {"model_sha256": "unbound"},
                        {"model_sha256": "a" * 64, "observation_spec": {"window": 1}}):
            with self.subTest(options=options):
                with self.assertRaises(ExactSearchRefused):
                    NativeExactBackend(self.native, self.model, memoize_leaves=True, **options)

    def test_complete_nonduplicate_authored_worlds_keep_tree_numerics_and_counts(self):
        from dataclasses import replace
        from pokezero.shallow_exact import exact_depth_two
        class RowEncoder(Encoder):
            def encode_leaf(self, state, fold, turn, lines, authoritative):
                row = super().encode_leaf(state, fold, turn, lines, authoritative)
                row["numeric_features"] = struct.pack("<dd", 0.25 if state == "a" else 0.75,
                                                      len(fold.lines) / 10)
                return row
        class ValueModel(Model):
            def eval_obs_flat(self, count, *arrays):
                return arrays[1][::2], [], [], 0
        encoder = RowEncoder()
        worlds = [replace(self.root, state_str="world" + str(i), encoder=encoder,
                          fold=Fold(["|root|" + str(i)] * (i + 1))) for i in range(4)]
        plain = NativeExactBackend(Native(), ValueModel())
        cached = NativeExactBackend(Native(), ValueModel(), memoize_leaves=True, model_sha256="a" * 64,
            observation_spec=dict(window=1, tokens=1, categorical_features=2, numeric_features=2))
        left, right = [exact_depth_two(worlds, backend, batch_size=3) for backend in (plain, cached)]
        self.assertEqual(left, right)
        self.assertEqual((left.leaf_evaluations, cached.encode_calls), (16, 16))
        self.assertEqual((cached.unique_leaf_evaluations, cached.leaf_cache_hits), (8, 8))
        self.assertEqual((plain.unique_leaf_evaluations, plain.leaf_cache_hits), (16, 0))


if __name__ == "__main__":
    unittest.main()
