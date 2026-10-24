"""Authored correspondence guards; no native measurement or pool credit."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from gpu_engine import future_state as f


def nested(shape, value=0):
    return [nested(shape[1:], value) for _ in range(shape[0])] if shape else value


def fixture():
    data = {name: nested(shape) for name, shape in f.SHAPES.items()}
    for name, value in (("level", 100), ("hp", 1000), ("max_hp", 1000),
                        ("happiness", 255), ("hidden_power_power", 70), ("pp", 8),
                        ("moves", 1), ("species", 1), ("trap_source", -1)):
        data[name] = nested(f.SHAPES[name], value)
    for side in range(2):
        for p in range(6):
            data["types"][side][p] = [1, 0]
            data["stats"][side][p] = [1000, 200, 200, 200, 200, 301 if side == 0 else 199]
    data.update(queued_actor=-1, done=False, winner=-1)
    metadata = {"vocabulary": {
        "species": ["_none", "ditto"], "types": ["_none", "normal", "fire", "ice"],
        "abilities": ["_none", "truant"], "items": ["_none", "choiceband"],
        "moves": ["_none", "tackle", "encore", "counter", "hiddenpowerfire", "solarbeam", "return", "lowkick", "transform"],
    }}
    return data, metadata


def wire_bool(value):
    return str(bool(value)).lower()


def wire_mon(mon):
    # Independent literal order from pinned Rust Pokemon::serialize_base.
    values = [mon.id, mon.level, *mon.types, *mon.base_types, mon.hp, mon.maxhp,
        mon.ability, mon.base_ability, mon.item, mon.nature, ";".join(map(str, mon.evs)),
        mon.attack, mon.defense, mon.special_attack, mon.special_defense, mon.speed,
        mon.status, mon.rest_turns, mon.sleep_turns, mon.weight_kg]
    values.extend(f"{m.id};{wire_bool(m.disabled)};{m.pp}" for m in mon.moves)
    values.extend((wire_bool(mon.terastallized), mon.tera_type, mon.gender))
    if mon.pre_transform:
        values.append(mon.pre_transform)
    if mon.rest_sleep_pending_refund:
        values.append(f"refund:{mon.rest_sleep_pending_refund}")
    return ",".join(map(str, values))


class FakeState(SimpleNamespace):
    def to_string(self):
        parts = []
        for side in (self.side_one, self.side_two):
            values = [wire_mon(m) for m in side.pokemon]
            values += [side.active_index,
                ";".join(str(getattr(side.side_conditions, k)) for k in f.CONDITIONS),
                ":".join(sorted(side.volatile_statuses)) + (":" if side.volatile_statuses else ""),
                ";".join(str(getattr(side.volatile_status_durations, k)) for k in f.DURATIONS),
                side.substitute_health]
            values += [getattr(side, k+"_boost") for k in f.BOOST_NAMES]
            values += [*side.wish, *side.future_sight, wire_bool(side.force_switch),
                side.switch_out_move_second_saved_move, wire_bool(side.baton_passing),
                wire_bool(side.shed_tailing), wire_bool(side.force_trapped), side.last_used_move,
                wire_bool(side.slow_uturn_move)]
            parts.append("=".join(map(str, values)))
        parts += [f"{self.weather};{self.weather_turns_remaining}",
            f"{self.terrain};{self.terrain_turns_remaining}",
            f"{wire_bool(self.trick_room)};{self.trick_room_turns_remaining}", wire_bool(self.team_preview)]
        return "/".join(parts)


FAKE = SimpleNamespace(State=FakeState, Side=SimpleNamespace, Pokemon=SimpleNamespace,
    Move=SimpleNamespace, SideConditions=SimpleNamespace, VolatileStatusDurations=SimpleNamespace)


class FutureStateTests(unittest.TestCase):
    def construct(self, data=None, meta=None):
        original, metadata = fixture()
        out = f.native_input_from_jax(original if data is None else data, None,
            metadata if meta is None else meta, module=FAKE)
        self.assertIsInstance(out, f.NativeInput, out)
        return out

    def test_every_current_jax_field_has_an_explicit_policy_and_shape(self):
        tree = ast.parse((Path(__file__).parents[1]/"gpu_engine/state.py").read_text())
        state = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BattleState")
        fields = {node.target.id for node in state.body if isinstance(node, ast.AnnAssign)}
        self.assertFalse(fields - f.FIELD_POLICY.keys())
        self.assertEqual(set(f.FIELD_POLICY), set(f.SHAPES))
        self.assertEqual(set(f.FIELD_POLICY), set(f.NATIVE_PATHS))

    def test_complete_constructor_and_both_keys_cover_all12_mons(self):
        data, meta = fixture()
        before = deepcopy(data)
        native = self.construct(data, meta)
        left = f.canonical_jax_future(data, None, meta, certificate=native.certificate)
        right = f.canonical_native_future(native.state, native.certificate)
        self.assertEqual(left.key, right.key)
        self.assertEqual(data, before)
        self.assertIn("side_two.pokemon[5].moves[3].pp", dict(left.key))
        self.assertIn("side_one.pokemon[0].base_ability", dict(left.key))
        self.assertEqual(set(dict(native.certificate.field_policies)), set(data))

    def test_status_boost_item_hazard_weather_and_toxic_are_not_anchor_only(self):
        data, meta = fixture()
        data["status"][0][0] = 4
        data["status_counter"][0][0] = 3
        data["status"][1][0] = 1
        data["item"][0][0] = 1
        data["boosts"][1][0] = [2, -1, 3, 0, -2, 1, 0]
        data["spikes"] = [2, 3]
        data.update(weather=3, weather_counter=4)
        native = self.construct(data, meta)
        self.assertEqual(native.state.side_one.side_conditions.toxic_count, 3)
        self.assertEqual(native.state.side_two.attack_boost, 2)
        self.assertEqual(native.state.weather, "sand")

    def test_all_fields_not_hp_pp_projection_and_raw_values_retained(self):
        data, meta = fixture()
        native = self.construct(data, meta)
        left = f.canonical_jax_future(data, None, meta, certificate=native.certificate)
        native.state.side_two.pokemon[4].ability = "truant"
        right = f.canonical_native_future(native.state, native.certificate)
        self.assertEqual(f.field_differences(left, right), (("side_two.pokemon[4].ability", "none", "truant"),))
        self.assertTrue(right.raw_native)
        self.assertEqual(dict(left.raw_jax)["pp"], f._freeze(data["pp"]))

    def test_conditional_tracking_comes_from_all12_including_bench_moves(self):
        data, meta = fixture()
        data["moves"][1][5][0] = 3
        native = self.construct(data, meta)
        self.assertTrue(native.certificate.conditional_damage)
        self.assertFalse(native.certificate.conditional_last)
        data["moves"][0][4][0] = 2
        native = self.construct(data, meta)
        self.assertTrue(native.certificate.conditional_last)
        self.assertEqual(native.state.side_one.last_used_move, "switch:0")

    def test_hidden_power_uses_actual_type_and_supported_power_not_alias_text(self):
        data, meta = fixture()
        data["moves"][0][0][0] = 4  # metadata alias Fire; actual retained type Ice
        data["hidden_power_type"][0][0] = 3
        native = self.construct(data, meta)
        self.assertEqual(native.state.side_one.pokemon[0].moves[0].id, "hiddenpowerice70")
        data["hidden_power_power"][0][0] = 60
        native = self.construct(data, meta)
        self.assertEqual(native.state.side_one.pokemon[0].moves[0].id, "hiddenpowerice60")
        data["hidden_power_power"][0][0] = 68
        out = f.native_input_from_jax(data, None, meta, module=FAKE)
        self.assertIn("NATIVE_POWER_CONVENTION", {g.code for g in out.gaps})

    def test_stale_draws_are_explicit_but_suspended_draws_are_never_dropped(self):
        data, meta = fixture()
        native = self.construct(data, meta)
        data["residual_draws"][0][3] = 99
        data["queued_actions"] = [3, 2]
        left = f.canonical_jax_future(data, None, meta, certificate=native.certificate)
        self.assertIsInstance(left, f.CanonicalFuture)
        data["pending_residual"] = True
        out = f.canonical_jax_future(data, None, meta, certificate=native.certificate)
        self.assertIsInstance(out, f.Refused)
        self.assertIn("LIVE_SUSPENSION_INTERFACE", {g.code for g in out.gaps})
        self.assertEqual(dict(out.raw_jax)["residual_draws"][0][3], 99)

    def test_named_latent_timer_rest_and_bench_gaps(self):
        for field, value in (("confusion_counter", 2), ("encore_counter", 3), ("status", 5)):
            data, meta = fixture()
            data[field][0][0] = value
            out = f.native_input_from_jax(data, None, meta, module=FAKE)
            self.assertIsInstance(out, f.Refused)
            self.assertTrue(any("DURATION" in g.code for g in out.gaps))
        data, meta = fixture()
        data["boosts"][0][1][0] = 1
        out = f.native_input_from_jax(data, None, meta, module=FAKE)
        self.assertIn("BENCH_STATE_NOT_REPRESENTED", {g.code for g in out.gaps})

    def test_unknown_nonzero_native_fields_refuse_not_disappear(self):
        mutations = (("side_conditions", "reflect", 4), ("volatile_status_durations", "yawn", 1))
        for parent, field, value in mutations:
            native = self.construct()
            setattr(getattr(native.state.side_one, parent), field, value)
            out = f.canonical_native_future(native.state, native.certificate)
            self.assertIsInstance(out, f.Refused)
            self.assertIn(field, " ".join(g.field for g in out.gaps))
            self.assertEqual(out.raw_native, native.state.to_string())

    def test_native_latent_branch_is_not_combined_with_an_equal_hp_pp_branch(self):
        native = self.construct()
        native.state.side_one.volatile_statuses.add("mustrecharge")
        out = f.canonical_native_future(native.state, native.certificate)
        self.assertIsInstance(out, f.Refused)
        self.assertIn("mustrecharge", out.gaps[0].detail)

    def test_new_schema_shape_binding_or_conditional_context_never_silently_admits(self):
        data, meta = fixture()
        native = self.construct(data, meta)
        altered = deepcopy(meta)
        altered["vocabulary"]["species"][1] = "slaking"
        out = f.canonical_jax_future(data, None, altered, certificate=native.certificate)
        self.assertIn("METADATA_BINDING_CHANGED", {g.code for g in out.gaps})
        data["moves"][0][1][0] = 3
        out = f.canonical_jax_future(data, None, meta, certificate=native.certificate)
        self.assertIn("CONDITIONAL_MECHANICS_CHANGED", {g.code for g in out.gaps})
        data, meta = fixture()
        data["hp"][0].append(1000)
        self.assertIsInstance(f.native_input_from_jax(data, None, meta, module=FAKE), f.Refused)
        data, meta = fixture()
        data["unreviewed_timer"] = 0
        self.assertIsInstance(f.native_input_from_jax(data, None, meta, module=FAKE), f.Refused)


if __name__ == "__main__":
    unittest.main()
