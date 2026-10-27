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
    data['last_action_index'] = [-1, -1]
    metadata = {"vocabulary": {
        "species": ["_none", "ditto"], "types": ["_none", "normal", "fire", "ice"],
        "abilities": ["_none", "truant"], "items": ["_none", "choiceband"],
        "moves": ["_none", "tackle", "encore", "counter", "hiddenpowerfire", "solarbeam", "return", "lowkick", "transform"],
    }}
    return data, metadata


def action_fixture():
    data, meta = fixture()
    meta["vocabulary"]["moves"] += ["splash", "growl", "recover"]
    meta["vocabulary"]["species"] += ["slaking", "pikachu", "blissey", "snorlax", "skarmory"]
    for side in range(2):
        for p in range(6):
            data["species"][side][p] = p+1
            data["moves"][side][p] = [1, 9, 10, 11]
    return data, meta


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
    def test_wish_history_maps_exactly_including_expired_and_zero_amount(self):
        for counter,amount in ((0,0),(0,150),(1,0),(1,150),(2,50),(127,32767)):
            data,meta=fixture()
            data['wish_counter']=[counter,0]
            data['wish_amount']=[amount,0]
            native=self.construct(data,meta)
            self.assertEqual(native.state.side_one.wish,(counter,amount))
            left=f.canonical_jax_future(data,None,meta,certificate=native.certificate)
            right=f.canonical_native_future(native.state,native.certificate)
            self.assertEqual(left.key,right.key)
            native.state.side_one.wish=(counter,amount+1 if amount<32767 else 0)
            changed=f.canonical_native_future(native.state,native.certificate)
            self.assertIsInstance(changed,f.CanonicalFuture,type(changed).__name__)
            self.assertNotEqual(left.key,changed.key)

    def test_missing_and_range_invalid_wish_history_is_never_reconstructed(self):
        data,meta=fixture();data.pop('wish_amount',None)
        refused=f.native_input_from_jax(data,None,meta,module=FAKE)
        self.assertIsInstance(refused,f.Refused,type(refused).__name__)
        self.assertIn('WISH_HISTORY_REQUIRED',{g.code for g in refused.gaps})
        for counter,amount in ((-1,0),(128,0),(1,-1),(0,32768),(True,0),(0,True)):
            data,meta=fixture();data['wish_counter']=[counter,0];data['wish_amount']=[amount,0]
            refused=f.native_input_from_jax(data,None,meta,module=FAKE)
            self.assertIsInstance(refused,f.Refused,type(refused).__name__)
            self.assertIn('INVALID_WISH_HISTORY',{g.code for g in refused.gaps})
        for counter,amount in ((-1,0),(128,0),(1,-1),(0,32768)):
            native=self.construct()
            native.state.side_one.wish=(counter,amount)
            refused=f.canonical_native_future(native.state,native.certificate)
            self.assertIsInstance(refused,f.Refused,type(refused).__name__)
            self.assertIn('INVALID_NATIVE_WISH_HISTORY',{g.code for g in refused.gaps})

    def test_explicit_last_action_distinguishes_none_switch_slot_and_struggle(self):
        for kind, index, expected in ((0, -1, 'move:none'), (1, 0, 'switch:0'),
                                      (2, 2, 'move:2'), (3, -1, 'move:struggle')):
            data, meta = fixture()
            data['moves'][1][5][0] = 2  # Bench Encore enables native tracking.
            data['last_action_kind'] = [kind, 0]
            data['last_action_index'] = [index, -1]
            data['active_turns'] = [7, 0]  # Never infer Switch from elapsed age.
            native = self.construct(data, meta)
            self.assertEqual(native.state.side_one.last_used_move, expected)
            self.assertEqual(native.state.side_two.last_used_move, 'move:none')
            left = f.canonical_jax_future(data, None, meta, certificate=native.certificate)
            right = f.canonical_native_future(native.state, native.certificate)
            self.assertEqual(left.key, right.key)

    def test_missing_or_invalid_last_action_history_is_never_guessed(self):
        for missing in ('last_action_kind', 'last_action_index'):
            data, meta = fixture()
            data.pop(missing, None)
            refused = f.native_input_from_jax(data, None, meta, module=FAKE)
            self.assertIsInstance(refused, f.Refused)
            self.assertIn('LAST_ACTION_HISTORY_REQUIRED', {g.code for g in refused.gaps})
        for kind, index in ((-1,-1), (4,0), (True,0), (0,0), (1,-1), (1,6),
                            (2,-1), (2,4), (3,0), (2,True)):
            data, meta = fixture()
            data['last_action_kind'] = [kind, 0]
            data['last_action_index'] = [index, -1]
            refused = f.native_input_from_jax(data, None, meta, module=FAKE)
            self.assertIsInstance(refused, f.Refused)
            self.assertIn('INVALID_LAST_ACTION_HISTORY', {g.code for g in refused.gaps})

    def test_diagnostic_identity_is_inactive_and_bench_fakeout_enables_history(self):
        data,meta=fixture()
        meta['vocabulary']['moves'].append('fakeout')
        data['moves'][1][5][0]=len(meta['vocabulary']['moves'])-1
        data['last_action_kind']=[2,0];data['last_action_index']=[3,-1]
        keys=[]
        for identity in (0,1,2):
            data['last_move'][0][0]=identity
            native=self.construct(data,meta)
            self.assertTrue(native.certificate.conditional_last)
            self.assertEqual(native.state.side_one.last_used_move,'move:3')
            keys.append(f.canonical_jax_future(data,None,meta,certificate=native.certificate).key)
        self.assertEqual(keys[0],keys[1]);self.assertEqual(keys[1],keys[2])

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

    def test_native_original_ability_cannot_contradict_its_certificate(self):
        native=self.construct()
        native.state.side_two.pokemon[5].base_ability='truant'
        for boundary in ('ordinary','suspended'):
            refused=f._canonical_native_future(native.state,native.certificate,boundary=boundary)
            self.assertIsInstance(refused,f.Refused)
            self.assertIn('NATIVE_BASE_ABILITY_CHANGED',{g.code for g in refused.gaps})
            self.assertIn('TRUANT',refused.raw_native.upper())

    def test_original_ability_is_explicit_and_certificate_bound(self):
        data,meta=fixture()
        data['base_ability'][0][0]=1
        native=self.construct(data,meta)
        self.assertEqual((native.state.side_one.pokemon[0].ability,
            native.state.side_one.pokemon[0].base_ability),('none','truant'))
        data['base_ability'][0][0]=0
        refused=f.canonical_jax_future(data,None,meta,certificate=native.certificate)
        self.assertIn('RETAINED_BASE_ABILITY_CHANGED',{g.code for g in refused.gaps})
        data,meta=fixture();del data['base_ability']
        refused=f.native_input_from_jax(data,None,meta,module=FAKE)
        self.assertEqual([g.code for g in refused.gaps],['ORIGINAL_BASE_ABILITY_REQUIRED'])
        for bad in (-1,2,True):
            data,meta=fixture();data['base_ability'][0][0]=bad
            self.assertIsInstance(f.native_input_from_jax(data,None,meta,module=FAKE),f.Refused)

    def test_permanent_weather_clock_convention_retains_raw_and_positive_clocks(self):
        for weather in range(5):
            for clock in (0,1,2,5,127):
                data,meta=fixture();data.update(weather=weather,weather_counter=clock)
                before=deepcopy(data);native=self.construct(data,meta)
                expected=-1 if weather and clock==0 else clock
                self.assertEqual(native.state.weather_turns_remaining,expected)
                left=f.canonical_jax_future(data,None,meta,certificate=native.certificate)
                right=f.canonical_native_future(native.state,native.certificate)
                self.assertEqual(left.key,right.key)
                self.assertEqual(dict(left.key)['weather_turns_remaining'],expected)
                self.assertEqual(dict(left.raw_jax)['weather_counter'],clock)
                self.assertEqual(dict(native.certificate.input_fields)['weather_counter'],clock)
                self.assertEqual(f._native_wire(right.raw_native)['weather_turns_remaining'],expected)
                self.assertEqual(data,before)

    def test_native_different_weather_clocks_are_not_normalized_away(self):
        data,meta=fixture();data.update(weather=1,weather_counter=0)
        native=self.construct(data,meta);keys=[]
        for clock in (-1,0,1,2):
            native.state.weather_turns_remaining=clock
            result=f.canonical_native_future(native.state,native.certificate)
            self.assertIsInstance(result,f.CanonicalFuture,result)
            self.assertEqual(dict(result.key)['weather_turns_remaining'],clock)
            self.assertEqual(f._native_wire(result.raw_native)['weather_turns_remaining'],clock)
            keys.append(result.key)
        self.assertEqual(len(set(keys)),4)

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
        self.assertEqual(native.state.side_one.last_used_move, "move:none")

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

    def test_protect_constructor_maps_denominator_not_streak_identity(self):
        for denominator, streak in ((0, 0), (2, 1), (4, 2), (8, 3)):
            data, meta = fixture()
            data["protect_counter"][0][0] = denominator
            native = self.construct(data, meta)
            self.assertEqual(native.state.side_one.side_conditions.protect, streak)
        for invalid in (-1, 1, 3, 16):
            data, meta = fixture()
            data["protect_counter"][0][0] = invalid
            out = f.native_input_from_jax(data, None, meta, module=FAKE)
            self.assertIn("INVALID_PROTECT_DENOMINATOR", {g.code for g in out.gaps})

    def test_protect_posttransition_capped_quotient_keeps_both_raw_states(self):
        data, meta = fixture()
        native = self.construct(data, meta)
        data["protect_counter"][0][0] = 8
        native.state.side_one.side_conditions.protect = 9
        left = f.canonical_jax_future(data, None, meta, certificate=native.certificate)
        right = f.canonical_native_future(native.state, native.certificate)
        self.assertEqual(left.key, right.key)
        self.assertEqual(dict(left.raw_jax)["protect_counter"][0][0], 8)
        self.assertEqual(f._native_wire(right.raw_native)["side_one"]["side_conditions"]["protect"], 9)

    def test_toxic_saturation_uses_future_equivalence_not_raw_identity(self):
        for counter in (0, 1, 13, 14, 15):
            data, meta = fixture()
            data["status"][0][0] = 4
            data["status_counter"][0][0] = counter
            native = self.construct(data, meta)
            self.assertEqual(native.state.side_one.side_conditions.toxic_count, min(counter, 14))
            left = f.canonical_jax_future(data, None, meta, certificate=native.certificate)
            right = f.canonical_native_future(native.state, native.certificate)
            self.assertEqual(left.key, right.key)
            self.assertEqual(dict(left.raw_jax)["status_counter"][0][0], counter)

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

    def test_actions_use_exact_native_slot_aliases_and_bare_switch_ids(self):
        data, meta = action_fixture()
        data["moves"][0][0][0] = 4
        data["hidden_power_type"][0][0] = 3
        native = self.construct(data, meta)
        result = f.native_actions_from_jax(data, None, (0, 4), native)
        self.assertIsInstance(result, f.NativeActions)
        self.assertEqual(result.choices, ("hiddenpowerice70", "slaking"))

    def test_actions_refuse_duplicate_move_even_when_the_other_slot_is_disabled(self):
        data, meta = action_fixture()
        data["moves"][0][0][1] = 1
        data["disabled"][0][0][1] = True
        native = self.construct(data, meta)
        out = f.native_actions_from_jax(data, None, (0, 0), native)
        self.assertIn("DUPLICATE_MOVE_ID", {g.code for g in out.gaps})
        self.assertEqual(out.raw_actions, (0, 0))

    def test_actions_refuse_switch_ambiguity_even_from_fainted_duplicate(self):
        data, meta = action_fixture()
        data["species"][0][2] = data["species"][0][1]
        data["hp"][0][2] = 0
        native = self.construct(data, meta)
        out = f.native_actions_from_jax(data, None, (4, 0), native)
        self.assertIn("DUPLICATE_OR_RESERVED_SWITCH_ID", {g.code for g in out.gaps})

    def test_waiter_is_not_struggle_and_exhaustion_is_not_wait(self):
        data, meta = action_fixture()
        data["pp"][0][0] = [0]*4
        native = self.construct(data, meta)
        self.assertEqual(f.native_actions_from_jax(data, None, (0, 0), native).choices, ("struggle", "tackle"))
        data, meta = action_fixture()
        data["hp"][0][0] = 0
        data["forced_switch"][0] = True
        native = self.construct(data, meta)
        self.assertEqual(f.native_actions_from_jax(data, None, (4, 0), native).choices, ("slaking", "none"))

    def test_action_legality_trap_and_exact_root_binding(self):
        data, meta = action_fixture()
        meta["vocabulary"]["abilities"].append("shadowtag")
        data["ability"][1][0] = 2
        native = self.construct(data, meta)
        out = f.native_actions_from_jax(data, None, (4, 0), native)
        self.assertIn("ILLEGAL_SWITCH", {g.code for g in out.gaps})
        out = f.native_actions_from_jax(data, None, (9, 0), native)
        self.assertIn("INVALID_NINE_ACTION_REQUEST", {g.code for g in out.gaps})
        data["pp"][0][0][0] -= 1
        out = f.native_actions_from_jax(data, None, (0, 0), native)
        self.assertIn("ACTION_ROOT_BINDING_CHANGED", {g.code for g in out.gaps})


if __name__ == "__main__":
    unittest.main()
