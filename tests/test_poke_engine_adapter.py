from __future__ import annotations

from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace
import unittest

from pokezero.poke_engine_adapter import (
    BattleSpec,
    MoveSpec,
    PokeEngineAttractUnsupportedError,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
    minimal_gen3_fixture,
    run_adapter_reversible_smoke,
)
from pokezero.poke_engine_backend import PokeEngineUnavailableError, probe_poke_engine


class RecordingState:
    """Records construction kwargs and supports the reversible round-trip."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        # A serialized form derived from the structure so apply/reverse can mutate it.
        self.serialized = "ORIG"

    @classmethod
    def from_string(cls, value: str) -> "RecordingState":
        state = cls()
        state.serialized = value
        return state

    def to_string(self) -> str:
        return self.serialized

    def apply_instructions(self, instruction: object) -> "RecordingState":
        after = RecordingState(**self.kwargs)
        after.serialized = self.serialized + instruction.delta
        return after

    def reverse_instructions(self, instruction: object) -> "RecordingState":
        after = RecordingState(**self.kwargs)
        delta = instruction.delta
        if delta and self.serialized.endswith(delta):
            after.serialized = self.serialized[: -len(delta)]
        else:
            after.serialized = self.serialized
        return after


def fake_construction_module(*, instructions: list | None = None) -> ModuleType:
    """A fake poke_engine exposing only the adapter construction surface."""

    module = ModuleType("poke_engine_fake")
    module.State = RecordingState
    module.Side = lambda **kwargs: SimpleNamespace(kind="Side", **kwargs)
    module.Pokemon = lambda **kwargs: SimpleNamespace(kind="Pokemon", **kwargs)
    module.Move = lambda **kwargs: SimpleNamespace(kind="Move", **kwargs)
    module.SideConditions = lambda **kwargs: SimpleNamespace(kind="SideConditions", **kwargs)
    module.generate_instructions = lambda *args, **kwargs: list(instructions or [])
    return module


def instruction(delta: str, percentage: float = 50.0) -> SimpleNamespace:
    return SimpleNamespace(delta=delta, percentage=percentage)


def attract_branch(*, moved: bool, percentage: float = 50.0) -> SimpleNamespace:
    """A minimal generate_instructions branch for the Attract patch probe."""

    instructions = ("Boost SideOne",) if moved else ("CantMove",)
    return SimpleNamespace(percentage=percentage, instruction_list=instructions)


class BuildPokeEngineStateTest(unittest.TestCase):
    def test_constructs_expected_kwargs_for_minimal_fixture(self) -> None:
        engine = fake_construction_module()
        state = build_poke_engine_state(minimal_gen3_fixture(), module=engine)

        self.assertIsInstance(state, RecordingState)
        self.assertEqual(state.kwargs["weather"], "none")
        self.assertEqual(state.kwargs["terrain"], "none")
        self.assertIs(state.kwargs["trick_room"], False)

        side_one = state.kwargs["side_one"]
        self.assertEqual(side_one.kind, "Side")
        # active_index must be a STRING for the real engine.
        self.assertEqual(side_one.active_index, "0")
        self.assertEqual(len(side_one.pokemon), 1)

        charmander = side_one.pokemon[0]
        self.assertEqual(charmander.id, "charmander")
        self.assertEqual(charmander.types, ("fire", "typeless"))
        self.assertEqual(charmander.hp, 100)
        self.assertEqual(charmander.maxhp, 100)
        self.assertEqual(charmander.status, "none")

        move_ids = [m.id for m in charmander.moves]
        self.assertEqual(move_ids, ["ember", "tackle"])
        self.assertTrue(all(m.kind == "Move" for m in charmander.moves))

        squirtle = state.kwargs["side_two"].pokemon[0]
        self.assertEqual(squirtle.id, "squirtle")
        self.assertEqual([m.id for m in squirtle.moves], ["watergun", "tackle"])

    def test_active_index_and_moves_preserved_for_multi_pokemon_side(self) -> None:
        engine = fake_construction_module()
        bench = replace(minimal_gen3_fixture().side_one.pokemon[0], id="charmeleon")
        active = minimal_gen3_fixture().side_one.pokemon[0]
        spec = replace(
            minimal_gen3_fixture(),
            side_one=SideSpec(pokemon=(bench, active), active_index=1),
        )

        state = build_poke_engine_state(spec, module=engine)
        side_one = state.kwargs["side_one"]

        self.assertEqual(side_one.active_index, "1")
        self.assertEqual([p.id for p in side_one.pokemon], ["charmeleon", "charmander"])
        self.assertEqual([m.id for m in side_one.pokemon[1].moves], ["ember", "tackle"])

    def test_optional_fields_only_passed_when_set(self) -> None:
        engine = fake_construction_module()
        base = minimal_gen3_fixture()
        without = build_poke_engine_state(base, module=engine)
        bare = without.kwargs["side_one"].pokemon[0]
        self.assertFalse(hasattr(bare, "ability"))
        self.assertFalse(hasattr(bare, "item"))
        self.assertFalse(hasattr(bare, "nature"))
        self.assertFalse(hasattr(bare, "gender"))

        decorated_member = replace(
            base.side_one.pokemon[0],
            ability="blaze",
            item="charcoal",
            nature="adamant",
            gender="F",
        )
        decorated = build_poke_engine_state(
            replace(base, side_one=SideSpec(pokemon=(decorated_member,), active_index=0)),
            module=engine,
        )
        member = decorated.kwargs["side_one"].pokemon[0]
        self.assertEqual(member.ability, "blaze")
        self.assertEqual(member.item, "charcoal")
        self.assertEqual(member.nature, "adamant")
        self.assertEqual(member.gender, "female")

    def test_side_conditions_built_via_engine_type(self) -> None:
        engine = fake_construction_module()
        base = minimal_gen3_fixture()
        spec = replace(
            base,
            side_one=SideSpec(
                pokemon=base.side_one.pokemon,
                active_index=0,
                side_conditions={"spikes": 2, "reflect": 1},
            ),
        )

        state = build_poke_engine_state(spec, module=engine)
        side_conditions = state.kwargs["side_one"].side_conditions
        self.assertEqual(side_conditions.kind, "SideConditions")
        self.assertEqual(side_conditions.spikes, 2)
        self.assertEqual(side_conditions.reflect, 1)

    def test_mono_type_padded_with_typeless(self) -> None:
        engine = fake_construction_module()
        state = build_poke_engine_state(minimal_gen3_fixture(), module=engine)
        self.assertEqual(state.kwargs["side_two"].pokemon[0].types, ("water", "typeless"))

    def test_dual_type_preserved(self) -> None:
        engine = fake_construction_module()
        base = minimal_gen3_fixture()
        member = replace(base.side_one.pokemon[0], types=("fire", "flying"))
        state = build_poke_engine_state(
            replace(base, side_one=SideSpec(pokemon=(member,), active_index=0)),
            module=engine,
        )
        self.assertEqual(state.kwargs["side_one"].pokemon[0].types, ("fire", "flying"))


class BuildPokeEngineStateValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = fake_construction_module()
        self.base = minimal_gen3_fixture()

    def _build(self, spec: object) -> object:
        return build_poke_engine_state(spec, module=self.engine)

    def test_rejects_non_battlespec(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self._build({"side_one": None})
        self.assertIn("BattleSpec", str(ctx.exception))

    def test_empty_side_raises_value_error(self) -> None:
        spec = replace(self.base, side_one=SideSpec(pokemon=(), active_index=0))
        with self.assertRaises(ValueError) as ctx:
            self._build(spec)
        self.assertIn("side_one", str(ctx.exception))
        self.assertIn("at least one Pokemon", str(ctx.exception))

    def test_invalid_gender_fails_closed(self) -> None:
        member = replace(self.base.side_one.pokemon[0], gender="unknown")
        spec = replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))
        with self.assertRaisesRegex(ValueError, "gender must be M, F, N, or None"):
            self._build(spec)

    def test_active_index_out_of_range_raises_value_error(self) -> None:
        spec = replace(
            self.base,
            side_one=SideSpec(pokemon=self.base.side_one.pokemon, active_index=3),
        )
        with self.assertRaises(ValueError) as ctx:
            self._build(spec)
        self.assertIn("out of range", str(ctx.exception))

    def test_active_index_wrong_type_raises_type_error(self) -> None:
        spec = replace(
            self.base,
            side_one=SideSpec(pokemon=self.base.side_one.pokemon, active_index="0"),
        )
        with self.assertRaises(TypeError) as ctx:
            self._build(spec)
        self.assertIn("active_index", str(ctx.exception))

    def test_empty_moves_raises_value_error(self) -> None:
        member = replace(self.base.side_one.pokemon[0], moves=())
        spec = replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))
        with self.assertRaises(ValueError) as ctx:
            self._build(spec)
        self.assertIn("moves", str(ctx.exception))

    def test_empty_species_id_raises_value_error(self) -> None:
        member = replace(self.base.side_one.pokemon[0], id="")
        spec = replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))
        with self.assertRaises(ValueError) as ctx:
            self._build(spec)
        self.assertIn("id", str(ctx.exception))

    def test_bare_string_types_raises_type_error(self) -> None:
        member = replace(self.base.side_one.pokemon[0], types="fire")
        spec = replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))
        with self.assertRaises(TypeError) as ctx:
            self._build(spec)
        self.assertIn("types", str(ctx.exception))

    def test_empty_types_raises_value_error(self) -> None:
        member = replace(self.base.side_one.pokemon[0], types=())
        spec = replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))
        with self.assertRaises(ValueError) as ctx:
            self._build(spec)
        self.assertIn("types", str(ctx.exception))

    def test_too_many_types_raises_value_error(self) -> None:
        member = replace(self.base.side_one.pokemon[0], types=("fire", "flying", "water"))
        spec = replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))
        with self.assertRaises(ValueError) as ctx:
            self._build(spec)
        self.assertIn("at most", str(ctx.exception))

    def test_trick_room_wrong_type_raises_type_error(self) -> None:
        spec = replace(self.base, trick_room="yes")
        with self.assertRaises(TypeError) as ctx:
            self._build(spec)
        self.assertIn("trick_room", str(ctx.exception))

    def _replace_active(self, **changes: object) -> object:
        member = replace(self.base.side_one.pokemon[0], **changes)
        return replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))

    def test_non_positive_level_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build(self._replace_active(level=0))
        self.assertIn("level", str(ctx.exception))

    def test_bool_level_raises_type_error(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self._build(self._replace_active(level=True))
        self.assertIn("level", str(ctx.exception))

    def test_non_int_hp_raises_type_error(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self._build(self._replace_active(hp=10.5))
        self.assertIn("hp", str(ctx.exception))

    def test_negative_hp_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build(self._replace_active(hp=-1))
        self.assertIn("hp", str(ctx.exception))

    def test_non_positive_maxhp_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build(self._replace_active(maxhp=0))
        self.assertIn("maxhp", str(ctx.exception))

    def test_hp_above_maxhp_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build(self._replace_active(hp=120, maxhp=100))
        self.assertIn("maxhp", str(ctx.exception))

    def test_non_positive_stat_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build(self._replace_active(speed=0))
        self.assertIn("speed", str(ctx.exception))

    def test_zero_hp_at_maxhp_is_allowed(self) -> None:
        # A fainted Pokemon (hp == 0) is valid as long as maxhp stays positive.
        self._build(self._replace_active(hp=0))

    def test_negative_move_pp_raises_value_error(self) -> None:
        member = replace(
            self.base.side_one.pokemon[0], moves=(MoveSpec(id="ember", pp=-1),)
        )
        spec = replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))
        with self.assertRaises(ValueError) as ctx:
            self._build(spec)
        self.assertIn("pp", str(ctx.exception))

    def test_bool_move_pp_raises_type_error(self) -> None:
        member = replace(
            self.base.side_one.pokemon[0], moves=(MoveSpec(id="ember", pp=True),)
        )
        spec = replace(self.base, side_one=SideSpec(pokemon=(member,), active_index=0))
        with self.assertRaises(TypeError) as ctx:
            self._build(spec)
        self.assertIn("pp", str(ctx.exception))

    def test_missing_construction_api_raises_unavailable(self) -> None:
        engine = fake_construction_module()
        del engine.Move
        with self.assertRaises(PokeEngineUnavailableError) as ctx:
            build_poke_engine_state(self.base, module=engine)
        self.assertIn("Move", str(ctx.exception))


class AttractPatchCapabilityTest(unittest.TestCase):
    def _attracted_spec(self) -> BattleSpec:
        base = minimal_gen3_fixture()
        return replace(
            base,
            side_one=replace(base.side_one, volatile_statuses=("attract",)),
        )

    def test_attract_build_requires_a_real_immobilization_branch(self) -> None:
        engine = fake_construction_module(
            instructions=[
                attract_branch(moved=True),
                attract_branch(moved=False),
            ]
        )

        state = build_poke_engine_state(self._attracted_spec(), module=engine)

        self.assertEqual(state.kwargs["side_one"].volatile_statuses, {"attract"})

    def test_attract_build_fails_closed_when_engine_accepts_but_ignores_it(self) -> None:
        engine = fake_construction_module(
            instructions=[attract_branch(moved=True, percentage=100.0)]
        )

        with self.assertRaises(PokeEngineAttractUnsupportedError) as caught:
            build_poke_engine_state(self._attracted_spec(), module=engine)

        self.assertIn("patched poke-engine", str(caught.exception))


class FakeModuleIsolationTest(unittest.TestCase):
    def test_no_real_poke_engine_import_when_module_supplied(self) -> None:
        # Guard: building with an explicit fake module must not import real poke_engine.
        had_real = "poke_engine" in sys.modules
        engine = fake_construction_module(instructions=[instruction("A")])
        build_poke_engine_state(minimal_gen3_fixture(), module=engine)
        run_adapter_reversible_smoke(module=engine)
        if not had_real:
            self.assertNotIn(
                "poke_engine",
                sys.modules,
                "adapter imported real poke_engine despite a supplied fake module",
            )


class AdapterReversibleSmokeTest(unittest.TestCase):
    def test_smoke_passes_with_mutating_reversible_branches(self) -> None:
        engine = fake_construction_module(
            instructions=[instruction("A"), instruction("B"), instruction("", 0.0)]
        )
        result = run_adapter_reversible_smoke(module=engine)
        self.assertEqual(result.instruction_count, 3)
        self.assertTrue(result.mutated_any_state)
        self.assertTrue(result.round_trip_ok)
        self.assertTrue(result.succeeded)

    def test_smoke_raises_when_no_instructions(self) -> None:
        engine = fake_construction_module(instructions=[])
        with self.assertRaises(PokeEngineUnavailableError) as ctx:
            run_adapter_reversible_smoke(module=engine)
        self.assertIn("no instructions", str(ctx.exception))

    def test_smoke_raises_clear_error_when_move_not_on_active(self) -> None:
        engine = fake_construction_module(instructions=[instruction("A")])
        with self.assertRaises(ValueError) as ctx:
            run_adapter_reversible_smoke(module=engine, move_one="surf")
        message = str(ctx.exception)
        self.assertIn("surf", message)
        self.assertIn("charmander", message)


class RealEngineIntegrationTest(unittest.TestCase):
    def test_real_fixture_builds_and_round_trips(self) -> None:
        probe = probe_poke_engine()
        if not probe.ready:
            self.skipTest("poke-engine is not installed/ready")

        import poke_engine

        state = build_poke_engine_state(minimal_gen3_fixture())
        original = state.to_string()
        self.assertIn("CHARMANDER", original)
        self.assertIn("SQUIRTLE", original)

        instructions = list(poke_engine.generate_instructions(state, "ember", "watergun"))
        self.assertGreater(len(instructions), 0, "expected instruction branches")

        mutated = False
        for instr in instructions:
            after = state.apply_instructions(instr)
            if after.to_string() != original:
                mutated = True
            restored = after.reverse_instructions(instr)
            self.assertEqual(restored.to_string(), original, "branch did not reverse cleanly")
        self.assertTrue(mutated, "expected at least one branch to mutate state")

        result = run_adapter_reversible_smoke()
        self.assertTrue(result.succeeded, result.summary())


if __name__ == "__main__":
    unittest.main()
