"""Public Struggle history must survive the producer and the Python ABI."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

from pokezero.engine_world import battle_spec_from_payload
from pokezero.poke_engine_adapter import (
    PokeEngineStruggleLastMoveUnsupportedError,
    build_poke_engine_state,
    minimal_gen3_fixture,
    require_struggle_last_move_support,
)
from tests.test_engine_world import _dex, _override, _payload
from tests.test_poke_engine_adapter import fake_construction_module


def engine_with_roundtrip():
    engine = fake_construction_module()

    class State:
        reads = 0

        def __init__(self, **kwargs):
            self.side_one = kwargs.get("side_one", SimpleNamespace(last_used_move=""))
            self.side_two = kwargs.get("side_two", SimpleNamespace(last_used_move=""))

        def to_string(self):
            return json.dumps([getattr(self.side_one, "last_used_move", ""),
                               getattr(self.side_two, "last_used_move", "")])

        @classmethod
        def from_string(cls, value):
            cls.reads += 1
            one, two = json.loads(value)
            return cls(side_one=SimpleNamespace(last_used_move=one),
                       side_two=SimpleNamespace(last_used_move=two))

    engine.State = State
    return engine


class StruggleLastMoveAdapterTest(unittest.TestCase):
    def test_reserved_token_reaches_both_sides(self):
        for seat in ("side_one", "side_two"):
            with self.subTest(seat=seat):
                spec = minimal_gen3_fixture()
                spec = replace(spec, **{seat: replace(getattr(spec, seat), last_used_move="move:struggle")})
                state = build_poke_engine_state(spec, module=engine_with_roundtrip())
                self.assertEqual(getattr(state, seat).last_used_move, "move:struggle")

    def test_public_producer_reaches_the_adapter_on_both_seats(self):
        for seat, attribute in (("p1", "side_one"), ("p2", "side_two")):
            with self.subTest(seat=seat):
                dex = _dex(); payload = _payload(dex)
                payload["sides"][seat]["lastUsedMove"] = "struggle"
                spec = battle_spec_from_payload(payload, _override(), dex=dex).spec
                state = build_poke_engine_state(spec, module=engine_with_roundtrip())
                self.assertEqual(getattr(state, attribute).last_used_move, "move:struggle")
                self.assertNotIn("struggle", [move.id for move in getattr(spec, attribute).pokemon[0].moves])

    def test_unsupported_producer_state_is_a_typed_refusal(self):
        dex = _dex(); payload = _payload(dex)
        payload["sides"]["p2"]["lastUsedMove"] = "struggle"
        spec = battle_spec_from_payload(payload, _override(), dex=dex).spec
        with self.assertRaises(PokeEngineStruggleLastMoveUnsupportedError):
            build_poke_engine_state(spec, module=fake_construction_module())

    def test_ordinary_numeric_tokens_are_unchanged(self):
        for token in ("move:0", "move:3", "switch:0", "switch:5"):
            with self.subTest(token=token):
                spec = minimal_gen3_fixture()
                state = build_poke_engine_state(replace(spec, side_one=replace(spec.side_one, last_used_move=token)),
                                                module=engine_with_roundtrip())
                self.assertEqual(state.side_one.last_used_move, token)

    def test_arbitrary_move_ids_are_still_rejected(self):
        for token in ("move:tackle", "switch:struggle", "move:Struggle", "move:-1"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                spec = minimal_gen3_fixture()
                build_poke_engine_state(replace(spec, side_one=replace(spec.side_one, last_used_move=token)),
                                        module=engine_with_roundtrip())

    def test_dropped_token_is_not_capability(self):
        engine = engine_with_roundtrip()
        engine.State.to_string = lambda self: "[]"
        with self.assertRaises(PokeEngineStruggleLastMoveUnsupportedError):
            require_struggle_last_move_support(engine)

    def test_roundtrip_rewrite_is_not_capability(self):
        engine = engine_with_roundtrip()
        engine.State.from_string = lambda value: engine.State()
        with self.assertRaises(PokeEngineStruggleLastMoveUnsupportedError):
            require_struggle_last_move_support(engine)

    def test_ordinary_parser_failure_is_a_typed_refusal(self):
        engine = engine_with_roundtrip()
        def fail(value): raise ValueError("unsupported token")
        engine.State.from_string = fail
        with self.assertRaises(PokeEngineStruggleLastMoveUnsupportedError):
            require_struggle_last_move_support(engine)

    def test_legacy_pyo3_panic_is_a_typed_refusal(self):
        engine = engine_with_roundtrip()
        panic = type("PanicException", (BaseException,), {"__module__": "pyo3_runtime"})
        def fail(self): raise panic("Invalid PokemonMoveIndex: struggle")
        engine.State.to_string = fail
        with self.assertRaises(PokeEngineStruggleLastMoveUnsupportedError):
            require_struggle_last_move_support(engine)

    def test_interrupts_and_other_base_failures_are_not_swallowed(self):
        for failure in (KeyboardInterrupt, SystemExit, BaseException):
            with self.subTest(failure=failure):
                engine = engine_with_roundtrip()
                def fail(self): raise failure()
                engine.State.to_string = fail
                with self.assertRaises(failure):
                    require_struggle_last_move_support(engine)

    def test_capability_is_cached_for_one_engine_module(self):
        engine = engine_with_roundtrip()
        require_struggle_last_move_support(engine)
        require_struggle_last_move_support(engine)
        self.assertEqual(engine.State.reads, 2)

    def test_unhashable_explicit_engine_is_supported(self):
        engine = SimpleNamespace(**vars(engine_with_roundtrip()))
        require_struggle_last_move_support(engine)

    def test_legacy_states_do_not_invoke_the_new_probe(self):
        engine = fake_construction_module()
        build_poke_engine_state(minimal_gen3_fixture(), module=engine)


if __name__ == "__main__":
    unittest.main()
