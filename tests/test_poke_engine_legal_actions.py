from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

from pokezero.poke_engine_adapter import build_poke_engine_state, minimal_gen3_fixture
from pokezero.poke_engine_legal_actions import (
    ENGINE_OPTION_PROVIDER_CANDIDATES,
    EngineLegalActions,
    LegalActionEquivalence,
    compare_legal_actions,
    engine_legal_actions,
    move_action_label,
    request_legal_actions,
    switch_action_label,
)
from pokezero.poke_engine_backend import probe_poke_engine


def move_request(moves, *, pokemon=None, trapped=False, maybe_trapped=False):
    """A singles 'move' request: one active row plus a side party list."""

    active = {"moves": list(moves)}
    if trapped:
        active["trapped"] = True
    if maybe_trapped:
        active["maybeTrapped"] = True
    return {
        "active": [active],
        "side": {"pokemon": list(pokemon or [])},
    }


def party_member(ident, *, active=False, condition="100/100"):
    return {"ident": ident, "active": active, "condition": condition}


def fixture_request(side_id: str):
    path = Path(__file__).parent / "fixtures" / "showdown" / "p2_seat_replay.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|request|"):
            continue
        request = json.loads(line.removeprefix("|request|"))
        if request.get("side", {}).get("id") == side_id:
            return request
    raise AssertionError(f"fixture has no request for side {side_id!r}")


# ---- request parser ------------------------------------------------------


class RequestLegalActionsTest(unittest.TestCase):
    def test_moves_in_request_order_skipping_disabled(self) -> None:
        request = move_request(
            [
                {"id": "ember", "disabled": False},
                {"id": "tackle", "disabled": True},
                {"id": "scratch"},
            ]
        )
        self.assertEqual(
            request_legal_actions(request),
            (move_action_label("ember"), move_action_label("scratch")),
        )

    def test_move_label_normalizes_display_name_fallback(self) -> None:
        request = move_request([{"move": "Water Gun"}])
        self.assertEqual(request_legal_actions(request), ("move:watergun",))

    def test_switches_exclude_active_and_fainted(self) -> None:
        request = move_request(
            [{"id": "ember"}],
            pokemon=[
                party_member("p1: Charmander", active=True),
                party_member("p1: Squirtle", condition="100/100"),
                party_member("p1: Bulbasaur", condition="0 fnt"),
                party_member("p1: Pikachu", condition="48/100"),
            ],
        )
        self.assertEqual(
            request_legal_actions(request),
            (move_action_label("ember"), switch_action_label(1), switch_action_label(3)),
        )

    def test_trapped_active_yields_no_switches(self) -> None:
        request = move_request(
            [{"id": "ember"}],
            pokemon=[
                party_member("p1: Charmander", active=True),
                party_member("p1: Squirtle"),
            ],
            trapped=True,
        )
        self.assertEqual(request_legal_actions(request), (move_action_label("ember"),))

    def test_maybe_trapped_active_still_offers_switches(self) -> None:
        request = move_request(
            [{"id": "ember"}],
            pokemon=[
                party_member("p1: Charmander", active=True),
                party_member("p1: Squirtle"),
            ],
            maybe_trapped=True,
        )
        self.assertEqual(
            request_legal_actions(request),
            (move_action_label("ember"), switch_action_label(1)),
        )

    def test_real_replay_request_shape_derives_move_and_switch_labels(self) -> None:
        request = fixture_request("p2")
        self.assertEqual(
            request_legal_actions(request),
            (
                move_action_label("flamethrower"),
                move_action_label("earthquake"),
                move_action_label("toxic"),
                switch_action_label(1),
                switch_action_label(3),
                switch_action_label(4),
                switch_action_label(5),
            ),
        )

    def test_force_switch_yields_only_switches(self) -> None:
        request = {
            "forceSwitch": [True],
            "active": [{"moves": [{"id": "ember"}]}],
            "side": {
                "pokemon": [
                    party_member("p1: Charmander", active=True, condition="0 fnt"),
                    party_member("p1: Squirtle"),
                    party_member("p1: Bulbasaur"),
                ]
            },
        }
        self.assertEqual(
            request_legal_actions(request),
            (switch_action_label(1), switch_action_label(2)),
        )

    def test_wait_request_has_no_actions(self) -> None:
        self.assertEqual(request_legal_actions({"wait": True}), ())

    def test_rejects_doubles_active(self) -> None:
        request = {"active": [{"moves": []}, {"moves": []}], "side": {"pokemon": []}}
        with self.assertRaises(ValueError) as ctx:
            request_legal_actions(request)
        self.assertIn("singles only", str(ctx.exception))

    def test_rejects_doubles_force_switch(self) -> None:
        request = {"forceSwitch": [True, False], "side": {"pokemon": []}}
        with self.assertRaises(ValueError) as ctx:
            request_legal_actions(request)
        self.assertIn("singles only", str(ctx.exception))


# ---- engine option provider (fake) ---------------------------------------


def move_option(move_index: int) -> SimpleNamespace:
    return SimpleNamespace(kind="move", move_index=move_index)


def switch_option(switch_index: int) -> SimpleNamespace:
    return SimpleNamespace(kind="switch", switch_index=switch_index)


def fake_engine_state(side_one_party, *, active_index=0):
    """Minimal navigable engine-style state for option resolution.

    side_one_party is a list of (species_id, [move_ids]) tuples.
    """

    def make_side(party):
        pokemon = [
            SimpleNamespace(
                id=species,
                moves=[SimpleNamespace(id=move_id) for move_id in moves],
            )
            for species, moves in party
        ]
        return SimpleNamespace(active_index=active_index, pokemon=pokemon)

    return SimpleNamespace(side_one=make_side(side_one_party), side_two=make_side([]))


class EngineLegalActionsTest(unittest.TestCase):
    def test_resolves_options_against_engine_state(self) -> None:
        state = fake_engine_state(
            [
                ("charmander", ["ember", "tackle"]),
                ("squirtle", ["watergun"]),
            ]
        )

        def provider(_state):
            return ([move_option(0), switch_option(1)], [])

        result = engine_legal_actions(state, "side_one", option_provider=provider)
        self.assertIsInstance(result, EngineLegalActions)
        self.assertTrue(result.supported)
        self.assertEqual(result.actions, (move_action_label("ember"), switch_action_label(1)))

    def test_selects_requested_seat(self) -> None:
        state = SimpleNamespace(
            side_one=SimpleNamespace(active_index=0, pokemon=[SimpleNamespace(id="a", moves=[])]),
            side_two=SimpleNamespace(active_index=0, pokemon=[SimpleNamespace(id="b", moves=[])]),
        )

        def provider(_state):
            return ([switch_option(5)], [switch_option(9)])

        self.assertEqual(
            engine_legal_actions(state, "side_two", option_provider=provider).actions,
            (switch_action_label(9),),
        )

    def test_unsupported_when_no_provider_on_state_or_module(self) -> None:
        state = SimpleNamespace(side_one=None, side_two=None)
        empty_module = ModuleType("poke_engine_no_options")
        result = engine_legal_actions(state, "side_one", module=empty_module)
        self.assertFalse(result.supported)
        self.assertEqual(result.actions, ())
        self.assertIn("root_get_all_options", result.reason)
        for candidate in ENGINE_OPTION_PROVIDER_CANDIDATES:
            self.assertIn(candidate, result.reason)

    def test_discovers_provider_method_on_state(self) -> None:
        state = fake_engine_state([("charmander", ["ember"])])
        state.get_all_options = lambda: ([move_option(0)], [])
        result = engine_legal_actions(state, "side_one")
        self.assertTrue(result.supported)
        self.assertEqual(result.actions, (move_action_label("ember"),))

    def test_rejects_unknown_side(self) -> None:
        with self.assertRaises(ValueError):
            engine_legal_actions(SimpleNamespace(), "side_three", option_provider=lambda _s: ([], []))


# ---- equivalence ---------------------------------------------------------


class CompareLegalActionsTest(unittest.TestCase):
    def test_equivalent_when_engine_matches_request(self) -> None:
        request = move_request(
            [{"id": "ember"}, {"id": "tackle"}],
            pokemon=[
                party_member("p1: Charmander", active=True),
                party_member("p1: Squirtle"),
            ],
        )
        state = fake_engine_state(
            [
                ("charmander", ["ember", "tackle"]),
                ("squirtle", ["watergun"]),
            ]
        )

        def provider(_state):
            return ([move_option(0), move_option(1), switch_option(1)], [])

        result = compare_legal_actions(request, state, "side_one", option_provider=provider)
        self.assertTrue(result.supported)
        self.assertTrue(result.equivalent)
        self.assertEqual(result.missing_from_engine, ())
        self.assertEqual(result.extra_from_engine, ())

    def test_reports_missing_and_extra(self) -> None:
        request = move_request(
            [{"id": "ember"}, {"id": "tackle"}],
            pokemon=[party_member("p1: Charmander", active=True)],
        )
        state = fake_engine_state([("charmander", ["ember", "surf"])])

        def provider(_state):
            # Engine offers surf (index 1) which the request did not, and omits tackle.
            return ([move_option(0), move_option(1)], [])

        result = compare_legal_actions(request, state, "side_one", option_provider=provider)
        self.assertTrue(result.supported)
        self.assertFalse(result.equivalent)
        self.assertEqual(result.missing_from_engine, (move_action_label("tackle"),))
        self.assertEqual(result.extra_from_engine, (move_action_label("surf"),))

    def test_unsupported_when_engine_lacks_options(self) -> None:
        request = move_request([{"id": "ember"}], pokemon=[party_member("p1: Charmander", active=True)])
        state = SimpleNamespace(side_one=None, side_two=None)
        empty_module = ModuleType("poke_engine_no_options")

        result = compare_legal_actions(request, state, "side_one", module=empty_module)
        self.assertFalse(result.supported)
        self.assertFalse(result.equivalent)
        # Request side still parsed; engine side empty with an actionable reason.
        self.assertEqual(result.request_actions, (move_action_label("ember"),))
        self.assertEqual(result.engine_actions, ())
        self.assertEqual(result.missing_from_engine, ())
        self.assertEqual(result.extra_from_engine, ())
        self.assertIn("PyO3", result.reason)

    def test_to_dict_round_trips_fields(self) -> None:
        result = LegalActionEquivalence(
            supported=True,
            request_actions=("move:ember",),
            engine_actions=("move:ember",),
            missing_from_engine=(),
            extra_from_engine=(),
            reason=None,
        )
        payload = result.to_dict()
        self.assertTrue(payload["supported"])
        self.assertTrue(payload["equivalent"])
        self.assertEqual(payload["request_actions"], ["move:ember"])


# ---- isolation -----------------------------------------------------------


class FakeModuleIsolationTest(unittest.TestCase):
    def test_no_real_poke_engine_import_when_provider_supplied(self) -> None:
        had_real = "poke_engine" in sys.modules
        state = fake_engine_state([("charmander", ["ember"])])
        compare_legal_actions(
            move_request([{"id": "ember"}], pokemon=[party_member("p1: Charmander", active=True)]),
            state,
            "side_one",
            option_provider=lambda _s: ([move_option(0)], []),
        )
        if not had_real:
            self.assertNotIn(
                "poke_engine",
                sys.modules,
                "legal-action comparison imported real poke_engine despite a supplied provider",
            )


# ---- optional real-engine integration ------------------------------------


class RealEngineLegalActionsTest(unittest.TestCase):
    def test_real_engine_reports_unsupported_with_clear_reason(self) -> None:
        probe = probe_poke_engine()
        if not probe.ready:
            self.skipTest("poke-engine is not installed/ready")

        # The real fixture builds a true engine state; the binding exposes no
        # root-option enumeration, so the engine side must report unsupported
        # (with an actionable reason) rather than fail.
        state = build_poke_engine_state(minimal_gen3_fixture())
        request = move_request(
            [{"id": "ember"}, {"id": "tackle"}],
            pokemon=[party_member("p1: Charmander", active=True)],
        )
        result = compare_legal_actions(request, state, "side_one")
        self.assertFalse(result.supported)
        self.assertIsNotNone(result.reason)
        self.assertIn("root_get_all_options", result.reason)
        self.assertEqual(result.engine_actions, ())
        # The request side is still derived even when the engine side is unsupported.
        self.assertEqual(result.request_actions, ("move:ember", "move:tackle"))


if __name__ == "__main__":
    unittest.main()
