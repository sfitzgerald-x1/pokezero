"""Live in-battle retype (Kecleon Color Change) must reach the constructed world.

The parser has produced `live_type_override` since the v3 observation work, but only the
OBSERVATION path consumed it (`showdown._apply_live_type_override`). The world was still
built from base Pokedex types, so a Kecleon whose Color Change had retyped it arrived at the
engine as plain Normal — wrong in both directions at once:

* **attacking**: it keeps Normal STAB it no longer has (seed 1500074 step 12 — Showdown's
  Return does 27, the engine 43, and 43 / 1.5 = 28.7);
* **defending**: incoming effectiveness is priced against the wrong type (seed 1500191 step
  20 — Hidden Power Ice, Showdown `-resisted` for 16, the engine neutral for 34).

Both fall out of one `types` field, so one seeding fixes both.

Precedence is the part worth pinning. Three arms can retype the active mon, and they are
applied transform -> forecast -> **typechange**, deliberately in that order:

| arm | source | kind |
| --- | --- | --- |
| `_apply_transform` | donor's types | derived from a rule |
| `_apply_forecast_types` | public weather | derived from a rule |
| `_apply_live_typechange` | a `-start ... typechange <type>` protocol line | OBSERVED |

Observation beats derivation, so the observed arm runs last. Showdown gets the same answer
by a different route: Color Change's `onAfterMoveSecondary` calls `setType(type)`
(data/abilities.ts:554-562), and because every arm mutates `pokemon.types` in event order,
whichever fired most recently wins. Applying the observation last reproduces that without
modelling the ordering.
"""

from __future__ import annotations

import unittest

from _showdown_root import showdown_root_str  # noqa: E402
from pokezero.dex import load_showdown_dex
from pokezero.engine_world import _apply_forecast_types, _apply_live_typechange, _apply_transform
from pokezero.poke_engine_adapter import MoveSpec, PokemonSpec, SideSpec


def _mon(species: str, ability: str, types: tuple[str, ...]) -> PokemonSpec:
    return PokemonSpec(
        id=species,
        level=80,
        types=types,
        hp=200,
        maxhp=200,
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=100,
        moves=(MoveSpec("tackle"),),
        ability=ability,
    )


def _payload(p1: str | None = None, p2: str | None = None) -> dict:
    sides: dict = {}
    if p1 is not None:
        sides["p1"] = {"liveTypeOverride": p1}
    if p2 is not None:
        sides["p2"] = {"liveTypeOverride": p2}
    return {"sides": sides}


def _sides(p1_types=("Normal",), p2_types=("Normal",)) -> dict[str, SideSpec]:
    return {
        "p1": SideSpec((_mon("kecleon", "colorchange", p1_types),)),
        "p2": SideSpec((_mon("snorlax", "immunity", p2_types),)),
    }


class LiveTypechangeSeedingTest(unittest.TestCase):
    def test_observed_typechange_replaces_the_active_types(self) -> None:
        out = _apply_live_typechange(_sides(), _payload(p1="type:Steel"))
        self.assertEqual(out["p1"].pokemon[0].types, ("Steel",))

    def test_the_retype_is_mono_type(self) -> None:
        # Showdown's `setType` REPLACES the type list rather than appending, so a retyped
        # dual-typed mon ends up single-typed. Pinned because "add a type" is the plausible
        # wrong reading, and it would leave the old type resisting things it should not.
        sides = _sides(p1_types=("Bug", "Steel"))
        out = _apply_live_typechange(sides, _payload(p1="type:Fire"))
        self.assertEqual(out["p1"].pokemon[0].types, ("Fire",))

    def test_absent_or_empty_override_is_a_no_op(self) -> None:
        # The parser CLEARS the override on switch-out, so empty means "base types", not
        # "unknown". This must reproduce pre-change behaviour exactly for every side that
        # has no live retype — which is nearly all of them.
        base = _sides()
        for payload in (_payload(), _payload(p1=""), {"sides": {}}, {}):
            out = _apply_live_typechange(base, payload)
            self.assertEqual(out["p1"].pokemon[0].types, ("Normal",))
            self.assertEqual(out["p2"].pokemon[0].types, ("Normal",))

    def test_only_the_named_side_is_touched(self) -> None:
        out = _apply_live_typechange(_sides(), _payload(p1="type:Ghost"))
        self.assertEqual(out["p1"].pokemon[0].types, ("Ghost",))
        self.assertEqual(out["p2"].pokemon[0].types, ("Normal",))

    def test_the_forme_form_is_deliberately_not_consumed_here(self) -> None:
        # `forme:` is Castform Forecast, which `_apply_forecast_types` already derives from
        # the same public weather. Consuming it here too would give Castform TWO writers
        # that must agree — the exact shape that let the encoder-vocabulary bug survive for
        # months. Forecast is `onUpdate`, so the derived arm cannot lag the observation.
        out = _apply_live_typechange(_sides(), _payload(p1="forme:Castform-Rainy"))
        self.assertEqual(out["p1"].pokemon[0].types, ("Normal",))

    def test_a_garbage_payload_is_ignored_rather_than_guessed(self) -> None:
        for raw in ("type:", "type:   ", "nonsense", "Steel"):
            out = _apply_live_typechange(_sides(), _payload(p1=raw))
            self.assertEqual(
                out["p1"].pokemon[0].types, ("Normal",), f"payload {raw!r} should be inert"
            )


class RetypePrecedenceTest(unittest.TestCase):
    """The ordering contract between the three retype arms."""

    def test_observed_typechange_wins_over_derived_forecast(self) -> None:
        # Construct the collision explicitly: a Castform that the protocol says is Steel.
        # Unreachable in gen3 randbats (Castform has Forecast, not Color Change), so this
        # pins the RULE rather than a scenario — observation beats derivation, whichever
        # mon it lands on.
        sides = {
            "p1": SideSpec((_mon("castform", "forecast", ("Normal",)),)),
            "p2": SideSpec((_mon("snorlax", "immunity", ("Normal",)),)),
        }
        after_forecast = _apply_forecast_types(sides, weather="rain")
        self.assertEqual(after_forecast["p1"].pokemon[0].types, ("Water",))
        after_typechange = _apply_live_typechange(after_forecast, _payload(p1="type:Steel"))
        self.assertEqual(
            after_typechange["p1"].pokemon[0].types,
            ("Steel",),
            "an observed typechange must override a weather-derived Forecast type",
        )

    def test_observed_typechange_wins_over_transform(self) -> None:
        # A transformed mon that is then Color Change'd: Showdown's later `setType` wins
        # because both mutate `pokemon.types` in event order. The world reproduces that by
        # applying the observation after the transform copy.
        sides = {
            "p1": SideSpec((_mon("kecleon", "colorchange", ("Normal",)),)),
            "p2": SideSpec((_mon("gengar", "levitate", ("Ghost", "Poison")),)),
        }
        # `dex` became required when the copied moveset started taking its PP from the
        # catalog; this call site predates that.
        transformed = _apply_transform(
            sides, {"p1": "gengar"}, dex=load_showdown_dex(showdown_root_str())
        )
        self.assertEqual(transformed["p1"].pokemon[0].types, ("Ghost", "Poison"))
        after = _apply_live_typechange(transformed, _payload(p1="type:Water"))
        self.assertEqual(
            after["p1"].pokemon[0].types,
            ("Water",),
            "an observed typechange must override the transform-copied types",
        )

    def test_forecast_still_works_when_no_typechange_is_observed(self) -> None:
        # Guard against the new arm quietly clobbering the existing ones: with no observed
        # retype, forecast's answer must survive untouched.
        sides = {
            "p1": SideSpec((_mon("castform", "forecast", ("Normal",)),)),
            "p2": SideSpec((_mon("snorlax", "immunity", ("Normal",)),)),
        }
        out = _apply_live_typechange(_apply_forecast_types(sides, weather="sun"), _payload())
        self.assertEqual(out["p1"].pokemon[0].types, ("Fire",))


if __name__ == "__main__":
    unittest.main()
