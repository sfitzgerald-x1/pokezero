"""Pressure's double PP charge is decided by the move's ENGINE target, never by the wire slot.

``sim/pokemon.ts getMoveTargets`` sets ``pressureTargets = targets``, so a foe is present for
every target kind except ``self`` and the ally-side ones, and ``pressure.onDeductPP`` then
returns an extra 1 (``data/abilities.ts:3395``, not overridden anywhere in the gen3 chain).

The obvious proxy for "is a foe targeted" -- the target slot on the ``|move|`` line -- is not
usable, because ``sim/battle.ts:3155-3159`` deliberately blanks it whenever the move's animation
is suppressed:

    } else if (args.includes('[still]')) {
        // If no animation plays, the target should never be known
        const parts = this.log[this.lastMoveLine].split('|');
        parts[4] = '';

Toxic against a sleeping/immune target, Encore, Will-O-Wisp, Leech Seed and Spikes all hit that
path routinely -- measured over 60 games, Toxic's target slot was blank on 53 of 328 lines. Every
one of those lost its Pressure double and the PP ledger drifted for the rest of the battle.

These tests pin the two directions (pressured / never pressured) plus the two members of the
never-pressured set that are not self-evident, and the second producer of a running Pressure.
"""
import unittest

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.showdown import parse_showdown_replay


def _uses(lines, slot, species, move):
    engine = PublicBattleBeliefEngine.from_events(
        parse_showdown_replay(
            lines, battle_id="battle-gen3randombattle-pressure"
        ).public_events
    )
    for mon in engine.snapshot().side(slot):
        if mon.species.lower().startswith(species.lower()):
            return dict(mon.move_uses).get(move, 0)
    raise AssertionError(f"no {species} on {slot}")


# Zapdos announces Pressure on entry (`data/mods/gen3/abilities.ts:145-150` keeps the shared
# `onStart`, split to its own side). Snorlax is the foe throughout: it carries Curse, Rest and
# Toxic in the pool, so every line below is reachable against a Pressure lead.
OPENING = [
    "|start",
    "|switch|p1a: Zapdos|Zapdos, L77|100/100",
    "|switch|p2a: Snorlax|Snorlax, L79, M|100/100",
    "|-ability|p1a: Zapdos|Pressure|[silent]",
]


class PressureTargetingTest(unittest.TestCase):
    def test_foe_targeted_move_with_a_blanked_target_slot_is_still_pressured(self) -> None:
        """The live defect: `[still]` erases the target and the double charge went with it."""
        lines = [*OPENING, "|move|p2a: Snorlax|Toxic|", "|turn|1"]
        self.assertEqual(_uses(lines, "p2", "Snorlax", "toxic"), 2)

    def test_foe_targeted_move_with_a_named_target_slot_is_pressured(self) -> None:
        lines = [*OPENING, "|move|p2a: Snorlax|Toxic|p1a: Zapdos", "|turn|1"]
        self.assertEqual(_uses(lines, "p2", "Snorlax", "toxic"), 2)

    def test_self_targeted_move_is_never_pressured(self) -> None:
        lines = [*OPENING, "|move|p2a: Snorlax|Rest|p2a: Snorlax", "|turn|1"]
        self.assertEqual(_uses(lines, "p2", "Snorlax", "rest"), 1)

    def test_self_targeted_move_is_not_pressured_even_with_a_blank_slot(self) -> None:
        """The mirror risk of ignoring the wire: a blank slot must not turn Rest into a double."""
        lines = [*OPENING, "|move|p2a: Snorlax|Rest|", "|turn|1"]
        self.assertEqual(_uses(lines, "p2", "Snorlax", "rest"), 1)

    def test_curse_is_not_pressured_for_the_pools_non_ghost_carriers(self) -> None:
        """Curse's dex target is `normal`; `sim/pokemon.ts:999-1002` retargets it to
        `nonGhostTarget` (self) unless the user is Ghost, and all five pool carriers -- Muk
        (Poison), Snorlax, Dunsparce, Miltank (Normal) and Regirock (Rock) -- are non-Ghost."""
        lines = [*OPENING, "|move|p2a: Snorlax|Curse|p2a: Snorlax", "|turn|1"]
        self.assertEqual(_uses(lines, "p2", "Snorlax", "curse"), 1)

    def test_ally_team_move_is_not_pressured(self) -> None:
        """`allyTeam` never reaches a foe. Heal Bell's pool carriers include Miltank."""
        lines = [
            "|start",
            "|switch|p1a: Zapdos|Zapdos, L77|100/100",
            "|switch|p2a: Miltank|Miltank, L80, F|100/100",
            "|-ability|p1a: Zapdos|Pressure|[silent]",
            "|move|p2a: Miltank|Heal Bell|p2a: Miltank",
            "|turn|1",
        ]
        self.assertEqual(_uses(lines, "p2", "Miltank", "healbell"), 1)

    def test_no_pressure_on_the_field_means_a_single_charge(self) -> None:
        lines = [
            "|start",
            "|switch|p1a: Blissey|Blissey, L75, F|100/100",
            "|switch|p2a: Snorlax|Snorlax, L79, M|100/100",
            "|move|p2a: Snorlax|Toxic|",
            "|turn|1",
        ]
        self.assertEqual(_uses(lines, "p2", "Snorlax", "toxic"), 1)


class TransformedPressureTest(unittest.TestCase):
    """Transform copies the target's ABILITY, so the transformer pressures for real.

    ``sim/pokemon.ts:1353`` -- ``if (this.battle.gen > 2) this.setAbility(pokemon.ability, ...)``
    inside ``transformInto``, and nothing in the gen3 -> gen4 -> ... mod chain overrides it. Ditto
    is in the gen3 randbats pool, so a Ditto copying any of its Pressure members is reachable.
    """

    LINES = [
        "|start",
        "|switch|p1a: Ditto|Ditto, L83|100/100",
        "|switch|p2a: Zapdos|Zapdos, L77|100/100",
        "|-ability|p2a: Zapdos|Pressure|[silent]",
        "|move|p1a: Ditto|Transform|p2a: Zapdos",
        "|-transform|p1a: Ditto|p2a: Zapdos",
        "|turn|1",
        "|move|p2a: Zapdos|Thunderbolt|p1a: Ditto",
        "|turn|2",
    ]

    def test_transformed_ditto_pressures_the_mon_it_copied(self) -> None:
        self.assertEqual(_uses(self.LINES, "p2", "Zapdos", "thunderbolt"), 2)

    def test_the_copy_does_not_overwrite_dittos_revealed_ability(self) -> None:
        """`revealed_ability` stays Ditto's own set ability; only the RUNNING one changes."""
        engine = PublicBattleBeliefEngine.from_events(
            parse_showdown_replay(
                self.LINES, battle_id="battle-gen3randombattle-pressure"
            ).public_events
        )
        ditto = next(m for m in engine.snapshot().side("p1") if m.species.startswith("Ditto"))
        self.assertNotEqual((ditto.revealed_ability or "").lower(), "pressure")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
