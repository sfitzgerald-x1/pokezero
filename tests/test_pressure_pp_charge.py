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


class NeverPressuredSetMatchesThePoolTest(unittest.TestCase):
    """`_NEVER_PRESSURED_POOL_MOVES` is a STATIC set derived from a live data file. Guard it.

    The set was derived once, by hand, from `data/random-battles/gen3/sets.json` and the
    gen3-effective move targets. Two premises make it correct, and neither was checked by anything
    until this test:

    1. the pool's `self` + `allyTeam` moves are exactly the 25 non-Curse members, and
    2. Curse -- whose dex target is `normal` -- is retargeted to `self` for every pool carrier,
       because `sim/pokemon.ts` only applies `nonGhostTarget` to a non-Ghost user and all five
       gen3-randbats carriers (Muk, Snorlax, Dunsparce, Miltank, Regirock) are non-Ghost.

    The failure this guards is SILENT and in the dangerous direction: a move wrongly IN the set
    never gets its Pressure double, and the PP ledger drifts for the rest of the battle with
    nothing raised. That is precisely the defect this whole change fixed, so re-introducing it via
    a data change rather than a code change should not be possible without a red test.

    Skipped without a Showdown checkout, like the rest of the data-dependent suites.
    """

    def _pool_and_targets(self):
        import json
        import os
        import re

        root = os.environ.get("POKEZERO_SHOWDOWN_ROOT")
        if not root or not os.path.isdir(root):
            self.skipTest("requires POKEZERO_SHOWDOWN_ROOT")
        sets_path = os.path.join(root, "data", "random-battles", "gen3", "sets.json")
        if not os.path.exists(sets_path):
            self.skipTest("no gen3 randbats sets.json")

        def blocks(path):
            """Top-level `\\tmoveid: {` blocks, brace-matched."""
            if not os.path.exists(path):
                return {}
            text = open(path).read()
            out = {}
            for match in re.finditer(r"^\t([a-z0-9]+): \{$", text, re.M):
                index, depth = match.end() - 1, 0
                while index < len(text):
                    if text[index] == "{":
                        depth += 1
                    elif text[index] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    index += 1
                out[match.group(1)] = text[match.end():index]
            return out

        # Resolved along the MOD CHAIN, not off shared data/moves.ts: `surf` is allAdjacentFoes in
        # gen3 and `curse` is re-declared in gen7. Note the checkout writes `target: "self",` with
        # DOUBLE quotes; a single-quote pattern silently matches nothing and every move then looks
        # target-less, which is a failure mode this parser has already hit once.
        chain = [
            os.path.join(root, "data", "mods", g, "moves.ts")
            for g in ("gen3", "gen4", "gen5", "gen6", "gen7", "gen8")
        ] + [os.path.join(root, "data", "moves.ts")]
        layers = [blocks(p) for p in chain]

        def effective_target(move_id):
            for layer in layers:
                body = layer.get(move_id)
                if body is None:
                    continue
                found = re.search(r"""^\t\ttarget: ['"]([a-zA-Z]+)['"],""", body, re.M)
                if found:
                    return found.group(1)
            return None

        pool = set()
        for entry in json.load(open(sets_path)).values():
            for spec in entry.get("sets", []):
                pool |= {
                    m.lower().replace(" ", "").replace("-", "").replace("'", "")
                    for m in spec.get("movepool", [])
                }
        return pool, effective_target

    def test_the_static_set_still_equals_what_the_pool_implies(self) -> None:
        from pokezero.belief import _NEVER_PRESSURED_POOL_MOVES

        pool, effective_target = self._pool_and_targets()
        unresolved = sorted(m for m in pool if effective_target(m) is None)
        self.assertEqual(unresolved, [], "pool moves with no resolvable target")

        derived = {m for m in pool if effective_target(m) in ("self", "allyTeam")} | {"curse"}
        self.assertEqual(
            derived,
            set(_NEVER_PRESSURED_POOL_MOVES),
            "the pool no longer implies the hardcoded never-pressured set; a move added to it "
            "loses its Pressure double SILENTLY",
        )

    def test_every_pool_curse_carrier_is_still_non_ghost(self) -> None:
        """Curse is in the set only because `nonGhostTarget` retargets it for these carriers."""
        import json
        import os
        import re

        pool, _ = self._pool_and_targets()
        self.assertIn("curse", pool, "Curse left the pool; its entry in the set is now dead")

        root = os.environ["POKEZERO_SHOWDOWN_ROOT"]
        sets_path = os.path.join(root, "data", "random-battles", "gen3", "sets.json")
        carriers = [
            species
            for species, entry in json.load(open(sets_path)).items()
            for spec in entry.get("sets", [])
            if any(m.lower().replace(" ", "") == "curse" for m in spec.get("movepool", []))
        ]
        self.assertTrue(carriers, "no Curse carriers found; the derivation cannot be checked")

        dex = open(os.path.join(root, "data", "pokedex.ts")).read()

        def types_of(species):
            block = re.search(
                r"^\t" + re.escape(species) + r": \{(.*?)^\t\},", dex, re.S | re.M
            )
            if block is None:
                return None
            found = re.search(r"types: \[([^\]]*)\]", block.group(1))
            return found.group(1) if found is not None else None

        # POSITIVE CONTROL, before trusting a single negative result below. This whole assertion
        # is "no carrier is Ghost", which passes just as happily when the extractor returns
        # nothing at all -- and a silent-no-match regex is not hypothetical here: the sibling test
        # in this file parses `target:` out of moves.ts, where a single-quote pattern matches
        # nothing because the checkout uses double quotes. Prove the extractor can find a Ghost
        # before concluding there are none.
        for ghost_species in ("dusclops", "gengar", "misdreavus"):
            self.assertIn(
                "Ghost",
                types_of(ghost_species) or "",
                f"type extraction is broken: it cannot see that {ghost_species} is Ghost, so "
                "the non-Ghost conclusion below would be vacuous",
            )

        ghosts = []
        for species in sorted(set(carriers)):
            types = types_of(species)
            if types is None:
                self.fail(f"no types resolved for Curse carrier {species} in pokedex.ts")
            if "Ghost" in types:
                ghosts.append(species)
        self.assertEqual(
            ghosts,
            [],
            "a Ghost-type Curse carrier is in the pool, so Curse is FOE-targeted for it and "
            "must leave _NEVER_PRESSURED_POOL_MOVES",
        )
