"""High-risk regression gates for the Gen 3 randbats ability audit."""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT  # noqa: E402
from pokezero.randbat import Gen3RandbatSource  # noqa: E402

try:
    import poke_engine
except ImportError:  # pragma: no cover - native wheel absent
    poke_engine = None


AUDITED_ABILITIES = frozenset(
    {
        "Air Lock", "Arena Trap", "Battle Armor", "Blaze", "Chlorophyll", "Clear Body",
        "Cloud Nine", "Color Change", "Compound Eyes", "Cute Charm", "Drizzle", "Drought",
        "Early Bird", "Effect Spore", "Flame Body", "Flash Fire", "Forecast", "Guts",
        "Huge Power", "Hustle", "Hyper Cutter", "Immunity", "Inner Focus", "Insomnia",
        "Intimidate", "Keen Eye", "Levitate", "Limber", "Liquid Ooze",
        "Magma Armor", "Magnet Pull", "Marvel Scale", "Minus", "Natural Cure", "Oblivious",
        "Overgrow", "Own Tempo", "Pickup", "Plus", "Poison Point", "Pressure", "Pure Power",
        "Rock Head", "Rough Skin", "Run Away", "Sand Stream", "Sand Veil", "Serene Grace",
        "Shadow Tag", "Shed Skin", "Shell Armor", "Shield Dust", "Soundproof", "Speed Boost",
        "Static", "Sticky Hold", "Sturdy", "Suction Cups", "Swarm", "Swift Swim",
        "Synchronize", "Thick Fat", "Torrent", "Trace", "Truant", "Vital Spirit",
        "Volt Absorb", "Water Absorb", "Water Veil", "White Smoke", "Wonder Guard",
    }
)


class AbilityCatalogTests(unittest.TestCase):
    def test_published_ledger_has_one_row_per_audited_ability(self) -> None:
        report = (ROOT / "docs/gen3_randbat_ability_audit.md").read_text(encoding="utf-8")
        ledger = report.split("## Ability ledger", 1)[1].split("## Verification evidence", 1)[0]
        reported = {
            line.split("|", 2)[1].strip()
            for line in ledger.splitlines()
            if line.startswith("| ") and not line.startswith(("| Ability ", "|---"))
        }
        self.assertEqual(reported, AUDITED_ABILITIES)

    @unittest.skipUnless(
        (Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
         / "data/random-battles/gen3/sets.json").exists(),
        "requires a local Pokemon Showdown checkout",
    )
    def test_audit_covers_the_live_randbat_ability_universe(self) -> None:
        root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
        source = Gen3RandbatSource.from_showdown_root(root)
        actual = {variant.ability for universe in source.universes.values() for variant in universe.variants}
        self.assertEqual(actual, AUDITED_ABILITIES)
        report = (ROOT / "docs/gen3_randbat_ability_audit.md").read_text(encoding="utf-8")
        self.assertIn(source.metadata.source_hash, report)


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class AbilityMechanicsTests(unittest.TestCase):
    def _mon(
        self,
        species: str,
        ability: str,
        move: str | tuple[str, ...],
        *,
        types: tuple[str, str] = ("normal", "typeless"),
        level: int = 80,
        hp: int = 300,
        maxhp: int = 300,
        attack: int = 180,
        defense: int = 180,
        special_attack: int = 180,
        special_defense: int = 180,
        speed: int = 100,
        status: str = "none",
        sleep_turns: int = 0,
        pp: int = 16,
        item: str = "none",
        gender: str = "none",
    ):
        return poke_engine.Pokemon(
            id=species,
            level=level,
            gender=gender,
            types=types,
            base_types=types,
            hp=hp,
            maxhp=maxhp,
            ability=ability,
            item=item,
            attack=attack,
            defense=defense,
            special_attack=special_attack,
            special_defense=special_defense,
            speed=speed,
            status=status,
            sleep_turns=sleep_turns,
            moves=[
                poke_engine.Move(id=move_id, pp=pp)
                for move_id in ((move,) if isinstance(move, str) else move)
            ],
        )

    def _state(
        self,
        attacker,
        defender,
        *,
        weather: str = "none",
        attacker_party=(),
        defender_party=(),
        attacker_volatiles=(),
        defender_volatiles=(),
        substitute_health: int = 0,
        weather_turns_remaining: int = 0,
        attacker_safeguard: int = 0,
        defender_safeguard: int = 0,
        attacker_spikes: int = 0,
        attacker_yawn_duration: int = 0,
        attacker_speed_boost: int = 0,
    ):
        dummy = poke_engine.Pokemon(id="pikachu", level=1, hp=0)
        p1 = [attacker, *attacker_party]
        p2 = [defender, *defender_party]
        p1.extend([dummy] * (6 - len(p1)))
        p2.extend([dummy] * (6 - len(p2)))
        return poke_engine.State(
            side_one=poke_engine.Side(
                active_index="0",
                speed_boost=attacker_speed_boost,
                pokemon=p1,
                volatile_statuses=set(attacker_volatiles),
                volatile_status_durations=poke_engine.VolatileStatusDurations(
                    yawn=attacker_yawn_duration
                ),
                side_conditions=poke_engine.SideConditions(
                    safeguard=attacker_safeguard,
                    spikes=attacker_spikes,
                ),
            ),
            side_two=poke_engine.Side(
                active_index="0",
                pokemon=p2,
                volatile_statuses=set(defender_volatiles),
                substitute_health=substitute_health,
                side_conditions=poke_engine.SideConditions(safeguard=defender_safeguard),
            ),
            weather=weather,
            weather_turns_remaining=weather_turns_remaining,
            terrain="none",
            trick_room=False,
        )

    def test_blaze_boosts_only_fire_damage_at_or_below_one_third_hp(self) -> None:
        defender = self._mon("snorlax", "immunity", "splash", hp=1000, maxhp=1000)

        def opening_damage(hp: int, move: str) -> tuple[int, ...]:
            attacker = self._mon(
                "charizard",
                "blaze",
                move,
                types=("fire", "flying"),
                hp=hp,
                maxhp=300,
                speed=200,
            )
            branches = poke_engine.generate_instructions(
                self._state(attacker, defender), move, "splash"
            )
            return tuple(
                sorted(
                    {
                        int(self._text(branch).split("Damage SideTwo: ", 1)[1].split()[0])
                        for branch in branches
                    }
                )
            )

        boosted_fire = opening_damage(100, "flamethrower")
        ordinary_fire = opening_damage(101, "flamethrower")
        self.assertEqual(len(boosted_fire), len(ordinary_fire))
        for boosted, ordinary in zip(boosted_fire, ordinary_fire, strict=True):
            self.assertGreater(boosted, ordinary)
            self.assertAlmostEqual(boosted / ordinary, 1.5, delta=0.03)

        self.assertEqual(opening_damage(100, "tackle"), opening_damage(101, "tackle"))

    def test_flail_uses_the_hp_band_reached_by_the_first_move_roll(self) -> None:
        """Retained c15 row 2201005/55: Crunch can leave Dodrio at 39/222."""
        mightyena = self._mon(
            "mightyena", "intimidate", "crunch", types=("dark", "typeless"),
            level=92, hp=137, maxhp=278, attack=218, defense=180,
            special_attack=162, special_defense=162, speed=180, item="leftovers",
        )
        dodrio = self._mon(
            "dodrio", "earlybird", "flail", types=("normal", "flying"),
            level=78, hp=147, maxhp=222, attack=217, defense=154,
            special_attack=139, special_defense=139, speed=201,
            status="paralyze", item="liechiberry",
        )
        branches = poke_engine.generate_instructions(
            self._state(mightyena, dodrio), "crunch", "flail"
        )

        # Showdown's retained line is Crunch 108 -> Dodrio 39/222 -> Flail 110.
        # The engine represents Flail's own random damage as its average (111),
        # but the preceding 108 roll must enter Flail's 100-BP HP band.
        crossed_band = [branch for branch in branches if "Damage SideTwo: 108" in self._text(branch)]
        self.assertTrue(crossed_band)
        self.assertTrue(any("Damage SideOne: 111" in self._text(branch) for branch in crossed_band))

    def test_dynamic_power_fanout_preserves_the_first_moves_hit_count(self) -> None:
        attacker = self._mon(
            "hitmonlee", "limber", "doublekick", types=("fighting", "typeless"),
            hp=220, maxhp=220, attack=180, speed=200,
        )
        pending_flail = self._mon(
            "dodrio", "earlybird", "flail", types=("normal", "flying"),
            hp=180, maxhp=220, defense=180, speed=100,
        )

        branches = poke_engine.generate_instructions(
            self._state(attacker, pending_flail), "doublekick", "flail"
        )

        self.assertTrue(branches)
        self.assertTrue(all(
            sum(
                instruction.startswith("Damage SideTwo:")
                for instruction in self._text(branch).split(" | ")
            ) == 2
            for branch in branches
        ))

    def test_cloud_nine_entry_does_not_change_forecast_but_exit_restores_it(self) -> None:
        """Retained c15 row 2400451/56: Cloud Nine suppresses, not changes, rain."""
        castform_water = self._mon(
            "castform", "forecast", "return102", types=("water", "typeless"),
            hp=28, maxhp=272, speed=177,
        )
        magikarp = self._mon(
            "magikarp", "swiftswim", "splash", types=("water", "typeless"), hp=1, maxhp=100,
        )
        golduck = self._mon(
            "golduck", "cloudnine", "splash", types=("water", "typeless"),
            hp=239, maxhp=262, speed=184,
        )
        entering = self._state(
            castform_water, magikarp, weather="rain", weather_turns_remaining=-1,
            defender_party=(golduck,),
        )
        entering_branches = poke_engine.generate_instructions(entering, "return102", "golduck")
        self.assertTrue(entering_branches)
        self.assertTrue(all("ChangeType SideOne" not in self._text(branch) for branch in entering_branches))
        self.assertTrue(all(
            tuple(str(value).upper() for value in entering.apply_instructions(branch).side_one.pokemon[0].types)
            == ("WATER", "TYPELESS")
            for branch in entering_branches
        ))

        castform_normal = self._mon(
            "castform", "forecast", "return102", types=("normal", "typeless"),
            hp=28, maxhp=272, speed=177,
        )
        leaving = self._state(
            castform_normal, golduck, weather="rain", weather_turns_remaining=-1,
            defender_party=(magikarp,),
        )
        leaving_branches = poke_engine.generate_instructions(leaving, "return102", "magikarp")
        self.assertTrue(all("ChangeType SideOne" in self._text(branch) for branch in leaving_branches))
        self.assertTrue(all(
            tuple(str(value).upper() for value in leaving.apply_instructions(branch).side_one.pokemon[0].types)
            == ("WATER", "TYPELESS")
            for branch in leaving_branches
        ))

    def test_cloud_nine_handoff_exposes_weather_before_the_incoming_suppressor(self) -> None:
        """Gen 3 runs Forecast on suppressor exit, but not suppressor entry."""
        castform_normal = self._mon(
            "castform", "forecast", "return102", types=("normal", "typeless"),
            hp=272, maxhp=272, speed=177,
        )
        golduck = self._mon(
            "golduck", "cloudnine", "splash", types=("water", "typeless"),
            hp=239, maxhp=262, speed=184,
        )
        incoming = self._mon(
            "shedinja", "cloudnine", "splash", types=("bug", "ghost"),
            hp=1, maxhp=1, speed=120,
        )
        state = self._state(
            castform_normal, golduck, weather="rain", weather_turns_remaining=-1,
            defender_party=(incoming,),
        )

        branches = poke_engine.generate_instructions(state, "return102", "shedinja")
        self.assertTrue(branches)
        for branch in branches:
            text = self._text(branch)
            self.assertIn("ChangeType SideOne", text)
            self.assertLess(text.index("ChangeType SideOne"), text.index("Switch SideTwo"))
            applied = state.apply_instructions(branch)
            self.assertEqual(
                tuple(
                    str(value).upper()
                    for value in applied.side_one.pokemon[0].types
                ),
                ("WATER", "TYPELESS"),
            )

    def test_color_change_waits_until_after_ice_beam_freeze(self) -> None:
        """Retained incap row 2700752/65: Kecleon freezes before becoming Ice."""
        kecleon = self._mon(
            "kecleon", "colorchange", "brickbreak", types=("normal", "typeless"),
            level=92, hp=162, maxhp=260, attack=218, defense=181,
            special_attack=163, special_defense=273, speed=126,
        )
        seaking = self._mon(
            "seaking", "swiftswim", "icebeam", types=("water", "typeless"),
            level=90, hp=179, maxhp=290, attack=172, defense=168,
            special_attack=167, special_defense=195, speed=174,
        )
        branches = poke_engine.generate_instructions(
            self._state(kecleon, seaking, weather="rain", weather_turns_remaining=4),
            "brickbreak", "icebeam",
        )
        frozen = [branch for branch in branches if "NONE -> FREEZE" in self._text(branch)]
        self.assertTrue(frozen)
        self.assertAlmostEqual(self._mass(branches, "NONE -> FREEZE"), 10.0, places=4)
        for branch in frozen:
            text = self._text(branch)
            self.assertLess(text.index("NONE -> FREEZE"), text.index("ChangeType SideOne"))
            self.assertNotIn("Damage SideTwo", text)

    def test_color_change_does_not_run_when_endure_reduces_actual_damage_to_zero(self) -> None:
        kecleon = self._mon(
            "kecleon", "colorchange", "endure", types=("normal", "typeless"),
            hp=1, maxhp=260, speed=126,
        )
        seaking = self._mon(
            "seaking", "swiftswim", "icebeam", types=("water", "typeless"),
            hp=179, maxhp=290, special_attack=167, speed=174,
        )

        branches = poke_engine.generate_instructions(
            self._state(kecleon, seaking), "endure", "icebeam"
        )

        self.assertTrue(branches)
        self.assertTrue(any("Damage SideOne: 0" in self._text(branch) for branch in branches))
        self.assertTrue(all("ChangeType SideOne" not in self._text(branch) for branch in branches))

    @staticmethod
    def _text(branch) -> str:
        return " | ".join(str(instruction) for instruction in branch.instruction_list)

    @classmethod
    def _mass(cls, branches, fragment: str) -> float:
        return sum(float(branch.percentage) for branch in branches if fragment in cls._text(branch))

    def test_gen3_contact_proc_probabilities_and_substitute_gate(self) -> None:
        expected = {
            "poisonpoint": {"POISON": 100.0 / 3.0},
            "flamebody": {"BURN": 100.0 / 3.0},
            "static": {"PARALYZE": 100.0 / 3.0},
            "effectspore": {"POISON": 100.0 / 30.0, "PARALYZE": 100.0 / 30.0, "SLEEP": 100.0 / 30.0},
        }
        attacker = self._mon("tauros", "intimidate", "tackle", speed=200)
        for ability, status_masses in expected.items():
            with self.subTest(ability=ability):
                defender = self._mon("nidoqueen", ability, "splash")
                branches = poke_engine.generate_instructions(
                    self._state(attacker, defender), "tackle", "splash"
                )
                for status, mass in status_masses.items():
                    self.assertAlmostEqual(self._mass(branches, f"-> {status}"), mass, places=4)

                behind_sub = poke_engine.generate_instructions(
                    self._state(
                        attacker,
                        defender,
                        defender_volatiles={"SUBSTITUTE"},
                        substitute_health=100,
                    ),
                    "tackle",
                    "splash",
                )
                self.assertFalse(any("ChangeStatus SideOne" in self._text(branch) for branch in behind_sub))

    def test_cute_charm_uses_public_gender_and_gen3_contact_gates(self) -> None:
        attacker = self._mon(
            "tauros", "intimidate", "tackle", speed=200, gender="male"
        )
        defender = self._mon("delcatty", "cutecharm", "splash", gender="female")
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "tackle", "splash"
        )
        self.assertAlmostEqual(self._mass(branches, "ATTRACT"), 100.0 / 3.0, places=4)

        blocked_cases = (
            ("same gender", self._mon("delcatty", "cutecharm", "splash", gender="male"), {}),
            ("genderless", self._mon("delcatty", "cutecharm", "splash"), {}),
            ("substitute", defender, {"defender_volatiles": {"SUBSTITUTE"}, "substitute_health": 100}),
            (
                "oblivious",
                defender,
                {
                    "attacker": self._mon(
                        "slowbro", "oblivious", "tackle", speed=200, gender="male"
                    )
                },
            ),
        )
        for label, case_defender, options in blocked_cases:
            with self.subTest(label=label):
                state_options = dict(options)
                case_attacker = state_options.pop("attacker", attacker)
                case_branches = poke_engine.generate_instructions(
                    self._state(case_attacker, case_defender, **state_options), "tackle", "splash"
                )
                self.assertEqual(self._mass(case_branches, "ATTRACT"), 0.0)

    def test_cute_charm_attract_ends_when_its_source_switches(self) -> None:
        attacker = self._mon(
            "tauros", "intimidate", "splash", speed=200, gender="male"
        )
        source = self._mon("delcatty", "cutecharm", "splash", gender="female")
        replacement = self._mon("snorlax", "immunity", "splash", gender="male")
        branches = poke_engine.generate_instructions(
            self._state(
                attacker,
                source,
                attacker_volatiles={"ATTRACT"},
                defender_party=(replacement,),
            ),
            "splash",
            "snorlax",
        )

        self.assertTrue(branches)
        self.assertTrue(
            all("RemoveVolatileStatus SideOne: ATTRACT" in self._text(branch) for branch in branches)
        )

    def test_contact_status_abilities_respect_safeguard(self) -> None:
        attacker = self._mon("tauros", "intimidate", "tackle", speed=200)
        for ability in ("poisonpoint", "flamebody", "static", "effectspore"):
            with self.subTest(ability=ability):
                defender = self._mon("nidoqueen", ability, "splash")
                branches = poke_engine.generate_instructions(
                    self._state(attacker, defender, attacker_safeguard=2),
                    "tackle",
                    "splash",
                )
                self.assertFalse(
                    any("ChangeStatus SideOne" in self._text(branch) for branch in branches)
                )

    def test_contact_secondary_sees_the_attacker_wake_on_the_same_turn(self) -> None:
        # A5. `ability_modify_attack_against` decides the contact secondary from
        # `before_move`, one call before the attacker's own sleep/freeze is
        # resolved, so a mon that wakes and attacks on the same turn used to be
        # read as still-statused and the secondary was refused outright. Showdown
        # decides these in `onDamagingHit`, well after `slp.onBeforeMove` cleared
        # the status. gen3 MAX_SLEEP_TURNS is 4, so `chance_to_wake_up` is
        # 1/(1+4-turns): certain at 4 turns, 1/4 at 1 turn. Freeze thaws at 20%.
        third = 100.0 / 3.0
        cases = (
            ("certain wake", {"status": "sleep", "sleep_turns": 4}, third),
            ("forked wake", {"status": "sleep", "sleep_turns": 1}, third / 4.0),
            ("thaw", {"status": "freeze"}, third / 5.0),
        )
        defender = self._mon("nidoqueen", "poisonpoint", "splash")
        for label, mon_kwargs, expected in cases:
            with self.subTest(case=label):
                attacker = self._mon(
                    "tauros", "intimidate", "tackle", speed=200, **mon_kwargs
                )
                branches = poke_engine.generate_instructions(
                    self._state(attacker, defender), "tackle", "splash"
                )
                self.assertAlmostEqual(
                    self._mass(branches, "-> POISON"), expected, places=4
                )

    def test_a_status_that_survives_the_move_still_refuses_the_contact_secondary(
        self,
    ) -> None:
        # The controls for the fix above, and the reason it is a predicate change
        # rather than an unconditional one. Sleep Talk and the move it calls are
        # gen3's only actions that land a hit while their user is still asleep, so
        # their user really does hold a status at `onDamagingHit`. Paralysis is
        # never cured by moving at all.
        defender = self._mon("nidoqueen", "poisonpoint", "splash")
        sleep_talker = self._mon(
            "tauros",
            "intimidate",
            ("sleeptalk", "tackle"),
            speed=200,
            status="sleep",
            sleep_turns=1,
        )
        branches = poke_engine.generate_instructions(
            self._state(sleep_talker, defender), "sleeptalk", "splash"
        )
        # Anti-vacuity: 25% wakes and does nothing, 75% stays asleep (sleep turns
        # 1 -> 2) and lands Tackle. Without this the POISON assertion below would
        # also pass on a state where no contact ever happened.
        self.assertAlmostEqual(
            self._mass(branches, "Damage SideTwo"),
            75.0,
            places=4,
            msg="Sleep Talk must land its contact move while asleep, or this control proves nothing",
        )
        self.assertAlmostEqual(self._mass(branches, "-> POISON"), 0.0, places=4)

        paralyzed = self._mon(
            "tauros", "intimidate", "tackle", speed=200, status="paralyze"
        )
        branches = poke_engine.generate_instructions(
            self._state(paralyzed, defender), "tackle", "splash"
        )
        self.assertAlmostEqual(
            self._mass(branches, "Damage SideTwo"),
            75.0,
            places=4,
            msg="a paralyzed Tauros must still land Tackle 75% of the time",
        )
        self.assertAlmostEqual(self._mass(branches, "-> POISON"), 0.0, places=4)

    def test_white_herb_restores_a_self_inflicted_drop_and_is_consumed(self) -> None:
        # A6. gen3 White Herb was absent from the engine entirely (WHITEHERB
        # existed only under src/genx/), so a Deoxys never restored Superpower's
        # -1 Def and the next physical hit landed against a dropped stat.
        # Showdown: data/items.ts:7653, no gen3/gen4 override -- zero the
        # NEGATIVE boosts, consume, render `-enditem` + `-clearnegativeboost`.
        # The item is knowable: teams.ts:471 gives it to Deoxys unconditionally.
        attacker = self._mon(
            "deoxys", "pressure", "superpower", speed=200, item="whiteherb"
        )
        defender = self._mon("golem", "rockhead", "splash")
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "superpower", "splash"
        )
        text = " || ".join(self._text(b) for b in branches)
        # Superpower drops the USER's Attack and Defense by one each.
        self.assertIn("Boost SideOne Attack: -1", text)
        self.assertIn("Boost SideOne Defense: -1", text)
        # White Herb restores both to zero and is consumed.
        self.assertIn("Boost SideOne Attack: 1", text)
        self.assertIn("Boost SideOne Defense: 1", text)
        self.assertIn("ChangeItem SideOne: WHITEHERB -> NONE", text)
        # Exactly once. Three call sites each check both sides, so the
        # early-return-on-consumed guard is what stops a double proc; this locks
        # it rather than leaving it structurally implied.
        for branch in branches:
            self.assertLessEqual(
                self._text(branch).count("ChangeItem SideOne: WHITEHERB -> NONE"),
                1,
                "White Herb must be consumed at most once per branch",
            )

    def test_white_herb_leaves_positive_boosts_alone(self) -> None:
        # It is not a reset. Showdown zeroes only the NEGATIVE entries, so a
        # Pokemon that is +2 Speed and -1 Defense keeps the +2. Without this the
        # implementation could pass the test above by clearing every boost.
        attacker = self._mon(
            "deoxys", "pressure", "superpower", speed=200, item="whiteherb"
        )
        defender = self._mon("golem", "rockhead", "splash")
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender, attacker_speed_boost=2),
            "superpower",
            "splash",
        )
        text = " || ".join(self._text(b) for b in branches)
        self.assertIn("ChangeItem SideOne: WHITEHERB -> NONE", text)
        self.assertNotIn("Boost SideOne Speed", text)

    def test_white_herb_fires_after_a_secondary_effect_drop(self) -> None:
        # Site 2. Secondaries are applied AFTER the boost-effect site, so the
        # first version of this patch missed them entirely: a Deoxys hit by
        # Crunch kept its -1 SpD. Richly reachable -- counted from sets.json
        # movepools, Ice Beam is on 79 pool species, Shadow Ball 59, Thunderbolt
        # 50, Psychic 42, Flamethrower 23, Crunch 9, Iron Tail 7.
        # gen3 Crunch drops SpD, not Def.
        deoxys = self._mon("deoxys", "pressure", "splash", item="whiteherb", speed=50)
        opponent = self._mon("tyranitar", "sandstream", "crunch", speed=200)
        branches = poke_engine.generate_instructions(
            self._state(deoxys, opponent), "splash", "crunch"
        )
        dropped = [b for b in branches if "SpecialDefense: -1" in self._text(b)]
        self.assertTrue(dropped, "Crunch must actually drop SpD, or this pin is vacuous")
        for branch in dropped:
            text = self._text(branch)
            self.assertIn("Boost SideOne SpecialDefense: 1", text)
            self.assertIn("ChangeItem SideOne: WHITEHERB -> NONE", text)

    def test_white_herb_fires_on_an_intimidate_switch_in(self) -> None:
        # Site 3. Intimidate drops the opposing Attack as the holder switches in,
        # which is neither a move's boost effect nor a secondary, so the first
        # version of this patch missed it. 18 gen3 randbats species carry
        # Intimidate: arbok, arcanine, granbull, gyarados, hitmontop, masquerain,
        # mawile, mightyena, salamence, stantler, tauros and more.
        deoxys = self._mon("deoxys", "pressure", "splash", item="whiteherb", speed=50)
        active = self._mon("pidgey", "keeneye", "splash", speed=200)
        gyarados = self._mon("gyarados", "intimidate", "splash", speed=200)
        branches = poke_engine.generate_instructions(
            self._state(deoxys, active, defender_party=[gyarados]),
            "splash",
            "gyarados",
        )
        text = " || ".join(self._text(b) for b in branches)
        self.assertIn("Boost SideOne Attack: -1", text)
        self.assertIn("Boost SideOne Attack: 1", text)
        self.assertIn("ChangeItem SideOne: WHITEHERB -> NONE", text)

    def test_a_multi_hit_move_partitions_on_its_TOTAL_not_its_per_hit_damage(self) -> None:
        # The KO partitions compared a PER-HIT maximum against the defender's
        # FULL HP with no hit_count scaling, so a two-hit move never partitioned
        # on a threshold only its TOTAL can straddle. Holdout row 19100113/62's
        # shape: per-hit max 140 against 253 HP while two hits reach 280.
        #
        # Sizing matters and a first draft of this pin got it wrong. Bonemerang
        # here has per-hit max 87, so the straddle band is
        # 2*0.85*87 = 148 < hp <= 2*87 = 174. At hp=120 EVERY roll kills either
        # way and the pin passed against the unfixed engine -- vacuous. 165 sits
        # inside the band: the collapsed average (80+80 = 160) survives, while
        # the top rolls (2*87 = 174) kill.
        attacker = self._mon("marowak", "rockhead", "bonemerang", attack=200, speed=200)
        defender = self._mon("registeel", "clearbody", "splash", hp=165, maxhp=165, defense=80)
        state = self._state(attacker, defender)

        per_hit_max = poke_engine.calculate_damage(state, "bonemerang", "splash", False)[0][0]
        per_hit_min = int(per_hit_max * 0.85)
        self.assertLess(per_hit_max, 165, "one hit must not KO, or there is nothing to fix")
        self.assertGreaterEqual(2 * per_hit_max, 165, "two hits must be able to KO")
        self.assertLess(2 * per_hit_min, 165, "the TOTAL must straddle, not merely exceed")

        branches = poke_engine.generate_instructions(state, "bonemerang", "splash")
        # Non-crit arms only: the crit arm partitions separately and already did.
        non_crit = [
            b for b in branches
            if "Damage SideTwo" in self._text(b) and float(b.percentage) > 10.0
        ]
        self.assertTrue(non_crit, "expected a non-crit damaging arm")

        def total(branch) -> int:
            return sum(
                int(m) for m in re.findall(r"Damage SideTwo: (\d+)", self._text(branch))
            )

        totals = sorted({total(b) for b in non_crit})
        # Before the fix the whole non-crit mass collapses to ONE total (160,
        # surviving). After it, the fan straddles 165 and both outcomes appear.
        self.assertGreater(
            len(totals), 1,
            f"non-crit mass did not partition: totals={totals}, "
            f"branches={[self._text(b)[:70] for b in branches]}",
        )
        self.assertTrue(any(t >= 165 for t in totals), f"no killing roll: {totals}")
        self.assertTrue(any(t < 165 for t in totals), f"no surviving roll: {totals}")

    def test_the_multi_hit_ko_gate_never_divides_by_zero(self) -> None:
        # Scaling the KO gate by hit_count introduced a REACHABLE PANIC, found by
        # the independent review of #1116 and reproduced before this pin existed.
        #
        # The gate floored THEN multiplied -- floor(0.85*max)*h -- while
        # `compare_health_with_damage_multiples` multiplies THEN scales,
        # (max*h)*0.85. So the gate's lowest total sat up to h*frac(0.85*max)
        # BELOW the comparator's lowest roll. When hp landed in that window the
        # gate fired but all 16 of the comparator's rolls were >= health, leaving
        # num_less_than == 0 and executing `total_less_than / num_less_than` =
        # 0/0 on i16: an integer-division panic.
        #
        # Unscaled the mismatch was impossible: for integer t, floor(x) < t
        # implies x < t. Multiplying by hit_count breaks exactly that proof.
        #
        # RED RUN, on the engine as #1116 first shipped it:
        #   bonemerang  hp=147 -> PanicException (146 and 148 were fine)
        #   bonerush    hp=112 -> PanicException
        #   furyswipes  hp=121 -> PanicException
        #   spikecannon hp=136 -> PanicException
        # A 400-game differential sweep reported engine_errors: 0 the whole time,
        # because the window is one or two HP values per matchup. That sweep was
        # sample size, not safety -- which is the reason this pin sweeps an axis
        # rather than asserting one state.
        multi_hit = [
            "bonemerang", "bonerush", "doublekick", "doubleslap", "furyattack",
            "furyswipes", "pinmissile", "spikecannon", "twineedle", "barrage",
            "cometpunch",
        ]
        panics: list[tuple[str, int, str]] = []
        for move in multi_hit:
            attacker = self._mon("marowak", "rockhead", move, attack=200, speed=200)
            for hp in range(1, 301):
                defender = self._mon(
                    "registeel", "clearbody", "splash", hp=hp, maxhp=hp, defense=80
                )
                try:
                    poke_engine.generate_instructions(
                        self._state(attacker, defender), move, "splash"
                    )
                except BaseException as exc:  # noqa: BLE001 - a panic is the bug
                    panics.append((move, hp, type(exc).__name__))
        self.assertEqual(panics, [], f"multi-hit KO gate panicked: {panics[:12]}")

        # Anti-vacuity: the sweep must actually cross the straddle band, or it
        # would pass on an axis that never reaches the partition at all.
        attacker = self._mon("marowak", "rockhead", "bonemerang", attack=200, speed=200)
        probe = self._mon("registeel", "clearbody", "splash", hp=165, maxhp=165, defense=80)
        per_hit_max = poke_engine.calculate_damage(
            self._state(attacker, probe), "bonemerang", "splash", False
        )[0][0]
        self.assertLess(int(per_hit_max * 0.85) * 2, 300, "band must lie inside the swept axis")
        self.assertGreater(per_hit_max * 2, 1, "band must be non-empty")

    def test_hit_count_scaling_does_not_kill_the_residual_third_arm(self) -> None:
        # Third blocking finding of #1116's review. The residual arm counts the
        # rolls that SURVIVE the move but die to residuals, and it gets that count
        # by SUBTRACTING num_kill_rolls -- so both counts must share a basis.
        # Scaling the KO partition made num_kill_rolls TOTAL while this arm still
        # compared a PER-HIT max against the threshold. For a multi-hit move the
        # total clears hp on more rolls than the per-hit damage clears the LOWER
        # threshold, so num_at_or_above - num_kill_rolls went NEGATIVE, failed
        # `> 0`, and the arm silently vanished for exactly the population #1116 is
        # about. These are i16, so it underflowed to a skip rather than panicking,
        # which is why no gate caught it and why this pin exists.
        #
        # THE FIRST VERSION OF THIS PIN WAS VACUOUS AND IS RECORDED BECAUSE IT
        # PASSED. It used hp=164 / maxhp=200 and matched "two hits plus a third
        # smaller chunk that finishes it" -- but that shape is also what the
        # ORDINARY survive arm looks like once residuals are applied downstream, so
        # it passed against the unfixed engine. Matching the residual arm's SHAPE
        # cannot distinguish it; only the MASS SPLIT it introduces can.
        #
        # What the arm actually does is separate "survives the move, dies to the
        # residual" from "survives both". So the discriminator needs all THREE
        # outcome classes to be simultaneously reachable, which requires the roll
        # spread to exceed the residual chunk:
        #     0.3 * per_hit_max > maxhp / 8
        # At attack=400 / level=100 / defense=85, per_hit_max is 199, so the total
        # band is 338..398 and poison is 50; hp=395 puts pure survival (338..344),
        # residual death (345..394) and direct death (395..398) all in range.
        #
        # RED RUN on the engine as #1116 shipped it -- the surviving mass was ONE
        # undifferentiated arm that always died to poison:
        #     hp=395 buggy: [(10.0, [50]), (79.1, [183, 183, 29]), (10.9, [395])]
        #     hp=395 fixed: [(10.0, [50]), (10.55, [170, 170, 50]),
        #                    (10.9, [395]), (64.27, [173, 173, 49])]
        # 10.55% of the mass is priced as a survival by the fix and as a death by
        # the bug. That is the regression this pin exists to catch.
        #
        # An earlier revision of this comment recorded the fixed output's last arm
        # as `(68.55, [345, 50])` and called that correct. IT WAS THE SECOND BUG,
        # baked into the test record as expected behaviour -- see
        # `test_the_residual_arm_lets_the_defender_take_its_turn` below.
        attacker = self._mon(
            "marowak", "rockhead", "bonemerang", attack=400, speed=200, level=100
        )
        defender = self._mon(
            "registeel", "clearbody", "splash",
            hp=395, maxhp=400, defense=85, status="poison",
        )
        state = self._state(attacker, defender)

        per_hit_max = poke_engine.calculate_damage(state, "bonemerang", "splash", False)[0][0]
        residual = 400 // 8
        self.assertLess(per_hit_max, 395, "one hit must not KO, or there is no partition")
        self.assertGreaterEqual(2 * per_hit_max, 395, "two hits must be able to KO")
        self.assertLess(2 * int(per_hit_max * 0.85), 395, "the TOTAL must straddle 395")
        # The band must be wide enough for a PURE survival to exist, or the pin
        # degenerates back into the vacuous version above.
        self.assertLess(
            2 * int(per_hit_max * 0.85) + residual, 395,
            "lowest total plus the residual must still survive, or there is no third class",
        )

        # SECOND VACUITY, ALSO RECORDED. "sum(hits) < hp" alone is satisfied by the
        # 10% MISS arm, which on this state is a bare [50] of poison -- so that
        # predicate passed on the buggy engine too. The arm being looked for must
        # contain BOTH of the move's hits as well as the residual chunk, hence
        # len(hits) >= 3. Against the recorded runs above that separates them
        # exactly: buggy [183, 183, 29] totals 395 and is lethal, fixed
        # [170, 170, 50] totals 390 and is not.
        branches = poke_engine.generate_instructions(state, "bonemerang", "splash")
        survives_everything = []
        missed = []
        for branch in branches:
            hits = [
                int(m) for m in re.findall(r"Damage SideTwo: (\d+)", self._text(branch))
            ]
            if float(branch.percentage) <= 1.0:
                continue
            if len(hits) < 3:
                missed.append((float(branch.percentage), hits))
                continue
            if sum(hits) < 395:
                survives_everything.append((float(branch.percentage), hits))

        self.assertTrue(
            survives_everything,
            "no arm lands both hits and still survives the residual, so the residual "
            "third arm collapsed for a multi-hit move: "
            f"{[(round(float(b.percentage), 2), self._text(b)[:60]) for b in branches]}",
        )
        # Anti-vacuity on the exclusion itself: the miss arm must really have been
        # filtered out, not absent, or len(hits) >= 3 is doing nothing.
        self.assertTrue(
            any(h == [residual] for _, h in missed),
            f"expected the miss arm to be excluded by the hit-count filter: {missed}",
        )

    def test_the_residual_arm_lets_the_defender_take_its_turn(self) -> None:
        # Second review of #1116, BLOCK 1. `run_move` applies `damage_amount` ONCE
        # PER HIT, so the residual arm's threshold has to be per-hit like every
        # other arm this patch touches. Pushing the TOTAL made the arm deal
        # `hit_count * residual_threshold`, clamped at HP -- so the arm meaning
        # "survives the move, dies to the residual" killed on the MOVE and deleted
        # the defender's turn.
        #
        # THE RENDERING IS WHAT FOOLED ME. The arm came out as `[345, 50]`, which
        # reads naturally as "345 from the move, then 50 from poison" -- and poison
        # on a 400 maxhp defender really is 50, so the numbers corroborated the
        # wrong story. It was hit one for 345 and hit two clamped to the remaining
        # 50. I called it cosmetic in the PR body on that reading. It was a
        # 68.55pp fidelity regression against main.
        #
        # The tell is structural, not numeric: 345 alone cannot kill a 395 HP
        # defender, so if the defender never acts the damage must have been applied
        # more than once. This pin gives the defender a damaging move and makes it
        # SLOWER, then asserts it acts -- which no amount of reading damage
        # magnitudes would have caught.
        attacker = self._mon(
            "marowak", "rockhead", "bonemerang", attack=400, speed=200, level=100
        )
        defender = self._mon(
            "registeel", "clearbody", "tackle",
            hp=395, maxhp=400, defense=85, status="poison", speed=10,
        )
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "bonemerang", "tackle"
        )

        offenders = []
        for branch in branches:
            text = self._text(branch)
            hits = [int(m) for m in re.findall(r"Damage SideTwo: (\d+)", text)]
            if len(hits) < 3 or float(branch.percentage) <= 1.0:
                continue
            # A multi-hit arm that keeps the defender below its HP must leave it
            # alive to move. Only the genuine KO arm may skip the defender's turn.
            if sum(hits[:2]) < 395 and "Damage SideOne" not in text:
                offenders.append((float(branch.percentage), hits, text[:90]))

        self.assertEqual(
            offenders, [],
            "a residual arm killed on the move and deleted the defender's turn: "
            f"{offenders}",
        )

        # Anti-vacuity: the residual arm must actually be present and carry mass,
        # or the assertion above is trivially satisfied by its absence.
        acting = [
            (float(b.percentage), self._text(b))
            for b in branches
            if len(re.findall(r"Damage SideTwo: (\d+)", self._text(b))) >= 3
            and "Damage SideOne" in self._text(b)
            and float(b.percentage) > 50.0
        ]
        self.assertTrue(
            acting,
            "expected a majority-mass multi-hit residual arm in which the defender "
            f"acts: {[(round(float(b.percentage), 2), self._text(b)[:70]) for b in branches]}",
        )

    def test_effect_spore_invalid_outcomes_keep_their_probability_mass(self) -> None:
        defender = self._mon("breloom", "effectspore", "splash")
        poison_attacker = self._mon(
            "nidoking", "poisonpoint", "tackle", types=("poison", "ground"), speed=200
        )
        poison_branches = poke_engine.generate_instructions(
            self._state(poison_attacker, defender), "tackle", "splash"
        )
        self.assertEqual(self._mass(poison_branches, "-> POISON"), 0.0)
        self.assertAlmostEqual(self._mass(poison_branches, "-> PARALYZE"), 100.0 / 30.0, places=4)
        self.assertAlmostEqual(self._mass(poison_branches, "-> SLEEP"), 100.0 / 30.0, places=4)

        attacker = self._mon("tauros", "intimidate", "tackle", speed=200)
        sleeping_reserve = self._mon("snorlax", "immunity", "splash", status="sleep")
        clause_branches = poke_engine.generate_instructions(
            self._state(attacker, defender, attacker_party=(sleeping_reserve,)),
            "tackle",
            "splash",
        )
        self.assertEqual(self._mass(clause_branches, "-> SLEEP"), 0.0)
        self.assertAlmostEqual(self._mass(clause_branches, "-> POISON"), 100.0 / 30.0, places=4)
        self.assertAlmostEqual(self._mass(clause_branches, "-> PARALYZE"), 100.0 / 30.0, places=4)

    def test_persistent_status_immunity_matrix(self) -> None:
        cases = (
            ("waterveil", "willowisp", "BURN"),
            ("magmaarmor", "icebeam", "FREEZE"),
            ("insomnia", "sleeppowder", "SLEEP"),
            ("vitalspirit", "sleeppowder", "SLEEP"),
            ("limber", "thunderwave", "PARALYZE"),
            ("immunity", "toxic", "TOXIC"),
        )
        for ability, move, status in cases:
            with self.subTest(ability=ability):
                attacker = self._mon("mew", "synchronize", move, speed=200)
                defender = self._mon("snorlax", ability, "splash")
                branches = poke_engine.generate_instructions(self._state(attacker, defender), move, "splash")
                self.assertFalse(any(f"SideTwo-P0: NONE -> {status}" in self._text(branch) for branch in branches))

    def test_trace_cures_status_incompatible_with_copied_ability(self) -> None:
        cases = (
            ("waterveil", "burn"),
            ("magmaarmor", "freeze"),
            ("limber", "paralyze"),
            ("immunity", "poison"),
            ("immunity", "toxic"),
            ("insomnia", "sleep"),
            ("vitalspirit", "sleep"),
        )
        for copied_ability, status in cases:
            with self.subTest(ability=copied_ability, status=status):
                lead = self._mon("tauros", "intimidate", "splash")
                tracer = self._mon("gardevoir", "trace", "splash", status=status)
                opponent = self._mon("snorlax", copied_ability, "splash")
                state = self._state(lead, opponent, attacker_party=(tracer,))
                branches = poke_engine.generate_instructions(state, "gardevoir", "splash")
                self.assertTrue(branches)
                for branch in branches:
                    applied = state.apply_instructions(branch)
                    active = applied.side_one.pokemon[1]
                    self.assertEqual(str(active.ability).upper(), copied_ability.upper())
                    self.assertEqual(str(active.status).upper(), "NONE")

    def test_own_tempo_and_oblivious_prevent_turn_denial(self) -> None:
        for ability, volatile in (("owntempo", "CONFUSION"), ("oblivious", "ATTRACT")):
            with self.subTest(ability=ability):
                attacker = self._mon("spinda", ability, "tackle", speed=200)
                defender = self._mon("snorlax", "thickfat", "splash")
                branches = poke_engine.generate_instructions(
                    self._state(attacker, defender, attacker_volatiles={volatile}),
                    "tackle",
                    "splash",
                )
                self.assertAlmostEqual(self._mass(branches, "Damage SideTwo"), 100.0, places=4)

    def test_wonder_guard_blocks_only_non_super_effective_direct_damage(self) -> None:
        shedinja = self._mon(
            "shedinja", "wonderguard", "splash", types=("bug", "ghost"), hp=1, maxhp=1
        )
        cases = (("tackle", False), ("rockslide", True), ("struggle", True))
        for move, should_damage in cases:
            with self.subTest(move=move):
                attacker = self._mon("tauros", "intimidate", move, speed=200)
                branches = poke_engine.generate_instructions(self._state(attacker, shedinja), move, "splash")
                has_damage = any("Damage SideTwo" in self._text(branch) for branch in branches)
                self.assertEqual(has_damage, should_damage)

        poisoner = self._mon("mew", "synchronize", "toxic", speed=200)
        status_branches = poke_engine.generate_instructions(
            self._state(poisoner, shedinja), "toxic", "splash"
        )
        self.assertTrue(any("SideTwo-P0: NONE -> TOXIC" in self._text(branch) for branch in status_branches))

    def test_rock_head_suppresses_move_recoil_but_not_struggle(self) -> None:
        defender = self._mon("snorlax", "thickfat", "splash", hp=600, maxhp=600)
        for move, expect_recoil in (("doubleedge", False), ("struggle", True)):
            with self.subTest(move=move):
                attacker = self._mon("aerodactyl", "rockhead", move, speed=200)
                branches = poke_engine.generate_instructions(self._state(attacker, defender), move, "splash")
                has_recoil = any("Damage SideOne" in self._text(branch) for branch in branches)
                self.assertEqual(has_recoil, expect_recoil)

    def test_lightning_rod_is_inert_in_gen3_singles(self) -> None:
        attacker = self._mon("raikou", "pressure", "thunderbolt", speed=200)
        defender = self._mon("rhydon", "lightningrod", "splash", types=("rock", "normal"))
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "thunderbolt", "splash"
        )
        self.assertTrue(any("Damage SideTwo" in self._text(branch) for branch in branches))
        self.assertFalse(any("Boost SideTwo SpecialAttack" in self._text(branch) for branch in branches))

    def test_intimidate_does_not_cross_substitute(self) -> None:
        lead = self._mon("tauros", "none", "splash")
        intimidator = self._mon("gyarados", "intimidate", "splash")
        defender = self._mon("snorlax", "thickfat", "splash")
        open_branch = poke_engine.generate_instructions(
            self._state(lead, defender, attacker_party=(intimidator,)), "gyarados", "splash"
        )[0]
        sub_branch = poke_engine.generate_instructions(
            self._state(
                lead,
                defender,
                attacker_party=(intimidator,),
                defender_volatiles={"SUBSTITUTE"},
                substitute_health=100,
            ),
            "gyarados",
            "splash",
        )[0]
        self.assertIn("Boost SideTwo Attack: -1", self._text(open_branch))
        self.assertNotIn("Boost SideTwo Attack: -1", self._text(sub_branch))

    def test_flash_fire_will_o_wisp_edge_does_not_false_activate(self) -> None:
        attacker = self._mon("mew", "synchronize", "willowisp", speed=200)
        poisoned = self._mon("houndoom", "flashfire", "splash", status="poison")
        branches = poke_engine.generate_instructions(
            self._state(attacker, poisoned), "willowisp", "splash"
        )
        self.assertFalse(any("FLASHFIRE" in self._text(branch) for branch in branches))

        fresh = self._mon("houndoom", "flashfire", "splash")
        fresh_branches = poke_engine.generate_instructions(
            self._state(attacker, fresh), "willowisp", "splash"
        )
        self.assertTrue(any("FLASHFIRE" in self._text(branch) for branch in fresh_branches))
        self.assertFalse(any("SideTwo-P0: NONE -> BURN" in self._text(branch) for branch in fresh_branches))

    def test_flash_fire_does_not_absorb_while_frozen_and_fire_hit_thaws(self) -> None:
        attacker = self._mon("charizard", "blaze", "flamethrower", speed=200)
        frozen = self._mon(
            "houndoom",
            "flashfire",
            "splash",
            types=("fire", "dark"),
            status="freeze",
        )
        branches = poke_engine.generate_instructions(
            self._state(attacker, frozen), "flamethrower", "splash"
        )
        self.assertTrue(any("Damage SideTwo" in self._text(branch) for branch in branches))
        self.assertTrue(all("FREEZE -> NONE" in self._text(branch) for branch in branches))
        self.assertFalse(any("FLASHFIRE" in self._text(branch) for branch in branches))

    def test_sand_veil_is_exact_and_weather_suppressors_disable_it(self) -> None:
        attacker = self._mon(
            "flygon", "levitate", "tackle", types=("ground", "dragon"), speed=200
        )
        defender = self._mon("cacturne", "sandveil", "splash")
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender, weather="sand"), "tackle", "splash"
        )
        self.assertAlmostEqual(self._mass(branches, "Damage SideTwo"), 80.0, places=4)

        suppressor = self._mon("rayquaza", "airlock", "tackle", speed=200)
        suppressed = poke_engine.generate_instructions(
            self._state(suppressor, defender, weather="sand"), "tackle", "splash"
        )
        self.assertAlmostEqual(self._mass(suppressed, "Damage SideTwo"), 100.0, places=4)

    def test_fainted_weather_suppressor_no_longer_blocks_residual(self) -> None:
        # The suppressor faints, so the residual block it was blocking is DEFERRED
        # past the forced replacement (poke-engine-gen3-residual-defer-on-faint.patch)
        # and lands on the ply that resolves the switch. The ability assertion is
        # unchanged — a fainted Air Lock holder stops suppressing the weather, so
        # side one finally takes its sandstorm tick — it just moved one ply.
        attacker = self._mon("tauros", "intimidate", "tackle", speed=200)
        suppressor = self._mon(
            "rayquaza", "airlock", "splash", types=("dragon", "flying"), hp=1, maxhp=300
        )
        replacement = self._mon("snorlax", "immunity", "splash")
        state = self._state(
            attacker, suppressor, weather="sand", defender_party=(replacement,)
        )

        faint = poke_engine.generate_instructions(state, "tackle", "splash")
        self.assertAlmostEqual(self._mass(faint, "Damage SideOne: 18"), 0.0, places=4)
        self.assertAlmostEqual(
            self._mass(faint, "ToggleSideTwoForceSwitch"), 100.0, places=4
        )

        replaced = state.apply_instructions(faint[0])
        deferred = poke_engine.generate_instructions(replaced, "none", "snorlax")
        self.assertAlmostEqual(
            self._mass(deferred, "Damage SideOne: 18"), 100.0, places=4
        )

    def test_weather_ability_replaces_finite_same_weather(self) -> None:
        lead = self._mon("tauros", "intimidate", "splash")
        setter = self._mon("tyranitar", "sandstream", "splash", types=("rock", "dark"))
        defender = self._mon("snorlax", "immunity", "splash")
        state = self._state(
            lead,
            defender,
            weather="sand",
            weather_turns_remaining=2,
            attacker_party=(setter,),
        )
        branches = poke_engine.generate_instructions(state, "tyranitar", "splash")
        self.assertTrue(all("ChangeWeather" in self._text(branch) for branch in branches))
        for branch in branches:
            applied = state.apply_instructions(branch)
            self.assertEqual(applied.weather_turns_remaining, -1)

    def test_liquid_ooze_damage_is_capped_at_current_hp(self) -> None:
        attacker = self._mon("venusaur", "overgrow", "gigadrain", hp=10, maxhp=300, speed=200)
        defender = self._mon("swalot", "liquidooze", "splash", hp=500, maxhp=500)
        state = self._state(attacker, defender)
        branches = poke_engine.generate_instructions(state, "gigadrain", "splash")
        self.assertTrue(branches)
        for branch in branches:
            applied = state.apply_instructions(branch)
            self.assertEqual(applied.side_one.pokemon[0].hp, 0)
            self.assertIn("Heal SideOne: -10", self._text(branch))

    def test_liquid_ooze_reverses_leech_seed_recovery(self) -> None:
        seeder = self._mon("venusaur", "overgrow", "splash", hp=10, maxhp=300, speed=200)
        seeded = self._mon("swalot", "liquidooze", "splash", hp=160, maxhp=160)
        state = self._state(seeder, seeded, defender_volatiles={"LEECHSEED"})
        branches = poke_engine.generate_instructions(state, "splash", "splash")
        self.assertTrue(branches)
        for branch in branches:
            applied = state.apply_instructions(branch)
            self.assertEqual(applied.side_one.pokemon[0].hp, 0)
            self.assertEqual(applied.side_two.pokemon[0].hp, 140)
            self.assertIn("Heal SideOne: -10", self._text(branch))

    def test_python_pokemon_positional_types_remain_backward_compatible(self) -> None:
        mon = poke_engine.Pokemon("pikachu", 50, ("electric", "typeless"))
        self.assertEqual(tuple(str(value).upper() for value in mon.types), ("ELECTRIC", "TYPELESS"))
        self.assertEqual(str(mon.gender).upper(), "NONE")
        female = poke_engine.Pokemon(
            "pikachu", 50, ("electric", "typeless"), gender="female"
        )
        self.assertEqual(str(female.gender).upper(), "FEMALE")

    def test_early_bird_first_attempt_wake_probability(self) -> None:
        defender = self._mon("snorlax", "thickfat", "splash")
        early = self._mon(
            "xatu", "earlybird", "tackle", speed=200, status="sleep", sleep_turns=0
        )
        branches = poke_engine.generate_instructions(
            self._state(early, defender), "tackle", "splash"
        )
        self.assertAlmostEqual(self._mass(branches, "Damage SideTwo"), 25.0, places=4)

    def test_synchronize_reflects_toxic_as_regular_poison(self) -> None:
        attacker = self._mon("mew", "pressure", "toxic", speed=200)
        defender = self._mon("alakazam", "synchronize", "splash")
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "toxic", "splash"
        )
        reflected = [
            branch
            for branch in branches
            if "SideTwo-P0: NONE -> TOXIC" in self._text(branch)
        ]
        self.assertTrue(reflected)
        self.assertTrue(
            all("SideOne-P0: NONE -> POISON" in self._text(branch) for branch in reflected)
        )

    def test_synchronize_reflects_contact_ability_status(self) -> None:
        attacker = self._mon("alakazam", "synchronize", "tackle", speed=200)
        defender = self._mon("breloom", "effectspore", "splash", types=("grass", "fighting"))
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "tackle", "splash"
        )
        poisoned = [
            branch
            for branch in branches
            if "SideOne-P0: NONE -> POISON" in self._text(branch)
        ]
        self.assertTrue(poisoned)
        self.assertTrue(
            all("SideTwo-P0: NONE -> POISON" in self._text(branch) for branch in poisoned)
        )

    def test_synchronize_reflects_before_lum_berry_cures(self) -> None:
        attacker = self._mon("mew", "pressure", "willowisp", speed=200)
        defender = self._mon("alakazam", "synchronize", "splash", item="lumberry")
        state = self._state(attacker, defender)
        branches = poke_engine.generate_instructions(state, "willowisp", "splash")
        hit_branches = [branch for branch in branches if "LUMBERRY -> NONE" in self._text(branch)]
        self.assertTrue(hit_branches)
        for branch in hit_branches:
            applied = state.apply_instructions(branch)
            self.assertEqual(str(applied.side_one.pokemon[0].status).upper(), "BURN")
            self.assertEqual(str(applied.side_two.pokemon[0].status).upper(), "NONE")
            self.assertEqual(str(applied.side_two.pokemon[0].item).upper(), "NONE")

    def test_synchronize_reflection_triggers_source_lum_berry(self) -> None:
        attacker = self._mon("mew", "pressure", "toxic", speed=200, item="lumberry")
        defender = self._mon("alakazam", "synchronize", "splash")
        state = self._state(attacker, defender)
        branches = poke_engine.generate_instructions(state, "toxic", "splash")
        hit_branches = [
            branch
            for branch in branches
            if "SideTwo-P0: NONE -> TOXIC" in self._text(branch)
        ]
        self.assertTrue(hit_branches)
        for branch in hit_branches:
            applied = state.apply_instructions(branch)
            self.assertEqual(str(applied.side_one.pokemon[0].status).upper(), "NONE")
            self.assertEqual(str(applied.side_one.pokemon[0].item).upper(), "NONE")
            self.assertEqual(str(applied.side_two.pokemon[0].status).upper(), "TOXIC")

    def test_synchronize_bypasses_substitute_but_respects_safeguard(self) -> None:
        attacker = self._mon("mew", "pressure", "toxic", speed=200)
        defender = self._mon("alakazam", "synchronize", "splash")
        behind_substitute = poke_engine.generate_instructions(
            self._state(attacker, defender, attacker_volatiles={"SUBSTITUTE"}),
            "toxic",
            "splash",
        )
        reflected = [
            branch
            for branch in behind_substitute
            if "SideOne-P0: NONE -> POISON" in self._text(branch)
        ]
        self.assertTrue(reflected)

        safeguarded = poke_engine.generate_instructions(
            self._state(attacker, defender, attacker_safeguard=2),
            "toxic",
            "splash",
        )
        self.assertFalse(
            any("SideOne-P0: NONE -> POISON" in self._text(branch) for branch in safeguarded)
        )

    def test_pressure_consumes_two_pp_in_the_engine_relevant_range(self) -> None:
        attacker = self._mon("tauros", "intimidate", "tackle", speed=200, pp=9)
        defender = self._mon("lugia", "pressure", "splash")
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "tackle", "splash"
        )
        self.assertTrue(
            all("DecrementPP SideOne: M0 2" in self._text(branch) for branch in branches)
        )

    def test_sturdy_is_not_modern_focus_sash(self) -> None:
        attacker = self._mon("metagross", "clearbody", "explosion", speed=200)
        defender = self._mon("donphan", "sturdy", "splash", hp=200, maxhp=200)
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "explosion", "splash"
        )
        self.assertTrue(any("Damage SideTwo: 200" in self._text(branch) for branch in branches))

    def test_heal_bell_respects_soundproof_party_boundary(self) -> None:
        user = self._mon("exploud", "soundproof", "healbell", status="burn", speed=200)
        blocked_ally = self._mon("mr-mime", "soundproof", "splash", status="poison")
        cured_ally = self._mon("snorlax", "immunity", "splash", status="paralyze")
        # The opposing Soundproof holder must not consume this team-targeted move.
        defender = self._mon("exploud", "soundproof", "splash")
        state = self._state(
            user,
            defender,
            attacker_party=(blocked_ally, cured_ally),
        )

        branches = poke_engine.generate_instructions(state, "healbell", "splash")
        self.assertTrue(branches)
        for branch in branches:
            applied = state.apply_instructions(branch)
            self.assertEqual(str(applied.side_one.pokemon[0].status).upper(), "BURN")
            self.assertEqual(str(applied.side_one.pokemon[1].status).upper(), "POISON")
            self.assertEqual(str(applied.side_one.pokemon[2].status).upper(), "NONE")

    def test_yawn_resolution_rechecks_sleep_clause(self) -> None:
        yawned = self._mon("tauros", "intimidate", "tackle", speed=200)
        sleeping_ally = self._mon("snorlax", "immunity", "splash", status="sleep")
        defender = self._mon("swalot", "liquidooze", "splash")
        state = self._state(
            yawned,
            defender,
            attacker_party=(sleeping_ally,),
            attacker_volatiles={"YAWN"},
            attacker_yawn_duration=1,
        )
        branches = poke_engine.generate_instructions(state, "tackle", "splash")
        self.assertFalse(
            any("SideOne-P0: NONE -> SLEEP" in self._text(branch) for branch in branches)
        )

        control = self._state(
            yawned,
            defender,
            attacker_volatiles={"YAWN"},
            attacker_yawn_duration=1,
        )
        control_branches = poke_engine.generate_instructions(control, "tackle", "splash")
        self.assertTrue(
            any("SideOne-P0: NONE -> SLEEP" in self._text(branch) for branch in control_branches)
        )

    def test_yawn_initial_and_resolution_checks_match_showdown_phases(self) -> None:
        user = self._mon("swalot", "liquidooze", "yawn", speed=200)
        target = self._mon("tauros", "intimidate", "splash")
        sleeping_ally = self._mon("snorlax", "immunity", "splash", status="sleep")

        # Sleep Clause does not stop the Yawn volatile from landing.
        clause_state = self._state(user, target, defender_party=(sleeping_ally,))
        clause_branches = poke_engine.generate_instructions(clause_state, "yawn", "splash")
        self.assertTrue(
            any("ApplyVolatileStatus SideTwo: YAWN" in self._text(branch) for branch in clause_branches)
        )

        # Safeguard does stop the initial volatile.
        safeguarded = poke_engine.generate_instructions(
            self._state(user, target, defender_safeguard=2), "yawn", "splash"
        )
        self.assertFalse(
            any("ApplyVolatileStatus SideTwo: YAWN" in self._text(branch) for branch in safeguarded)
        )

        # Once Yawn is present, a later Safeguard does not block resolution.
        resolving = self._state(
            target,
            user,
            attacker_volatiles={"YAWN"},
            attacker_yawn_duration=1,
            attacker_safeguard=2,
        )
        resolution_branches = poke_engine.generate_instructions(resolving, "splash", "yawn")
        self.assertTrue(
            any("SideOne-P0: NONE -> SLEEP" in self._text(branch) for branch in resolution_branches)
        )

    def test_spikes_ko_prevents_switch_in_abilities(self) -> None:
        lead = self._mon("tauros", "intimidate", "splash")
        defender = self._mon("snorlax", "immunity", "splash")
        entrants = (
            self._mon(
                "tyranitar",
                "sandstream",
                "splash",
                types=("rock", "dark"),
                hp=1,
                maxhp=8,
            ),
            self._mon("gyarados", "intimidate", "splash", hp=1, maxhp=8),
        )
        for entrant in entrants:
            with self.subTest(ability=str(entrant.ability)):
                branches = poke_engine.generate_instructions(
                    self._state(lead, defender, attacker_party=(entrant,), attacker_spikes=1),
                    str(entrant.id).lower(),
                    "splash",
                )
                self.assertTrue(branches)
                for branch in branches:
                    text = self._text(branch)
                    self.assertIn("Damage SideOne: 1", text)
                    self.assertNotIn("ChangeWeather", text)
                    self.assertNotIn("Boost SideTwo Attack: -1", text)

    def test_speed_tie_does_not_compound_ability_modifiers(self) -> None:
        attacker = self._mon("butterfree", "compoundeyes", "thunder", speed=100)
        defender = self._mon("snorlax", "immunity", "tackle", hp=500, maxhp=500, speed=100)
        branches = poke_engine.generate_instructions(
            self._state(attacker, defender), "thunder", "tackle"
        )
        self.assertAlmostEqual(self._mass(branches, "Damage SideTwo"), 91.0, places=4)

    def test_forecast_updates_after_in_tree_weather_change(self) -> None:
        castform = self._mon(
            "castform",
            "forecast",
            "sunnyday",
            types=("normal", "typeless"),
            speed=200,
        )
        defender = self._mon("snorlax", "immunity", "splash")
        state = self._state(castform, defender)
        branches = poke_engine.generate_instructions(state, "sunnyday", "splash")
        self.assertTrue(branches)
        for branch in branches:
            applied = state.apply_instructions(branch)
            self.assertEqual(str(applied.weather).upper(), "SUN")
            self.assertEqual(
                tuple(str(t).upper() for t in applied.side_one.pokemon[0].types),
                ("FIRE", "TYPELESS"),
            )

    def test_early_bird_doubles_rest_countdown(self) -> None:
        early = self._mon(
            "kangaskhan",
            "earlybird",
            ("rest", "tackle"),
            hp=100,
            maxhp=300,
            speed=200,
        )
        defender = self._mon("snorlax", "immunity", "splash")
        state = self._state(early, defender)
        rested = state.apply_instructions(
            poke_engine.generate_instructions(state, "rest", "splash")[0]
        )
        self.assertEqual(rested.side_one.pokemon[0].rest_turns, 3)

        first_sleep_turn = poke_engine.generate_instructions(rested, "tackle", "splash")
        self.assertTrue(first_sleep_turn)
        for branch in first_sleep_turn:
            applied = rested.apply_instructions(branch)
            self.assertEqual(str(applied.side_one.pokemon[0].status).upper(), "SLEEP")
            self.assertEqual(applied.side_one.pokemon[0].rest_turns, 1)
            self.assertNotIn("Damage SideTwo", self._text(branch))

        sleeping = rested.apply_instructions(first_sleep_turn[0])
        wake_turn = poke_engine.generate_instructions(sleeping, "tackle", "splash")
        self.assertTrue(any("Damage SideTwo" in self._text(branch) for branch in wake_turn))
        for branch in wake_turn:
            applied = sleeping.apply_instructions(branch)
            self.assertEqual(str(applied.side_one.pokemon[0].status).upper(), "NONE")
            self.assertEqual(applied.side_one.pokemon[0].rest_turns, 0)


if __name__ == "__main__":
    unittest.main()
