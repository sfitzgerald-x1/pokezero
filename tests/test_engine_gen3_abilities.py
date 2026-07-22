"""High-risk regression gates for the Gen 3 randbats ability audit."""

from __future__ import annotations

import os
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
        hp: int = 300,
        maxhp: int = 300,
        speed: int = 100,
        status: str = "none",
        sleep_turns: int = 0,
        pp: int = 16,
        item: str = "none",
    ):
        return poke_engine.Pokemon(
            id=species,
            level=80,
            types=types,
            base_types=types,
            hp=hp,
            maxhp=maxhp,
            ability=ability,
            item=item,
            attack=180,
            defense=180,
            special_attack=180,
            special_defense=180,
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
        attacker_yawn_duration: int = 0,
    ):
        dummy = poke_engine.Pokemon(id="pikachu", level=1, hp=0)
        p1 = [attacker, *attacker_party]
        p2 = [defender, *defender_party]
        p1.extend([dummy] * (6 - len(p1)))
        p2.extend([dummy] * (6 - len(p2)))
        return poke_engine.State(
            side_one=poke_engine.Side(
                active_index="0",
                pokemon=p1,
                volatile_statuses=set(attacker_volatiles),
                volatile_status_durations=poke_engine.VolatileStatusDurations(
                    yawn=attacker_yawn_duration
                ),
                side_conditions=poke_engine.SideConditions(safeguard=attacker_safeguard),
            ),
            side_two=poke_engine.Side(
                active_index="0",
                pokemon=p2,
                volatile_statuses=set(defender_volatiles),
                substitute_health=substitute_health,
            ),
            weather=weather,
            weather_turns_remaining=weather_turns_remaining,
            terrain="none",
            trick_room=False,
        )

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
        attacker = self._mon("tauros", "intimidate", "tackle", speed=200)
        suppressor = self._mon(
            "rayquaza", "airlock", "splash", types=("dragon", "flying"), hp=1, maxhp=300
        )
        branches = poke_engine.generate_instructions(
            self._state(attacker, suppressor, weather="sand"), "tackle", "splash"
        )
        self.assertAlmostEqual(self._mass(branches, "Damage SideOne: 18"), 100.0, places=4)

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
            self.assertEqual(str(applied.side_one.pokemon[0].status).upper(), "NONE")
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
