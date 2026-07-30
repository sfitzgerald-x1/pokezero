"""Regression pins for the batch-E gen3 patches (slots 37-41).

Requires the patched wheel (``scripts/setup_poke_engine.sh``). Five patches:

``poke-engine-gen3-weather-move-targeting.patch``
    Field moves never TryHit the opposing Pokemon; the engine's default
    target of Opponent sailed Rain Dance past the defender-ability guard and
    Water Absorb healed maxhp/4 off it (row 1500124/58).

``poke-engine-gen3-recoil-rounding.patch``
    Sim recoil is clampIntRange(floor(dealt * recoil[0]/recoil[1]), 1);
    engine used trunc(f32 0.33 * dealt) (row 1500207/27).

``poke-engine-gen3-fixed-damage-pipeline.patch``
    check_move_hit_or_miss zeroed percent_hit on the (0,0) damage
    placeholder, so fixed-damage moves skipped run_move's whole post-damage
    suite: no damage_dealt (Counter after Seismic Toss returned nothing, row
    1500192/91), no contact abilities (rows 1500103/76, 1500274/15,
    1500287/76), no Super Fang accuracy roll, and Night Shade dealt nothing
    at all (Appendix K.3). Super Fang's amount was also a CEILING (hp - hp/2)
    where the sim floors with a min of 1.

``poke-engine-gen3-counter-hiddenpower-category.patch``
    gen3 Counter bounces Hidden Power whatever its elemental type; Mirror
    Coat never does (data/mods/gen3/moves.ts, explicit hiddenpower clauses;
    rows 1500155/22, 1500264/40).

``poke-engine-gen3-flashfire-phase1.patch``
    The Flash Fire volatile is a ModifyDamagePhase1 handler (pre-+2,
    fixed-point floor); the engine applied it as a trailing float after the
    type steps (row 1500123/79: sim max 92, engine 93).

Sim ground truth (gen3 Custom Game fixtures via pokezero.showdown_fixture,
2026-07-29, transcribed from live probe runs; per-scenario observations in
the test docstrings).
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import poke_engine
except ImportError:  # pragma: no cover - native wheel absent
    poke_engine = None


def _one_on_one(p1_kwargs, p2_kwargs, *, p1_side=None, p2_side=None):
    pe = poke_engine
    dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
    side_one = pe.Side(active_index="0", pokemon=[pe.Pokemon(**p1_kwargs)] + [dummy] * 5,
                       **(p1_side or {}))
    side_two = pe.Side(active_index="0", pokemon=[pe.Pokemon(**p2_kwargs)] + [dummy] * 5,
                       **(p2_side or {}))
    return pe.State(side_one=side_one, side_two=side_two,
                    weather="none", terrain="none", trick_room=False)


def _mon(**kwargs):
    base = dict(item="none", status="none")
    base.update(kwargs)
    return base


def _instructions(state, m1, m2):
    return [
        (round(b.percentage, 2), [str(i) for i in b.instruction_list])
        for b in poke_engine.generate_instructions(state, m1, m2)
    ]


def _flat(branches):
    return [ins for _, instrs in branches for ins in instrs]


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class WeatherMoveTargetingTests(unittest.TestCase):
    def _state(self, defender_ability):
        return _one_on_one(
            _mon(id="seaking", level=80, types=("water", "typeless"), hp=259, maxhp=259,
                 ability="swiftswim", attack=193, defense=150, special_attack=150,
                 special_defense=174, speed=155,
                 moves=[poke_engine.Move(id="raindance", pp=8)]),
            _mon(id="politoed", level=84, types=("water", "typeless"), hp=200, maxhp=288,
                 ability=defender_ability, attack=174, defense=174, special_attack=199,
                 special_defense=216, speed=166,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )

    def test_raindance_is_not_absorbed(self) -> None:
        """Sim probe (Seaking Rain Dance vs Water Absorb Politoed, seeds 1-3):
        the protocol shows |-weather|RainDance and NOTHING else — no heal, no
        -immune. Unpatched, the engine healed the Politoed maxhp/4 = 72 right
        after ChangeWeather (the recorded row's exact shape)."""
        flat = _flat(_instructions(self._state("waterabsorb"), "raindance", "splash"))
        self.assertTrue(any("ChangeWeather" in i for i in flat))
        self.assertFalse(any(i.startswith("Heal SideTwo: 72") for i in flat))

    def test_water_move_is_still_absorbed(self) -> None:
        """Control (both builds): a TARGETED Water move still activates Water
        Absorb — sim probe: Surf into Water Absorb answers |-immune (full HP)
        / heals; the engine's absorb arm converts the move to a heal and no
        damage instruction appears."""
        state = _one_on_one(
            _mon(id="seaking", level=80, types=("water", "typeless"), hp=259, maxhp=259,
                 ability="swiftswim", attack=193, defense=150, special_attack=150,
                 special_defense=174, speed=155,
                 moves=[poke_engine.Move(id="surf", pp=24)]),
            _mon(id="politoed", level=84, types=("water", "typeless"), hp=200, maxhp=288,
                 ability="waterabsorb", attack=174, defense=174, special_attack=199,
                 special_defense=216, speed=166,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        flat = _flat(_instructions(state, "surf", "splash"))
        self.assertFalse(any(i.startswith("Damage SideTwo") for i in flat))


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class RecoilRoundingTests(unittest.TestCase):
    def test_double_edge_recoil_floors_exact_thirds(self) -> None:
        """Sim probe (Wigglytuff L87 Double-Edge vs Cloyster L82, seeds 1-8):
        every (damage, recoil) pair satisfies recoil = floor(damage/3); seed 4
        is the live discriminator (damage 60 -> recoil 20, where f32
        trunc(0.33*60) = 19). This pin uses stats whose branch damage is 120:
        patched recoil 40, unpatched 39."""
        state = _one_on_one(
            _mon(id="wigglytuff", level=87, types=("normal", "typeless"), hp=385, maxhp=385,
                 ability="cutecharm", attack=150, defense=128, special_attack=180,
                 special_defense=137, speed=128,
                 moves=[poke_engine.Move(id="doubleedge", pp=15)]),
            _mon(id="cloyster", level=82, types=("water", "ice"), hp=400, maxhp=400,
                 ability="shellarmor", attack=203, defense=151, special_attack=187,
                 special_defense=121, speed=162,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        flat = _flat(_instructions(state, "doubleedge", "splash"))
        self.assertIn("Damage SideTwo: 120", flat)
        self.assertIn("Damage SideOne: 40", flat)
        self.assertNotIn("Damage SideOne: 39", flat)


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class FixedDamagePipelineTests(unittest.TestCase):
    def test_seismic_toss_triggers_flame_body(self) -> None:
        """Sim probe (Dusclops L83 Seismic Toss vs Flame Body Magcargo, seeds
        1-12): damage 83 = level every time; |-status brn [from] ability:
        Flame Body on 7/12 seeds (the 1/3 proc). Unpatched the engine emitted
        a single branch with no burn."""
        state = _one_on_one(
            _mon(id="dusclops", level=83, types=("ghost", "typeless"), hp=202, maxhp=202,
                 ability="pressure", attack=164, defense=263, special_attack=147,
                 special_defense=263, speed=89,
                 moves=[poke_engine.Move(id="seismictoss", pp=16)]),
            _mon(id="magcargo", level=83, types=("fire", "rock"), hp=219, maxhp=219,
                 ability="flamebody", attack=131, defense=247, special_attack=180,
                 special_defense=180, speed=97,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        branches = _instructions(state, "seismictoss", "splash")
        flat = _flat(branches)
        self.assertIn("Damage SideTwo: 83", flat)
        self.assertTrue(any("ChangeStatus SideOne-P0: NONE -> BURN" in i for i in flat))

    def test_seismic_toss_triggers_rough_skin(self) -> None:
        """Sim probe (Togetic L100 maxhp 272 Seismic Toss vs Rough Skin
        Sharpedo, seeds 1-3): attacker takes 17 = floor(272/16) every time.
        Unpatched: nothing."""
        state = _one_on_one(
            _mon(id="togetic", level=100, types=("normal", "flying"), hp=272, maxhp=272,
                 ability="serenegrace", attack=137, defense=227, special_attack=217,
                 special_defense=267, speed=137,
                 moves=[poke_engine.Move(id="seismictoss", pp=16)]),
            _mon(id="sharpedo", level=76, types=("water", "dark"), hp=231, maxhp=231,
                 ability="roughskin", attack=226, defense=105, special_attack=188,
                 special_defense=105, speed=188,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        flat = _flat(_instructions(state, "seismictoss", "splash"))
        self.assertIn("Damage SideTwo: 100", flat)
        self.assertIn("Damage SideOne: 17", flat)

    def test_counter_bounces_seismic_toss(self) -> None:
        """Sim probe (Illumise L92 Seismic Toss then Wobbuffet Counter, seeds
        1-3): Wobbuffet takes 92, Counter returns 184 = 2x. Unpatched the
        engine registered no damage_dealt for fixed damage and Counter
        returned nothing (row 1500192/91)."""
        state = _one_on_one(
            _mon(id="wobbuffet", level=100, types=("psychic", "typeless"), hp=542, maxhp=542,
                 ability="shadowtag", attack=123, defense=173, special_attack=123,
                 special_defense=173, speed=123,
                 moves=[poke_engine.Move(id="counter", pp=16)]),
            _mon(id="illumise", level=92, types=("bug", "typeless"), hp=269, maxhp=269,
                 ability="oblivious", attack=139, defense=154, special_attack=187,
                 special_defense=190, speed=209,
                 moves=[poke_engine.Move(id="seismictoss", pp=16)]),
        )
        flat = _flat(_instructions(state, "counter", "seismictoss"))
        self.assertIn("Damage SideOne: 92", flat)
        self.assertIn("Damage SideTwo: 184", flat)

    def test_super_fang_floors_and_can_miss(self) -> None:
        """Sim probe (Raticate Super Fang vs Milotic hp 352, seeds 1-12):
        damage exactly floor(352/2) = 176, with misses on 2/12 seeds (90%
        accuracy). This pin uses an ODD current hp (353): patched deals
        floor(353/2) = 176 and carries a miss branch; unpatched dealt the
        CEILING 177 with no miss branch."""
        state = _one_on_one(
            _mon(id="raticate", level=80, types=("normal", "typeless"), hp=219, maxhp=219,
                 ability="guts", attack=176, defense=142, special_attack=126,
                 special_defense=158, speed=201,
                 moves=[poke_engine.Move(id="superfang", pp=16)]),
            _mon(id="milotic", level=100, types=("water", "typeless"), hp=353, maxhp=400,
                 ability="marvelscale", attack=177, defense=215, special_attack=257,
                 special_defense=307, speed=219,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        branches = _instructions(state, "superfang", "splash")
        flat = _flat(branches)
        self.assertIn("Damage SideTwo: 176", flat)
        self.assertNotIn("Damage SideTwo: 177", flat)
        # 10% miss branch exists: some branch lacks the damage entirely
        self.assertTrue(any(not any(i.startswith("Damage SideTwo") for i in instrs)
                            for _, instrs in branches))

    def test_night_shade_deals_level(self) -> None:
        """Sim probe (Misdreavus L88 Night Shade vs Milotic, seeds 1-2):
        damage 88 = level. The engine dealt NOTHING before this patch
        (Appendix K.3's documented inertness)."""
        state = _one_on_one(
            _mon(id="misdreavus", level=88, types=("ghost", "typeless"), hp=249, maxhp=249,
                 ability="levitate", attack=156, defense=156, special_attack=200,
                 special_defense=200, speed=200,
                 moves=[poke_engine.Move(id="nightshade", pp=16)]),
            _mon(id="milotic", level=100, types=("water", "typeless"), hp=352, maxhp=400,
                 ability="marvelscale", attack=177, defense=215, special_attack=257,
                 special_defense=307, speed=219,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        flat = _flat(_instructions(state, "nightshade", "splash"))
        self.assertIn("Damage SideTwo: 88", flat)

    def test_seismic_toss_plain_target_single_shape(self) -> None:
        """Control (both builds): Seismic Toss into a hook-free target deals
        level and nothing else fires."""
        state = _one_on_one(
            _mon(id="dusclops", level=83, types=("ghost", "typeless"), hp=202, maxhp=202,
                 ability="pressure", attack=164, defense=263, special_attack=147,
                 special_defense=263, speed=89,
                 moves=[poke_engine.Move(id="seismictoss", pp=16)]),
            _mon(id="milotic", level=100, types=("water", "typeless"), hp=352, maxhp=400,
                 ability="owntempo", attack=177, defense=215, special_attack=257,
                 special_defense=307, speed=219,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        flat = _flat(_instructions(state, "seismictoss", "splash"))
        self.assertIn("Damage SideTwo: 83", flat)
        self.assertFalse(any("BURN" in i or "Damage SideOne" in i for i in flat))


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class CounterHiddenPowerTests(unittest.TestCase):
    def _state(self, wobb_move):
        return _one_on_one(
            _mon(id="seaking", level=83, types=("water", "typeless"), hp=268, maxhp=268,
                 ability="swiftswim", attack=200, defense=156, special_attack=155,
                 special_defense=180, speed=161,
                 moves=[poke_engine.Move(id="hiddenpowergrass70", pp=24)]),
            _mon(id="wobbuffet", level=100, types=("psychic", "typeless"), hp=542, maxhp=542,
                 ability="shadowtag", attack=123, defense=173, special_attack=123,
                 special_defense=173, speed=123,
                 moves=[poke_engine.Move(id=wobb_move, pp=16)]),
        )

    def test_counter_bounces_special_typed_hidden_power(self) -> None:
        """Sim probe (Seaking HP Grass — special-typed — into Wobbuffet
        COUNTER, seeds 1-4): Counter retaliates exactly 2x the damage taken
        (43->86, 42->84, 41->82, 40->80). Unpatched the engine recorded HP's
        type-derived Special category and Counter saw nothing."""
        branches = _instructions(self._state("counter"), "hiddenpowergrass70", "counter")
        flat = _flat(branches)
        dmg = next(int(i.rsplit(":", 1)[1]) for i in flat if i.startswith("Damage SideTwo:"))
        self.assertIn(f"Damage SideOne: {dmg * 2}", flat)

    def test_mirror_coat_never_bounces_hidden_power(self) -> None:
        """Sim probe (same attacker into MIRROR COAT, seeds 1-4): the Mirror
        Coat move line appears with NO damage. Unpatched the engine bounced
        2x (rows 1500155/22, 1500264/40)."""
        branches = _instructions(self._state("mirrorcoat"), "hiddenpowergrass70", "mirrorcoat")
        flat = _flat(branches)
        self.assertFalse(any(i.startswith("Damage SideOne") for i in flat))

    def test_mirror_coat_still_bounces_true_special(self) -> None:
        """Control (both builds; sim probe: Surf into Mirror Coat retaliates
        2x — 88->176, 86->172, 83->166)."""
        state = _one_on_one(
            _mon(id="seaking", level=83, types=("water", "typeless"), hp=268, maxhp=268,
                 ability="swiftswim", attack=200, defense=156, special_attack=156,
                 special_defense=180, speed=161,
                 moves=[poke_engine.Move(id="surf", pp=24)]),
            _mon(id="wobbuffet", level=100, types=("psychic", "typeless"), hp=542, maxhp=542,
                 ability="shadowtag", attack=123, defense=173, special_attack=123,
                 special_defense=173, speed=123,
                 moves=[poke_engine.Move(id="mirrorcoat", pp=16)]),
        )
        flat = _flat(_instructions(state, "surf", "mirrorcoat"))
        dmg = next(int(i.rsplit(":", 1)[1]) for i in flat if i.startswith("Damage SideTwo:"))
        self.assertIn(f"Damage SideOne: {dmg * 2}", flat)


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class FlashFirePhase1Tests(unittest.TestCase):
    def test_flash_fire_floors_before_the_plus_two(self) -> None:
        """Row 1500123/79 mirror (recorded state stats: Ninetales L82 spa 179
        with the FLASHFIRE volatile, Fire Blast, vs Octillery spd 180, Water
        resist). Sim pipeline: t4 81 -> Phase1 modify(81,1.5) = 121 -> +2 ->
        STAB 184 -> resist 92; the observed 78 is floor(92*0.85), the sim's
        exact min roll. Trailing-float placement gives 93. Branch damage:
        floor(92*0.925) = 85 patched, floor(93*0.925) = 86 unpatched."""
        state = _one_on_one(
            _mon(id="ninetales", level=82, types=("fire", "typeless"), hp=235, maxhp=254,
                 ability="flashfire", item="leftovers", attack=131, defense=170,
                 special_attack=179, special_defense=211, speed=211,
                 moves=[poke_engine.Move(id="fireblast", pp=8)]),
            _mon(id="octillery", level=87, types=("water", "typeless"), hp=272, maxhp=272,
                 ability="suctioncups", item="leftovers", attack=187, defense=180,
                 special_attack=232, special_defense=180, speed=128,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
            p1_side={"volatile_statuses": {"flashfire"}},
        )
        flat = _flat(_instructions(state, "fireblast", "splash"))
        self.assertTrue(any(i.startswith("Damage SideTwo: 85") for i in flat))
        self.assertFalse(any(i.startswith("Damage SideTwo: 86") for i in flat))

    def test_no_volatile_no_boost(self) -> None:
        """Control (both builds): without the volatile the same state deals
        the plain pipeline value (max 62, branch floor(62*0.925) = 57)."""
        state = _one_on_one(
            _mon(id="ninetales", level=82, types=("fire", "typeless"), hp=235, maxhp=254,
                 ability="flashfire", item="leftovers", attack=131, defense=170,
                 special_attack=179, special_defense=211, speed=211,
                 moves=[poke_engine.Move(id="fireblast", pp=8)]),
            _mon(id="octillery", level=87, types=("water", "typeless"), hp=272, maxhp=272,
                 ability="suctioncups", item="leftovers", attack=187, defense=180,
                 special_attack=232, special_defense=180, speed=128,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        flat = _flat(_instructions(state, "fireblast", "splash"))
        self.assertTrue(any(i.startswith("Damage SideTwo: 57") for i in flat))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
