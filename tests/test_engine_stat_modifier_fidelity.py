"""Regression pins for the gen3 stat-modifier-flooring and type-netting patches.

Requires the patched wheel (``scripts/setup_poke_engine.sh``). Two patches:

``poke-engine-gen3-stat-modifier-flooring.patch``
    RSE applies the item/ability stat modifiers (Choice Band, Guts, Huge/Pure
    Power, Hustle, Thick Club, Light Ball, Soul Dew, Marvel Scale, Metal
    Powder, the gen3 1.1x type items, Plus/Minus) to the boost-floored STAT
    via ``battle.modify``'s 4096-based fixed point BEFORE the base formula's
    /Defense and /50 truncations (sim/battle-actions.ts getDamage +
    data/mods/gen3/items.ts). Upstream multiplied ``choice.base_power`` in
    f32, carrying the modifier's half-point through the truncations. Also
    covered by the same sim reading: Light Ball is SpA-only in gen3, Thick
    Club Atk-only, Metal Powder x2 Def for an untransformed Ditto
    (physical-only), Explosion/Self-Destruct halve the MODIFIED defense stat
    instead of doubling BP, gen3 Plus/Minus fire across the field, and the
    upstream Guts burn-compensation double is gone (it triple-counted a
    burned Guts attacker once patch 33's Guts-guarded burn halving landed).

``poke-engine-gen3-type-effectiveness-netting.patch``
    The sim NETS the type exponent before applying it (mods/gen3/scripts.ts
    modifyDamage:88-104); patch 33's transcription applied the defender's two
    types as independent steps, flooring inside a net-neutral (2x, 0.5x) pair
    whenever the resisted type comes first in the type tuple.

Sim ground truth (gen3 Custom Game fixtures via pokezero.showdown_fixture,
2026-07-29, transcribed from live probe runs — stats below mirror the fixture
requests exactly; each divergence test's docstring carries its observed roll
set). Engine expectations are the branch-applied ``floor(0.925 * max)``
values, computed from the observed sim max via the exact pipeline in
``pokezero.gen3_damage`` (itself live-sim cross-checked in
tests/test_gen3_damage.py).
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
    p1 = pe.Pokemon(**p1_kwargs)
    p2 = pe.Pokemon(**p2_kwargs)
    side_one = pe.Side(active_index="0", pokemon=[p1] + [dummy] * 5, **(p1_side or {}))
    side_two = pe.Side(active_index="0", pokemon=[p2] + [dummy] * 5, **(p2_side or {}))
    return pe.State(
        side_one=side_one, side_two=side_two,
        weather="none", terrain="none", trick_room=False,
    )


def _damages(branches, side="SideTwo"):
    out = []
    prefix = f"Damage {side}:"
    for branch in branches:
        for ins in branch.instruction_list:
            text = str(ins)
            if text.startswith(prefix):
                out.append(int(text.rsplit(":", 1)[1]))
    return out


def _mon(**kwargs):
    base = dict(item="none", status="none")
    base.update(kwargs)
    return base


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class StatModifierFlooringTests(unittest.TestCase):
    """Divergence pins: each FAILS on a 33-patch (pre-stat-flooring) build."""

    def test_choice_band_floors_at_the_attack_stat(self) -> None:
        """Proof row 1500221/13 mirror. Sim probe (Pidgeot L87 atk 189 + Choice
        Band, Aerial Ace, vs Unown L100 def 153; seeds 1-10): observed
        {102,104,107,110,111,113,118,121} — the roll set of max 121
        (atk' = floor(189*1.5) = 283), with both the min (102) and max (121)
        hit. BP-side 1.5 gives max 123 (legal floor 104; the recorded row's
        observed 102 sat below it). Branch damage: floor(121*0.925) = 111
        patched, floor(123*0.925) = 113 unpatched."""
        state = _one_on_one(
            _mon(id="pidgeot", level=87, types=("normal", "flying"), hp=286, maxhp=286,
                 ability="keeneye", item="choiceband", attack=189, defense=180,
                 special_attack=171, special_defense=171, speed=208,
                 moves=[poke_engine.Move(id="aerialace", pp=32)]),
            _mon(id="unown", level=100, types=("psychic", "typeless"), hp=258, maxhp=258,
                 ability="levitate", attack=201, defense=153, special_attack=201,
                 special_defense=153, speed=153,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "aerialace", "splash"))
        self.assertIn(111, damages)
        self.assertNotIn(113, damages)

    def test_guts_boosted_stat_floors_at_the_stat(self) -> None:
        """Proof row 1500024/9 mirror. Sim probe (Ursaring L81 atk 257 Guts,
        Toxic'd then Growled to -1, Return, vs Slowbro L82 def 228; seeds
        1-10): observed {101,105,106,109,116,118} on the poisoned+unboosted
        turns — the roll set of max 118 (atk' = floor(floor(257/1.5)*1.5) =
        floor(171*1.5) = 256), max hit twice. BP-side gives max 120. Branch:
        floor(118*0.925) = 109 patched, floor(120*0.925) = 111 unpatched."""
        state = _one_on_one(
            _mon(id="ursaring", level=81, types=("normal", "typeless"), hp=278, maxhp=278,
                 ability="guts", item="leftovers", attack=257, defense=167,
                 special_attack=168, special_defense=167, speed=136, status="poison",
                 moves=[poke_engine.Move(id="return", pp=32)]),
            _mon(id="slowbro", level=82, types=("water", "psychic"), hp=290, maxhp=290,
                 ability="owntempo", item="leftovers", attack=170, defense=228,
                 special_attack=211, special_defense=178, speed=96,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
            p1_side={"attack_boost": -1},
        )
        damages = _damages(poke_engine.generate_instructions(state, "return", "splash"))
        self.assertIn(109, damages)
        self.assertNotIn(111, damages)

    def test_burned_guts_attacks_at_plain_1_5x(self) -> None:
        """Burned Guts is 1.5x with the burn halving skipped — identical to any
        other status. Sim probe (Machamp L80 atk 254 Guts, Will-O-Wisp'd, Rock
        Slide vs Smeargle L90 def 114; seeds 1-10): observed {158,161,166,168,
        170} inside the rolls of max 172 = the SAME max as a paralyzed Guts
        Machamp. The unpatched engine kept upstream's burn-compensation double
        on top of the Guts 1.5x after patch 33's Guts-guarded halving stopped
        halving, tripling the move (this state: max 383, branch 354).
        This pin uses the Z.2 fixture stats (atk 237 / def 95, target hp
        raised clear of the cap): patched branch floor(192*0.925) = 177 for
        BOTH burn and paralyze; unpatched burn branch 354."""
        def branches_for(status):
            state = _one_on_one(
                _mon(id="machamp", level=80, types=("fighting", "typeless"), hp=258,
                     maxhp=258, ability="guts", attack=237, defense=157,
                     special_attack=133, special_defense=165, speed=117, status=status,
                     moves=[poke_engine.Move(id="rockslide", pp=16)]),
                _mon(id="smeargle", level=90, types=("normal", "typeless"), hp=500,
                     maxhp=500, ability="owntempo", attack=68, defense=95,
                     special_attack=68, special_defense=113, speed=167,
                     moves=[poke_engine.Move(id="splash", pp=16)]),
            )
            return _damages(poke_engine.generate_instructions(state, "rockslide", "splash"))

        burn = branches_for("burn")
        paralyze = branches_for("paralyze")
        self.assertIn(177, burn)
        self.assertNotIn(354, burn)
        self.assertEqual(sorted(burn), sorted(paralyze))

    def test_lightball_is_spa_only(self) -> None:
        """gen3 Light Ball doubles SpA ONLY (mods/gen3/items.ts). Sim probe
        (Pikachu L87 atk 145 + Light Ball, Quick Attack, vs Slowbro L82 def
        228; seeds 1-10): observed {17,18,19,20} — the roll set of the
        UNBOOSTED max 20. The unpatched engine doubled the physical move's BP
        too (max 38, branch 35). Patched branch: floor(20*0.925) = 18."""
        state = _one_on_one(
            _mon(id="pikachu", level=87, types=("electric", "typeless"), hp=203, maxhp=203,
                 ability="static", item="lightball", attack=145, defense=102,
                 special_attack=137, special_defense=119, speed=206,
                 moves=[poke_engine.Move(id="quickattack", pp=48)]),
            _mon(id="slowbro", level=82, types=("water", "psychic"), hp=290, maxhp=290,
                 ability="owntempo", attack=170, defense=228, special_attack=211,
                 special_defense=178, speed=96,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "quickattack", "splash"))
        self.assertIn(18, damages)
        self.assertNotIn(35, damages)

    def test_thickclub_is_atk_only(self) -> None:
        """Thick Club doubles Atk only (gen4-mod onModifyAtk, inherited). Sim
        probe (Marowak L87 spa 137 + Thick Club, Blizzard, vs Slowbro L82 spd
        178, Ice resisted; seeds 1-10): observed {29,30,31,32} — rolls of the
        UNBOOSTED max 34. Unpatched doubled the special BP (max 67, branch
        61). Patched branch: floor(34*0.925) = 31."""
        state = _one_on_one(
            _mon(id="marowak", level=87, types=("ground", "typeless"), hp=246, maxhp=246,
                 ability="rockhead", item="thickclub", attack=189, defense=241,
                 special_attack=137, special_defense=189, speed=128,
                 moves=[poke_engine.Move(id="blizzard", pp=8)]),
            _mon(id="slowbro", level=82, types=("water", "psychic"), hp=290, maxhp=290,
                 ability="owntempo", attack=170, defense=228, special_attack=211,
                 special_defense=178, speed=96,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "blizzard", "splash"))
        self.assertIn(31, damages)
        self.assertNotIn(61, damages)

    def test_silkscarf_1_1_rounds_at_the_stat(self) -> None:
        """The gen3 1.1x type items chainModify the STAT (4505/4096,
        round-half-down at 4096 scale). Sim probe (Linoone L100 atk 197 +
        Silk Scarf, Return, vs Slowbro L76 def 211; seeds 1-10): observed
        {117,121,122,124,128,130,135} — max 135 hit, which only the stat-side
        pipeline produces (atk' = modify(197, 4505/4096) = 217; BP-side float
        1.1 gives max 133). Branch: floor(135*0.925) = 124 patched,
        floor(133*0.925) = 123 unpatched."""
        state = _one_on_one(
            _mon(id="linoone", level=100, types=("normal", "typeless"), hp=318, maxhp=318,
                 ability="pickup", item="silkscarf", attack=197, defense=179,
                 special_attack=157, special_defense=179, speed=257,
                 moves=[poke_engine.Move(id="return", pp=32)]),
            _mon(id="slowbro", level=76, types=("water", "psychic"), hp=269, maxhp=269,
                 ability="owntempo", attack=158, defense=211, special_attack=196,
                 special_defense=166, speed=90,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "return", "splash"))
        self.assertIn(124, damages)
        self.assertNotIn(123, damages)

    def test_souldew_spa_floors_at_the_stat(self) -> None:
        """Soul Dew SpA x1.5 is stat-side. Sim probe (Latias L81 spa 225 + Soul
        Dew, Psychic, vs Slowbro L82 spd 178, resisted; seeds 1-10): observed
        {74,76,79,84,85,86,87} — rolls of max 87 (spa' = floor(225*1.5) =
        337), max hit. BP-side gives max 88. Branch: floor(87*0.925) = 80
        patched, floor(88*0.925) = 81 unpatched."""
        state = _one_on_one(
            _mon(id="latias", level=81, types=("dragon", "psychic"), hp=262, maxhp=262,
                 ability="levitate", item="souldew", attack=176, defense=192,
                 special_attack=225, special_defense=257, speed=225,
                 moves=[poke_engine.Move(id="psychic", pp=16)]),
            _mon(id="slowbro", level=82, types=("water", "psychic"), hp=290, maxhp=290,
                 ability="owntempo", attack=170, defense=228, special_attack=211,
                 special_defense=178, speed=96,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "psychic", "splash"))
        self.assertIn(80, damages)
        self.assertNotIn(81, damages)

    def test_souldew_spd_floors_at_the_stat(self) -> None:
        """Soul Dew's defensive half is a ModifySpD x1.5 STAT modifier. Sim
        probe (Slowbro L100 spa 257 Surf vs Latias L83 spd 263 + Soul Dew,
        resisted; seeds 1-10): observed {34,35,36,38,39,40} — rolls of max 40
        (spd' = floor(263*1.5) = 394), max hit; BP-side /1.5 gives max 39, so
        the observed 40 sits above its whole roll set. Branch:
        floor(40*0.925) = 37 patched, floor(39*0.925) = 36 unpatched."""
        state = _one_on_one(
            _mon(id="slowbro", level=100, types=("water", "psychic"), hp=352, maxhp=352,
                 ability="owntempo", attack=207, defense=277, special_attack=257,
                 special_defense=217, speed=117,
                 moves=[poke_engine.Move(id="surf", pp=24)]),
            _mon(id="latias", level=83, types=("dragon", "psychic"), hp=268, maxhp=268,
                 ability="levitate", item="souldew", attack=180, defense=197,
                 special_attack=230, special_defense=263, speed=230,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "surf", "splash"))
        self.assertIn(37, damages)
        self.assertNotIn(36, damages)

    def test_marvelscale_floors_at_the_defense_stat(self) -> None:
        """Marvel Scale is ModifyDef x1.5 while statused. Sim probe (Snorlax
        L72 atk 200 Return vs toxic'd Milotic L100 def 215; seeds 1-12):
        observed {53,54,55,56,58,59} — inside the rolls of max 60
        (def' = floor(215*1.5) = 322); the BP-side /1.5 build has max 58, so
        the observed 59 sits above its whole roll set. Branch:
        floor(60*0.925) = 55 patched, floor(58*0.925) = 53 unpatched — both
        legal sim rolls, but only one is the engine's branch value."""
        state = _one_on_one(
            _mon(id="snorlax", level=72, types=("normal", "typeless"), hp=349, maxhp=349,
                 ability="immunity", attack=200, defense=136, special_attack=136,
                 special_defense=200, speed=85,
                 moves=[poke_engine.Move(id="return", pp=32)]),
            _mon(id="milotic", level=100, types=("water", "typeless"), hp=352, maxhp=352,
                 ability="marvelscale", attack=177, defense=215, special_attack=257,
                 special_defense=307, speed=219, status="toxic",
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "return", "splash"))
        self.assertIn(55, damages)
        self.assertNotIn(53, damages)

    def test_metalpowder_is_def_x2_untransformed(self) -> None:
        """Metal Powder is x2 Defense for an UNTRANSFORMED Ditto, physical
        only (base data/items.ts through the unmodified gen3 chain). Upstream
        divided BP by 1.5 — and behind a guard that checked the ATTACKER for
        being Ditto, so against a non-Ditto attacker it silently did nothing.
        Pool-unreachable in randbats (Ditto's item table has no Metal Powder),
        pinned like Fake Out rather than left to be rediscovered; expectations
        from the sim-exact pipeline in pokezero.gen3_damage (oracle-derived).
        Snorlax L100 atk 277 STAB Return vs Metal Powder Ditto def 153:
        def' = 306 -> max 118, branch floor(118*0.925) = 109; unpatched
        (no-op arm) max 235, branch 217."""
        state = _one_on_one(
            _mon(id="snorlax", level=100, types=("normal", "typeless"), hp=482, maxhp=482,
                 ability="immunity", attack=277, defense=187, special_attack=187,
                 special_defense=277, speed=117,
                 moves=[poke_engine.Move(id="return", pp=32)]),
            _mon(id="ditto", level=100, types=("normal", "typeless"), hp=500, maxhp=500,
                 ability="limber", item="metalpowder", attack=159, defense=153,
                 special_attack=153, special_defense=153, speed=153,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "return", "splash"))
        self.assertIn(109, damages)
        self.assertNotIn(217, damages)

    def test_explosion_halves_the_defense_stat(self) -> None:
        """Explosion halves the MODIFIED defense stat with a floor
        (clampIntRange(floor(def/2), 1) — sim/battle-actions.ts, gen<=4),
        not the base power. Sim anchor: Machamp L80 atk 254 Explosion vs
        Skarmory L100 def 337 (odd), observed {112,...,129} = exactly the
        rolls of max 129 = floor(halved-def pipeline / Steel resist); the
        no-halving reading gives max 65. This pin discriminates the
        RELOCATION on an odd neutral defense (def 215, oracle-derived):
        floor(def/2) = 107 -> max 405, branch floor(405*0.925) = 374;
        BP-doubling gives max 403, branch 372."""
        state = _one_on_one(
            _mon(id="machamp", level=80, types=("fighting", "typeless"), hp=275, maxhp=275,
                 ability="guts", attack=254, defense=174, special_attack=150,
                 special_defense=182, speed=134,
                 moves=[poke_engine.Move(id="explosion", pp=8)]),
            _mon(id="milotic", level=100, types=("water", "typeless"), hp=600, maxhp=600,
                 ability="owntempo", attack=177, defense=215, special_attack=257,
                 special_defense=307, speed=219,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "explosion", "splash"))
        self.assertIn(374, damages)
        self.assertNotIn(372, damages)

    def test_plus_minus_fire_across_the_field(self) -> None:
        """gen3 Plus/Minus scan ALL actives (mods/gen3/abilities.ts uses
        getAllActive), so the boost fires in singles against the partner
        ability. Sim probe (Minun L100 spa 207 Thunderbolt vs Plusle L100 spd
        207, Electric resisted; seeds 1-10): observed {78,81,82,84,86,88,90}
        — rolls of max 90 (spa' = floor(207*1.5) = 310), max hit; unboosted
        max is 60 (control probe vs Slowbro matched the unboosted oracle).
        Branch: floor(90*0.925) = 83 patched; the unpatched engine has no
        Plus/Minus at all (max 60, branch 55)."""
        state = _one_on_one(
            _mon(id="minun", level=100, types=("electric", "typeless"), hp=282, maxhp=282,
                 ability="minus", attack=137, defense=157, special_attack=207,
                 special_defense=227, speed=247,
                 moves=[poke_engine.Move(id="thunderbolt", pp=24)]),
            _mon(id="plusle", level=100, types=("electric", "typeless"), hp=282, maxhp=282,
                 ability="plus", attack=157, defense=137, special_attack=227,
                 special_defense=207, speed=247,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "thunderbolt", "splash"))
        self.assertIn(83, damages)
        self.assertNotIn(55, damages)

    def test_hustle_modifies_the_stat_directly_then_chains(self) -> None:
        """Hustle is the one DIRECT stat modify ("applied directly to the stat
        as opposed to chaining with the others" — data/abilities.ts), so
        Hustle + Choice Band truncates twice. Sim probe (Togetic L100 atk 137
        Hustle + Choice Band, Return, vs Slowbro L76 def 211; seeds 1-14):
        observed {164,168,171,173,179,183,185} — 183 and 185 are rolls of the
        double-truncated max 189 (floor(floor(137*1.5)*1.5) = floor(205*1.5)
        = 307) and sit OUTSIDE the BP-side x2.25 build's roll set (max 190).
        Branch: floor(189*0.925) = 174 patched, floor(190*0.925) = 175
        unpatched."""
        state = _one_on_one(
            _mon(id="togetic", level=100, types=("normal", "flying"), hp=272, maxhp=272,
                 ability="hustle", item="choiceband", attack=137, defense=227,
                 special_attack=217, special_defense=267, speed=137,
                 moves=[poke_engine.Move(id="return", pp=32)]),
            _mon(id="slowbro", level=76, types=("water", "psychic"), hp=269, maxhp=269,
                 ability="owntempo", attack=158, defense=211, special_attack=196,
                 special_defense=166, speed=90,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "return", "splash"))
        self.assertIn(174, damages)
        self.assertNotIn(175, damages)


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class TypeEffectivenessNettingTests(unittest.TestCase):
    """Divergence pin for the netting patch (fails on a build without it)."""

    def test_net_neutral_pair_applies_no_steps(self) -> None:
        """Proof row 1500076/96 mirror. Sim probe (Arcanine L78 spa 233 HP
        Grass 70 vs Shuckle L98 spd 506, BUG/ROCK; seeds 1-10): observed
        {20,21,22,23} — rolls of max 23 with the max hit: the sim nets
        typeMod to 0 and applies NO type steps. Per-type application floors
        the odd 23 at the leading Bug resist (floor(23/2)*2 = 22). Branch:
        floor(23*0.925) = 21 patched, floor(22*0.925) = 20 unpatched."""
        state = _one_on_one(
            _mon(id="arcanine", level=78, types=("fire", "typeless"), hp=268, maxhp=268,
                 ability="intimidate", attack=216, defense=170, special_attack=233,
                 special_defense=170, speed=193,
                 moves=[poke_engine.Move(id="hiddenpowergrass70", pp=24)]),
            _mon(id="shuckle", level=98, types=("bug", "rock"), hp=198, maxhp=198,
                 ability="sturdy", attack=75, defense=506, special_attack=75,
                 special_defense=506, speed=65,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(
            poke_engine.generate_instructions(state, "hiddenpowergrass70", "splash"))
        self.assertIn(21, damages)
        self.assertNotIn(20, damages)


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class StatModifierControlTests(unittest.TestCase):
    """Controls: identical on patched and unpatched builds."""

    def test_no_item_attack_is_untouched(self) -> None:
        """The Choice Band pin's state minus the item: max 82, branch
        floor(82*0.925) = 75 on BOTH builds (no modifier applies)."""
        state = _one_on_one(
            _mon(id="pidgeot", level=87, types=("normal", "flying"), hp=286, maxhp=286,
                 ability="keeneye", attack=189, defense=180, special_attack=171,
                 special_defense=171, speed=208,
                 moves=[poke_engine.Move(id="aerialace", pp=32)]),
            _mon(id="unown", level=100, types=("psychic", "typeless"), hp=258, maxhp=258,
                 ability="levitate", attack=201, defense=153, special_attack=201,
                 special_defense=153, speed=153,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "aerialace", "splash"))
        self.assertIn(75, damages)

    def test_lightball_special_x2_is_locus_invariant(self) -> None:
        """An exact x2 lands identically whether applied to BP or the stat
        (2*BP*A == BP*2A with no fraction to floor): Light Ball Pikachu
        Thunderbolt vs Slowbro — max 320, branch floor(320*0.925) = 296 on
        BOTH builds. Guards against over-reach: the patch must not change the
        already-correct special doubling."""
        state = _one_on_one(
            _mon(id="pikachu", level=87, types=("electric", "typeless"), hp=203, maxhp=203,
                 ability="static", item="lightball", attack=145, defense=102,
                 special_attack=137, special_defense=119, speed=206,
                 moves=[poke_engine.Move(id="thunderbolt", pp=24)]),
            _mon(id="slowbro", level=82, types=("water", "psychic"), hp=400, maxhp=400,
                 ability="owntempo", attack=170, defense=228, special_attack=211,
                 special_defense=178, speed=96,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "thunderbolt", "splash"))
        self.assertIn(296, damages)

    def test_marvelscale_unstatused_is_untouched(self) -> None:
        """Marvel Scale without a status: plain defense on both builds.
        Snorlax L72 atk 200 Return vs healthy Milotic L100 def 215:
        max 87 (oracle-exact), branch floor(87*0.925) = 80."""
        state = _one_on_one(
            _mon(id="snorlax", level=72, types=("normal", "typeless"), hp=349, maxhp=349,
                 ability="immunity", attack=200, defense=136, special_attack=136,
                 special_defense=200, speed=85,
                 moves=[poke_engine.Move(id="return", pp=32)]),
            _mon(id="milotic", level=100, types=("water", "typeless"), hp=352, maxhp=352,
                 ability="marvelscale", attack=177, defense=215, special_attack=257,
                 special_defense=307, speed=219,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "return", "splash"))
        self.assertIn(80, damages)

    def test_minus_without_plus_is_unboosted(self) -> None:
        """Plus/Minus need the partner ability on the field. Sim control probe
        (Minun Thunderbolt vs Slowbro L82; seeds 1-10): observed
        {245,...,282} — exactly the UNBOOSTED oracle rolls (max 282, hit).
        Branch: floor(282*0.925) = 260 on the patched build; the unpatched
        build never boosts anyway."""
        state = _one_on_one(
            _mon(id="minun", level=100, types=("electric", "typeless"), hp=282, maxhp=282,
                 ability="minus", attack=137, defense=157, special_attack=207,
                 special_defense=227, speed=247,
                 moves=[poke_engine.Move(id="thunderbolt", pp=24)]),
            _mon(id="slowbro", level=82, types=("water", "psychic"), hp=400, maxhp=400,
                 ability="owntempo", attack=170, defense=228, special_attack=211,
                 special_defense=178, speed=96,
                 moves=[poke_engine.Move(id="splash", pp=16)]),
        )
        damages = _damages(poke_engine.generate_instructions(state, "thunderbolt", "splash"))
        self.assertIn(260, damages)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
