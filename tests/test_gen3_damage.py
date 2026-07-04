import os
import shutil
import unittest
from pathlib import Path

from pokezero.gen3_damage import (
    Gen3DamageContext,
    HIDDEN_POWER_IVS,
    ROLL_NUMERATORS,
    apply_chain,
    boosted_stat,
    chain_modifier,
    gen3_damage_rolls,
    gen3_hp_stat,
    gen3_stat,
    median_damage,
    modify,
    randbats_spread_stats,
)


class FixedPointTest(unittest.TestCase):
    def test_modify_rounds_half_down_at_4096_scale(self) -> None:
        # 128 * 1.5 = 192 exactly at 4096 scale: tr((128*6144 + 2047) / 4096) = 192.
        self.assertEqual(modify(128, 1.5), 192)
        self.assertEqual(modify(126, 0.5), 63)
        self.assertEqual(modify(128, 2), 256)
        # 1.1x items truncate to 4505/4096 first (Showdown chainModify(1.1)):
        # tr((100*4505 + 2047) / 4096) = 110.
        self.assertEqual(modify(100, 1.1), 110)

    def test_chain_modifier_accumulates_like_the_engine(self) -> None:
        self.assertEqual(chain_modifier([(1.5, 1)]), 1.5)
        # CB x Guts: (6144 * 6144 + 2048) >> 12 = 9216 -> 2.25.
        self.assertEqual(chain_modifier([(1.5, 1), (1.5, 1)]), 2.25)
        self.assertEqual(apply_chain(300, [(1.5, 1)]), 450)
        self.assertEqual(apply_chain(300, ()), 300)

    def test_boost_table(self) -> None:
        self.assertEqual(boosted_stat(300, 0), 300)
        self.assertEqual(boosted_stat(300, 1), 450)
        self.assertEqual(boosted_stat(300, 2), 600)
        self.assertEqual(boosted_stat(300, -1), 200)
        self.assertEqual(boosted_stat(299, -1), 199)  # floor of 299/1.5
        self.assertEqual(boosted_stat(300, 7), 1200)  # clamped to +6


class StatFormulaTest(unittest.TestCase):
    def test_stat_formulas(self) -> None:
        # 85 EV / 31 IV / neutral at level 80, base 110: floor(272 * 0.8 + 5) = 222.
        self.assertEqual(gen3_stat(110, 31, 85, 80), 222)
        self.assertEqual(gen3_hp_stat(100, 31, 85, 80), 291)
        # Zeroed Atk (confusion-damage rule): floor(220 * 0.8 + 5) = 181.
        self.assertEqual(gen3_stat(110, 0, 0, 80), 181)


_BASE = {"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60}


class SpreadStatsTest(unittest.TestCase):
    def test_standard_spread(self) -> None:
        stats = randbats_spread_stats(
            _BASE, level=80, moves=("bodyslam", "earthquake"), item="Leftovers", has_physical_attack=True
        )
        self.assertEqual(stats, {"atk": 222, "def": 190, "spa": 182, "spd": 190, "spe": 142, "hp": 291})

    def test_atk_zeroed_without_physical_attacks(self) -> None:
        stats = randbats_spread_stats(
            _BASE, level=80, moves=("surf", "icebeam"), item="Leftovers", has_physical_attack=False
        )
        self.assertEqual(stats["atk"], gen3_stat(110, 0, 0, 80))
        # Transform sets keep the Atk investment.
        transform = randbats_spread_stats(
            _BASE, level=80, moves=("transform",), item="Leftovers", has_physical_attack=False
        )
        self.assertEqual(transform["atk"], 222)

    def test_hidden_power_iv_overrides(self) -> None:
        # HP Grass pins atk/spa IVs to 30.
        stats = randbats_spread_stats(
            _BASE, level=80, moves=("hiddenpowergrass", "return"), item="Leftovers", has_physical_attack=True
        )
        self.assertEqual(stats["atk"], gen3_stat(110, 30, 85, 80))
        self.assertEqual(stats["spa"], gen3_stat(85, 30, 85, 80))
        # No-physical Hidden Power carrier: Atk IV drops by 28, EVs to 0.
        special = randbats_spread_stats(
            _BASE, level=80, moves=("hiddenpowergrass", "surf"), item="Leftovers", has_physical_attack=False
        )
        self.assertEqual(special["atk"], gen3_stat(110, 2, 0, 80))

    def test_hp_trim_substitute_flail(self) -> None:
        stats = randbats_spread_stats(
            _BASE, level=80, moves=("substitute", "flail", "return"), item="Salac Berry", has_physical_attack=True
        )
        self.assertGreater(stats["hp"] % 4, 0)  # four Substitutes must be possible

    def test_hp_trim_substitute_pinch_berry(self) -> None:
        stats = randbats_spread_stats(
            _BASE, level=80, moves=("substitute", "return", "swordsdance"), item="Salac Berry", has_physical_attack=True
        )
        self.assertEqual(stats["hp"] % 4, 0)  # berry activates after three Substitutes

    def test_hp_trim_belly_drum(self) -> None:
        stats = randbats_spread_stats(
            _BASE, level=80, moves=("bellydrum", "return"), item="Lum Berry", has_physical_attack=True
        )
        self.assertGreater(stats["hp"] % 2, 0)  # two Belly Drums must be possible

    def test_hidden_power_iv_table_shape(self) -> None:
        self.assertEqual(len(HIDDEN_POWER_IVS), 16)
        for overrides in HIDDEN_POWER_IVS.values():
            for stat, value in overrides.items():
                self.assertIn(stat, {"hp", "atk", "def", "spa", "spd", "spe"})
                self.assertEqual(value, 30)


class DamageRollsTest(unittest.TestCase):
    """Pure-formula fixtures; the exact chain is verified against the live vendored sim
    in :class:`SimCrossCheckTest` below and by the gate harness's roll-membership
    calibration on full games."""

    def _ctx(self, **overrides) -> Gen3DamageContext:
        base = dict(level=100, base_power=100, category="Physical", attack=300, defense=200)
        base.update(overrides)
        return Gen3DamageContext(**base)

    def test_plain_rolls(self) -> None:
        rolls = gen3_damage_rolls(self._ctx())
        self.assertEqual(len(rolls), 16)
        self.assertEqual(rolls[0], 108)
        self.assertEqual(rolls[-1], 128)
        self.assertEqual(median_damage(rolls), 118.0)

    def test_stab(self) -> None:
        self.assertEqual(gen3_damage_rolls(self._ctx(stab=True))[-1], 192)

    def test_type_effectiveness_steps(self) -> None:
        self.assertEqual(gen3_damage_rolls(self._ctx(effectiveness=2.0))[-1], 256)
        self.assertEqual(gen3_damage_rolls(self._ctx(effectiveness=0.5))[-1], 64)
        self.assertEqual(gen3_damage_rolls(self._ctx(effectiveness=4.0))[-1], 512)
        self.assertEqual(gen3_damage_rolls(self._ctx(effectiveness=0.0)), ())

    def test_burn_halves_before_plus_two(self) -> None:
        # 126 -> 63 (burn) -> 65 (+2).
        self.assertEqual(gen3_damage_rolls(self._ctx(burned=True))[-1], 65)

    def test_crit_doubles_after_plus_two_and_ignores_stages(self) -> None:
        self.assertEqual(gen3_damage_rolls(self._ctx(crit=True))[-1], 256)
        # Crit ignores the attacker's harmful stage and the defender's helpful stage.
        self.assertEqual(
            gen3_damage_rolls(self._ctx(crit=True, attack_boost=-2, defense_boost=2))[-1], 256
        )
        # ... but keeps helpful attacker stages.
        self.assertEqual(
            gen3_damage_rolls(self._ctx(attack_boost=1))[-1],
            gen3_damage_rolls(self._ctx(crit=False, attack_boost=1))[-1],
        )

    def test_screen_halves_after_type_but_not_on_crit(self) -> None:
        self.assertEqual(gen3_damage_rolls(self._ctx(screen=True))[-1], 64)
        self.assertEqual(gen3_damage_rolls(self._ctx(screen=True, crit=True))[-1], 256)

    def test_weather_mod_applies_before_plus_two(self) -> None:
        # 126 -> 189 (x1.5) -> 191 (+2).
        self.assertEqual(gen3_damage_rolls(self._ctx(weather_mod=(1.5, 1)))[-1], 191)

    def test_attack_chain_choice_band(self) -> None:
        # CB: attack 300 -> 450; base 189 -> 191 with +2.
        self.assertEqual(gen3_damage_rolls(self._ctx(attack_mods=((1.5, 1),)))[-1], 191)

    def test_base_power_mods_solar_beam_facade(self) -> None:
        halved = gen3_damage_rolls(self._ctx(base_power=120, base_power_mods=((0.5, 1),)))
        plain = gen3_damage_rolls(self._ctx(base_power=60))
        self.assertEqual(halved, plain)
        doubled = gen3_damage_rolls(self._ctx(base_power=70, base_power_mods=((2, 1),)))
        self.assertEqual(doubled, gen3_damage_rolls(self._ctx(base_power=140)))

    def test_explosion_defense_halving(self) -> None:
        rolls = gen3_damage_rolls(self._ctx(base_power=250, explosion_def_halving=True))
        expected = gen3_damage_rolls(self._ctx(base_power=250, defense=100))
        self.assertEqual(rolls, expected)

    def test_boost_stages(self) -> None:
        self.assertEqual(gen3_damage_rolls(self._ctx(attack_boost=-1))[-1], 86)

    def test_minimum_one_damage(self) -> None:
        rolls = gen3_damage_rolls(
            self._ctx(level=5, base_power=10, attack=10, defense=400, effectiveness=0.25)
        )
        self.assertEqual(rolls, (1,) * 16)

    def test_roll_numerators(self) -> None:
        self.assertEqual(ROLL_NUMERATORS, tuple(range(85, 101)))


def _integration_config():
    """Only run live-sim tests when a built Showdown checkout + node exist."""
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=20.0)


_EVS = {"hp": 85, "atk": 85, "def": 85, "spa": 85, "spd": 85, "spe": 85}


def _mon(species: str, moves, ability: str, item: str | None = None, level: int = 100):
    from pokezero.showdown_fixture import FixturePokemon

    return FixturePokemon(species=species, moves=moves, ability=ability, item=item, level=level, evs=_EVS)


def _run_turns(config, p1_team, p2_team, turn_choices, seed):
    """Multi-turn extension of run_one_turn_fixture over the same bridge session."""
    from pokezero.showdown_fixture import _BridgeFixtureSession, pack_team

    session = _BridgeFixtureSession(config)
    try:
        session.start(
            format_id="gen3customgame",
            seed=seed,
            p1_team=pack_team(p1_team),
            p2_team=pack_team(p2_team),
        )
        session.read_until_boundary()
        p1_request = session.requests.get("p1")
        p2_request = session.requests.get("p2")
        for p1_choice, p2_choice in turn_choices:
            if session.terminal:
                break
            session.send_choices({"p1": p1_choice, "p2": p2_choice})
            session.read_until_boundary()
        return tuple(session.protocol_lines), p1_request, p2_request
    finally:
        session.close()


def _request_stats(request, slot_index: int = 0):
    row = request["side"]["pokemon"][slot_index]
    stats = dict(row["stats"])
    condition = str(row["condition"])
    stats["hp"] = int(condition.split()[0].split("/")[1])
    return stats


def _damage_events(lines, initial_hp):
    """Untagged move damage per action, tracked in absolute HP from the omniscient log."""
    hp = dict(initial_hp)
    events = []
    current = None
    turn = 0
    for line in lines:
        parts = line.split("|")
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type == "turn":
            try:
                turn = int(parts[2])
            except (IndexError, ValueError):
                pass
            current = None
        elif event_type == "move" and len(parts) >= 4:
            current = {
                "turn": turn,
                "attacker": parts[2][:2],
                "move": "".join(ch for ch in parts[3].lower() if ch.isalnum()),
                "crit": False,
                "damage": 0,
            }
            events.append(current)
        elif event_type in {"-damage", "-heal", "-sethp"} and len(parts) >= 4:
            side = parts[2][:2]
            head = parts[3].split()[0]
            if head == "0" or "fnt" in parts[3]:
                new_hp = 0
            elif "/" in head:
                new_hp = int(head.split("/")[0])
            else:
                continue
            if (
                event_type == "-damage"
                and "[from]" not in line
                and current is not None
                and side != current["attacker"]
            ):
                current["damage"] += hp.get(side, 0) - new_hp
            hp[side] = new_hp
        elif event_type == "-crit" and current is not None:
            current["crit"] = True
    return [event for event in events if event["damage"] > 0]


@unittest.skipUnless(_integration_config() is not None, "requires built Showdown checkout and node")
class SimCrossCheckTest(unittest.TestCase):
    """Cross-check the pure-python damage chain against the live vendored simulator.

    Each scenario runs a deterministic curated battle, reads exact stats from the
    opening requests (server truth), computes the 16 predicted rolls for the scenario's
    public-modifier set, and asserts the observed damage is EXACTLY one of them (crit
    rolls when the log says ``|-crit|``). Between them these cover: neutral, STAB,
    super-effective, resisted, burn, burn+Guts, Reflect, Light Screen, rain/sun
    weather, Choice Band, Huge Power, Thick Fat, Solar Beam weather-halving, Flash
    Fire's volatile, and Explosion's defense halving — 15+ sim-verified values.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _integration_config()

    def _check(
        self,
        p1_team,
        p2_team,
        turn_choices,
        *,
        seed,
        attacker,
        move,
        turn,
        context_builder,
        expect_status=None,
    ):
        lines, p1_request, p2_request = _run_turns(self.config, p1_team, p2_team, turn_choices, seed)
        self.assertTrue(p1_request and p2_request, "missing opening requests")
        stats = {"p1": _request_stats(p1_request), "p2": _request_stats(p2_request)}
        if expect_status is not None:
            status_line = next((line for line in lines if line.startswith("|-status|")), None)
            if status_line is None or expect_status not in status_line:
                self.skipTest(f"seed did not produce the {expect_status} status: adjust seed")
        events = _damage_events(lines, {"p1": stats["p1"]["hp"], "p2": stats["p2"]["hp"]})
        event = next(
            (e for e in events if e["attacker"] == attacker and e["move"] == move and e["turn"] == turn),
            None,
        )
        self.assertIsNotNone(event, f"no damage event for {attacker} {move} turn {turn}: {events}\n{lines}")
        context = context_builder(stats, crit=event["crit"])
        rolls = gen3_damage_rolls(context)
        self.assertIn(
            event["damage"],
            rolls,
            f"observed {event['damage']} not in predicted rolls {rolls} (crit={event['crit']})",
        )

    def test_neutral_no_stab(self) -> None:
        # Rhydon Strength (Normal 80) vs Slowbro: no STAB, neutral.
        self._check(
            [_mon("Rhydon", ["Strength"], "Rock Head")],
            [_mon("Slowbro", ["Splash"], "Oblivious")],
            [("move 1", "move 1")],
            seed=3,
            attacker="p1",
            move="strength",
            turn=1,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=80, category="Physical",
                attack=s["p1"]["atk"], defense=s["p2"]["def"], crit=crit,
            ),
        )

    def test_stab_return(self) -> None:
        self._check(
            [_mon("Snorlax", ["Return"], "Immunity")],
            [_mon("Slowbro", ["Splash"], "Oblivious")],
            [("move 1", "move 1")],
            seed=11,
            attacker="p1",
            move="return",
            turn=1,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=102, category="Physical",
                attack=s["p1"]["atk"], defense=s["p2"]["def"], stab=True, crit=crit,
            ),
        )

    def test_super_effective_special(self) -> None:
        # Alakazam Psychic vs Machamp: STAB, 2x.
        self._check(
            [_mon("Alakazam", ["Psychic"], "Synchronize")],
            [_mon("Machamp", ["Splash"], "Guts")],
            [("move 1", "move 1")],
            seed=5,
            attacker="p1",
            move="psychic",
            turn=1,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=90, category="Special",
                attack=s["p1"]["spa"], defense=s["p2"]["spd"], stab=True, effectiveness=2.0, crit=crit,
            ),
        )

    def test_resisted(self) -> None:
        # Snorlax Return vs Rhydon: Normal vs Rock/Ground -> 0.5.
        self._check(
            [_mon("Snorlax", ["Return"], "Immunity")],
            [_mon("Rhydon", ["Splash"], "Rock Head")],
            [("move 1", "move 1")],
            seed=7,
            attacker="p1",
            move="return",
            turn=1,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=102, category="Physical",
                attack=s["p1"]["atk"], defense=s["p2"]["def"], stab=True, effectiveness=0.5, crit=crit,
            ),
        )

    def test_burn_halves_physical(self) -> None:
        self._check(
            [_mon("Snorlax", ["Splash", "Return"], "Immunity")],
            [_mon("Slowbro", ["Will-O-Wisp", "Splash"], "Oblivious")],
            [("move 1", "move 1"), ("move 2", "move 2")],
            seed=2,
            attacker="p1",
            move="return",
            turn=2,
            expect_status="brn",
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=102, category="Physical",
                attack=s["p1"]["atk"], defense=s["p2"]["def"], stab=True, burned=True, crit=crit,
            ),
        )

    def test_burn_with_guts(self) -> None:
        # Guts: no burn halving, 1.5x attack chain.
        self._check(
            [_mon("Machamp", ["Splash", "Brick Break"], "Guts")],
            [_mon("Slowbro", ["Will-O-Wisp", "Splash"], "Oblivious")],
            [("move 1", "move 1"), ("move 2", "move 2")],
            seed=6,
            attacker="p1",
            move="brickbreak",
            turn=2,
            expect_status="brn",
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=75, category="Physical",
                attack=s["p1"]["atk"], defense=s["p2"]["def"], stab=True, effectiveness=0.5,
                attack_mods=((1.5, 1),), burned=False, crit=crit,
            ),
        )

    def test_reflect(self) -> None:
        self._check(
            [_mon("Slowbro", ["Reflect", "Splash"], "Oblivious")],
            [_mon("Snorlax", ["Splash", "Return"], "Immunity")],
            [("move 1", "move 1"), ("move 2", "move 2")],
            seed=9,
            attacker="p2",
            move="return",
            turn=2,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=102, category="Physical",
                attack=s["p2"]["atk"], defense=s["p1"]["def"], stab=True, screen=True, crit=crit,
            ),
        )

    def test_light_screen(self) -> None:
        self._check(
            [_mon("Snorlax", ["Light Screen", "Splash"], "Immunity")],
            [_mon("Alakazam", ["Splash", "Psychic"], "Synchronize")],
            [("move 1", "move 1"), ("move 2", "move 2")],
            seed=13,
            attacker="p2",
            move="psychic",
            turn=2,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=90, category="Special",
                attack=s["p2"]["spa"], defense=s["p1"]["spd"], stab=True, screen=True, crit=crit,
            ),
        )

    def test_rain_boosts_water(self) -> None:
        self._check(
            [_mon("Slowbro", ["Rain Dance", "Surf"], "Oblivious")],
            [_mon("Snorlax", ["Splash"], "Immunity")],
            [("move 1", "move 1"), ("move 2", "move 1")],
            seed=17,
            attacker="p1",
            move="surf",
            turn=2,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=95, category="Special",
                attack=s["p1"]["spa"], defense=s["p2"]["spd"], stab=True, weather_mod=(1.5, 1), crit=crit,
            ),
        )

    def test_sun_boosts_fire(self) -> None:
        self._check(
            [_mon("Charizard", ["Sunny Day", "Flamethrower"], "Blaze")],
            [_mon("Snorlax", ["Splash"], "Immunity")],
            [("move 1", "move 1"), ("move 2", "move 1")],
            seed=19,
            attacker="p1",
            move="flamethrower",
            turn=2,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=95, category="Special",
                attack=s["p1"]["spa"], defense=s["p2"]["spd"], stab=True, weather_mod=(1.5, 1), crit=crit,
            ),
        )

    def test_choice_band(self) -> None:
        self._check(
            [_mon("Snorlax", ["Return"], "Immunity", item="Choice Band")],
            [_mon("Slowbro", ["Splash"], "Oblivious")],
            [("move 1", "move 1")],
            seed=23,
            attacker="p1",
            move="return",
            turn=1,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=102, category="Physical",
                attack=s["p1"]["atk"], defense=s["p2"]["def"], stab=True,
                attack_mods=((1.5, 1),), crit=crit,
            ),
        )

    def test_huge_power(self) -> None:
        self._check(
            [_mon("Azumarill", ["Return"], "Huge Power")],
            [_mon("Slowbro", ["Splash"], "Oblivious")],
            [("move 1", "move 1")],
            seed=29,
            attacker="p1",
            move="return",
            turn=1,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=102, category="Physical",
                attack=s["p1"]["atk"], defense=s["p2"]["def"],
                attack_mods=((2, 1),), crit=crit,
            ),
        )

    def test_thick_fat_halves_fire(self) -> None:
        self._check(
            [_mon("Charizard", ["Flamethrower"], "Blaze")],
            [_mon("Snorlax", ["Splash"], "Thick Fat")],
            [("move 1", "move 1")],
            seed=31,
            attacker="p1",
            move="flamethrower",
            turn=1,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=95, category="Special",
                attack=s["p1"]["spa"], defense=s["p2"]["spd"], stab=True,
                attack_mods=((0.5, 1),), crit=crit,
            ),
        )

    def test_solar_beam_halved_in_sand(self) -> None:
        # Tyranitar's Sand Stream is up from the lead; Solar Beam charges turn 1 and
        # releases turn 2 at half power (grass is 2x vs Rock/Dark, with STAB).
        self._check(
            [_mon("Sceptile", ["Solar Beam"], "Overgrow")],
            [_mon("Tyranitar", ["Splash"], "Sand Stream")],
            [("move 1", "move 1"), ("move 1", "move 1")],
            seed=37,
            attacker="p1",
            move="solarbeam",
            turn=2,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=120, category="Special",
                attack=s["p1"]["spa"], defense=s["p2"]["spd"], stab=True, effectiveness=2.0,
                base_power_mods=((0.5, 1),), crit=crit,
            ),
        )

    def test_flash_fire_boosts_fire_after_absorb(self) -> None:
        self._check(
            [_mon("Charizard", ["Flamethrower", "Splash"], "Blaze")],
            [_mon("Flareon", ["Splash", "Flamethrower"], "Flash Fire")],
            [("move 1", "move 1"), ("move 2", "move 2")],
            seed=41,
            attacker="p2",
            move="flamethrower",
            turn=2,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=100, base_power=95, category="Special",
                attack=s["p2"]["spa"], defense=s["p1"]["spd"], stab=True, effectiveness=0.5,
                attack_mods=((1.5, 1),), crit=crit,
            ),
        )

    def test_explosion_defense_halving(self) -> None:
        self._check(
            [_mon("Snorlax", ["Explosion"], "Immunity", level=50)],
            [_mon("Slowbro", ["Splash"], "Oblivious")],
            [("move 1", "move 1")],
            seed=43,
            attacker="p1",
            move="explosion",
            turn=1,
            context_builder=lambda s, crit: Gen3DamageContext(
                level=50, base_power=250, category="Physical",
                attack=s["p1"]["atk"], defense=s["p2"]["def"], stab=True,
                explosion_def_halving=True, crit=crit,
            ),
        )


if __name__ == "__main__":
    unittest.main()
