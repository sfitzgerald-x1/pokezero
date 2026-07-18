"""Focused tests for the trait-extraction protocol parser (scripts/trait_extract.py).

The extractor is a standalone script (not a package module); we importlib-load it and drive the
stateful GameParse over hand-built protocol snippets so the gen3-specific classification rules —
conditional rapid-spin, phazing-justified, immunity switch-in, sleep-vs-yawn, solar-beam-by-sun,
PP-exhaustion dedup — fail loudly on regression.
"""
import gzip
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trait_extract.py"


def _load():
    spec = importlib.util.spec_from_file_location("trait_extract", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TE = _load()


def parse(protocol, movesets=None):
    gp = TE.GameParse(movesets or {})
    gp.walk(protocol)
    return gp


class MoveClassification(unittest.TestCase):
    def test_toxic_and_sleep_not_yawn(self):
        gp = parse([
            "|turn|1",
            "|move|p1a: Skarmory|Toxic|p2a: Blissey",
            "|move|p1a: Skarmory|Yawn|p2a: Blissey",
            "|move|p1a: Skarmory|Spore|p2a: Blissey",
        ])
        self.assertEqual(gp.ev["p1"]["cat_toxic"], 1)
        self.assertEqual(gp.ev["p1"]["cat_yawn"], 1)
        self.assertEqual(gp.ev["p1"]["cat_sleep"], 1)  # Spore counts, Yawn does not

    def test_phaze_justified_vs_neutral(self):
        # Roar into a boosted opponent = justified; into a clean opponent = neutral.
        gp = parse([
            "|turn|1",
            "|-boost|p2a: Salamence|atk|1",
            "|move|p1a: Skarmory|Roar|p2a: Salamence",
            "|turn|2",
            "|switch|p2a: Blissey|Blissey|300/300",
            "|move|p1a: Skarmory|Whirlwind|p2a: Blissey",
        ])
        self.assertEqual(gp.ev["p1"]["cat_phaze"], 2)
        self.assertEqual(gp.ev["p1"]["cat_phaze_justified"], 1)
        self.assertEqual(gp.ev["p1"]["cat_phaze_neutral"], 1)

    def test_rapid_spin_only_counts_spikesdown(self):
        gp = parse([
            "|turn|1",
            "|move|p1a: Starmie|Rapid Spin|p2a: X",          # no spikes on p1 side
            "|turn|2",
            "|-sidestart|p1: PokeZero|Spikes",
            "|move|p1a: Starmie|Rapid Spin|p2a: X",          # spikes present -> conditional counts
        ])
        self.assertEqual(gp.ev["p1"]["cat_rapidspin_total"], 2)
        self.assertEqual(gp.ev["p1"]["cat_rapidspin_spikesdown"], 1)

    def test_solar_beam_by_sun(self):
        gp = parse([
            "|turn|1",
            "|move|p1a: Sunflora|Solar Beam|p2a: X",         # no sun
            "|turn|2",
            "|-weather|SunnyDay",
            "|move|p1a: Sunflora|Solar Beam|p2a: X",         # sun
        ])
        self.assertEqual(gp.ev["p1"]["cat_solarbeam"], 2)
        self.assertEqual(gp.ev["p1"]["cat_solarbeam_sun"], 1)
        self.assertEqual(gp.ev["p1"]["cat_solarbeam_nosun"], 1)

    def test_lockedmove_continuation_is_not_a_decision(self):
        # Solar Beam with no sun charges (the decision), then is re-emitted next turn as
        # [from] lockedmove (forced). Counting the locked line double-counts the move and, since
        # only no-sun beams ever charge, deflates the in-sun rate. Only the charge should count.
        gp = parse([
            "|turn|1",
            "|move|p1a: Tangela|Solar Beam||[still]",      # charge = the decision
            "|-prepare|p1a: Tangela|Solar Beam",
            "|turn|2",
            "|move|p1a: Tangela|Solar Beam|p2a: Y|[from] lockedmove",   # forced, not a decision
            "|-damage|p2a: Y|100/262",
        ])
        self.assertEqual(gp.ev["p1"]["cat_solarbeam"], 1)
        self.assertEqual(gp.moves_total["p1"], 1)

    def test_instant_solarbeam_in_sun_counts_once(self):
        gp = parse([
            "|turn|1",
            "|-weather|SunnyDay",
            "|move|p1a: Sunflora|Solar Beam||[still]",     # under sun: fires same turn
            "|-anim|p1a: Sunflora|Solar Beam|p2a: Y",
            "|-damage|p2a: Y|50/262",
        ])
        self.assertEqual(gp.ev["p1"]["cat_solarbeam"], 1)
        self.assertEqual(gp.ev["p1"]["cat_solarbeam_sun"], 1)

    def test_status_moves_and_burn(self):
        # dedicated status-only moves count toward cat_status_move (aggregate) and their specific
        # category; Will-O-Wisp fills the burn gap. A secondary status (Body Slam paralysis) does not.
        gp = parse([
            "|turn|1",
            "|move|p1a: Gengar|Will-O-Wisp|p2a: Skarmory",     # burn
            "|move|p1a: Gengar|Thunder Wave|p2a: Blissey",     # paralysis
            "|move|p1a: Gengar|Toxic|p2a: Blissey",            # poison
            "|move|p1a: Gengar|Spore|p2a: Blissey",            # sleep
            "|move|p1a: Snorlax|Body Slam|p2a: Blissey",       # attack w/ 2ndary para -> NOT a status move
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["cat_burn"], 1)
        self.assertEqual(e["cat_para"], 1)          # Thunder Wave only (not Body Slam)
        self.assertEqual(e["cat_status_move"], 4)   # WoW + T-Wave + Toxic + Spore, Body Slam excluded

    def test_knock_off_tracked(self):
        gp = parse(["|turn|1", "|move|p1a: Tyranitar|Knock Off|p2a: Blissey"])
        self.assertEqual(gp.ev["p1"]["cat_knockoff"], 1)

    def test_reversal_bp_tracks_hp_at_use(self):
        # Reversal fired at high HP scores low BP; fired near death scores high BP. The average over
        # both uses reflects whether the policy times them at low HP (the correct play).
        gp = parse([
            "|switch|p1a: Heracross|Heracross, M|250/250",
            "|turn|1",
            "|move|p1a: Heracross|Reversal|p2a: Blissey",         # full HP -> BP 20
            "|turn|2",
            "|-damage|p1a: Heracross|8/250",                       # ratio floor(8*48/250)=1 -> BP 200
            "|move|p1a: Heracross|Flail|p2a: Blissey",             # grouped with Reversal
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["cat_reversal"], 2)
        self.assertEqual(e["reversal_bp_sum"], 220)               # 20 + 200
        self.assertEqual(TE.reversal_bp(250, 250), 20)
        self.assertEqual(TE.reversal_bp(8, 250), 200)

    def test_belly_drum_ko_attribution(self):
        # After Belly Drum, count opponent mons the drummer KOs before it leaves. A faint on the
        # OTHER seat while the drummer is active is credited; the window closes when the drummer switches.
        gp = parse([
            "|switch|p1a: Linoone|Linoone, M|300/300",
            "|switch|p2a: Blissey|Blissey|300/300",
            "|turn|1",
            "|move|p1a: Linoone|Belly Drum|p1a: Linoone",
            "|turn|2",
            "|move|p1a: Linoone|Extreme Speed|p2a: Blissey",
            "|faint|p2a: Blissey",                                 # KO #1 -> credited to p1 drummer
            "|switch|p2a: Snorlax|Snorlax, M|400/400",
            "|turn|3",
            "|move|p1a: Linoone|Extreme Speed|p2a: Snorlax",
            "|faint|p2a: Snorlax",                                 # KO #2
            "|switch|p2a: Suicune|Suicune|360/360",
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["cat_bellydrum"], 1)
        self.assertEqual(e["bellydrum_ko_sum"], 2)
        self.assertEqual(e["bellydrum_success"], 1)   # this use converted to >=1 KO

    def test_belly_drum_no_ko_is_not_a_success(self):
        # a Belly Drum whose user is revenge-killed before it lands a KO: counts as a use, avg-KOs 0,
        # and NOT a success (0 KOs).
        gp = parse([
            "|switch|p1a: Linoone|Linoone, M|300/300",
            "|switch|p2a: Salamence|Salamence, M|330/330",
            "|turn|1",
            "|move|p1a: Linoone|Belly Drum|p1a: Linoone",
            "|turn|2",
            "|move|p2a: Salamence|Earthquake|p1a: Linoone",
            "|faint|p1a: Linoone",                                 # drummer dies with 0 KOs
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["cat_bellydrum"], 1)
        self.assertEqual(e["bellydrum_ko_sum"], 0)
        self.assertEqual(e["bellydrum_success"], 0)

    def test_belly_drum_window_closes_on_drummer_faint(self):
        # If the drummer itself faints, its window is flushed with whatever it had scored so far.
        gp = parse([
            "|switch|p1a: Linoone|Linoone, M|300/300",
            "|switch|p2a: Blissey|Blissey|300/300",
            "|turn|1",
            "|move|p1a: Linoone|Belly Drum|p1a: Linoone",
            "|turn|2",
            "|move|p1a: Linoone|Extreme Speed|p2a: Blissey",
            "|faint|p2a: Blissey",                                 # KO #1
            "|switch|p2a: Weezing|Weezing, M|280/280",
            "|turn|3",
            "|faint|p1a: Linoone",                                 # drummer dies -> window flushes at 1
        ])
        self.assertEqual(gp.ev["p1"]["cat_bellydrum"], 1)
        self.assertEqual(gp.ev["p1"]["bellydrum_ko_sum"], 1)
        self.assertEqual(gp.ev["p1"]["bellydrum_success"], 1)

    def test_priority_vs_faster_and_ko(self):
        # p2 outspeeds p1 (moves first in a neutral turn); next turn p1's priority move is therefore
        # "vs faster" AND lands the KO immediately.
        gp = parse([
            "|switch|p1a: Scizor|Scizor, M|300/300",
            "|switch|p2a: Starmie|Starmie|260/260",
            "|turn|1",
            "|move|p2a: Starmie|Surf|p1a: Scizor",         # p2 first (neutral) -> p2 faster
            "|-damage|p1a: Scizor|150/300",
            "|move|p1a: Scizor|Recover|p1a: Scizor",       # p1 second (neutral) -> p1 slower
            "|turn|2",
            "|move|p1a: Scizor|Mach Punch|p2a: Starmie",   # slower[p1]=True -> vs_faster; immediate KO
            "|-damage|p2a: Starmie|0 fnt",
            "|faint|p2a: Starmie",
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["cat_priority"], 1)
        self.assertEqual(e["cat_priority_vs_faster"], 1)
        self.assertEqual(e["cat_priority_ko"], 1)

    def test_priority_not_vs_faster_when_we_are_faster(self):
        gp = parse([
            "|switch|p1a: Dodrio|Dodrio, M|250/250",
            "|switch|p2a: Snorlax|Snorlax, M|400/400",
            "|turn|1",
            "|move|p1a: Dodrio|Return|p2a: Snorlax",       # p1 first (neutral) -> p1 faster
            "|-damage|p2a: Snorlax|300/400",
            "|move|p2a: Snorlax|Body Slam|p1a: Dodrio",
            "|-damage|p1a: Dodrio|150/250",
            "|turn|2",
            "|move|p1a: Dodrio|Quick Attack|p2a: Snorlax", # slower[p1]=False -> NOT vs_faster
            "|-damage|p2a: Snorlax|280/400",
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["cat_priority"], 1)
        self.assertEqual(e["cat_priority_vs_faster"], 0)
        self.assertEqual(e["cat_priority_ko"], 0)

    def test_priority_ko_not_credited_for_residual_faint(self):
        # A priority move that only chips, followed by a residual (poison) faint, is not a priority KO.
        gp = parse([
            "|switch|p1a: Dugtrio|Dugtrio, M|200/200",
            "|switch|p2a: Gengar|Gengar|260/260",
            "|turn|1",
            "|move|p2a: Gengar|Toxic|p1a: Dugtrio",
            "|move|p1a: Dugtrio|Quick Attack|p2a: Gengar",
            "|-damage|p2a: Gengar|20/260",                                  # survives the priority hit
            "|-damage|p2a: Gengar|0 fnt|[from] psn|[of] p1a: Dugtrio",      # residual -> clears window
            "|faint|p2a: Gengar",
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["cat_priority"], 1)
        self.assertEqual(e["cat_priority_ko"], 0)

    def test_priority_speed_unknown_first_turn(self):
        # No prior neutral turn -> speed unknown -> a leading priority move is not counted "vs faster".
        gp = parse([
            "|switch|p1a: Dodrio|Dodrio, M|250/250",
            "|switch|p2a: Snorlax|Snorlax, M|400/400",
            "|turn|1",
            "|move|p1a: Dodrio|Quick Attack|p2a: Snorlax",
            "|-damage|p2a: Snorlax|360/400",
        ])
        self.assertEqual(gp.ev["p1"]["cat_priority"], 1)
        self.assertEqual(gp.ev["p1"]["cat_priority_vs_faster"], 0)

    def test_destiny_bond_success_and_failure(self):
        # DB drags the attacker down -> -activate fires -> success. A DB whose user survives does not.
        won = parse([
            "|switch|p1a: Misdreavus|Misdreavus, F|230/230",
            "|switch|p2a: Tyranitar|Tyranitar, M|340/340",
            "|turn|1",
            "|move|p1a: Misdreavus|Destiny Bond|p1a: Misdreavus",
            "|-singlemove|p1a: Misdreavus|Destiny Bond",
            "|turn|2",
            "|move|p2a: Tyranitar|Crunch|p1a: Misdreavus",
            "|-damage|p1a: Misdreavus|0 fnt",
            "|faint|p1a: Misdreavus",
            "|-activate|p1a: Misdreavus|move: Destiny Bond",
            "|faint|p2a: Tyranitar",
        ])
        self.assertEqual(won.ev["p1"]["cat_destinybond"], 1)
        self.assertEqual(won.ev["p1"]["destinybond_success"], 1)

        missed = parse([
            "|turn|1",
            "|move|p1a: Misdreavus|Destiny Bond|p1a: Misdreavus",
            "|-singlemove|p1a: Misdreavus|Destiny Bond",
            "|turn|2",
            "|move|p1a: Misdreavus|Shadow Ball|p2a: Blissey",   # user lived; DB wore off, no -activate
        ])
        self.assertEqual(missed.ev["p1"]["cat_destinybond"], 1)
        self.assertEqual(missed.ev["p1"]["destinybond_success"], 0)

    def test_sunny_day_counts_to_weather_sun(self):
        gp = parse(["|turn|1", "|move|p1a: Groudon|Sunny Day|p1a: Groudon"])
        self.assertEqual(gp.ev["p1"]["cat_weather_sun"], 1)

    def test_intimidate_activation(self):
        gp = parse([
            "|switch|p1a: Gyarados|Gyarados, M|300/300",
            "|-ability|p1a: Gyarados|Intimidate|boost",
            "|-unboost|p2a: Snorlax|atk|1",
            "|turn|1",
        ])
        self.assertEqual(gp.ev["p1"]["intimidate_activation"], 1)

    def test_absorb_switchin_read_counts_both(self):
        # Volt Absorb negating a Thunderbolt on the switch-in turn is BOTH a switch-in immunity and
        # the more advantageous absorb read (per design, it credits both).
        gp = parse([
            "|turn|3",
            "|switch|p1a: Jolteon|Jolteon, F|260/260",
            "|move|p2a: Zapdos|Thunderbolt|p1a: Jolteon",
            "|-immune|p1a: Jolteon|[from] ability: Volt Absorb",
            "|turn|4",
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["absorb_switchin"], 1)
        self.assertEqual(e["absorb_activation"], 1)
        self.assertEqual(e["immunity_switchin"], 1)

    def test_absorb_heal_variant_and_not_switchin(self):
        # A damaged Water Absorb mon that stays in and eats a Surf heals (a -heal absorb line) — an
        # activation (so the ability reads as present) but not a switch-in read.
        gp = parse([
            "|switch|p1a: Vaporeon|Vaporeon, F|300/400",
            "|turn|5",                                          # not the switch-in turn
            "|move|p2a: Starmie|Surf|p1a: Vaporeon",
            "|-heal|p1a: Vaporeon|360/400|[from] ability: Water Absorb",
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["absorb_activation"], 1)
        self.assertEqual(e["absorb_switchin"], 0)

    def test_toxic_stage_peak_and_switch_reset(self):
        # p1's mon is toxiced and takes 3 ticks before switching out (peak 3). A second mon is toxiced
        # and switched out after 1 tick (peak 1). Average peak across the two episodes = 2.0.
        gp = parse([
            "|switch|p1a: Skarmory|Skarmory, M|270/270",
            "|switch|p2a: Gengar|Gengar|260/260",
            "|turn|1",
            "|move|p2a: Gengar|Toxic|p1a: Skarmory",
            "|-status|p1a: Skarmory|tox",
            "|-damage|p1a: Skarmory|253/270 tox|[from] psn",     # tick 1
            "|turn|2",
            "|-damage|p1a: Skarmory|219/270 tox|[from] psn",     # tick 2
            "|turn|3",
            "|-damage|p1a: Skarmory|168/270 tox|[from] psn",     # tick 3
            "|switch|p1a: Blissey|Blissey, F|360/360",           # peak 3 recorded, counter resets
            "|turn|4",
            "|move|p2a: Gengar|Toxic|p1a: Blissey",
            "|-status|p1a: Blissey|tox",
            "|-damage|p1a: Blissey|337/360 tox|[from] psn",      # tick 1
            "|switch|p1a: Skarmory|Skarmory, M|168/270",         # peak 1 recorded
            "|turn|5",
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["tox_episodes"], 2)
        self.assertEqual(e["tox_stage_sum"], 4)                  # 3 + 1

    def test_boom_block_protect_sub_ghost_and_miss(self):
        # p1 blocks four enemy booms four different ways (Protect, Substitute, Ghost immunity) and
        # eats one; boom_faced counts all, boom_block counts the three that were neutralized.
        gp = parse([
            # 1) Protect
            "|turn|1",
            "|move|p1a: Zapdos|Protect|p1a: Zapdos",
            "|-singleturn|p1a: Zapdos|Protect",
            "|move|p2a: Forretress|Explosion|p1a: Zapdos",
            "|-activate|p1a: Zapdos|move: Protect",
            "|faint|p2a: Forretress",
            # 2) Substitute up
            "|turn|2",
            "|switch|p2a: Metagross|Metagross|320/320",
            "|-start|p1a: Zapdos|Substitute",
            "|turn|3",
            "|move|p2a: Metagross|Explosion|p1a: Zapdos",
            "|-end|p1a: Zapdos|Substitute",
            "|faint|p2a: Metagross",
            # 3) Ghost immunity
            "|turn|4",
            "|switch|p2a: Snorlax|Snorlax, M|400/400",
            "|switch|p1a: Gengar|Gengar|260/260",
            "|turn|5",
            "|move|p2a: Snorlax|Self-Destruct|p1a: Gengar",
            "|-immune|p1a: Gengar",
            "|faint|p2a: Snorlax",
            # 4) not blocked — takes the hit
            "|turn|6",
            "|switch|p2a: Regirock|Regirock|350/350",
            "|switch|p1a: Blissey|Blissey, F|360/360",
            "|turn|7",
            "|move|p2a: Regirock|Explosion|p1a: Blissey",
            "|-damage|p1a: Blissey|0 fnt",
            "|faint|p1a: Blissey",
            "|faint|p2a: Regirock",
        ])
        e = gp.ev["p1"]
        self.assertEqual(e["boom_faced"], 4)
        self.assertEqual(e["boom_block"], 3)

    def test_weather_move_not_ability(self):
        gp = parse([
            "|turn|1",
            "|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Politoed",  # ability, not a move
            "|move|p1a: Kingdra|Rain Dance|p1a: Kingdra",                      # the move
        ])
        self.assertEqual(gp.ev["p1"]["cat_weather_rain"], 1)  # only the move is counted


class SwitchBehavior(unittest.TestCase):
    def test_immunity_switchin_same_turn_only(self):
        # p1 switches Gengar in; p2's Earthquake that turn is immune -> counts once.
        gp = parse([
            "|turn|3",
            "|switch|p1a: Gengar|Gengar|260/260",
            "|move|p2a: Flygon|Earthquake|p1a: Gengar",
            "|-immune|p1a: Gengar",
        ])
        self.assertEqual(gp.ev["p1"]["immunity_switchin"], 1)

    def test_immunity_next_turn_does_not_count(self):
        gp = parse([
            "|turn|3",
            "|switch|p1a: Gengar|Gengar|260/260",
            "|turn|4",                                   # new turn resets the switch-in window
            "|move|p2a: Flygon|Earthquake|p1a: Gengar",
            "|-immune|p1a: Gengar",
        ])
        self.assertEqual(gp.ev["p1"]["immunity_switchin"], 0)

    def test_forced_switch_not_a_pivot(self):
        gp = parse([
            "|turn|1",
            "|faint|p1a: Skarmory",
            "|switch|p1a: Blissey|Blissey|300/300",   # forced replacement
            "|turn|2",
            "|switch|p1a: Gengar|Gengar|260/260",     # voluntary pivot
        ])
        self.assertEqual(gp.ev["p1"]["pivot"], 1)
        self.assertEqual(gp.ev["p1"]["forced_switch"], 1)

    def test_switch_out_sleeping(self):
        gp = parse([
            "|turn|1",
            "|switch|p1a: Snorlax|Snorlax|500/500",
            "|-status|p1a: Snorlax|slp",              # enemy-inflicted sleep
            "|turn|2",
            "|switch|p1a: Gengar|Gengar|260/260",     # sleeping Snorlax pivoted out
        ])
        self.assertEqual(gp.ev["p1"]["switch_out_sleeping"], 1)

    def test_rest_sleeper_switch_out_not_counted(self):
        # Rest is a self-chosen heal; pivoting the Rest-sleeper is not the tracked behavior.
        gp = parse([
            "|turn|1",
            "|switch|p1a: Snorlax|Snorlax|500/500",
            "|move|p1a: Snorlax|Rest|p1a: Snorlax",
            "|-status|p1a: Snorlax|slp|[from] move: Rest",
            "|turn|2",
            "|switch|p1a: Gengar|Gengar|260/260",     # Rest-sleeper pivoted out -> NOT counted
        ])
        self.assertEqual(gp.ev["p1"]["switch_out_sleeping"], 0)


class PPAndSpecies(unittest.TestCase):
    def test_pp_exhaustion_dedup_by_mon_move(self):
        # same move at 0 PP across three snapshots = ONE exhausted move, not three.
        pp_track = [
            {"turn": 5, "seat": "p1", "mon": "Skarmory", "moves": [{"id": "spikes", "pp": 0}]},
            {"turn": 6, "seat": "p1", "mon": "Skarmory", "moves": [{"id": "spikes", "pp": 0}]},
            {"turn": 7, "seat": "p1", "mon": "Skarmory", "moves": [{"id": "spikes", "pp": 0}]},
        ]
        bot, opp = TE._pp_exhaustions(pp_track)
        self.assertEqual(bot, 1)
        self.assertEqual(opp, 0)

    def test_species_vector_win_delta_self_play(self):
        team = TE.Counter({"Zapdos": 100, "Weezing": 100})
        win = TE.Counter({"Zapdos": 70, "Weezing": 30})
        sv = TE._species_vector(team, win, wins_bot=50, n=100, opponent="self")
        self.assertAlmostEqual(sv["Zapdos"]["p_win_given_on_team"], 0.7, places=6)
        self.assertAlmostEqual(sv["Zapdos"]["win_delta"], 0.2, places=6)   # vs 0.5 self-play baseline
        self.assertAlmostEqual(sv["Weezing"]["win_delta"], -0.2, places=6)


class PerGameCorrelation(unittest.TestCase):
    """The per-game trait->win correlation: n is games, and self-play is a paired winner-vs-loser
    design (same policy both seats, same game), so game-level quantities cannot leak in."""

    def _game(self, seed, winner, p1_subs, p2_subs, capped=False):
        proto = ["|player|p1|Bot p1|", "|player|p2|Bot p2|", "|turn|1"]
        proto += ["|move|p1a: X|Substitute|p1a: X"] * p1_subs
        proto += ["|move|p2a: Y|Substitute|p2a: Y"] * p2_subs
        if winner:
            proto.append(f"|win|Bot {winner}")
        ms = {"p1": [{"species": "X", "moves": ["Substitute"]}],
              "p2": [{"species": "Y", "moves": ["Substitute"]}]}
        return {"seed": seed, "opponent": "self", "winner": winner, "turn_count": 30,
                "capped": capped, "protocol": proto, "movesets": ms, "pp_track": []}

    def _extract(self, games):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events-0.jsonl.gz")
            with gzip.open(path, "wt") as f:
                f.write(json.dumps({"record": "manifest", "opponent": "self"}) + "\n")
                for g in games:
                    f.write(json.dumps(g) + "\n")
            return TE.extract([path])

    def test_winner_uses_more_gives_positive_r(self):
        # in every game the winning seat used more Substitute -> strong positive r
        games = [self._game(i, "p1" if i % 2 else "p2",
                            p1_subs=4 if i % 2 else 0, p2_subs=0 if i % 2 else 4) for i in range(10)]
        m = self._extract(games)
        c = m["per_game_correlations"]["cat_substitute"]
        self.assertEqual(c["n"], 20)               # 10 games x 2 behavioral seats
        self.assertGreater(c["r"], 0.9)

    def test_loser_uses_more_gives_negative_r(self):
        games = [self._game(i, "p1" if i % 2 else "p2",
                            p1_subs=0 if i % 2 else 4, p2_subs=4 if i % 2 else 0) for i in range(10)]
        m = self._extract(games)
        self.assertLess(m["per_game_correlations"]["cat_substitute"]["r"], -0.9)

    def test_timeouts_excluded_from_per_game_rows(self):
        # a timeout has no winner; counting it would silently score both seats as losses
        games = [self._game(i, "p1", p1_subs=2, p2_subs=1) for i in range(5)]
        games.append(self._game(99, None, p1_subs=9, p2_subs=9, capped=True))
        m = self._extract(games)
        self.assertEqual(m["per_game_rows"], 10)   # 5 decided games x 2 seats; timeout dropped


    def test_outcome_definitional_traits_excluded(self):
        # forced_switch is caused by your mon fainting, so it is ~"mons lost" — correlating it
        # with losing restates the outcome. It must never be in the per-game trait set.
        self.assertNotIn("forced_switch", TE.PER_GAME_TRAITS)
        for banned in ("forced_switch", "mons_alive", "last_active"):
            self.assertFalse(any(banned in t for t in TE.PER_GAME_TRAITS), f"{banned} is circular")

    def test_no_signal_when_usage_unrelated_to_winning(self):
        # both seats always use the same amount -> no within-game variance -> r undefined/dropped
        games = [self._game(i, "p1" if i % 2 else "p2", p1_subs=3, p2_subs=3) for i in range(10)]
        m = self._extract(games)
        self.assertNotIn("cat_substitute", m["per_game_correlations"])


class EndToEnd(unittest.TestCase):
    def test_extract_over_tiny_events_file(self):
        movesets = {"p1": [{"species": "Skarmory", "moves": ["Toxic", "Spikes", "Roar", "Protect"]}],
                    "p2": [{"species": "Blissey", "moves": ["Softboiled", "Seismic Toss", "Toxic", "Protect"]}]}
        protocol = [
            "|player|p1|PokeZero p1|", "|player|p2|PokeZero p2|",
            "|turn|1",
            "|move|p1a: Skarmory|Toxic|p2a: Blissey",
            "|move|p2a: Blissey|Softboiled|p2a: Blissey",
            "|turn|2",
            "|move|p1a: Skarmory|Spikes|p2a: Blissey",
            "|faint|p2a: Blissey",
            "|win|PokeZero p1",
        ]
        game = {"seed": 1, "opponent": "self", "winner": "p1", "turn_count": 2, "capped": False,
                "protocol": protocol, "movesets": movesets, "pp_track": []}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events-0.jsonl.gz")
            with gzip.open(path, "wt") as f:
                f.write(json.dumps({"record": "manifest", "opponent": "self"}) + "\n")
                f.write(json.dumps(game) + "\n")
            m = TE.extract([path], lineage="testlin", milestone=500000)
        self.assertEqual(m["n_games"], 1)
        self.assertEqual(m["seat_games"], 2)
        self.assertEqual(m["lineage"], "testlin")
        self.assertEqual(m["avg_turns"], 2.0)
        self.assertEqual(m["timeout_rate"], 0.0)
        self.assertEqual(m["decided_games"], 1)
        # Toxic used once (p1); Softboiled once (p2) — both behavioral in self-play
        mc = m["move_categories"]
        self.assertEqual(mc["cat_toxic"]["total_uses"], 1)
        self.assertEqual(mc["cat_heal"]["total_uses"], 1)
        self.assertEqual(m["bot_win_rate"], 1.0)

    def test_ability_presence_gating_exact(self):
        # movesets carry an ability -> gate exactly on team composition. p1 has Volt Absorb (present,
        # and its switch-in read fires); p2 has Pressure (absent). No intimidator on either team.
        movesets = {"p1": [{"species": "Jolteon", "moves": ["Thunderbolt", "Substitute"], "ability": "Volt Absorb"}],
                    "p2": [{"species": "Zapdos", "moves": ["Thunderbolt", "Roar"], "ability": "Pressure"}]}
        protocol = [
            "|turn|3",
            "|switch|p1a: Jolteon|Jolteon, F|260/260",
            "|move|p2a: Zapdos|Thunderbolt|p1a: Jolteon",
            "|-immune|p1a: Jolteon|[from] ability: Volt Absorb",
            "|turn|4",
            "|faint|p2a: Zapdos",
            "|win|Bot p1",
        ]
        game = {"seed": 1, "opponent": "self", "winner": "p1", "turn_count": 4, "capped": False,
                "protocol": protocol, "movesets": movesets, "pp_track": []}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events-0.jsonl.gz")
            with gzip.open(path, "wt") as f:
                f.write(json.dumps({"record": "manifest", "opponent": "self"}) + "\n")
                f.write(json.dumps(game) + "\n")
            m = TE.extract([path])
        self.assertEqual(m["absorb_present_seat_games"], 1)          # only p1's team carries an absorber
        self.assertEqual(m["absorb_switchins_per_game"], 1.0)
        self.assertEqual(m["intimidate_present_seat_games"], 0)      # no intimidator on either team
        self.assertIsNone(m["intimidate_activations_per_game"])

    def test_toxic_and_boom_output_metrics(self):
        # one badly-poisoned mon reaches stage 1, and one enemy Explosion is blocked by Protect.
        movesets = {"p1": [{"species": "Skarmory", "moves": ["Protect", "Spikes", "Roar", "Whirlwind"]}],
                    "p2": [{"species": "Forretress", "moves": ["Explosion", "Spikes", "Toxic", "Rapid Spin"]}]}
        protocol = [
            "|turn|1",
            "|switch|p1a: Skarmory|Skarmory, M|270/270",
            "|switch|p2a: Forretress|Forretress, M|350/350",
            "|turn|2",
            "|move|p2a: Forretress|Toxic|p1a: Skarmory",
            "|-status|p1a: Skarmory|tox",
            "|-damage|p1a: Skarmory|253/270 tox|[from] psn",
            "|turn|3",
            "|move|p1a: Skarmory|Protect|p1a: Skarmory",
            "|-singleturn|p1a: Skarmory|Protect",
            "|move|p2a: Forretress|Explosion|p1a: Skarmory",
            "|-activate|p1a: Skarmory|move: Protect",
            "|faint|p2a: Forretress",
            "|win|Bot p1",
        ]
        game = {"seed": 1, "opponent": "self", "winner": "p1", "turn_count": 3, "capped": False,
                "protocol": protocol, "movesets": movesets, "pp_track": []}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events-0.jsonl.gz")
            with gzip.open(path, "wt") as f:
                f.write(json.dumps({"record": "manifest", "opponent": "self"}) + "\n")
                f.write(json.dumps(game) + "\n")
            m = TE.extract([path])
        self.assertEqual(m["toxic_episodes"], 1)
        self.assertEqual(m["avg_toxic_stage"], 1.0)                  # one tick before game end
        self.assertEqual(m["boom_faced"], 1)
        self.assertEqual(m["boom_blocks"], 1)
        self.assertEqual(m["boom_block_rate"], 1.0)

    def test_ability_per_game_excludes_capped(self):
        # A timeout stall inflates per-game ability counts (endless intimidator pivoting), so capped
        # games must not contribute to intimidate/absorb per-game rates. Decided game: 1 activation;
        # capped game: 5 activations that must be ignored -> per-game stays 1.0 over 1 present game.
        ms = {"p1": [{"species": "Gyarados", "moves": ["Earthquake"], "ability": "Intimidate"}],
              "p2": [{"species": "Snorlax", "moves": ["Body Slam"], "ability": "Immunity"}]}
        decided = ["|turn|1", "|switch|p1a: Gyarados|Gyarados, M|300/300",
                   "|-ability|p1a: Gyarados|Intimidate|boost", "|-unboost|p2a: Snorlax|atk|1", "|win|Bot p1"]
        stall = ["|turn|1"] + sum(([f"|switch|p1a: Gyarados|Gyarados, M|300/300",
                                    "|-ability|p1a: Gyarados|Intimidate|boost",
                                    "|-unboost|p2a: Snorlax|atk|1"] for _ in range(5)), [])
        games = [
            {"seed": 1, "opponent": "self", "winner": "p1", "turn_count": 20, "capped": False,
             "protocol": decided, "movesets": ms, "pp_track": []},
            {"seed": 2, "opponent": "self", "winner": None, "turn_count": 200, "capped": True,
             "protocol": stall, "movesets": ms, "pp_track": []},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events-0.jsonl.gz")
            with gzip.open(path, "wt") as f:
                f.write(json.dumps({"record": "manifest", "opponent": "self"}) + "\n")
                for g in games:
                    f.write(json.dumps(g) + "\n")
            m = TE.extract([path])
        self.assertEqual(m["intimidate_present_seat_games"], 1)          # only the decided game counts
        self.assertEqual(m["intimidate_activations_per_game"], 1.0)      # the 5-activation stall ignored

    def test_timeout_excluded_from_avg_turns(self):
        # one decided 30-turn game + one 200-turn timeout: avg_turns is 30 (decided only),
        # timeout_rate 0.5, and the timeout still contributes its moves to the categories.
        proto_decided = ["|player|p1|Bot p1|", "|player|p2|Bot p2|", "|turn|1",
                         "|move|p1a: X|Toxic|p2a: Y", "|win|Bot p1"]
        proto_timeout = ["|player|p1|Bot p1|", "|player|p2|Bot p2|", "|turn|1",
                         "|move|p1a: X|Toxic|p2a: Y"]  # no |win| — stalled
        ms = {"p1": [{"species": "X", "moves": ["Toxic"]}], "p2": [{"species": "Y", "moves": ["Toxic"]}]}
        games = [
            {"seed": 1, "opponent": "self", "winner": "p1", "turn_count": 30, "capped": False,
             "protocol": proto_decided, "movesets": ms, "pp_track": []},
            {"seed": 2, "opponent": "self", "winner": None, "turn_count": 200, "capped": True,
             "protocol": proto_timeout, "movesets": ms, "pp_track": []},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events-0.jsonl.gz")
            with gzip.open(path, "wt") as f:
                f.write(json.dumps({"record": "manifest", "opponent": "self"}) + "\n")
                for g in games:
                    f.write(json.dumps(g) + "\n")
            m = TE.extract([path])
        self.assertEqual(m["n_games"], 2)
        self.assertEqual(m["avg_turns"], 30.0)       # timeout's 200 excluded
        self.assertEqual(m["decided_games"], 1)
        self.assertEqual(m["timeout_rate"], 0.5)
        self.assertEqual(m["move_categories"]["cat_toxic"]["total_uses"], 2)  # timeout still counted


if __name__ == "__main__":
    unittest.main()
