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
