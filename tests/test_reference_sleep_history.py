"""Public protocol contracts, independent of hidden abilities and timer draws."""
import unittest
from fractions import Fraction

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.showdown_fixture import FixturePokemon, pack_team

from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.reference_sleep_history import public_sleep_history
from pokezero.mcts_eval.reference_sleep import (
    SleepAttempt, SleepHypothesis, SleepReentry, condition_sleep,
)


OPEN = ("|gen|3", "|gametype|singles", "|start",
        "|switch|p1a: Articuno|Articuno|100/100",
        "|switch|p2a: Breloom|Breloom|100/100", "|turn|1")
ONSET = "|-status|p1a: Articuno|slp"
CANT = "|cant|p1a: Articuno|slp"


def extract(*suffix, boundary=True):
    return public_sleep_history(OPEN + suffix, at_action_boundary=boundary).episodes


class PublicSleepHistoryTests(unittest.TestCase):
    def test_external_source_and_rest_self_source_are_distinct(self):
        row, = extract(ONSET)
        self.assertEqual((row.key, row.source_key, row.kind),
                         ("p1:articuno", "p2:breloom", "induced"))
        row, = extract(ONSET + "|[from] move: Rest")
        self.assertEqual((row.source_key, row.kind), (row.key, "rest"))

    def test_direct_sleep_talk_and_snore_mark_one_refundable_attempt(self):
        for move in ("Sleep Talk", "Snore"):
            row, = extract(ONSET, CANT, f"|move|p1a: Articuno|{move}|p1a: Articuno",
                           "|move|p1a: Articuno|Splash|p1a: Articuno|[from] move: Sleep Talk")
            attempts = [event for event in row.events if event.kind == "attempt"]
            self.assertEqual([event.sleep_usable for event in attempts], [True])

    def test_confusion_attract_and_later_cant_prove_sleep_usable_without_move(self):
        for proof in ("|-activate|p1a: Articuno|confusion",
                      "|-activate|p1a: Articuno|move: Attract|[of] p2a: Breloom",
                      "|cant|p1a: Articuno|flinch"):
            with self.subTest(proof=proof):
                row, = extract(ONSET, CANT, proof, "|upkeep")
                self.assertEqual([event.sleep_usable for event in row.events
                                  if event.kind == "attempt"], [True])

    def test_unclassified_attempt_only_settles_at_a_proved_boundary(self):
        with self.assertRaisesRegex(ReferenceRefusal, "unsettled"):
            extract(ONSET, CANT, boundary=False)
        for suffix in ((), ("|upkeep",), ("|turn|2",)):
            row, = extract(ONSET, CANT, *suffix)
            self.assertEqual([event.sleep_usable for event in row.events
                              if event.kind == "attempt"], [False])

    def test_switch_refund_is_one_event_and_benched_residuals_are_not_attempts(self):
        row, = extract(ONSET, CANT, "|move|p1a: Articuno|Sleep Talk|p1a: Articuno",
            "|upkeep", "|switch|p1a: Skarmory|Skarmory|100/100", "|upkeep", "|upkeep",
            "|switch|p1a: Articuno|Articuno|100/100 slp")
        kinds = [event.kind for event in row.events]
        self.assertEqual((kinds.count("attempt"), kinds.count("reentry"), kinds.count("switch_out")), (1, 1, 1))
        self.assertEqual([event.fields for event in row.events if event.kind == "residual_boundary"],
                         [("active",), ("bench",), ("bench",)])

    def test_aromatherapy_closes_bench_episode_and_new_onset_is_separate(self):
        rows = extract(ONSET, "|switch|p1a: Skarmory|Skarmory|100/100",
            "|-cureteam|p1a: Skarmory|[from] move: Aromatherapy",
            "|switch|p1a: Articuno|Articuno|100/100", ONSET)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].closed)
        self.assertEqual(rows[0].events[-1].kind, "team_cure")
        self.assertFalse(rows[1].closed)

    def test_heal_bell_records_exposure_but_does_not_invent_soundproof_cure(self):
        row, = extract(ONSET, "|-activate|p1a: Articuno|move: Heal Bell")
        self.assertFalse(row.closed)
        self.assertEqual(row.events[-1].kind, "heal_bell")
        row, = extract(ONSET, "|-activate|p1a: Articuno|move: Heal Bell",
                       "|-curestatus|p1: Articuno|slp|[silent]")
        self.assertTrue(row.closed)
        self.assertEqual(row.events[-1].kind, "cure")

    def test_untagged_cure_is_not_mislabeled_as_a_natural_wake(self):
        for cause in ("|-activate|p1a: Articuno|ability: Shed Skin",
                      "|-enditem|p1a: Articuno|Lum Berry|[eat]",
                      "|-enditem|p1a: Articuno|Chesto Berry|[eat]"):
            row, = extract(ONSET, cause, "|-curestatus|p1a: Articuno|slp|[msg]")
            self.assertTrue(row.closed)
            self.assertEqual(row.events[-1].kind, "cure")
            self.assertEqual(row.events[-1].fields, ("slp", "[msg]"))
            self.assertEqual(row.events[-2].fields, tuple(cause.split("|")[3:]))

    def test_ability_transform_and_item_events_are_preserved_not_assumed_away(self):
        row, = extract(ONSET, "|-ability|p1a: Articuno|Early Bird|Pressure|[from] move: Skill Swap",
                       "|-transform|p1a: Articuno|p2a: Breloom",
                       "|-item|p1a: Articuno|Lum Berry|[from] move: Trick")
        self.assertEqual([event.kind for event in row.events], ["onset", "ability", "transform", "item"])

    def test_pre_onset_traits_and_other_mon_sources_remain_in_the_full_ledger(self):
        lines = OPEN + ("|-ability|p1a: Articuno|Early Bird|Pressure|[from] move: Skill Swap",
                        "|-item|p2a: Breloom|Lum Berry|[from] move: Trick", ONSET)
        ledger = public_sleep_history(lines, at_action_boundary=True)
        self.assertEqual(ledger.public_lines, lines)
        self.assertEqual([event.kind for event in ledger.episodes[0].events], ["onset"])

    def test_ordinary_sleep_attempt_then_phazing_settles_before_upkeep(self):
        row, = extract(ONSET, CANT, "|move|p2a: Breloom|Roar|p1a: Articuno",
                       "|drag|p1a: Skarmory|Skarmory|100/100", "|upkeep")
        self.assertEqual([event.sleep_usable for event in row.events
                          if event.kind == "attempt"], [False])
        self.assertEqual([event.kind for event in row.events].count("switch_out"), 1)

    def test_delayed_yawn_retains_original_source_after_opponent_switch(self):
        row, = extract("|-start|p1a: Articuno|move: Yawn|[of] p2a: Breloom",
            "|switch|p2a: Wailord|Wailord|100/100", "|upkeep",
            "|-end|p1a: Articuno|move: Yawn|[silent]", ONSET)
        self.assertEqual(row.source_key, "p2:breloom")

    def test_failed_yawn_does_not_lend_source_to_later_spore(self):
        row, = extract("|-start|p1a: Articuno|move: Yawn|[of] p2a: Breloom",
            "|switch|p2a: Wailord|Wailord|100/100",
            "|-end|p1a: Articuno|move: Yawn|[silent]", "|-immune|p1a: Articuno", ONSET)
        self.assertEqual(row.source_key, "p2:wailord")

    def test_faint_and_repeated_onset_preserve_closed_episode(self):
        row, = extract(ONSET, "|faint|p1a: Articuno")
        self.assertTrue(row.closed)
        self.assertEqual(row.events[-1].kind, "faint")
        with self.assertRaises(ReferenceRefusal):
            extract(ONSET, ONSET)

    def test_private_inputs_missing_onset_and_inconsistent_switch_refuse(self):
        for suffix in (("|request|{}",), ("|split|p1",), (CANT,),
                       ("|switch|p1a: Articuno|Articuno|100/100 slp",),
                       (ONSET, "|switch|p1a: Skarmory|Skarmory|100/100", "|switch|p1a: Articuno|Articuno|100/100"),
                       (ONSET, "|move|p1a: Articuno|Splash|p1a: Articuno", CANT,
                        "|move|p1a: Articuno|Splash|p1a: Articuno"),
                       (ONSET, "|-status|p1a: Articuno|par")):
            with self.subTest(suffix=suffix), self.assertRaises(ReferenceRefusal):
                extract(*suffix)
        with self.assertRaises(ReferenceRefusal):
            public_sleep_history((ONSET,), at_action_boundary=True)
        with self.assertRaises(ReferenceRefusal):
            extract("|gametype|doubles")


@requires_showdown()
class PublicSleepServerTests(unittest.TestCase):
    def fixture(self, *, species="Articuno", ability="Pressure", healer="Splash", yawn=False):
        team = pack_team((FixturePokemon(species=species, ability=ability,
            moves=("Splash", "Sleep Talk", "Snore")),
            FixturePokemon(species="Skarmory", ability="Keen Eye", moves=(healer, "Splash"))))
        opponent = pack_team((FixturePokemon(species="Snorlax" if yawn else "Breloom",
            ability="Immunity" if yawn else "Effect Spore", moves=("Yawn" if yawn else "Spore", "Splash")),
            FixturePokemon(species="Wailord", ability="Water Veil", moves=("Splash",))))
        live = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root()))
        self.addCleanup(live.close)
        live.reset_with_start_override(seed=17,
            start_override=BattleStartOverride(player_teams={"p1": team, "p2": opponent}))
        return live

    def history(self, live):
        return public_sleep_history(live.public_materialization_state("p1").replay.public_lines,
                                    at_action_boundary=True).episodes

    def sleeper(self, live, species="Articuno"):
        return next(row for row in live.snapshot().bridge_snapshot["battle"]["sides"][0]["pokemon"]
                    if row["details"].split(",", 1)[0] == species)

    def test_real_public_attempt_ledger_matches_live_authored_timer_and_refund(self):
        live = self.fixture()
        live.step({"p1": 0, "p2": 0})
        # Private timer is an oracle output from this authored fixture only;
        # it is never an input to the ledger or conditional reconstruction.
        initial = self.sleeper(live)["statusState"]["time"]
        for choices in ({"p1": 1, "p2": 1}, {"p1": 4, "p2": 1},
                        {"p1": 1, "p2": 1}, {"p1": 4, "p2": 1}):
            live.step(choices)
            episode, = self.history(live)
            events = tuple(SleepAttempt(False, event.sleep_usable, True) if event.kind == "attempt"
                           else SleepReentry() for event in episode.events
                           if event.kind in {"attempt", "reentry"})
            posterior = condition_sleep((SleepHypothesis("authored", Fraction(1), events),), kind="induced")
            exact, = [row for row in posterior.outcomes if row.initial_time == initial]
            actual = self.sleeper(live)["statusState"]
            self.assertEqual((exact.time, exact.skipped_time), (actual["time"], actual["skippedTime"]))

    def test_real_aromatherapy_and_heal_bell_bench_cures_and_soundproof_exception(self):
        for healer, species, ability, closed in (("Aromatherapy", "Articuno", "Pressure", True),
                ("Heal Bell", "Articuno", "Pressure", True),
                ("Heal Bell", "Exploud", "Soundproof", False)):
            with self.subTest(healer=healer, species=species):
                live = self.fixture(species=species, ability=ability, healer=healer)
                live.step({"p1": 0, "p2": 0})
                live.step({"p1": 4, "p2": 1})
                live.step({"p1": 0, "p2": 1})
                episode, = self.history(live)
                self.assertEqual(episode.closed, closed)
                self.assertEqual(self.sleeper(live, species)["status"] == "", closed)

    def test_real_yawn_source_is_not_replaced_by_current_opponent(self):
        live = self.fixture(yawn=True)
        live.step({"p1": 0, "p2": 0})
        live.step({"p1": 0, "p2": 4})
        episode, = self.history(live)
        self.assertEqual(episode.source_key, "p2:snorlax")
        self.assertEqual(self.sleeper(live)["status"], "slp")

    def test_real_natural_cure_closes_episode_before_next_switch(self):
        live = self.fixture(species="Starmie", ability="Natural Cure")
        live.step({"p1": 0, "p2": 0})
        live.step({"p1": 4, "p2": 1})
        episode, = self.history(live)
        self.assertTrue(episode.closed)
        self.assertEqual(episode.events[-1].kind, "cure")
        self.assertEqual(self.sleeper(live, "Starmie")["status"], "")


if __name__ == "__main__":
    unittest.main()
