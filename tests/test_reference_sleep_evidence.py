"""Conditional evidence qualification, never a team-proposal or strength test."""
from fractions import Fraction
import json
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.mcts_eval.reference_sleep_evidence import (
    _ABILITIES, SleepTraits, qualify_sleep_episode,
)
from pokezero.mcts_eval.reference_sleep_history import public_sleep_history
from tests import test_reference_sleep_history as history_fixture
from tests.test_reference_sleep_history import OPEN, ONSET, CANT


TRAITS = {"p1:articuno": SleepTraits("Pressure"), "p1:skarmory": SleepTraits("Keen Eye"),
          "p2:breloom": SleepTraits("Effect Spore"), "p2:wailord": SleepTraits("Water Veil")}


def qualify(*suffix, traits=None, lines=None, index=0, boundary=True):
    ledger = public_sleep_history(tuple(lines) if lines is not None else OPEN + suffix,
                                  at_action_boundary=True)
    return qualify_sleep_episode(ledger, TRAITS if traits is None else traits,
                                 episode_index=index, at_action_boundary=boundary)


def actor(ability="Pressure", item=""):
    return dict(TRAITS, **{"p1:articuno": SleepTraits(ability, item)})


class ReferenceSleepEvidenceTests(unittest.TestCase):
    def test_explicit_episode_and_supplied_trait_likelihood(self):
        ordinary = qualify(ONSET, CANT, "|upkeep", CANT, "|upkeep")
        bird = qualify(ONSET, CANT, "|upkeep", CANT, "|upkeep", traits=actor("Early Bird"))
        self.assertEqual((ordinary.status, ordinary.evidence_probability), ("compatible", Fraction(3, 4)))
        self.assertEqual((bird.status, bird.evidence_probability), ("compatible", Fraction(1, 4)))
        self.assertEqual({row.hypothesis for row in bird.posterior.outcomes}, {"supplied-team"})

    def test_unknown_and_impossible_are_not_the_same_probability(self):
        unknown = qualify(ONSET, traits=actor("not-a-known-ability"))
        impossible = qualify(ONSET, traits=actor("Insomnia"))
        self.assertEqual((unknown.status, unknown.evidence_probability), ("unresolved", None))
        self.assertEqual((impossible.status, impossible.evidence_probability), ("incompatible", Fraction(0)))

    def test_invalid_identity_selection_and_boundary_remain_unresolved(self):
        for kwargs in (dict(index=True), dict(index=1), dict(boundary=False), dict(traits={}),
                       dict(traits=actor(item="not-a-known-item"))):
            with self.subTest(kwargs=kwargs):
                row = qualify(ONSET, **kwargs)
                self.assertEqual((row.status, row.evidence_probability), ("unresolved", None))
        lines = list(OPEN); lines[3] = "|switch|p1a: Skarmory|Articuno|100/100"
        row = qualify("|-status|p1a: Skarmory|slp", lines=tuple(lines) + ("|-status|p1a: Skarmory|slp",))
        self.assertEqual(row.reason, "nickname-species-ambiguity")

    def test_pre_onset_skill_swap_uses_both_current_traits_without_reveals(self):
        traits = dict(TRAITS, **{"p2:breloom": SleepTraits("Early Bird")})
        row = qualify("|-activate|p1a: Articuno|Skill Swap|||[of] p2a: Breloom",
                      ONSET, CANT, "|upkeep", "|turn|2", CANT, traits=traits)
        self.assertEqual((row.status, row.evidence_probability), ("compatible", Fraction(1, 4)))

    def test_role_play_and_transform_copy_the_preceding_source_trait(self):
        traits = dict(TRAITS, **{"p2:breloom": SleepTraits("Early Bird")})
        for change in ("|-ability|p1a: Articuno|Early Bird|Pressure|[from] move: Role Play|[of] p2a: Breloom",
                       "|-transform|p1a: Articuno|p2a: Breloom|[from] move: Transform"):
            with self.subTest(change=change):
                row = qualify(change, ONSET, CANT, "|upkeep", "|turn|2", CANT, traits=traits)
                self.assertEqual(row.evidence_probability, Fraction(1, 4))
        bad = qualify("|-ability|p1a: Articuno|Early Bird|Pressure|[from] move: Role Play|[of] p2a: Breloom",
                      ONSET)
        self.assertEqual((bad.status, bad.evidence_probability), ("incompatible", Fraction(0)))

    def test_switch_reset_and_reentry_refund_use_different_ability_frames(self):
        traits = dict(TRAITS, **{"p2:breloom": SleepTraits("Early Bird")})
        row = qualify("|-transform|p1a: Articuno|p2a: Breloom|[from] move: Transform", ONSET, CANT,
            "|move|p1a: Articuno|Sleep Talk|p1a: Articuno",
            "|switch|p1a: Skarmory|Skarmory|100/100", "|upkeep",
            "|switch|p1a: Articuno|Articuno|100/100 slp", CANT, traits=traits)
        self.assertEqual(row.status, "compatible")
        self.assertEqual([(value.initial_time, value.time, value.skipped_time)
                          for value in row.posterior.outcomes], [(3, 1, 0), (4, 2, 0), (5, 3, 0)])

    def test_copy_reveals_bind_the_old_ability_and_active_source(self):
        traits = dict(TRAITS, **{"p2:breloom": SleepTraits("Early Bird")})
        wrong_old = qualify("|-ability|p1a: Articuno|Early Bird|Trace|[from] move: Role Play|[of] p2a: Breloom",
                            ONSET, traits=traits)
        self.assertEqual(wrong_old.reason, "original-ability-reveal-contradiction")
        missing_old = qualify("|-ability|p1a: Articuno|Early Bird|[from] move: Role Play|[of] p2a: Breloom",
                              ONSET, traits=traits)
        self.assertEqual(missing_old.status, "unresolved")
        for change in (
            "|-ability|p1a: Articuno|Water Veil|Pressure|[from] move: Role Play|[of] p2: Wailord",
            "|-transform|p1a: Articuno|p2: Wailord",
            "|-activate|p1a: Articuno|Skill Swap|||[of] p2: Wailord",
            "|-activate|p1a: Articuno|move: Trick|[of] p2: Wailord",
        ):
            with self.subTest(change=change):
                self.assertEqual(qualify(change, ONSET).reason, "unproved-active-mutation-participants")

    def test_gen3_copy_restrictions_do_not_exclude_legal_trace_or_transform(self):
        swap = "|-activate|p1a: Articuno|Skill Swap|||[of] p2a: Breloom"
        role = "|-ability|p1a: Articuno|Wonder Guard|Pressure|[from] move: Role Play|[of] p2a: Breloom"
        traits = dict(TRAITS, **{"p2:breloom": SleepTraits("Wonder Guard")})
        self.assertEqual(qualify(swap, ONSET, traits=traits).evidence_probability, 0)
        self.assertEqual(qualify(role, ONSET, traits=traits).evidence_probability, 0)
        same = "|-ability|p1a: Articuno|Pressure|Pressure|[from] move: Role Play|[of] p2a: Breloom"
        self.assertEqual(qualify(same, ONSET,
            traits=dict(TRAITS, **{"p2:breloom": SleepTraits("Pressure")})).evidence_probability, 0)
        trace = "|-ability|p1a: Articuno|Wonder Guard|Trace|[from] ability: Trace|[of] p2a: Breloom"
        self.assertEqual(qualify(trace, ONSET,
            traits=dict(traits, **{"p1:articuno": SleepTraits("Trace")})).status, "compatible")
        self.assertEqual(qualify("|-transform|p1a: Articuno|p2a: Breloom", ONSET,
                                 traits=traits).status, "compatible")

    def test_known_shed_skin_no_cures_have_witnesses_but_bench_has_no_check(self):
        row = qualify(ONSET, "|upkeep", "|turn|2", "|upkeep", traits=actor("Shed Skin"))
        self.assertEqual(row.evidence_probability, Fraction(67, 100) ** 2)
        self.assertEqual(sum(w.kind == "stable-shed-skin-no-cure" for w in row.witnesses), 2)
        bench = qualify(ONSET, "|switch|p1a: Skarmory|Skarmory|100/100", "|upkeep",
                        traits=actor("Shed Skin"))
        self.assertEqual(bench.evidence_probability, Fraction(1))

    def test_copied_residual_membership_unknown_is_not_silent_non_cure(self):
        traits = dict(actor("Trace"), **{"p2:breloom": SleepTraits("Shed Skin")})
        trace = "|-ability|p1a: Articuno|Shed Skin|Trace|[from] ability: Trace|[of] p2a: Breloom"
        unknown = qualify(trace, ONSET, "|upkeep", traits=traits)
        self.assertEqual((unknown.status, unknown.evidence_probability), ("unresolved", None))
        self.assertEqual(unknown.reason, "copied-residual-membership")
        stable = qualify(trace, "|turn|2", ONSET, "|upkeep", traits=traits)
        self.assertEqual(stable.evidence_probability, Fraction(67, 100))

    def test_yawn_late_onset_skips_only_its_initial_residual(self):
        prefix = ("|-start|p1a: Articuno|move: Yawn|[of] p2a: Breloom",
                  "|upkeep", "|-end|p1a: Articuno|move: Yawn|[silent]", ONSET)
        first = qualify(*prefix, "|upkeep", traits=actor("Shed Skin"))
        second = qualify(*prefix, "|upkeep", "|turn|3", "|upkeep", traits=actor("Shed Skin"))
        self.assertEqual(first.evidence_probability, Fraction(1))
        self.assertEqual(second.evidence_probability, Fraction(67, 100))
        impossible = qualify(*prefix, "|-activate|p1a: Articuno|ability: Shed Skin",
                             "|-curestatus|p1a: Articuno|slp|[msg]", traits=actor("Shed Skin"))
        self.assertEqual(impossible.status, "unresolved")

    def test_yawn_immunity_is_checked_before_a_later_ability_swap(self):
        for ability in ("Insomnia", "Vital Spirit"):
            with self.subTest(ability=ability):
                traits = dict(actor(ability), **{"p2:breloom": SleepTraits("Pressure")})
                row = qualify("|-start|p1a: Articuno|move: Yawn|[of] p2a: Breloom",
                    "|-activate|p1a: Articuno|Skill Swap|||[of] p2a: Breloom",
                    "|-end|p1a: Articuno|move: Yawn|[silent]", ONSET, traits=traits)
                self.assertEqual((row.status, row.evidence_probability), ("incompatible", Fraction(0)))
                self.assertEqual(row.reason, "public-sleep-immunity-contradiction")
        possible = dict(actor("Early Bird"), **{"p2:breloom": SleepTraits("Pressure")})
        row = qualify("|-start|p1a: Articuno|move: Yawn|[of] p2a: Breloom",
            "|-activate|p1a: Articuno|Skill Swap|||[of] p2a: Breloom",
            "|-end|p1a: Articuno|move: Yawn|[silent]", ONSET, traits=possible)
        self.assertEqual(row.status, "compatible")

    def test_explicit_shed_skin_cure_and_untagged_cure_are_distinct(self):
        proved = qualify(ONSET, "|-activate|p1a: Articuno|ability: Shed Skin",
                         "|-curestatus|p1a: Articuno|slp|[msg]", traits=actor("Shed Skin"))
        self.assertEqual((proved.status, proved.evidence_probability), ("compatible", Fraction(33, 100)))
        unknown = qualify(ONSET, "|-curestatus|p1a: Articuno|slp|[msg]")
        self.assertEqual((unknown.status, unknown.evidence_probability), ("unresolved", None))

    def test_natural_cure_and_missing_switch_cure_are_deterministic(self):
        cure = "|-curestatus|p1a: Articuno|slp|[from] ability: Natural Cure"
        self.assertEqual(qualify(ONSET, cure, traits=actor("Natural Cure")).evidence_probability, 1)
        self.assertEqual(qualify(ONSET, cure).evidence_probability, 0)
        missing = qualify(ONSET, "|switch|p1a: Skarmory|Skarmory|100/100", traits=actor("Natural Cure"))
        self.assertEqual((missing.status, missing.evidence_probability), ("incompatible", Fraction(0)))

    def test_berry_possession_consumption_and_unfinished_cure(self):
        for item in ("Lum Berry", "Chesto Berry"):
            with self.subTest(item=item):
                absent = qualify(ONSET, "|upkeep", traits=actor(item=item))
                self.assertEqual(absent.evidence_probability, 0)
                cured = qualify(ONSET, f"|-enditem|p1a: Articuno|{item}|[eat]",
                                "|-curestatus|p1a: Articuno|slp|[msg]", traits=actor(item=item))
                self.assertEqual(cured.evidence_probability, 1)
                unfinished = qualify(ONSET, f"|-enditem|p1a: Articuno|{item}|[eat]", traits=actor(item=item))
                self.assertEqual((unfinished.status, unfinished.evidence_probability), ("unresolved", None))

    def test_pre_onset_trick_is_bound_to_both_original_items(self):
        traits = dict(actor(item="Leftovers"), **{"p2:breloom": SleepTraits("Effect Spore", "Lum Berry")})
        transfer = ("|-activate|p1a: Articuno|move: Trick|[of] p2a: Breloom",
                    "|-item|p2a: Breloom|Leftovers|[from] move: Trick",
                    "|-item|p1a: Articuno|Lum Berry|[from] move: Trick")
        row = qualify(*transfer, ONSET, "|-enditem|p1a: Articuno|Lum Berry|[eat]",
                      "|-curestatus|p1a: Articuno|slp|[msg]", traits=traits)
        self.assertEqual(row.evidence_probability, 1)
        bad = qualify(*transfer[:-1], "|-item|p1a: Articuno|Choice Band|[from] move: Trick", ONSET, traits=traits)
        self.assertEqual(bad.evidence_probability, 0)

    def test_team_cures_and_soundproof_exclusion_are_not_unconditional_cures(self):
        heal = "|-activate|p1a: Articuno|move: Heal Bell"
        silent = "|-curestatus|p1: Articuno|slp|[silent]"
        self.assertEqual(qualify(ONSET, heal).evidence_probability, 0)
        self.assertEqual(qualify(ONSET, heal, traits=actor("Soundproof")).evidence_probability, 1)
        self.assertEqual(qualify(ONSET, heal, silent).evidence_probability, 1)
        self.assertEqual(qualify(ONSET, heal, silent, traits=actor("Soundproof")).evidence_probability, 0)
        self.assertEqual(qualify(ONSET, "|-cureteam|p1a: Articuno|[from] move: Aromatherapy").evidence_probability, 1)

    def test_sticky_hold_blocks_target_item_removal_not_owners_trick(self):
        transfer = ("|-activate|p1a: Articuno|move: Trick|[of] p2a: Breloom",
                    "|-item|p2a: Breloom|Leftovers|[from] move: Trick",
                    "|-item|p1a: Articuno|Choice Band|[from] move: Trick")
        target_sticky = dict(actor(item="Leftovers"),
                             **{"p2:breloom": SleepTraits("Sticky Hold", "Choice Band")})
        self.assertEqual(qualify(*transfer, ONSET, traits=target_sticky).evidence_probability, 0)
        source_sticky = dict(actor("Sticky Hold", "Leftovers"),
                             **{"p2:breloom": SleepTraits("Effect Spore", "Choice Band")})
        self.assertEqual(qualify(*transfer, ONSET, traits=source_sticky).status, "compatible")
        removal = "|-enditem|p1a: Articuno|Leftovers|[from] move: Knock Off|[of] p2a: Breloom"
        self.assertEqual(qualify(removal, ONSET, traits=source_sticky).evidence_probability, 0)

    def test_unsupported_mutations_are_unknown_not_rejected_team_mass(self):
        for change in ("|-activate|p1a: Articuno|Skill Swap|||[of] p2a: Breloom",
                       "|-transform|p1a: Articuno|p2a: Breloom",
                       "|-endability|p1a: Articuno|Pressure",
                       "|-item|p1a: Articuno|Leftovers|[from] move: Thief|[of] p2a: Breloom"):
            with self.subTest(change=change):
                row = qualify(ONSET, change)
                self.assertEqual((row.status, row.evidence_probability), ("unresolved", None))

    def test_closed_unselected_episode_is_not_added_to_sleep_likelihood(self):
        lines = OPEN + (ONSET, "|-curestatus|p1a: Articuno|slp|[msg]", ONSET, CANT)
        row = qualify(lines=lines, index=1)
        self.assertEqual(row.evidence_probability, 1)
        self.assertEqual(qualify(lines=lines, index=0).status, "unresolved")

    @requires_showdown()
    def test_closed_ability_universe_matches_pinned_gen3_dex(self):
        program = """const {Dex}=require(process.argv[1]+'/dist/sim');
const abilities=Dex.mod('gen3').abilities.all().filter(a=>a.num>0&&a.gen<=3);
process.stdout.write(JSON.stringify({all:abilities.map(a=>a.id),
swap:abilities.filter(a=>a.flags.failskillswap).map(a=>a.id),
role:abilities.filter(a=>a.flags.failroleplay).map(a=>a.id),
suppress:abilities.filter(a=>a.flags.cantsuppress).map(a=>a.id)}));"""
        result = subprocess.run(["node", "-e", program, str(showdown_root())],
                                check=True, capture_output=True, text=True, timeout=30)
        result = json.loads(result.stdout)
        self.assertEqual(_ABILITIES, frozenset(result["all"]))
        self.assertEqual({key: result[key] for key in ("swap", "role", "suppress")},
                         {"swap": ["wonderguard"], "role": ["wonderguard"], "suppress": []})


@requires_showdown()
class ReferenceSleepEvidenceServerTests(unittest.TestCase):
    def fixture(self, **kwargs):
        return history_fixture.PublicSleepServerTests.fixture(self, **kwargs)

    def sleeper(self, live):
        return history_fixture.PublicSleepServerTests.sleeper(self, live)

    def test_authored_public_only_qualifier_contains_oracle_timer(self):
        for ability in ("Pressure", "Early Bird", "Shed Skin"):
            with self.subTest(ability=ability):
                live = self.fixture(ability=ability)
                live.step({"p1": 0, "p2": 0})
                state = live.public_materialization_state("p1")
                ledger = public_sleep_history(state.replay.public_lines, at_action_boundary=True)
                result = qualify_sleep_episode(ledger, actor(ability), episode_index=0, at_action_boundary=True)
                oracle = self.sleeper(live)
                self.assertEqual(result.status, "compatible", result)
                if oracle["status"] == "slp":
                    self.assertTrue(any((row.time, row.skipped_time) ==
                        (oracle["statusState"]["time"], oracle["statusState"]["skippedTime"])
                        for row in result.posterior.outcomes))
                else:
                    self.assertTrue(all(not row.asleep for row in result.posterior.outcomes))


if __name__ == "__main__":
    unittest.main()
