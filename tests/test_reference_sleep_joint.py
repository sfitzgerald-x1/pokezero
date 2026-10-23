"""Joint proposal law, bounded refusal and public-only bridge integration."""
from dataclasses import replace
from fractions import Fraction
import random
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference import Evaluation
from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
from pokezero.mcts_eval.reference_sleep_evidence import SleepTraits
from pokezero.mcts_eval.reference_sleep_history import public_sleep_history
from pokezero.mcts_eval.reference_sleep_joint import joint_sleep_proposal, opening_sleep_traits
from pokezero.showdown_fixture import FixturePokemon, pack_team
from tests.test_reference_sleep_evidence import TRAITS, actor
from tests.test_reference_sleep_history import OPEN, ONSET, CANT
from tests.test_paper_reference import ROOT, ToyWorld, searcher
from tests.test_reference_sleep_binding import run_node, snapshot, assignment


def ledger(*suffix):
    return public_sleep_history(OPEN + suffix, at_action_boundary=True)


def qualify(history, traits=TRAITS, rng=None, check=lambda: None):
    return joint_sleep_proposal(history, traits, rng or random.Random(1), check_deadline=check)


class RejectionRng(random.Random):
    def __init__(self, draw):
        super().__init__(7)
        self.draw = draw
        self.denominator = None

    def randrange(self, stop):
        self.denominator = stop
        return self.draw


class JointSleepTests(unittest.TestCase):
    def test_bridge_timer_restoration_is_atomic_and_requires_full_sleep_roster(self):
        program = r'''
const {restoreReferenceSleep: restore} = await import(process.argv[1]);
const [snapshot, rows] = JSON.parse(process.argv[2]);
const before = JSON.stringify(snapshot);
try {restore(snapshot,rows);process.stdout.write(JSON.stringify({snapshot}));}
catch(error) {process.stdout.write(JSON.stringify({error:error.message,unchanged:before===JSON.stringify(snapshot)}));}
'''
        state = snapshot()
        valid = dict(assignment(), time=2, skippedTime=1)
        result = run_node(program, json.dumps([state, [valid]]))
        timer = result["snapshot"]["sides"][0]["pokemon"][0]["statusState"]
        self.assertEqual(timer, dict(id="slp", effectOrder=0, time=2, skippedTime=1,
                                    source="[Pokemon:p2a]", target="[Pokemon:p1a]"))
        for rows in ([], [valid, valid], [dict(valid, time=True)], [dict(valid, time=0)],
                     [dict(valid, source=dict(side="p2", species="missing"))],
                     [dict(valid, kind="rest")], [dict(valid, skippedTime=4)]):
            result = run_node(program, json.dumps([state, rows]))
            self.assertIn("error", result)
            self.assertTrue(result["unchanged"])

    def test_team_likelihood_reweighting_not_per_team_timer_normalization(self):
        history = ledger(ONSET, CANT, "|upkeep", CANT, "|upkeep")
        accepted = []
        # Uniform proposal over two authored teams. Enumerating the exact
        # rejection coin proves the 3:1 joint law without statistical tolerance.
        for ability in ("Pressure", "Early Bird"):
            for coin in range(4):
                rng = RejectionRng(coin)
                assignment, receipt = qualify(history, actor(ability), rng)
                self.assertEqual(rng.denominator, 4)
                self.assertEqual(receipt["likelihood"], [3 if ability == "Pressure" else 1, 4])
                if assignment is not None:
                    accepted.append(ability)
                    self.assertGreater(assignment[0]["time"], 0)
        self.assertEqual(accepted, ["Pressure"] * 3 + ["Early Bird"])

    def test_distinct_open_episode_likelihoods_multiply_conditional_on_team(self):
        history = ledger(ONSET, "|-status|p2a: Breloom|slp", CANT,
            "|cant|p2a: Breloom|slp", "|upkeep", CANT, "|cant|p2a: Breloom|slp", "|upkeep")
        _, receipt = qualify(history, rng=RejectionRng(0))
        self.assertEqual(receipt["likelihood"], [9, 16])
        self.assertEqual(len(receipt["evidence"]), 2)

    def test_closed_episode_omits_likelihood_but_retains_trait_constraints(self):
        history = ledger(ONSET, CANT, "|-curestatus|p1a: Articuno|slp|[msg]")
        assignment, receipt = qualify(history)
        self.assertEqual((assignment, receipt["likelihood"]), ([], [1, 1]))
        _, receipt = qualify(history, actor("Insomnia"))
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["likelihood"], [0, 1])

    def test_closed_announced_cure_is_a_trait_constraint_without_weight(self):
        history = ledger(ONSET, "|-activate|p1a: Articuno|ability: Shed Skin",
                         "|-curestatus|p1a: Articuno|slp|[msg]")
        assignment, receipt = qualify(history, actor("Shed Skin"))
        self.assertEqual((assignment, receipt["likelihood"]), ([], [1, 1]))
        self.assertEqual(qualify(history)[1]["likelihood"], [0, 1])
        history = ledger(ONSET, "|-curestatus|p1a: Articuno|slp|[from] ability: Natural Cure")
        self.assertEqual(qualify(history)[1]["likelihood"], [0, 1])
        self.assertEqual(qualify(history, actor("Natural Cure"))[1]["likelihood"], [1, 1])

    def test_unrepresented_current_ability_aborts_instead_of_reverting_to_original(self):
        history = ledger("|-activate|p1a: Articuno|Skill Swap|||[of] p2a: Breloom", ONSET)
        with self.assertRaisesRegex(ReferenceRefusal, "current ability.*not represented"):
            qualify(history)

    def test_unknown_is_refusal_not_rejection(self):
        with self.assertRaisesRegex(ReferenceRefusal, "unresolved"):
            qualify(ledger(ONSET), actor("unknown"))

    def test_all_open_sleepers_sources_refunds_and_rest_are_distinct(self):
        history = ledger(ONSET, CANT, "|move|p1a: Articuno|Sleep Talk|p1a: Articuno",
            "|upkeep", "|switch|p1a: Skarmory|Skarmory|100/100",
            "|-status|p2a: Breloom|slp|[from] move: Rest")
        assignments, receipt = qualify(history, rng=RejectionRng(0))
        self.assertEqual(receipt["likelihood"], [1, 1])
        first, second = assignments
        self.assertEqual((first["skippedTime"], first["source"]), (1, dict(side="p2", species="breloom")))
        self.assertEqual((second["kind"], second["time"], second["source"]),
                         ("rest", 3, dict(side="p2", species="breloom")))

    def test_yawn_shed_skin_first_residual_does_not_get_prior_opportunity(self):
        history = ledger("|-start|p1a: Articuno|move: Yawn|[of] p2a: Breloom", "|upkeep",
            "|-end|p1a: Articuno|move: Yawn|[silent]", ONSET, "|upkeep")
        _, receipt = qualify(history, actor("Shed Skin"))
        self.assertEqual(receipt["likelihood"], [1, 1])

    def test_opening_traits_use_original_team_and_species_not_order(self):
        mon = FixturePokemon("Articuno", ("Rest",), "Pressure", "Leftovers")
        self.assertEqual(opening_sleep_traits({"p2": (mon,)}),
                         {"p2:articuno": SleepTraits("Pressure", "Leftovers")})
        with self.assertRaises(ReferenceRefusal):
            opening_sleep_traits({"p1": (mon, mon)})
        with self.assertRaises(ReferenceRefusal):
            opening_sleep_traits({"p1": (replace(mon, ability=None),)})

    def factory(self):
        factory = object.__new__(PublicRootWorldFactory)
        factory.active, factory.receipts = False, []
        factory.joint_sleep_proposal_cap, factory.deadline_at = 32, None
        factory.state = SimpleNamespace(player_id="p1")
        factory.known, factory.own_team, factory.sleep_ledger = (), (), ledger()
        draw = SimpleNamespace(packed_team_sha256="authored", known=(), unknown_party_seeds=(), team=())
        factory.sampler = SimpleNamespace(draw=Mock(return_value=draw))
        return factory

    def test_exact_cap_is_terminal_and_generator_errors_never_retry(self):
        factory = self.factory()
        with patch("pokezero.mcts_eval.paper_reference_factory.joint_sleep_proposal",
                   return_value=(None, dict(accepted=False, likelihood=[0, 1]))):
            with self.assertRaisesRegex(ReferenceRefusal, "cap exhausted"):
                factory(random.Random(1))
        self.assertEqual(factory.sampler.draw.call_count, 32)
        self.assertEqual(len(factory.receipts[0]["joint_sleep_proposals"]), 32)
        self.assertFalse(factory.active)
        factory = self.factory()
        factory.sampler.draw.side_effect = RuntimeError("authored generator failure")
        with self.assertRaisesRegex(RuntimeError, "generator failure"):
            factory(random.Random(1))
        self.assertEqual(factory.sampler.draw.call_count, 1)

    def test_expired_deadline_draws_nothing_and_releases_ownership(self):
        factory = self.factory()
        factory.bind_deadline(-1.)
        with self.assertRaises(SamplingDeadlineExceeded):
            factory(random.Random(1))
        factory.sampler.draw.assert_not_called()
        self.assertFalse(factory.active)
        self.assertEqual(factory.receipts[0]["status"], "DEADLINE")

    def test_qualifier_checks_deadline_before_any_evidence(self):
        def expired():
            raise SamplingDeadlineExceeded()
        with self.assertRaises(SamplingDeadlineExceeded):
            qualify(ledger(), check=expired)

    def test_opted_in_deadline_cancels_sampling_without_committing_work(self):
        now = [0.]
        calls = [0]
        def sample(rng):
            calls[0] += 1
            if calls[0] == 1:
                return ToyWorld()
            now[0] = 1.
            raise SamplingDeadlineExceeded()
        search = searcher()
        result = search.search_batch(ROOT, battle_id="battle-1", evaluate_root=lambda _: Evaluation((.5, .5), 0.),
            sample_world=sample, seed=1, trajectories=3, deadline_at=1., clock=lambda: now[0],
            allow_sampling_deadline=True)
        self.assertEqual((result.trajectories, result.world_draws), (1, 1))
        self.assertTrue(result.deadline_exhausted)
        now[0] = 0.
        def premature(rng):
            raise SamplingDeadlineExceeded()
        with self.assertRaisesRegex(ReferenceRefusal, "before the search deadline"):
            searcher().search_batch(ROOT, battle_id="battle-1", evaluate_root=lambda _: Evaluation((.5, .5), 0.),
                sample_world=premature, seed=1, trajectories=1, deadline_at=1., clock=lambda: now[0],
                allow_sampling_deadline=True)


@requires_showdown()
class JointSleepBridgeTests(unittest.TestCase):
    def test_public_only_ordinary_sleep_materializes_in_both_seats(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        parties = {"p1": (FixturePokemon("Articuno", ("Sleep Talk", "Splash"), "Pressure"),),
                   "p2": (FixturePokemon("Breloom", ("Spore", "Splash"), "Effect Spore"),)}
        override = BattleStartOverride(player_teams={s: pack_team(t) for s, t in parties.items()})
        with LocalShowdownEnv(config) as live, LocalShowdownEnv(config) as restored:
            live.reset_with_start_override(seed=17, start_override=override)
            live.step({"p1": 1, "p2": 0})
            for subject in ("p1", "p2"):
                state = live.public_materialization_state(subject)
                history = public_sleep_history(tuple(state.replay.public_lines), at_action_boundary=True)
                assignments, _ = qualify(history, opening_sleep_traits(parties))
                self.assertIsNotNone(assignments)
                restored.materialize_public_world(state=state, start_override=override, seed=19,
                                                  reference_joint_sleep=assignments)
                self.assertEqual(restored.observe(subject).legal_action_mask,
                                 live.observe(subject).legal_action_mask)
                # The restored branch is usable, not merely a serialized timer.
                restored.step({"p1": 0, "p2": 1})
                self.assertTrue(any("|cant|p1a: Articuno|slp" in line
                                    for line in restored.protocol_lines))


if __name__ == "__main__":
    unittest.main()
