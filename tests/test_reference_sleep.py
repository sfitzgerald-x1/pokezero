"""Authored finite-history contracts; no stopped battle or private timer input."""
from collections import Counter
from fractions import Fraction
import random
import json
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.reference_sleep import (
    SleepAttempt, SleepCure, SleepHypothesis, SleepReentry, condition_sleep,
)


ORACLE_PROGRAM = r'''
const {Battle} = require(process.argv[1] + '/dist/sim');
const patterns = [
 ['ordinary','ordinary','ordinary','ordinary'],
 ['usable','out','idle','in','ordinary','ordinary'],
 ['usable','usable','out','in','ordinary','ordinary'],
 ['usable','ordinary','out','in'],
 ['snore','out','in','ordinary'],
 ['out','idle','idle','in','ordinary'],
 ['usable','to_bird','usable','out','in','ordinary'],
];
const results = [];
for (const ability of ['Pressure','Early Bird']) for (const initial of [2,3,4,5]) {
 for (const pattern of patterns) {
  const battle = new Battle({formatid:'gen3randombattle',seed:[1,2,3,4],strictChoices:true});
  battle.setPlayer('p1',{team:[{species:'Articuno',ability,moves:['Splash','Sleep Talk','Snore']},
    {species:'Skarmory',ability:'Keen Eye',moves:['Splash']}]});
  battle.setPlayer('p2',{team:[{species:'Wailord',ability:'Water Veil',moves:['Splash']}]});
  const pokemon = battle.p1.active[0];
  pokemon.setStatus('slp',battle.p2.active[0],battle.dex.moves.get('spore'));
  // AUTHORED control: enumerate every possible initial duration, never recover
  // a stopped game's timer. All subsequent state transitions are real Gen 3.
  pokemon.statusState.time = initial;
  battle.makeRequest('move');
  const steps = [];
  for (const action of pattern) {
   let event = null;
   if (action === 'to_bird') {
    pokemon.setAbility('earlybird');
   } else {
    const early = pokemon.hasAbility('earlybird');
    const choice = action === 'in' || action === 'out' ? 'switch 2' :
      action === 'usable' ? 'move sleeptalk' : action === 'snore' ? 'move snore' : 'move splash';
    battle.makeChoices(choice,'move splash');
    if (action === 'in') event = {kind:'reentry'};
    else if (['ordinary','usable','snore'].includes(action)) event = {
      kind:'attempt', early, usable:action !== 'ordinary', asleep:pokemon.status === 'slp',
    };
   }
   steps.push({action,event,time:pokemon.status === 'slp' ? pokemon.statusState.time : 0,
     skipped:pokemon.status === 'slp' ? pokemon.statusState.skippedTime : 0});
   if (pokemon.status !== 'slp') break;
  }
  results.push({ability,initial,pattern,steps});
  battle.destroy();
 }
}
process.stdout.write(JSON.stringify(results));
'''


class ReferenceSleepContractTests(unittest.TestCase):
    def check_oracle(self, rows):
        self.assertEqual(len(rows), 56)
        for row in rows:
            events = []
            for step in row["steps"]:
                event = step["event"]
                if event is not None:
                    events.append(SleepReentry() if event["kind"] == "reentry" else
                                  SleepAttempt(event["early"], event["usable"], event["asleep"]))
                posterior = condition_sleep((SleepHypothesis("known", Fraction(1), tuple(events)),),
                                            kind="induced")
                own = [value for value in posterior.outcomes if value.initial_time == row["initial"]]
                self.assertEqual(len(own), 1, (row, step))
                self.assertEqual((own[0].time, own[0].skipped_time),
                                 (step["time"], step["skipped"]), (row, step))

    @requires_showdown("requires pinned Gen 3 server for authored sleep transitions")
    def test_gen3_server_all_durations_abilities_wake_and_switch_refunds(self):
        result = subprocess.run(["node", "-e", ORACLE_PROGRAM, str(showdown_root())],
                                check=True, capture_output=True, text=True, timeout=30)
        self.check_oracle(json.loads(result.stdout))

    def test_initial_induced_distribution_and_exact_rest_are_distinct(self):
        row = SleepHypothesis("known", Fraction(1), ())
        induced = condition_sleep((row,), kind="induced")
        self.assertEqual([(r.initial_time, r.time, r.probability) for r in induced.outcomes],
                         [(i, i, Fraction(1, 4)) for i in range(2, 6)])
        rest = condition_sleep((row,), kind="rest")
        self.assertEqual([(r.time, r.probability) for r in rest.outcomes], [(3, Fraction(1))])

    def test_joint_ability_weights_include_survival_likelihood(self):
        rows = (SleepHypothesis("ordinary", Fraction(1, 2), (SleepAttempt(False, False, True),) * 2),
                SleepHypothesis("earlybird", Fraction(1, 2), (SleepAttempt(True, False, True),) * 2))
        result = condition_sleep(rows, kind="induced")
        weights = Counter()
        for row in result.outcomes:
            weights[row.hypothesis] += row.probability
        self.assertEqual(weights, {"ordinary": Fraction(3, 4), "earlybird": Fraction(1, 4)})
        self.assertEqual(result.evidence_probability, Fraction(1, 2))

    def test_intermediate_survival_cannot_be_resurrected_by_a_later_refund(self):
        history = (SleepAttempt(True, True, True), SleepReentry())
        result = condition_sleep((SleepHypothesis("bird", Fraction(1), history),), kind="induced")
        self.assertEqual([r.initial_time for r in result.outcomes], [3, 4, 5])
        self.assertEqual([r.time for r in result.outcomes], [2, 3, 4])
        self.assertTrue(all(r.skipped_time == 0 for r in result.outcomes))

    def test_only_sleep_usable_trailing_run_is_refunded_and_bench_does_not_tick(self):
        history = (SleepAttempt(False, True, True), SleepAttempt(False, False, True),
                   SleepReentry(), SleepReentry())
        result = condition_sleep((SleepHypothesis("known", Fraction(1), history),), kind="induced")
        self.assertEqual([(r.initial_time, r.time, r.skipped_time) for r in result.outcomes],
                         [(3, 1, 0), (4, 2, 0), (5, 3, 0)])

    def test_natural_wake_filters_but_explicit_cure_does_not_invent_a_tick(self):
        wake = condition_sleep((SleepHypothesis("bird", Fraction(1),
                               (SleepAttempt(True, False, False),)),), kind="induced")
        self.assertEqual([(r.initial_time, r.time) for r in wake.outcomes], [(2, 0)])
        cure = condition_sleep((SleepHypothesis("known", Fraction(1), (SleepCure(),)),), kind="induced")
        self.assertEqual([r.initial_time for r in cure.outcomes], [2, 3, 4, 5])
        self.assertTrue(all(not r.asleep for r in cure.outcomes))

    def test_ability_at_each_attempt_is_explicit_not_assumed_constant(self):
        result = condition_sleep((SleepHypothesis("publicchange", Fraction(1),
            (SleepAttempt(False, False, True), SleepAttempt(True, False, True))),), kind="induced")
        self.assertEqual([r.initial_time for r in result.outcomes], [4, 5])

    def test_impossible_unresolved_and_malformed_histories_refuse(self):
        with self.assertRaises(ReferenceRefusal):
            condition_sleep((SleepHypothesis("known", Fraction(1),
                (SleepAttempt(False, False, True),) * 5),), kind="induced")
        with self.assertRaises(ReferenceRefusal):
            condition_sleep((SleepHypothesis("known", Fraction(1),
                (SleepCure(), SleepReentry())),), kind="induced")
        for bad in (None, 0, "unknown"):
            with self.subTest(bad=bad), self.assertRaises(ReferenceRefusal):
                SleepAttempt(bad, False, True)
        for mass in (0, .5, True, Fraction(0), Fraction(-1)):
            with self.subTest(mass=mass), self.assertRaises(ReferenceRefusal):
                SleepHypothesis("known", mass, ())
        row = SleepHypothesis("known", Fraction(1), ())
        for roster, kind in (((), "induced"), ((row, row), "induced"), ((row,), "unknown")):
            with self.assertRaises(ReferenceRefusal):
                condition_sleep(roster, kind=kind)

    def test_draw_replays_exact_integer_probability_bins(self):
        posterior = condition_sleep((SleepHypothesis("known", Fraction(1), ()),), kind="induced")
        class IntegerProbe:
            def __init__(self, value): self.value = value
            def randrange(self, maximum):
                self.maximum = maximum
                return self.value
        for bin_number in range(4):
            rng = IntegerProbe(bin_number)
            self.assertEqual(posterior.draw(rng).initial_time, bin_number + 2)
            self.assertEqual(rng.maximum, 4)
        first, second = random.Random(17), random.Random(17)
        self.assertEqual([posterior.draw(first) for _ in range(20)],
                         [posterior.draw(second) for _ in range(20)])


if __name__ == "__main__":
    unittest.main()
