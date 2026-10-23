"""Authored public Trace boundaries; no historical root or game is replayed."""
import json
from pathlib import Path
import random
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.reference_sleep_evidence import SleepTraits
from pokezero.mcts_eval.reference_sleep_history import public_sleep_history
from pokezero.mcts_eval.reference_sleep_joint import joint_sleep_proposal, opening_sleep_traits
from pokezero.showdown_fixture import FixturePokemon, pack_team


HELPER = Path(__file__).resolve().parents[1] / "scripts/battle_bridge_reference_trace.mjs"
OPEN = ("|gametype|singles", "|gen|3", "|start",
        "|switch|p1a: Porygon2|Porygon2|100/100",
        "|switch|p2a: Lapras|Lapras|100/100")
COPY = "|-ability|p1a: Porygon2|Water Absorb|Trace|[from] ability: Trace|[of] p2a: Lapras"
TRAITS = {"p1:porygon2": SleepTraits("Trace"), "p1:abra": SleepTraits("Trace"),
          "p1:skarmory": SleepTraits("Keen Eye"), "p2:lapras": SleepTraits("Water Absorb"),
          "p2:jolteon": SleepTraits("Volt Absorb")}


def qualify(suffix=(), *, traits=None, opt=True, rng=None):
    ledger = public_sleep_history(OPEN + suffix, at_action_boundary=True)
    return joint_sleep_proposal(ledger, TRAITS if traits is None else traits, rng or random.Random(8),
                               check_deadline=lambda: None, restore_public_trace=opt)


def run_node(program, *args):
    result = subprocess.run(["node", "--input-type=module", "--eval", program, str(HELPER), *args],
                            check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


class PublicTraceProofTests(unittest.TestCase):
    def test_default_remains_strict_and_opt_in_binds_both_original_and_current(self):
        with self.assertRaisesRegex(ReferenceRefusal, "not represented"):
            qualify((COPY,), opt=False)
        assignment, receipt = qualify((COPY,))
        self.assertEqual(assignment, [])
        self.assertEqual(receipt["likelihood"], [1, 1])
        self.assertEqual(receipt["trace_assignments"], [
            dict(side="p1", species="abra", active=False, ability="trace"),
            dict(side="p1", species="porygon2", active=True, ability="waterabsorb")])

    def test_donor_switch_and_faint_do_not_change_the_copy(self):
        for suffix in (("|switch|p2a: Jolteon|Jolteon|100/100",), ("|faint|p2a: Lapras",)):
            with self.subTest(suffix=suffix):
                rows = qualify((COPY,) + suffix)[1]["trace_assignments"]
                self.assertEqual(rows[1]["ability"], "waterabsorb")

    def test_holder_reentry_without_copy_refuses_instead_of_using_stale_acquisition(self):
        with self.assertRaisesRegex(ReferenceRefusal, "stateless public history"):
            qualify((COPY, "|switch|p1a: Skarmory|Skarmory|100/100",
                     "|switch|p1a: Porygon2|Porygon2|100/100"))

    def test_acquisition_from_an_already_fainted_donor_refuses(self):
        with self.assertRaisesRegex(ReferenceRefusal, "stateless public history"):
            qualify(("|faint|p2a: Lapras", COPY))

    def test_missing_earlier_acquisition_cannot_hide_stateful_history(self):
        traits = dict(TRAITS, **{"p2:lapras": SleepTraits("Truant")})
        suffixes = (("|faint|p1a: Porygon2",), ("|switch|p1a: Skarmory|Skarmory|100/100",),
            ("|switch|p1a: Skarmory|Skarmory|100/100", "|switch|p2a: Jolteon|Jolteon|100/100",
             "|switch|p1a: Porygon2|Porygon2|100/100",
             "|-ability|p1a: Porygon2|Volt Absorb|Trace|[from] ability: Trace|[of] p2a: Jolteon"))
        for suffix in suffixes:
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ReferenceRefusal, "stateless public history"):
                qualify(suffix, traits=traits)

    def test_new_entry_can_copy_a_different_donor(self):
        rows = qualify((COPY, "|switch|p1a: Skarmory|Skarmory|100/100",
            "|switch|p2a: Jolteon|Jolteon|100/100", "|switch|p1a: Porygon2|Porygon2|100/100",
            "|-ability|p1a: Porygon2|Volt Absorb|Trace|[from] ability: Trace|[of] p2a: Jolteon"))[1]
        self.assertEqual(rows["trace_assignments"][1]["ability"], "voltabsorb")

    def test_unsupported_old_copy_refuses_even_after_supported_reentry(self):
        traits = dict(TRAITS, **{"p2:lapras": SleepTraits("Truant")})
        suffix = (COPY.replace("Water Absorb", "Truant"),
            "|switch|p1a: Skarmory|Skarmory|100/100", "|switch|p2a: Jolteon|Jolteon|100/100",
            "|switch|p1a: Porygon2|Porygon2|100/100",
            "|-ability|p1a: Porygon2|Volt Absorb|Trace|[from] ability: Trace|[of] p2a: Jolteon")
        with self.assertRaisesRegex(ReferenceRefusal, "stateless public history"):
            qualify(suffix, traits=traits)

    def test_role_play_transform_and_skill_swap_stay_unsupported(self):
        suffixes = (
            (COPY.replace("[from] ability: Trace", "[from] move: Role Play"),),
            ("|-transform|p1a: Porygon2|p2a: Lapras|[from] move: Transform",),
            ("|-activate|p1a: Porygon2|Skill Swap|||[of] p2a: Lapras",),
        )
        for suffix in suffixes:
            with self.subTest(suffix=suffix), self.assertRaises(ReferenceRefusal):
                qualify(suffix)

    def test_incompatible_donor_is_zero_not_a_trace_certificate(self):
        assignments, receipt = qualify((COPY,), traits=dict(TRAITS, **{"p2:lapras": SleepTraits("Pressure")}))
        self.assertIsNone(assignments)
        self.assertEqual(receipt["likelihood"], [0, 1])
        self.assertNotIn("trace_assignments", receipt)

    def test_non_trace_mutation_still_refuses_and_never_draws_replacement(self):
        suffix = ("|-activate|p1a: Porygon2|Skill Swap|||[of] p2a: Lapras",)
        with self.assertRaises(ReferenceRefusal):
            qualify(suffix)
        suffix = (COPY, "|switch|p1a: Skarmory|Skarmory|100/100",
                  "|-ability|p1a: Skarmory|Water Absorb|Keen Eye|[from] move: Role Play|[of] p2a: Lapras")
        with self.assertRaisesRegex(ReferenceRefusal, "not represented"):
            qualify(suffix)

    def test_trace_proof_consumes_no_extra_rng_or_sleep_likelihood(self):
        a, b = random.Random(123), random.Random(123)
        traits = dict(TRAITS, **{"p1:porygon2": SleepTraits("Pressure"), "p1:abra": SleepTraits("Pressure")})
        old = qualify((), opt=False, rng=a, traits=traits)
        new = qualify((), rng=b, traits=traits)
        self.assertEqual(old[0], new[0])
        self.assertEqual(old[1]["likelihood"], new[1]["likelihood"])
        self.assertEqual(a.getstate(), b.getstate())


APPLY = r'''
const {restoreReferenceTrace: restore} = await import(process.argv[1]);
const [snapshot,rows]=JSON.parse(process.argv[2]),before=JSON.stringify(snapshot);
try {const count=restore(snapshot,rows);process.stdout.write(JSON.stringify({count,snapshot}));}
catch(error) {process.stdout.write(JSON.stringify({error:error.message,unchanged:JSON.stringify(snapshot)===before}));}
'''


def snapshot():
    def mon(species, ability, position, active=False):
        return dict(set=dict(species=species, ability=ability), baseAbility=ability.lower(),
                    ability="truant" if ability == "Trace" else ability.lower(),
                    abilityState=dict(id="truant", effectOrder=2, target="wrong", started=True),
                    position=position, isActive=active, hp=100, fainted=False, truantTurn=True)
    return dict(formatid="gen3randombattle", effectOrder=10, sides=[
        dict(id="p1", active=["[Pokemon:p1a]"], pokemon=[mon("Porygon2", "Trace", 0, True),
                                                             mon("Abra", "Trace", 1)]),
        dict(id="p2", active=["[Pokemon:p2a]"], pokemon=[mon("Lapras", "Water Absorb", 0, True)])])


def rows():
    return [dict(side="p1", species="abra", active=False, ability="trace"),
            dict(side="p1", species="porygon2", active=True, ability="waterabsorb")]


class TraceBindingTests(unittest.TestCase):
    def test_complete_roster_overwrites_warm_abilities_and_persistent_truant_state(self):
        result = run_node(APPLY, json.dumps([snapshot(), rows()]))
        self.assertEqual(result["count"], 2)
        state = result["snapshot"]
        active, bench = state["sides"][0]["pokemon"]
        self.assertEqual((active["ability"], bench["ability"]), ("waterabsorb", "trace"))
        for index, mon in enumerate((active, bench)):
            self.assertEqual(mon["set"]["ability"], "Trace")
            self.assertEqual(mon["baseAbility"], "trace")
            self.assertFalse(mon["truantTurn"])
            self.assertEqual(mon["abilityState"], dict(id=mon["ability"], target=f"[Pokemon:p1{'ab'[index]}]",
                                                     effectOrder=10 if index == 0 else 0))
        self.assertEqual(state["effectOrder"], 11)

    def test_bad_late_row_missing_roster_and_unsupported_values_are_atomic(self):
        for roster in (rows()[:1], rows() + rows()[:1], [rows()[0], dict(rows()[1], ability="truant")],
                       [dict(rows()[0], active=True), rows()[1]],
                       [rows()[0], dict(rows()[1], species="Porygon2")],
                       [rows()[0], dict(rows()[1], active=1)]):
            with self.subTest(roster=roster):
                result = run_node(APPLY, json.dumps([snapshot(), roster]))
                self.assertIn("error", result)
                self.assertTrue(result["unchanged"])

    def test_bad_snapshot_identity_and_counter_are_atomic(self):
        variants = []
        for field, value in (("baseAbility", "pressure"), ("position", 1), ("isActive", False)):
            state = snapshot(); state["sides"][0]["pokemon"][0][field] = value; variants.append(state)
        state = snapshot(); state["effectOrder"] = 2**53 - 1; variants.append(state)
        for state in variants:
            with self.subTest(state=state):
                result = run_node(APPLY, json.dumps([state, rows()]))
                self.assertIn("error", result)
                self.assertTrue(result["unchanged"])


ORACLE = r'''
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {Battle}=require(process.argv[2]+'/dist/sim');
const {State}=require(process.argv[2]+'/dist/sim/state');
const {restoreReferenceTrace:restore}=await import(process.argv[1]);
const result=[];
for(const seat of ['p1','p2']) for(const [ability,attack] of [
 ['Water Absorb','watergun'],['Volt Absorb','thundershock'],['Levitate','earthquake'],['Pressure','splash']]) {
 const foe=seat==='p1'?'p2':'p1', b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4],strictChoices:true});
 b.setPlayer(seat,{team:[{species:'Porygon2',ability:'Trace',moves:['Splash']},
   {species:'Abra',ability:'Trace',moves:['Splash']}]});
 b.setPlayer(foe,{team:[{species:'Lapras',ability,moves:[attack]},
   {species:'Wailord',ability:'Water Veil',moves:['Splash']}]});
 b[seat].active[0].hp-=40;
 const clean=State.serializeBattle(b), dirty=structuredClone(clean);
 const side=dirty.sides[seat==='p1'?0:1];
 for(const p of side.pokemon) {p.ability='truant';p.abilityState={id:'truant',effectOrder:2,target:'[Pokemon:'+seat+'a]'};p.truantTurn=true;}
 restore(dirty,[{side:seat,species:'porygon2',active:true,ability:ability.toLowerCase().replace(/ /g,'')},
                {side:seat,species:'abra',active:false,ability:'trace'}]);
 const expected=State.deserializeBattle(clean),actual=State.deserializeBattle(dirty);
 expected.restart(()=>{});actual.restart(()=>{});
 const play=battle=>battle.makeChoices(...(seat==='p1'?['move splash','move '+attack]:['move '+attack,'move splash']));
 play(expected);play(actual);
 const state=battle=>({hp:battle[seat].active[0].hp,ability:battle[seat].active[0].ability,
    base:battle[seat].active[0].baseAbility,pp:battle[foe].active[0].moveSlots[0].pp,
    truant:battle[seat].active[0].truantTurn});
 const first=[state(actual),state(expected)];
 // On holder switch, the never-seen bench Trace copies the actual donor; it
 // must not retain the warm template's copied Truant or persistent phase.
 const switchIn=battle=>battle.makeChoices(...(seat==='p1'?['switch 2','move '+attack]:['move '+attack,'switch 2']));
 switchIn(actual);switchIn(expected);
 result.push({seat,ability,first,second:[state(actual),state(expected)],
   target:actual[seat].active[0].abilityState.target===actual[seat].active[0]});
 b.destroy();actual.destroy();expected.destroy();
}
process.stdout.write(JSON.stringify(result));
'''


@requires_showdown()
class TraceOracleTests(unittest.TestCase):
    def test_clean_gen3_oracle_behavior_for_four_abilities_and_both_seats(self):
        results = run_node(ORACLE, str(showdown_root()))
        self.assertEqual(len(results), 8)
        for row in results:
            with self.subTest(seat=row["seat"], ability=row["ability"]):
                self.assertEqual(*row["first"])
                self.assertEqual(*row["second"])
                self.assertTrue(row["target"])

    def test_public_materialization_round_trip_in_both_actor_seats(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        parties = {"p1": (FixturePokemon("Porygon2", ("Splash",), "Trace"),),
                   "p2": (FixturePokemon("Lapras", ("Water Gun",), "Water Absorb"),)}
        override = BattleStartOverride(player_teams={side: pack_team(team) for side, team in parties.items()})
        with LocalShowdownEnv(config) as live, LocalShowdownEnv(config) as restored:
            live.reset_with_start_override(seed=71, start_override=override)
            for actor in ("p1", "p2"):
                state = live.public_materialization_state(actor)
                assignments, receipt = joint_sleep_proposal(public_sleep_history(
                    tuple(state.replay.public_lines), at_action_boundary=True), opening_sleep_traits(parties),
                    random.Random(7), check_deadline=lambda: None, restore_public_trace=True)
                restored.materialize_public_world(state=state, start_override=override, seed=19,
                    reference_joint_sleep=assignments, reference_trace=receipt["trace_assignments"])
                self.assertEqual(live.observe(actor).legal_action_mask, restored.observe(actor).legal_action_mask)
                restored.step({"p1": 0, "p2": 0})
                self.assertTrue(any("[from] ability: Water Absorb" in line for line in restored.protocol_lines))
