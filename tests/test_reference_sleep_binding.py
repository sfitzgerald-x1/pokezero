"""Authored source-pointer restoration, not sleep-sampler qualification."""
import json
from pathlib import Path
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root


HELPER = Path(__file__).resolve().parents[1] / "scripts/battle_bridge_reference_sleep.mjs"


def run_node(program, *args):
    result = subprocess.run(["node", "--input-type=module", "--eval", program,
                             str(HELPER), *args], check=True, capture_output=True,
                            text=True, timeout=30)
    return json.loads(result.stdout)


SYNTHETIC_PROGRAM = r'''
const {bindReferenceSleepSources: bind} = await import(process.argv[1]);
const [snapshot, assignments] = JSON.parse(process.argv[2]);
const before = JSON.stringify(snapshot);
try {process.stdout.write(JSON.stringify({count:bind(snapshot,assignments),snapshot}));}
catch(error) {process.stdout.write(JSON.stringify({error:error.message,unchanged:before===JSON.stringify(snapshot)}));}
'''


ORACLE_PROGRAM = r'''
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const {Battle} = require(process.argv[2] + '/dist/sim');
const {State} = require(process.argv[2] + '/dist/sim/state');
const {bindReferenceSleepSources: bind} = await import(process.argv[1]);
const results=[];
for (const victim of ['p1','p2']) for (const kind of ['induced','rest']) for (const sourceState of ['active','bench','fainted']) {
 const moveSource=sourceState!=='active';
 const foe=victim==='p1'?'p2':'p1';
 const b=new Battle({formatid:'gen3randombattle',seed:[1,2,3,4],strictChoices:true});
 b.setPlayer(victim,{team:[{species:'Articuno',ability:'Pressure',moves:['Rest','Sleep Talk','Splash']},
   {species:'Skarmory',ability:'Keen Eye',moves:['Splash']}]});
 b.setPlayer(foe,{team:[{species:'Breloom',ability:'Effect Spore',moves:['Spore','Splash']},
   {species:'Wailord',ability:'Water Veil',moves:['Spore','Splash']}]});
 const choices=(battle,own,enemy)=>battle.makeChoices(...(victim==='p1'?[own,enemy]:[enemy,own]));
 const sleeper=b[victim].active[0];
 if (kind==='rest') {sleeper.hp--; choices(b,'move rest','move splash');}
 else {sleeper.setStatus('slp',b[foe].active[0],b.dex.moves.get('spore'));b.makeRequest('move');}
 choices(b,'move sleeptalk','move splash');
 choices(b,'switch 2',moveSource?'switch 2':'move splash');
 if (sourceState==='fainted') {
  b[foe].pokemon.find(p=>p.species.id==='breloom').faint();
  b.faintMessages(); b.makeRequest('move');
 }
 const snapshot=State.serializeBattle(b);
 const side=snapshot.sides[victim==='p1'?0:1];
 const row=side.pokemon.find(p=>p.set.species==='Articuno');
 const expectedTime=[row.statusState.time,row.statusState.skippedTime];
 delete row.statusState.source; delete row.statusState.target;
 const identity=(side,species)=>({side,species});
 const count=bind(snapshot,[{kind,sleeper:identity(victim,'Articuno'),
   source:identity(kind==='rest'?victim:foe,kind==='rest'?'Articuno':'Breloom')}]);
 const source=row.statusState.source,target=row.statusState.target;
 const restored=State.deserializeBattle(snapshot); restored.restart(()=>{});
 const oracle=State.deserializeBattle(State.serializeBattle(b)); oracle.restart(()=>{});
 choices(restored,'switch 2','move splash'); choices(oracle,'switch 2','move splash');
 const timer=battle=>{const p=battle[victim].active[0];return [p.status,p.statusState.time,p.statusState.skippedTime];};
 const actual=timer(restored),expected=timer(oracle);
 const clause=State.deserializeBattle(snapshot); clause.restart(()=>{});
 choices(clause,'move splash','move spore');
 results.push({victim,foe,kind,sourceState,moveSource,count,source,target,expectedTime,
   restoredTimer:actual,oracleTimer:expected,clauseStatus:clause[victim].active[0].status});
 b.destroy();restored.destroy();oracle.destroy();clause.destroy();
}
process.stdout.write(JSON.stringify(results));
'''


def snapshot():
    def mon(species, *, sleeping=False, position=0):
        return dict(set=dict(species=species), position=position, hp=100, fainted=False,
                    status="slp" if sleeping else "",
                    statusState=dict(id="slp", time=2, skippedTime=1) if sleeping else {})
    return dict(formatid="gen3randombattle", sides=[
        dict(id="p1", pokemon=[mon("Articuno", sleeping=True), mon("Skarmory", position=1)]),
        dict(id="p2", pokemon=[mon("Breloom"), mon("Wailord", position=1)])])


def assignment(kind="induced"):
    return dict(kind=kind, sleeper=dict(side="p1", species="Articuno"),
                source=dict(side="p1" if kind == "rest" else "p2",
                            species="Articuno" if kind == "rest" else "Breloom"))


class ReferenceSleepBindingTests(unittest.TestCase):
    def apply(self, state, rows):
        return run_node(SYNTHETIC_PROGRAM, json.dumps([state, rows]))

    def test_exact_external_identity_and_timers_unchanged(self):
        state = snapshot()
        result = self.apply(state, [assignment()])
        timer = result["snapshot"]["sides"][0]["pokemon"][0]["statusState"]
        self.assertEqual(timer, dict(id="slp", time=2, skippedTime=1,
                                    source="[Pokemon:p2a]", target="[Pokemon:p1a]"))
        self.assertEqual(result["count"], 1)

    def test_rest_requires_self_identity_and_its_fixed_origin_timer(self):
        state = snapshot()
        timer = state["sides"][0]["pokemon"][0]["statusState"]
        timer["startTime"] = 3
        self.assertEqual(self.apply(state, [assignment("rest")])["count"], 1)
        timer["time"] = 3
        result = self.apply(state, [assignment("rest")])
        self.assertIn("error", result)
        self.assertTrue(result["unchanged"])

    def test_absent_duplicate_awake_and_wrong_side_origins_refuse(self):
        for rows in ([], [assignment(), assignment()],
                     [dict(assignment(), source=dict(side="p2", species="Missing"))],
                     [dict(assignment(), source=dict(side="p1", species="Skarmory"))],
                     [dict(assignment(), sleeper=dict(side="p1", species="Skarmory"))],
                     [dict(assignment("rest"), source=dict(side="p2", species="Breloom"))]):
            with self.subTest(rows=rows):
                result = self.apply(snapshot(), rows)
                self.assertIn("error", result)
                self.assertTrue(result["unchanged"])

    def test_late_bad_source_cannot_partially_rebind_first_sleeper(self):
        state = snapshot()
        second = state["sides"][0]["pokemon"][1]
        second.update(status="slp", statusState=dict(id="slp", time=3, skippedTime=0, startTime=3))
        bad = dict(kind="rest", sleeper=dict(side="p1", species="Skarmory"),
                   source=dict(side="p2", species="Missing"))
        result = self.apply(state, [assignment(), bad])
        self.assertIn("error", result)
        self.assertTrue(result["unchanged"])

    def test_complete_mixed_rest_and_induced_roster_keeps_distinct_sources(self):
        state = snapshot()
        state["sides"][0]["pokemon"][1].update(status="slp",
            statusState=dict(id="slp", time=3, skippedTime=0, startTime=3))
        rest = dict(kind="rest", sleeper=dict(side="p1", species="Skarmory"),
                    source=dict(side="p1", species="Skarmory"))
        result = self.apply(state, [assignment(), rest])
        self.assertEqual(result["count"], 2)
        rows = result["snapshot"]["sides"][0]["pokemon"]
        self.assertEqual([p["statusState"]["source"] for p in rows],
                         ["[Pokemon:p2a]", "[Pokemon:p1b]"])

    def test_fainted_sleeper_is_not_silently_bound_as_a_living_status(self):
        for fields in (dict(hp=0), dict(fainted=True)):
            state = snapshot(); state["sides"][0]["pokemon"][0].update(fields)
            result = self.apply(state, [assignment()])
            self.assertIn("error", result)
            self.assertTrue(result["unchanged"])

    def test_fainted_source_remains_the_same_source_object(self):
        state = snapshot()
        state["sides"][1]["pokemon"][0].update(hp=0, fainted=True)
        self.assertEqual(self.apply(state, [assignment()])["count"], 1)

    def test_unfinalized_positions_unknown_format_and_duplicate_species_refuse(self):
        states = []
        state = snapshot(); state["sides"][1]["pokemon"][0]["position"] = 1; states.append(state)
        state = snapshot(); state["formatid"] = "gen4randombattle"; states.append(state)
        state = snapshot(); state["sides"][0]["id"] = "p2"; states.append(state)
        state = snapshot(); state["sides"][1]["pokemon"][1]["set"]["species"] = "Breloom"; states.append(state)
        for state in states:
            result = self.apply(state, [assignment()])
            self.assertIn("error", result)
            self.assertTrue(result["unchanged"])

    def test_invalid_timer_or_extra_assignment_fields_refuse(self):
        for fields in (dict(time=True), dict(time=0), dict(time=6), dict(skippedTime=-1),
                       dict(skippedTime=4), dict(startTime=3)):
            state = snapshot(); state["sides"][0]["pokemon"][0]["statusState"].update(fields)
            result = self.apply(state, [assignment()])
            self.assertIn("error", result)
            self.assertTrue(result["unchanged"])
        self.assertIn("error", self.apply(snapshot(), [dict(assignment(), timer=3)]))

    @requires_showdown()
    def test_real_both_seats_benched_sources_reentry_refunds_and_sleep_clause(self):
        self.check_oracle(run_node(ORACLE_PROGRAM, str(showdown_root())))

    def check_oracle(self, rows):
        self.assertEqual(len(rows), 12)
        for row in rows:
            with self.subTest(row=row):
                self.assertEqual(row["count"], 1)
                self.assertEqual(row["target"], f"[Pokemon:{row['victim']}b]")
                expected_source = f"[Pokemon:{row['victim']}b]" if row["kind"] == "rest" else \
                    f"[Pokemon:{row['foe']}{'b' if row['moveSource'] else 'a'}]"
                self.assertEqual(row["source"], expected_source)
                self.assertEqual(row["restoredTimer"], row["oracleTimer"])
                self.assertEqual(row["clauseStatus"] == "slp", row["kind"] == "rest")


if __name__ == "__main__":
    unittest.main()
