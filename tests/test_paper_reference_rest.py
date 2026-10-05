"""Public Rest timers, including real pinned-server wake and switch refunds."""
import json
from pathlib import Path
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
from pokezero.showdown_fixture import FixturePokemon, pack_team

HELPER = Path(__file__).resolve().parents[1] / "scripts/battle_bridge_reference_rest.mjs"


def rest_state(row, ability="Pressure"):
    program = """
const {referenceRestState} = await import(process.argv[1]);
const [row, ability] = JSON.parse(process.argv[2]);
try {process.stdout.write(JSON.stringify({state: referenceRestState(row, ability)}));}
catch (e) {process.stdout.write(JSON.stringify({error: e.message}));}
"""
    result = subprocess.run(["node", "--input-type=module", "--eval", program,
        str(HELPER), json.dumps([row, ability])], check=True, capture_output=True,
        text=True, timeout=10)
    return json.loads(result.stdout)


class ReferenceRestCounterTests(unittest.TestCase):
    def test_active_skipped_time_is_not_refunded_until_switch_in(self):
        result = rest_state(dict(active=True, restSleepAttempts=1,
            restSleepSkippedTime=1, restSleepActiveRefundPending=True, restSleepRefundPending=True))
        self.assertEqual(result["state"], dict(id="slp", effectOrder=0,
            startTime=3, time=2, skippedTime=1))

    def test_prior_refund_and_early_bird_are_accounted_for_separately(self):
        self.assertEqual(rest_state(dict(restSleepAttempts=3, restSleepRefundedTime=1))["state"]["time"], 1)
        self.assertEqual(rest_state(dict(restSleepAttempts=1), "Early Bird")["state"]["time"], 1)
        self.assertEqual(rest_state(dict(active=False, restSleepAttempts=1,
            restSleepSkippedTime=1))["state"]["time"], 2)

    def test_unknown_induced_unsettled_and_malformed_states_refuse(self):
        rows = [{}, {"restSleepAttempts": True}, {"restSleepAttempts": -1},
            {"restSleepAttempts": 1, "restSleepRefundedTime": 2},
            {"restSleepAttempts": 3}, {"restSleepAttempts": 0, "restSleepSkippedTime": 1},
            {"restSleepAttempts": 1, "restSleepAttemptUnsettled": True},
            {"restSleepAttempts": 1, "restSleepProvenanceUnrepresentable": True},
            {"restSleepAttempts": 1, "restSleepRefundPending": True},
            {"restSleepAttempts": 1, "restSleepActiveRefundPending": True}]
        for row in rows:
            with self.subTest(row=row):
                self.assertIn("error", rest_state(row))


@requires_showdown()
class ReferenceRestServerTests(unittest.TestCase):
    def test_custom_nickname_species_collision_refuses_instead_of_swapping_timers(self):
        config = LocalShowdownConfig(showdown_root=showdown_root())
        team = pack_team((FixturePokemon(species="Articuno", ability="Pressure", moves=("Rest",)),
            FixturePokemon(species="Skarmory", ability="Keen Eye", moves=("Rest",))))
        first, second = team.split("]")
        nicknamed = "Skarmory|Articuno" + first[len("Articuno|"):] + "]" + \
            "Articuno|Skarmory" + second[len("Skarmory|"):]
        override = BattleStartOverride(player_teams={"p1": nicknamed,
            "p2": pack_team((FixturePokemon(species="Wailord", ability="Water Veil", moves=("Splash",)),))})
        with LocalShowdownEnv(config) as live, LocalShowdownEnv(config) as restored:
            live.reset_with_start_override(seed=17, start_override=override)
            state = live.public_materialization_state("p1")
            with self.assertRaisesRegex(LocalShowdownError, "nicknames"):
                restored.materialize_public_world(state=state, start_override=override,
                    seed=19, reference_rest_sleep=True)

    def test_public_bridge_restoration_matches_wake_and_benched_refund_for_both_seats(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = BattleStartOverride(player_teams={
            "p1": pack_team((FixturePokemon(species="Articuno", ability="Pressure",
                moves=("Rest", "Sleep Talk", "Splash")),
                FixturePokemon(species="Skarmory", ability="Keen Eye", moves=("Splash",)))),
            "p2": pack_team((FixturePokemon(species="Wailord", ability="Water Veil",
                moves=("Night Shade", "Splash")),))})
        with LocalShowdownEnv(config) as live, LocalShowdownEnv(config) as restored:
            live.reset_with_start_override(seed=17, start_override=override)
            live.step({"p1": 2, "p2": 0})  # Public damage allows Rest to succeed.
            live.step({"p1": 0, "p2": 1})  # Rest fixes the timer to 3.
            live.step({"p1": 1, "p2": 1})  # Sleep Talk: time 2, skippedTime 1.
            for actor in ("p1", "p2"):
                with self.subTest(actor=actor):
                    state = live.public_materialization_state(actor)
                    with self.assertRaisesRegex(LocalShowdownError, "sleep counters"):
                        restored.materialize_public_world(state=state, start_override=override, seed=19)
                    restored.materialize_public_world(state=state, start_override=override, seed=19,
                        reference_rest_sleep=True)
                    timer = restored.snapshot().bridge_snapshot["battle"]["sides"][0]["pokemon"][0]["statusState"]
                    self.assertEqual((timer["time"], timer["skippedTime"], timer["startTime"]), (2, 1, 3))
                    self.assertEqual(timer["source"], timer["target"])
                    # Keep sleeping one more attempt, then wake on the next.
                    restored.step({"p1": 2, "p2": 1})
                    self.assertEqual(restored.snapshot().bridge_snapshot["battle"]["sides"][0]["pokemon"][0]["status"], "slp")
                    restored.step({"p1": 2, "p2": 1})
                    self.assertEqual(restored.snapshot().bridge_snapshot["battle"]["sides"][0]["pokemon"][0]["status"], "")
            live.step({"p1": 4, "p2": 1})  # Dense first bench candidate: Skarmory.
            state = live.public_materialization_state("p1")
            restored.materialize_public_world(state=state, start_override=override, seed=19,
                reference_rest_sleep=True)
            side = restored.snapshot().bridge_snapshot["battle"]["sides"][0]
            sleeper = next(p for p in side["pokemon"] if p["status"] == "slp")
            self.assertEqual(sleeper["statusState"]["source"], f"[Pokemon:p1{'abcdef'[sleeper['position']]}]")
            live.step({"p1": 4, "p2": 1})
            restored.step({"p1": 4, "p2": 1})
            for env in (live, restored):
                timer = env.snapshot().bridge_snapshot["battle"]["sides"][0]["pokemon"][0]["statusState"]
                self.assertEqual((timer["time"], timer["skippedTime"]), (3, 0))

    def test_real_server_early_bird_and_sleep_clause_self_source(self):
        program = """
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const {Battle} = require(process.argv[1] + '/dist/sim');
const {State} = require(process.argv[1] + '/dist/sim/state');
const {referenceRestState, bindReferenceRestSources} = await import(process.argv[2]);
const outcomes = [];
for (const ability of ['Pressure', 'Early Bird']) {
  const b = new Battle({formatid:'gen3randombattle', seed:[1,2,3,4]});
  b.setPlayer('p1', {team:[{species:'Articuno', ability, moves:['Rest','Splash']},
    {species:'Skarmory',ability:'Keen Eye',moves:['Splash']}]});
  b.setPlayer('p2', {team:[{species:'Wailord',ability:'Water Veil',moves:['Splash','Spore']}]});
  b.p1.active[0].hp--;
  b.makeChoices('move rest','move splash');
  b.makeChoices('move splash','move splash');
  const snapshot = State.serializeBattle(b);
  const p = snapshot.sides[0].pokemon[0];
  const expected = [p.statusState.time, p.statusState.skippedTime];
  p.statusState = referenceRestState({restSleepAttempts:1}, ability);
  bindReferenceRestSources(snapshot.sides[0], 'p1');
  const restored = State.deserializeBattle(snapshot);
  restored.restart(() => {});
  restored.makeChoices('move splash','move splash');
  outcomes.push({ability, expected, status:restored.p1.active[0].status});
  // Rest on the bench must not block the opponent inducing a DIFFERENT sleeper.
  if (ability === 'Pressure') {
    const clause = State.deserializeBattle(snapshot); clause.restart(() => {});
    clause.makeChoices('switch 2','move spore');
    outcomes.push({clauseStatus:clause.p1.active[0].status,
      sameSource:clause.p1.pokemon[1].statusState.source === clause.p1.pokemon[1]});
  }
}
process.stdout.write(JSON.stringify(outcomes));
"""
        result = subprocess.run(["node", "--input-type=module", "--eval", program,
            str(showdown_root()), str(HELPER)], check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(json.loads(result.stdout), [
            dict(ability="Pressure", expected=[2, 0], status="slp"),
            dict(clauseStatus="slp", sameSource=True),
            dict(ability="Early Bird", expected=[1, 0], status="")])


if __name__ == "__main__":
    unittest.main()
