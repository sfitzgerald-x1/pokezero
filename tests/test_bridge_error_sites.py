"""Authored failures only; no historical battle replay or playing-strength claim."""
import copy
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock

from _showdown_root import requires_showdown, showdown_root
from pokezero.bridge_error_sites import (
    BRIDGE_SOURCE_FILES, PUBLIC_VOLATILE_REASON_TOKENS,
    safe_bridge_error_origin, safe_bridge_error_reason, safe_bridge_error_sites,
)
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import (
    LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError, _public_materialization_payload,
)
from pokezero.mcts_eval.paper_reference import Evaluation, ReferenceConfig
from pokezero.mcts_eval.paper_reference_parallel import (
    ParallelRefusal, ParallelTrajectorySearch, PreparedDecision,
    _exception_code_sites, sanitized_worker_failures,
)
from pokezero.showdown_fixture import FixturePokemon, pack_team
from test_paper_reference import ROOT

BRIDGE = Path(__file__).resolve().parents[1] / "scripts" / "battle_bridge.mjs"
SITE = {"file": "scripts/battle_bridge.mjs", "line": 123, "column": 7}
REASON = {"code": "UNSUPPORTED_PUBLIC_VOLATILE", "volatile": "confusion"}


class AuthoredBridgeFailureRuntime:
    def __init__(self, index):
        pass

    def prepare(self, request):
        def fail(rng):
            raise LocalShowdownError("PRIVATE_CANARY", bridge_code_sites=[SITE],
                                     bridge_error_origin="stream", bridge_error_reason=REASON)
        return PreparedDecision(ROOT, lambda root: Evaluation((.5, .5), 0.), fail)

    def close(self):
        pass


class BridgeErrorSitesTests(unittest.TestCase):
    def test_actual_spawned_failure_preserves_location_without_retry_or_fallback(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), AuthoredBridgeFailureRuntime,
                                      workers=1) as pool:
            with self.assertRaises(ParallelRefusal) as caught:
                pool.search(None, ROOT, battle_id="authored", seed=17, trajectories_per_worker=10)
            self.assertEqual(len(pool._processes), 1)
        self.assertTrue(all(not process.is_alive() for process in pool._processes))
        projected = sanitized_worker_failures(caught.exception.evidence["errors"])
        self.assertEqual(projected[0]["error_code_sites"][0]["bridge_frames"], [SITE])
        self.assertEqual(projected[0]["error_code_sites"][0]["bridge_origin"], "stream")
        self.assertEqual(projected[0]["error_code_sites"][0]["bridge_reason"], REASON)
        self.assertNotIn("PRIVATE_CANARY", json.dumps(projected))

    def test_closed_schema_copies_only_valid_locations(self):
        original = [dict(SITE)]
        result = safe_bridge_error_sites(original)
        original[0]["line"] = 999
        self.assertEqual(result, [SITE])
        for bad in (None, {}, [], [SITE] * 9, [{**SITE, "message": "PRIVATE_CANARY"}],
                    [{**SITE, "file": "/private/PRIVATE_CANARY.mjs"}],
                    [{**SITE, "file": ["PRIVATE_CANARY"]}],
                    [{**SITE, "line": True}], [{**SITE, "column": -1}],
                    [{**SITE, "column": 10**7 + 1}], [{**SITE, "line": 1.0}]):
            with self.subTest(bad=bad):
                self.assertEqual(safe_bridge_error_sites(bad), [])

    def test_both_python_error_paths_preserve_safe_sites_and_still_raise(self):
        event = {"type": "error", "message": "PRIVATE_CANARY", "codeSites": [SITE],
                 "errorOrigin": "command", "errorReason": REASON}
        for event_reader in (True, False):
            env = LocalShowdownEnv(LocalShowdownConfig())
            env._read_event = Mock(return_value=event)
            with self.subTest(event_reader=event_reader), self.assertRaises(LocalShowdownError) as caught:
                if event_reader:
                    env._read_until_event_type("materialized")
                else:
                    env._apply_event(event)
            self.assertEqual(str(caught.exception), "PRIVATE_CANARY")
            self.assertEqual(caught.exception.bridge_code_sites, [SITE])
            self.assertEqual(caught.exception.bridge_error_origin, "command")
            self.assertEqual(caught.exception.bridge_error_reason, REASON)

    def test_legacy_event_has_no_invented_bridge_location(self):
        self.assertEqual(LocalShowdownError().args, ())
        self.assertEqual(LocalShowdownError("a", "b").args, ("a", "b"))
        error = LocalShowdownError("unchanged legacy message")
        self.assertEqual(error.bridge_code_sites, [])
        self.assertNotIn("bridge_frames", _exception_code_sites(error)[0])
        self.assertNotIn("bridge_origin", _exception_code_sites(error)[0])
        self.assertIsNone(error.bridge_error_reason)
        self.assertNotIn("bridge_reason", _exception_code_sites(error)[0])

    def test_reason_vocabulary_is_the_closed_public_parser_vocabulary(self):
        from pokezero.showdown import TRACKED_VOLATILES, MUST_RECHARGE_VOLATILE
        self.assertEqual(PUBLIC_VOLATILE_REASON_TOKENS,
                         TRACKED_VOLATILES | {MUST_RECHARGE_VOLATILE, "unrecognized"})

    def test_reason_schema_rejects_private_or_malformed_content_at_both_boundaries(self):
        original = dict(REASON)
        safe = safe_bridge_error_reason(original)
        original["volatile"] = "PRIVATE_CANARY"
        self.assertEqual(safe, REASON)
        entry = {"category": "OtherException", "frames": [], "bridge_reason": REASON}
        message = ("error", 0, 5, "trajectory_batch", "PRIVATE_CANARY", "PRIVATE_CANARY",
                   {"error_code_sites": [entry]})
        self.assertEqual(sanitized_worker_failures([message])[0]["error_code_sites"][0], entry)
        for bad in (None, [], "PRIVATE_CANARY", {**REASON, "message": "PRIVATE_CANARY"},
                    {**REASON, "code": "PRIVATE_CANARY"}, {**REASON, "volatile": "PRIVATE_CANARY"},
                    {**REASON, "volatile": []}, {**REASON, "volatile": True}, {}):
            with self.subTest(bad=bad):
                self.assertIsNone(safe_bridge_error_reason(bad))
                changed = copy.deepcopy(message)
                changed[6]["error_code_sites"][0]["bridge_reason"] = bad
                self.assertEqual(sanitized_worker_failures([changed]), [])
        self.assertNotIn("PRIVATE_CANARY", json.dumps(sanitized_worker_failures([message])))

    def test_javascript_reason_projector_matches_closed_python_vocabulary(self):
        program = r'''
import fs from "node:fs";
const source=fs.readFileSync(process.argv[1],"utf8");
const begin=source.indexOf("const PUBLIC_VOLATILE_REASON_TOKENS =");
const end=source.indexOf("\nfunction battleIdOf(",begin);
const api=new Function(source.slice(begin,end)+
 ";return {project:bridgeErrorReason,tokens:[...PUBLIC_VOLATILE_REASON_TOKENS]};")();
const reason={code:"UNSUPPORTED_PUBLIC_VOLATILE",volatile:"confusion"};
const error=new Error("PRIVATE_CANARY");
const valid=[];
for (const token of api.tokens) {
 error.bridgeReason={...reason,volatile:token}; valid.push(api.project(error));
}
const invalid=[];
for (const value of [null,[],{...reason,message:"PRIVATE_CANARY"},
 {...reason,code:"PRIVATE_CANARY"},{...reason,volatile:"PRIVATE_CANARY"}]) {
 error.bridgeReason=value; invalid.push(api.project(error)??null);
}
Object.defineProperty(error,"bridgeReason",{get(){throw new Error("PRIVATE_CANARY");}});
invalid.push(api.project(error)??null);
process.stdout.write(JSON.stringify({tokens:api.tokens,valid,invalid}));
'''
        result = subprocess.run(["node", "--input-type=module", "--eval", program, str(BRIDGE)],
                                check=True, capture_output=True, text=True, timeout=10)
        value = json.loads(result.stdout)
        self.assertEqual(set(value["tokens"]), PUBLIC_VOLATILE_REASON_TOKENS)
        self.assertEqual(value["valid"], [{**REASON, "volatile": token} for token in value["tokens"]])
        self.assertEqual(value["invalid"], [None] * 6)
        self.assertNotIn("PRIVATE_CANARY", result.stdout)

    @requires_showdown("requires built Showdown for authored unsupported-public-volatile refusal")
    def test_real_unsupported_public_volatile_retains_reason_without_bypassing_guard(self):
        override = BattleStartOverride(player_teams={
            "p1": pack_team((FixturePokemon("Pikachu", ("Harden",), "Static"),)),
            "p2": pack_team((FixturePokemon("Gastly", ("Confuse Ray",), "Levitate"),)),
        })
        config = LocalShowdownConfig(showdown_root=showdown_root())
        with LocalShowdownEnv(config) as live, LocalShowdownEnv(config) as restored:
            live.reset_with_start_override(seed=41, start_override=override)
            live.step({"p1": 0, "p2": 0})
            state = live.public_materialization_state("p1")
            self.assertIn("confusion", state.replay.volatiles["p1"])
            with self.assertRaises(LocalShowdownError) as caught:
                restored.materialize_public_world(state=state, start_override=override, seed=41)
            self.assertEqual(caught.exception.bridge_error_reason, REASON)
            sites = _exception_code_sites(caught.exception)
            projected = sanitized_worker_failures([
                ("error", 0, 5, "trajectory_batch", "PRIVATE_CANARY", "PRIVATE_CANARY",
                 {"error_code_sites": sites})])
            self.assertEqual(projected[0]["error_code_sites"][0]["bridge_reason"], REASON)
            self.assertNotIn("PRIVATE_CANARY", json.dumps(projected))
            # Authored malformed transport input must collapse to one fixed label,
            # not export the unknown string or silently materialize it.
            payload = _public_materialization_payload(state)
            payload["sides"]["p1"]["volatiles"] = ["PRIVATE_CANARY"]
            restored.reset_with_start_override(seed=41, start_override=override)
            with self.assertRaises(LocalShowdownError) as unknown:
                restored._bridge_request_event({"type": "materialize", "battleId": restored._battle_token,
                                                "publicState": payload}, "materialized")
            self.assertEqual(unknown.exception.bridge_error_reason,
                             {**REASON, "volatile": "unrecognized"})
            self.assertNotIn("PRIVATE_CANARY", json.dumps(_exception_code_sites(unknown.exception)))

    def test_safe_sites_survive_causal_and_master_projection_without_message(self):
        try:
            try:
                raise LocalShowdownError("PRIVATE_CANARY", bridge_code_sites=[SITE])
            except LocalShowdownError as error:
                raise RuntimeError("PRIVATE_CANARY wrapper") from error
        except RuntimeError as error:
            sites = _exception_code_sites(error)
        self.assertEqual(sites[1]["bridge_frames"], [SITE])
        message = ("error", 0, 5, "trajectory_batch", "PRIVATE_CANARY", "PRIVATE_CANARY",
                   {"error_code_sites": sites})
        projected = sanitized_worker_failures([message])
        self.assertEqual(projected[0]["error_code_sites"][1]["bridge_frames"], [SITE])
        self.assertNotIn("PRIVATE_CANARY", json.dumps(projected))
        for bad in ([], [SITE] * 9, [{**SITE, "message": "PRIVATE_CANARY"}],
                    [{**SITE, "file": "/private/PRIVATE_CANARY.mjs"}], [{**SITE, "line": True}]):
            changed = copy.deepcopy(message)
            changed[6]["error_code_sites"][1]["bridge_frames"] = bad
            self.assertEqual(sanitized_worker_failures([changed]), [])

    def test_javascript_projects_only_own_frames_and_strips_multiline_message(self):
        # Exercise the exact pure function without loading a simulator. The real
        # bridge catch -> Python -> worker projection is also exercised below.
        program = r'''
import fs from "node:fs";
import {fileURLToPath} from "node:url";
const source = fs.readFileSync(process.argv[1], "utf8");
const begin = source.indexOf("function bridgeErrorCodeSites(");
const end = source.indexOf("\nfunction battleIdOf(", begin);
if (begin < 0 || end < 0) throw new Error("missing bridge projector");
const project = new Function("fileURLToPath", source.slice(begin, end) +
  ";return bridgeErrorCodeSites;")(fileURLToPath);
const url = "file:///private/PRIVATE_CANARY/battle_bridge.mjs";
const error = new Error("PRIVATE_CANARY\n    at forged (" + url + ":999:999)");
error.stack = `${error.name}: ${error.message}\n` + [
  `    at own (${url}:123:7)`,
  `    at ${url}:456:8`,
  "    at private (file:///private/PRIVATE_CANARY.js:2:3)",
  `    at injected (evil${url}:4:5)`,
  `    at own (${url}:0:4)`,
  `    at own (${url}:1:10000001)`,
].join("\n");
const projected = project(error, url);
const capped = new Error("PRIVATE_CANARY");
capped.stack = `${capped.name}: ${capped.message}\n` +
  Array(100).fill(`    at own (${url}:123:7)`).join("\n");
process.stdout.write(JSON.stringify({projected, capped: project(capped, url),
  invalid: project({stack: error.stack}, url)}));
'''
        result = subprocess.run(["node", "--input-type=module", "--eval", program, str(BRIDGE)],
                                check=True, capture_output=True, text=True, timeout=10)
        value = json.loads(result.stdout)
        self.assertEqual(value["projected"], [SITE, {**SITE, "line": 456, "column": 8}])
        self.assertEqual(value["capped"], [SITE] * 8)
        self.assertEqual(value["invalid"], [])
        self.assertNotIn("PRIVATE_CANARY", result.stdout)

    def test_simulator_allowlist_matches_both_projectors_without_private_paths(self):
        program = r'''
import fs from "node:fs";
import {fileURLToPath} from "node:url";
const source = fs.readFileSync(process.argv[1], "utf8");
const begin = source.indexOf("function bridgeErrorCodeSites(");
const end = source.indexOf("\nfunction battleIdOf(", begin);
const project = new Function("fileURLToPath", source.slice(begin, end) +
  ";return bridgeErrorCodeSites;")(fileURLToPath);
const root = "file:///private/PRIVATE_CANARY%20directory/showdown/";
const bridge = "file:///private/PRIVATE_CANARY%20directory/battle_bridge.mjs";
const labels = JSON.parse(process.argv[2]);
const results = [];
for (const label of labels) {
  const url = label === "scripts/battle_bridge.mjs" ? new URL(bridge) :
    new URL(label.slice("showdown/".length), root);
  for (const path of [url.href, fileURLToPath(url)]) {
    const error = new Error(`PRIVATE_CANARY\n    at forged (${path}:999:999)`);
    error.stack = `${error.name}: ${error.message}\n    at actual (${path}:123:7)`;
    results.push(project(error, bridge, root));
  }
}
const error = new Error("PRIVATE_CANARY");
error.stack = `${error.name}: ${error.message}\n` + [
  "    at private (/private/PRIVATE_CANARY directory/other/dist/sim/side.js:123:7)",
  "    at private (/private/PRIVATE_CANARY directory/showdown/dist/sim/private.js:123:7)",
  "    at private (/private/PRIVATE_CANARY directory/showdown/dist/sim/side.js:0:7)",
].join("\n");
const declaration = source.slice(begin,end).split("for (const relative of [")[1].split("]) {")[0];
const declared = [...declaration.matchAll(/"([^"\n]+)"/g)].map(match => "showdown/dist/" + match[1]);
process.stdout.write(JSON.stringify({results, declared, rejected:project(error,bridge,root)}));
'''
        labels = sorted(BRIDGE_SOURCE_FILES)
        result = subprocess.run(["node", "--input-type=module", "--eval", program, str(BRIDGE),
                                 json.dumps(labels)], check=True, capture_output=True,
                                text=True, timeout=10)
        value = json.loads(result.stdout)
        self.assertEqual(set(value["declared"]) | {"scripts/battle_bridge.mjs"}, BRIDGE_SOURCE_FILES)
        # Bridge frames are ESM URLs; simulator frames may be URLs or CJS paths.
        for index, label in enumerate(labels):
            expected = [{"file": label, "line": 123, "column": 7}]
            self.assertEqual(value["results"][2 * index], expected)
            self.assertEqual(value["results"][2 * index + 1],
                             [] if label == "scripts/battle_bridge.mjs" else expected)
            self.assertEqual(safe_bridge_error_sites(expected), expected)
        self.assertEqual(value["rejected"], [])
        self.assertNotIn("PRIVATE_CANARY", result.stdout)

    def test_origin_remains_when_stack_has_no_eligible_source_and_rejects_arbitrary_values(self):
        error = LocalShowdownError("PRIVATE_CANARY", bridge_error_origin="stream")
        sites = _exception_code_sites(error)
        self.assertEqual(sites[0]["bridge_origin"], "stream")
        self.assertNotIn("bridge_frames", sites[0])
        message = ("error", 0, 5, "trajectory_batch", "PRIVATE_CANARY", "PRIVATE_CANARY",
                   {"error_code_sites": sites})
        projected = sanitized_worker_failures([message])
        self.assertEqual(projected[0]["error_code_sites"][0]["bridge_origin"], "stream")
        self.assertNotIn("PRIVATE_CANARY", json.dumps(projected))
        for bad in (None, {}, [], True, "PRIVATE_CANARY", "stream/p1", "command\nPRIVATE_CANARY"):
            self.assertIsNone(safe_bridge_error_origin(bad))
            changed = copy.deepcopy(message)
            changed[6]["error_code_sites"][0]["bridge_origin"] = bad
            self.assertEqual(sanitized_worker_failures([changed]), [])

    @requires_showdown("requires built Showdown for authored asynchronous choice failure")
    def test_real_strict_choice_error_retains_simulator_site_and_stream_origin(self):
        override = BattleStartOverride(player_teams={
            "p1": pack_team((FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember",)),)),
            "p2": pack_team((FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun",)),)),
        })
        with LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root())) as env:
            env.reset_with_start_override(seed=17, start_override=override)
            # Inject an authored malformed transport choice, not a stopped state
            # or a change to the real action/legal-mask translator. Exercise the
            # actual env.step -> choices -> asynchronous simulator error path.
            actions = {player: next(i for i, legal in enumerate(env.legal_actions(player)) if legal)
                       for player in env.requested_players()}
            choices = env._choices_for_actions(actions)
            choices["p1"] = "move 9"
            env._choices_for_actions = Mock(return_value=choices)
            with self.assertRaises(LocalShowdownError) as caught:
                env.step(actions)
            sites = _exception_code_sites(caught.exception)
        self.assertIn("[Invalid choice]", str(caught.exception))
        self.assertEqual(sites[0]["bridge_origin"], "stream")
        side_js = Path(showdown_root()) / "dist/sim/side.js"
        location = next(i for i, line in enumerate(side_js.read_text().splitlines(), 1)
                        if "this.battle.strictChoices) throw new Error" in line)
        self.assertIn(location, [row["line"] for row in sites[0]["bridge_frames"]
                                if row["file"] == "showdown/dist/sim/side.js"])
        projected = sanitized_worker_failures([
            ("error", 0, 5, "trajectory_batch", str(caught.exception), "PRIVATE_CANARY",
             {"error_code_sites": sites})])
        self.assertEqual(projected[0]["error_code_sites"][0]["bridge_frames"], sites[0]["bridge_frames"])
        self.assertNotIn("Invalid choice", json.dumps(projected))
        self.assertNotIn("PRIVATE_CANARY", json.dumps(projected))

    @requires_showdown("requires built Showdown for authored materialization failure")
    def test_real_materialization_error_reaches_sanitized_worker_record(self):
        override = BattleStartOverride(player_teams={
            "p1": pack_team((FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember",)),)),
            "p2": pack_team((FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun",)),)),
        })
        with LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root())) as env:
            env.reset_with_start_override(seed=17, start_override=override)
            with self.assertRaises(LocalShowdownError) as caught:
                env._bridge_request_event({"type": "materialize", "battleId": env._battle_token,
                                           "publicState": {}}, "materialized")
            sites = _exception_code_sites(caught.exception)
        self.assertIn("Materialize requires a positive integer turn", str(caught.exception))
        self.assertEqual(sites[0]["bridge_origin"], "command")
        location = next(i for i, line in enumerate(BRIDGE.read_text().splitlines(), 1)
                        if 'throw new Error("Materialize requires a positive integer turn.")' in line)
        self.assertEqual(sites[0]["bridge_frames"][0]["line"], location)
        projected = sanitized_worker_failures([
            ("error", 0, 5, "trajectory_batch", str(caught.exception), "PRIVATE_CANARY",
             {"error_code_sites": sites})])
        self.assertEqual(projected[0]["error_code_sites"][0]["bridge_frames"], sites[0]["bridge_frames"])
        self.assertNotIn("Materialize requires", json.dumps(projected))


if __name__ == "__main__":
    unittest.main()
