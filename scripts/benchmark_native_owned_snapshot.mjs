// Isolated execution-cost probe, not a sampler/strength qualification. Both
// alternatives fully deserialize the same owned hypothetical native snapshots.
import assert from "node:assert/strict";
import {createRequire} from "node:module";
import path from "node:path";
import {performance} from "node:perf_hooks";
import {copyDeserializationAliases} from "./battle_bridge_owned_snapshot.mjs";

const root = process.env.POKEZERO_SHOWDOWN_ROOT;
if (!root) throw new Error("Pinned POKEZERO_SHOWDOWN_ROOT required.");
const require = createRequire(import.meta.url);
const {Battle} = require(path.join(root, "dist/sim/index.js"));
const {State} = require(path.join(root, "dist/sim/state.js"));
const oldRestore = input => State.deserializeBattle(structuredClone(input));
const newRestore = input => State.deserializeBattle(copyDeserializationAliases(input));
const normalized = battle => State.normalize(JSON.parse(JSON.stringify(State.serializeBattle(battle))));
function freeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}
function fixture(turns) {
  const team = species => species.map(name => ({species: name, item: "Leftovers",
    moves: ["Substitute", "Splash", "Rest", "Protect"]}));
  const battle = new Battle({formatid: "gen3customgame", seed: "1,2,3,4",
    p1: {name: "owned", team: team(["Lugia", "Zapdos", "Mewtwo", "Tauros", "Gengar", "Celebi"])},
    p2: {name: "hypothetical", team: team(["Snorlax", "Marowak", "Blissey", "Mr. Mime", "Tyranitar", "Suicune"])}});
  if (turns) battle.makeChoices("move 1", "move 1");
  for (let i = 1; i < turns; i++) battle.makeChoices("move 2", "move 2");
  return freeze(JSON.parse(JSON.stringify(State.serializeBattle(battle))));
}
function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const i = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[i] : (ordered[i - 1] + ordered[i]) / 2;
}
const rounds = 16, restoresPerRound = 32;
const cases = [];
for (const turns of [0, 12, 40]) {
  const input = fixture(turns), before = JSON.stringify(input);
  const a = oldRestore(input), b = newRestore(input);
  assert.deepEqual(normalized(a), normalized(b));
  a.makeChoices("move 2", "move 2");
  b.makeChoices("move 2", "move 2");
  assert.deepEqual(normalized(a), normalized(b));
  assert.equal(JSON.stringify(input), before);
  for (let i = 0; i < 32; i++) {oldRestore(input); newRestore(input);}
  const oldTimes = [], newTimes = [];
  function measure(fn, output) {
    const start = performance.now();
    for (let i = 0; i < restoresPerRound; i++) {
      const restored = fn(input);
      assert.equal(restored.turn, input.turn);
    }
    output.push((performance.now() - start) / restoresPerRound);
  }
  for (let i = 0; i < rounds; i++) {
    if (i % 2) {measure(newRestore, newTimes); measure(oldRestore, oldTimes);}
    else {measure(oldRestore, oldTimes); measure(newRestore, newTimes);}
  }
  assert.equal(JSON.stringify(input), before);
  cases.push({turns, serializedBytes: Buffer.byteLength(before), rounds, restoresPerRound,
    oldMillisecondsPerRestore: oldTimes, candidateMillisecondsPerRestore: newTimes,
    oldMedianMilliseconds: median(oldTimes), candidateMedianMilliseconds: median(newTimes),
    medianRatio: median(newTimes) / median(oldTimes),
    fullStateAndContinuationEqual: true, retainedInputUnchanged: true});
}
process.stdout.write(JSON.stringify({schema: "native-owned-copy-cost-probe-v1",
  scope: "synthetic owned six-Pokemon teams; full native deserialize; not conditioning or strength",
  clock: "counterbalanced sequential rounds; warm-up excluded; GC not forced", cases}, null, 2) + "\n");
