import assert from "node:assert/strict";
import {createRequire} from "node:module";
import path from "node:path";
import test from "node:test";
import {copyDeserializationAliases} from "../scripts/battle_bridge_owned_snapshot.mjs";

const root = process.env.POKEZERO_SHOWDOWN_ROOT;
if (!root) throw new Error("Tests require the pinned POKEZERO_SHOWDOWN_ROOT.");
const require = createRequire(import.meta.url);
const {Battle} = require(path.join(root, "dist/sim/index.js"));
const {State} = require(path.join(root, "dist/sim/state.js"));

function normalized(battle) {
  return State.normalize(JSON.parse(JSON.stringify(State.serializeBattle(battle))));
}

function deepFreeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

function fixture() {
  const battle = new Battle({formatid: "gen3customgame", seed: "1,2,3,4",
    p1: {name: "own", team: [
      {species: "Zapdos", ability: "Pressure", item: "Leftovers",
       moves: ["Substitute", "Thunderbolt", "Baton Pass", "Rest"]},
      {species: "Lugia", ability: "Pressure", item: "Leftovers",
       moves: ["Psychic", "Rest", "Sleep Talk", "Recover"]},
    ]},
    p2: {name: "hypothetical", team: [
      {species: "Mr. Mime", ability: "Soundproof", item: "Leftovers",
       moves: ["Encore", "Psychic", "Baton Pass", "Substitute"]},
      {species: "Marowak", ability: "Rock Head", item: "Thick Club",
       moves: ["Earthquake", "Rock Slide", "Swords Dance", "Substitute"]},
    ]},
  });
  // Native chance/choices remain native. This is an independent owned fixture,
  // never a snapshot or reconstruction of an actual opponent's hidden state.
  battle.makeChoices("move 1", "move 4");
  return JSON.parse(JSON.stringify(State.serializeBattle(battle)));
}

test("only direct mutable aliases are copied; recursive decoder input is shared read-only", () => {
  const input = fixture();
  const copy = copyDeserializationAliases(input);
  assert.notEqual(copy, input);
  assert.equal(copy.field, input.field);
  assert.notEqual(copy.log, input.log);
  assert.notEqual(copy.sides[0].pokemon[0].set, input.sides[0].pokemon[0].set);
  assert.notEqual(copy.sides[0].pokemon[0].set.moves, input.sides[0].pokemon[0].set.moves);
  assert.equal(copy.sides[0].pokemon[0].volatiles, input.sides[0].pokemon[0].volatiles);
});

test("full native state matches the old whole-clone restore", () => {
  const input = fixture();
  const legacy = State.deserializeBattle(structuredClone(input));
  const candidate = State.deserializeBattle(copyDeserializationAliases(input));
  assert.deepEqual(normalized(candidate), normalized(legacy));
});

test("frozen retained graph can be restored and advanced without mutation", () => {
  const input = deepFreeze(fixture());
  const before = JSON.stringify(input);
  const candidate = State.deserializeBattle(copyDeserializationAliases(input));
  candidate.makeChoices("move 2", "move 1");
  candidate.sides[0].pokemon[0].set.moves[0] = "Protect";
  candidate.log.push("owned branch only");
  assert.equal(JSON.stringify(input), before);
});

test("two restored branches own sets/logs and native transient state independently", () => {
  const input = deepFreeze(fixture());
  const a = State.deserializeBattle(copyDeserializationAliases(input));
  const b = State.deserializeBattle(copyDeserializationAliases(input));
  const before = normalized(b);
  a.makeChoices("move 2", "move 1");
  a.sides[0].pokemon[0].set.evs.hp = 0;
  a.sides[0].pokemon[0].volatiles.substitute.hp = 1;
  a.log.push("not branch b");
  assert.deepEqual(normalized(b), before);
});

test("same ordered native continuation matches whole-clone restoration", () => {
  const input = deepFreeze(fixture());
  const legacy = State.deserializeBattle(structuredClone(input));
  const candidate = State.deserializeBattle(copyDeserializationAliases(input));
  for (const choices of [["move 2", "move 1"], ["move 2", "move 2"]]) {
    legacy.makeChoices(...choices);
    candidate.makeChoices(...choices);
    assert.deepEqual(normalized(candidate), normalized(legacy));
  }
});

test("unsupported state refuses rather than borrowing an unvalidated graph", () => {
  assert.throws(() => copyDeserializationAliases(null), TypeError);
  assert.throws(() => copyDeserializationAliases({}), TypeError);
});
