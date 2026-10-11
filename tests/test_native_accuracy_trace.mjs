import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import path from 'node:path';
import test from 'node:test';
import {installChanceTrace} from '../scripts/battle_bridge_chance_trace.mjs';

const root = process.env.POKEZERO_SHOWDOWN_ROOT;
if (!root) throw new Error('Pinned POKEZERO_SHOWDOWN_ROOT required.');
const require = createRequire(import.meta.url);
const {Battle} = require(path.join(root, 'dist/sim/index.js'));
const {State} = require(path.join(root, 'dist/sim/state.js'));
const normalized = b => State.normalize(JSON.parse(JSON.stringify(State.serializeBattle(b))));

function fixture(seed, sourceMoves = ['Toxic', 'Sleep Talk', 'Rest', 'Splash'],
  targetMoves = ['Substitute', 'Splash', 'Fly', 'Thunder Wave'], targetSpecies = 'Cacturne') {
  return new Battle({formatid: 'gen3customgame', seed: `1,2,3,${seed}`,
    p1: {name: 'own', team: [{species: 'Lanturn', ability: 'Volt Absorb', moves: sourceMoves}]},
    p2: {name: 'hypothetical', team: [{species: targetSpecies, ability: 'Sand Veil', moves: targetMoves}]}});
}

function pairedStep(a, b, choices) {
  const before = [b.runEvent, b.actions.tryMoveHit, b.prng.randomChance, b.prng.rng.next];
  const trace = installChanceTrace(b);
  try { a.makeChoices(...choices); b.makeChoices(...choices); }
  finally { trace.close(); }
  assert.deepEqual(normalized(b), normalized(a), 'instrumentation must preserve full native state/RNG');
  assert.deepEqual([b.runEvent, b.actions.tryMoveHit, b.prng.randomChance, b.prng.rng.next], before);
  assert.deepEqual(trace.records.map(r => r.ordinal), trace.records.map((_, i) => i + 1));
  return trace.records;
}

test('native Toxic hit and miss coins carry returned Accuracy context, not a guessed 100-roll', () => {
  const seen = new Set();
  for (let seed = 1; seed <= 32; seed++) {
    const a = fixture(seed), b = fixture(seed);
    const records = pairedStep(a, b, ['move 1', 'move 2']);
    const rows = records.filter(r => r.accuracy?.move === 'toxic');
    assert.equal(rows.length, 1);
    const row = rows[0];
    assert.equal(row.accuracy.source_ident, 'p1a: Lanturn');
    assert.equal(row.accuracy.target_ident, 'p2a: Cacturne');
    assert.deepEqual(row.chance, {numerator: row.accuracy.numerator, denominator: 100});
    seen.add(a.log.some(line => line.includes('|Toxic|') && line.includes('[miss]')) ? 'miss' : 'hit');
  }
  assert.deepEqual([...seen].sort(), ['hit', 'miss']);
});

test('Substitute-blocked still branch retains its native accuracy coin and public outcome', () => {
  const seen = new Set();
  for (let seed = 1; seed <= 32; seed++) {
    const a = fixture(seed), b = fixture(seed);
    pairedStep(a, b, ['move 4', 'move 1']);
    const at = a.log.length;
    const rows = pairedStep(a, b, ['move 1', 'move 2']).filter(r => r.accuracy?.move === 'toxic');
    assert.equal(rows.length, 1);
    const suffix = a.log.slice(at);
    seen.add(suffix.some(line => line.includes('|Toxic|') && line.includes('[still]')) ? 'still' : 'miss');
  }
  assert.deepEqual([...seen].sort(), ['miss', 'still']);
});

test('called Sleep Talk Toxic is contextualized by the native called move, not Sleep Talk', () => {
  let calls = 0;
  for (let seed = 1; seed <= 16; seed++) {
    const a = fixture(seed), b = fixture(seed);
    for (const battle of [a, b]) {
      battle.p1.active[0].hp -= 20;
      battle.p1.active[0].status = 'slp';
      battle.p1.active[0].statusState = {id: 'slp', time: 4, startTime: 4, target: battle.p1.active[0]};
    }
    const records = pairedStep(a, b, ['move 2', 'move 2']);
    assert.equal(records.filter(r => r.accuracy?.move === 'sleeptalk').length, 0);
    if (a.log.some(line => line.includes('|Toxic|'))) {
      calls++;
      assert.equal(records.filter(r => r.accuracy?.move === 'toxic').length, 1);
    }
  }
  assert.ok(calls > 0);
});

test('invulnerability and secondary rolls cannot borrow stale accuracy context', () => {
  const a = fixture(5, ['Toxic'], ['Fly'], 'Aerodactyl'),
    b = fixture(5, ['Toxic'], ['Fly'], 'Aerodactyl');
  const records = pairedStep(a, b, ['move 1', 'move 1']);
  const accuracy = records.filter(r => r.accuracy);
  assert.ok(accuracy.every(r => r.chance?.denominator === 100));
  assert.equal(accuracy.filter(r => r.accuracy.move === 'toxic').length, 0);
});

test('damage secondary-effect denominator-100 rolls are not tagged as accuracy', () => {
  const a = fixture(8, ['Thunderbolt']), b = fixture(8, ['Thunderbolt']);
  const records = pairedStep(a, b, ['move 1', 'move 2']);
  // Gen3 draws this as random(100), not randomChance(10, 100).
  const secondary = records.filter(r => r.chance === null && r.range?.[0] === 100);
  assert.ok(secondary.length > 0);
  assert.ok(secondary.every(r => r.accuracy === null));
});

test('event-to-coin binding expires on intervening events or random draws; true accuracy is not tagged', () => {
  const b = fixture(7), trace = installChanceTrace(b);
  const source = b.p1.active[0], target = b.p2.active[0], move = b.dex.moves.get('toxic');
  b.runEvent('Accuracy', target, source, move, 85);
  b.random(16);
  b.randomChance(85, 100);
  b.runEvent('Accuracy', target, source, move, 85);
  b.runEvent('ModifyAccuracy', target, source, move, 85);
  b.randomChance(85, 100);
  b.runEvent('Accuracy', target, source, move, true);
  b.randomChance(85, 100);
  trace.close();
  assert.ok(trace.records.every(r => r.accuracy === null));
});

test('intervening hit calls do not resurrect old pending events', () => {
  const b = fixture(9);
  b.actions.tryMoveHit = () => b.random(16);
  const trace = installChanceTrace(b);
  const source = b.p1.active[0], target = b.p2.active[0], move = b.dex.moves.get('toxic');
  b.runEvent('Accuracy', target, source, move, 85);
  b.actions.tryMoveHit();
  b.randomChance(85, 100);
  trace.close();
  assert.ok(trace.records.every(r => r.accuracy === null));
});

test('nested and throwing outer events cannot leak an inner Accuracy context', () => {
  const b = fixture(10);
  const native = b.runEvent;
  b.runEvent = function (event, target, source, move, ...args) {
    if (event === 'Outer' || event === 'OuterThrow') {
      this.runEvent('Accuracy', target, source, move, 85);
      if (event === 'OuterThrow') throw new Error('native fixture error');
      return 85;
    }
    return native.call(this, event, target, source, move, ...args);
  };
  const trace = installChanceTrace(b);
  const source = b.p1.active[0], target = b.p2.active[0], move = b.dex.moves.get('toxic');
  b.runEvent('Outer', target, source, move);
  b.randomChance(85, 100);
  assert.throws(() => b.runEvent('OuterThrow', target, source, move), /native fixture error/);
  b.randomChance(85, 100);
  trace.close();
  assert.ok(trace.records.every(r => r.accuracy === null));
});
