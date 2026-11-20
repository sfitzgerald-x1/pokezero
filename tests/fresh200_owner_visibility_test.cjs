'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { test } = require('node:test');
const { createVisibility, readPolicy } = require('../gpu_engine/fresh200_owner_visibility.cjs');

const root = process.env.POKEZERO_SHOWDOWN_ROOT;
assert.ok(root, 'pinned Native runtime required');
const { BattleStream } = require(path.join(root, 'dist/sim/battle-stream.js'));
const { Battle } = require(path.join(root, 'dist/sim/battle.js'));

const policy = Object.freeze({
  schema: 'pokezero.fresh200.owner_visibility.v1',
  formatid: 'gen3customgame@@@sleepclausemod',
  native_commit: 'f76228a1354b5d0f307ca2d16101294ad3a2308b',
  compiled_tree_sha256: 'd22c0cc51aa32b983c2946108834418e824b0056a727e121e19dc56919350a01',
  root_index: 0, stream_id: 2120000000, reference_owner: 'p2', endpoint_frame: 8,
  session: 'fixture', battle_id: 'fixture',
  scope_receipt_sha256: '1'.repeat(64), fixed_input_binding_sha256: '2'.repeat(64),
  start_wire_sha256: '3'.repeat(64), prior_recording_sha256: '4'.repeat(64),
  prior_physical_pre_sha256: '5'.repeat(64),
});
const wire = value => Buffer.from(JSON.stringify(value) + '\n');
const context = Object.freeze({
  session: policy.session, battleId: policy.battle_id, formatid: policy.formatid, stdinSha256: policy.start_wire_sha256,
});

test('v2 binds the frozen roster and selected address without accepting a v1 substitution', () => {
  const { prior_recording_sha256, prior_physical_pre_sha256, ...base } = policy;
  const next = {
    ...base, schema: 'pokezero.fresh200.owner_visibility.v2', root_index: 1,
    storage_row: 37, stream_id: 2120000037, reference_owner: 'p1', endpoint_frame: 9,
    capture_sha256: 'f6c606ffbce81b0b43351d88188c448d5b54a16334e6f75c20681aca2800b13a',
    device_replay_sha256: 'f6520ff94bbea899f5ba05dcd80cfba8fdc1f0628626f20956f58aaf7de52b29',
    ordered_roster_sha256: '6'.repeat(64), selected_address_sha256: '7'.repeat(64),
  };
  assert.deepEqual(readPolicy(wire(next)).policy, next);
  for (const mutation of [
    { schema: policy.schema }, { prior_recording_sha256 }, { prior_physical_pre_sha256 },
    { root_index: 0 }, { storage_row: false }, { endpoint_frame: 64 },
    { capture_sha256: '0'.repeat(64) }, { selected_address_sha256: undefined },
  ]) {
    assert.throws(() => readPolicy(wire({ ...next, ...mutation })), /EXACT_(FIXED|FROZEN)_OWNER/);
  }
});

async function withNativeBattle(run) {
  const stream = new BattleStream({ keepAlive: true });
  const result = { formatid: policy.formatid, strictChoices: true, seed: '1,2,3,4' };
  try {
    await stream.write('>start ' + JSON.stringify(result));
    return run(stream.battle);
  } finally {
    await stream.writeEnd();
  }
}

test('the fixed root and literal expected start are independently required', async () => {
  assert.throws(() => readPolicy(wire({ ...policy, root_index: 1 })), /EXACT_FIXED_OWNER/);
  assert.throws(() => readPolicy(wire({ ...policy, reference_owner: 'p1' })), /EXACT_FIXED_OWNER/);
  await withNativeBattle(battle => {
    const original = battle.reportExactHP;
    const visibility = createVisibility(wire(policy), Battle);
    assert.throws(() => visibility.apply(battle, { ...context, stdinSha256: 'a'.repeat(64) }), /START_BINDING_DIFFERS/);
    assert.equal(battle.reportExactHP, original);
  });
});

test('there is one pre-player visibility transition on the original Battle', async () => {
  await withNativeBattle(battle => {
    const visibility = createVisibility(wire(policy), Battle);
    const before = Object.getOwnPropertyDescriptors(battle);
    const event = visibility.apply(battle, context);
    assert.equal(event.policy_sha256, readPolicy(wire(policy)).sha256);
    assert.equal(event.installed_players, 0);
    before.reportExactHP.value = false;
    assert.deepEqual(Object.getOwnPropertyDescriptors(battle), before);
    assert.throws(() => visibility.apply(battle, context), /START_BINDING_DIFFERS/);
    assert.throws(() => visibility.verifyBoundary(battle), /PERCENTAGE_VISIBILITY_NOT_ACTIVE/);
  });
});

test('a late attempt and an accessor are refused before any assignment', async () => {
  await withNativeBattle(battle => {
    const visibility = createVisibility(wire(policy), Battle);
    battle.turn = 1;
    assert.throws(() => visibility.apply(battle, context), /MUST_PRECEDE_PLAYER_INSTALLATION/);
    assert.equal(battle.reportExactHP, true);
    battle.turn = 0;
    let read = false;
    Object.defineProperty(battle, 'reportExactHP', { get() { read = true; return true; } });
    assert.throws(() => visibility.apply(battle, context), /INITIAL_HP_VISIBILITY_DIFFERS/);
    assert.equal(read, false);
  });
});
