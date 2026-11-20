'use strict';

const { createHash } = require('node:crypto');

const V1 = 'pokezero.fresh200.owner_visibility.v1';
const V2 = 'pokezero.fresh200.owner_visibility.v2';
const FORMAT = 'gen3customgame@@@sleepclausemod';
const NATIVE_COMMIT = 'f76228a1354b5d0f307ca2d16101294ad3a2308b';
const COMPILED = 'd22c0cc51aa32b983c2946108834418e824b0056a727e121e19dc56919350a01';
const CAPTURE = 'f6c606ffbce81b0b43351d88188c448d5b54a16334e6f75c20681aca2800b13a';
const DEVICE_REPLAY = 'f6520ff94bbea899f5ba05dcd80cfba8fdc1f0628626f20956f58aaf7de52b29';
const DIGEST_FIELDS = [
  'compiled_tree_sha256', 'scope_receipt_sha256', 'fixed_input_binding_sha256', 'start_wire_sha256',
];
const V1_DIGEST_FIELDS = [...DIGEST_FIELDS, 'prior_recording_sha256', 'prior_physical_pre_sha256'];
const V2_DIGEST_FIELDS = [
  ...DIGEST_FIELDS, 'capture_sha256', 'device_replay_sha256', 'ordered_roster_sha256', 'selected_address_sha256',
];
const BASE_FIELDS = [
  'schema', 'formatid', 'native_commit', 'root_index', 'stream_id', 'reference_owner',
  'endpoint_frame', 'session', 'battle_id',
];
const V1_FIELDS = [...BASE_FIELDS, ...V1_DIGEST_FIELDS].sort().join(',');
const V2_FIELDS = [...BASE_FIELDS, ...V2_DIGEST_FIELDS, 'storage_row'].sort().join(',');

function need(condition, message) {
  if (!condition) throw Error(message);
}

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function readPolicy(bytes) {
  need(Buffer.isBuffer(bytes) && bytes.length > 0 && bytes.length <= 4096
    && bytes[bytes.length - 1] === 10, 'BOUNDED_OWNER_VISIBILITY_POLICY_REQUIRED');
  const policy = JSON.parse(bytes.toString('utf8'));
  const version2 = policy?.schema === V2;
  const fields = version2 ? V2_FIELDS : V1_FIELDS;
  const digests = version2 ? V2_DIGEST_FIELDS : V1_DIGEST_FIELDS;
  need(policy && Object.keys(policy).sort().join(',') === fields && policy.formatid === FORMAT
    && policy.native_commit === NATIVE_COMMIT && policy.compiled_tree_sha256 === COMPILED
    && [policy.session, policy.battle_id].every(value => typeof value === 'string' && /^[A-Za-z0-9_.:-]{1,200}$/.test(value))
    && digests.every(key => typeof policy[key] === 'string' && /^[a-f0-9]{64}$/.test(policy[key])),
  'EXACT_FIXED_OWNER_VISIBILITY_POLICY_REQUIRED');
  if (version2) {
    need(policy.capture_sha256 === CAPTURE && policy.device_replay_sha256 === DEVICE_REPLAY
      && Number.isInteger(policy.root_index) && policy.root_index > 0 && policy.root_index < 200
      && Number.isInteger(policy.storage_row) && policy.storage_row >= 0 && policy.storage_row < 256
      && Number.isInteger(policy.stream_id) && policy.stream_id > 0 && policy.stream_id < 2 ** 31
      && Number.isInteger(policy.endpoint_frame) && policy.endpoint_frame >= 8 && policy.endpoint_frame < 64
      && ['p1', 'p2'].includes(policy.reference_owner), 'EXACT_FROZEN_OWNER_VISIBILITY_ADDRESS_REQUIRED');
  } else {
    need(policy.schema === V1 && policy.root_index === 0 && policy.stream_id === 2120000000
      && policy.reference_owner === 'p2' && policy.endpoint_frame === 8,
    'EXACT_FIXED_OWNER_VISIBILITY_POLICY_REQUIRED');
  }
  return Object.freeze({ policy: Object.freeze(policy), sha256: digest(bytes) });
}

function dataProperty(object, key) {
  const descriptor = Object.getOwnPropertyDescriptor(object, key);
  need(descriptor && Object.hasOwn(descriptor, 'value'), 'NATIVE_VISIBILITY_OWN_DATA_PROPERTY_REQUIRED');
  return descriptor.value;
}

function sameDescriptor(original, current, changed) {
  return current && original.configurable === current.configurable
    && original.enumerable === current.enumerable && original.writable === current.writable
    && original.get === current.get && original.set === current.set
    && Object.is(current.value, changed ? false : original.value);
}

function createVisibility(bytes, Battle) {
  const admission = readPolicy(bytes);
  const { policy, sha256 } = admission;
  let appliedBattle = null;
  return Object.freeze({
    policy,
    sha256,
    apply(battle, { session, battleId, formatid, stdinSha256 }) {
      need(!appliedBattle && battle instanceof Battle && session === policy.session
        && battleId === policy.battle_id && formatid === policy.formatid && stdinSha256 === policy.start_wire_sha256,
      'OWNER_VISIBILITY_START_BINDING_DIFFERS');
      const sides = dataProperty(battle, 'sides');
      const format = dataProperty(battle, 'format');
      need(dataProperty(battle, 'started') === false && dataProperty(battle, 'turn') === 0
        && Array.isArray(sides) && sides.length === 2 && sides.every(side => side === null)
        && dataProperty(format, 'id') === 'gen3customgame'
        && dataProperty(battle, 'ruleTable').has('sleepclausemod'),
      'OWNER_VISIBILITY_MUST_PRECEDE_PLAYER_INSTALLATION');
      const before = Object.getOwnPropertyDescriptors(battle);
      const exact = before.reportExactHP;
      need(exact && Object.hasOwn(exact, 'value') && exact.writable === true
        && exact.value === true && dataProperty(battle, 'reportPercentages') === false,
      'NATIVE_INITIAL_HP_VISIBILITY_DIFFERS');
      battle.reportExactHP = false;
      const after = Object.getOwnPropertyDescriptors(battle);
      const keys = Reflect.ownKeys(before);
      need(keys.length === Reflect.ownKeys(after).length
        && keys.every(key => Object.hasOwn(after, key) && sameDescriptor(before[key], after[key], key === 'reportExactHP')),
      'NATIVE_VISIBILITY_CHANGED_UNRELATED_PROPERTY');
      appliedBattle = battle;
      return Object.freeze({ policy_sha256: sha256, before_exact: true, after_exact: false,
        before_percentage: false, after_percentage: false, turn: 0, installed_players: 0 });
    },
    verifyBoundary(battle) {
      need(battle === appliedBattle && battle instanceof Battle
        && dataProperty(battle, 'reportExactHP') === false && dataProperty(battle, 'reportPercentages') === true,
      'NATIVE_OWNER_PERCENTAGE_VISIBILITY_NOT_ACTIVE');
      return Object.freeze({ policy_sha256: sha256, report_exact_hp: false, report_percentages: true });
    },
  });
}

module.exports = { createVisibility, readPolicy };
