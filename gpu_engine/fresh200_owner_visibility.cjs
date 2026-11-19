'use strict';

const { createHash } = require('node:crypto');

const SCHEMA = 'pokezero.fresh200.owner_visibility.v1';
const FORMAT = 'gen3customgame@@@sleepclausemod';
const NATIVE_COMMIT = 'f76228a1354b5d0f307ca2d16101294ad3a2308b';
const COMPILED = 'd22c0cc51aa32b983c2946108834418e824b0056a727e121e19dc56919350a01';
const DIGEST_FIELDS = [
  'compiled_tree_sha256', 'scope_receipt_sha256', 'fixed_input_binding_sha256',
  'start_wire_sha256', 'prior_recording_sha256', 'prior_physical_pre_sha256',
];
const FIELDS = [
  'schema', 'formatid', 'native_commit', 'root_index', 'stream_id', 'reference_owner',
  'endpoint_frame', 'session', 'battle_id', ...DIGEST_FIELDS,
].sort().join(',');

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
  need(policy && Object.keys(policy).sort().join(',') === FIELDS
    && policy.schema === SCHEMA && policy.formatid === FORMAT
    && policy.native_commit === NATIVE_COMMIT && policy.compiled_tree_sha256 === COMPILED
    && policy.root_index === 0 && policy.stream_id === 2120000000
    && policy.reference_owner === 'p2' && policy.endpoint_frame === 8
    && [policy.session, policy.battle_id].every(value => typeof value === 'string' && /^[A-Za-z0-9_.:-]{1,200}$/.test(value))
    && DIGEST_FIELDS.every(key => typeof policy[key] === 'string' && /^[a-f0-9]{64}$/.test(policy[key])),
  'EXACT_FIXED_OWNER_VISIBILITY_POLICY_REQUIRED');
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
