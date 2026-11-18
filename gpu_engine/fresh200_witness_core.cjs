'use strict';
// Passive process-local audit plumbing. Never admission or a battle projection.
const crypto = require('node:crypto');
const apply = Reflect.apply;
const nativeApply = Function.prototype.apply;
const functionText = Function.prototype.toString;
const ZERO = '0'.repeat(64);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function literal(value, seen = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Object.is(value, -0) ? {$number: '-0'} : value;
  }
  if (!value || typeof value !== 'object' || seen.has(value)) throw Error('NON_LITERAL_EVIDENCE');
  const proto = Object.getPrototypeOf(value);
  if (!Array.isArray(value) && proto !== Object.prototype && proto !== null) throw Error('NON_LITERAL_EVIDENCE');
  seen.add(value);
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const result = Array.isArray(value) ? [] : {};
  const keys = Array.isArray(value) ? Array.from({length:value.length}, (_, i) => String(i)) : Object.keys(value).sort();
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw Error('EVIDENCE_ACCESSOR_OR_HOLE');
    // Define data properties, including a literal "__proto__" key.
    Object.defineProperty(result, key, {value:literal(descriptor.value, seen), enumerable:true, writable:true, configurable:true});
  }
  seen.delete(value);
  return result;
}

class WitnessCore {
  constructor({appendAudit, sourceRegistryHash, maxRows = 100000, maxBytes = 64*1024*1024}) {
    if (typeof appendAudit !== 'function' || !/^[a-f0-9]{64}$/.test(sourceRegistryHash) ||
        !Number.isSafeInteger(maxRows) || maxRows <= 0 || !Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw Error('BOUNDED_WITNESS_WRITER_AND_SOURCE_REQUIRED');
    }
    this.appendAudit = appendAudit; this.sourceRegistryHash = sourceRegistryHash;
    this.maxRows = maxRows; this.maxBytes = maxBytes;
    this.sequence = 0; this.chainHash = ZERO; this.callId = 0;
    this.poisoned = false; this.faultCode = null; this.active = null;
    this.frames = []; this.hooks = []; this.callbacks = new WeakMap();
    this.commands = new Map(); this.nonces = new Set(); this.lastReceiptHash = ZERO;
  }
  poison(code) { this.poisoned = true; this.faultCode ??= code; }
  observe(code, callback) {
    try { return callback(); } catch { this.poison(code); return undefined; }
  }
  record(operation, evidence = null, frame = null) {
    if (!this.active) throw Error('MISSING_COMMAND_CONTEXT');
    const context = this.active;
    const row = {schema:'pokezero.fresh200.native_witness_row.v1',
      run:context.control.run, stream:context.control.stream, reset:context.control.reset,
      command:context.control.command, nonce:context.control.nonce, phase:context.control.phase,
      battleId:context.battleId, sequence:this.sequence, previousHash:this.chainHash,
      operation, frame, evidence:literal(evidence)};
    const bytes = Buffer.from(JSON.stringify(row)+'\n', 'utf8');
    if (context.rows >= this.maxRows || context.bytes+bytes.length > this.maxBytes) throw Error('AUDIT_BOUND');
    const result = this.appendAudit(bytes);
    if (result && typeof result.then === 'function') throw Error('SYNCHRONOUS_AUDIT_WRITER_REQUIRED');
    this.chainHash = hash(bytes); this.sequence++; context.rows++; context.bytes += bytes.length;
  }
  begin(control, {battleId, phase}) {
    return this.observe('COMMAND_BEGIN_FAILED', () => {
      if (this.active || this.frames.length) throw Error('CONCURRENT_COMMAND');
      const value = literal(control);
      const expected = ['schema','run','stream','reset','command','phase','request','nonce','chance_nonce','previous_receipt_hash'].sort();
      if (Object.keys(value).sort().join(',') !== expected.join(',') ||
          value.schema !== 'pokezero.native_witness_control.v1' ||
          typeof value.run !== 'string' || !value.run || typeof battleId !== 'string' || !battleId ||
          !Number.isSafeInteger(value.stream) || value.stream < 0 ||
          !Number.isSafeInteger(value.reset) || value.reset < 0 ||
          !Number.isSafeInteger(value.command) || value.command < 0 ||
          !Number.isSafeInteger(value.request) || value.request < 0 ||
          value.phase !== phase || !['start','choices','snapshot','end','close'].includes(phase) ||
          typeof value.nonce !== 'string' || !value.nonce || this.nonces.has(value.nonce) ||
          !(value.chance_nonce === null || typeof value.chance_nonce === 'string') ||
          value.previous_receipt_hash !== this.lastReceiptHash) throw Error('CONTROL_BINDING');
      const key = JSON.stringify([value.run, value.stream, value.reset]);
      if (value.command !== (this.commands.get(key) ?? -1)+1) throw Error('NONCONTIGUOUS_COMMAND');
      this.commands.set(key, value.command); this.nonces.add(value.nonce);
      this.active = {control:Object.freeze(value), battleId, firstSequence:this.sequence, rows:0, bytes:0};
      this.record('COMMAND_BEGIN');
      return this.active;
    });
  }
  end(token, {originalThrew = false, boundary = null} = {}) {
    let receipt;
    this.observe('COMMAND_END_FAILED', () => {
      if (!token || token !== this.active) throw Error('WRONG_COMMAND_TOKEN');
      if (this.frames.length) this.poison('OPEN_CALL_FRAMES');
      if (originalThrew) this.poison('ORIGINAL_COMMAND_THROW');
      this.verifyHooks();
      this.record('COMMAND_END', {originalThrew, boundary});
      receipt = {schema:'pokezero.fresh200.native_witness_receipt.v1', control:token.control,
        battleId:token.battleId, firstSequence:token.firstSequence, lastSequence:this.sequence-1,
        rowCount:token.rows, byteCount:token.bytes, chainHash:this.chainHash,
        openFrameCount:this.frames.length, poisoned:this.poisoned, faultCode:this.faultCode,
        sourceRegistryHash:this.sourceRegistryHash, hookIdentityDigest:this.hookIdentityDigest(),
        boundary:literal(boundary), qualification:false};
      const bytes = Buffer.from(JSON.stringify(receipt)+'\n', 'utf8');
      this.lastReceiptHash = hash(bytes); this.active = null;
    });
    return receipt;
  }
  wrap(original, {operation, enter = () => null, leave = () => null}) {
    if (typeof original !== 'function' || typeof operation !== 'string' || !operation ||
        typeof enter !== 'function' || typeof leave !== 'function') throw Error('EXPLICIT_PASSIVE_OBSERVER_REQUIRED');
    const core = this;
    return function (...args) {
      let frame, before;
      core.observe('OBSERVER_ENTER_FAILED', () => {
        frame = {callId:core.callId++, parentCall:core.frames.at(-1)?.callId ?? null, operation};
        core.frames.push(frame); before = enter(this, args);
        core.record('CALL_ENTER', before, frame);
      });
      let value, thrown, didThrow = false;
      try { value = apply(original, this, args); }
      catch (error) { didThrow = true; thrown = error; }
      core.observe('OBSERVER_LEAVE_FAILED', () => {
        try { core.record(didThrow ? 'CALL_THROW' : 'CALL_RETURN',
          leave(before, this, args, {didThrow, value, thrown}), frame); }
        finally {
          if (!frame || core.frames.at(-1) !== frame) core.poison('UNBALANCED_CALL_FRAMES');
          else core.frames.pop();
        }
      });
      if (didThrow) throw thrown;
      return value;
    };
  }
  installMethod(target, name, observer) {
    const descriptor = Object.getOwnPropertyDescriptor(target, name);
    if (!descriptor || typeof descriptor.value !== 'function') throw Error('EXACT_DATA_METHOD_REQUIRED');
    const original = descriptor.value; const wrapped = this.wrap(original, observer);
    Object.defineProperty(target, name, {...descriptor, value:wrapped});
    this.hooks.push({target, name, wrapped, operation:observer.operation,
      originalSha256:hash(apply(functionText, original, [])), kind:'method'});
    return original;
  }
  instrumentCallback(fn, identity, observer) {
    const prior = this.callbacks.get(fn);
    if (prior) {
      if (prior.identity !== identity) throw Error('CALLBACK_SOURCE_ALIAS');
      return fn;
    }
    if (typeof fn !== 'function' || Object.hasOwn(fn, 'apply') || fn.apply !== nativeApply || !Object.isExtensible(fn)) {
      throw Error('UNSUPPORTED_CALLBACK_APPLY_SEAM');
    }
    const wrapped = this.wrap(nativeApply, observer);
    Object.defineProperty(fn, 'apply', {value:wrapped, enumerable:false, writable:false, configurable:true});
    this.callbacks.set(fn, {identity, wrapped});
    this.hooks.push({target:fn, name:'apply', wrapped, operation:identity,
      originalSha256:hash(apply(functionText, fn, [])), kind:'callback-own-apply'});
    return fn;
  }
  verifyHooks() {
    for (const hook of this.hooks) {
      if (Object.getOwnPropertyDescriptor(hook.target, hook.name)?.value !== hook.wrapped) this.poison('HOOK_IDENTITY_CHANGED');
    }
  }
  hookIdentityDigest() {
    return hash(Buffer.from(JSON.stringify(this.hooks.map(({operation, originalSha256, kind}) =>
      ({operation, originalSha256, kind})).sort((a,b) => a.operation < b.operation ? -1 : a.operation > b.operation ? 1 : 0))+'\n'));
  }
}

module.exports = {WitnessCore, literal, hash, ZERO};
