'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {WitnessCore, literal, hash, ZERO} = require('../gpu_engine/fresh200_witness_core.cjs');
function fixture(options = {}) {
  const bytes = [];
  const core = new WitnessCore({appendAudit:row => { bytes.push(Buffer.from(row)); },
    sourceRegistryHash:'1'.repeat(64), ...options});
  const packet = (command = 0, phase = 'choices', extra = {}) => ({
    schema:'pokezero.native_witness_control.v1', run:'fixture', stream:17, reset:0,
    command, phase, request:command, nonce:'fixture/17/0/'+command,
    chance_nonce:phase === 'choices' ? '17:'+command : null,
    previous_receipt_hash:core.lastReceiptHash, ...extra});
  return {core, bytes, packet, rows:() => bytes.map(b => JSON.parse(b))};
}

test('nested original methods run once with exact receivers, arguments and returns', () => {
  const f = fixture(); const marker = {}; const receiver = {calls:0}; const values = [];
  receiver.inner = f.core.wrap(function (value) { assert.equal(this, receiver); values.push(value); return marker; },
    {operation:'inner', enter:() => ({at:'inner'})});
  receiver.outer = f.core.wrap(function (value) { this.calls++; return this.inner(value); },
    {operation:'outer', enter:() => ({at:'outer'})});
  const token = f.core.begin(f.packet(), {battleId:'b',phase:'choices'});
  assert.equal(receiver.outer(marker), marker); assert.equal(receiver.calls,1); assert.deepEqual(values,[marker]);
  const receipt = f.core.end(token); assert.equal(receipt.poisoned,false); assert.equal(receipt.rowCount,6);
  const calls = f.rows().filter(row => row.operation === 'CALL_ENTER');
  assert.equal(calls[1].frame.parentCall,calls[0].frame.callId); assert.equal(receipt.openFrameCount,0);
});

test('every original thrown value survives a simultaneous observer failure', () => {
  for (const thrown of [undefined, null, false, 0, Symbol('throw'), new Error('original')]) {
    const f = fixture(); let count = 0, caught = false, actual;
    const fn = f.core.wrap(() => { count++; throw thrown; },
      {operation:'throw', leave:() => { throw Error('observer'); }});
    const token = f.core.begin(f.packet(), {battleId:'b',phase:'choices'});
    try { fn(); } catch (error) { caught = true; actual = error; }
    assert.equal(caught,true); assert.equal(actual,thrown); assert.equal(count,1);
    assert.equal(f.core.end(token).poisoned,true);
  }
});

test('enter and leave observation faults never suppress or retry the original', () => {
  for (const location of ['enter','leave']) {
    const f = fixture(); const marker = {}; let count = 0;
    const observer = {operation:location, [location]:() => { throw Error('observer'); }};
    const fn = f.core.wrap(() => { count++; return marker; },observer);
    const token = f.core.begin(f.packet(), {battleId:'b',phase:'choices'});
    assert.equal(fn(),marker); assert.equal(count,1);
    const receipt = f.core.end(token); assert.equal(receipt.poisoned,true); assert.equal(receipt.openFrameCount,0);
    const next = f.core.begin(f.packet(1),{battleId:'b',phase:'choices'});
    assert.equal(f.core.end(next).poisoned,true);
  }
});

test('callback identity and nonenumerable own apply preserve exact execution', () => {
  const f = fixture(); let count = 0; const marker = {}; const receiver = {};
  const callback = function (value) { count++; assert.equal(this,receiver); return value; };
  const same = f.core.instrumentCallback(callback,'fixture.callback',{operation:'callback'});
  assert.equal(same,callback); assert.deepEqual(Object.keys(callback),[]);
  assert.equal(f.core.instrumentCallback(callback,'fixture.callback',{operation:'callback'}),callback);
  const token = f.core.begin(f.packet(),{battleId:'b',phase:'choices'});
  assert.equal(callback.apply(receiver,[marker]),marker); assert.equal(count,1);
  assert.equal(f.core.end(token).poisoned,false);
  assert.throws(() => f.core.instrumentCallback(callback,'different',{operation:'callback'}),/SOURCE_ALIAS/);
});

test('frozen callback parents are safe and unsupported callback seams reject before replay', () => {
  const f = fixture(); const callback = () => 7; Object.freeze({callback});
  assert.equal(f.core.instrumentCallback(callback,'frozen-parent',{operation:'callback'}),callback);
  const ownApply = () => 0; ownApply.apply = Function.prototype.apply;
  assert.throws(() => f.core.instrumentCallback(ownApply,'own',{operation:'own'}),/UNSUPPORTED/);
  assert.throws(() => f.core.instrumentCallback(Object.preventExtensions(() => 0),'sealed',{operation:'sealed'}),/UNSUPPORTED/);
});

test('installed method identity changes poison qualification without repair', () => {
  const f = fixture(); const object = {method() {return 1;}};
  const original = f.core.installMethod(object,'method',{operation:'method'});
  const token = f.core.begin(f.packet(),{battleId:'b',phase:'choices'});
  object.method = original;
  assert.equal(f.core.end(token).faultCode,'HOOK_IDENTITY_CHANGED'); assert.equal(object.method,original);
});

test('exact UTF8 framed audit bytes form a contiguous global hash chain', () => {
  const f = fixture(); let previous = ZERO;
  for (let i=0;i<2;i++) {
    const token = f.core.begin(f.packet(i,'snapshot'),{battleId:'b',phase:'snapshot'});
    f.core.record('EXPLICIT_BOUNDARY',{text:'\u03b1',zero:-0});
    const receipt = f.core.end(token,{boundary:{kind:'ordinary'}});
    assert.equal(receipt.byteCount,f.bytes.slice(i*3,i*3+3).reduce((n,b) => n+b.length,0));
    assert.equal(receipt.qualification,false); assert.equal(f.core.lastReceiptHash,hash(Buffer.from(JSON.stringify(receipt)+'\n')));
  }
  for (const [i,bytes] of f.bytes.entries()) {
    const row = JSON.parse(bytes); assert.equal(row.sequence,i); assert.equal(row.previousHash,previous);
    assert.equal(bytes.at(-1),10); previous = hash(bytes);
  }
  assert.equal(previous,f.core.chainHash); assert.deepEqual(f.rows()[1].evidence.zero,{$number:'-0'});
});

test('zero-event commands still carry explicit begin and end receipts', () => {
  const f = fixture(); const token = f.core.begin(f.packet(0,'start'),{battleId:'b',phase:'start'});
  const receipt = f.core.end(token); assert.equal(receipt.rowCount,2);
  assert.deepEqual(f.rows().map(x => x.operation),['COMMAND_BEGIN','COMMAND_END']);
});

test('gaps, reused nonces, wrong parent receipt and concurrent commands fail closed', () => {
  for (const bad of [p => ({...p,command:2}), p => ({...p,previous_receipt_hash:'2'.repeat(64)})]) {
    const f = fixture(); assert.equal(f.core.begin(bad(f.packet()),{battleId:'b',phase:'choices'}),undefined);
    assert.equal(f.core.poisoned,true);
  }
  const f = fixture(); const first = f.packet(); const token = f.core.begin(first,{battleId:'b',phase:'choices'});
  assert.equal(f.core.begin(f.packet(),{battleId:'b',phase:'choices'}),undefined);
  assert.equal(f.core.end(token).poisoned,true);
  assert.equal(f.core.begin({...f.packet(1),nonce:first.nonce},{battleId:'b',phase:'choices'}),undefined);
});

test('getters, foreign objects, cycles and nonfinite evidence never execute a getter', () => {
  let getters = 0; const object = {};
  Object.defineProperty(object,'danger',{enumerable:true,get() {getters++; return 1;}});
  const cycle = {}; cycle.self = cycle;
  for (const value of [object,cycle,new (class Native {})(),NaN,Infinity,[,]]) assert.throws(() => literal(value));
  assert.equal(getters,0);
  const proto = JSON.parse('{"__proto__":{"literal":1}}'); const copy = literal(proto);
  assert.equal(Object.getPrototypeOf(copy),Object.prototype); assert.deepEqual(copy.__proto__,{literal:1});
});

test('audit write failure and row bounds preserve original returns but cannot produce clean receipts', () => {
  for (const options of [{appendAudit:() => {throw Error('disk');}}, {maxRows:1}, {maxBytes:1}, {appendAudit:() => Promise.resolve()}]) {
    const f = fixture(options); let count = 0;
    const fn = f.core.wrap(() => {count++; return 5;},{operation:'method'});
    const token = f.core.begin(f.packet(),{battleId:'b',phase:'choices'});
    assert.equal(fn(),5); assert.equal(count,1);
    const receipt = f.core.end(token); assert.equal(f.core.poisoned,true);
    assert.ok(receipt === undefined || receipt.poisoned);
  }
});

test('missing command context never prevents the original method call', () => {
  const f = fixture(); let count = 0;
  const fn = f.core.wrap(() => {count++; return 1;},{operation:'method'});
  assert.equal(fn(),1); assert.equal(count,1); assert.equal(f.core.poisoned,true);
});
