'use strict';
// NEW opt-in registration evidence guards. All methods/classes are synthetic;
// no actual Native tree/game, chance source, frozen root or old suite runs.
const test=require('node:test');
const assert=require('node:assert/strict');
const {WitnessCore,ZERO,hash}=require('../gpu_engine/fresh200_witness_core.cjs');
const {createRecorder,METHODS}=require('../gpu_engine/fresh200_witness_native.cjs');
const {createObserver}=require('../gpu_engine/fresh200_witness_bridge.cjs');
const site=owner=>({kind:'bootstrap-method',owner});
const packet=(index=0,phase='start',parent=ZERO)=>({schema:'pokezero.native_witness_control.v1',run:'hook-fixture',stream:0,reset:0,
  command:index,phase,request:index,nonce:'hook-fixture/0/0/'+index,chance_nonce:null,previous_receipt_hash:parent});
function coreFixture(extra={}){
  const bytes=[],rows=[];const core=new WitnessCore({registrationEvidence:true,sourceRegistryHash:'6'.repeat(64),
    appendAudit:data=>{bytes.push(data);rows.push(JSON.parse(data));},...extra});
  return {core,bytes,rows,begin:(index=0,phase='start')=>core.begin(packet(index,phase,core.lastReceiptHash),{battleId:'b',phase})};
}
function nativeFixture(overrides={}){
  const classes=Object.fromEntries(Object.keys(METHODS).map(name=>[name,class{}]));
  for(const [kind,names] of Object.entries(METHODS))for(const name of names)
    Object.defineProperty(classes[kind].prototype,name,{value:function(){return undefined;},writable:true,configurable:true});
  for(const [kind,names] of Object.entries(overrides))for(const [name,fn] of Object.entries(names))
    Object.defineProperty(classes[kind].prototype,name,{value:fn,writable:true,configurable:true});
  classes.Battle.prototype.setPlayer=function(side){this.players.push(side);return side;};
  const encore=function(){return true;};classes.Pokemon.prototype.addVolatile=encore;
  const bytes=[],rows=[],receipts=[];let control=packet(),recorder;
  const observer=createObserver({Battle:classes.Battle,registrationEvidence:true,sourceRegistryHash:'6'.repeat(64),
    loadControl:()=>control,appendAudit:data=>{bytes.push(data);rows.push(JSON.parse(data));},appendReceipt:data=>receipts.push(data),
    nativeFactory:core=>recorder=createRecorder(core,classes)});
  const battle=new classes.Battle();battle.players=[];battle.sides=[];battle.actions=new classes.BattleActions();battle.queue=new classes.BattleQueue();
  battle.actions.battle=battle;battle.queue.battle=battle;
  const token=observer.commandBegin({type:'start',battleId:'b'});battle.setPlayer('p1');battle.setPlayer('p2');
  return {classes,observer,core:observer.core,battle,bytes,rows,receipts,recorder,encore,
    finish:()=>observer.commandEnd(token,{originalThrew:false})};
}
function registrations(rows){return rows.filter(row=>row.operation==='HOOK_REGISTERED');}
function declaredDigest(rows){
  const hooks=registrations(rows).map(row=>{const e=row.evidence;return {operation:e.operation,originalSha256:e.originalSha256,kind:e.kind};});
  hooks.sort((a,b)=>a.operation<b.operation?-1:a.operation>b.operation?1:0);return hash(Buffer.from(JSON.stringify(hooks)+'\n'));
}

test('bootstrap declarations flush after begin and exactly reconstruct receipt roster digest',()=>{
  const f=coreFixture(),object={first(){return 1;},second(){return 2;}};
  for(const name of ['first','second'])f.core.installMethod(object,name,{operation:'Synthetic.'+name,registrationSite:site('Synthetic')});
  assert.equal(f.rows.length,0);const token=f.begin();const receipt=f.core.end(token);
  assert.deepEqual(f.rows.map(r=>r.operation),['COMMAND_BEGIN','HOOK_REGISTERED','HOOK_REGISTERED','COMMAND_END']);
  const hooks=registrations(f.rows);assert.deepEqual(hooks.map(r=>r.evidence.hook_id),[0,1]);
  assert.equal(hooks[0].evidence.target_token,hooks[1].evidence.target_token);assert.equal(hooks[0].frame,null);
  assert.equal(hooks[0].evidence.discovery_call_id,null);assert.equal(receipt.hookIdentityDigest,declaredDigest(f.rows));assert.equal(receipt.poisoned,false);
  let previous=ZERO;for(const [index,bytes] of f.bytes.entries()){assert.equal(f.rows[index].sequence,index);assert.equal(f.rows[index].previousHash,previous);previous=hash(bytes);}
  assert.equal(previous,receipt.chainHash);
});

test('instance installations in enter appear after parent enter and before original work',()=>{
  const f=coreFixture();let calls=0;const child={inner(){return 'inner';}},result={same:true};
  const target={outer(){calls++;assert.equal(f.rows.at(-1).operation,'HOOK_REGISTERED');return result;}};
  f.core.installMethod(target,'outer',{operation:'Synthetic.outer',registrationSite:site('Synthetic'),enter(){
    f.core.installMethod(child,'inner',{operation:'Child.inner',registrationSite:{kind:'instance-method',owner:'Child'}});return null;}});
  const token=f.begin();assert.equal(target.outer(),result);assert.equal(calls,1);const receipt=f.core.end(token);
  const installed=registrations(f.rows).find(r=>r.evidence.operation==='Child.inner');
  assert.equal(f.rows[installed.sequence-1].operation,'CALL_ENTER');assert.equal(installed.frame.operation,'Synthetic.outer');
  assert.equal(installed.evidence.discovery_call_id,installed.frame.callId);assert.equal(receipt.hookIdentityDigest,declaredDigest(f.rows));assert.equal(receipt.poisoned,false);
});

test('actual Native getCallback returns register unused functions and preserve exact object tokens',()=>{
  function unused(){return 'unused';}
  const f=nativeFixture({Battle:{getCallback:function(value){return value;}}});
  assert.equal(f.battle.getCallback(unused),unused);assert.equal(f.battle.getCallback(unused),unused);
  const rows=registrations(f.rows).filter(r=>r.evidence.kind==='callback-own-apply');assert.equal(rows.length,1);
  const registered=rows[0],returned=f.rows.find(r=>r.operation==='CALL_RETURN'&&r.frame.callId===registered.frame.callId);
  const entered=f.rows.find(r=>r.operation==='CALL_ENTER'&&r.frame.callId===registered.frame.callId);
  assert.deepEqual(registered.evidence.discovery_site,{kind:'getCallback-result'});
  assert.equal(returned.evidence.result.function_token,registered.evidence.target_token);
  assert.equal(entered.evidence.arguments.items[0].function_token,registered.evidence.target_token);
  assert.equal(returned.evidence.result.$function,registered.evidence.originalSha256);assert.ok(registered.sequence<returned.sequence);
  assert.equal(f.rows.filter(r=>r.operation==='CALL_ENTER'&&r.frame.operation.startsWith('Callback:')).length,0);
  const receipt=f.finish();assert.equal(receipt.poisoned,false);assert.equal(receipt.hookIdentityDigest,declaredDigest(f.rows));
  assert.equal(f.classes.Pokemon.prototype.addVolatile,f.encore);assert.equal(f.recorder.manifest().causalCoverageComplete,false);
});

test('distinct function objects with identical text remain distinct registration multiplicities',()=>{
  const make=()=>function same(){return 17;};const first=make(),second=make();assert.notEqual(first,second);
  const f=nativeFixture({Battle:{getCallback:function(value){return value;}}});f.battle.getCallback(first);f.battle.getCallback(second);
  const rows=registrations(f.rows).filter(r=>r.evidence.kind==='callback-own-apply');assert.equal(rows.length,2);
  assert.equal(rows[0].evidence.originalSha256,rows[1].evidence.originalSha256);assert.equal(rows[0].evidence.operation,rows[1].evidence.operation);
  assert.notEqual(rows[0].evidence.target_token,rows[1].evidence.target_token);assert.equal(f.finish().hookIdentityDigest,declaredDigest(f.rows));
});

test('handler discovery joins exact return-array positions without invoking callbacks',()=>{
  const first=function(){return 1;},second=function(){return 2;};let calls=0;
  const f=nativeFixture({Battle:{findEventHandlers:function(){calls++;return [{callback:first},{callback:second}];}}});
  const actual=f.battle.findEventHandlers();assert.equal(calls,1);assert.equal(actual[0].callback,first);assert.equal(actual[1].callback,second);
  const rows=registrations(f.rows).filter(r=>r.evidence.kind==='callback-own-apply');
  assert.deepEqual(rows.map(r=>r.evidence.discovery_site),[{index:0,kind:'handler-result'},{index:1,kind:'handler-result'}]);
  const returned=f.rows.find(r=>r.operation==='CALL_RETURN'&&r.frame.operation==='Battle.findEventHandlers');
  for(const [index,row] of rows.entries()){
    const value=returned.evidence.result.items[index].fields.find(x=>x[0]==='callback')[1];
    assert.equal(value.function_token,row.evidence.target_token);assert.equal(value.$function,row.evidence.originalSha256);assert.ok(row.sequence<returned.sequence);
  }assert.equal(f.finish().poisoned,false);
});

test('singleEvent selection records inherited effect versus custom callback and original apply ancestry',()=>{
  let calls=0;const result={same:true};function selected(){calls++;return result;}
  const effect=Object.create({onProbe:selected});
  const f=nativeFixture({Battle:{singleEvent:function(event,effect,_s,_t,_o,_e,_r,custom){return (custom||effect['on'+event]).apply(this,[]);}}});
  assert.equal(f.battle.singleEvent('Probe',effect,null,null,null,null,null,false),result);assert.equal(calls,1);
  const registered=registrations(f.rows).find(r=>r.evidence.kind==='callback-own-apply');
  const parent=f.rows.find(r=>r.operation==='CALL_ENTER'&&r.frame.operation==='Battle.singleEvent');
  assert.equal(parent.evidence.callbackSelection.selector,'effect-property');assert.equal(parent.evidence.callbackSelection.function_token,registered.evidence.target_token);
  assert.equal(parent.evidence.callbackSelection.originalSha256,registered.evidence.originalSha256);
  assert.equal(registered.sequence,parent.sequence+1);const invoked=f.rows.find(r=>r.operation==='CALL_ENTER'&&r.frame.operation===registered.evidence.operation);
  assert.equal(invoked.frame.parentCall,parent.frame.callId);assert.ok(invoked.sequence>registered.sequence);
  const custom=function(){return 3;};assert.equal(f.battle.singleEvent('Probe',effect,null,null,null,null,null,custom),3);
  assert.equal(f.rows.filter(r=>r.operation==='CALL_ENTER'&&r.frame.operation==='Battle.singleEvent').at(-1).evidence.callbackSelection.selector,'custom');
  assert.equal(f.finish().poisoned,false);
});

test('registration journal failure preserves original return and throw identities without retry',()=>{
  for(const throwing of [false,true]){
    let calls=0;const result={original:true},sentinel={thrown:true},rows=[];
    const f=coreFixture({appendAudit:data=>{const row=JSON.parse(data);if(row.operation==='HOOK_REGISTERED'&&row.evidence.discovery_call_id!==null)throw Error('SYNTHETIC_IO');rows.push(row);}});
    const child={work(){return 0;}},target={outer(){calls++;if(throwing)throw sentinel;return result;}};
    f.core.installMethod(target,'outer',{operation:'Synthetic.outer',registrationSite:site('Synthetic'),enter(){
      f.core.installMethod(child,'work',{operation:'Child.work',registrationSite:{kind:'instance-method',owner:'Child'}});return null;}});
    const token=f.begin();if(throwing)assert.throws(()=>target.outer(),value=>value===sentinel);else assert.equal(target.outer(),result);
    assert.equal(calls,1);assert.equal(f.core.end(token,{originalThrew:throwing}).poisoned,true);assert.equal(f.core.frames.length,0);
  }
});

test('registration IDs and target tokens remain contiguous across command boundaries',()=>{
  const f=coreFixture(),object={discover(fn){return fn;}};
  f.core.installMethod(object,'discover',{operation:'Synthetic.discover',registrationSite:site('Synthetic'),leave(_before,_self,_args,outcome){
    const fn=outcome.value;f.core.instrumentCallback(fn,'Callback:'+hash(Function.prototype.toString.call(fn)),{
      operation:'Callback:'+hash(Function.prototype.toString.call(fn)),registrationSite:{kind:'getCallback-result'}});return null;}});
  let token=f.begin();const first=function first(){return 1;};object.discover(first);const one=f.core.end(token);
  token=f.begin(1,'snapshot');object.discover(first);object.discover(function second(){return 2;});const two=f.core.end(token);
  const hooks=registrations(f.rows);assert.deepEqual(hooks.map(r=>r.evidence.hook_id),[0,1,2]);
  assert.notEqual(hooks[1].evidence.target_token,hooks[2].evidence.target_token);assert.equal(two.control.previous_receipt_hash,hash(Buffer.from(JSON.stringify(one)+'\n')));
  assert.equal(two.hookIdentityDigest,declaredDigest(f.rows));assert.equal(two.poisoned,false);
});

test('late out-of-command installations poison opt-in evidence and legacy mode adds no registration rows',()=>{
  const f=coreFixture(),token=f.begin();f.core.end(token);
  const object={work(){return 1;}};f.core.installMethod(object,'work',{operation:'Synthetic.work',registrationSite:site('Synthetic')});
  assert.equal(f.core.poisoned,true);assert.equal(f.core.faultCode,'HOOK_REGISTRATION_FAILED');
  const plain=coreFixture({registrationEvidence:false}),target={work(){return 2;}};
  plain.core.installMethod(target,'work',{operation:'Synthetic.work'});const old=plain.begin();assert.equal(target.work(),2);const receipt=plain.core.end(old);
  assert.equal(registrations(plain.rows).length,0);assert.equal(receipt.poisoned,false);
});

test('discovery metadata getters are refused without execution or suppression of the original',()=>{
  let getters=0,calls=0;const f=coreFixture(),object={work(){calls++;return 19;}},metadata={};
  Object.defineProperty(metadata,'kind',{enumerable:true,get(){getters++;throw Error('UNTRUSTED_ACCESSOR');}});
  f.core.installMethod(object,'work',{operation:'Synthetic.work',registrationSite:metadata});
  const token=f.begin();assert.equal(object.work(),19);assert.equal(calls,1);assert.equal(getters,0);assert.equal(f.core.end(token).poisoned,true);
});
