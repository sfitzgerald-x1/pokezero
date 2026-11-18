'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const path=require('node:path');
const {createObserver,wireValue,elidedProperties}=require('../gpu_engine/fresh200_witness_bridge.cjs');
const {ZERO}=require('../gpu_engine/fresh200_witness_core.cjs');
const control=(command,phase,parent=ZERO)=>({schema:'pokezero.native_witness_control.v1',run:'bridge-fixture',stream:0,reset:0,
  command,phase,request:command,nonce:'bridge-fixture/0/0/'+command,chance_nonce:null,previous_receipt_hash:parent});
function sides(){return Array.from({length:2},(_,n)=>{
  const team=Array.from({length:6},()=>({species:'duplicate'}));
  return {n,team,pokemon:team.map(set=>({set,level:80,happiness:255,pokeball:'pokeball'}))};
});}

test('ordinary port remains absent and configured observation faults cannot escape',async()=>{
  const port=await import(pathToFileURL(path.resolve('scripts/battle_bridge_witness_port.mjs')).href+'?fixture');
  assert.equal(port.witnessCommandBegin({type:'start'}),undefined);
  assert.equal(port.witnessSnapshotEnd('b',{}, {},{},null),undefined);
  let faults=0;const fail=()=>{throw Error('diagnostic');};
  port.setNativeWitness({commandBegin:fail,commandEnd:fail,snapshotBegin:fail,snapshotEnd:fail,poison:()=>{faults++;}});
  assert.equal(port.witnessCommandBegin({type:'start'}),undefined);
  assert.equal(port.witnessCommandEnd({},{}),undefined);
  assert.equal(port.witnessSnapshotBegin('b',{},null),undefined);
  assert.equal(port.witnessSnapshotEnd('b',{}, {},{},null),undefined);
  assert.equal(faults,4);
});

test('explicit snapshot port arguments preserve their exact identities',async()=>{
  const port=await import(pathToFileURL(path.resolve('scripts/battle_bridge_witness_port.mjs')).href+'?identity');
  const battle={},state={},requests={},marker={};let called=0;
  port.setNativeWitness({commandBegin:()=>null,commandEnd:()=>null,snapshotBegin:()=>null,poison:()=>{},
    snapshotEnd:(id,b,s,r,request)=>{called++;assert.equal(id,'b');assert.equal(b,battle);assert.equal(s,state);assert.equal(r,requests);assert.equal(request,'q');return marker;}});
  assert.equal(port.witnessSnapshotEnd('b',battle,state,requests,'q'),marker);assert.equal(called,1);
});

test('bridge observer closes snapshot once while declaring causal coverage open',()=>{
  class Battle {constructor(){this.sides=sides();}setPlayer(side){this.calls=(this.calls||0)+1;return side;}}
  let packet=control(0,'start');const audit=[],receipts=[];
  const observer=createObserver({Battle,loadControl:()=>packet,appendAudit:b=>{audit.push(b);},appendReceipt:b=>{receipts.push(b);},sourceRegistryHash:'4'.repeat(64)});
  const battle=new Battle();const start=observer.commandBegin({type:'start',battleId:'b'});
  assert.equal(battle.setPlayer('p1'),'p1');assert.equal(battle.setPlayer('p2'),'p2');observer.commandEnd(start,{originalThrew:false});
  packet=control(1,'snapshot',observer.core.lastReceiptHash);const token=observer.commandBegin({type:'snapshot',battleId:'b'});
  const state={turn:1,elided:undefined},requests={p1:{wait:false},p2:{wait:false}};
  observer.snapshotBegin('b',battle,'q');const result=observer.snapshotEnd('b',battle,state,requests,'q');
  const count=receipts.length;observer.commandEnd(token,{originalThrew:false});
  assert.equal(receipts.length,count);assert.equal(battle.calls,2);assert.equal(result.receipt.poisoned,false);
  assert.equal(result.causalCoverageComplete,false);assert.equal(result.qualification,false);
  assert.equal(result.elidedProperties.length,12);assert.equal(result.elidedProperties[0].happiness,255);
  assert.deepEqual(result.elidedProperties,result.receipt.boundary.elidedProperties);
  assert.equal(Buffer.from(result.wire.state_wire_base64,'base64').toString(),JSON.stringify(state));
  assert.ok(Object.hasOwn(state,'elided'));assert.equal(audit.length,10);
});

test('elided live scalars join by original set identity even for duplicate species and swaps',()=>{
  const battle={sides:sides()};const team=battle.sides[0].pokemon;
  [team[0],team[3]]=[team[3],team[0]];team[0].happiness=17;
  const facts=elidedProperties(battle);
  assert.equal(facts[0].original,3);assert.equal(facts[0].current,0);assert.equal(facts[0].happiness,17);
  assert.equal(facts[3].original,0);assert.equal(facts.length,12);
});

test('missing or accessor Native scalars are refused without invoking Native getters',()=>{
  const battle={sides:sides()};let calls=0;
  Object.defineProperty(battle.sides[0].pokemon[0],'happiness',{get(){calls++;return 255;}});
  assert.throws(()=>elidedProperties(battle),/DATA_PROPERTY/);assert.equal(calls,0);
  const missing={sides:sides()};delete missing.sides[0].pokemon[0].happiness;
  assert.throws(()=>elidedProperties(missing),/DATA_PROPERTY/);
});

test('wire projection matches Native JSON omission/array semantics without getters',()=>{
  const original={z:undefined,a:[undefined,-0,1e-7],nested:{alpha:'\u03b1'}};
  assert.equal(JSON.stringify(wireValue(original)),JSON.stringify(original));
  let getters=0;const bad={};Object.defineProperty(bad,'x',{enumerable:true,get(){getters++;return 1;}});
  assert.throws(()=>wireValue(bad));assert.equal(getters,0);
  assert.throws(()=>wireValue({number:Infinity}));assert.throws(()=>wireValue(new(class Native{})()));
});

test('receipt output failure cannot return a clean snapshot proof',()=>{
  class Battle {setPlayer(side){return side;}}
  let packet=control(0,'start');const observer=createObserver({Battle,loadControl:()=>packet,appendAudit:()=>{},
    appendReceipt:()=>{throw Error('disk');},sourceRegistryHash:'5'.repeat(64)});
  const battle=new Battle();const token=observer.commandBegin({type:'start',battleId:'b'});battle.setPlayer('p1');battle.setPlayer('p2');
  assert.equal(observer.commandEnd(token,{originalThrew:false}),undefined);assert.equal(observer.core.poisoned,true);
});
