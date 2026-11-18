'use strict';
// Synthetic parsed-span/catalog and actual recorder rows only. No Native
// runtime, frozen root, chance inputs, policy or physical state is loaded.
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
const {WitnessCore,ZERO,hash}=require('../gpu_engine/fresh200_witness_core.cjs');
const {createObserver}=require('../gpu_engine/fresh200_witness_bridge.cjs');
const {createRecorder,METHODS}=require('../gpu_engine/fresh200_witness_native.cjs');
const out=process.argv[2];fs.mkdirSync(out,{recursive:true});const wire=v=>Buffer.from(JSON.stringify(v)+'\n');
const source=fs.readFileSync(__filename);const entries=new Map();
function declare(operation,fn,kind,stage){
  const text=Buffer.from(Function.prototype.toString.call(fn)),sha=hash(text),start=source.indexOf(text);assert.ok(start>=0);
  const value={operation,originalSha256:sha,kind,stage,locations:[{source:__filename,start,end:start+text.length,
    declaration:operation,source_form:'function-expression'}]};
  const old=entries.get(operation);if(old)assert.deepEqual(old,value);else entries.set(operation,value);
}
const classes=Object.fromEntries(Object.keys(METHODS).map(name=>[name,class{}]));
for(const [owner,names] of Object.entries(METHODS))for(const name of names)
  Object.defineProperty(classes[owner].prototype,name,{value:function(){return undefined;},writable:true,configurable:true});
classes.Battle.prototype.setPlayer=function(side){this.players.push(side);return side;};
classes.Battle.prototype.getCallback=function(value){return value;};
classes.Battle.prototype.findEventHandlers=function(value){return value;};
classes.Battle.prototype.singleEvent=function(event,effect,_state,_target,_source,_effect,_relay,custom){return (custom||effect['on'+event]).apply(this,[]);};
classes.Pokemon.prototype.addVolatile=function(){return true;};
const guardedPointer=classes.Pokemon.prototype.addVolatile;
const make=()=>function sameBody(){return 7;};const first=make(),second=make();
const handlerOnly=function handlerOnly(){return 9;};const custom=function custom(){return 11;};
for(const fn of [first,second,handlerOnly,custom])declare('Callback:'+hash(Function.prototype.toString.call(fn)),fn,'callback-own-apply','callback');
const bootstrap=[],instances=[];
for(const owner of ['Pokemon','Side','Field'])for(const name of METHODS[owner]){
  const operation=owner+'.'+name;bootstrap.push(operation);declare(operation,classes[owner].prototype[name],'method','bootstrap-method');
}
bootstrap.push('Battle.setPlayer');declare('Battle.setPlayer',classes.Battle.prototype.setPlayer,'method','bootstrap-method');
for(const owner of ['Battle','BattleActions','BattleQueue'])for(const name of METHODS[owner]){
  const operation=owner+'.'+name;instances.push(operation);declare(operation,classes[owner].prototype[name],'method','instance-method');
}
let control;const rows=[],receipts=[],controls=[];
const observer=createObserver({Battle:classes.Battle,registrationEvidence:true,sourceRegistryHash:'b'.repeat(64),loadControl:()=>control,
  appendAudit:bytes=>rows.push(bytes),appendReceipt:bytes=>receipts.push(bytes),nativeFactory:core=>createRecorder(core,classes)});
const battle=new classes.Battle();battle.players=[];battle.sides=[];battle.actions=new classes.BattleActions();battle.queue=new classes.BattleQueue();
battle.actions.battle=battle;battle.queue.battle=battle;
function begin(index,phase){
  control={schema:'pokezero.native_witness_control.v1',run:'catalog-fixture',stream:0,reset:0,command:index,phase,request:index,
    nonce:'catalog-fixture/0/0/'+index,chance_nonce:null,previous_receipt_hash:observer.core.lastReceiptHash};
  controls.push(wire(control));
  return observer.commandBegin({type:phase,battleId:'b'});
}
let token=begin(0,'start');battle.setPlayer('p1');battle.setPlayer('p2');
battle.getCallback(first);battle.getCallback(second); // Identical bodies, unused distinct objects.
battle.findEventHandlers([{callback:first},{callback:handlerOnly}]);
const effect={id:'synthetic',effectType:'Condition',onProbe:first};assert.equal(battle.singleEvent('Probe',effect,null,null,null,null,null,false),7);
observer.commandEnd(token,{originalThrew:false});
token=begin(1,'choices');battle.getCallback(first);assert.equal(battle.singleEvent('Probe',effect,null,null,null,null,null,custom),11);
observer.commandEnd(token,{originalThrew:false});
assert.ok(receipts.every(bytes=>JSON.parse(bytes).poisoned===false));
fs.writeFileSync(path.join(out,'audit.jsonl'),Buffer.concat(rows));fs.writeFileSync(path.join(out,'receipts.jsonl'),Buffer.concat(receipts));
fs.writeFileSync(path.join(out,'controls.jsonl'),Buffer.concat(controls));
fs.writeFileSync(path.join(out,'catalog.json'),wire({synthetic_only:true,source_pins:{[__filename]:hash(source)},functions:[...entries.values()],
  bootstrap_operations:bootstrap,instance_operations:instances,source_registry_hash:'b'.repeat(64)}));
// New borrowed-apply guard. Observation faults preserve the exact other
// function's execution; such evidence is poisoned, never source admitted.
token=begin(2,'choices');let otherCalls=0;function other(){otherCalls++;return 17;}
assert.equal(first.apply.call(other,battle,[]),17);const failed=observer.commandEnd(token,{originalThrew:false});
fs.writeFileSync(path.join(out,'borrowed.json'),wire({original_result:17,original_calls:otherCalls,poisoned:failed.poisoned,
  fault:failed.faultCode,qualification:false,guarded_pointer_unchanged:classes.Pokemon.prototype.addVolatile===guardedPointer}));
