'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {WitnessCore,ZERO}=require('../gpu_engine/fresh200_witness_core.cjs');
const {createRecorder,METHODS}=require('../gpu_engine/fresh200_witness_native.cjs');

function fixture(overrides={}){
  const classes=Object.fromEntries(Object.keys(METHODS).map(name=>[name,class {}]));
  for(const [owner,names] of Object.entries(METHODS))for(const name of names){
    Object.defineProperty(classes[owner].prototype,name,{value:function(){return undefined;},writable:true,configurable:true});
  }
  for(const [owner,methods] of Object.entries(overrides))for(const [name,value] of Object.entries(methods)){
    Object.defineProperty(classes[owner].prototype,name,{value,writable:true,configurable:true});
  }
  const encore=function(){return 'guarded';};classes.Pokemon.prototype.addVolatile=encore;
  const rows=[];const core=new WitnessCore({sourceRegistryHash:'8'.repeat(64),appendAudit:b=>rows.push(JSON.parse(b))});
  const native=createRecorder(core,classes),battle=new classes.Battle();
  battle.sides=[];battle.actions=new classes.BattleActions();battle.queue=new classes.BattleQueue();
  battle.actions.battle=battle;battle.queue.battle=battle;
  native.bindBattle(battle);
  const control={schema:'pokezero.native_witness_control.v1',run:'native-recorder',stream:0,reset:0,command:0,phase:'start',request:0,
    nonce:'native-recorder/0/0/0',chance_nonce:null,previous_receipt_hash:ZERO};
  const command=core.begin(control,{battleId:'b',phase:'start'});
  return {classes,core,native,battle,rows,encore,finish:()=>core.end(command)};
}

test('records exact hidden-lock arguments and outcomes without querying Native again',()=>{
  let calls=0;const sentinel={locked:'encore'};
  const f=fixture({Pokemon:{getSemiLockedMove:function(...args){calls++;assert.equal(this,mon);return args.length?sentinel:null;}}});
  const mon=new f.classes.Pokemon();mon.battle=f.battle;
  assert.equal(mon.getSemiLockedMove(),null);assert.equal(mon.getSemiLockedMove(true),sentinel);
  assert.equal(calls,2);assert.equal(f.finish().poisoned,false);
  const enters=f.rows.filter(r=>r.operation==='CALL_ENTER');
  assert.deepEqual(enters.map(r=>r.evidence.arguments.items),[[],[true]]);
  const returns=f.rows.filter(r=>r.operation==='CALL_RETURN');
  assert.equal(returns[0].evidence.result,null);
  assert.deepEqual(returns[1].evidence.result.fields,[['locked','encore']]);
});

test('instruments actual inherited callback apply preserving function identity and Native result',()=>{
  let calls=0;const result={native:true};
  function onProbe(value){calls++;assert.equal(this,f.battle);assert.equal(value,17);return result;}
  const effect=Object.create({onProbe});
  const f=fixture({Battle:{singleEvent:function(event,effect,_state,_target,_source,_sourceEffect,_relay,custom){
    return (custom||effect['on'+event]).apply(this,[17]);
  }}});
  assert.equal(f.battle.singleEvent('Probe',effect,null,null,null,null,undefined,false),result);
  assert.equal(effect.onProbe,onProbe);assert.equal(calls,1);
  assert.equal(Object.getOwnPropertyDescriptor(onProbe,'apply').enumerable,false);
  assert.equal(f.finish().poisoned,false);
  const callbackRows=f.rows.filter(r=>r.frame?.operation.startsWith('Callback:'));
  assert.deepEqual(callbackRows.map(r=>r.operation),['CALL_ENTER','CALL_RETURN']);
  assert.equal(callbackRows[0].frame.parentCall,0);
  assert.equal(f.native.manifest().causalCoverageComplete,false);
});

test('retains executed original identity across duplicate species and roster swaps',()=>{
  const f=fixture();const side=new f.classes.Side();side.n=0;side.battle=f.battle;
  side.team=[{species:'duplicate'},{species:'duplicate'}];side.pokemon=side.team.map(set=>{
    const mon=new f.classes.Pokemon();mon.side=side;mon.set=set;mon.battle=f.battle;return mon;
  });
  [side.pokemon[0],side.pokemon[1]]=[side.pokemon[1],side.pokemon[0]];f.battle.sides=[side];
  const snapshot=f.native.snapshot(f.battle);
  assert.deepEqual(snapshot.sides[0].pokemon.map(m=>m.identity.original),[1,0]);
  assert.equal(f.finish().poisoned,false);
  assert.equal(f.classes.Pokemon.prototype.addVolatile,f.encore);
  assert.equal(f.native.manifest().guardedEncorePointerPreserved,true);
});

test('observer getter failure poisons evidence while original operation still runs exactly once',()=>{
  let getters=0,calls=0;const result={same:true};
  const f=fixture({Battle:{runAction:function(){calls++;return result;}}});
  Object.defineProperty(f.battle,'event',{get(){getters++;throw Error('forbidden');}});
  assert.equal(f.battle.runAction({choice:'move'}),result);assert.equal(calls,1);assert.equal(getters,0);
  assert.equal(f.finish().poisoned,true);
});

test('unknown Native class stays explicitly opaque and exact original throws are preserved',()=>{
  const sentinel={originalThrow:true};const f=fixture({Battle:{runAction:function(){throw sentinel;}}});
  class Unknown {}
  f.battle.effectState=new Unknown();
  assert.equal(f.native.snapshot(f.battle).fields.effectState.kind,'OpaqueNativeObject');
  assert.equal(f.native.snapshot(f.battle).fields.effectState.materialInterpretationComplete,false);
  assert.throws(()=>f.battle.runAction(),error=>error===sentinel);
  assert.ok(f.rows.some(r=>r.operation==='CALL_THROW'));
  assert.equal(f.native.manifest().qualification,false);f.finish();
});
