'use strict';
// Explicit bridge boundaries only. Full Native causal coverage remains OPEN.
const {WitnessCore,hash} = require('./fresh200_witness_core.cjs');

function ownData(object,key){
  const value=Object.getOwnPropertyDescriptor(object,key);
  if(!value||!Object.hasOwn(value,'value'))throw Error('LITERAL_NATIVE_DATA_PROPERTY_REQUIRED');
  return value.value;
}
function ownArray(value,length){
  if(!Array.isArray(value)||ownData(value,'length')!==length)throw Error('COMPLETE_NATIVE_ARRAY_REQUIRED');
  return Array.from({length},(_,i)=>ownData(value,String(i)));
}
function elidedProperties(battle){
  const result=[];
  for(const [sideIndex,side] of ownArray(ownData(battle,'sides'),2).entries()){
    if(ownData(side,'n')!==sideIndex)throw Error('NATIVE_SIDE_IDENTITY_DIFFERS');
    const original=ownArray(ownData(side,'team'),6),pokemon=ownArray(ownData(side,'pokemon'),6),seen=new Set();
    for(const [current,mon] of pokemon.entries()){
      const set=ownData(mon,'set');const matches=original.map((value,index)=>value===set?index:-1).filter(index=>index>=0);
      if(matches.length!==1||seen.has(matches[0]))throw Error('UNIQUE_NATIVE_SET_IDENTITY_REQUIRED');
      seen.add(matches[0]);
      const level=ownData(mon,'level'),happiness=ownData(mon,'happiness'),pokeball=ownData(mon,'pokeball');
      if(!Number.isInteger(level)||level<1||level>9999||!Number.isInteger(happiness)||happiness<0||happiness>255||typeof pokeball!=='string')
        throw Error('LITERAL_NATIVE_ELIDED_SCALARS_REQUIRED');
      result.push({side:sideIndex,original:matches[0],current,level,happiness,pokeball});
    }
  }
  return result;
}

function materializedRequests(battle){
  const result={};
  for(const [index,side] of ownArray(ownData(battle,'sides'),2).entries()){
    if(ownData(side,'n')!==index)throw Error('NATIVE_SIDE_IDENTITY_DIFFERS');
    const request=ownData(side,'activeRequest');
    if(request===null)continue;
    if(!request||typeof request!=='object'||Array.isArray(request))throw Error('LITERAL_NATIVE_REQUEST_REQUIRED');
    result['p'+(index+1)]=request;
  }
  return result;
}

function wireValue(value, array = false, seen = new Set()) {
  if (value === undefined) return array ? null : undefined;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (!value || typeof value !== 'object' || seen.has(value)) throw Error('NON_LITERAL_NATIVE_SERIALIZATION');
  if (!Array.isArray(value) && ![Object.prototype,null].includes(Object.getPrototypeOf(value))) throw Error('NON_LITERAL_NATIVE_SERIALIZATION');
  seen.add(value);const descriptors=Object.getOwnPropertyDescriptors(value),result=Array.isArray(value)?[]:{};
  const keys=Array.isArray(value)?Array.from({length:value.length},(_,i)=>String(i)):Object.keys(value);
  for (const key of keys) {
    const descriptor=descriptors[key];
    if (!descriptor || !Object.hasOwn(descriptor,'value')) throw Error('NATIVE_SERIALIZATION_ACCESSOR_OR_HOLE');
    const child=wireValue(descriptor.value,Array.isArray(value),seen);
    if (child!==undefined) Object.defineProperty(result,key,{value:child,enumerable:true,writable:true,configurable:true});
  }
  seen.delete(value);return result;
}

function createObserver({Battle,loadControl,appendAudit,appendReceipt,sourceRegistryHash,nativeFactory=null}) {
  const core=new WitnessCore({appendAudit,sourceRegistryHash});
  const native=nativeFactory===null?null:nativeFactory(core);
  const ended=new WeakSet(),battles=new WeakMap();
  const playerDescriptor=Object.getOwnPropertyDescriptor(Battle.prototype,'setPlayer');
  if (!playerDescriptor || typeof playerDescriptor.value!=='function') throw Error('PINNED_SETPLAYER_SEAM_REQUIRED');
  core.installMethod(Battle.prototype,'setPlayer',{operation:'Battle.setPlayer',
    enter(battle,args) {
      if (!core.active || core.active.control.phase!=='start') throw Error('UNDECLARED_NATIVE_CONSTRUCTION');
      const known=battles.get(battle);
      if (known!==undefined && known!==core.active.battleId) throw Error('NATIVE_BATTLE_REBOUND');
      battles.set(battle,core.active.battleId);
      if(native)native.bindBattle(battle);
      if (!['p1','p2'].includes(args[0])) throw Error('LITERAL_PLAYER_REQUIRED');
      return {side:args[0]};
    }});
  function finish(token,outcome) {
    if (token && ended.has(token)) {
      if (outcome?.originalThrew) core.poison('COMMAND_THROW_AFTER_SNAPSHOT');
      return undefined;
    }
    if (!token) {core.poison('MISSING_BRIDGE_COMMAND_TOKEN');return undefined;}
    const receipt=core.end(token,outcome);
    if (!receipt) return undefined;
    ended.add(token);
    const bytes=Buffer.from(JSON.stringify(receipt)+'\n','utf8');
    try {
      const result=appendReceipt(bytes);
      if (result && typeof result.then==='function') throw Error('SYNCHRONOUS_RECEIPT_WRITER_REQUIRED');
    } catch {core.poison('RECEIPT_WRITE_FAILED');return undefined;}
    return receipt;
  }
  function requireBattle(battleId,battle) {
    if (!core.active || core.active.battleId!==battleId || battles.get(battle)!==battleId ||
        core.active.control.phase!=='snapshot') throw Error('EXACT_CONSTRUCTED_SNAPSHOT_CONTEXT_REQUIRED');
  }
  return {
    core,
    poison(code) {core.poison(code);},
    commandBegin(command) {
      const battleId=command.battleId==null?'default':String(command.battleId);
      return core.begin(loadControl(),{battleId,phase:command.type});
    },
    commandEnd(token,outcome) {return finish(token,outcome);},
    snapshotBegin(battleId,battle,requestId) {
      requireBattle(battleId,battle);core.record('SNAPSHOT_BEGIN',{requestId});
    },
    snapshotRequests(battleId,battle,requestId) {
      requireBattle(battleId,battle);
      const requests=materializedRequests(battle);
      core.active.passiveRequests={requestId,requests};
      return requests;
    },
    snapshotEnd(battleId,battle,state,requests,requestId) {
      requireBattle(battleId,battle);
      const passive=core.active.passiveRequests;
      if(passive&&(passive.requestId!==requestId||passive.requests!==requests))throw Error('PASSIVE_REQUEST_BOUNDARY_DIFFERS');
      const requestOrigin=passive?'materialized_activeRequest':'getRequests';
      // State.serializePokemon intentionally elides these data properties.
      // Observe actual live values; never fill them from saved device state or
      // infer a constructor default from an absent serialized set field.
      const elided=elidedProperties(battle);
      const nativeObservation=native?{state:native.snapshot(battle),roster:native.manifest()}:null;
      const stateWire=Buffer.from(JSON.stringify(wireValue(state)),'utf8');
      const requestWire=Buffer.from(JSON.stringify(wireValue(requests)),'utf8');
      const packet={state_wire_base64:stateWire.toString('base64'),state_sha256:hash(stateWire),
        requests_wire_base64:requestWire.toString('base64'),requests_sha256:hash(requestWire)};
      core.record('SNAPSHOT_SERIALIZED',{requestId,requestOrigin,wire:packet,elidedProperties:elided,nativeObservation});
      const receipt=finish(core.active,{boundary:{kind:'snapshot',requestId,requestOrigin,wire:packet,elidedProperties:elided,nativeObservation,causalCoverageComplete:false}});
      if (!receipt) return undefined;
      return {receipt,wire:packet,elidedProperties:elided,nativeObservation,requestOrigin,causalCoverageComplete:false,qualification:false};
    },
  };
}

module.exports={createObserver,wireValue,elidedProperties,materializedRequests};
