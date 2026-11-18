'use strict';
// Passive Native method/callback observations. Interpretation/coverage is not
// certified here; opaque/missing evidence can never qualify physical parity.
const {hash}=require('./fresh200_witness_core.cjs');
const functionText=Function.prototype.toString;
const BATTLE=['choose','runAction','turnLoop','commitChoices','clearRequest','makeRequest','getRequests','endTurn','fieldEvent','singleEvent','runEvent','priorityEvent',
  'getCallback','findEventHandlers','findPokemonEventHandlers','findBattleEventHandlers','findSideEventHandlers','findFieldEventHandlers','faintMessages','speedSort','canSwitch'];
const POKEMON=['moveUsed','getMoveData','deductPP','transformInto','clearVolatile','copyVolatileFrom','removeVolatile','faint','disableMove','tryTrap','getMoveRequestData','getLockedMove','getSemiLockedMove',
  'getMoves','isLastActive','hasAbility','hasType','isGrounded','isAdjacent','runStatusImmunity'];
const SIDE=['chooseMove','chooseSwitch','commitChoices','clearChoice','getRequestData','emitRequest','addSlotCondition','removeSlotCondition','addSideCondition','removeSideCondition'];
const ACTIONS=['switchIn','dragIn','runSwitch','runMove','useMove','useMoveInner'];
const QUEUE=['resolveAction','addChoice','insertChoice','shift','push','unshift','cancelAction','cancelMove','changeAction','prioritizeAction','clear','sort'];
const FIELD=['setWeather','clearWeather','addPseudoWeather','removePseudoWeather'];
const MATERIAL=new Set(['choose','runAction','turnLoop','commitChoices','clearChoice','clearRequest','makeRequest','getRequests','endTurn','fieldEvent','faintMessages',
  'switchIn','dragIn','runSwitch','runMove','useMove','useMoveInner','chooseMove','chooseSwitch','getRequestData','getMoveRequestData',
  'addSlotCondition','removeSlotCondition','addSideCondition','removeSideCondition','moveUsed','deductPP','transformInto','clearVolatile','copyVolatileFrom','removeVolatile','faint','disableMove','tryTrap',...QUEUE,...FIELD]);
const METHODS=Object.freeze({Battle:Object.freeze(BATTLE),Pokemon:Object.freeze(POKEMON),Side:Object.freeze(SIDE),
  BattleActions:Object.freeze(ACTIONS),BattleQueue:Object.freeze(QUEUE),Field:Object.freeze(FIELD)});

function createRecorder(core,{Battle,Pokemon,Side,BattleActions,BattleQueue,Field},{transport=null}={}){
  const objects=new WeakMap(),bound=new WeakSet();let nextObject=0;
  const protectedAddVolatile=Pokemon.prototype.addVolatile;
  function token(value){let id=objects.get(value);if(id===undefined){id=nextObject++;objects.set(value,id);}return id;}
  function property(value,key){
    const descriptor=Object.getOwnPropertyDescriptor(value,key);
    if(!descriptor)return {present:false};
    if(!Object.hasOwn(descriptor,'value'))throw Error('NATIVE_OBSERVATION_ACCESSOR_FORBIDDEN');
    return {present:true,value:descriptor.value};
  }
  function data(value,key){return property(value,key).value;}
  function inheritedData(value,key){
    for(let owner=value;owner;owner=Object.getPrototypeOf(owner)){
      const descriptor=property(owner,key);if(descriptor.present)return descriptor.value;
    }
    return undefined;
  }
  function values(array){
    if(!Array.isArray(array))throw Error('NATIVE_OBSERVATION_ARRAY_REQUIRED');
    const length=data(array,'length');if(!Number.isInteger(length)||length<0||length>4096)throw Error('NATIVE_OBSERVATION_ARRAY_BOUND');
    return Array.from({length},(_,i)=>{const d=property(array,String(i));if(!d.present)throw Error('NATIVE_ARRAY_HOLE');return d.value;});
  }
  function physical(mon){
    const side=data(mon,'side'),set=data(mon,'set');
    if(!side||!set)return {object:token(mon),constructing:true};
    const team=data(side,'team'),n=data(side,'n');
    if(!Array.isArray(team))return {object:token(mon),constructingSide:token(side),set:token(set)};
    const matches=values(team).map((v,i)=>v===set?i:-1).filter(i=>i>=0);
    if(matches.length!==1||![0,1].includes(n))throw Error('NATIVE_ORIGINAL_IDENTITY_UNRESOLVED');
    return {object:token(mon),side:n,original:matches[0]};
  }
  function project(value,seen=new Set(),depth=0){
    if(depth>32)throw Error('NATIVE_OBSERVATION_DEPTH_BOUND');
    if(value===undefined)return {$native:'undefined'};
    if(value===null||typeof value==='string'||typeof value==='boolean')return value;
    if(typeof value==='number'){
      if(Number.isFinite(value))return value;
      return {$nativeNumber:Number.isNaN(value)?'NaN':value>0?'+Infinity':'-Infinity'};
    }
    if(typeof value==='function')return {$function:hash(Reflect.apply(functionText,value,[])),
      ...(core.registrationEvidence?{function_token:core.registrationToken(value)}:{})};
    if(typeof value!=='object')throw Error('UNSUPPORTED_NATIVE_OBSERVATION_VALUE');
    if(value instanceof Pokemon)return {kind:'Pokemon',...physical(value)};
    if(value instanceof Battle)return {kind:'Battle',object:token(value)};
    if(value instanceof Side)return {kind:'Side',object:token(value),n:project(data(value,'n'),seen,depth+1)};
    if(value instanceof BattleActions||value instanceof BattleQueue||value instanceof Field)return {kind:'NativeService',object:token(value)};
    const id=token(value);
    if(seen.has(value))return {$object:id};
    seen.add(value);let result;
    if(Array.isArray(value))result={object:id,items:values(value).map(v=>project(v,seen,depth+1))};
    else{
      const proto=Object.getPrototypeOf(value),identity=data(value,'id'),kind=data(value,'effectType');
      if(typeof identity==='string'&&typeof kind==='string')result={object:id,kind:'Effect',id:identity,effectType:kind};
      else if(proto===Object.prototype||proto===null){
        result={object:id,fields:[]};
        for(const key of Object.keys(value))result.fields.push([key,project(data(value,key),seen,depth+1)]);
      }else result={object:id,kind:'OpaqueNativeObject',materialInterpretationComplete:false};
    }
    seen.delete(value);return result;
  }
  function field(value,key){const d=property(value,key);return d.present?project(d.value):{$native:'missing'};}
  function member(mon){
    const result={identity:physical(mon),fields:{}};
    for(const key of ['hp','maxhp','fainted','faintQueued','isActive','isStarted','position','status','statusState','volatiles','trapped','maybeTrapped','maybeDisabled','maybeLocked',
      'moveSlots','lastMove','moveThisTurn','switchFlag','forceSwitchFlag','activeTurns','transformed','baseStoredStats','baseMoveSlots','baseHpType','baseHpPower',
      'species','types','baseTypes','ability','baseAbility','item','boosts','level','happiness','hpType','hpPower','gender','addedType',
      'storedStats','baseMaxhp','truantTurn','itemKnockedOff'])result.fields[key]=field(mon,key);
    return result;
  }
  function cache(battle){
    const result={battle:token(battle),fields:{},sides:[]};
    for(const key of ['turn','midTurn','requestState','ended','winner','event','effect','effectState','activePokemon','activeTarget','activeMove','faintQueue'])result.fields[key]=field(battle,key);
    const queue=data(battle,'queue'),nativeField=data(battle,'field');
    if(queue)result.fields.queueList=field(queue,'list');
    if(nativeField){
      result.fields.field={};
      for(const key of ['weather','weatherState','terrain','terrainState','pseudoWeather'])result.fields.field[key]=field(nativeField,key);
    }
    const sides=data(battle,'sides');
    if(Array.isArray(sides))for(const side of values(sides)){
      if(side===null){result.sides.push(null);continue;}
      const item={identity:project(side),fields:{},pokemon:[]};
      for(const key of ['choice','activeRequest','sideConditions','slotConditions','active'])item.fields[key]=field(side,key);
      const pokemon=data(side,'pokemon');if(Array.isArray(pokemon))item.pokemon=values(pokemon).map(member);
      result.sides.push(item);
    }
    return result;
  }
  function battleOf(subject){
    if(subject instanceof Battle)return subject;
    return data(subject,'battle');
  }
  function context(battle){
    if(!battle)return {$native:'constructing'};
    return {battle:token(battle),event:field(battle,'event'),effect:field(battle,'effect'),effectState:field(battle,'effectState')};
  }
  function callback(fn,registrationSite){
    if(typeof fn!=='function')return;
    const key='Callback:'+hash(Reflect.apply(functionText,fn,[]));
    core.instrumentCallback(fn,key,{operation:key,registrationSite,
      enter(actual,args){
        if(core.registrationEvidence&&actual!==fn)throw Error('EXACT_REGISTERED_CALLBACK_RECEIVER_REQUIRED');
        const battle=args[0];return {...(core.registrationEvidence?{function_token:core.registrationToken(actual)}:{}),
          context:context(battle),arguments:project(args[1]),before:cache(battle)};},
      leave(_before,actual,args,outcome){return {...(core.registrationEvidence?{function_token:core.registrationToken(actual)}:{}),
        context:context(args[0]),result:project(outcome.value),didThrow:outcome.didThrow,after:cache(args[0])};}});
  }
  function handlers(result){
    if(!Array.isArray(result))return;
    for(const [index,handler] of values(result).entries())if(handler&&typeof handler==='object')callback(data(handler,'callback'),{kind:'handler-result',index});
  }
  function observer(operation,name){return {operation,
    enter(subject,args){
      const battle=battleOf(subject);
      if(transport&&name==='emitRequest'){
        const frame=core.frames.at(-1),side=data(subject,'n');
        if(!frame||frame.operation!=='Side.emitRequest'||![0,1].includes(side)||core.active?.battleId!==transport.battleId)
          throw Error('EXACT_NATIVE_REQUEST_CALL_CONTEXT_REQUIRED');
        transport.requestBegin(frame.callId,'p'+(side+1));
      }
      let callbackSelection=null;
      if(name==='singleEvent'){
        const effect=args[1],custom=args[7];
        // Native uses customCallback || effect[onEvent], including false/0.
        const fn=custom||(effect&&inheritedData(effect,'on'+args[0]));callback(fn,{kind:'singleEvent-selection'});
        if(core.registrationEvidence&&typeof fn==='function')callbackSelection={function_token:core.registrationToken(fn),
          originalSha256:hash(Reflect.apply(functionText,fn,[])),selector:custom?'custom':'effect-property'};
      }
      return {subject:project(subject),arguments:project(args),context:context(battle),
        ...(core.registrationEvidence&&name==='singleEvent'?{callbackSelection}:{}),
        ...(MATERIAL.has(name)&&battle?{before:cache(battle)}:{})};
    },
    leave(_before,subject,_args,outcome){
      if(transport&&name==='emitRequest')transport.requestEnd(core.frames.at(-1)?.callId,outcome.didThrow);
      if(!outcome.didThrow){
        if(name==='getCallback')callback(outcome.value,{kind:'getCallback-result'});
        else if(name.startsWith('find'))handlers(outcome.value);
      }
      const battle=battleOf(subject);
      return {result:project(outcome.value),didThrow:outcome.didThrow,context:context(battle),
        ...(MATERIAL.has(name)&&battle?{after:cache(battle)}:{})};
    }};}
  function method(target,name,operation,stage='bootstrap-method'){
    let owner=target,descriptor;
    while(owner&&!(descriptor=Object.getOwnPropertyDescriptor(owner,name)))owner=Object.getPrototypeOf(owner);
    if(!descriptor||typeof descriptor.value!=='function')throw Error('PINNED_NATIVE_METHOD_MISSING:'+operation);
    if(owner!==target)Object.defineProperty(target,name,{value:descriptor.value,writable:true,configurable:true});
    core.installMethod(target,name,{...observer(operation,name),registrationSite:{kind:stage,owner:operation.split('.')[0]}});
  }
  for(const name of POKEMON)method(Pokemon.prototype,name,'Pokemon.'+name);
  for(const name of SIDE)method(Side.prototype,name,'Side.'+name);
  for(const name of FIELD)method(Field.prototype,name,'Field.'+name);
  return {
    bindBattle(battle){
      if(bound.has(battle))return;bound.add(battle);
      for(const name of BATTLE)method(battle,name,'Battle.'+name,'instance-method');
      const actions=data(battle,'actions'),queue=data(battle,'queue');
      if(!(actions instanceof BattleActions)||!(queue instanceof BattleQueue))throw Error('PINNED_NATIVE_SERVICES_REQUIRED');
      for(const name of ACTIONS)method(actions,name,'BattleActions.'+name,'instance-method');
      for(const name of QUEUE)method(queue,name,'BattleQueue.'+name,'instance-method');
    },
    snapshot(battle){return cache(battle);},
    manifest(){
      if(Pokemon.prototype.addVolatile!==protectedAddVolatile)core.poison('GUARDED_ENCORE_POINTER_CHANGED');
      return {schema:'pokezero.fresh200.native_hook_roster.v1',hooks:core.hooks.map(({operation,originalSha256,kind})=>({operation,originalSha256,kind})),
        guardedEncorePointerPreserved:Pokemon.prototype.addVolatile===protectedAddVolatile,
        callbackSeam:'actual Function.apply invocations only; direct call/invocation coverage remains open',
        causalCoverageComplete:false,qualification:false};
    },
  };
}
module.exports={createRecorder,METHODS};
