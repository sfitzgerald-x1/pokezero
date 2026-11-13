// Diagnostic-only semantic coupling. Never imported by the production bridge.
//
// Tape v1: draws [2][12], secondary [2], sleep_talk [2]. Tape v2 adds
// effect [2][4] with bounds (16,16,4,2): the engine's appended ChanceInputs.effect
// stream (fidelity-patch design 2.1). Roles that read it on a v1 tape stay
// unresolved (EFFECT_STREAM_ABSENT); a v1 tape never carries `effect`.
//   effect[s][0] second-hit critical, effect[s][1] second-hit damage roll (one
//   multihit move, per hit index; the engine consumes both for Bonemerang since
//   fidelity R, and a third hit stays unresolved); effect[s][2] Yawn's residual sleep duration
//   (s = the drowsy mon's side); effect[0][3] partial-trap residual pair tie;
//   effect[1][3] Perish residual pair tie. A residual pair tie puts the side
//   whose n equals the tape value first, as move_tie does with draws[0][0].
// draws[s][7] is shared by sleep, Encore, partial-trap (3+d) and confusion (2+d)
// durations; a second duration consumer on one side in one nonce stays REPEATED.
// selfDrops' random(100) (self.boosts with no chance) is an inert, audited role, as is a
// residual tie among one mon's callback-free countdowns (stall + protect).
// Fidelity M2: a Yawn pair (both actives drowsy at equal Speed) is an inert residual tie:
// each countdown and its sleep attempt touch only their holder (Sleep Clause reads the
// holder's own side, the duration is that side's effect[s][2]). An eachEvent (BeforeTurn,
// Update) Pokemon-order tie between equal-Speed actives is inert when neither Pokemon has a
// handler for that event (findEventHandlers is empty, so both runEvent calls do nothing);
// with a handler it stays unresolved (TIE_ORDER).
// Fidelity WH: White Herb handlers of two holders at equal Speed (onAnyAfterMove, onResidual order 29; SwitchIn ties
// are broken by effectOrder), with a Choice Band user's own onAfterMove (addVolatile choicelock) beside them, are an
// inert tie: each Item handler touches only its own holder and draws nothing, so the order cannot change the battle.
const TAPE_V1='pokezero.gpu_chance_tape.v1',TAPE_V2='pokezero.gpu_chance_tape.v2';
const DRAW_BOUNDS=[2,100,100,100,16,16,100,4,8,2,2,2],EFFECT_BOUNDS=[16,16,4,2];
function validTape(c){
  const v2=c.schema===TAPE_V2,row=(x,bounds)=>Array.isArray(x)&&x.length===bounds.length
    &&x.every((v,i)=>Number.isInteger(v)&&v>=0&&v<bounds[i]);
  return (c.schema===TAPE_V1||v2)&&typeof c.nonce==='string'&&!!c.nonce
    &&Array.isArray(c.draws)&&c.draws.length===2&&c.draws.every(x=>row(x,DRAW_BOUNDS))
    &&['secondary','sleep_talk'].every(k=>Array.isArray(c[k])&&c[k].length===2
      &&c[k].every(v=>Number.isInteger(v)&&v>=0&&v<(k==='secondary'?100:60)))
    &&(v2?Array.isArray(c.effect)&&c.effect.length===2&&c.effect.every(x=>row(x,EFFECT_BOUNDS)):c.effect===undefined);
}
function install(Battle,PRNG,load,audit,initializing=()=>null,pokemonPath='') {
  let nonce,used=new Set(),inert=0;
  const control=()=>{
    const c=load();if(!c)return null;
    if(!validTape(c))throw Error('invalid chance tape');
    if(c.nonce!==nonce){nonce=c.nonce;used=new Set();}return c;
  };
  function unknown(kind,detail){audit({kind:'UNRESOLVED',role:kind,detail});throw Error('SPIKE_UNRESOLVED:'+kind);}
  function take(c,role,side,value){
    const key=role+':'+side;if(used.has(key))return unknown('REPEATED_ROLE',key);
    used.add(key);audit({kind:'COUPLED',nonce:c.nonce,role,side,raw:value});return value;
  }
  function effectDraw(c,role,side,column){
    if(!Array.isArray(c.effect))return unknown('EFFECT_STREAM_ABSENT',{role,side});
    return take(c,role,side,c.effect[side][column]);
  }
  // Hit index inside one multihit move (gen3 trySpreadMoveHit sets move.hit=i+1).
  const multihitIndex=b=>b.activeMove?.multihit?b.activeMove.hit:undefined;
  // Yawn's residual sleep: slp onStart with no active move, from yawn's onEnd
  // (removeVolatile -> trySetStatus) while the drowsy mon still holds the volatile.
  const yawnSleeper=b=>{
    const t=b.effectState?.target;
    return b.effect?.id==='slp'&&!b.activeMove&&t?.volatiles?.yawn&&(t.side?.n===0||t.side?.n===1)
      &&new Error().stack.includes('trySetStatus')?t:null;
  };
  // Sleep and Encore share the SAME engine coordinate draws[actor][7].
  // An unexpected second duration consumer must remain unresolved, not become
  // two independent roles which silently reuse one finite tape coordinate.
  function duration(c,role,side){
    const key='duration_draw7:'+side;
    if(used.has(key))return unknown('REPEATED_ROLE',key);
    used.add(key);return take(c,role,side,c.draws[side][7]);
  }
  const seat=b=>b.activePokemon?.side?.n;
  const encoreFrames=new WeakMap();let boundRandom;
  let encoreCallbackPath='',trapCallbackPath='';
  if(pokemonPath){
    const path=require('path'),crypto=require('crypto');
    const resolved=require.resolve(pokemonPath),{Pokemon}=require(resolved);
    // Gen 3 inherits gen4's partiallytrapped.durationCallback: random(3,7).
    trapCallbackPath=path.resolve(path.dirname(resolved),'../data/mods/gen4/conditions.js');
    const {Dex}=require(path.join(path.dirname(resolved),'dex.js'));
    const dex=Dex.mod('gen3'),condition=dex.conditions.get('encore');
    const original=condition.durationCallback,start=condition.onStart,add=Pokemon.prototype.addVolatile;
    const hash=f=>typeof f==='function'?crypto.createHash('sha256').update(f.toString()).digest('hex'):'';
    const sourceOK=hash(original)==='5fa95c56ee1187568ff774129ba7816873d4d6a4806130268725463caea45786'
      &&hash(start)==='65cca25468443dea05dd74c855318f3668d29f5a791838ef10d845536937eeca'
      &&hash(add)==='cddf90eeb6711d33592f46b063e3f804f395d5dfe9ec8ccd833540d4280add51';
    encoreCallbackPath=path.resolve(path.dirname(resolved),'../data/mods/gen3/moves.js');
    // Original condition tables are frozen. Scope the original addVolatile
    // caller instead; never replace a condition/callback or fabricate a draw.
    Pokemon.prototype.addVolatile=function scopedEncoreAdd(status,source,effect,linkedStatus){
      const c=control();
      if(!c||(status!=='encore'&&status!==condition))return add.apply(this,arguments);
      if(!sourceOK||condition.durationCallback!==original||condition.onStart!==start
        ||Pokemon.prototype.addVolatile!==scopedEncoreAdd)return unknown('ENCORE_DURATION_SOURCE',null);
      const battle=this.battle,actor=battle?seat(battle):undefined,target=this,sides=battle?.sides;
      const same=(actor===0||actor===1)&&battle.gen===3&&battle.dex===dex
        &&source instanceof Pokemon&&target instanceof Pokemon&&source.battle===battle
        &&battle.activePokemon===source&&battle.activeTarget===target&&effect===battle.activeMove
        &&effect?.id==='encore'&&!effect.sourceEffect&&battle.random===boundRandom
        &&source.hp>0&&target.hp>0&&source.side?.n===actor&&target.side?.n===1-actor
        &&sides?.length===2&&sides[actor]===source.side&&sides[1-actor]===target.side
        &&source.side.active?.length===1&&source.side.active[0]===source
        &&target.side.active?.length===1&&target.side.active[0]===target
        &&target.addVolatile===scopedEncoreAdd;
      if(!same||encoreFrames.has(battle))return unknown('ENCORE_DURATION_CONTEXT',{actor});
      encoreFrames.set(battle,{nonce:c.nonce,actor,target,source,effect,pokemonPath:resolved});
      try{return add.apply(this,arguments);}finally{encoreFrames.delete(battle);}
    };
  }
  const caller=()=>new Error().stack.split('\n')[3]||'';
  const oldChance=Battle.prototype.randomChance,oldRandom=Battle.prototype.random,oldSample=Battle.prototype.sample;
  Battle.prototype.randomChance=function(n,d){
    const c=control();
    if(!c){
      if(initializing()&&caller().includes('Battle.endTurn')&&n===1&&d===5
        &&!this.sides.some(x=>x.active.some(p=>p?.item==='quickclaw'))){
        audit({kind:'INERT_QUICK_CLAW_ROLL',phase:'initial'});return false;
      }
      return oldChance.call(this,n,d);
    }
    const from=caller(),s=seat(this),effect=this.effect?.id;
    if(from.includes('Battle.endTurn')&&n===1&&d===5){
      if(this.sides.some(x=>x.active.some(p=>p?.item==='quickclaw')))return unknown('QUICK_CLAW',from);
      audit({kind:'INERT_QUICK_CLAW_ROLL',nonce:c.nonce});return false;
    }
    if(s!==0&&s!==1)return unknown('MISSING_ACTOR',from);
    if(from.includes('tryMoveHit')&&d===100)return take(c,'accuracy',s,c.draws[s][6])<n;
    if(from.includes('getDamage')&&Number.isInteger(16*n/d)){
      const hit=multihitIndex(this);
      if(hit===2)return effectDraw(c,'critical_hit2',s,0)<16*n/d;
      if(hit>2)return unknown('MULTIHIT_CRITICAL',{hit});
      return take(c,'critical',s,c.draws[s][4])<16*n/d;
    }
    if(effect==='par'&&n===1&&d===4)return take(c,'paralysis',s,c.draws[s][2])<25;
    if(effect==='frz'&&n===1&&d===5)return take(c,'thaw',s,c.draws[s][1])<20;
    if(effect==='confusion'&&n===1&&d===2)return take(c,'confusion',s,c.draws[s][3])<50;
    if(effect==='stall'&&n===1&&[1,2,4,8].includes(d))return take(c,'stall',s,c.draws[s][8])*d<8;
    return unknown('RANDOM_CHANCE',{from,n,d,effect});
  };
  Battle.prototype.random=function(m,n){
    const c=control();if(!c)return oldRandom.call(this,m,n);
    const from=caller(),s=seat(this);
    if(m===2&&n===6){const t=yawnSleeper(this);if(t)return 2+effectDraw(c,'yawn_sleep_duration',t.side.n,2);}
    if(s!==0&&s!==1)return unknown('MISSING_ACTOR',from);
    if(from.includes('randomizer')&&m===16&&n===undefined){
      const hit=multihitIndex(this);
      if(hit===2)return 15-effectDraw(c,'damage_hit2',s,1);
      if(hit>2)return unknown('MULTIHIT_DAMAGE',{hit});
      return 15-take(c,'damage',s,c.draws[s][5]);
    }
    if(from.includes('secondaries')&&m===100&&n===undefined)return take(c,'secondary',s,c.secondary[s]);
    if(from.includes('selfDrops')&&m===100&&n===undefined){
      // BA selfDrops rolls random(100) even for an unconditional self.boosts; the
      // value is unused when the chance is undefined. Any other shape stays unknown.
      const move=this.activeMove;
      if(!move?.self?.boosts||move.self.chance!==undefined)return unknown('SELF_DROPS_CHANCE',{move:move?.id});
      audit({kind:'INERT_SELF_DROPS',nonce:c.nonce,side:s,move:move.id});return 0;
    }
    const encore=encoreFrames.get(this);
    if(encore&&m===3&&n===7){
      const stack=new Error().stack.split('\n'),v=encore.target.volatiles?.encore;
      if(encore.nonce!==c.nonce||encore.actor!==s||this.activePokemon!==encore.source
        ||this.activeTarget!==encore.target||this.activeMove!==encore.effect
        ||v?.target!==encore.target||v.source!==encore.source||v.sourceEffect!==encore.effect
        ||!from.includes('durationCallback ('+encoreCallbackPath+':')
        ||!stack.some(x=>x.includes('Pokemon.addVolatile ('+encore.pokemonPath+':')))
        return unknown('ENCORE_DURATION_CONTEXT',{from,s});
      return 3+duration(c,'encore_duration',s);
    }
    if(this.effect?.id==='slp'&&m===2&&n===6)return 2+duration(c,'sleep_duration',s);
    if(this.effect?.id==='confusion'&&m===2&&n===6)return 2+duration(c,'confusion_duration',s);
    if(m===3&&n===7&&trapCallbackPath&&from.includes('durationCallback ('+trapCallbackPath+':'))
      return 3+duration(c,'partial_trap_duration',s);
    return unknown('RANDOM',{from,m,n,effect:this.effect?.id});
  };
  boundRandom=Battle.prototype.random;
  Battle.prototype.sample=function(items){
    const c=control();if(!c){
      const from=caller();
      if(initializing()?.allow_inert_gender&&pokemonPath
        &&from.includes('new Pokemon ('+pokemonPath+':108:')
        &&items.length===2&&items[0]==='M'&&items[1]==='F'){
        inert++;
        try{return oldSample.call(this,items);}finally{inert--;}
      }
      return oldSample.call(this,items);
    }
    const s=seat(this);if(s!==0&&s!==1)return unknown('MISSING_ACTOR','sample');
    if(this.activeMove?.id==='sleeptalk'&&items.length>=1&&items.length<=4)
      return items[take(c,'sleep_talk',s,c.sleep_talk[s])%items.length];
    return unknown('SAMPLE',{move:this.activeMove?.id,length:items.length});
  };
  const eachEvent=Battle.prototype.eachEvent;let eachEventId=null;
  Battle.prototype.eachEvent=function(eventid,...rest){
    const outer=eachEventId;eachEventId=eventid;
    try{return eachEvent.call(this,eventid,...rest);}finally{eachEventId=outer;}
  };
  const speedSort=Battle.prototype.speedSort;
  Battle.prototype.speedSort=function(list,comparator){
    const c=control();if(!c)return speedSort.call(this,list,comparator);
    const shuffle=this.prng.shuffle,battle=this;
    this.prng.shuffle=(items,start=0,end=items.length)=>{
      if(end-start<2)return;
      const group=items.slice(start,end);
      if(eachEventId&&group.every(x=>x&&x.side&&Array.isArray(x.moveSlots))
        &&group.every(x=>battle.findEventHandlers(x,eachEventId).length===0)){
        audit({kind:'INERT_EACH_EVENT_TIE',nonce:c.nonce,event:eachEventId,sides:group.map(x=>x.side.n)});return;
      }
      if(group.length===2&&group.every(x=>x.choice==='move')&&new Set(group.map(x=>x.pokemon.side.n)).size===2){
        const first=take(c,'move_tie',0,c.draws[0][0]);
        group.sort((a,b)=>(a.pokemon.side.n===first?-1:1));items.splice(start,2,...group);return;
      }
      // Same-holder residual countdowns (no callback, no onEnd: stall+protect, ...) commute;
      // their order cannot change the battle, so keep it and audit the tie as inert.
      if(group.every(x=>x.effectHolder&&x.effectHolder===group[0].effectHolder&&!x.callback&&x.end
        &&x.state?.duration&&x.effect&&!x.effect.onEnd)){
        audit({kind:'INERT_RESIDUAL_TIE',nonce:c.nonce,effects:group.map(x=>x.effect.id)});return;
      }
      if(group.length===2&&group.every(x=>x.effect?.id==='yawn'&&(x.effectHolder?.side?.n===0||x.effectHolder?.side?.n===1))
        &&new Set(group.map(x=>x.effectHolder.side.n)).size===2){
        audit({kind:'INERT_RESIDUAL_TIE',nonce:c.nonce,effects:['yawn','yawn']});return;
      }
      // Fidelity WH: White Herb (and Choice Band's onAfterMove) Item handlers on distinct holders commute.
      if(group.every(x=>x&&x.effect?.effectType==='Item'&&(x.effect.id==='whiteherb'||x.effect.id==='choiceband')
        &&x.effectHolder&&Array.isArray(x.effectHolder.moveSlots))&&group.some(x=>x.effect.id==='whiteherb')
        &&new Set(group.map(x=>x.effectHolder)).size===group.length){
        audit({kind:'INERT_ITEM_TIE',nonce:c.nonce,effects:group.map(x=>x.effect.id),sides:group.map(x=>x.effectHolder.side?.n)});return;
      }
      // Residual handler pairs on both actives: partiallytrapped (subOrder 9) and Perish.
      const residual={partiallytrapped:['partial_trap_tie',0],perishsong:['perish_tie',1]}[group[0]?.effect?.id];
      if(residual&&group.length===2&&group.every(x=>x.effect?.id===group[0].effect.id)
        &&group.every(x=>x.effectHolder?.side?.n===0||x.effectHolder?.side?.n===1)
        &&new Set(group.map(x=>x.effectHolder.side.n)).size===2){
        const [role,row]=residual,first=effectDraw(c,role,row,3);
        group.sort((a,b)=>(a.effectHolder.side.n===first?-1:1));items.splice(start,2,...group);return;
      }
      return unknown('TIE_ORDER',group.map(x=>x.choice||x.effect?.id||x.constructor?.name));
    };
    try{return speedSort.call(this,list,comparator);}finally{this.prng.shuffle=shuffle;}
  };
  const raw=PRNG.prototype.random;
  PRNG.prototype.random=function(...args){
    if(control())return unknown('UNMAPPED_RAW_RNG',args);
    const value=raw.apply(this,args);
    if(initializing()){
      if(inert&&args.length===1&&args[0]===2)audit({kind:'INERT_GENDER',args,raw:value});
      else audit({kind:'INITIAL_RANDOM',args,raw:value,stack:new Error().stack.split('\n').slice(1,9)});
    }
    return value;
  };
}
module.exports={install,validTape,TAPE_V1,TAPE_V2,DRAW_BOUNDS,EFFECT_BOUNDS};
