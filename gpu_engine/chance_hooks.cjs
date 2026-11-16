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
// Tape v4 also permits the reviewed own-holder Update callbacks below. Unknown
// callbacks still stop unresolved; an effect name alone is never a proof.
// Fidelity WH: White Herb handlers of two holders at equal Speed (onAnyAfterMove, onResidual order 29; SwitchIn ties
// are broken by effectOrder), with a Choice Band user's own onAfterMove (addVolatile choicelock) beside them, are an
// inert tie: each Item handler touches only its own holder and draws nothing, so the order cannot change the battle.
// Tape v4 appends phazing [2] and entry_ties [2], plus a resuming flag. Ordinary
// switch/entry ties read draws[0][11]/draws[1][10]; resumed requests read fresh
// entry_ties while the turn's phazing and residual draws remain saved. Lead
// entries use a separate initializing.initial_tie from the unsplit initial key.
const TAPE_V1='pokezero.gpu_chance_tape.v1',TAPE_V2='pokezero.gpu_chance_tape.v2',TAPE_V4='pokezero.gpu_chance_tape.v4';
const DRAW_BOUNDS=[2,100,100,100,16,16,100,4,8,2,2,2],EFFECT_BOUNDS=[16,16,4,2];
function validTape(c){
  const v4=c.schema===TAPE_V4,v2=c.schema===TAPE_V2||v4,row=(x,bounds)=>Array.isArray(x)&&x.length===bounds.length
    &&x.every((v,i)=>Number.isInteger(v)&&v>=0&&v<bounds[i]);
  return (c.schema===TAPE_V1||v2)&&typeof c.nonce==='string'&&!!c.nonce
    &&Array.isArray(c.draws)&&c.draws.length===2&&c.draws.every(x=>row(x,DRAW_BOUNDS))
    &&['secondary','sleep_talk'].every(k=>Array.isArray(c[k])&&c[k].length===2
      &&c[k].every(v=>Number.isInteger(v)&&v>=0&&v<(k==='secondary'?100:60)))
    &&(v2?Array.isArray(c.effect)&&c.effect.length===2&&c.effect.every(x=>row(x,EFFECT_BOUNDS)):c.effect===undefined)
    &&(v4?(row(c.phazing,[60,60])&&row(c.entry_ties,[2,2])&&typeof c.resuming==='boolean'):
      c.phazing===undefined&&c.entry_ties===undefined&&c.resuming===undefined);
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
  function phazingDraw(c,role,side){
    if(c.schema!==TAPE_V4)return unknown('PHAZING_STREAM_ABSENT',{role,side});
    const key='phazing_draw:'+side;
    if(used.has(key))return unknown('REPEATED_ROLE',key);
    used.add(key);return take(c,role,side,c.phazing[side]);
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
  const insertionFrames=new WeakMap();let gen3Dex;
  let fieldEventId=null,runEventId=null;
  const singles=b=>b.gen===3&&b.sides?.length===2&&b.sides.every(s=>s?.active?.length===1);
  const active=(b,p)=>p?.side&&b.sides[p.side.n]===p.side&&p.side.active[0]===p;
  const ownUpdates={Ability:new Set(['magmaarmor','limber','insomnia','vitalspirit','immunity','owntempo','oblivious','waterveil']),
    Item:new Set(['lumberry','cheriberry','chestoberry','pechaberry','rawstberry','aspearberry','persimberry','leppaberry']),
    // Attract's Update only removes its holder's volatile if the source left;
    // none of these Update handlers changes active membership.
    Condition:new Set(['attract'])};
  const originalCallback=(h,event)=>{
    const table={Ability:gen3Dex?.abilities,Item:gen3Dex?.items}[h.effect?.effectType]||gen3Dex?.conditions;
    return table&&typeof h.callback==='function'&&h.callback===table.get(h.effect.id)['on'+event];
  };
  const ownUpdate=(h,p)=>h.effectHolder===p&&ownUpdates[h.effect?.effectType]?.has(h.effect.id)&&originalCallback(h,'Update');
  function insertion(b,c,m,n){
    const frame=insertionFrames.get(b),other=b.queue?.list[m],p=frame?.pokemon;
    if(!frame||frame.choice!=='runSwitch'||n!==m+2||other?.choice!=='runSwitch'
      ||!singles(b)||!active(b,p)||!active(b,other.pokemon)||p.side===other.pokemon.side)
      return unknown('ENTRY_INSERTION_CONTEXT',{m,n});
    let first;
    if(c)first=take(c,'entry_tie',0,c.resuming?c.entry_ties[1]:c.draws[1][10]);
    else{
      first=initializing()?.initial_tie;
      if(first!==0&&first!==1)return unknown('INITIAL_TIE_ABSENT',null);
      audit({kind:'COUPLED_INITIAL',role:'initial_tie',raw:first});
    }
    // insertChoice samples an absolute queue index; insert before the existing
    // action exactly when the incoming side is the engine's first side.
    return m+(p.side.n===first?0:1);
  }
  let encoreCallbackPath='',trapCallbackPath='';
  if(pokemonPath){
    const path=require('path'),crypto=require('crypto');
    const resolved=require.resolve(pokemonPath),{Pokemon}=require(resolved);
    // Gen 3 inherits gen4's partiallytrapped.durationCallback: random(3,7).
    trapCallbackPath=path.resolve(path.dirname(resolved),'../data/mods/gen4/conditions.js');
    const {Dex}=require(path.join(path.dirname(resolved),'dex.js'));
    const dex=Dex.mod('gen3'),condition=dex.conditions.get('encore');gen3Dex=dex;
    const {BattleQueue}=require(path.join(path.dirname(resolved),'battle-queue.js'));
    const insert=BattleQueue.prototype.insertChoice;
    BattleQueue.prototype.insertChoice=function(choice,...rest){
      const outer=insertionFrames.get(this.battle);insertionFrames.set(this.battle,choice);
      try{return insert.call(this,choice,...rest);}finally{
        if(outer)insertionFrames.set(this.battle,outer);else insertionFrames.delete(this.battle);
      }
    };
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
    if(effect==='static'&&n===1&&d===3&&singles(this)&&from.includes('Battle.onDamagingHit')
      &&this.activeMove?.flags?.contact&&this.effectState?.target===this.sides[1-s].active[0])
      return phazingDraw(c,'static',s)<20;
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
    const c=control(),init=c?null:initializing();
    // Legacy probe initialization has no engine coordinate: keep its native
    // draw and INITIAL_RANDOM audit, which makes that case unresolved.
    // A supplied coordinate, including a malformed one, must use validation.
    if((c||(init&&Object.prototype.hasOwnProperty.call(init,'initial_tie')))
      &&insertionFrames.has(this)&&caller().includes('BattleQueue.insertChoice'))return insertion(this,c,m,n);
    if(!c)return oldRandom.call(this,m,n);
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
    const c=control();
    if((c||initializing())&&singles(this)&&this.effect?.id==='trace'&&caller().includes('Side.randomFoe')
      &&items.length===1&&active(this,items[0])){
      audit({kind:'INERT_FORCED_FOE',phase:c?'turn':'initial',nonce:c?.nonce});return items[0];
    }
    if(c&&caller().includes('Battle.getRandomSwitchable')&&['roar','whirlwind'].includes(this.activeMove?.id)){
      const s=seat(this),foe=this.sides[1-s];
      if(!singles(this)||!foe||!items.length||items.length>5||items.some(p=>p.side!==foe||p.isActive||p.fainted))
        return unknown('PHAZING_CONTEXT',null);
      // Team sets retain identity when Showdown swaps side.pokemon entries.
      const original=foe.team,ordered=items.slice().sort((a,b)=>original.indexOf(a.set)-original.indexOf(b.set));
      if(ordered.some(p=>!original.includes(p.set))||new Set(ordered.map(p=>original.indexOf(p.set))).size!==items.length)
        return unknown('PHAZING_IDENTITY',null);
      return ordered[phazingDraw(c,'phazing',s)%ordered.length];
    }
    if(!c){
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
  for(const name of ['fieldEvent','runEvent']){
    const original=Battle.prototype[name];
    Battle.prototype[name]=function(id,...rest){
      const outer=name==='fieldEvent'?fieldEventId:runEventId;
      if(name==='fieldEvent')fieldEventId=id;else runEventId=id;
      try{return original.call(this,id,...rest);}finally{
        if(name==='fieldEvent')fieldEventId=outer;else runEventId=outer;
      }
    };
  }
  const speedSort=Battle.prototype.speedSort;
  Battle.prototype.speedSort=function(list,comparator){
    const c=control();if(!c&&!initializing())return speedSort.call(this,list,comparator);
    const shuffle=this.prng.shuffle,battle=this;
    this.prng.shuffle=(items,start=0,end=items.length)=>{
      if(end-start<2)return;
      const group=items.slice(start,end);
      if(eachEventId&&group.every(x=>x&&x.side&&Array.isArray(x.moveSlots))
        &&group.every(x=>battle.findEventHandlers(x,eachEventId).every(h=>
          singles(battle)&&active(battle,x)&&(
            (eachEventId==='Update'&&ownUpdate(h,x))||
            // Gen3 weather chip reaches both actives before faintMessages.
            // Only the field's own sand/hail handler is covered; any other
            // Weather handler must acquire its own commutation proof.
            (eachEventId==='Weather'&&['sandstorm','hail'].includes(h.effect?.id)
              &&h.effectHolder===battle.field&&originalCallback(h,'Weather')))))){
        audit({kind:'INERT_EACH_EVENT_TIE',nonce:c?.nonce,phase:c?'turn':'initial',event:eachEventId,sides:group.map(x=>x.side.n)});return;
      }
      if(!c)return shuffle.call(battle.prng,items,start,end);
      if(group.length===2&&group.every(x=>x.choice==='move')&&new Set(group.map(x=>x.pokemon.side.n)).size===2){
        const first=take(c,'move_tie',0,c.draws[0][0]);
        group.sort((a,b)=>(a.pokemon.side.n===first?-1:1));items.splice(start,2,...group);return;
      }
      if(singles(battle)&&group.length===2&&['switch','instaswitch'].includes(group[0].choice)
        &&group.every(x=>x.choice===group[0].choice&&active(battle,x.pokemon))
        &&group[0].pokemon.side!==group[1].pokemon.side){
        const first=take(c,'switch_tie',0,c.resuming?c.entry_ties[0]:c.draws[0][11]);
        group.sort((a,b)=>a.pokemon.side.n===first?-1:1);items.splice(start,2,...group);return;
      }
      if(singles(battle)&&runEventId==='DisableMove'&&group.length===2
        &&new Set(group.map(h=>h.effect?.id)).size===2
        &&group.every(h=>['choicelock','encore'].includes(h.effect?.id)&&h.effectHolder===group[0].effectHolder
          &&active(battle,h.effectHolder)&&originalCallback(h,'DisableMove'))){
        audit({kind:'INERT_DISABLE_MOVE_TIE',nonce:c.nonce});return;
      }
      if(singles(battle)&&['TrapPokemon','MaybeTrapPokemon'].includes(runEventId)&&group.length===2
        &&group.every(h=>h.effect?.id==='magnetpull'&&active(battle,h.effectHolder)&&originalCallback(h,'Any'+runEventId))
        &&group[0].effectHolder.side!==group[1].effectHolder.side){
        // In singles the holder is not adjacent to itself; only the foe can
        // set the target's trapped/maybeTrapped flag. Neither handler clears it.
        audit({kind:'INERT_MAGNET_PULL_TIE',nonce:c.nonce,event:runEventId});return;
      }
      if(singles(battle)&&fieldEventId==='Residual'&&group.length===2
        &&group.every(h=>active(battle,h.effectHolder))&&group[0].effectHolder.side!==group[1].effectHolder.side){
        if(group.every(h=>h.effect?.id==='truant'&&h.order===27&&originalCallback(h,'Residual'))){
          audit({kind:'INERT_TRUANT_TIE',nonce:c.nonce});return;
        }
        const heals=new Set(['leftovers','oranberry','sitrusberry','liechiberry','ganlonberry','salacberry','petayaberry','apicotberry','lansatberry','starfberry']);
        const category=h=>heals.has(h.effect?.id)?['heal_tie',0,9,4]:
          ['psn','tox','brn'].includes(h.effect?.id)?['status_tie',1,9,6]:
          {speedboost:['ability_tie',1,0,3],leechseed:['seed_tie',0,10,5],encore:['encore_tie',1,11,14]}[h.effect?.id];
        const row=category(group[0]);
        if(row&&group.every(h=>category(h)?.[0]===row[0]&&h.order===10&&h.subOrder===row[3]&&originalCallback(h,'Residual'))){
          const [role,side,column]=row,first=take(c,role,side,c.draws[side][column]);
          group.sort((a,b)=>a.effectHolder.side.n===first?-1:1);items.splice(start,2,...group);return;
        }
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
module.exports={install,validTape,TAPE_V1,TAPE_V2,TAPE_V4,DRAW_BOUNDS,EFFECT_BOUNDS};
