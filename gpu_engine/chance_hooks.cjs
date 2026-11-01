// Diagnostic-only semantic coupling. Never imported by the production bridge.
function install(Battle,PRNG,load,audit,initializing=()=>null,pokemonPath='') {
  let nonce,used=new Set(),inert=0;
  const control=()=>{
    const c=load();if(!c)return null;
    const bounds=[2,100,100,100,16,16,100,4,8,2,2,2];
    if(c.schema!=='pokezero.gpu_chance_tape.v1'||typeof c.nonce!=='string'||!c.nonce
      ||!Array.isArray(c.draws)||c.draws.length!==2
      ||c.draws.some(x=>x.length!==12||x.some((v,i)=>!Number.isInteger(v)||v<0||v>=bounds[i]))
      ||['secondary','sleep_talk'].some(k=>!Array.isArray(c[k])||c[k].length!==2
        ||c[k].some(v=>!Number.isInteger(v)||v<0||v>=(k==='secondary'?100:60))))throw Error('invalid chance tape');
    if(c.nonce!==nonce){nonce=c.nonce;used=new Set();}return c;
  };
  function unknown(kind,detail){audit({kind:'UNRESOLVED',role:kind,detail});throw Error('SPIKE_UNRESOLVED:'+kind);}
  function take(c,role,side,value){
    const key=role+':'+side;if(used.has(key))return unknown('REPEATED_ROLE',key);
    used.add(key);audit({kind:'COUPLED',nonce:c.nonce,role,side,raw:value});return value;
  }
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
  let encoreCallbackPath='';
  if(pokemonPath){
    const path=require('path'),crypto=require('crypto');
    const resolved=require.resolve(pokemonPath),{Pokemon}=require(resolved);
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
    if(from.includes('getDamage')&&Number.isInteger(16*n/d))return take(c,'critical',s,c.draws[s][4])<16*n/d;
    if(effect==='par'&&n===1&&d===4)return take(c,'paralysis',s,c.draws[s][2])<25;
    if(effect==='frz'&&n===1&&d===5)return take(c,'thaw',s,c.draws[s][1])<20;
    if(effect==='confusion'&&n===1&&d===2)return take(c,'confusion',s,c.draws[s][3])<50;
    if(effect==='stall'&&n===1&&[1,2,4,8].includes(d))return take(c,'stall',s,c.draws[s][8])*d<8;
    return unknown('RANDOM_CHANCE',{from,n,d,effect});
  };
  Battle.prototype.random=function(m,n){
    const c=control();if(!c)return oldRandom.call(this,m,n);
    const from=caller(),s=seat(this);if(s!==0&&s!==1)return unknown('MISSING_ACTOR',from);
    if(from.includes('randomizer')&&m===16&&n===undefined)return 15-take(c,'damage',s,c.draws[s][5]);
    if(from.includes('secondaries')&&m===100&&n===undefined)return take(c,'secondary',s,c.secondary[s]);
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
  const speedSort=Battle.prototype.speedSort;
  Battle.prototype.speedSort=function(list,comparator){
    const c=control();if(!c)return speedSort.call(this,list,comparator);
    const shuffle=this.prng.shuffle;
    this.prng.shuffle=(items,start=0,end=items.length)=>{
      if(end-start<2)return;
      const group=items.slice(start,end);
      if(group.length===2&&group.every(x=>x.choice==='move')&&new Set(group.map(x=>x.pokemon.side.n)).size===2){
        const first=take(c,'move_tie',0,c.draws[0][0]);
        group.sort((a,b)=>(a.pokemon.side.n===first?-1:1));items.splice(start,2,...group);return;
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
module.exports={install};
