// Real Showdown callback tests. No engine is loaded by this script.
const assert=require('node:assert/strict'),path=require('node:path');
const root=process.argv[2],hooks=process.argv[3];
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const {PRNG}=require(path.join(root,'dist/sim/prng.js'));
let tape=null,initial=null,events=[];
require(hooks).install(Battle,PRNG,()=>tape,e=>events.push(e),()=>initial,path.join(root,'dist/sim/pokemon.js'));
const row=()=>[0,99,99,99,15,0,0,0,0,0,0,0];let nonce=0;
function T(){return {schema:'pokezero.gpu_chance_tape.v4',nonce:String(++nonce),draws:[row(),row()],effect:[[15,0,0,0],[15,0,0,0]],
  secondary:[99,99],sleep_talk:[0,0],phazing:[0,0],entry_ties:[0,0],resuming:false};}
const mon=(species='Snorlax',ability='Thick Fat',item='')=>({species,ability,item,gender:'M',level:100,moves:['splash','tackle'],
  evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31}});
function battle(a=[mon()],b=[mon()]){
  tape=null;const x=new Battle({formatid:'gen3customgame@@@sleepclausemod',seed:[1,2,3,4]});
  x.setPlayer('p1',{name:'p1',team:a});x.setPlayer('p2',{name:'p2',team:b});return x;
}
const view=b=>b.sides.map(s=>s.pokemon.map(p=>({hp:p.hp,status:p.status,item:p.item,boosts:p.boosts,
  volatiles:Object.keys(p.volatiles).sort(),truant:p.truantTurn,moves:p.moveSlots.map(m=>[m.id,m.pp,!!m.disabled])})));
function controlled(t,fn){events=[];tape=t;try{return fn();}finally{tape=null;}}
function coupled(role){return events.filter(e=>e.kind==='COUPLED'&&e.role===role);}
let checks=0;
const updateCases=[['magmaarmor','','frz'],['limber','','par'],['insomnia','','slp'],['vitalspirit','','slp'],['immunity','','tox'],['waterveil','','brn'],
  ['owntempo','','confusion'],['oblivious','','attract'],['thickfat','lumberry','tox'],['thickfat','cheriberry','par'],
  ['thickfat','chestoberry','slp'],['thickfat','pechaberry','psn'],['thickfat','rawstberry','brn'],
  ['thickfat','aspearberry','frz'],['thickfat','persimberry','confusion'],['thickfat','leppaberry','pp']];
for(const [ability,item,condition] of updateCases){
  function run(order,tied){
    const b=battle([mon('Snorlax',ability,item)],[mon('Snorlax',ability,item)]);
    for(const s of b.sides){const p=s.active[0];
      if(condition==='pp')p.moveSlots[0].pp=0;
      else if(['confusion','attract'].includes(condition))p.volatiles[condition]={id:condition,target:p,source:b.sides[1-s.n].active[0]};
      else{p.status=condition;p.statusState={id:condition,target:p,time:3};}
    }
    const before=JSON.stringify(view(b));
    try{controlled(T(),()=>{if(tied)b.eachEvent('Update');else for(const s of order)b.runEvent('Update',b.sides[s].active[0]);});}
    catch(e){throw new Error(`${ability}/${item}: ${e.message}`);}
    if(tied)assert(events.some(e=>e.kind==='INERT_EACH_EVENT_TIE'));
    assert.notEqual(JSON.stringify(view(b)),before,`${ability}/${item} must actually change state`);
    return view(b);
  }
  assert.deepEqual(run([0,1]),run([1,0]));assert.deepEqual(run([0,1],true),run([1,0]));checks++;
}
// Truant is inert only in Residual; both flags really change in both orders.
for(const reverse of [false,true]){
  const b=battle([mon('Slaking','Truant')],[mon('Slaking','Truant')]);
  b.sides[0].active[0].truantTurn=false;b.sides[1].active[0].truantTurn=true;
  controlled(T(),()=>{if(!reverse)b.fieldEvent('Residual');else for(const s of [1,0]){
    const p=b.sides[s].active[0];b.singleEvent('Residual',p.getAbility(),p.abilityState,p);
  }});
  if(!reverse)assert(events.some(e=>e.kind==='INERT_TRUANT_TIE'));
  assert.deepEqual(b.sides.map(s=>s.active[0].truantTurn),[true,false]);
  checks++;
}
// DisableMove: the enabled-move intersection is independent of handler order,
// including stale Choice lock removal when the item or remembered move is gone.
for(const mode of ['normal','missing-item','missing-move']){
  const run=reverse=>{
    const b=battle([mon('Snorlax','Thick Fat',mode==='missing-item'?'':'choiceband')],[mon('Blissey')]),p=b.p1.active[0];
    p.volatiles.choicelock={id:'choicelock',target:p,move:mode==='missing-move'?'surf':'splash'};
    p.volatiles.encore={id:'encore',target:p,move:'tackle',duration:3};
    const hs=b.findEventHandlers(p,'DisableMove').filter(h=>['choicelock','encore'].includes(h.effect.id));
    assert.equal(hs.length,2);
    if(reverse)hs.reverse();
    controlled(T(),()=>{for(const h of hs)b.singleEvent('DisableMove',h.effect,h.state,p);});
    return view(b);
  };
  assert.deepEqual(run(false),run(true));checks++;
  const b=battle([mon('Snorlax','Thick Fat','choiceband')],[mon('Blissey')]),p=b.p1.active[0];
  p.volatiles.choicelock={id:'choicelock',target:p,move:'splash'};p.volatiles.encore={id:'encore',target:p,move:'tackle'};
  controlled(T(),()=>b.runEvent('DisableMove',p));assert(events.some(e=>e.kind==='INERT_DISABLE_MOVE_TIE'));
}
// Every residual family reads its own bit, with both bit values. Real handlers
// are driven by fieldEvent, so event/callback guards are exercised too.
for(const [id,role,r,c] of [['leftovers','heal_tie',0,9],['psn','status_tie',1,9],['speedboost','ability_tie',1,0],
  ['leechseed','seed_tie',0,10],['encore','encore_tie',1,11]])for(const bit of [0,1]){
  const b=battle([mon('Snorlax',id==='speedboost'?'Speed Boost':'Thick Fat',id==='leftovers'?id:'')],
    [mon('Snorlax',id==='speedboost'?'Speed Boost':'Thick Fat',id==='leftovers'?id:'')]);
  for(const s of b.sides){const p=s.active[0];p.hp=Math.floor(p.maxhp/2);p.activeTurns=2;
    if(id==='psn'){p.status=id;p.statusState={id,target:p};}
    if(id==='leechseed')p.volatiles[id]={id,target:p,sourceSlot:b.sides[1-s.n].active[0].getSlot()};
    if(id==='encore')p.volatiles[id]={id,target:p,move:'splash',duration:3};
  }
  const t=T();t.draws[r][c]=bit;controlled(t,()=>b.fieldEvent('Residual'));
  assert.deepEqual(coupled(role).map(e=>e.raw),[bit]);checks++;
}
// Entry insertion has absolute bounds, both incoming seats and queue offsets.
for(const incoming of [0,1])for(const offset of [0,1])for(const bit of [0,1])for(const resumed of [false,true]){
  const b=battle(),p=b.sides[incoming].active[0],q=b.sides[1-incoming].active[0];
  b.queue.clear();if(offset)b.queue.addChoice({choice:'start'});
  b.queue.addChoice({choice:'runSwitch',pokemon:q});
  const t=T();t.draws[1][10]=bit;t.resuming=resumed;t.entry_ties[1]=bit;
  controlled(t,()=>b.queue.insertChoice({choice:'runSwitch',pokemon:p}));
  assert.equal(b.queue.list.find(a=>a.choice==='runSwitch').pokemon.side.n,bit);assert.equal(coupled('entry_tie').length,1);checks++;
}
// Forced singles Trace samples consume no random draw even without an actor.
for(const initialPhase of [false,true]){
  const b=battle([mon('Porygon2','Trace')],[mon('Blissey','Natural Cure')]);b.effect=b.dex.abilities.get('trace');b.activePokemon=null;
  if(initialPhase)initial={initial_tie:0,allow_inert_gender:true};
  controlled(initialPhase?null:T(),()=>assert.equal(b.p1.randomFoe(),b.p2.active[0]));initial=null;
  assert(events.some(e=>e.kind==='INERT_FORCED_FOE'));assert(!events.some(e=>e.kind==='COUPLED'));checks++;
}
// Phazing uses original team identity even after the mutable party is reordered.
for(let n=1;n<=5;n++)for(let raw=0;raw<60;raw++){
  const team=['Blissey','Chansey','Ditto','Eevee','Pikachu','Mew'].map(s=>mon(s));
  const b=battle([mon('Skarmory')],team);b.activePokemon=b.p1.active[0];b.activeMove=b.dex.getActiveMove('whirlwind');
  for(let i=n+1;i<6;i++){b.p2.pokemon[i].fainted=true;b.p2.pokemon[i].hp=0;}
  const wanted=b.p2.pokemon[1+raw%n];b.p2.pokemon.splice(1,5,...b.p2.pokemon.slice(1).reverse());
  const t=T();t.phazing[0]=raw;
  controlled(t,()=>assert.equal(b.getRandomSwitchable(b.p2),wanted));checks++;
}
for(const bit of [0,1]){
  initial={initial_tie:bit,allow_inert_gender:true};events=[];
  const b=battle([mon('Groudon','Drought')],[mon('Kyogre','Drizzle')]);initial=null;
  assert.equal(b.field.weather,bit?'sunnyday':'raindance');
  assert(events.some(e=>e.kind==='COUPLED_INITIAL'));assert(!events.some(e=>e.kind==='INITIAL_RANDOM'),JSON.stringify(events));checks++;
  const sw=battle([mon(),mon('Groudon','Drought')],[mon(),mon('Kyogre','Drizzle')]);
  const t=T();t.draws[0][11]=bit;
  controlled(t,()=>sw.makeChoices('switch 2','switch 2'));
  assert.equal(sw.field.weather,bit?'sunnyday':'raindance');assert.equal(coupled('switch_tie').length,1);checks++;
}
// A handler with the same name but a different callback never inherits an
// order-irrelevance proof. Nor does an unreviewed own-holder ability.
for(const forged of [false,true]){
  const b=battle([mon('Snorlax',forged?'Limber':'Disguise')],[mon('Snorlax',forged?'Limber':'Disguise')]);
  if(forged){const original=b.findEventHandlers;b.findEventHandlers=function(...args){return original.apply(this,args).map(h=>({...h,callback(){}}));};}
  assert.throws(()=>controlled(T(),()=>b.eachEvent('Update')),/SPIKE_UNRESOLVED:TIE_ORDER/);checks++;
}
process.stdout.write(JSON.stringify({checks})+'\n');
