// Pinned Gen 3 authored hit/recoil boundaries, not a game-fidelity panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create(ability='Limber') {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++) b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:side?'Limber':ability,moves:['splash'],level:73},
    {species:'Ditto',ability:'Limber',moves:['splash'],level:73},
  ]});
  for(const side of b.sides){const p=side.active[0];p.baseMaxhp=p.maxhp=1000;p.hp=1000;}
  return b;
}
function fixed(move,types) {
  const b=create(),p=b.p1.active[0],q=b.p2.active[0];
  p.types=['Fighting'];p.boosts.atk=6;p.setStatus('brn');q.types=types;q.boosts.def=6;
  b.randomizer=damage=>Math.floor(damage*0.85);
  const action=b.dex.getActiveMove(move);action.accuracy=true;
  b.actions.useMove(action,p,q);const dealt=1000-q.hp;b.destroy();return dealt;
}
function sub(move,ability) {
  const b=create(ability),p=b.p1.active[0],q=b.p2.active[0];
  p.hp=50;q.addVolatile('substitute');q.volatiles.substitute.hp=19;q.hp=1000;
  b.actions.useMove(move,p,q);
  const result=[p.hp,q.hp,q.volatiles.substitute?.hp||0];b.destroy();return result;
}
const b=create();
const amounts=[1,3,6,7,11,19];
const recoil=Object.fromEntries(['struggle','doubleedge','takedown','submission','volttackle'].map(move=>
  [move,amounts.map(d=>b.actions.calcRecoilDamage(d,b.dex.moves.get(move)))]));
b.destroy();
process.stdout.write(JSON.stringify({amounts,recoil,fixed:{
  toss_rock:fixed('seismictoss',['Rock']),toss_ghost:fixed('seismictoss',['Ghost']),
  shade_psychic:fixed('nightshade',['Psychic']),shade_dark:fixed('nightshade',['Dark']),
  shade_normal:fixed('nightshade',['Normal']),sonic_ghost:fixed('sonicboom',['Ghost']),
  sonic_normal:fixed('sonicboom',['Normal']),
  dragon_steel:fixed('dragonrage',['Steel'])},sub:{
  struggle:sub('struggle','Limber'),struggle_rockhead:sub('struggle','Rock Head'),
  doubleedge:sub('doubleedge','Limber'),doubleedge_rockhead:sub('doubleedge','Rock Head')
}}));
