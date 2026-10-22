// Pinned Gen 3 singles interception boundaries, not a gameplay panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create() {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  b.setPlayer('p1',{name:'one',team:[{species:'Mew',ability:'Limber',moves:['pursuit','splash'],level:73},
    {species:'Ditto',ability:'Limber',moves:['splash'],level:73}]});
  b.setPlayer('p2',{name:'two',team:[{species:'Mew',ability:'Pressure',moves:['splash','wish'],level:73},
    {species:'Kyogre',ability:'Drizzle',moves:['splash'],level:73}]});
  for(const side of b.sides)for(const p of side.pokemon){
    p.baseMaxhp=p.maxhp=p.hp=1000;
    Object.assign(p.storedStats,{atk:77,def:200,spa:333,spd:200,spe:200});p.updateSpeed();
  }
  b.p1.active[0].types=['Dark'];
  b.randomizer=x=>x;
  // Force full paralysis/confusion if a BeforeMove event incorrectly runs,
  // but keep ordinary critical hits off for controlled damage comparisons.
  b.randomChance=(n,d)=>n===1&&d===16?false:true;
  return b;
}
function interception({hp=1000,status='',confusion=false,sub=false,wish=false,band=false}={}) {
  const b=create(),p=b.p1.active[0],old=b.p2.active[0],incoming=b.p2.pokemon[1];
  old.hp=hp;if(status)p.setStatus(status);
  if(status==='slp')p.statusState.time=1;
  if(confusion){p.addVolatile('confusion');p.volatiles.confusion.time=3;p.addVolatile('flinch');}
  if(sub){old.addVolatile('substitute');old.volatiles.substitute.hp=13;old.hp=hp;}
  if(band)p.setItem('choiceband');
  if(wish){b.actions.useMove('wish',old);b.fieldEvent('Residual');incoming.hp=600;}
  b.p2.addSideCondition('spikes',p,b.dex.moves.get('spikes'));b.p2.sideConditions.spikes.layers=3;
  const before=p.moveSlots[0].pp;
  b.makeChoices('move 1','switch 2');
  const out={oldHP:old.hp,incomingChosen:b.p2.active[0]===incoming,incomingHP:incoming.hp,
    weather:b.field.weather,usedPP:before-p.moveSlots[0].pp,confusion:p.volatiles.confusion?.time||0,
    lock:p.volatiles.choicelock?.move||'',wish:!!b.p2.slotConditions[0].wish};
  b.destroy();return out;
}
function damage(item='',boost=0,critical=false) {
  const b=create(),p=b.p1.active[0],q=b.p2.active[0];
  p.setItem(item);if(item==='lightball'){p.species=b.dex.species.get('pikachu');}
  p.boosts.spa=boost;q.boosts.spd=2;q.beingCalledBack=true;
  const move=b.dex.getActiveMove('pursuit');move.willCrit=critical;
  const amount=b.actions.getDamage(p,q,move);b.destroy();return amount;
}
process.stdout.write(JSON.stringify({ordinary:interception(),sub:interception({sub:true,band:true}),
  bypass:interception({status:'par',confusion:true,band:true}),sleep:interception({status:'slp'}),
  ko:interception({hp:1,wish:true,band:true}),
  damage:[damage(),damage('choiceband'),damage('lightball'),damage('',-2),damage('',-2,true),damage('',2,true)]}));
