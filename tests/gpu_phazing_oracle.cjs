// Authored event-order fixtures. Bench choice is fixed, not RNG parity.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create(own='roar',foe='splash',ability='Limber'){
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
  {species:'Mew',ability:side?ability:'Limber',moves:[side?foe:own],level:73},
  {species:'Ditto',ability:'Limber',moves:['splash'],level:73},
  {species:'Chansey',ability:'Limber',moves:['splash'],level:73}]});
 for(const side of b.sides)for(const p of side.pokemon){p.baseMaxhp=p.maxhp=p.hp=1000;}
 b.p1.active[0].storedStats.spe=300;b.p2.active[0].storedStats.spe=200;
 for(const side of b.sides)side.active[0].updateSpeed();
 b.randomChance=(n,d)=>n>=d;
 b.getRandomSwitchable=side=>side.pokemon.find(p=>!p.isActive&&p.hp>0)||null;
 return b;
}
function ordinary(own='roar',ability='Limber',condition=''){
 const b=create(own,'splash',ability),old=b.p2.active[0],next=b.p2.pokemon[1];
 if(condition)old.addVolatile(condition);
 if(ability==='Natural Cure')old.setStatus('tox');
 old.boosts.atk=2;old.addVolatile('substitute');old.volatiles.substitute.hp=100;
 b.makeChoices('move 1','move 1');
 const result={active:b.p2.active[0].species.id,oldStatus:old.status,oldBoost:old.boosts.atk,
  oldSub:!!old.volatiles.substitute,newHP:next.hp,request:b.requestState,turn:b.turn};
 b.destroy();return result;
}
function cancelled(){
 const b=create('roar','whirlwind'),old=b.p2.active[0];
 b.makeChoices('move 1','move 1');
 const result={active:b.sides.map(s=>s.active[0].species.id),oldPP:old.moveSlots[0].pp,request:b.requestState};
 b.destroy();return result;
}
function hazards(){
 const b=create(),p=b.p1.active[0],old=b.p2.active[0],next=b.p2.pokemon[1];
 p.hp=500;next.hp=1;b.actions.useMove('wish',p);b.fieldEvent('Residual');
 b.p2.addSideCondition('spikes',p,b.dex.moves.get('spikes'));b.p2.sideConditions.spikes.layers=3;
 b.field.setWeather('sunnyday',p,b.dex.moves.get('sunnyday'));b.field.weatherState.duration=4;
 b.makeChoices('move 1','move 1');
 const before={request:b.requestState,oldHP:old.hp,newHP:next.hp,sourceHP:p.hp,
  wish:b.p1.slotConditions[0].wish.duration,weather:b.field.weatherState.duration};
 // The outgoing Mew is still living; choose the remaining Chansey instead.
 const slot=b.p2.pokemon.findIndex(p=>p.species.id==='chansey')+1;b.makeChoices(undefined,`switch ${slot}`);
 const after={request:b.requestState,sourceHP:p.hp,wish:!!b.p1.slotConditions[0].wish,
  weather:b.field.weatherState.duration,active:b.p2.active[0].species.id,hp:b.p2.active[0].hp};
 b.destroy();return{before,after};
}
process.stdout.write(JSON.stringify({normal:ordinary(),natural:ordinary('roar','Natural Cure'),
 suction:ordinary('roar','Suction Cups'),soundRoar:ordinary('roar','Soundproof'),
 soundWind:ordinary('whirlwind','Soundproof'),rooted:ordinary('roar','Limber','ingrain'),
 protect:ordinary('roar','Limber','protect'),cancelled:cancelled(),hazards:hazards()}));
