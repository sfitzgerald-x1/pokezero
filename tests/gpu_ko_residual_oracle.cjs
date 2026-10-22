// Gen 3 action-phase replacement precedes residuals; authored fixtures only.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create() {
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
  {species:'Mew',ability:'Limber',moves:side?['splash','wish']:['seismictoss','wish'],level:73},
  {species:'Ditto',ability:'Limber',moves:['splash'],level:73},
  {species:'Wobbuffet',ability:'Limber',moves:['splash'],level:73}]});
 for(const side of b.sides)for(const p of side.pokemon){p.baseMaxhp=p.maxhp=p.hp=1000;p.storedStats.spe=200;p.updateSpeed();}
 b.p1.active[0].storedStats.spe=300;b.p1.active[0].updateSpeed();
 b.p1.active[0].hp=500;b.p2.active[0].hp=1;
 return b;
}
function snapshot(b){return{request:b.requestState,hp:b.p1.active[0].hp,wish:b.p1.slotConditions[0].wish?.duration||0,
 weatherDuration:b.field.weatherState.duration||0,turn:b.turn};}
function sourceWish(hazards=false){
 const b=create(),p=b.p1.active[0],first=b.p2.pokemon[1],last=b.p2.pokemon[2];
 b.actions.useMove('wish',p);b.fieldEvent('Residual');
 if(hazards){first.hp=1;last.hp=600;b.p2.addSideCondition('spikes',p,b.dex.moves.get('spikes'));b.p2.sideConditions.spikes.layers=3;
  b.field.setWeather('sunnyday',p,b.dex.moves.get('sunnyday'));b.field.weatherState.duration=4;}
 b.makeChoices('move 1','move 1');const before=snapshot(b);
 b.makeChoices(undefined,'switch 2');const middle=snapshot(b);
 if(hazards)b.makeChoices(undefined,'switch 3');
 const after=snapshot(b);const result={before,middle,after,recipient:b.p2.active[0].hp};b.destroy();return result;
}
function recipientWish(){
 const b=create(),q=b.p2.active[0],next=b.p2.pokemon[1];next.hp=600;
 b.actions.useMove('wish',q);b.fieldEvent('Residual');b.makeChoices('move 1','move 1');
 const before={request:b.requestState,wish:b.p2.slotConditions[0].wish?.duration||0,oldHP:q.hp};
 b.makeChoices(undefined,'switch 2');const after={hp:b.p2.active[0].hp,wish:!!b.p2.slotConditions[0].wish};
 b.destroy();return{before,after};
}
function suppressorKO(){
 const b=create(),p=b.p1.active[0],q=b.p2.active[0];p.hp=1000;
 q.setAbility('airlock');b.field.setWeather('sandstorm',p,b.dex.moves.get('sandstorm'));
 b.field.weatherState.duration=4;b.makeChoices('move 1','move 1');
 const before=snapshot(b);b.makeChoices(undefined,'switch 2');const after=snapshot(b);
 b.destroy();return{before,after};
}
process.stdout.write(JSON.stringify({source:sourceWish(),hazards:sourceWish(true),recipient:recipientWish(),suppressor:suppressorKO()}));
