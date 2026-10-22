// Authored direct-Baton scheduler fixtures; fixed damage, not damage parity.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create(own='batonpass',foe='pursuit',fast=true){
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
  {species:'Mew',ability:'Limber',moves:[side?foe:own],level:73},
  {species:'Ditto',ability:'Limber',item:'Choice Band',moves:['splash'],level:73}]});
 for(const side of b.sides)for(const p of side.pokemon){p.baseMaxhp=p.maxhp=p.hp=1000;}
 b.p1.active[0].storedStats.spe=fast?300:100;b.p2.active[0].storedStats.spe=200;
 for(const side of b.sides)side.active[0].updateSpeed();
 b.randomChance=(n,d)=>n>=d;
 const hits=[];b.actions.getDamage=(source,target,move)=>{
  if(move.basePower>0)hits.push({source:source.side.id,target:target.species.id,power:move.basePower});
  return move.basePower||0;};
 return{b,hits};
}
function pursuit(fast){
 const{b,hits}=create('batonpass','pursuit',fast),old=b.p1.active[0],q=b.p2.active[0];
 b.makeChoices('move 1','move 1');
 const before={request:b.requestState,hits:[...hits],pp:[old.moveSlots[0].pp,q.moveSlots[0].pp]};
 b.makeChoices('switch 2',undefined);
 const after={request:b.requestState,hits:[...hits],oldHP:old.hp,newHP:b.p1.active[0].hp,pp:q.moveSlots[0].pp};
 b.destroy();return{before,after};
}
function twice(){
 const{b}=create('batonpass','batonpass');b.makeChoices('move 1','move 1');
 const first={request:b.requestState,seats:b.sides.map(s=>!!s.activeRequest?.forceSwitch)};
 b.makeChoices('switch 2',undefined);
 const middle={request:b.requestState,active:b.sides.map(s=>s.active[0].species.id)};
 b.makeChoices(undefined,'switch 2');
 const after={request:b.requestState,active:b.sides.map(s=>s.active[0].species.id),turn:b.turn};
 b.destroy();return{first,middle,after};
}
function copied(){
 const{b}=create('batonpass','splash'),p=b.p1.active[0],next=b.p1.pokemon[1],q=b.p2.active[0];
 p.boosts.atk=2;p.boosts.spd=1;
 p.addVolatile('substitute');p.volatiles.substitute.hp=100;
 p.addVolatile('leechseed',q,b.dex.moves.get('leechseed'));
 p.addVolatile('flashfire');p.addVolatile('choicelock');
 next.hp=800;next.setStatus('psn');
 b.makeChoices('move 1','move 1');
 // Isolate copy of an existing confusion counter from BeforeMove RNG.
 p.addVolatile('confusion');p.volatiles.confusion.time=3;
 b.makeChoices('switch 2',undefined);
 const result={atk:next.boosts.atk,spd:next.boosts.spd,sub:next.volatiles.substitute?.hp,
  confusion:next.volatiles.confusion?.time,seed:!!next.volatiles.leechseed,
  flash:!!next.volatiles.flashfire,choice:!!next.volatiles.choicelock,
  status:next.status,item:next.item,hp:next.hp};
 b.destroy();return result;
}
process.stdout.write(JSON.stringify({fast:pursuit(true),slow:pursuit(false),twice:twice(),copied:copied()}));
