// Authored Gen 3 linked-trap fixtures, not a gameplay/RNG parity campaign.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create(move='meanlook',foe='Mew'){
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
  {species:side?foe:'Mew',ability:'Limber',moves:side?['batonpass','splash','seismictoss']:[move,'batonpass','splash'],level:73},
  {species:'Ditto',ability:'Limber',moves:['splash','batonpass'],level:73}]});
 for(const side of b.sides)for(const p of side.pokemon){p.baseMaxhp=p.maxhp=p.hp=1000;}
 b.p1.active[0].storedStats.spe=300;b.p2.active[0].storedStats.spe=200;
 for(const side of b.sides)side.active[0].updateSpeed();b.randomChance=(n,d)=>n>=d;
 return b;
}
function bound(b){return{active:b.sides.map(s=>s.active[0].species.id),
 trapped:!!b.p2.active[0].volatiles.trapped,owner:b.p2.active[0].volatiles.trapped?.linkedPokemon?.[0]?.species.id||null,
 request:b.requestState,canSwitch:!b.p2.activeRequest?.active?.[0]?.trapped};}
function chain(){
 const b=create();b.makeChoices('move 1','move 2');const initial=bound(b);
 b.makeChoices('move 2','move 2');b.makeChoices('switch 2',undefined);const ownerPass=bound(b);
 b.makeChoices('move 1','move 1');b.makeChoices(undefined,'switch 2');const targetPass=bound(b);
 const oldOwner=b.p1.pokemon.findIndex(p=>p.species.id==='mew')+1;
 b.makeChoices(`switch ${oldOwner}`,'move 1');const ownerLeaves=bound(b);
 b.destroy();return{initial,ownerPass,targetPass,ownerLeaves};
}
function eligibility(move,foe='Mew',condition=''){
 const b=create(move,foe),target=b.p2.active[0];
 if(condition){target.addVolatile(condition);if(condition==='substitute')target.volatiles.substitute.hp=100;}
 b.makeChoices('move 1','move 2');const result=bound(b);b.destroy();return result;
}
function ownerFaint(){
 const b=create();b.makeChoices('move 1','move 2');b.p1.active[0].hp=1;
 b.makeChoices('move 3','move 3');const result=bound(b);b.destroy();return result;
}
process.stdout.write(JSON.stringify({chain:chain(),meanSub:eligibility('meanlook','Mew','substitute'),
 webSub:eligibility('spiderweb','Mew','substitute'),blockProtect:eligibility('block','Mew','protect'),
 ghost:eligibility('meanlook','Gengar'),web:eligibility('spiderweb'),block:eligibility('block'),ownerFaint:ownerFaint()}));
