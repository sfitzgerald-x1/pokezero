// Pinned Gen3 authored battles, not real-team roots or qualification credit.
const fs=require('node:fs');
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const rows=JSON.parse(fs.readFileSync(0,'utf8')).map(spec=>{
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Run Away',moves:['bellydrum','splash','protect','seismictoss'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const own=b.sides[spec.side].active[0],foe=b.sides[1-spec.side].active[0];
  for(const side of b.sides)for(const mon of side.pokemon){
    mon.baseMaxhp=mon.maxhp=mon.hp=1000;
    for(const key of Object.keys(mon.storedStats))mon.storedStats[key]=200;
    mon.storedStats.spe=mon===own?(spec.prior_damage?199:301):200;mon.updateSpeed();
    for(const slot of mon.moveSlots)slot.pp=8;
  }
  own.baseMaxhp=own.maxhp=spec.maxhp;own.hp=spec.hp;own.boosts.atk=spec.atk;
  if(spec.sub){own.addVolatile('substitute');own.volatiles.substitute.hp=spec.sub;}
  b.randomChance=(n,d)=>n>=d;b.random=()=>0;b.randomizer=x=>x;
  const choices=['move splash','move splash'];choices[spec.side]='move bellydrum';
  choices[1-spec.side]=spec.protect?'move protect':spec.prior_damage?'move seismictoss':'move splash';
  b.makeChoices(...choices);
  const row={hp:own.hp,atk:own.boosts.atk,pp:own.moveSlots[0].pp,sub:own.volatiles.substitute?.hp||0};
  b.destroy();return row;
});
process.stdout.write(JSON.stringify(rows));
