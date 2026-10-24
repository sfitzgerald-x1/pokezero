// Actual pinned Gen3 authored battles; not real-team roots or coverage credit.
const fs=require('node:fs');
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const specs=JSON.parse(fs.readFileSync(0,'utf8'));
const rows=specs.map(spec=>{
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++) b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Run Away',moves:['painsplit','splash','protect','seismictoss'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const own=b.sides[spec.side].active[0],foe=b.sides[1-spec.side].active[0];
  for(const side of b.sides)for(const mon of side.pokemon){
    mon.baseMaxhp=mon.maxhp=mon.hp=1000;
    for(const key of Object.keys(mon.storedStats))mon.storedStats[key]=200;
    mon.storedStats.spe=mon===own?(spec.prior_damage?199:301):200;mon.updateSpeed();
    for(const slot of mon.moveSlots)slot.pp=8;
  }
  own.baseMaxhp=own.maxhp=spec.own_max;own.hp=spec.own;
  foe.baseMaxhp=foe.maxhp=spec.foe_max;foe.hp=spec.foe;
  if(spec.ghost)foe.types=['Ghost'];
  if(spec.own_sub){own.addVolatile('substitute');own.volatiles.substitute.hp=spec.own_sub;}
  if(spec.target_sub){foe.addVolatile('substitute');foe.volatiles.substitute.hp=spec.target_sub;}
  if(spec.switch)b.sides[1-spec.side].pokemon[1].hp=800;
  b.randomChance=(n,d)=>n>=d;b.random=()=>0;b.randomizer=x=>x;
  const choices=['move splash','move splash'];choices[spec.side]='move painsplit';
  choices[1-spec.side]=spec.switch?'switch 2':spec.protect?'move protect':spec.prior_damage?'move seismictoss':'move splash';
  b.makeChoices(...choices);
  const target=b.sides[1-spec.side].active[0];
  const result={hp:[own.hp,target.hp],pp:own.moveSlots[0].pp,
    own_sub:own.volatiles.substitute?.hp||0,target_sub:target.volatiles.substitute?.hp||0,
    active:b.sides[1-spec.side].pokemon.indexOf(target)};
  // Showdown moves the active to pokemon[0]; preserve the authored team identity.
  result.active=target.species.id==='ditto'?1:0;
  b.destroy();return result;
});
process.stdout.write(JSON.stringify(rows));
