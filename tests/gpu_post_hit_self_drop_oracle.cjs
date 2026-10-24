// Authored pinned Gen3 battles; no fresh-panel replay or coverage credit.
const fs=require('node:fs'),path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const specs=JSON.parse(fs.readFileSync(0,'utf8'));
const rows=specs.map(spec=>{
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Run Away',moves:['overheat','psychoboost','splash',spec.endure?'endure':'protect'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const own=b.sides[spec.side].active[0],foe=b.sides[1-spec.side].active[0];
  for(const side of b.sides)for(const mon of side.pokemon){
    mon.baseMaxhp=mon.maxhp=mon.hp=1000;
    for(const key of Object.keys(mon.storedStats))mon.storedStats[key]=200;
    for(const slot of mon.moveSlots)slot.pp=8;
    mon.types=['Normal'];
  }
  own.storedStats.spe=301;own.updateSpeed();foe.updateSpeed();
  own.boosts.spa=spec.stage||0;
  if(spec.dark)foe.types=['Dark'];
  if(spec.absorb)foe.setAbility('flashfire');
  if(spec.sub){foe.addVolatile('substitute');foe.volatiles.substitute.hp=spec.sub;}
  if(spec.ko||spec.endure)foe.hp=1;
  b.randomChance=(n,d)=>n>=d||(d===100&&!spec.miss);
  b.random=()=>spec.miss?99:0;b.randomizer=x=>x;
  const choices=['move splash','move splash'];choices[spec.side]=`move ${spec.move}`;
  choices[1-spec.side]=spec.protect?'move protect':spec.endure?'move endure':'move splash';
  const before=foe.hp;
  b.makeChoices(...choices);
  const row={stage:own.boosts.spa,pp:own.moveSlots[spec.move==='overheat'?0:1].pp,
    hp:foe.hp,damaged:foe.hp<before,sub:foe.volatiles.substitute?.hp||0};
  b.destroy();return row;
});
process.stdout.write(JSON.stringify(rows));
