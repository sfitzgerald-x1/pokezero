// Independent Gen3 normal-switch versus completed Baton Pass lifecycle.
const fs=require('node:fs'),path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const rows=JSON.parse(fs.readFileSync(0,'utf8')).map(spec=>{
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:'Mew',ability:'Run Away',moves:['batonpass','splash'],level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const own=b.sides[spec.side].active[0];
  own.addVolatile('substitute');own.volatiles.substitute.hp=73;
  own.storedStats.spe=301;own.updateSpeed();
  b.randomChance=(n,d)=>n>=d;
  const choices=['move splash','move splash'];choices[spec.side]=spec.baton?'move batonpass':'switch 2';
  b.makeChoices(...choices);
  if(spec.baton){const replace=[undefined,undefined];replace[spec.side]='switch 2';b.makeChoices(...replace);}
  const active=b.sides[spec.side].active[0];
  const result={species:active.species.id,sub:active.volatiles.substitute?.hp||0,request:b.requestState};
  b.destroy();return result;
});
process.stdout.write(JSON.stringify(rows));
