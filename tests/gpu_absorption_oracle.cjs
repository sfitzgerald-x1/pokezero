// Authored pinned Gen 3 ability hooks; not a gameplay divergence panel.
const fs = require('node:fs');
const path = require('node:path');
const {Dex} = require(path.join(process.argv[2],'dist/sim/dex.js'));
const dex = Dex.mod('gen3');
const rows = JSON.parse(fs.readFileSync(0,'utf8'));
const results = rows.map(r=>{
  let hp=r.hp, flash=!!r.flash;
  const target = {status:r.status,volatiles:r.sub?{substitute:{}}:{},baseMaxhp:r.max_hp,
    hasType:type=>r.types.includes(type),
    addVolatile:()=>{const added=!flash;flash=true;return added;}};
  const ctx = {heal:amount=>{const old=hp;hp=Math.min(r.max_hp,hp+Math.max(1,Math.floor(amount)));return hp-old;},
    add:()=>{}};
  const move = {...dex.moves.get(r.move),type:r.type};
  const result = dex.abilities.get(r.ability).onTryHit.call(ctx,target,{},move);
  return {blocked:result===null,hp,flash};
});
const modifier = dex.abilities.get('flashfire').condition.onModifyDamagePhase1;
process.stdout.write(JSON.stringify({results,phase1:['Fire','Water'].map(type=>
  modifier.call({debug:()=>{},chainModify:x=>x},100,{},{},{type})??1)}));
