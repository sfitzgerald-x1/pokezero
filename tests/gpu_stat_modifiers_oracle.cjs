// Authored pinned Gen3 stat/damage events, not historical roots or a game panel.
const fs=require('node:fs'),path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const rows=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(rows.map(row=>{
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer('p'+(side+1),{name:'seat'+side,team:[{
    species:side?row.defender:row.attacker,
    item:side?row.defender_item:row.item,
    ability:side?'Run Away':row.ability,moves:['tackle','watergun','pursuit','splash'],level:73,
  }]});
  const p=b.p1.active[0],q=b.p2.active[0];
  for(const mon of [p,q]){
    mon.baseMaxhp=mon.maxhp=mon.hp=10000;mon.types=['Bug'];
    Object.assign(mon.storedStats,{atk:row.attack,spa:row.attack,def:row.defense,spd:row.defense,spe:200});
    mon.boosts.atk=mon.boosts.spa=row.attack_stage;
    mon.boosts.def=mon.boosts.spd=row.defense_stage;
  }
  if(row.burned)p.setStatus('brn');
  b.randomizer=x=>x;b.randomChance=()=>!!row.critical;
  const move=b.dex.getActiveMove(row.pursuit?'pursuit':row.physical?'tackle':'watergun');
  move.basePower=row.pursuit?40:50;move.willCrit=!!row.critical;
  if(row.pursuit)q.beingCalledBack=true; // pinned ModifyMove doubles to80
  const amount=row.confusion?b.actions.getDamage(p,p,40):b.actions.getDamage(p,q,move);
  const out={damage:amount,
    modified_atk:b.runEvent('ModifyAtk',p,p,move,row.attack),
    modified_spa:b.runEvent('ModifySpA',p,p,move,row.attack),
    modified_spd:b.runEvent('ModifySpD',q,q,move,row.defense)};
  b.destroy();return out;
})));
