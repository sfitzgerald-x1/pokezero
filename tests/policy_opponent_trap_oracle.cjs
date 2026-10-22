// Authored Gen3 private-request events, not historical states or strength games.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const cases=[
  ['shadowtag','Wobbuffet','Shadow Tag','Pikachu','Static'],
  ['arenatrap','Dugtrio','Arena Trap','Pikachu','Static'],
  ['magnetpull','Magneton','Magnet Pull','Metagross','Clear Body'],
];
const rows=[];
for(const [kind,foe,ability,own,ownAbility] of cases)for(const slot of ['p1','p2']){
  const battle=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  const mon=(species,ability)=>({species,name:species,ability,level:100,moves:['Tackle'],nature:'Serious'});
  const other=slot==='p1'?'p2':'p1';
  battle.setPlayer(slot,{name:'Acting',team:[mon(own,ownAbility),mon('Snorlax','Immunity')]});
  battle.setPlayer(other,{name:'Foe',team:[mon(foe,ability),mon('Snorlax','Immunity')]});
  const side=battle[slot], before=structuredClone(side.activeRequest);
  const turnBefore=battle.turn;
  const accepted=side.choose('switch 2');
  const after=structuredClone(side.activeRequest);
  rows.push({case:kind,seat:slot,turn_before:turnBefore,turn_after:battle.turn,
    actual_trapped:side.active[0].trapped,before:before.active[0],
    switch_accepted:accepted,choice_error:side.choice.error,after:after.active[0]});
  battle.destroy();
}
console.log(JSON.stringify({schema:'pokezero.authored_trap_request_probe.v1',scientific_measurement:false,
  historical_cause_established:false,rows}));
