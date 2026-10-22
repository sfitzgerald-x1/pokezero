// Generated normal-ability Gen3 teams: request facts only, no search/outcomes.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Teams}=require(path.join(process.argv[2],'dist/sim/teams.js'));
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const rows=[];
for(const slot of ['p1','p2']){
  const b=new Battle({formatid:'gen3randombattle',seed:[1,2,3,4]});
  const holder=slot==='p1'?'p2':'p1';
  const trappedTeam=Teams.generate('gen3randombattle',{seed:[1,0,0,1]});
  const holderTeam=Teams.generate('gen3randombattle',{seed:[220,0,0,1]});
  if(holderTeam[0].species!=='Wobbuffet'||holderTeam[0].ability!=='Shadow Tag')throw Error('source fixture changed');
  b.setPlayer(slot,{name:'Acting',team:trappedTeam});
  b.setPlayer(holder,{name:'Holder',team:holderTeam});
  rows.push({slot,holder,acting_species:b[slot].active[0].species.name,
    format_id:'gen3randombattle',lines:b.log.slice(),trapped:b[slot].activeRequest.active[0].trapped});
  b.destroy();
}
console.log(JSON.stringify({rows,default_abilities:['Wobbuffet','Wynaut'].map(name=>
  ({species:name,abilities:Dex.forGen(3).species.get(name).abilities})),scientific_measurement:false}));
