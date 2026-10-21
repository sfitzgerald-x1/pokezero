// Authored Gen 3 PP events and target classification, not a game-fidelity panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const dex=Dex.mod('gen3');
function cost(move,{ghost=false,protected=false,asleep=false,pp=10,switchIn=false}={}) {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++) b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:side?'Mew':ghost?'Gengar':'Mew',ability:side?(switchIn?'Limber':'Pressure'):'Limber',
      moves:side?['splash','protect']:[move],level:100},
    {species:'Ditto',ability:side?'Pressure':'Limber',moves:['splash'],level:100},
  ]});
  const p=b.p1.active[0];
  p.moveSlots[0].pp=pp;
  if(asleep){p.setStatus('slp');p.statusState.time=10;}
  b.makeChoices('move 1',switchIn?'switch 2':protected?'move 2':'move 1');
  const used=pp-p.moveSlots[0].pp;
  b.destroy();return used;
}
const nonFoe=dex.moves.all().filter(m=>m.exists&&m.gen<=3&&
  ['self','allySide','allyTeam','allies','adjacentAlly'].includes(m.target)&&!m.flags.mustpressure)
  .map(m=>m.id).sort();
process.stdout.write(JSON.stringify({non_foe:nonFoe,costs:{
  tackle:cost('tackle'),toxic:cost('toxic'),seismictoss:cost('seismictoss'),
  recover:cost('recover'),haze:cost('haze'),raindance:cost('raindance'),
  spikes:cost('spikes'),reflect:cost('reflect'),healbell:cost('healbell'),
  curse_normal:cost('curse'),curse_ghost:cost('curse',{ghost:true}),
  protected:cost('tackle',{protected:true}),asleep:cost('tackle',{asleep:true}),
  one_pp:cost('tackle',{pp:1}),incoming_pressure:cost('tackle',{switchIn:true}),
}}));
