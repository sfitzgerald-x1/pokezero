// Authored real Gen3 move/entry boundaries, not scientific games or roster replay.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const rows=[];
for(const hazardSide of [0,1])for(const layers of [0,1,2,3])for(const terminal of [false,true])for(const late of [false,true]){
  if((!layers&&terminal)||(!layers&&late))continue;
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let side=0;side<2;side++)b.setPlayer(`p${side+1}`,{name:`seat${side}`,team:[
    {species:side===0?'Electrode':'Snorlax',moves:[side===0?'Explosion':'Body Slam'],level:100},
    {species:'Swampert',ability:'Torrent',moves:['Protect'],level:100,item:'Leftovers'},
    {species:'Swampert',ability:'Torrent',moves:['Protect'],level:100}]});
  const old=b.sides.map(s=>s.active[0]),incoming=b.sides.map(s=>s.pokemon[1]);
  const later=b.sides.map(s=>s.pokemon[2]);
  for(let side=0;side<2;side++)for(const p of b.sides[side].pokemon){
    p.baseMaxhp=p.maxhp=p.hp=1000;p.types=['Normal'];
    for(const key of Object.keys(p.storedStats))p.storedStats[key]=200;
    p.storedStats.spe=side===0?301:101;p.updateSpeed();
  }
  for(let side=0;side<2;side++){
    old[side].hp=1;incoming[side].hp=800;incoming[side].setStatus('psn');
    incoming[side].storedStats.spe=(side===hazardSide)!==late?401:201;incoming[side].updateSpeed();
  }
  if(layers){
    for(let n=0;n<layers;n++)b.sides[hazardSide].addSideCondition('spikes',old[1-hazardSide],b.dex.moves.get('spikes'));
    if(late)for(let n=0;n<layers;n++)b.sides[1-hazardSide].addSideCondition('spikes',old[hazardSide],b.dex.moves.get('spikes'));
    incoming[hazardSide].hp=[0,125,166,250][layers];incoming[hazardSide].item='';
  }
  if(terminal){later[hazardSide].faint();b.faintMessages();}
  b.choose('p1','move 1');b.choose('p2','move 1');
  const requested=b.sides.map(s=>s.activeRequest?.forceSwitch?.[0]===true);
  b.choose('p1','switch 2');b.choose('p2','switch 2');
  const first={activeIsIncoming:b.sides.map((s,i)=>s.active[0]===incoming[i]),
    hp:incoming.map(p=>p.hp),requested:b.sides.map(s=>s.activeRequest?.forceSwitch?.[0]===true),ended:b.ended};
  let afterChain=null;
  if(layers&&!terminal){
    b.choose(`p${hazardSide+1}`,'switch 3');
    afterChain={hp:b.sides.map(s=>s.active[0].hp),ended:b.ended,
      requested:b.sides.map(s=>s.activeRequest?.forceSwitch?.[0]===true)};
  }
  rows.push({hazardSide,layers,terminal,late,requested,first,afterChain});b.destroy();
}
process.stdout.write(JSON.stringify({scientific_measurement:false,rows}));
