// Authored Gen3 status-event controls; no scientific roster or saved-root replay.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const dex=Dex.mod('gen3');
function sample(o={}) {
  const side=o.side||0;
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let seat=0;seat<2;seat++)b.setPlayer(`p${seat+1}`,{name:`seat${seat}`,team:[
    {species:'Mew',ability:seat===side?(o.sourceAbility||'Run Away'):(o.targetAbility||'Synchronize'),
      moves:[o.move||'thunderbolt','splash'],item:seat===side?(o.sourceLum?'Lum Berry':''):(o.targetLum?'Lum Berry':''),level:100},
    {species:'Ditto',ability:'Run Away',moves:['splash'],level:100}]});
  const p=b.sides[side].active[0],q=b.sides[1-side].active[0];
  p.baseMaxhp=p.maxhp=p.hp=q.baseMaxhp=q.maxhp=q.hp=1000;
  p.types=[o.sourceType||'Water'];q.types=['Water'];
  if(o.oldSource)p.setStatus(o.oldSource);
  if(o.oldTarget)q.setStatus(o.oldTarget);
  if(o.sourceSub)p.addVolatile('substitute');
  if(o.targetSub)q.addVolatile('substitute');
  if(o.safeguard)p.side.addSideCondition('safeguard',p,b.dex.moves.get('safeguard'));
  if(o.sourceKO)p.hp=0;
  if(o.targetKO)q.hp=1;
  b.randomChance=()=>true;b.random=(lo,hi)=>hi===undefined?0:lo;b.randomizer=x=>x;
  const move=b.dex.getActiveMove(o.move||'thunderbolt');move.accuracy=true;move.willCrit=false;
  b.actions.useMove(move,p,q);
  const result=[p.status,q.status,p.item,q.item];b.destroy();return result;
}
const cases={};
const inputs={
 secondary:[{},['par','par','','']],opposite:[{side:1},['par','par','','']],
 toxic:[{move:'toxic'},['psn','tox','','']],burn:[{move:'willowisp'},['brn','brn','','']],
 sleep:[{move:'sleeppowder'},['','slp','','']],freeze:[{move:'icebeam'},['','frz','','']],
 targetLum:[{targetLum:true},['par','','','']],sourceLum:[{sourceLum:true},['','par','','']],
 bothLum:[{sourceLum:true,targetLum:true},['','','','']],
 limber:[{sourceAbility:'Limber'},['','par','','']],
 waterVeil:[{move:'firepunch',sourceAbility:'Water Veil'},['','brn','','']],
 poisonImmune:[{move:'sludgebomb',sourceAbility:'Immunity'},['','psn','','']],
 fireType:[{move:'firepunch',sourceType:'Fire'},['','brn','','']],
 steelType:[{move:'sludgebomb',sourceType:'Steel'},['','psn','','']],
 groundPar:[{sourceType:'Ground'},['par','par','','']],
 ownSub:[{sourceSub:true},['par','par','','']],targetSub:[{targetSub:true},['','','','']],
 safeguard:[{safeguard:true},['','par','','']],
 existingSource:[{oldSource:'brn'},['brn','par','','']],existingTarget:[{oldTarget:'brn'},['','brn','','']],
 compound:[{move:'bodyslam',sourceAbility:'Synchronize',targetAbility:'Static',targetLum:true},['par','par','','']],
 compoundOpposite:[{side:1,move:'bodyslam',sourceAbility:'Synchronize',targetAbility:'Static',targetLum:true},['par','par','','']],
 faintTarget:[{targetKO:true},['','','','']],
};
for(const [name,[options,expected]]of Object.entries(inputs)) {
  const actual=sample(options);cases[name]={actual,expected};
  if(JSON.stringify(actual)!==JSON.stringify(expected))throw new Error(name+': '+JSON.stringify(actual)+' != '+JSON.stringify(expected));
}
const names=['_none','thunderbolt','firepunch','sludgebomb','icebeam','toxic','willowisp',
 'thunderwave','sleeppowder','bodyslam','splash','batonpass','struggle'];
const types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
 'steel','fire','water','grass','electric','psychic','ice','dragon','dark'];
const physical=new Set(['normal','fighting','flying','poison','ground','rock','bug','ghost','steel']);
const metadata={vocabulary:{moves:names,types,abilities:['_none','synchronize','static','limber','waterveil','immunity','shielddust'],
 items:['_none','lumberry'],species:['_none','ditto','pikachu','blissey','snorlax','skarmory','slaking']},
 moves:[null,...names.slice(1).map(id=>{const m=dex.moves.get(id);return {id,source_type:m.type,
 type:types.indexOf(m.type.toLowerCase()),base_power:m.basePower,accuracy:m.accuracy,priority:m.priority,
 category:m.category==='Status'?'Status':physical.has(m.type.toLowerCase())?'Physical':'Special',flags:m.flags};})],
 type_chart:types.slice(1).map(a=>types.slice(1).map(d=>{const k=dex.types.get(d).damageTaken[a[0].toUpperCase()+a.slice(1)];return k===1?2:k===2?0.5:k===3?0:1;}))};
process.stdout.write(JSON.stringify({metadata,catalog:{thunderboltChance:dex.moves.get('thunderbolt').secondaries[0].chance,
 synchronize:dex.abilities.get('synchronize').onAfterSetStatus.toString()},cases}));
