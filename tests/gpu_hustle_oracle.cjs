// Authored events against pinned Showdown; no scientific-root replay.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const dex=Dex.mod('gen3');
function battle(ability='Hustle',item='Choice Band') {
  const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
  for(let s=0;s<2;s++)b.setPlayer('p'+(s+1),{name:'seat'+s,team:[{
    species:s?'Blissey':'Delibird',ability:s?'Natural Cure':ability,item:s?'':item,
    moves:['hiddenpowerground','hiddenpowerice','aerialace','toxic'],level:73}]});
  b.randomizer=x=>x;
  return b;
}
const b=battle(),p=b.p1.active[0],q=b.p2.active[0];
const physical=b.dex.getActiveMove('hiddenpowerground');physical.type='Ground';
const counts=[],boosts=[1,4/3,5/3,2,7/3,8/3,3];
for(let base=0;base<=100;base++)for(let a=-6;a<=6;a++)for(let e=-6;e<=6;e++){
  let value=base;
  value=a>0?value*boosts[a]:value/boosts[-a];
  value=e>0?value/boosts[e]:value*boosts[-e];
  const actual=b.runEvent('ModifyAccuracy',q,p,physical,value);
  counts.push(Array.from({length:100},(_,roll)=>roll<actual).filter(Boolean).length);
}
// Read actual move execution's accuracy event, including staged float quirks,
// Status/type handling, dynamic Hidden Power and always-hit bypass.
const accuracy=[];
for(const [name,a,e]of [['hiddenpowerground',0,0],['hiddenpowerice',0,0],
    ['toxic',0,0],['aerialace',-6,6],['hiddenpowerground',-4,0],
    ['toxic',1,2],['hiddenpowerground',1,2]]){
  const c=battle(),source=c.p1.active[0],target=c.p2.active[0];
  source.hpType=name==='hiddenpowerice'?'Ice':'Ground';source.hpPower=70;
  source.boosts.accuracy=a;target.boosts.evasion=e;
  let seen=null;
  c.randomChance=(n,d)=>{if(d===100){seen=n;return false;}return false;};
  c.actions.useMove(name,source,target);
  accuracy.push({name,a,e,threshold:seen});c.destroy();
}
const damage=[];
for(const ability of ['Hustle','Vital Spirit'])for(const item of ['','Choice Band'])
 for(const physicalMove of [false,true])for(const attack_stage of [-2,0,2])for(const critical of [false,true]){
  const c=battle(ability,item),source=c.p1.active[0],target=c.p2.active[0];
  source.types=target.types=['Bug'];
  Object.assign(source.storedStats,{atk:103,spa:103,def:101,spd:101});
  Object.assign(target.storedStats,{atk:103,spa:103,def:101,spd:101});
  source.boosts.atk=source.boosts.spa=attack_stage;
  const move=c.dex.getActiveMove(physicalMove?'doubleedge':'icebeam');
  move.basePower=50;move.willCrit=critical;c.randomChance=()=>critical;
  const value=c.actions.getDamage(source,target,move);
  damage.push({ability,item,physical:physicalMove,attack_stage,critical,value});c.destroy();
 }
const confusion=[];
for(const burned of [false,true])for(const item of ['','Choice Band']){
  const c=battle('Hustle',item),source=c.p1.active[0];
  Object.assign(source.storedStats,{atk:103,def:101});
  if(burned)source.setStatus('brn');
  confusion.push({burned,item,value:c.actions.getDamage(source,source,40)});c.destroy();
}
const names=['_none','hiddenpowerground','hiddenpowerice','doubleedge','aerialace','toxic','splash','struggle'];
const types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
 'steel','fire','water','grass','electric','psychic','ice','dragon','dark'];
const physicalTypes=new Set(types.slice(1,10));
const metadata={vocabulary:{moves:names,types,abilities:['_none','hustle','vitalspirit','naturalcure','trace'],
 items:['_none','choiceband'],species:['_none','delibird','blissey','ditto','togetic','scizor','porygon2']},
 moves:[null,...names.slice(1).map(id=>{const m=dex.moves.get(id);return {id,source_type:m.type,
 type:types.indexOf(m.type.toLowerCase()),base_power:m.basePower,accuracy:m.accuracy,priority:m.priority,
 category:m.category==='Status'?'Status':physicalTypes.has(m.type.toLowerCase())?'Physical':'Special',flags:m.flags};})],
 type_chart:types.slice(1).map(a=>types.slice(1).map(d=>{const k=dex.types.get(d).damageTaken[a[0].toUpperCase()+a.slice(1)];return k===1?2:k===2?0.5:k===3?0:1;}))};
b.destroy();
process.stdout.write(JSON.stringify({metadata,counts,accuracy,damage,confusion}));
