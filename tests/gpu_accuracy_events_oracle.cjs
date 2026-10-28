// Authored pinned Gen3 accuracy events; no saved roots or scientific collection.
const fs=require('node:fs'),path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const dex=Dex.mod('gen3');
const nativeIds=process.argv[3]?JSON.parse(fs.readFileSync(process.argv[3],'utf8')):
 JSON.parse(fs.readFileSync(path.join(__dirname,'data/gpu_accuracy_events_catalog.json'),'utf8')).native_choice_ids;
const always=nativeIds.filter(id=>{const m=dex.moves.get(id.toLowerCase());return m.exists&&m.gen<=3&&m.accuracy===true;}).sort();
const combinations=[['Run Away','Run Away',false],['Hustle','Run Away',false],
 ['Hustle','Sand Veil',true],['Compound Eyes','Run Away',false],['Compound Eyes','Sand Veil',true],
 ['Run Away','Sand Veil',true]];
const stages=[1,4/3,5/3,2,7/3,8/3,3];
const counts=[];
for(const [sourceAbility,targetAbility,sand]of combinations){
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let s=0;s<2;s++)b.setPlayer('p'+(s+1),{name:'seat'+s,team:[{
  species:s?'Sandslash':'Delibird',ability:s?targetAbility:sourceAbility,
  moves:['hiddenpowerground','toxic','aerialace','splash'],level:73}]});
 if(sand)b.field.setWeather('sandstorm',b.p1.active[0],b.dex.moves.get('sandstorm'));
 const p=b.p1.active[0],q=b.p2.active[0],move=b.dex.getActiveMove('hiddenpowerground');move.type='Ground';
 for(let base=0;base<=100;base++)for(let a=-6;a<=6;a++)for(let e=-6;e<=6;e++){
  let value=a>0?base*stages[a]:base/stages[-a];
  value=e>0?value/stages[e]:value*stages[-e];
  const actual=b.runEvent('ModifyAccuracy',q,p,move,value);
  counts.push(Math.max(0,Math.min(100,Math.ceil(actual))));
 }
 b.destroy();
}
let checksum=14695981039346656037n;
for(const count of counts)checksum=BigInt.asUintN(64,(checksum^BigInt(count))*1099511628211n);
const metadata=JSON.parse(fs.readFileSync(path.join(__dirname,'data/gpu_hustle_catalog.json'),'utf8')).metadata;
metadata.vocabulary.abilities.push('compoundeyes','sandveil','airlock','cloudnine');
metadata.vocabulary.moves.push('thunder','sleeppowder','haze','doubleteam','raindance','sandstorm','pursuit','hydropump');
metadata.vocabulary.species.push('butterfree','sandslash');
const physical=new Set(metadata.vocabulary.types.slice(1,10));
metadata.moves=[null,...metadata.vocabulary.moves.slice(1).map(id=>{const m=dex.moves.get(id);return {
 id,source_type:m.type,type:metadata.vocabulary.types.indexOf(m.type.toLowerCase()),
 base_power:m.basePower,accuracy:m.accuracy,priority:m.priority,flags:m.flags,
 category:m.category==='Status'?'Status':physical.has(m.type.toLowerCase())?'Physical':'Special'};})];
const pursuit=[];
for(const switching of [true,false]){
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 for(let seat=0;seat<2;seat++)b.setPlayer('p'+(seat+1),{name:'seat'+seat,team:[{
  species:seat?'Sandslash':'Delibird',ability:seat?'Sand Veil':'Vital Spirit',moves:['pursuit','splash'],level:73}]});
 const p=b.p1.active[0],q=b.p2.active[0];p.boosts.accuracy=-6;q.boosts.evasion=6;q.beingCalledBack=switching;
 b.field.setWeather('sandstorm',p,b.dex.moves.get('sandstorm'));
 let threshold=null;b.randomChance=(n,d)=>{if(d===100)threshold=n;return false;};
 b.actions.useMove('pursuit',p,q);pursuit.push({switching,threshold});b.destroy();
}
process.stdout.write(JSON.stringify({native_choice_ids:nativeIds,always_hit_native_ids:always,counts,fnv64:checksum.toString(),pursuit,
 combinations,metadata,pinned_source:'f76228a1354b5d0f307ca2d16101294ad3a2308b'}));
