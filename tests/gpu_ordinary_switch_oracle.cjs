// Authored Gen3 randbats action vocabulary; never reads collected roots.
const path=require('node:path');
const {Dex}=require(path.join(process.argv[2],'dist/sim/dex.js'));
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
const dex=Dex.mod('gen3');
const moves=['_none','splash','earthquake','thunderbolt','bodyslam','explosion','batonpass','protect','endure','willowisp','struggle','fireblast','rockslide'];
const types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost','steel','fire','water','grass','electric','psychic','ice','dragon','dark'];
const physical=new Set(types.slice(1,10));
const metadata={vocabulary:{moves,types,
 abilities:['_none','cloudnine','flashfire','magmaarmor','levitate','static','pressure'],
 items:['_none','leftovers','choiceband','lightball'],
 species:['_none','golduck','flareon','camerupt','flygon','ditto','weezing']},
 moves:[null,...moves.slice(1).map(id=>{const m=dex.moves.get(id);return {
  id,source_type:m.type,type:types.indexOf(m.type.toLowerCase()),base_power:m.basePower,
  accuracy:m.accuracy,priority:m.priority,flags:m.flags,
  category:m.category==='Status'?'Status':physical.has(m.type.toLowerCase())?'Physical':'Special'};})],
 type_chart:types.slice(1).map(a=>types.slice(1).map(d=>{const k=dex.types.get(d).damageTaken[a[0].toUpperCase()+a.slice(1)];return k===1?2:k===2?0.5:k===3?0:1;}))};
const flash_fire=[];
for(const seat of [0,1])for(const hit of [false,true])for(const protect of [false,true]){
 const b=new Battle({formatid:'gen3customgame',seed:[11,22,33,44]});
 for(let s=0;s<2;s++)b.setPlayer('p'+(s+1),{name:'seat'+s,team:[{
  species:s===seat?'Porygon2':'Gengar',ability:s===seat?'Trace':'Levitate',
  moves:s===seat?['protect','splash']:['willowisp','splash'],level:77}]});
 const target=b.sides[seat].active[0],source=b.sides[1-seat].active[0];
 target.setAbility('flashfire');if(protect)target.addVolatile('protect');
 let threshold=null;b.randomChance=(n,d)=>{if(d===100)threshold=n;return hit;};
 b.actions.useMove('willowisp',source,target);
 flash_fire.push({seat,hit,protect,threshold,activated:!!target.volatiles.flashfire});b.destroy();
}
const terminal_switch=[];
for(const seat of [0,1]){
 const b=new Battle({formatid:'gen3customgame',seed:[21,31,41,51]});
 for(let s=0;s<2;s++)b.setPlayer('p'+(s+1),{name:'seat'+s,team:s===seat?[
  {species:'Golduck',ability:'Cloud Nine',moves:['surf'],level:77},
  {species:'Flareon',ability:'Flash Fire',moves:['fireblast'],level:77}]:[
  {species:'Camerupt',ability:'Magma Armor',item:'Choice Band',moves:['explosion'],level:77}]});
 b.sides[seat].pokemon[1].hp=1;
 const choices=seat===0?['switch 2','move explosion']:['move explosion','switch 2'];
 b.makeChoices(...choices);
 terminal_switch.push({seat,ended:b.ended,winner:b.winner,request:b.requestState,
  incoming_hp:b.sides[seat].active[0].hp,remaining:b.sides[seat].pokemonLeft});b.destroy();
}
process.stdout.write(JSON.stringify({pinned_source:'f76228a1354b5d0f307ca2d16101294ad3a2308b',metadata,flash_fire,terminal_switch}));
