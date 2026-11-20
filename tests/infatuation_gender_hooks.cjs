// Real pinned-Showdown constructor coupling; no repaired snapshot or state.
const assert=require('node:assert/strict'),path=require('node:path');
const root=process.argv[2];
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const {PRNG}=require(path.join(root,'dist/sim/prng.js'));
const {install}=require('../gpu_engine/chance_hooks.cjs');
let init=null,audit=[];
install(Battle,PRNG,()=>null,row=>audit.push(row),()=>init,path.join(root,'dist/sim/pokemon.js'));
const species=[['Alakazam','Latias','Latios','Metagross','Blissey','Tauros'],
 ['Snorlax','Delcatty','Clefable','Wigglytuff','Ditto','Magneton']];
const teams=species.map(team=>team.map(species=>({species,level:80,moves:['splash'],ability:'Pressure'})));
const expected=[[2,2,1,0,2,1],[1,2,1,2,0,0]];
function build(genders,sets=teams){
 init={genders,allow_inert_gender:false};audit=[];
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 try{b.setPlayer('p1',{name:'p1',team:sets[0]});b.setPlayer('p2',{name:'p2',team:sets[1]});}
 finally{init=null;}
 return b;
}
const b=build(expected);
assert.deepEqual(b.sides.map(s=>s.pokemon.map(p=>({M:1,F:2})[p.gender]||0)),expected);
assert.deepEqual(audit.filter(x=>x.kind==='COUPLED_INITIAL_GENDER').map(({side,slot,raw})=>[side,slot,raw]),
 [[0,0,1],[1,0,0],[1,1,1],[1,2,0],[1,3,1]]);
assert.equal(audit.filter(x=>x.kind==='INERT_GENDER'||x.kind==='INITIAL_RANDOM').length,0);
const advanced=new PRNG([1,2,3,4]);
for(let i=0;i<5;i++)advanced.random(2);
assert.equal(b.prng.getSeed(),advanced.getSeed(),'one battle-PRNG advance per mixed slot');
const override=structuredClone(teams),values=structuredClone(expected);
override[0][1].gender='M';values[0][1]=1;
assert.equal(build(values,override).sides[0].pokemon[1].gender,'M');
const wrong=structuredClone(expected);wrong[0][1]=1;
assert.throws(()=>build(wrong),/fixed construction gender differs/);
const missing=structuredClone(expected);missing[1][1]=0;
assert.throws(()=>build(missing),/mixed construction gender must be M\/F/);
assert.throws(()=>build([[1],[2]]),/invalid construction genders/);
console.log('PASS construction law, stream order, explicit override, and malformed controls');
