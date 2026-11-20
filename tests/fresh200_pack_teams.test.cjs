'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {literalTeams,packTeams}=require('../gpu_engine/fresh200_pack_teams.cjs');
function fixture(){
  const one=()=>({species:'ditto',level:80,ability:'limber',item:'',nature:'Serious',moves:['transform'],
    ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},evs:{hp:85,atk:85,def:85,spa:85,spd:85,spe:85},gender:'N',happiness:255});
  return [Array.from({length:6},one),Array.from({length:6},one)];
}
test('literal set admission preserves exact source values without a Native import',()=>{
  const value=fixture(),before=JSON.stringify(value);assert.equal(literalTeams(value),value);assert.equal(JSON.stringify(value),before);
});
test('malformed literal input fails before any Native serializer or generator can load',()=>{
  const cases=[()=>null,()=>['random','random'],()=>[[],[]],()=>{const t=fixture();delete t[0][0].ivs;return t;},
    ()=>{const t=fixture();t[0][0].evs.hp=-1;return t;},()=>{const t=fixture();t[0][0].ivs.atk=32;return t;},
    ()=>{const t=fixture();t[0][0].gender='';return t;},()=>{const t=fixture();t[0][0].level=true;return t;}];
  for(const make of cases)assert.throws(()=>packTeams('/unavailable-native-source',make()),/LITERAL/);
});
