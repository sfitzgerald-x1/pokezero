'use strict';
// Native wire serialization of already sealed literal sets, never generation.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const stats=['hp','atk','def','spa','spd','spe'];
const fields=['species','level','ability','item','nature','moves','ivs','evs','gender','happiness'].sort();
function literalTeams(value){
  if(!Array.isArray(value)||value.length!==2||value.some(team=>!Array.isArray(team)||team.length!==6))throw Error('LITERAL_TWO_SIX_MEMBER_TEAMS_REQUIRED');
  for(const team of value)for(const set of team){
    if(!set||typeof set!=='object'||Array.isArray(set)||Object.keys(set).sort().join(',')!==fields.join(',')||
      ['species','ability','nature'].some(key=>typeof set[key]!=='string'||!set[key])||typeof set.item!=='string'||
      !Number.isInteger(set.level)||set.level<1||set.level>100||!Number.isInteger(set.happiness)||set.happiness<0||set.happiness>255||
      !['N','M','F'].includes(set.gender)||!Array.isArray(set.moves)||set.moves.length<1||set.moves.length>4||set.moves.some(x=>typeof x!=='string'||!x))throw Error('COMPLETE_LITERAL_SET_REQUIRED');
    for(const [key,max]of [['ivs',31],['evs',255]]){
      const values=set[key];
      if(!values||typeof values!=='object'||Array.isArray(values)||Object.keys(values).sort().join(',')!==[...stats].sort().join(',')||
        stats.some(stat=>!Number.isInteger(values[stat])||values[stat]<0||values[stat]>max))throw Error('LITERAL_IV_EV_SET_REQUIRED');
    }
    if(stats.reduce((n,key)=>n+set.evs[key],0)>510)throw Error('LITERAL_EV_BUDGET_REQUIRED');
  }
  return value;
}
function packTeams(root,value){
  literalTeams(value);const before=JSON.stringify(value);
  const file=path.join(root,'dist/sim/teams.js');const {Teams}=require(file);
  const packed=value.map(team=>Teams.pack(team));
  if(JSON.stringify(value)!==before)throw Error('NATIVE_PACK_MUTATED_LITERAL_SETS');
  for(const team of packed){
    if(typeof team!=='string'||team.split(']').length!==6||team.split(']').some(set=>!set.includes('|'))||Teams.unpack(team)?.length!==6)
      throw Error('EXACT_SIX_MEMBER_PACKED_WIRE_REQUIRED');
  }
  return {packed,evidence:{schema:'pokezero.fresh200.native_team_wire.v1',node:process.version,arch:process.arch,
    native_serializer_file:file,native_serializer_sha256:sha(fs.readFileSync(file)),literal_sets_sha256:sha(before),
    packed:packed.map(value=>({bytes:Buffer.byteLength(value),sha256:sha(value)})),
    literal_sets_unchanged:true,no_team_generation:true,no_stat_calculation:true,no_RNG:true,qualification:false}};
}
if(require.main===module){
  const input=fs.readFileSync(0);const result=packTeams(process.argv[2],JSON.parse(input));
  result.evidence.input_wire_sha256=sha(input);process.stdout.write(JSON.stringify(result)+'\n');
}
module.exports={literalTeams,packTeams};
