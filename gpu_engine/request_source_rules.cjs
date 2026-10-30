// Static source policy only: no Battle, Teams, generated roster or game.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {compiledTree} = require('./gender_policy.cjs');
const PIN = 'f76228a1354b5d0f307ca2d16101294ad3a2308b';
const FILES = ['sim/pokemon.ts','sim/battle.ts','sim/side.ts','sim/dex.ts',
  'data/abilities.ts','data/conditions.ts','data/moves.ts',
  'data/mods/gen3/abilities.ts','data/mods/gen3/moves.ts','data/mods/gen5/typechart.ts',
  'sim/SIM-PROTOCOL.md'];
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

function build(root, metadata) {
  if (execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim() !== PIN ||
      metadata.source_commit_label !== PIN) throw new Error('pinned source revision required');
  execFileSync('git',['-C',root,'diff','--quiet','HEAD','--','sim','data','config']);
  const tree = compiledTree(path.join(root,'dist'));
  if (tree.sha256 !== metadata.compiled_tree?.sha256 || tree.files !== metadata.compiled_tree?.files)
    throw new Error('constructor/source compiled Dex tree differs');
  const source_files=Object.fromEntries(FILES.map(f=>[f,sha(path.join(root,f))]));
  const descriptor={revision:PIN,git_tree:execFileSync('git',['-C',root,'rev-parse',PIN+'^{tree}'],
    {encoding:'utf8'}).trim(),source_files,compiled_tree:tree};
  const source_tree={revision:PIN,sha256:crypto.createHash('sha256')
    .update(JSON.stringify(descriptor)).digest('hex'),descriptor};
  const {Dex,toID} = require(path.join(root,'dist/sim/dex.js'));
  const dex = Dex.mod('gen3');
  const format = dex.formats.get('gen3randombattle');
  const rules = dex.formats.getRuleTable(format);
  if (dex.gen !== 3 || format.gameType !== 'singles' || format.team !== 'random')
    throw new Error('exact Gen3 random singles source required');
  const species = metadata.vocabulary.species.map((id,i) => {
    if (!i) return {supported:false,abilities:[],types:[]};
    const s = dex.species.get(id);
    if (!s.exists || s.gen > 3 || toID(s.id) !== id)
      return {supported:false,abilities:[],types:[]};
    // battle.ts nextTurn's alternate-ability iteration, including its rules.
    const abilities = Object.entries(s.abilities).filter(([slot,name]) =>
      !(slot === 'H' && s.unreleasedHidden) && !rules.has('-ability:'+toID(name)))
      .map(([,name]) => toID(name));
    return {supported:true,abilities,types:s.types.map(toID)};
  });
  return {schema:'pokezero.gen3-request-source-rules.v1',revision:PIN,
    format:'gen3randombattle',compiled_tree:tree,source_tree,
    source_files,
    generator_sha256:sha(__filename),vocabulary:metadata.vocabulary,species,
    abilities:metadata.vocabulary.abilities.map((id,i)=>({supported:!i ||
      (dex.abilities.get(id).exists && dex.abilities.get(id).gen <= 3),
      arena:id==='arenatrap',magnet:id==='magnetpull',shadow:id==='shadowtag',levitate:id==='levitate'})),
    types:metadata.vocabulary.types.map((id,i)=>({immune:!!i &&
      dex.types.get(id).damageTaken.trapped===3,flying:id==='flying',steel:id==='steel'})),
    moves:metadata.vocabulary.moves.map((id,i)=>{
      const move=dex.moves.get(id);
      const condition=move.volatileStatus?dex.conditions.get(move.volatileStatus):move.condition;
      const altersRequest=Object.keys(condition||{}).some(key=>
        /(?:DisableMove|TrapPokemon|MaybeTrapPokemon|LockMove|SemiLockMove)$/.test(key)&&
        typeof condition[key]!=='undefined');
      // Encore's actual engine counters are represented and explicitly refused
      // while active. Other missing request-changing conditions must stay unknown.
      const missingCondition=altersRequest&&id!=='encore';
      const missingOrigin=['transform','mimic','sketch','conversion','conversion2',
        'sleeptalk','metronome','assist','mirrormove','naturepower','copycat'].includes(id);
      return {unsupported_history:!!i&&(missingCondition||missingOrigin||!!move.flags.recharge)};
    }),
    // Pokemon.knownType starts true. Transform is a separate unsupported origin.
    constructor_known_type:true,direct_simulator_request_id:-1};
}
module.exports={build,PIN};
if (require.main===module) {
  const root=process.argv[2];
  const metadata=JSON.parse(fs.readFileSync(0,'utf8'));
  process.stdout.write(JSON.stringify(build(root,metadata))+'\n');
}
