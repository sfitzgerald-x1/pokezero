// Offline table for a NEW explicitly initialized cohort, never old gender recovery.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

function compiledTree(root) {
  const files = [];
  function visit(dir) {
    for (const e of fs.readdirSync(dir, {withFileTypes:true})) {
      const p = path.join(dir,e.name);
      if (e.isDirectory()) visit(p);
      else if (e.isFile() && /\.(js|json)$/.test(e.name)) files.push(p);
    }
  }
  visit(root);
  const hash = crypto.createHash('sha256');
  for (const p of files.sort()) {
    hash.update(path.relative(root,p).split(path.sep).join('/')+'\0');
    hash.update(crypto.createHash('sha256').update(fs.readFileSync(p)).digest());
  }
  return {sha256:hash.digest('hex'),files:files.length};
}

function buildPolicy(dex, species) {
  if (!Array.isArray(species) || species[0] !== '_none' || new Set(species).size !== species.length)
    throw new Error('exact unique species vocabulary required');
  const code = g => g === undefined || g === '' ? 3 : ({N:0,M:1,F:2})[g];
  const unresolved = [];
  const codes = species.map((id,i) => {
    if (i===0) return -1;
    const s = dex.species.get(id);
    const base = s && dex.species.get(s.baseSpecies || s.name || id);
    // The generator's explicit species.gender and retained cosmetic/form
    // constructor must have the SAME law; otherwise leave that form unresolved.
    if (!s?.exists || s.gen>3 || !base?.exists || code(s.gender) === undefined ||
        code(base.gender) === undefined || code(s.gender)!==code(base.gender)) {
      unresolved.push(id); return -1;
    }
    return code(s.gender); // no use of genderRatio; Showdown's fallback is fair
  });
  return {schema:'pokezero.gen3-gender-policy.v1',species,codes,unresolved,
    law:'explicit M/F/N override, fixed species sex, otherwise fair M/F; no historical recovery'};
}

module.exports = {buildPolicy,compiledTree};
if (require.main===module) {
  const [root,metadataPath,expectedCommit] = process.argv.slice(2);
  if (!root || !metadataPath || !/^[a-f0-9]{40}$/.test(expectedCommit || ''))
    throw new Error('usage: gender_policy.cjs SHOWDOWN_ROOT POOL_METADATA_JSON EXPECTED_SHOWDOWN_COMMIT');
  const actual = execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  if (actual!==expectedCommit) throw new Error('pinned Showdown commit changed');
  execFileSync('git',['-C',root,'diff','--quiet','HEAD','--','sim/pokemon.ts','data/random-battles/gen3/teams.ts']);
  const metadata = JSON.parse(fs.readFileSync(metadataPath,'utf8'));
  if (!/^[a-f0-9]{64}$/.test(metadata.pool_sha256 || ''))
    throw new Error('immutable source pool hash required');
  const compiled = compiledTree(path.join(root,'dist'));
  if (compiled.sha256!==metadata.compiled_tree?.sha256 || compiled.files!==metadata.compiled_tree?.files)
    throw new Error('actual compiled Dex tree differs from immutable pool generator');
  const {Dex} = require(path.join(root,'dist/sim/dex.js'));
  const policy = buildPolicy(Dex.mod('gen3'),metadata.vocabulary.species);
  policy.provenance = {showdown_commit:actual,compiled_tree:compiled,node:process.version,
    pool_sha256:metadata.pool_sha256,
    source_files:Object.fromEntries(['sim/pokemon.ts','data/random-battles/gen3/teams.ts'].map(
      n=>[n,sha(path.join(root,n))])),
    generator_sources:{'gender_policy.cjs':sha(__filename),'generate_teams.cjs':sha(path.join(__dirname,'generate_teams.cjs'))}};
  process.stdout.write(JSON.stringify(policy)+'\n');
}
