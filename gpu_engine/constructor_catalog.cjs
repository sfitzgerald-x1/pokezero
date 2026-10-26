// Offline constructor weights for a new cohort; no historical recovery or battle.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {compiledTree} = require('./gender_policy.cjs');
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

function buildCatalog(dex, species) {
  if (!Array.isArray(species) || species[0] !== '_none' || new Set(species).size !== species.length)
    throw new Error('exact unique source species vocabulary required');
  const unresolved = [];
  const weights = species.map((id, i) => {
    if (i === 0) return 0;
    const s = dex.species.get(id);
    if (!s?.exists || s.gen > 3 || !Number.isFinite(s.weightkg) || s.weightkg <= 0) {
      unresolved.push(id); return null;
    }
    return s.weightkg;
  });
  return {schema:'pokezero.gen3-constructor-catalog.v1',species,weights,unresolved};
}
module.exports = {buildCatalog};

if (require.main === module) {
  const [root, metadataPath, expectedCommit] = process.argv.slice(2);
  if (!root || !metadataPath || !/^[a-f0-9]{40}$/.test(expectedCommit || ''))
    throw new Error('usage: constructor_catalog.cjs SHOWDOWN_ROOT POOL_METADATA_JSON EXPECTED_COMMIT');
  const commit = execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  if (commit !== expectedCommit) throw new Error('pinned Showdown commit changed');
  const metadata = JSON.parse(fs.readFileSync(metadataPath,'utf8'));
  if (!/^[a-f0-9]{64}$/.test(metadata.pool_sha256 || '')) throw new Error('verified immutable pool hash required');
  const compiled = compiledTree(path.join(root,'dist'));
  if (compiled.sha256 !== metadata.compiled_tree?.sha256 || compiled.files !== metadata.compiled_tree?.files)
    throw new Error('compiled source tree differs from pool generator');
  const {Dex} = require(path.join(root,'dist/sim/dex.js'));
  const catalog = buildCatalog(Dex.mod('gen3'),metadata.vocabulary.species);
  catalog.provenance = {showdown_commit:commit,compiled_tree:compiled,pool_sha256:metadata.pool_sha256,
    generator_sources:{'constructor_catalog.cjs':sha(__filename),
      'gender_policy.cjs':sha(path.join(__dirname,'gender_policy.cjs'))},node:process.version};
  process.stdout.write(JSON.stringify(catalog)+'\n');
}
