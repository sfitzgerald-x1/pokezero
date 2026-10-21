// Read the actual pinned Gen 3 catalog; no simulation or coverage claim.
const path = require('node:path');
const fs = require('node:fs');
const {Dex} = require(path.join(process.argv[2], 'dist/sim/dex.js'));
const dex = Dex.mod('gen3');
const names = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(names.map(name => {
  const m = dex.moves.get(name);
  if (!m.exists) throw new Error(`unknown move ${name}`);
  return {id:m.id, status:m.status || '', boosts:m.boosts || {},
    heal:m.heal || null, target:m.target};
})));
