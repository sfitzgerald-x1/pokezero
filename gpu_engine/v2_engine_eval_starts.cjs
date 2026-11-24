// Decode the registered packed starts. No simulator or team generator is run.
const fs = require('node:fs');
const path = require('node:path');
const {buildPolicy, compiledTree} = require('./gender_policy.cjs');
const {load, memberRow} = require('./team_rows.cjs');

const [root, expectedTree] = process.argv.slice(2);
if (compiledTree(path.join(root, 'dist')).sha256 !== expectedTree) {
  throw new Error('registered dex tree differs');
}
const {Teams} = require(path.join(root, 'dist/sim/teams.js'));
const {dex, id, vocabulary} = load(root);
const genderPolicy = buildPolicy(dex, vocabulary.species);
const packed = JSON.parse(fs.readFileSync(0, 'utf8'));
const teams = packed.map((value, index) => {
  const team = Teams.unpack(value);
  if (team?.length !== 6) throw new Error(`invalid registered team at index ${index}`);
  return team.map(mon => {
    const gender = ({M: 1, F: 2, N: 0})[mon.gender] ?? -1;
    const happiness = mon.happiness ?? 255;
    if (!Number.isInteger(happiness) || happiness < 0 || happiness > 255) {
      throw new Error(`invalid happiness at team index ${index}`);
    }
    return {...memberRow(dex, id, mon), gender, happiness};
  });
});
process.stdout.write(JSON.stringify({vocabulary, genderPolicy, teams}));
