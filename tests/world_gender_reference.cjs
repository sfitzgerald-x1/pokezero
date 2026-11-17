// Actual randomTeam -> Pokemon constructor at the pinned Gen 3 source.
const path = require('node:path');
const root = process.argv[2];
const {Teams, Battle, Dex} = require(path.join(root, 'dist/sim'));
const rows = {};
for (let i = 0; i < 2048; i++) {
  const seed = [19, 731, i, 821];
  const team = Teams.generate('gen3randombattle', {seed});
  const battle = new Battle({formatid: 'gen3customgame', seed: [71, i, 91, 37]});
  battle.setPlayer('p1', {name: 'p1', team});
  for (const mon of battle.sides[0].pokemon) {
    const species = Dex.mod('gen3').species.get(mon.species.name);
    const key = species.id;
    rows[key] ||= {metadata: {gender: species.gender, genderRatio: species.genderRatio}, M: 0, F: 0, N: 0};
    rows[key][mon.gender || 'N']++;
  }
  battle.destroy();
}
process.stdout.write(JSON.stringify(rows) + '\n');
