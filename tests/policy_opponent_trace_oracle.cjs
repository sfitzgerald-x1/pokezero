// Authored replacement boundary, not a strength game or historical replay.
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));
const rows = [];
const mon = (species, ability, moves) => ({species, name: species, ability,
  level: 100, moves, nature: 'Serious'});
const cases = process.argv.includes('--maybe-trap') ? [
  ['Dugtrio', 'Arena Trap', ['Tackle']],
  ['Magneton', 'Magnet Pull', ['Tackle']],
] : [['Wobbuffet', 'Shadow Tag', ['Safeguard', 'Counter']]];
for (const [holderSpecies, holderAbility, holderMoves] of cases) for (const seat of ['p1', 'p2']) {
  const other = seat === 'p1' ? 'p2' : 'p1';
  const b = new Battle({formatid: 'gen3randombattle', seed: [1, 2, 3, 4]});
  b.setPlayer(seat, {name: 'Holder', team: [
    mon(holderSpecies, holderAbility, holderMoves),
    mon('Blissey', 'Natural Cure', ['Soft-Boiled']),
  ]});
  b.setPlayer(other, {name: 'Tracer', team: [
    mon('Snorlax', 'Immunity', ['Rest']),
    mon('Gardevoir', 'Trace', ['Psychic', 'Calm Mind']),
    mon('Blissey', 'Natural Cure', ['Soft-Boiled']),
  ]});
  // Exercise the real faint/request lifecycle before the replacement.
  b[other].active[0].faint();
  b.faintMessages();
  b.checkFainted();
  b.makeRequest('switch');
  const before = b.log.slice();
  b.makeChoices(...(seat === 'p1' ? ['pass', 'switch 2'] : ['switch 2', 'pass']));
  if (b[other].active[0].species.id !== 'gardevoir') throw new Error('replacement rejected');
  rows.push({seat, holderSpecies, holderAbility, holderMoves, before, after: b.log.slice(),
    copied_ability: b[other].active[0].ability,
    request: b[seat].activeRequest.active[0]});
  b.destroy();
}
console.log(JSON.stringify({scientific_measurement: false, historical_cause_established: false, rows}));
