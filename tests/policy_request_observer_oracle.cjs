// Authored source fixtures, never scientific games or historical run states.
const path = require('node:path');
// Freeze only this authored fixture's metadata clock, never production input.
Date.now = () => 1700000000000;
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));
const mon = (species, ability) => ({species, name: species, ability, level: 100,
  moves: ['Tackle'], nature: 'Serious'});
const traced = process.argv.includes('--traced');
const cases = traced ? ['Gardevoir', 'Porygon2'].flatMap(species => [
  [species, 'Trace', 'Dugtrio', 'Arena Trap'],
  [species, 'Trace', 'Magneton', 'Magnet Pull'],
]) : [
  ['Dugtrio', 'Arena Trap', 'Pikachu', 'Static'],
  ['Dugtrio', 'Sand Veil', 'Pikachu', 'Static'],
  ['Dugtrio', 'Arena Trap', 'Charizard', 'Blaze'],
  ['Dugtrio', 'Sand Veil', 'Charizard', 'Blaze'],
  ['Dugtrio', 'Arena Trap', 'Gengar', 'Levitate'],
  ['Dugtrio', 'Sand Veil', 'Gengar', 'Levitate'],
  ['Magneton', 'Magnet Pull', 'Metagross', 'Clear Body'],
  ['Magneton', 'Sturdy', 'Metagross', 'Clear Body'],
  ['Magneton', 'Magnet Pull', 'Pikachu', 'Static'],
  ['Magneton', 'Sturdy', 'Pikachu', 'Static'],
];
const rows = [];
for (const [foe, ability, own, ownAbility] of cases) for (const slot of ['p1', 'p2']) {
  const b = new Battle({formatid: 'gen3randombattle', seed: [1, 2, 3, 4]});
  const other = slot === 'p1' ? 'p2' : 'p1';
  b.setPlayer(slot, {name: 'Acting', team: [mon(own, ownAbility), mon('Snorlax', 'Immunity')]});
  b.setPlayer(other, {name: 'Holder', team: [mon(foe, ability), mon('Snorlax', 'Immunity')]});
  rows.push({foe, ability: traced ? ownAbility : ability,
    baseAbility: ability, copiedAbility: b[other].active[0].ability,
    own, ownAbility, slot, lines: b.log.slice(),
    request: structuredClone(b[slot].activeRequest), actual_trapped: b[slot].active[0].trapped});
  b.destroy();
}
console.log(JSON.stringify({schema: 'pokezero.authored_private_request_observer.v1',
  scientific_measurement: false, rows}));
