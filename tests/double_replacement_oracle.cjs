// Authored boundary fixtures, not scientific games or failed-state replays.
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));
const mon = species => ({species, name: species, level: 100, moves: ['Tackle']});
const rows = [];
for (const fainted of [['p1'], ['p2'], ['p1', 'p2']]) {
  const battle = new Battle({formatid: 'gen3randombattle', seed: [1, 2, 3, 4]});
  battle.setPlayer('p1', {name: 'One', team: [mon('Swampert'), mon('Starmie')]});
  battle.setPlayer('p2', {name: 'Two', team: [mon('Snorlax'), mon('Starmie')]});
  // Use the real faint path, not HP=0 without its lifecycle changes.
  for (const slot of fainted) battle[slot].active[0].faint();
  battle.faintMessages();
  battle.checkFainted();
  const requests = battle.getRequests('switch');
  rows.push({fainted, requests: Object.fromEntries(['p1', 'p2'].map((slot, i) =>
    [slot, requests[i]?.forceSwitch?.[0] === true]))});
  battle.destroy();
}
console.log(JSON.stringify({scientific_measurement: false, rows}));
