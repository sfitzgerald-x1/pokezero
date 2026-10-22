// Authored Baton Pass queue boundary, never a stopped-panel replay.
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));
const rows = [];
for (const waitingSeat of ['p1', 'p2']) {
 for (const scenario of ['fast', 'slow', 'confusion']) {
  const fastPass = scenario === 'fast';
  const passingSeat = waitingSeat === 'p1' ? 'p2' : 'p1';
  const b = new Battle({formatid: 'gen3customgame', seed: [1, 2, 3, 4]});
  b.reportPercentages = true;
  b.reportExactHP = false;
  for (const seat of ['p1', 'p2']) {
    b.setPlayer(seat, {name: seat, team: [
      {species: seat === waitingSeat ? 'Charmander' : 'Mew', ability: 'None',
        moves: seat === waitingSeat ? ['Ember', 'Growl'] : ['Baton Pass', 'Splash'], level: 100},
      {species: 'Ditto', ability: 'None', moves: ['Splash'], level: 100},
    ]});
  }
  for (const side of b.sides) {
    for (const mon of side.pokemon) mon.baseMaxhp = mon.maxhp = mon.hp = 1000;
    side.active[0].storedStats.spe = (side.id === passingSeat) === fastPass ? 300 : 100;
    side.active[0].updateSpeed();
  }
  b.randomChance = (n, d) => n >= d;
  // Fixed damage isolates queue order/identity; this is not damage parity.
  const damage = [];
  b.actions.getDamage = (source, target, move) => {
    if (typeof move === 'number') return 20; // confusion self-hit, not an attack choice
    if (move.basePower > 0) damage.push({move: move.id, source: source.side.id,
      target: target.species.id, targetSeat: target.side.id});
    return move.basePower > 0 ? 50 : undefined;
  };
  const actor = b[waitingSeat].active[0];
  actor.moveSlots[0].pp = 9;
  if (scenario === 'confusion') {
    actor.addVolatile('confusion');
    actor.volatiles.confusion.time = 3;
  }
  const start = b.log.length;
  b.makeChoices('move 1', 'move 1');
  const before = {request: b[waitingSeat].activeRequest, pp: actor.moveSlots[0].pp, hp: actor.hp,
    damage: damage.slice(), events: b.log.slice(start), public_lines: b.log.slice()};
  b.makeChoices(...(waitingSeat === 'p1' ? [undefined, 'switch 2'] : ['switch 2', undefined]));
  rows.push({waitingSeat, passingSeat, fastPass, scenario, before,
    after: {pp: actor.moveSlots[0].pp, damage, events: b.log.slice(start),
      recipient: b[passingSeat].active[0].species.id,
      recipientHP: b[passingSeat].active[0].hp}});
  b.destroy();
 }
}
console.log(JSON.stringify({scientific_measurement: false, historical_replay: false, rows}));
