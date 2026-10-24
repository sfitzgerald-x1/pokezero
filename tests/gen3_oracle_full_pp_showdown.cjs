// Authored Gen3 PP controls against the pinned Showdown checkout.
const assert = require('node:assert/strict');
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));
const rows = [];
for (const pp of [1, 8, 32]) for (const pressure of [false, true]) {
  for (const called of ['earthquake', 'recover']) {
    const battle = new Battle({formatid: 'gen3customgame', seed: [1, 2, 3, 4]});
    battle.setPlayer('p1', {name: 'caller', team: [{species: 'Snorlax',
      ability: 'Immunity', moves: ['sleeptalk', called], level: 100}]});
    battle.setPlayer('p2', {name: 'target', team: [{species: 'Dusclops',
      ability: pressure ? 'Pressure' : 'Limber', moves: ['splash'], level: 100}]});
    for (const side of battle.sides) {
      const mon = side.active[0];
      mon.baseMaxhp = mon.maxhp = 5000;
      mon.hp = 4000;
      mon.storedStats.spe = side.id === 'p1' ? 301 : 199;
      mon.updateSpeed();
    }
    const caller = battle.p1.active[0];
    caller.setStatus('slp');
    caller.statusState.time = 5;
    caller.moveSlots[0].pp = pp;
    caller.moveSlots[1].pp = 32;
    battle.randomChance = (n, d) => n >= d;
    battle.makeChoices('move sleeptalk', 'move splash');
    // Gen3's useMoveInner override excludes all sourceEffect calls from
    // Pressure charging, unlike the modern callerMoveForPressure branch.
    const expected = Math.max(0, pp - 1);
    assert.equal(caller.moveSlots[0].pp, expected);
    assert.equal(caller.moveSlots[1].pp, 32);
    rows.push({pp, pressure, called, after: caller.moveSlots.map(slot => slot.pp),
      moves: battle.log.filter(line => line.startsWith('|move|'))});
    battle.destroy();
  }
}
process.stdout.write(JSON.stringify({schema: 'pokezero.gen3-oracle-full-pp-controls.v1', rows}));
