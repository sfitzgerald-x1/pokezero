// Finite authored Gen3 queue boundaries; not sampled games or pool coverage.
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));

function fixture({frozen = false, thaw = true, item = '', reverse = false} = {}) {
  const b = new Battle({formatid: 'gen3customgame', seed: [1, 2, 3, 4]});
  for (let seat = 0; seat < 2; seat++) {
    b.setPlayer(`p${seat + 1}`, {name: `seat${seat}`, team: [
      {species: 'Mew', ability: 'Run Away', item: seat === Number(!reverse) ? item : '',
        moves: ['icebeam', 'splash'], level: 100},
      {species: 'Ditto', ability: 'Run Away', moves: ['splash'], level: 100},
    ]});
  }
  const attacker = b.sides[Number(reverse)].active[0];
  const target = b.sides[Number(!reverse)].active[0];
  for (const side of b.sides) for (const p of side.pokemon) {
    p.baseMaxhp = p.maxhp = p.hp = 1000;
    for (const key of Object.keys(p.storedStats)) p.storedStats[key] = 200;
    p.storedStats.spe = p === attacker ? 301 : 199;
    p.updateSpeed();
    for (const slot of p.moveSlots) slot.pp = 8;
  }
  let thawCalls = 0;
  let freezeBeforeMove = false;
  const runEvent = b.runEvent;
  b.runEvent = function (event, pokemon, ...args) {
    const previous = freezeBeforeMove;
    freezeBeforeMove = event === 'BeforeMove' && pokemon === target && pokemon.status === 'frz';
    try { return runEvent.call(this, event, pokemon, ...args); }
    finally { freezeBeforeMove = previous; }
  };
  b.randomChance = (n, d) => {
    if (freezeBeforeMove && n === 1 && d === 5) {thawCalls++; return thaw;}
    // Force only Ice Beam's secondary; accuracy succeeds and crits fail.
    return n >= d || (n === 10 && d === 100) || (n === 1 && d === 10);
  };
  // Gen3's secondary dispatcher uses random(100), rather than randomChance.
  b.random = () => 0;
  b.randomizer = x => x;
  if (frozen) target.setStatus('frz');
  const start = b.log.length;
  b.makeChoices(reverse ? 'move splash' : 'move icebeam',
    reverse ? 'move icebeam' : 'move splash');
  const result = {status: target.status, item: target.item,
    pp: target.moveSlots[1].pp, thawCalls,
    events: b.log.slice(start).filter(s => /^\|(move|cant|-status|-curestatus)\|/.test(s))};
  b.destroy();
  return result;
}
process.stdout.write(JSON.stringify({fresh: fixture(), fresh_stays: fixture({thaw: false}),
  fresh_other_seat: fixture({reverse: true}), fresh_stays_other_seat: fixture({thaw: false, reverse: true}),
  preexisting_thaw: fixture({frozen: true}), preexisting_stays: fixture({frozen: true, thaw: false}),
  fresh_lum: fixture({item: 'Lum Berry'})}));
