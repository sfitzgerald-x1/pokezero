// Finite authored Gen3 component executions, not a real-team fidelity panel.
const fs = require('node:fs');
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));
const {Dex} = require(path.join(process.argv[2], 'dist/sim/dex.js'));
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const dex = Dex.mod('gen3');
const stats = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
function run(spec) {
  const b = new Battle({formatid: 'gen3customgame', seed: [11, 22, 33, 44]});
  const actor = spec.actor || 0, target = 1 - actor, draws = [];
  for (let side = 0; side < 2; side++) b.setPlayer(`p${side + 1}`, {name: `seat${side}`, team: [
    {species: 'Mew', ability: side === target ? spec.ability || 'Run Away' : 'Run Away',
      moves: side === actor ? [spec.move, 'splash'] : [spec.foe || 'splash', spec.foe ? 'splash' : 'growl'], level: 100},
    {species: 'Ditto', ability: 'Run Away', moves: ['splash'], level: 100},
  ]});
  const mons = b.sides.map(s => s.pokemon.slice()); // immutable identity across switches
  for (let side = 0; side < 2; side++) for (const p of mons[side]) {
    p.baseMaxhp = p.maxhp = p.hp = spec.maxhp || 101;
    p.types = ['Normal'];
    for (const stat of Object.keys(p.storedStats)) p.storedStats[stat] = 200;
    p.storedStats.spe = side === actor ? 1000 : 100;
    for (const slot of p.moveSlots) slot.pp = slot.maxpp = 20;
  }
  const p = mons[actor][0], q = mons[target][0];
  if (spec.type) q.types = [spec.type];
  if (spec.hp !== undefined) p.hp = spec.hp;
  if (spec.status) p.setStatus(spec.status);
  if (spec.targetStatus) q.setStatus(spec.targetStatus);
  if (spec.toxicStage !== undefined) q.statusState.stage = spec.toxicStage;
  if (spec.substitute) {q.addVolatile('substitute'); q.volatiles.substitute.hp = 25;}
  if (spec.item) p.item = p.baseItem = spec.item;
  if (spec.targetItem) q.item = q.baseItem = spec.targetItem;
  if (spec.stage !== undefined) for (const stat of stats) p.boosts[stat] = spec.stage;
  if (spec.accuracyStage !== undefined) p.boosts.accuracy = spec.accuracyStage;
  if (spec.evasionStage !== undefined) q.boosts.evasion = spec.evasionStage;
  for (const side of b.sides) for (const mon of side.pokemon) mon.updateSpeed();
  const originalChance = b.randomChance.bind(b);
  b.random = (from, to) => {
    // Only primary sleep duration is overridden. PRNG draws for the global
    // Gen3 Quick Claw clock and queue ties remain seeded and are irrelevant:
    // neither holder has Quick Claw and the active speeds are distinct.
    if (from !== 2 || to !== 6) throw new Error(`Unexpected random(${from},${to})`);
    const result = spec.duration || 2;
    draws.push({event: 'sleep-duration', args: [from, to], result}); return result;
  };
  b.randomChance = (n, d) => {
    let event, result;
    if (d === 100) {event = 'accuracy'; result = (spec.roll || 0) < n;}
    else if (n === 1 && d === 4) {event = 'paralysis'; result = false;}
    else if (n === 1 && d === 5) {event = 'global-quick-claw-no-holder'; result = originalChance(n,d);}
    else if (n === 1 && d === 1 && spec.foe === 'protect') {event = 'protect'; result = true;}
    else throw new Error(`Unexpected randomChance(${n},${d})`);
    draws.push({event, args: [n, d], result}); return result;
  };
  const snapshot = () => ({
    active: b.sides.map((s, side) => mons[side].indexOf(s.active[0])),
    mons: mons.map(side => side.map(mon => ({hp: mon.hp, status: mon.status,
      counter: mon.status === 'slp' ? mon.statusState.time : mon.status === 'tox' ? mon.statusState.stage : 0,
      boosts: stats.map(stat => mon.boosts[stat]), item: mon.item,
      pp: mon.moveSlots.map(slot => slot.pp)}))),
    spikes: b.sides.map(s => s.sideConditions.spikes?.layers || 0),
  });
  const rows = [];
  for (const choices of spec.choices) {
    const start = b.log.length;
    b.makeChoices(...choices);
    rows.push({...snapshot(), events: b.log.slice(start).filter(line => /^\|(move|cant|-status|-curestatus|-heal|-damage|-boost|-fail|-sidestart|switch)\|/.test(line))});
  }
  b.destroy(); return {id: spec.id, rows, draws};
}
const catalog = input.names.map(id => {
  const m = dex.moves.get(id);
  return {id: m.id, type: m.type.toLowerCase(), accuracy: m.accuracy, priority: m.priority,
    flags: m.flags, status: m.status || '', boosts: m.boosts || {}, heal: m.heal || null};
});
process.stdout.write(JSON.stringify({catalog, cases: input.cases.map(run)}));
