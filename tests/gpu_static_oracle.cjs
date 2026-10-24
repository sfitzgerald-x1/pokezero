// Actual pinned Gen3 ability events on authored battles, not a game panel.
const path = require('node:path');
const {Battle} = require(path.join(process.argv[2], 'dist/sim/battle.js'));
const {Dex} = require(path.join(process.argv[2], 'dist/sim/dex.js'));
const dex = Dex.mod('gen3');
const contact = Object.fromEntries(['dragonclaw', 'brickbreak', 'pursuit', 'surf', 'roar', 'whirlwind']
  .map(id => [id, !!dex.moves.get(id).flags.contact]));
let staticChance;
function sample(options = {}) {
  const side = options.side || 0;
  const b = new Battle({formatid: 'gen3customgame', seed: [1, 2, 3, 4]});
  for (let seat = 0; seat < 2; seat++) b.setPlayer(`p${seat+1}`, {name: `seat${seat}`, team: [
    {species: 'Mew', ability: seat === side ? (options.ability || 'Run Away') : 'Static',
      moves: [options.move || 'dragonclaw', 'splash'], level: 100,
      item: seat === side ? (options.lum ? 'Lum Berry' : '') : ''},
    {species: 'Ditto', ability: 'Run Away', moves: ['splash'], level: 100}]});
  const p = b.sides[side].active[0], q = b.sides[1-side].active[0];
  p.baseMaxhp = p.maxhp = p.hp = q.baseMaxhp = q.maxhp = q.hp = 1000;
  p.types = [options.type || 'Water'];
  if (options.recoilKO) p.hp = 1;
  if (options.status) p.setStatus(options.status);
  if (options.ko) q.hp = 1;
  if (options.sub) {q.addVolatile('substitute'); q.volatiles.substitute.hp = 1; q.hp = 1000;}
  if (options.protect) q.addVolatile('protect');
  if (options.immune) q.types = ['Ghost'];
  b.randomChance = (n, d) => {
    if (n === 1 && d === 3) {staticChance = [n, d]; return (options.roll || 0) === 0;}
    return false;
  };
  b.randomizer = x => x;
  if (options.switching) {
    const choices = side === 0 ? ['move 1', 'switch 2'] : ['switch 2', 'move 1'];
    b.makeChoices(...choices);
  } else {
    const move = b.dex.getActiveMove(options.move || 'dragonclaw');
    move.accuracy = options.miss ? 0 : true; move.willCrit = false;
    b.actions.useMove(move, p, q);
  }
  const out = `${p.status}:${p.item}`;
  b.destroy(); return out;
}
const cases = {}, expected = {};
const inputs = {
  ordinary: [{}, 'par:'], opposite: [{side: 1}, 'par:'],
  fail1: [{roll: 1}, ':'], fail2: [{roll: 2}, ':'],
  noncontact: [{move: 'surf'}, ':'], sub: [{sub: true}, ':'],
  protect: [{protect: true}, ':'], miss: [{miss: true}, ':'],
  immune: [{move: 'brickbreak', immune: true}, ':'],
  limber: [{ability: 'Limber'}, ':'], burn: [{status: 'brn'}, 'brn:'],
  dust: [{ability: 'Shield Dust'}, 'par:'], grace: [{ability: 'Serene Grace'}, 'par:'],
  electric: [{type: 'Electric'}, 'par:'], knockout: [{ko: true}, 'par:'],
  lum: [{lum: true}, ':'], pursuit: [{move: 'pursuit', switching: true}, 'par:'],
  pursuit2: [{side: 1, move: 'pursuit', switching: true}, 'par:'],
  pursuit_lum: [{move: 'pursuit', switching: true, lum: true}, ':'],
  recoil_lum: [{move: 'doubleedge', recoilKO: true, lum: true}, ':'],
  recoil_status: [{move: 'doubleedge', recoilKO: true}, 'par:'],
  struggle_lum: [{move: 'struggle', recoilKO: true, lum: true}, ':'],
};
for (const [name, [options, result]] of Object.entries(inputs)) {
  cases[name] = sample(options); expected[name] = result;
}
process.stdout.write(JSON.stringify({contact, static_chance: staticChance, cases, expected}));
