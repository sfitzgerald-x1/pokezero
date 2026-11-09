// Pool vocabulary and per-member row builder, shared by gpu_engine/effect_probe.cjs.
//
// This is generate_teams.cjs's vocabulary block (generate(), the `ids` .. `maps`
// lines) and its member row (the `generated.map(mon => ...)` body), factored into
// functions. generate_teams.cjs itself is NOT edited: the live gender-policy
// provenance pins its sha256 (src/pokezero/v5_python_backend.py, `generator_sources`),
// so any byte change there makes every existing V2 runtime refuse to prepare.
// `node team_rows.cjs SHOWDOWN_ROOT COUNT SEED` emits generate_teams.cjs's team
// lines through these functions; tests/test_gpu_engine_team_rows.py proves the two
// streams byte-identical.
const path = require('node:path');
const crypto = require('node:crypto');
const {calculatedStats, gen3Species} = require('./generate_teams.cjs');

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const TYPES = ['Normal','Fighting','Flying','Poison','Ground','Rock','Bug','Ghost','Steel',
  'Fire','Water','Grass','Electric','Psychic','Ice','Dragon','Dark'];

function poolVocabulary(dex, toID) {
  const ids = rows => ['_none', ...[...new Set(rows.map(x => x.id).filter(Boolean))].sort()];
  const moveIds = new Set(dex.moves.all().filter(x => x.exists && x.gen <= 3).map(x => x.id));
  moveIds.add('struggle');
  for (const type of TYPES.filter(x => !['Normal'].includes(x))) moveIds.add('hiddenpower' + type.toLowerCase());
  const vocabulary = {
    species: ids(gen3Species(dex)),
    moves: ['_none', ...[...moveIds].sort()],
    items: ids(dex.items.all().filter(x => x.exists && x.gen <= 3)),
    abilities: ids(dex.abilities.all().filter(x => x.exists && x.gen <= 3)),
    natures: ids(dex.natures.all()), types: ['_none', ...TYPES.map(x => x.toLowerCase())],
  };
  const maps = Object.fromEntries(Object.entries(vocabulary).map(([k,v]) => [k, new Map(v.map((s,i) => [s,i]))]));
  function id(kind, value) {
    const key = value ? toID(value) : '_none';
    if (!maps[kind].has(key)) throw new Error(`unknown ${kind} ID: ${key}`);
    return maps[kind].get(key);
  }
  return {vocabulary, id};
}

// One generated (or authored, same shape) Showdown set -> the pool member row.
function memberRow(dex, id, mon) {
  const species = dex.species.get(mon.species);
  const ivs = Object.fromEntries(STATS.map(s => [s, mon.ivs?.[s] ?? 31]));
  const evs = Object.fromEntries(STATS.map(s => [s, mon.evs?.[s] ?? 0]));
  const nature = dex.natures.get(mon.nature || 'Serious');
  const hp = dex.getHiddenPower(ivs);
  if (mon.moves.length < 1 || mon.moves.length > 4) throw new Error('invalid move count');
  const paddedMoves = [...mon.moves, ...Array(4-mon.moves.length).fill(null)];
  return {species:id('species', species.id), level:mon.level || 100,
    types:[id('types',species.types[0]), id('types',species.types[1])],
    stats:calculatedStats(species.baseStats, ivs, evs, mon.level || 100, nature),
    moves:paddedMoves.map(x => id('moves',x)),
    pp:paddedMoves.map(x => {if (x === null) return 0; const m=dex.moves.get(x); return m.noPPBoosts ? m.pp : Math.floor(m.pp*8/5);}),
    item:id('items',mon.item), ability:id('abilities',mon.ability), nature:id('natures',nature.id),
    ivs:STATS.map(s => ivs[s]), evs:STATS.map(s => evs[s]),
    hidden_power_type:id('types',hp.type), hidden_power_power:hp.power};
}

function load(root) {
  const {Dex, toID} = require(path.join(root, 'dist/sim/dex.js'));
  const dex = Dex.mod('gen3');
  return {dex, toID, ...poolVocabulary(dex, toID)};
}

function teamSeed(seed) {
  const bytes = crypto.createHash('sha256').update('pokezero-gpu-pool-v1:' + seed).digest();
  return Array.from({length:4}, (_, k) => bytes.readUInt16LE(2*k));
}

module.exports = {STATS, TYPES, poolVocabulary, memberRow, load, teamSeed};

if (require.main === module) {
  const [root, countText, seedText] = process.argv.slice(2);
  const count = Number(countText), seedStart = Number(seedText);
  if (!root || !Number.isSafeInteger(count) || count <= 0 || !Number.isSafeInteger(seedStart) || seedStart < 0) {
    throw new Error('usage: team_rows.cjs SHOWDOWN_ROOT POSITIVE_COUNT UINT32_SEED');
  }
  const {Teams} = require(path.join(root, 'dist/sim/teams.js'));
  const {dex, id, vocabulary} = load(root);
  const lines = [JSON.stringify({kind:'vocabulary', vocabulary})];
  for (let i = 0; i < count; i++) {
    const seed = seedStart + i;
    const generated = Teams.generate('gen3randombattle', {seed:teamSeed(seed)});
    if (generated.length !== 6) throw new Error('generator did not return six members');
    lines.push(JSON.stringify({kind:'team', index:i, seed, members:generated.map(mon => memberRow(dex, id, mon))}));
    if (lines.length >= 4096) {process.stdout.write(lines.join('\n') + '\n'); lines.length = 0;}
  }
  if (lines.length) process.stdout.write(lines.join('\n') + '\n');
}
