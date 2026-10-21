// Offline real-team generator; never part of the device step or timed benchmark.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {once} = require('node:events');

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const TYPES = ['Normal','Fighting','Flying','Poison','Ground','Rock','Bug','Ghost','Steel',
  'Fire','Water','Grass','Electric','Psychic','Ice','Dragon','Dark'];
const SPECIAL = new Set(['Fire','Water','Grass','Electric','Psychic','Ice','Dragon','Dark']);

function moveTypeId(type, types) {
  // Gen 3 Curse is genuinely typeless, not Normal. Zero is explicit here only.
  if (type === '???') return 0;
  const index = types.indexOf(type.toLowerCase());
  if (index <= 0) throw new Error(`unknown move type: ${type}`);
  return index;
}

function calculatedStats(base, ivs, evs, level, nature) {
  return STATS.map(s => {
    const core = Math.floor((2 * base[s] + ivs[s] + Math.floor(evs[s] / 4)) * level / 100);
    if (s === 'hp') return base.hp === 1 ? 1 : core + level + 10;
    const multiplier = nature.plus === s ? 110 : nature.minus === s ? 90 : 100;
    return Math.floor((core + 5) * multiplier / 100);
  });
}

function compiledTree(root) {
  const files = [];
  function visit(dir) {
    for (const entry of fs.readdirSync(dir, {withFileTypes:true})) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile() && /\.(js|json)$/.test(entry.name)) files.push(file);
    }
  }
  visit(root);
  const hash = crypto.createHash('sha256');
  for (const file of files.sort()) {
    hash.update(path.relative(root, file).split(path.sep).join('/') + '\0');
    hash.update(crypto.createHash('sha256').update(fs.readFileSync(file)).digest());
  }
  return {sha256:hash.digest('hex'), files:files.length};
}

async function generate(root, count, seedStart, sourceCommitLabel) {
  const {Dex, toID} = require(path.join(root, 'dist/sim/dex.js'));
  const {Teams} = require(path.join(root, 'dist/sim/teams.js'));
  const rngPackage = require.resolve('ts-chacha20/package.json', {paths:[root]});
  const dex = Dex.mod('gen3');
  const ids = rows => ['_none', ...[...new Set(rows.map(x => x.id).filter(Boolean))].sort()];
  const moveIds = new Set(dex.moves.all().filter(x => x.exists && x.gen <= 3).map(x => x.id));
  moveIds.add('struggle');
  for (const type of TYPES.filter(x => !['Normal'].includes(x))) moveIds.add('hiddenpower' + type.toLowerCase());
  const vocabulary = {
    species: ids(dex.species.all().filter(x => x.exists && x.gen <= 3)),
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
  const moves = vocabulary.moves.map(name => {
    if (name === '_none') return null;
    const m = dex.moves.get(name);
    return {id:name, type:moveTypeId(m.type, vocabulary.types), source_type:m.type,
      category:m.category === 'Status' ? 'Status' : SPECIAL.has(m.type) ? 'Special' : 'Physical',
      base_power:m.basePower, pp:m.pp, accuracy:m.accuracy, priority:m.priority,
      flags:m.flags, implemented:false};
  });
  const chart = TYPES.map(attack => TYPES.map(defend => {
    const code = dex.types.get(defend).damageTaken[attack] || 0;
    return [1,2,0.5,0][code];
  }));
  async function emit(record) {
    if (!process.stdout.write(JSON.stringify(record) + '\n')) await once(process.stdout, 'drain');
  }
  await emit({kind:'header', schema:'pokezero.gpu-team-stream.v1', format:'gen3randombattle',
    team_count:count, seed_start:seedStart, source_commit_label:sourceCommitLabel,
    compiled_tree:compiledTree(path.join(root, 'dist')),
    rng_dependency:{name:'ts-chacha20', version:require(rngPackage).version,
      tree:compiledTree(path.dirname(rngPackage))}, vocabulary, moves,
    type_chart:chart, stats_order:STATS, seed_domain:'pokezero-gpu-pool-v1',
    node_version:process.version, scope:'Real generated teams; no transitions or game outcomes'});
  for (let i=0; i<count; i++) {
    const seed = seedStart + i;
    const bytes = crypto.createHash('sha256').update('pokezero-gpu-pool-v1:' + seed).digest();
    const teamSeed = Array.from({length:4}, (_, k) => bytes.readUInt16LE(2*k));
    const generated = Teams.generate('gen3randombattle', {seed:teamSeed});
    if (generated.length !== 6) throw new Error('generator did not return six members');
    const members = generated.map(mon => {
      const species = dex.species.get(mon.species);
      const ivs = Object.fromEntries(STATS.map(s => [s, mon.ivs?.[s] ?? 31]));
      const evs = Object.fromEntries(STATS.map(s => [s, mon.evs?.[s] ?? 0]));
      const nature = dex.natures.get(mon.nature || 'Serious');
      const hp = dex.getHiddenPower(ivs);
      if (mon.moves.length !== 4) throw new Error('generator did not return four moves');
      return {species:id('species', species.id), level:mon.level || 100,
        types:[id('types',species.types[0]), id('types',species.types[1])],
        stats:calculatedStats(species.baseStats, ivs, evs, mon.level || 100, nature),
        moves:mon.moves.map(x => id('moves',x)),
        pp:mon.moves.map(x => {const m=dex.moves.get(x); return m.noPPBoosts ? m.pp : Math.floor(m.pp*8/5);}),
        item:id('items',mon.item), ability:id('abilities',mon.ability), nature:id('natures',nature.id),
        ivs:STATS.map(s => ivs[s]), evs:STATS.map(s => evs[s]),
        hidden_power_type:id('types',hp.type), hidden_power_power:hp.power};
    });
    await emit({kind:'team', index:i, seed, members});
  }
  await emit({kind:'terminal', team_count:count});
}

module.exports = {calculatedStats, moveTypeId};
if (require.main === module) {
  const [root, countText, seedText, sourceCommitLabel] = process.argv.slice(2);
  const count = Number(countText), seed = Number(seedText);
  if (!root || !Number.isSafeInteger(count) || count <= 0 || !Number.isSafeInteger(seed)
      || seed < 0 || seed + count > 2**32 || !/^[a-f0-9]{40}$/.test(sourceCommitLabel || '')) {
    throw new Error('usage: generate_teams.cjs SHOWDOWN_ROOT POSITIVE_COUNT UINT32_SEED SOURCE_COMMIT_LABEL');
  }
  generate(root, count, seed, sourceCommitLabel).catch(error => {console.error(error); process.exitCode=1;});
}
