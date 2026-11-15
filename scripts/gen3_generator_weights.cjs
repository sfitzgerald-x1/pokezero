#!/usr/bin/env node
'use strict';

// Offline exact finite-choice interpreter for the pinned Gen 3 generator.
// Calls the real randomSet, including inherited methods; no move-selection port.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');

const PIN = 'f76228a1354b5d0f307ca2d16101294ad3a2308b';
// Rebuilt from clean PIN with the pinned lockfile; includes executed dependencies.
const EXECUTED_PIN = 'a27248481987e94a95cd2a1261ec7f2900b097c84bc1b670ab85f92c0db63f78';
const SCHEMA = 'gen3-generator-weights-v1';
const CONTEXT_KEYS = ['rain', 'sun', 'statusCure', 'rapidSpin'];
const PROBABILITY_MODEL = 'independent-uniform-api-choices';
const id = value => String(value).toLowerCase().replace(/[^a-z0-9]/g, '');
const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
const gcd = (a, b) => { while (b) [a, b] = [b, a % b]; return a; };
const rational = (n, d = 1n) => { const g = gcd(n, d); return [n / g, d / g]; };
const add = (a, b) => rational(a[0] * b[1] + b[0] * a[1], a[1] * b[1]);

class Fork {
  constructor(weights) { this.weights = weights; }
}

class BranchingPRNG {
  constructor(choices) { this.choices = choices; this.position = 0; this.probability = [1n, 1n]; }
  choose(weights) {
    const total = weights.reduce((a, b) => a + b, 0n);
    if (!total || weights.some(w => w < 0n)) throw new Error('invalid branch weights');
    const alive = weights.flatMap((w, i) => w ? [i] : []);
    if (alive.length === 1) return alive[0];
    if (this.position === this.choices.length) throw new Fork(weights);
    const choice = this.choices[this.position++];
    if (!weights[choice]) throw new Error('invalid replay choice');
    this.probability = rational(this.probability[0] * weights[choice], this.probability[1] * total);
    return choice;
  }
  random(from, to) {
    if (from === undefined) throw new Error('continuous random is not enumerable');
    if (to === undefined) [from, to] = [0, from];
    from = Math.floor(from); to = Math.floor(to);
    if (!(to > from)) throw new Error('invalid random range');
    return from + this.choose(Array(to - from).fill(1n));
  }
  randomChance(numerator, denominator) {
    return this.choose([BigInt(denominator - numerator), BigInt(numerator)]) === 1;
  }
  sample(items) { return items[this.random(items.length)]; }
  shuffle(items, start = 0, end = items.length) {
    while (start < end - 1) {
      const next = this.random(start, end);
      [items[start], items[next]] = [items[next], items[start]];
      start++;
    }
  }
  getSeed() { return '0,0,0,0'; }
}

function contextFromIndex(index) {
  if (!Number.isInteger(index) || index < 0 || index >= 16) throw new Error('invalid context');
  return Object.fromEntries(CONTEXT_KEYS.map((key, bit) => [key, index >> bit & 1]));
}

function contextIndex(context) {
  // randomTeam assigns spikes=1 (never increments), so the >=2 cull is
  // unreachable. Refuse arbitrary custom team contexts outside that contract.
  if (context.spikes >= 2) throw new Error('spikes >= 2 is unreachable in pinned randomTeam');
  return CONTEXT_KEYS.reduce((result, key, bit) => result | (!!context[key] << bit), 0);
}

// Cosmetic forme, shiny and move order are summed out, not treated as new sets.
// Hidden Power remains typed. Role is a generator route, not set identity.
function setKey(set) {
  return JSON.stringify([id(set.speciesId || set.species), set.level, set.moves.map(id).sort(), id(set.ability), id(set.item)]);
}

function enumerate(generator, species, context = {}, {maxLeaves = 2000000} = {}) {
  const originalPRNG = generator.prng;
  const pending = [[]];
  const outcomes = new Map();
  let leaves = 0;
  try {
  while (pending.length) {
    const choices = pending.pop();
    const prng = new BranchingPRNG(choices);
    generator.prng = prng;
    try {
      const set = generator.randomSet(species, {...context});
      if (++leaves > maxLeaves) throw new Error(`enumeration leaf budget exceeded: ${species}`);
      if (prng.position !== choices.length) throw new Error('unused replay choices');
      const key = setKey(set);
      const stats = JSON.stringify([set.evs, set.ivs]);
      const previous = outcomes.get(key);
      if (previous && previous.stats !== stats) throw new Error(`non-unique stats for ${key}`);
      outcomes.set(key, {stats, probability: add(previous?.probability || [0n, 1n], prng.probability)});
    } catch (error) {
      if (!(error instanceof Fork)) throw error;
      for (let i = error.weights.length - 1; i >= 0; i--) {
        if (error.weights[i]) pending.push([...choices, i]);
      }
    }
  }
  } finally {
    generator.prng = originalPRNG;
  }
  const total = [...outcomes.values()].reduce((sum, value) => add(sum, value.probability), [0n, 1n]);
  if (total[0] !== 1n || total[1] !== 1n) throw new Error(`probability mass is not one: ${species}`);
  return {leaves, outcomes};
}

function integerDistribution(outcomes) {
  let denominator = 1n;
  for (const {probability: [, d]} of outcomes.values()) denominator = denominator / gcd(denominator, d) * d;
  return {
    denominator: String(denominator),
    weights: Object.fromEntries([...outcomes].sort(([a], [b]) => a.localeCompare(b)).map(([key, {probability: [n, d]}]) =>
      [key, String(n * (denominator / d))])),
  };
}

function loadGenerator(root) {
  root = path.resolve(root);
  const commit = execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
  if (commit !== PIN) throw new Error(`expected Showdown ${PIN}, found ${commit}`);
  execFileSync('git', ['-C', root, 'diff', '--quiet', PIN, '--', 'data', 'sim', 'lib', 'config/formats.ts']);
  const {RandomGen3Teams} = require(path.join(root, 'dist/data/random-battles/gen3/teams.js'));
  const generator = new RandomGen3Teams('gen3randombattle', [1, 2, 3, 4]);
  generator.enumerationRoot = root;
  if (executedDigest(root) !== EXECUTED_PIN) throw new Error('executed Showdown files differ from pinned build');
  return generator;
}

function executedDigest(root) {
  // Bind the actual JS/JSON dependencies, including ignored build output.
  // The source commit alone cannot identify a stale or patched dist tree.
  const modulesRoot = fs.realpathSync(path.join(root, 'node_modules'));
  const files = Object.keys(require.cache).flatMap(file => {
    if (file.startsWith(modulesRoot + path.sep)) return [[`node_modules/${path.relative(modulesRoot, file)}`, file]];
    if (file.startsWith(root + path.sep)) return [[path.relative(root, file), file]];
    return [];
  }).sort(([a], [b]) => a.localeCompare(b));
  const hash = crypto.createHash('sha256');
  for (const [relative, file] of files) {
    hash.update(relative.split(path.sep).join('/') + '\0');
    hash.update(fs.readFileSync(file)); hash.update('\0');
  }
  return hash.digest('hex');
}

function compile(generator, speciesList, contexts) {
  const species = {};
  for (const name of speciesList) {
    const rows = {};
    for (const index of contexts) {
      const result = enumerate(generator, name, contextFromIndex(index));
      rows[index] = integerDistribution(result.outcomes);
    }
    species[name] = rows;
    process.stderr.write(`${name}: ${Object.keys(rows).length} contexts\n`);
  }
  const complete = JSON.stringify([...speciesList].sort()) === JSON.stringify(Object.keys(generator.randomSets).sort()) &&
    JSON.stringify([...contexts].sort((a, b) => a - b)) === JSON.stringify(Array.from({length:16}, (_, i) => i));
  const payload = {schema: SCHEMA, showdown_commit: PIN, probability_model: PROBABILITY_MODEL,
    compiler_sha256: sha256(fs.readFileSync(__filename)), executed_sha256: executedDigest(generator.enumerationRoot),
    complete, context_keys: CONTEXT_KEYS, species};
  return {...payload, sha256: sha256(JSON.stringify(payload))};
}

function validateEnumeration(enumeration, manifest, {diagnostic = false} = {}) {
  const {sha256: digest, ...payload} = enumeration;
  if (digest !== sha256(JSON.stringify(payload))) throw new Error('enumeration hash mismatch');
  if (enumeration.schema !== SCHEMA || enumeration.showdown_commit !== PIN ||
      enumeration.probability_model !== PROBABILITY_MODEL ||
      JSON.stringify(enumeration.context_keys) !== JSON.stringify(CONTEXT_KEYS) ||
      enumeration.executed_sha256 !== EXECUTED_PIN ||
      !/^[a-f0-9]{64}$/.test(enumeration.compiler_sha256)) throw new Error('enumeration provenance mismatch');
  if (!diagnostic && (!enumeration.complete ||
      JSON.stringify(Object.keys(enumeration.species).sort()) !== JSON.stringify(Object.keys(manifest.species).sort()))) {
    throw new Error('incomplete species coverage');
  }
  for (const [name, rows] of Object.entries(enumeration.species)) {
    if (!diagnostic && JSON.stringify(Object.keys(rows)) !== JSON.stringify(Array.from({length:16}, (_, i) => String(i)))) {
      throw new Error(`incomplete context coverage: ${name}`);
    }
    for (const [context, row] of Object.entries(rows)) {
      contextFromIndex(Number(context));
      if (!/^[1-9][0-9]*$/.test(row.denominator)) throw new Error('invalid denominator');
      let sum = 0n;
      for (const weight of Object.values(row.weights)) {
        if (!/^[1-9][0-9]*$/.test(weight)) throw new Error('invalid weight');
        sum += BigInt(weight);
      }
      if (sum !== BigInt(row.denominator)) throw new Error('invalid probability mass');
    }
  }
}

function mapToVariants(enumeration, manifest, options = {}) {
  validateEnumeration(enumeration, manifest, options);
  const species = {};
  for (const [name, rows] of Object.entries(enumeration.species)) {
    const variants = manifest.species[name];
    if (!variants || variants.length > 64) throw new Error(`invalid variant list: ${name}`);
    const indices = new Map();
    variants.forEach((variant, index) => {
      const key = setKey(variant);
      // The same physical set can appear under multiple roles. Its generator
      // paths are already summed: keep one representative in source order.
      if (!indices.has(key)) indices.set(key, index);
    });
    let denominator = 1n;
    for (const row of Object.values(rows)) {
      const d = BigInt(row.denominator);
      denominator = denominator / gcd(denominator, d) * d;
    }
    const contexts = {};
    for (const [context, row] of Object.entries(rows)) {
      const weights = variants.map(() => 0n);
      for (const [key, weight] of Object.entries(row.weights)) {
        if (!indices.has(key)) throw new Error(`generated set missing from variant list: ${name} context=${context} ${key}`);
        weights[indices.get(key)] += BigInt(weight) * (denominator / BigInt(row.denominator));
      }
      if (weights.reduce((a, b) => a + b, 0n) !== denominator) throw new Error(`invalid mapped mass: ${name}`);
      contexts[context] = weights.map(String);
    }
    species[name] = {denominator: String(denominator), variant_ids: variants.map(v => v.variant_id), contexts};
  }
  const payload = {schema: SCHEMA, showdown_commit: enumeration.showdown_commit, source_hash: manifest.source_hash,
    probability_model: enumeration.probability_model, executed_sha256: enumeration.executed_sha256,
    compiler_sha256: enumeration.compiler_sha256, complete: enumeration.complete && !options.diagnostic,
    variant_manifest_sha256: sha256(JSON.stringify(manifest)), enumeration_sha256: enumeration.sha256,
    context_keys: enumeration.context_keys, species};
  return {...payload, sha256: sha256(JSON.stringify(payload))};
}

function main(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith('--') || !argv[i + 1]) throw new Error('expected --key value');
    options[argv[i].slice(2)] = argv[i + 1];
  }
  if (!options.root || !options.output) throw new Error('required: --root BUILT_SHOWDOWN --output FILE');
  const generator = loadGenerator(options.root);
  const species = options.species ? options.species.split(',') : Object.keys(generator.randomSets).sort();
  const contexts = options.contexts ? options.contexts.split(',').map(Number) : Array.from({length: 16}, (_, i) => i);
  const enumeration = compile(generator, species, contexts);
  const result = options.variants ? mapToVariants(enumeration, JSON.parse(fs.readFileSync(options.variants))) : enumeration;
  fs.writeFileSync(options.output, JSON.stringify(result) + '\n');
}

module.exports = {PIN, EXECUTED_PIN, SCHEMA, PROBABILITY_MODEL, CONTEXT_KEYS, sha256, BranchingPRNG, contextFromIndex, contextIndex,
  setKey, enumerate, integerDistribution, loadGenerator, executedDigest, compile, validateEnumeration, mapToVariants};
if (require.main === module) main(process.argv.slice(2));
