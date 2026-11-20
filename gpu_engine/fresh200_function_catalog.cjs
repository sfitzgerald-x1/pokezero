'use strict';

// Construct the policy from a separate, pinned source-only process. Derive
// method identities from the format's constructor override rules and match
// complete Function#toString bytes to parser spans in the admitted files.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {preflight} = require('./fresh200_source_preflight.cjs');

const FORMAT = 'gen3customgame@@@sleepclausemod';
const ACORN_SHA = 'fdb08546776ec6228b03e8d02b40d4ab3255bae5f401adba7ff5dad927ac5c9c';
const SIM_FILES = {
  Battle: 'battle', BattleActions: 'battle-actions', BattleQueue: 'battle-queue',
  Pokemon: 'pokemon', Side: 'side', Field: 'field',
};
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function requireEvidence(condition, message) {
  if (!condition) throw new Error(message);
}

function properties(scope, name, {enumerable = false} = {}) {
  if (!scope) return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(scope, name);
  if (!descriptor || (enumerable && !descriptor.enumerable)) return undefined;
  requireEvidence(Object.hasOwn(descriptor, 'value'), 'accessor used as method override: ' + name);
  return descriptor;
}

function original(prototype, name, ...overrides) {
  for (const scope of overrides) {
    const descriptor = properties(scope, name, {enumerable: true});
    if (descriptor) return descriptor.value;
  }
  for (let owner = prototype; owner !== null; owner = Object.getPrototypeOf(owner)) {
    const descriptor = properties(owner, name);
    if (descriptor) return descriptor.value;
  }
  throw new Error('original method is absent: ' + name);
}

function battleScriptOverride(scripts, name) {
  for (let owner = scripts; owner != null; owner = Object.getPrototypeOf(owner)) {
    const descriptor = properties(owner, name);
    if (descriptor) return descriptor.enumerable ? descriptor.value : undefined;
  }
  return undefined;
}

function propertyKey(node) {
  if (!node || node.computed) return null;
  if (node.key?.type === 'Identifier') return node.key.name;
  if (typeof node.key?.value === 'string') return node.key.value;
  return null;
}

function parseSpans(parser, name, bytes) {
  const source = new TextDecoder('utf-8', {fatal: true, ignoreBOM: true}).decode(bytes);
  const ast = parser.parse(source, {ecmaVersion: 'latest', sourceType: 'script'});
  const spans = [];
  function visit(node, ancestors) {
    const parent = ancestors.at(-1);
    const parsedMethod = (node.type === 'MethodDefinition' && !node.static && node.kind !== 'get' && node.kind !== 'set')
      || (node.type === 'Property' && node.method);
    const alreadyRepresented = parent?.value === node
      && (parent.method || parent.type === 'MethodDefinition' || parent.kind === 'get' || parent.kind === 'set');
    const forms = {
      FunctionDeclaration: 'function-declaration', FunctionExpression: 'function-expression',
      ArrowFunctionExpression: 'arrow-function',
    };
    const form = parsedMethod ? 'method-definition' : (!alreadyRepresented && forms[node.type]);
    if (form) {
      const property = parsedMethod ? propertyKey(node) : propertyKey(parent);
      const text = source.slice(node.start, node.end);
      const declaration = property ?? node.id?.name ?? 'anonymous';
      const isCallback = /^on[A-Z]/.test(property ?? '') || /Callback$/.test(property ?? '');
      spans.push({start: node.start, end: node.end, sha256: sha(text), declaration, form, isCallback});
    }
    for (const value of Object.values(node)) {
      if (Array.isArray(value)) {
        for (const child of value) if (child?.type) visit(child, [...ancestors, node]);
      } else if (value?.type) visit(value, [...ancestors, node]);
    }
  }
  visit(ast, []);
  const offsets = [...new Set(spans.flatMap(({start, end}) => [start, end]))].sort((a, b) => a - b);
  const byteOffsets = new Map();
  let last = 0;
  let total = 0;
  for (const position of offsets) {
    total += Buffer.byteLength(source.slice(last, position));
    byteOffsets.set(position, total);
    last = position;
  }
  return spans.map(span => ({
    hash: span.sha256,
    isCallback: span.isCallback,
    location: {source: name, start: byteOffsets.get(span.start), end: byteOffsets.get(span.end),
      declaration: span.declaration, source_form: span.form},
  }));
}

function sourceFiles(pins) {
  const dist = path.join(pins.native_root, 'dist');
  const core = Object.values(SIM_FILES).map(name => path.join(dist, 'sim', name + '.js'));
  const callbackTables = ['moves', 'abilities', 'items', 'conditions', 'rulesets', 'scripts', 'pokedex'];
  const tables = callbackTables.map(name => path.join(dist, 'data', name + '.js'));
  for (const mod of ['gen3', 'gen4', 'gen5', 'gen6', 'gen7', 'gen8']) {
    for (const name of callbackTables) {
      const file = path.join(dist, 'data/mods', mod, name + '.js');
      if (fs.existsSync(file)) tables.push(file);
    }
  }
  const formats = path.join(dist, 'config/formats.js');
  return [...core, ...tables, formats, path.join(__dirname, 'chance_hooks.cjs')];
}

function constructCatalog(pins, identity, parserPath) {
  requireEvidence(path.isAbsolute(parserPath), 'absolute parser path required');
  const parserBytes = fs.readFileSync(parserPath);
  requireEvidence(sha(parserBytes) === ACORN_SHA, 'pinned Acorn source differs');
  const parser = require(parserPath);
  requireEvidence(parser.version === '8.15.0', 'pinned Acorn version differs');
  const admission = preflight(pins, identity);
  const dist = path.join(pins.native_root, 'dist');
  const sim = name => require(path.join(dist, 'sim', name + '.js'));
  const {Dex} = sim('dex');
  const format = Dex.formats.get(FORMAT, true);
  const dex = Dex.forFormat(format);
  requireEvidence(dex.gen === 3 && format.id === 'gen3customgame', 'frozen Native format differs');
  const scripts = dex.data.Scripts;
  const {METHODS} = require('./fresh200_witness_native.cjs');
  const originals = new Map();
  const bootstrap = [];
  const instances = [];
  const battlePrototype = sim('battle').Battle.prototype;
  const add = (owner, name, value, stage) => {
    const key = owner + '.' + name;
    requireEvidence(typeof value === 'function', 'unsupported effective original: ' + key);
    originals.set(key, {hash: sha(Function.prototype.toString.call(value)), stage});
    (stage === 'bootstrap-method' ? bootstrap : instances).push(key);
  };
  for (const owner of ['Pokemon', 'Side', 'Field']) {
    const prototype = sim(SIM_FILES[owner])[owner].prototype;
    for (const name of METHODS[owner]) add(owner, name, original(prototype, name), 'bootstrap-method');
  }
  add('Battle', 'setPlayer', original(battlePrototype, 'setPlayer'), 'bootstrap-method');
  for (const owner of ['Battle', 'BattleActions', 'BattleQueue']) {
    const prototype = sim(SIM_FILES[owner])[owner].prototype;
    for (const name of METHODS[owner]) {
      let value;
      if (owner === 'Battle') {
        const mod = battleScriptOverride(scripts, name);
        const formatOverride = properties(format.battle, name, {enumerable: true});
        value = formatOverride ? formatOverride.value :
          (typeof mod === 'function' ? mod : original(prototype, name));
      } else if (owner === 'BattleActions') {
        value = original(prototype, name, format.actions, scripts.actions);
      } else {
        value = original(prototype, name, format.queue || scripts.queue);
      }
      add(owner, name, value, 'instance-method');
    }
  }

  const filePins = {};
  const allSpans = new Map();
  const callbackHashes = new Set();
  let total = 0;
  for (const file of sourceFiles(pins)) {
    const bytes = fs.readFileSync(file);
    total += bytes.length;
    requireEvidence(total <= 64 * 1024 ** 2 && bytes.length <= 16 * 1024 ** 2, 'source catalog byte limit exceeded');
    filePins[file] = sha(bytes);
    const simulatorSource = path.dirname(file) === path.join(dist, 'sim');
    for (const span of parseSpans(parser, file, bytes)) {
      if (!allSpans.has(span.hash)) allSpans.set(span.hash, []);
      allSpans.get(span.hash).push(span.location);
      if (simulatorSource || span.isCallback) callbackHashes.add(span.hash);
    }
  }
  const functions = [];
  for (const [operation, {hash, stage}] of originals) {
    const locations = allSpans.get(hash);
    requireEvidence(locations?.length > 0, 'effective original lacks a complete source span: ' + operation);
    functions.push({operation, originalSha256: hash, kind: 'method', stage, locations});
  }
  for (const hash of [...callbackHashes].sort()) {
    functions.push({operation: 'Callback:' + hash, originalSha256: hash, kind: 'callback-own-apply',
      stage: 'callback', locations: allSpans.get(hash)});
  }
  return {
    schema: 'pokezero.fresh200.parsed_native_catalog.v1',
    qualification: false,
    format: FORMAT,
    parser_sha256: ACORN_SHA,
    admission,
    source_pins: filePins,
    functions,
    bootstrap_operations: bootstrap,
    instance_operations: instances,
  };
}

if (require.main === module) {
  const [pinFile, identityFile, parserFile] = process.argv.slice(2);
  requireEvidence(process.argv.length === 5, 'usage: fresh200_function_catalog.cjs <pins.json> <identity.json> <acorn.cjs>');
  const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
  const result = constructCatalog(read(pinFile), read(identityFile), parserFile);
  process.stdout.write(JSON.stringify(result) + '\n');
}
