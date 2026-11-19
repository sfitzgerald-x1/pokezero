'use strict';

// Prepare independent source identities for the recorder. This process loads
// the pinned Native modules and installs the existing chance guards without
// constructing a battle, receiving stdin commands, or running callbacks.
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const HELPERS = [
  'fresh200_witness_core.cjs', 'fresh200_witness_bridge.cjs', 'chance_hooks.cjs',
  'fresh200_bridge.mjs', 'fresh200_witness_native.cjs', 'fresh200_witness_transport.cjs',
];
const BRIDGE = 'scripts/battle_bridge.mjs';
const PORT = 'scripts/battle_bridge_witness_port.mjs';
const SOURCE_FILES = [
  ...HELPERS.map(name => 'gpu_engine/' + name), BRIDGE, PORT,
  'scripts/battle_bridge_reference_rest.mjs', 'scripts/battle_bridge_reference_sleep.mjs',
  'scripts/battle_bridge_reference_trace.mjs', 'scripts/battle_bridge_boundary_requests.mjs',
  'gpu_engine/fresh200_source_preflight.cjs', 'gpu_engine/gender_policy.cjs',
];
const REPO = path.dirname(__dirname);
const ORIGINAL_NATIVE = 'f76228a1354b5d0f307ca2d16101294ad3a2308b';
const INTEGRATION = 'f9b0c48ef0d14c15f017c3e4b97dddc6695c5b3d';
const COMPILED = 'd22c0cc51aa32b983c2946108834418e824b0056a727e121e19dc56919350a01';
const NODE = '3159f9115ab4be7d318b7c28e946837a4dceb7f2b3c43232aa2f2e3852550b90';
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const wire = value => Buffer.from(JSON.stringify(value) + '\n', 'utf8');
const hex = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const text = value => typeof value === 'string' && value.length > 0;

function requireEvidence(condition, message) {
  if (!condition) throw new Error(message);
}

function keysMatch(object, keys) {
  if (!object || typeof object !== 'object' || Array.isArray(object)) return false;
  const names = Object.keys(object);
  return names.length === keys.length && keys.every(key => Object.hasOwn(object, key));
}

function hashFile(file) {
  const hash = crypto.createHash('sha256');
  const buffer = Buffer.alloc(64 * 1024);
  const fd = fs.openSync(file, 'r');
  try {
    requireEvidence(fs.fstatSync(fd).isFile(), 'expected a regular source file');
    let count;
    while ((count = fs.readSync(fd, buffer, 0, buffer.length, null)) !== 0) {
      hash.update(buffer.subarray(0, count));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest('hex');
}

function functionSha(value) {
  requireEvidence(typeof value === 'function', 'expected an original Native function');
  const source = Function.prototype.toString.call(value);
  requireEvidence(!/\{\s*\[native code\]\s*\}$/.test(source), 'Native function lacks literal source');
  return sha(source);
}

function methodSha(prototype, name) {
  for (let owner = prototype; owner !== null; owner = Object.getPrototypeOf(owner)) {
    const descriptor = Object.getOwnPropertyDescriptor(owner, name);
    if (descriptor) return functionSha(descriptor.value);
  }
  throw new Error('missing original method ' + String(name));
}

function preflight(pins, identity) {
  requireEvidence(keysMatch(pins, ['integration_commit', 'native_commit', 'native_root',
    'compiled_tree_sha256', 'node_path', 'node_sha256', 'sources']), 'closed source pin schema required');
  requireEvidence(pins.integration_commit === INTEGRATION && pins.native_commit === ORIGINAL_NATIVE
    && pins.compiled_tree_sha256 === COMPILED && pins.node_sha256 === NODE, 'pinned runtime identity differs');
  requireEvidence(typeof pins.native_root === 'string' && path.isAbsolute(pins.native_root)
    && path.normalize(pins.native_root) === pins.native_root, 'literal absolute Native root required');
  requireEvidence(typeof pins.node_path === 'string' && path.isAbsolute(pins.node_path)
    && fs.realpathSync(pins.node_path) === fs.realpathSync(process.execPath)
    && process.version === 'v22.22.2' && hashFile(process.execPath) === NODE, 'pinned Node executable differs');
  requireEvidence(keysMatch(identity, ['session', 'battle_id', 'completion_fd'])
    && text(identity.session) && text(identity.battle_id) && Number.isSafeInteger(identity.completion_fd)
    && identity.completion_fd >= 3 && identity.completion_fd <= 255, 'literal bridge identifiers required');
  requireEvidence(keysMatch(pins.sources, SOURCE_FILES), 'complete producer file pins required');
  for (const name of SOURCE_FILES) {
    requireEvidence(hex(pins.sources[name]) && hashFile(path.join(REPO, name)) === pins.sources[name],
      'producer file identity differs: ' + name);
  }

  const dist = path.join(pins.native_root, 'dist');
  const compiled = require('./gender_policy.cjs').compiledTree(dist);
  requireEvidence(compiled.sha256 === COMPILED && compiled.files === 500, 'pinned compiled Native tree differs');
  const sim = name => require(path.join(dist, 'sim', name + '.js'));
  const {Battle} = sim('battle');
  const {PRNG} = sim('prng');
  const {BattleStream, getPlayerStreams} = sim('battle-stream');
  const bridgeExports = sim('index');
  requireEvidence(bridgeExports.BattleStream === BattleStream && bridgeExports.getPlayerStreams === getPlayerStreams,
    'bridge and preflight use different Native stream exports');
  const {ObjectReadStream} = require(path.join(dist, 'lib/streams.js'));
  const originalSetPlayer = methodSha(Battle.prototype, 'setPlayer');
  const noRuntimeAction = () => { throw new Error('source preflight attempted a runtime action'); };
  require('./chance_hooks.cjs').install(Battle, PRNG, noRuntimeAction, noRuntimeAction,
    noRuntimeAction, path.join(dist, 'sim/pokemon.js'), {strictNoRng: true});

  const stream = BattleStream.prototype;
  const reader = ObjectReadStream.prototype;
  const methods = {
    'source.iterator': methodSha(stream, Symbol.asyncIterator),
    'source.next': methodSha(stream, 'next'),
    'source.write': methodSha(stream, '_write'),
    'source.push': methodSha(stream, 'push'),
    'source.end': methodSha(stream, 'pushEnd'),
    'source.error': methodSha(stream, 'pushError'),
    'stream.push': methodSha(reader, 'push'),
    'stream.end': methodSha(reader, 'pushEnd'),
    'stream.error': methodSha(reader, 'pushError'),
  };
  const declaration = wire({
    schema: 'pokezero.native_transport_declaration.v1',
    session: identity.session,
    battle_id: identity.battle_id,
    completion_fd: identity.completion_fd,
    expected_methods: methods,
  });
  const registry = {
    nativeRoot: pins.native_root,
    node: process.version,
    nativeMode: true,
    transportMode: true,
    registrationEvidence: true,
    transportFunctions: {
      methods,
      getPlayerStreams: functionSha(getPlayerStreams),
      battleStreamFile: hashFile(path.join(dist, 'sim/battle-stream.js')),
      streamLibraryFile: hashFile(path.join(dist, 'lib/streams.js')),
      declaration: sha(declaration),
    },
    helpers: Object.fromEntries(HELPERS.map(name => [name, pins.sources['gpu_engine/' + name]])),
    bridge: pins.sources[BRIDGE],
    port: pins.sources[PORT],
    setPlayer: methodSha(Battle.prototype, 'setPlayer'),
  };
  const sourceRegistryHash = sha(wire(registry));
  return {
    schema: 'pokezero.fresh200.source_preflight.v1',
    qualification: false,
    source_registry_hash: sourceRegistryHash,
    expected_registry_base64: wire({registry, sourceRegistryHash}).toString('base64'),
    transport_declaration_base64: declaration.toString('base64'),
    native_declaration_base64: wire({enabled: true, qualification: false, registrationEvidence: true}).toString('base64'),
    original_set_player_sha256: originalSetPlayer,
    pins,
    compiled,
  };
}

if (require.main === module) {
  const args = process.argv.slice(2);
  requireEvidence(args.length === 2, 'usage: fresh200_source_preflight.cjs <pins.json> <bridge-identity.json>');
  const read = file => {
    const bytes = fs.readFileSync(file);
    requireEvidence(bytes.length <= 64 * 1024, 'bounded preflight input required');
    return JSON.parse(bytes);
  };
  process.stdout.write(wire(preflight(read(args[0]), read(args[1]))));
}
