// Synthetic transport/lifecycle tests only: no Showdown import or game.
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import {spawn} from "node:child_process";
import {fileURLToPath} from "node:url";
import test from "node:test";
import {createOwnedContinuations} from "../scripts/battle_bridge_owned_continuation.mjs";

const tick = () => new Promise(resolve => setImmediate(resolve));
const hash = value => crypto.createHash("sha256").update(value).digest("hex");
function snapshot() {
  return {battle: {sides: [{pokemon: [{set: {moves: ["fixture"]}}]}], log: [], ended: false},
    boundaryRequests: {p1: {active: [{}]}, p2: {wait: true}}, terminalScheduled: false};
}
function fixture(mode = {}) {
  const events = [], battles = new Map(), created = [], decoded = [], calls = [];
  class Stream {
    constructor() { this.atEOF = false; this.channels = {}; this.endCount = 0; created.push(this); }
    pushMessage(type, data) { calls.push([this, type, data]); }
    async writeEnd() {
      this.endCount++;
      if (mode.hangEnd) return new Promise(() => {});
      this.atEOF = !mode.noEOF;
      if (this.battle && !mode.noDestroy) {
        this.battle.field = null; this.battle.queue = null; this.battle.sides = [null];
      }
      for (const channel of Object.values(this.channels)) {
        if (mode.hangPump) continue;
        if (mode.pumpError) channel.reject(new Error("synthetic pump failure"));
        else channel.resolve();
      }
      if (mode.endError) throw new Error("synthetic teardown failure");
    }
  }
  const api = createOwnedContinuations({BattleStream: Stream,
    getPlayerStreams: stream => stream.channels = Object.fromEntries(["omniscient", "p1", "p2"].map(seat => [seat, {}])),
    State: {serializeBattle: battle => {
      if (mode.serializeError) throw new Error("synthetic serialization failure");
      return {...structuredClone(battle.serialized), ended: battle.ended};
    }, deserializeBattle: input => {
      if (mode.decodeError) throw new Error("synthetic decode failure");
      const battle = {field: {}, queue: {}, sides: [{}], ended: !!mode.decodedTerminal,
        set: input.sides[0].pokemon[0].set, log: input.log, serialized: structuredClone(input),
        restart(send) { if (mode.restartError) throw new Error("synthetic restart failure"); this.send = send; }};
      decoded.push(battle); return battle;
    }}, battles,
    newBattleState: battleId => ({battleId, readyEpoch: 0, boundaryGeneration: 0, terminalScheduled: false}),
    listenToStream: (battle, seat, channel, strict) => {
      assert.equal(strict, true);
      // Closure retains the ORIGINAL state/token permanently.
      channel.emit = lines => events.push({type: "stream", battleId: battle.battleId, stream: seat, lines});
      return new Promise((resolve, reject) => Object.assign(channel, {resolve, reject}));
    },
    actionableRequestedPlayers: battle => ["p1", "p2"].filter(seat =>
      battle.boundaryRequests[seat]?.active?.length && !battle.boundaryRequests[seat]?.wait),
    emit: event => events.push(event), cleanupTimeoutMs: 30});
  const clone = (token = "cell-1", root = snapshot()) => api.execute({type: "clone_snapshot",
    battleId: token, cloneId: `clone-${token}`, snapshotJson: JSON.stringify(root)}, () => assert.fail("legacy clone"));
  const release = (token = "cell-1", changes = {}) => {
    const ack = events.find(event => event.type === "cloned" && event.battleId === token);
    return api.execute({type: "release_clone", battleId: token, cloneId: ack?.cloneId,
      snapshotJsonSha256: ack?.snapshotJsonSha256, ...changes}, () => assert.fail("legacy release"));
  };
  return {api, events, battles, created, decoded, calls, clone, release};
}

test("clone binds received bytes and restores one empty stream without >start", async () => {
  const f = fixture(), root = snapshot(), encoded = JSON.stringify(root);
  await f.clone("cell-1", root);
  assert.deepEqual(f.events, [{type: "cloned", battleId: "cell-1", cloneId: "clone-cell-1",
    snapshotJsonSha256: hash(encoded), requested: ["p1"]}]);
  assert.equal(f.created.length, 1);
  assert.deepEqual(f.calls, []);
  f.decoded[0].send("update", ["a", "b"]);
  assert.deepEqual(f.calls[0], [f.created[0], "update", "a\nb"]);
  root.boundaryRequests.p1.active.pop(); root.battle.log.push("external");
  f.decoded[0].set.moves.push("owned");
  assert.equal(f.battles.get("cell-1").boundaryRequests.p1.active.length, 1);
  assert.deepEqual(f.decoded[0].log, []);
  assert.deepEqual(root.battle.sides[0].pokemon[0].set.moves, ["fixture"]);
  await f.release();
});

test("strict release observes destroy/EOF/pumps and invalidates scheduled callbacks", async () => {
  const f = fixture(); await f.clone();
  const battle = f.battles.get("cell-1"), oldEpoch = battle.readyEpoch;
  // Simulate a final drained chunk scheduling an old-shell boundary.
  const stream = f.created[0], end = stream.writeEnd.bind(stream);
  stream.writeEnd = async () => {
    await end(); const epoch = battle.readyEpoch; battle.terminalScheduled = true;
    setImmediate(() => { if (epoch === battle.readyEpoch && battle.terminalScheduled) f.events.push({type: "terminal"}); });
  };
  await f.release();
  assert.equal(f.events.at(-1).type, "clone_released");
  assert.equal(f.events.some(event => event.type === "terminal"), false);
  assert.ok(battle.readyEpoch > oldEpoch);
  assert.deepEqual(battle.boundaryRequests, {});
  assert.equal(f.battles.size, 0); assert.equal(stream.endCount, 1);
  assert.equal(stream.atEOF, true); assert.equal(f.decoded[0].field, null);
});

test("owned snapshot is synchronous read-only and binds clone, bytes, index and actual generation", async () => {
  const f = fixture(), root = snapshot(); await f.clone("cell-1", root);
  const ack = f.events[0], battle = f.battles.get("cell-1");
  const read = requestId => f.api.execute({type: "snapshot_owned_boundary", battleId: "cell-1",
    cloneId: ack.cloneId, snapshotJsonSha256: ack.snapshotJsonSha256, requestId},
    () => assert.fail("snapshot cannot call legacy or regenerate requests"));
  const before = JSON.stringify(battle.boundaryRequests);
  await read("root");
  const first = f.events.at(-1);
  assert.equal(first.type, "owned_boundary_snapshot"); assert.equal(first.snapshotIndex, 0);
  assert.equal(first.boundaryGeneration, 0); assert.equal(first.requestId, "root");
  assert.equal(first.currentSnapshotJsonSha256, hash(first.currentSnapshotJson));
  assert.deepEqual(JSON.parse(first.currentSnapshotJson), root);
  assert.equal(JSON.stringify(battle.boundaryRequests), before); assert.deepEqual(f.calls, []);
  await assert.rejects(read("root"), /unique request/);
  battle.boundaryGeneration++; battle.boundaryRequestGeneration = battle.boundaryGeneration;
  f.decoded[0].ended = true; battle.terminalScheduled = true;
  await read("terminal");
  assert.equal(f.events.at(-1).snapshotIndex, 1); assert.equal(f.events.at(-1).boundaryGeneration, 1);
  assert.deepEqual(JSON.parse(f.events.at(-1).currentSnapshotJson).boundaryRequests, {});
  await f.release(); await assert.rejects(read("after-release"), /live admitted/);
});

test("owned snapshot rejects binding drift and overlapping commands before serialization", async () => {
  const f = fixture(); await f.clone(); const ack = f.events[0];
  const base = {type: "snapshot_owned_boundary", battleId: "cell-1", cloneId: ack.cloneId,
    snapshotJsonSha256: ack.snapshotJsonSha256, requestId: "root"};
  for (const changed of [{cloneId: "wrong"}, {snapshotJsonSha256: "a".repeat(64)},
    {requestId: "invalid request"}, {extra: true}]) {
    await assert.rejects(f.api.execute({...base, ...changed}, () => assert.fail()), /mismatch|Exact/);
  }
  let done; const pending = f.api.execute({type: "choices", battleId: "cell-1"},
    () => new Promise(resolve => { done = resolve; }));
  await assert.rejects(f.api.execute(base, () => assert.fail()), /live admitted/);
  done(); await pending; await f.release();
  assert.equal(f.events.some(event => event.type === "owned_boundary_snapshot"), false);
});

test("missing current-generation requests and serialization failure never return an inferred snapshot", async () => {
  for (const mode of [{stale: true}, {missing: true}, {serializeError: true}]) {
    const f = fixture(mode); await f.clone(); const ack = f.events[0], battle = f.battles.get("cell-1");
    if (mode.stale) battle.boundaryGeneration++;
    if (mode.missing) delete battle.boundaryRequests.p2;
    await assert.rejects(f.api.execute({type: "snapshot_owned_boundary", battleId: "cell-1",
      cloneId: ack.cloneId, snapshotJsonSha256: ack.snapshotJsonSha256, requestId: "root"}, () => assert.fail()),
    /unavailable|serialization failure/);
    assert.equal(f.events.some(event => event.type === "owned_boundary_snapshot"), false);
    assert.equal(f.events.at(-1).type, "clone_failed"); assert.equal(f.created[0].endCount, 1);
    await assert.rejects(f.clone("new"), /failed/i);
  }
});

test("tokens remain tombstoned; delayed old shell facts never acquire new token", async () => {
  const f = fixture(); await f.clone("old");
  const old = f.created[0], send = f.decoded[0].send;
  await f.release("old"); await f.clone("new");
  old.channels.omniscient.emit(["|win|old winner"]);
  assert.equal(f.events.at(-1).battleId, "old");
  send("update", "old delayed update"); assert.equal(f.calls.at(-1)[0], old);
  await f.release("new");
  await assert.rejects(f.clone("old"), /cannot be reused/);
  assert.equal(f.created.length, 2);
});

test("absent, duplicate, mismatched release never produce successful acknowledgement", async () => {
  const f = fixture(); await assert.rejects(f.release(), /live admitted/);
  await f.clone();
  for (const changes of [{cloneId: "wrong"}, {snapshotJsonSha256: "a".repeat(64)}, {extra: true}]) {
    await assert.rejects(f.release("cell-1", changes), /mismatch|Exact/);
    assert.equal(f.events.filter(event => event.type === "clone_released").length, 0);
  }
  await f.release(); await assert.rejects(f.release(), /live admitted/);
  assert.equal(f.events.filter(event => event.type === "clone_released").length, 1);
});

test("legacy mutators, alias tokens, other shells and overlapping commands are rejected", async () => {
  const f = fixture(); await f.clone();
  for (const type of ["start", "restore", "end", "materialize", "snapshot", "choice"]) {
    await assert.rejects(f.api.execute({type, battleId: "cell-1"}, () => assert.fail("mutation")), /cannot mutate/);
  }
  await assert.rejects(f.api.execute({type: "restore", battleId: {toString: () => "cell-1"}}, () => assert.fail("alias")), /cannot mutate/);
  await assert.rejects(f.api.execute({type: "start", battleId: "other"}, () => assert.fail("other")), /Only the active/);
  await assert.rejects(f.clone("other"), /already active/);
  let done; const pending = f.api.execute({type: "choices", battleId: "cell-1"}, () => new Promise(resolve => { done = resolve; }));
  await assert.rejects(f.release(), /live admitted/);
  await assert.rejects(f.api.execute({type: "reseed", battleId: "cell-1"}, () => {}), /cannot mutate/);
  done(); await pending; await f.release();
});

test("clone refuses terminal, malformed, nonactionable and duplicate attempts", async () => {
  for (const kind of ["terminal", "ended", "win", "requests", "structure"]) {
    const f = fixture(), root = snapshot();
    if (kind === "terminal") root.terminalScheduled = true;
    if (kind === "ended") root.battle.ended = true;
    if (kind === "win") root.battle.log.push("|win|fixture");
    if (kind === "requests") root.boundaryRequests = {};
    if (kind === "structure") root.battle.sides = null;
    await assert.rejects(f.clone("bad", root));
    assert.equal(f.events.some(event => event.type === "cloned"), false);
    assert.equal(f.events.at(-1).type, "clone_failed");
    assert.equal(f.events.at(-1).cleanupReleased, true);
    assert.equal(f.battles.size, 0);
    await assert.rejects(f.clone("bad")); await assert.rejects(f.clone("new"), /failed/i);
  }
});

test("decoder/restart failures tear down once, preserve primary and poison admission", async () => {
  for (const mode of [{decodeError: true}, {restartError: true}, {decodedTerminal: true}, {decodeError: true, endError: true}]) {
    const f = fixture(mode);
    await assert.rejects(f.clone(), mode.decodeError ? /decode failure/ : mode.restartError ? /restart failure/ : /terminal/);
    assert.equal(f.created[0].endCount, 1);
    assert.equal(f.events.some(event => event.type === "cloned"), false);
    assert.equal(f.events.at(-1).cleanupReleased, !mode.endError);
    await assert.rejects(f.clone("new"), /failed/i);
  }
});

test("teardown error, pump error, missing destroy/EOF or timeout cannot earn release", async () => {
  for (const mode of [{endError: true}, {pumpError: true}, {noDestroy: true}, {noEOF: true}, {hangEnd: true}, {hangPump: true}]) {
    const f = fixture(mode); await f.clone(); await assert.rejects(f.release());
    assert.equal(f.created[0].endCount, 1); // no implicit retry of failed teardown
    assert.equal(f.events.some(event => event.type === "clone_released"), false);
    assert.equal(f.events.at(-1).cleanupReleased, false);
    await assert.rejects(f.release(), /live admitted|Failed owned/);
    await assert.rejects(f.clone("new"), /failed/i);
    await assert.rejects(f.api.shutdown(), /Unsettled|Failed owned/);
  }
});

test("step failure cleans up; primary survives secondary teardown error", async () => {
  for (const mode of [{}, {endError: true}]) {
    const f = fixture(mode); await f.clone();
    const primary = new Error("actual synthetic step failed");
    await assert.rejects(f.api.execute({type: "choices", battleId: "cell-1"}, () => { throw primary; }), error => {
      assert.equal(error, primary); if (mode.endError) assert.match(error.cause.message, /teardown/); return true;
    });
    assert.equal(f.created[0].endCount, 1); assert.equal(f.events.at(-1).cleanupReleased, !mode.endError);
    await assert.rejects(f.clone("new"), /failed/i);
  }
});

test("graceful shutdown strictly releases owned shell; historical dispatch stays opt-in", async () => {
  const f = fixture(); let legacyCalls = 0;
  await f.api.execute({type: "start", battleId: "legacy"}, () => legacyCalls++);
  assert.equal(legacyCalls, 1); await f.clone(); await f.api.shutdown();
  assert.equal(f.events.at(-1).type, "clone_released"); await f.api.shutdown();
  await tick(); assert.equal(f.battles.size, 0);
});

test("previous legacy tokens cannot be promoted into fresh outcome shells", async () => {
  const f = fixture(); let legacyCalls = 0;
  const late = () => f.events.push({type: "stream", battleId: "legacy", stream: "omniscient", lines: ["|win|late"]});
  for (const type of ["start", "end", "start", "end"]) {
    await f.api.execute({type, battleId: "legacy"}, () => legacyCalls++);
  }
  assert.equal(legacyCalls, 4); // historical reuse behavior unchanged
  await assert.rejects(f.clone("legacy"), /cannot be reused/);
  await f.clone("fresh"); late(); assert.equal(f.events.at(-1).battleId, "legacy");
  await f.release("fresh");
});

test("failed owned subsystem cannot continue through historical dispatch or exit success", async () => {
  const f = fixture({decodeError: true}); await assert.rejects(f.clone());
  for (const type of ["start", "restore", "end", "reseed", "choices", "snapshot"]) {
    await assert.rejects(f.api.execute({type, battleId: "legacy"}, () => assert.fail("fallback")), /cannot fall back/);
  }
  await assert.rejects(f.api.execute({type: "close"}, () => f.api.shutdown()), /graceful close/);
});

// Deliberately synthetic stand-ins. Launches the REAL bridge entrypoint and
// its listener/command code without importing Showdown or generating a root.
const fakeRuntime = `
class Channel {
  constructor(write) { this.write = write; this.items = []; this.readers = []; }
  push(item) { if (this.readers.length) this.readers.shift()(item); else this.items.push(item); }
  async *[Symbol.asyncIterator]() { while (true) {
    const item = this.items.length ? this.items.shift() : await new Promise(resolve => this.readers.push(resolve));
    if (item === null) return; yield item;
  } }
}
class BattleStream {
  constructor() { this.atEOF = false; this.battle = null; }
  pushMessage(type, data) { this.channels.omniscient.push(data); }
  async write(data) {
    if (data.startsWith('>reseed ')) return;
    if (data.startsWith('>p2 ')) return;
    if (data.startsWith('>p1 ')) { this.battle.ended = true; this.battle.send('update', ['|win|Fixture One']); return; }
    throw new Error('Forbidden synthetic source/start command: ' + data);
  }
  async writeEnd() {
    this.atEOF = true;
    if (this.battle) { this.battle.field = null; this.battle.queue = null; this.battle.sides = [null]; }
    for (const channel of Object.values(this.channels)) channel.push(null);
    if (process.env.SYNTHETIC_END_ERROR === '1') throw new Error('synthetic entrypoint teardown failure');
  }
}
function getPlayerStreams(stream) {
  stream.channels = Object.fromEntries(['omniscient', 'p1', 'p2'].map(seat => [seat,
    new Channel(data => stream.write(seat === 'omniscient' ? data : '>' + seat + ' ' + data))]));
  return stream.channels;
}
module.exports = {BattleStream, getPlayerStreams, Battle: class { constructor() { throw new Error('Forbidden fresh Battle'); } },
  Teams: {getGenerator() { throw new Error('Forbidden source generation'); }}};
`;
const fakeState = `module.exports = {State: {serializeBattle(battle) {
  return {...structuredClone(battle.serialized), ended: battle.ended};
}, deserializeBattle(input) {
  return {field: {}, queue: {}, sides: [{}], ended: false, log: input.log, serialized: structuredClone(input),
    restart(send) { this.send = send; }};
}}};`;

async function entrypoint(t, teardownError = false, sourceFixture = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pokezero-synthetic-bridge-"));
  const sim = path.join(root, "dist", "sim"); fs.mkdirSync(sim, {recursive: true});
  fs.writeFileSync(path.join(sim, "index.js"), fakeRuntime);
  fs.writeFileSync(path.join(sim, "state.js"), fakeState);
  let bridge = fileURLToPath(new URL("../scripts/battle_bridge.mjs", import.meta.url));
  if (sourceFixture) {
    const directory = path.join(root, "authored-source-bridge"); fs.mkdirSync(directory);
    bridge = path.join(directory, "battle_bridge.mjs");
    fs.copyFileSync(fileURLToPath(new URL("./source_bridge_fixture.mjs", import.meta.url)), bridge);
    for (const name of ["battle_bridge_owned_snapshot.mjs", "battle_bridge_owned_continuation.mjs"])
      fs.copyFileSync(fileURLToPath(new URL("../scripts/" + name, import.meta.url)), path.join(directory, name));
    const clones = path.join(directory, "authored-clones"); fs.mkdirSync(clones);
    fs.writeFileSync(path.join(clones, "index.js"), fakeRuntime);
    fs.writeFileSync(path.join(clones, "state.js"), fakeState);
    fs.writeFileSync(path.join(directory, "source-settings.json"), JSON.stringify({rounds:1,
      mode:"complete",snapshotTransport:true,ownedContinuationTransport:true}));
  }
  const child = spawn(process.execPath, [bridge, "--showdown-root", root], {
    env: {PATH: process.env.PATH, SYNTHETIC_END_ERROR: teardownError ? "1" : "0"}, stdio: ["pipe", "pipe", "pipe"]});
  let stderr = "", pending;
  const events = [], queue = [];
  child.stderr.on("data", chunk => stderr += chunk);
  const exited = new Promise(resolve => child.once("exit", (code, signal) => resolve({code, signal})));
  readline.createInterface({input: child.stdout}).on("line", line => {
    const event = JSON.parse(line); events.push(event); queue.push(event);
    if (pending) { const resolve = pending; pending = null; resolve(); }
  });
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    await exited;
    // Only the exact disposable synthetic fixture allocated by this test.
    fs.rmSync(root, {recursive: true});
  });
  async function next(type, token) {
    const deadline = Date.now() + 3000;
    while (true) {
      while (queue.length) {
        const event = queue.shift();
        if (event.type === type && (token === undefined || event.battleId === token)) return event;
        if (event.type === "error" && type !== "error") assert.fail(event.message);
      }
      assert.ok(Date.now() < deadline, "Synthetic bridge acknowledgement timeout: " + stderr);
      let timer;
      await new Promise(resolve => { pending = resolve; timer = setTimeout(resolve, 20); });
      clearTimeout(timer); pending = null;
    }
  }
  return {child, exited, events, next, send: command => child.stdin.write(JSON.stringify(command) + "\n")};
}

async function exerciseEntrypointClones(t, sourceFixture = false) {
  const f = await entrypoint(t, false, sourceFixture), encoded = JSON.stringify(snapshot());
  for (const token of ["first", "second"]) {
    f.send({type: "clone_snapshot", battleId: token, cloneId: token, snapshotJson: encoded});
    const clone = await f.next("cloned", token);
    assert.equal(clone.snapshotJsonSha256, hash(encoded));
    f.send({type: "snapshot_owned_boundary", battleId: token, cloneId: token,
      snapshotJsonSha256: clone.snapshotJsonSha256, requestId: "root"});
    const root = await f.next("owned_boundary_snapshot", token);
    assert.equal(root.currentSnapshotJsonSha256, hash(root.currentSnapshotJson));
    assert.equal(root.boundaryGeneration, 0); assert.equal(root.snapshotIndex, 0);
    f.send({type: "reseed", battleId: token, seed: "1,2,3,4"});
    assert.equal((await f.next("reseeded", token)).seed, "1,2,3,4");
    f.send({type: "choices", battleId: token, choices: {p1: "move 1"}});
    await f.next("choice_ack", token); await f.next("terminal", token);
    f.send({type: "snapshot_owned_boundary", battleId: token, cloneId: token,
      snapshotJsonSha256: clone.snapshotJsonSha256, requestId: "terminal"});
    const terminal = await f.next("owned_boundary_snapshot", token);
    assert.equal(terminal.boundaryGeneration, 1); assert.equal(terminal.snapshotIndex, 1);
    assert.equal(JSON.parse(terminal.currentSnapshotJson).battle.ended, true);
    assert.ok(f.events.some(event => event.battleId === token && event.lines?.includes("|win|Fixture One")));
    f.send({type: "release_clone", battleId: token, cloneId: token, snapshotJsonSha256: clone.snapshotJsonSha256});
    const release = await f.next("clone_released", token);
    assert.equal(release.cloneId, token);
    const ends = f.events.filter(event => event.type === "stream_end" && event.battleId === token);
    assert.deepEqual(ends.map(event => event.stream).sort(), ["omniscient", "p1", "p2"]);
    assert.ok(ends.every(event => f.events.indexOf(event) < f.events.indexOf(release)));
  }
  f.send({type: "close"}); await f.next("closed");
  assert.deepEqual(await f.exited, {code: 0, signal: null});
}

test("real bridge entrypoint clones/steps/releases two synthetic shells and drains actual pumps", async t => {
  await exerciseEntrypointClones(t);
});

test("authored source fixture delegates isolated clones to the original strict helper", async t => {
  await exerciseEntrypointClones(t, true);
});

test("authored source fixture keeps its source lifecycle with clone transport enabled", async t => {
  const f = await entrypoint(t, false, true), token = "authored-source";
  f.send({type:"start",battleId:token,formatid:"gen3randombattle",seed:"1,2,3,4"});
  await f.next("started", token);
  assert.deepEqual((await f.next("ready", token)).requested, ["p1","p2"]);
  f.send({type:"snapshot_actionable_boundary",battleId:token});
  const captured = await f.next("snapshot_actionable_boundary", token);
  assert.equal(captured.available, true);
  assert.equal(captured.snapshot.terminalScheduled, false);
  f.send({type:"choices",battleId:token,choices:{p1:"move 1",p2:"move 1"}});
  await f.next("terminal", token);
  f.send({type:"end",battleId:token});
  await f.next("ended", token);
  assert.deepEqual(f.events.filter(event => event.type === "stream_end").map(event => event.stream).sort(),
    ["omniscient","p1","p2"]);
  assert.equal(f.events.some(event => event.type === "cloned"), false);
  f.send({type:"close"}); await f.next("closed");
  assert.deepEqual(await f.exited, {code:0,signal:null});
});

test("authored source fixture cannot bank a release after actual clone teardown failure", async t => {
  const f = await entrypoint(t, true, true);
  f.send({type:"clone_snapshot",battleId:"failed",cloneId:"failed",snapshotJson:JSON.stringify(snapshot())});
  const clone = await f.next("cloned", "failed");
  f.send({type:"release_clone",battleId:"failed",cloneId:"failed",snapshotJsonSha256:clone.snapshotJsonSha256});
  assert.equal((await f.next("clone_failed", "failed")).cleanupReleased, false);
  assert.deepEqual(await f.exited, {code:1,signal:null});
  assert.equal(f.events.some(event => event.type === "clone_released" || event.type === "closed"), false);
});

test("real bridge entrypoint never converts strict teardown failure or poisoned EOF to exit zero", async t => {
  const f = await entrypoint(t, true);
  f.send({type: "clone_snapshot", battleId: "first", cloneId: "first", snapshotJson: JSON.stringify(snapshot())});
  const clone = await f.next("cloned");
  f.send({type: "release_clone", battleId: "first", cloneId: "first", snapshotJsonSha256: clone.snapshotJsonSha256});
  assert.equal((await f.next("clone_failed")).cleanupReleased, false);
  assert.match((await f.next("error")).message, /teardown failure/);
  f.child.stdin.end();
  assert.deepEqual(await f.exited, {code: 1, signal: null});
  assert.equal(f.events.some(event => event.type === "clone_released" || event.type === "closed"), false);
});
