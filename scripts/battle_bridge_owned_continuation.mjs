// Prospective outcome shells only. No start/reset, source generation or engine
// equivalence claim. Dependencies are injected so tests need no simulator/game.
import crypto from "node:crypto";
import {copyDeserializationAliases} from "./battle_bridge_owned_snapshot.mjs";

const seats = ["omniscient", "p1", "p2"];
const immediate = () => new Promise(resolve => setImmediate(resolve));
const sha = text => crypto.createHash("sha256").update(text, "utf8").digest("hex");
const identity = value => typeof value === "string" && /^[A-Za-z0-9._:-]{1,200}$/.test(value);
function require(condition, message) {
  if (!condition) throw new Error(message);
}
function exact(command, keys) {
  require(Object.keys(command).sort().join(",") === keys.sort().join(","), "Exact owned lifecycle command required.");
}

export function createOwnedContinuations({BattleStream, getPlayerStreams, State,
  battles, newBattleState, listenToStream, actionableRequestedPlayers, emit,
  cleanupTimeoutMs = 10000,
  destroyed = battle => battle.field === null && battle.queue === null
    && Array.isArray(battle.sides) && battle.sides.every(side => side === null)}) {
  const used = new Set(); // Tombstones survive release AND failed admission.
  const legacyTokens = new Set(); // Old listeners can outlive legacy Map entries.
  const records = new Map();
  let active = null;
  let poisoned = false;
  require(Number.isInteger(cleanupTimeoutMs) && cleanupTimeoutMs > 0, "Positive owned cleanup timeout required.");

  async function boundedCleanup(operation) {
    let timer;
    try {
      return await Promise.race([operation(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("Owned cleanup timed out; no release acknowledgement.")), cleanupTimeoutMs);
      })]);
    } finally { clearTimeout(timer); }
  }

  function binding(record) {
    return {battleId: record.token, cloneId: record.cloneId,
      snapshotJsonSha256: record.snapshotSha};
  }

  async function teardown(record) {
    const battle = record.battle;
    record.phase = "releasing";
    // Invalidate scheduled callbacks both before and after draining stream
    // pumps: final buffered chunks may schedule another old-shell callback.
    const invalidate = () => {
      battle.readyEpoch += 1;
      battle.readyScheduled = false;
      battle.terminalScheduled = false;
    };
    invalidate();
    const simulator = battle.battleStream?.battle;
    const results = await boundedCleanup(async () => {
      let failure;
      try {
        if (battle.battleStream) await battle.battleStream.writeEnd();
      } catch (error) { failure = error; }
      // Drain pumps even after a writeEnd error, but never wait indefinitely.
      const results = await Promise.allSettled(record.pumps);
      invalidate();
      await immediate();
      if (failure) throw failure;
      return results;
    });
    require(results.every(result => result.status === "fulfilled"), "Owned stream pump failed during release.");
    require(!simulator || destroyed(simulator), "Owned simulator destruction was not observed.");
    require(!battle.battleStream || battle.battleStream.atEOF === true, "Owned stream EOF was not observed.");
    battle.streams = null;
    battle.battleStream = null;
    battle.boundaryRequests = {};
    battles.delete(record.token);
    record.phase = "released";
    record.battle = null;
    record.pumps = []; // Keep identity tombstones, not old private state graphs.
    if (active === record) active = null;
  }

  async function fail(record, primary, cleanupAttempted = false) {
    poisoned = true;
    let cleanupFailure;
    try {
      if (cleanupAttempted) throw primary;
      await teardown(record);
    } catch (error) {
      record.phase = "failed";
      cleanupFailure = error;
    }
    emit({type: "clone_failed", ...binding(record),
      cleanupReleased: !cleanupFailure,
      message: primary.message,
      cleanupError: cleanupFailure ? cleanupFailure.message : null});
    // Preserve the original failure; cleanup is additional evidence, never a
    // replacement success or permission to retry the token/cell/pool.
    if (cleanupFailure && cleanupFailure !== primary) primary.cause = cleanupFailure;
    throw primary;
  }

  async function clone(command) {
    exact(command, ["type", "battleId", "cloneId", "snapshotJson"]);
    require(identity(command.battleId) && identity(command.cloneId), "Explicit owned token and clone identity required.");
    require(!poisoned && active === null, "Owned continuation subsystem is failed or already active.");
    require(!used.has(command.battleId) && !legacyTokens.has(command.battleId)
      && !battles.has(command.battleId), "Owned token cannot be reused.");
    used.add(command.battleId);
    const battle = newBattleState(command.battleId);
    const record = {token: command.battleId, cloneId: command.cloneId,
      snapshotSha: typeof command.snapshotJson === "string" ? sha(command.snapshotJson) : null,
      battle, pumps: [], phase: "cloning", busy: true,
      boundaryRequestsSeen: new Set(), snapshotIndex: 0, initialGeneration: null};
    records.set(record.token, record);
    active = record;
    battles.set(record.token, battle);
    try {
      require(typeof command.snapshotJson === "string", "Received snapshot JSON bytes required.");
      const snapshot = JSON.parse(command.snapshotJson);
      require(snapshot && typeof snapshot === "object" && !Array.isArray(snapshot)
        && Object.keys(snapshot).sort().join(",") === "battle,boundaryRequests,terminalScheduled"
        && snapshot.battle && snapshot.battle.ended !== true && snapshot.terminalScheduled === false
        && !snapshot.battle.log?.some(line => typeof line === "string"
          && (line.startsWith("|win|") || line === "|tie" || line.startsWith("|tie|"))),
      "A nonterminal frozen bridge snapshot is required.");
      // This is an empty stream, NOT >start. Restore only the supplied frozen
      // serialized world. The send closure is permanently tied to this stream.
      const stream = new BattleStream({keepAlive: true});
      battle.battleStream = stream;
      battle.streams = getPlayerStreams(stream);
      for (const seat of seats) {
        const pump = listenToStream(battle, seat, battle.streams[seat], true);
        // Keep the original rejection for strict release, but attach a handler
        // immediately so a late asynchronous error cannot go unobserved.
        pump.catch(() => {});
        record.pumps.push(pump);
      }
      stream.battle = State.deserializeBattle(copyDeserializationAliases(snapshot.battle));
      stream.battle.restart((type, data) => {
        stream.pushMessage(type, Array.isArray(data) ? data.join("\n") : data);
      });
      require(stream.battle.ended !== true, "Decoded outcome root is terminal.");
      battle.boundaryRequests = structuredClone(snapshot.boundaryRequests);
      battle.boundaryGeneration += 1;
      battle.boundaryRequestGeneration = battle.boundaryGeneration;
      record.initialGeneration = battle.boundaryGeneration;
      const requested = actionableRequestedPlayers(battle);
      require(requested.length > 0, "Cloned root has no actionable frozen request.");
      record.phase = "live";
      record.busy = false;
      emit({type: "cloned", ...binding(record), requested});
    } catch (error) { await fail(record, error); }
  }

  async function release(command) {
    exact(command, ["type", "battleId", "cloneId", "snapshotJsonSha256"]);
    const record = records.get(command.battleId);
    require(record && active === record && record.phase === "live" && !record.busy,
      "Release requires a live admitted owned clone.");
    require(command.cloneId === record.cloneId && command.snapshotJsonSha256 === record.snapshotSha,
      "Owned release identity/snapshot mismatch.");
    record.busy = true;
    try {
      await teardown(record);
      emit({type: "clone_released", ...binding(record)});
    } catch (error) { await fail(record, error, true); }
  }

  async function snapshotBoundary(command) {
    exact(command, ["type", "battleId", "cloneId", "snapshotJsonSha256", "requestId"]);
    const record = records.get(command.battleId);
    require(!poisoned && record && active === record && record.phase === "live" && !record.busy,
      "Snapshot requires a live admitted owned clone.");
    require(command.cloneId === record.cloneId && command.snapshotJsonSha256 === record.snapshotSha
      && identity(command.requestId) && !record.boundaryRequestsSeen.has(command.requestId),
      "Owned snapshot identity or unique request mismatch.");
    // Reserve before serialization. An ambiguous read cannot be retried.
    record.boundaryRequestsSeen.add(command.requestId);
    record.busy = true;
    try {
      const battle = record.battle, simulator = battle.battleStream?.battle;
      require(simulator && Number.isSafeInteger(battle.boundaryGeneration)
        && battle.boundaryGeneration >= record.initialGeneration,
        "Actual owned simulator generation required.");
      const terminal = simulator.ended === true;
      require(terminal || (battle.boundaryRequestGeneration === battle.boundaryGeneration
        && ["p1", "p2"].every(seat => battle.boundaryRequests[seat])
        && actionableRequestedPlayers(battle).length > 0),
        "Current paired owned boundary requests unavailable.");
      // Synchronous serialization only: no settling, request regeneration,
      // cache replacement, restore, choice submission or RNG mutation.
      const currentSnapshotJson = JSON.stringify({battle: State.serializeBattle(simulator),
        boundaryRequests: terminal ? {} : battle.boundaryRequests,
        terminalScheduled: battle.terminalScheduled});
      emit({type: "owned_boundary_snapshot", ...binding(record), requestId: command.requestId,
        snapshotIndex: record.snapshotIndex++, boundaryGeneration: battle.boundaryGeneration - record.initialGeneration,
        currentSnapshotJson, currentSnapshotJsonSha256: sha(currentSnapshotJson)});
    } catch (error) { await fail(record, error); }
    finally { record.busy = false; }
  }

  async function execute(command, legacy) {
    require(!poisoned || command.type === "close", "Failed owned subsystem cannot fall back to legacy commands.");
    if (command.type === "clone_snapshot") return clone(command);
    if (command.type === "release_clone") return release(command);
    if (command.type === "snapshot_owned_boundary") return snapshotBoundary(command);
    const token = command.battleId == null ? "default" : String(command.battleId);
    const record = records.get(token);
    if (!record) {
      require(active === null || command.type === "close", "Only the active owned clone may receive commands.");
      // Preserve ordinary legacy reuse, but exclude every previously observed
      // legacy token from the prospective no-reuse admission path forever.
      legacyTokens.add(token);
      return legacy();
    }
    require(active === record && record.phase === "live" && !record.busy
      && ["choices", "reseed"].includes(command.type), "Command cannot mutate an owned clone or tombstone.");
    record.busy = true;
    try { return await legacy(); }
    catch (error) { return await fail(record, error); }
    finally { record.busy = false; }
  }

  async function shutdown() {
    require(!poisoned, "Failed owned subsystem cannot report graceful close success.");
    require(!active || (active.phase === "live" && !active.busy), "Unsettled owned clone blocks graceful close.");
    if (active) await release({type: "release_clone", ...binding(active)});
  }

  return {execute, shutdown};
}
