#!/usr/bin/env node

import { createRequire } from "node:module";
import crypto from "node:crypto";
import path from "node:path";
import process from "node:process";
import readline from "node:readline";

const require = createRequire(import.meta.url);

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--showdown-root") {
      args.showdownRoot = argv[index + 1];
      index += 1;
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const showdownRoot = args.showdownRoot || process.env.POKEZERO_SHOWDOWN_ROOT;
if (!showdownRoot) {
  emit({ type: "error", message: "Missing --showdown-root or POKEZERO_SHOWDOWN_ROOT." });
  process.exit(2);
}

let BattleStream;
let getPlayerStreams;
let State;
try {
  ({ BattleStream, getPlayerStreams } = require(path.join(showdownRoot, "dist", "sim", "index.js")));
  ({ State } = require(path.join(showdownRoot, "dist", "sim", "state.js")));
} catch (error) {
  emit({
    type: "error",
    message: `Unable to load Pokemon Showdown simulator from ${showdownRoot}: ${error.message}`,
  });
  process.exit(2);
}

// One bridge process can host many battles concurrently, keyed by battleId. A single live process
// serves both a warm pool (reused across battles, never exiting between games) and batched
// collection (many battles stepped per round-trip). Every command and emitted event carries its
// battleId, so the driver routes events and ignores stale events from a finished battle on a
// reused process. battleId is optional; it defaults to "default" for single-battle callers.
const DEFAULT_BATTLE_ID = "default";
const battles = new Map();
const searchSnapshots = new Map();
let nextSearchSnapshotId = 1;

function emit(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function battleIdOf(command) {
  return command.battleId == null ? DEFAULT_BATTLE_ID : String(command.battleId);
}

function newBattleState(battleId) {
  return {
    battleId,
    battleStream: null,
    streams: null,
    boundaryRequests: {},
    readyScheduled: false,
    terminalScheduled: false,
  };
}

function emitStreamChunk(battle, stream, chunk) {
  const lines = String(chunk)
    .split("\n")
    .filter(line => line.length > 0);
  if (lines.length > 0) {
    emit({ type: "stream", battleId: battle.battleId, stream, lines });
    recordBoundaryLines(battle, stream, lines);
  }
}

function listenToStream(battle, name, stream) {
  void (async () => {
    try {
      for await (const chunk of stream) {
        emitStreamChunk(battle, name, chunk);
      }
      emit({ type: "stream_end", battleId: battle.battleId, stream: name });
    } catch (error) {
      emit({ type: "error", battleId: battle.battleId, stream: name, message: error.message });
    }
  })();
}

function deriveSeed(seed, label) {
  const digest = crypto.createHash("sha256").update(`${seed}:${label}`).digest();
  const parts = [];
  for (let index = 0; index < 8; index += 2) {
    parts.push(digest.readUInt16BE(index));
  }
  return parts.join(",");
}

function recordBoundaryLines(battle, stream, lines) {
  if (stream === "omniscient" && lines.some(isTerminalLine)) {
    scheduleTerminal(battle);
    return;
  }
  if (!["p1", "p2"].includes(stream)) return;
  for (const line of lines) {
    if (!line.startsWith("|request|")) continue;
    const request = JSON.parse(line.slice("|request|".length));
    const sideId = request?.side?.id;
    if (sideId === stream) {
      battle.boundaryRequests[stream] = request;
    }
  }
  if (battle.boundaryRequests.p1 && battle.boundaryRequests.p2) {
    scheduleReady(battle);
  }
}

function nodeProcMs(battle) {
  // Milliseconds of node-side processing for this step (compute + node-side serialization +
  // async event emission): from receiving the choices command to detecting the boundary.
  if (battle.tRecv == null) return null;
  return Number(process.hrtime.bigint() - battle.tRecv) / 1e6;
}

function scheduleReady(battle) {
  if (battle.readyScheduled || battle.terminalScheduled) return;
  battle.readyScheduled = true;
  const procMs = nodeProcMs(battle);
  setImmediate(() => {
    emit({
      type: "ready",
      battleId: battle.battleId,
      requested: ["p1", "p2"].filter(player => isActionableRequest(battle.boundaryRequests[player])),
      nodeProcMs: procMs,
    });
  });
}

function scheduleTerminal(battle) {
  if (battle.terminalScheduled) return;
  battle.terminalScheduled = true;
  const procMs = nodeProcMs(battle);
  setImmediate(() => {
    emit({ type: "terminal", battleId: battle.battleId, nodeProcMs: procMs });
  });
}

function isTerminalLine(line) {
  return line.startsWith("|win|") || line === "|tie" || line.startsWith("|tie|");
}

function isActionableRequest(request) {
  if (!request || request.wait || request.teamPreview) return false;
  if (Array.isArray(request.forceSwitch) && request.forceSwitch.some(Boolean)) return true;
  return Array.isArray(request.active) && request.active.length > 0;
}

async function teardownBattle(battle) {
  if (battle && battle.streams) {
    try {
      await battle.streams.omniscient.writeEnd();
    } catch (error) {
      // best-effort teardown; the stream may already be ending
    }
    battle.streams = null;
  }
}

async function startBattle(command) {
  const battleId = battleIdOf(command);
  // Reusing a live process: tear down any prior battle under this id so a fresh battle can begin
  // without exiting the process.
  if (battles.has(battleId)) {
    await teardownBattle(battles.get(battleId));
  }
  const battle = newBattleState(battleId);
  battles.set(battleId, battle);
  const battleStream = new BattleStream({ keepAlive: true });
  battle.battleStream = battleStream;
  battle.streams = getPlayerStreams(battleStream);
  for (const name of ["omniscient", "p1", "p2"]) {
    listenToStream(battle, name, battle.streams[name]);
  }

  const formatid = command.formatid || "gen3randombattle";
  const seed = command.seed;
  const players = command.players || {};
  const startOptions = { formatid, strictChoices: true };
  if (seed) startOptions.seed = seed;
  const p1 = normalizePlayerOptions(players.p1, "PokeZero p1");
  const p2 = normalizePlayerOptions(players.p2, "PokeZero p2");
  if (seed) {
    p1.seed = deriveSeed(seed, "p1");
    p2.seed = deriveSeed(seed, "p2");
  }
  await battle.streams.omniscient.write(
    `>start ${JSON.stringify(startOptions)}\n` +
      `>player p1 ${JSON.stringify(p1)}\n` +
      `>player p2 ${JSON.stringify(p2)}`
  );
  emit({ type: "started", battleId, formatid, seed: seed || null });
}

function snapshotBattle(command) {
  const battle = requireBattle(command);
  if (!battle.battleStream?.battle) {
    throw new Error(`No simulator state for battleId ${battle.battleId}.`);
  }
  emit({
    type: "snapshot",
    battleId: battle.battleId,
    snapshot: {
      // Keep the engine state API explicit: snapshots are cloned simulator worlds, never
      // protocol replays and never a serialization of the live decision environment.
      battle: State.serializeBattle(battle.battleStream.battle),
      boundaryRequests: battle.boundaryRequests,
      terminalScheduled: battle.terminalScheduled,
    },
  });
}

function snapshotSearchBattle(command) {
  const battle = requireBattle(command);
  if (!battle.battleStream?.battle) {
    throw new Error(`No simulator state for battleId ${battle.battleId}.`);
  }
  const snapshotId = `search-${nextSearchSnapshotId}`;
  nextSearchSnapshotId += 1;
  // This is only called from a separate, belief-sampled search environment. Keep the
  // serialized state in Node so each visit sends a tiny handle, never a live-battle payload.
  searchSnapshots.set(snapshotId, {
    // Match the existing JSON bridge contract before retaining the snapshot. In particular,
    // State.deserializeBattle expects the JSON-normalized form that generic snapshots receive
    // after their round trip through Python, not JavaScript-only undefined-valued properties.
    battle: jsonSnapshotClone(State.serializeBattle(battle.battleStream.battle)),
    boundaryRequests: jsonSnapshotClone(battle.boundaryRequests),
    terminalScheduled: battle.terminalScheduled,
  });
  emit({ type: "search_snapshot", battleId: battle.battleId, snapshotId });
}

function jsonSnapshotClone(value) {
  const encoded = JSON.stringify(value);
  return encoded === undefined ? null : JSON.parse(encoded);
}

function restoreSerializedBattle(battle, snapshot, { cloneSnapshot = false } = {}) {
  const send = battle.battleStream.battle.send;
  // Search restores clone the retained serialized world for every root visit. Keep the generic
  // snapshot path byte-for-byte compatible with its prior bridge contract.
  battle.battleStream.battle = State.deserializeBattle(
    cloneSnapshot ? structuredClone(snapshot.battle) : snapshot.battle
  );
  battle.battleStream.battle.restart(send);
  battle.boundaryRequests =
    snapshot.boundaryRequests && typeof snapshot.boundaryRequests === "object"
      ? cloneSnapshot
        ? structuredClone(snapshot.boundaryRequests)
        : snapshot.boundaryRequests
      : {};
  battle.readyScheduled = false;
  battle.terminalScheduled = Boolean(snapshot.terminalScheduled);
  battle.tRecv = null;
}

function restoreBattle(command) {
  const battle = requireBattle(command);
  const snapshot = command.snapshot;
  if (!snapshot || typeof snapshot !== "object" || !snapshot.battle) {
    throw new Error("Restore requires a battle snapshot.");
  }
  if (!battle.battleStream?.battle) {
    throw new Error(`No simulator state for battleId ${battle.battleId}.`);
  }
  restoreSerializedBattle(battle, snapshot);
  emit({
    type: "restored",
    battleId: battle.battleId,
    requested: ["p1", "p2"].filter(player => isActionableRequest(battle.boundaryRequests[player])),
  });
}

function restoreSearchBattle(command) {
  const battle = requireBattle(command);
  const snapshotId = command.snapshotId;
  if (typeof snapshotId !== "string" || snapshotId.length === 0) {
    throw new Error("Search restore requires a snapshotId.");
  }
  const snapshot = searchSnapshots.get(snapshotId);
  if (!snapshot) {
    throw new Error(`Unknown search snapshot ${snapshotId}.`);
  }
  if (!battle.battleStream?.battle) {
    throw new Error(`No simulator state for battleId ${battle.battleId}.`);
  }
  restoreSerializedBattle(battle, snapshot, { cloneSnapshot: true });
  emit({
    type: "search_restored",
    battleId: battle.battleId,
    requested: ["p1", "p2"].filter(player => isActionableRequest(battle.boundaryRequests[player])),
  });
}

function releaseSearchSnapshot(command) {
  requireBattle(command);
  const snapshotId = command.snapshotId;
  if (typeof snapshotId !== "string" || snapshotId.length === 0) {
    throw new Error("Search snapshot release requires a snapshotId.");
  }
  const released = searchSnapshots.delete(snapshotId);
  emit({
    type: "search_snapshot_released",
    battleId: battleIdOf(command),
    snapshotId,
    released,
  });
}

function materializeBattle(command) {
  const battle = requireBattle(command);
  const publicState = command.publicState;
  if (!publicState || typeof publicState !== "object") {
    throw new Error("Materialize requires a publicState object.");
  }
  if (!battle.battleStream?.battle) {
    throw new Error(`No simulator state for battleId ${battle.battleId}.`);
  }
  // This template belongs to the already belief-sampled search world. We construct a new
  // public branch-point payload from it, then let Showdown deserialize that payload directly.
  const snapshot = State.serializeBattle(battle.battleStream.battle);
  applyPublicState(snapshot, publicState);
  const send = battle.battleStream.battle.send;
  battle.battleStream.battle = State.deserializeBattle(snapshot);
  battle.battleStream.battle.restart(send);
  battle.boundaryRequests = boundaryRequestsFromBattle(battle.battleStream.battle);
  battle.readyScheduled = false;
  battle.terminalScheduled = false;
  battle.tRecv = null;
  emit({
    type: "materialized",
    battleId: battle.battleId,
    boundaryRequests: battle.boundaryRequests,
    requested: ["p1", "p2"].filter(player => isActionableRequest(battle.boundaryRequests[player])),
  });
}

function applyPublicState(snapshot, publicState) {
  if (!Number.isInteger(publicState.turn) || publicState.turn < 1) {
    throw new Error("Materialize requires a positive integer turn.");
  }
  if (!publicState.sides || typeof publicState.sides !== "object") {
    throw new Error("Materialize requires public state for both sides.");
  }
  if (publicState.selfBenchedMoveHistory) {
    throw new Error("Materialize cannot reconstruct spent PP for a benched acting Pokemon.");
  }
  if (hasEntries(publicState.futureSight)) {
    throw new Error("Materialize does not yet support Future Sight.");
  }
  snapshot.turn = publicState.turn;
  const selfForceSwitch = publicState.selfRequestKind === "force-switch";
  snapshot.requestState = selfForceSwitch ? "switch" : "move";
  snapshot.lastMove = null;
  snapshot.lastMoveLine = 0;
  snapshot.lastSuccessfulMoveThisTurn = null;
  snapshot.lastDamage = 0;
  snapshot.midTurn = false;
  snapshot.queue = [];
  snapshot.faintQueue = [];
  snapshot.activeMove = null;
  snapshot.activePokemon = null;
  snapshot.activeTarget = null;
  applyPublicWeather(snapshot.field, publicState);
  snapshot.field.pseudoWeather = {};
  const wishSetTurns = publicState.wishSetTurns;
  if (wishSetTurns != null && typeof wishSetTurns !== "object") {
    throw new Error("Materialize received invalid Wish timing.");
  }
  const leechSeedSourceSides = publicState.leechSeedSourceSides;
  if (leechSeedSourceSides != null && typeof leechSeedSourceSides !== "object") {
    throw new Error("Materialize received invalid Leech Seed provenance.");
  }
  const pendingBatonPassSides = publicState.pendingBatonPassSides ?? [];
  if (!Array.isArray(pendingBatonPassSides) ||
      !pendingBatonPassSides.every(sideId => sideId === "p1" || sideId === "p2")) {
    throw new Error("Materialize received invalid Baton Pass state.");
  }
  if (pendingBatonPassSides.length > 1 ||
      (pendingBatonPassSides.length === 1 &&
       (pendingBatonPassSides[0] !== publicState.selfPlayer || !selfForceSwitch))) {
    throw new Error("Materialize received a Baton Pass state without its forced switch.");
  }

  for (const [sideIndex, sideId] of ["p1", "p2"].entries()) {
    const publicSide = publicState.sides[sideId];
    if (!publicSide || typeof publicSide !== "object") {
      throw new Error(`Materialize is missing public state for ${sideId}.`);
    }
    if (!Array.isArray(publicSide.volatiles)) {
      throw new Error(`Materialize received invalid volatile effects for ${sideId}.`);
    }
    if (!Array.isArray(publicSide.materializationBlockers)) {
      throw new Error(`Materialize received invalid state blockers for ${sideId}.`);
    }
    if (publicSide.materializationBlockers.length > 0) {
      throw new Error(
        `Materialize cannot reconstruct public state for ${sideId}: ` +
        publicSide.materializationBlockers.join(", "),
      );
    }
    const rows = Array.isArray(publicSide.pokemon) ? publicSide.pokemon : [];
    const serializedSide = snapshot.sides[sideIndex];
    let activeIndex = null;
    for (const row of rows) {
      if (!row || typeof row !== "object" || typeof row.species !== "string") {
        throw new Error(`Materialize contains an invalid ${sideId} Pokemon row.`);
      }
      const matchingIndices = serializedSide.pokemon
        .map((pokemon, index) => (sameSpecies(pokemon, row.species) ? index : -1))
        .filter(index => index >= 0);
      if (matchingIndices.length !== 1) {
        throw new Error(`Materialize cannot uniquely match ${sideId} ${row.species}.`);
      }
      const index = matchingIndices[0];
      applyPokemonCondition(
        serializedSide.pokemon[index],
        row.condition,
        sideId,
        row.species,
        publicSide.toxicStage,
      );
      serializedSide.pokemon[index].boosts = row.active
        ? normalizedBoosts(publicSide.boosts)
        : normalizedBoosts(null);
      applyPublicVolatiles(
        serializedSide.pokemon[index],
        row.active ? publicSide.volatiles : [],
        sideId,
        leechSeedSourceSides,
      );
      serializedSide.pokemon[index].lastMove = null;
      serializedSide.pokemon[index].lastMoveUsed = null;
      serializedSide.pokemon[index].attackedBy = [];
      serializedSide.pokemon[index].lastDamage = 0;
      serializedSide.pokemon[index].activeMoveActions = 0;
      serializedSide.pokemon[index].moveThisTurn = "";
      serializedSide.pokemon[index].isActive = Boolean(row.active);
      if (row.active) {
        if (activeIndex !== null) throw new Error(`Materialize found multiple active ${sideId} Pokemon.`);
        activeIndex = index;
      }
    }
    if (activeIndex === null) {
      throw new Error(`Materialize requires one active ${sideId} Pokemon.`);
    }
    // Showdown treats `position < side.active.length` as the request-level active
    // predicate. A direct construction therefore needs the same active-first team
    // ordering that a real switch produces; changing only `side.active` leaves the
    // sampled lead in slot zero and exposes the wrong request to the policy.
    const active = moveActivePokemonToFront(serializedSide, activeIndex);
    active.activeTurns = Math.max(1, Number(active.activeTurns) || 1);
    serializedSide.active = [`[Pokemon:${sideId}a]`];
    // The acting request is private, but a fainted public active with a surviving
    // bench deterministically requires a replacement for either side.
    if (active.fainted && serializedSide.pokemon.some(pokemon => !pokemon.fainted)) {
      active.switchFlag = true;
    }
    if (sideId === publicState.selfPlayer) {
      preserveActorTeamOrder(serializedSide, publicState.selfTeamOrder);
      if (selfForceSwitch) active.switchFlag = true;
      applyKnownMoveState(active, publicState.selfActiveMoves);
      applySelfActiveRequestState(active, publicState.selfActiveRequestState);
      for (const row of rows) {
        const matchingIndex = serializedSide.pokemon.findIndex(pokemon => sameSpecies(pokemon, row.species));
        if (matchingIndex >= 0) applyKnownMoveState(serializedSide.pokemon[matchingIndex], row.moves);
      }
    }
    if (pendingBatonPassSides.includes(sideId)) {
      // BattleQueue turns this exact flag into the Baton Pass source effect when it resolves the
      // switch. The skip flag mirrors the already-completed BeforeSwitchOut phase.
      active.switchFlag = "batonpass";
      active.skipBeforeSwitchOutEventFlag = true;
    }
    applyPublicSideConditions(serializedSide, publicSide, sideId, publicState.turn);
    serializedSide.slotConditions = [{}];
    applyPublicWish(serializedSide, wishSetTurns?.[sideId], sideId, publicState.turn);
    serializedSide.pokemonLeft = serializedSide.pokemon.filter(pokemon => !pokemon.fainted).length;
    serializedSide.totalFainted = serializedSide.pokemon.length - serializedSide.pokemonLeft;
    delete serializedSide.activeRequest;
  }
}

function applyPublicWish(serializedSide, setTurn, sideId, currentTurn) {
  if (setTurn == null) return;
  const age = currentTurn - setTurn;
  if (!Number.isInteger(setTurn) || (age !== 0 && age !== 1)) {
    throw new Error(`Materialize received expired or invalid Wish timing for ${sideId}.`);
  }
  const source = serializedSide.pokemon[0];
  if (!source || !Number.isFinite(source.maxhp) || source.maxhp < 1) {
    throw new Error(`Materialize cannot restore Wish without a ${sideId} source.`);
  }
  const sourceSlot = `${sideId}a`;
  serializedSide.slotConditions[0].wish = {
    id: "wish",
    target: `[Side:${sideId}]`,
    source: `[Pokemon:${sourceSlot}]`,
    sourceSlot,
    isSlotCondition: true,
    // A normal request is already on the turn after Wish. A forced-switch
    // boundary can still be on the declaration turn, requiring one extra
    // residual countdown before the heal lands.
    duration: 2 - age,
    effectOrder: 2,
    // Gen 3 Wish heals half the user's maximum HP, captured before any later
    // switch can occur. At this request boundary the user remains the active
    // slot, possibly fainted while awaiting a forced switch.
    hp: source.maxhp / 2,
    startingTurn: setTurn,
  };
}

const TIMED_SIDE_CONDITIONS = new Set(["reflect", "lightscreen", "safeguard", "mist"]);
// These conditions are persistent, public flags. They neither carry an unknown duration nor
// require a source/target relationship to reproduce their Gen 3 mechanics.
const STATIC_PUBLIC_VOLATILES = new Set([
  "focusenergy", "ingrain", "mudsport", "watersport",
]);

function applyPublicWeather(field, publicState) {
  const weather = normalizeId(publicState.weather);
  if (!weather) {
    field.weather = "";
    field.weatherState = {id: "", effectOrder: 0};
    return;
  }
  const weatherState = {id: weather, effectOrder: 0, target: "[Field]"};
  if (!publicState.weatherFromAbility) {
    weatherState.duration = remainingTimedTurns(
      publicState.turn,
      publicState.weatherSetTurn,
      "weather",
    );
  }
  field.weather = weather;
  field.weatherState = weatherState;
}

function applyPublicSideConditions(serializedSide, publicSide, sideId, currentTurn) {
  const sideConditions = publicSide.sideConditions;
  if (sideConditions != null && typeof sideConditions !== "object") {
    throw new Error(`Materialize received invalid side conditions for ${sideId}.`);
  }
  const setTurns = publicSide.sideConditionSetTurns;
  if (setTurns != null && typeof setTurns !== "object") {
    throw new Error(`Materialize received invalid side-condition timing for ${sideId}.`);
  }
  serializedSide.sideConditions = {};
  for (const [rawCondition, rawCount] of Object.entries(sideConditions || {})) {
    const condition = normalizeId(rawCondition);
    const count = Number(rawCount);
    if (!Number.isInteger(count) || count < 1) {
      throw new Error(`Materialize received invalid ${rawCondition} count for ${sideId}.`);
    }
    const state = {id: condition, effectOrder: 2, target: `[Side:${sideId}]`};
    if (condition === "spikes") {
      if (count > 3) throw new Error(`Materialize received invalid Spikes layers for ${sideId}.`);
      state.layers = count;
    } else if (TIMED_SIDE_CONDITIONS.has(condition)) {
      state.duration = remainingTimedTurns(currentTurn, setTurns?.[rawCondition], condition);
    } else {
      throw new Error(`Materialize does not yet support side condition ${rawCondition}.`);
    }
    serializedSide.sideConditions[condition] = state;
  }
}

function remainingTimedTurns(currentTurn, setTurn, label) {
  if (!Number.isInteger(currentTurn) || currentTurn < 1 || !Number.isInteger(setTurn) || setTurn < 1) {
    throw new Error(`Materialize requires a public set turn for active ${label}.`);
  }
  const remaining = 5 - (currentTurn - setTurn);
  if (remaining < 1 || remaining > 5) {
    throw new Error(`Materialize received an expired or invalid ${label} duration.`);
  }
  return remaining;
}

function moveActivePokemonToFront(serializedSide, activeIndex) {
  const originalIndices = [activeIndex];
  for (let index = 0; index < serializedSide.pokemon.length; index++) {
    if (index !== activeIndex) originalIndices.push(index);
  }
  return reorderSerializedSide(serializedSide, originalIndices);
}

function preserveActorTeamOrder(serializedSide, teamOrder) {
  if (!Array.isArray(teamOrder) || teamOrder.length !== serializedSide.pokemon.length) {
    throw new Error("Materialize requires the acting player's full team order.");
  }
  const indices = [];
  const used = new Set();
  for (const species of teamOrder) {
    const matches = serializedSide.pokemon
      .map((pokemon, index) => (sameSpecies(pokemon, species) && !used.has(index) ? index : -1))
      .filter(index => index >= 0);
    if (matches.length !== 1) {
      throw new Error(`Materialize cannot uniquely preserve acting team order for ${species}.`);
    }
    const index = matches[0];
    used.add(index);
    indices.push(index);
  }
  if (indices[0] !== 0) {
    throw new Error("Materialize acting team order must keep the active Pokemon first.");
  }
  reorderSerializedSide(serializedSide, indices);
}

function reorderSerializedSide(serializedSide, originalIndices) {
  const originalPokemon = serializedSide.pokemon;
  if (originalIndices.length !== originalPokemon.length || new Set(originalIndices).size !== originalPokemon.length) {
    throw new Error("Materialize received an invalid serialized team permutation.");
  }
  const originalIndexByCurrentIndex = originalTeamIndexByCurrentIndex(
    serializedSide.team,
    originalPokemon.length,
  );
  const reorderedTeam = Array(originalPokemon.length);
  serializedSide.pokemon = originalIndices.map(index => originalPokemon[index]);
  for (const [newIndex, oldIndex] of originalIndices.entries()) {
    const pokemon = serializedSide.pokemon[newIndex];
    pokemon.position = newIndex;
    pokemon.isActive = newIndex === 0;
    reorderedTeam[originalIndexByCurrentIndex[oldIndex]] = newIndex + 1;
  }
  serializedSide.team = encodeTeamOrder(reorderedTeam);
  return serializedSide.pokemon[0];
}

function originalTeamIndexByCurrentIndex(team, expectedLength) {
  const tokens = String(team || "").split(String(team || "").includes(",") ? "," : "");
  if (tokens.length !== expectedLength) {
    throw new Error("Materialize received an invalid serialized team order.");
  }
  const result = Array(expectedLength);
  for (const [originalIndex, token] of tokens.entries()) {
    const currentIndex = Number(token) - 1;
    if (!Number.isInteger(currentIndex) || currentIndex < 0 || currentIndex >= expectedLength) {
      throw new Error("Materialize received an invalid serialized team order.");
    }
    if (result[currentIndex] !== undefined) {
      throw new Error("Materialize received a non-permutation serialized team order.");
    }
    result[currentIndex] = originalIndex;
  }
  return result;
}

function encodeTeamOrder(order) {
  return order.join(order.length > 9 ? "," : "");
}

function sameSpecies(pokemon, species) {
  const packedSpecies = pokemon?.set?.species || pokemon?.set?.name || "";
  return normalizeId(packedSpecies) === normalizeId(species);
}

function normalizeId(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function hasEntries(value) {
  return value && typeof value === "object" && Object.keys(value).length > 0;
}

function normalizedBoosts(boosts) {
  const normalized = {atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0};
  if (!boosts || typeof boosts !== "object") return normalized;
  for (const key of Object.keys(normalized)) {
    if (Number.isInteger(boosts[key])) normalized[key] = boosts[key];
  }
  return normalized;
}

function applyPublicVolatiles(pokemon, rawVolatiles, sideId, leechSeedSourceSides) {
  if (!Array.isArray(rawVolatiles)) {
    throw new Error(`Materialize received invalid volatile effects for ${sideId}.`);
  }
  pokemon.volatiles = {};
  const seen = new Set();
  for (const rawVolatile of rawVolatiles) {
    if (typeof rawVolatile !== "string") {
      throw new Error(`Materialize received invalid volatile effect for ${sideId}.`);
    }
    const volatile = normalizeId(rawVolatile);
    if (volatile === "leechseed") {
      const sourceSide = leechSeedSourceSides?.[sideId];
      if (!["p1", "p2"].includes(sourceSide) || sourceSide === sideId) {
        throw new Error(`Materialize cannot restore Leech Seed on ${sideId} without a public source side.`);
      }
      if (seen.has(volatile)) {
        throw new Error(`Materialize received duplicate volatile effect ${rawVolatile}.`);
      }
      seen.add(volatile);
      const sourceSlot = `${sourceSide}a`;
      pokemon.volatiles[volatile] = {
        id: volatile,
        effectOrder: 0,
        target: `[Pokemon:${sideId}a]`,
        source: `[Pokemon:${sourceSlot}]`,
        sourceSlot,
      };
      continue;
    }
    if (!STATIC_PUBLIC_VOLATILES.has(volatile)) {
      throw new Error(`Materialize does not yet support volatile effect ${rawVolatile}.`);
    }
    if (seen.has(volatile)) {
      throw new Error(`Materialize received duplicate volatile effect ${rawVolatile}.`);
    }
    seen.add(volatile);
    pokemon.volatiles[volatile] = {
      id: volatile,
      effectOrder: 0,
      target: `[Pokemon:${sideId}a]`,
    };
  }
}

function applyPokemonCondition(pokemon, condition, sideId, species, toxicStage) {
  if (typeof condition !== "string" || !condition.trim()) {
    throw new Error(`Materialize is missing a condition for ${sideId} ${species}.`);
  }
  const parts = condition.trim().split(/\s+/);
  const fainted = parts.includes("fnt") || parts[0] === "0";
  const status = parts.find(part => ["brn", "frz", "par", "psn", "tox"].includes(part)) || "";
  if (parts.includes("slp")) {
    throw new Error("Materialize does not yet support sleep counters.");
  }
  let hp = 0;
  let maxhp = pokemon.maxhp;
  if (!fainted) {
    const match = /^(\d+)\/(\d+)$/.exec(parts[0]);
    if (!match) throw new Error(`Materialize received an invalid condition for ${sideId} ${species}.`);
    hp = Number(match[1]);
    const publicMaxhp = Number(match[2]);
    if (publicMaxhp !== pokemon.maxhp) {
      throw new Error(`Materialize max HP mismatch for ${sideId} ${species}.`);
    }
    maxhp = publicMaxhp;
  }
  pokemon.hp = hp;
  pokemon.maxhp = maxhp;
  pokemon.baseMaxhp = maxhp;
  pokemon.fainted = fainted;
  pokemon.status = status;
  pokemon.statusState = {id: status, effectOrder: 0};
  if (status === "tox") {
    if (!Number.isInteger(toxicStage) || toxicStage < 0 || toxicStage > 15) {
      throw new Error(`Materialize requires a valid toxic stage for ${sideId} ${species}.`);
    }
    // The replay fold records the current toxic ramp from public protocol events. It is
    // sufficient to restore the exact next residual without reading source-world state.
    pokemon.statusState.stage = toxicStage;
  }
}

function applyKnownMoveState(pokemon, moves) {
  if (!Array.isArray(moves)) return;
  for (const state of moves) {
    if (!state || typeof state.id !== "string" || !Number.isInteger(state.pp) || !Number.isInteger(state.maxpp)) {
      continue;
    }
    const slot = pokemon.moveSlots.find(move => normalizeId(move.id) === normalizeId(state.id));
    if (!slot) throw new Error(`Materialize cannot match acting move ${state.id}.`);
    if (slot.maxpp !== state.maxpp) {
      throw new Error(`Materialize max PP mismatch for acting move ${state.id}.`);
    }
    slot.pp = state.pp;
    slot.disabled = Boolean(state.disabled);
    slot.used = state.pp < state.maxpp;
  }
}

function applySelfActiveRequestState(pokemon, state) {
  if (!state || typeof state !== "object") return;
  // These are actor-visible request flags, not inferred opponent state. They must survive
  // reconstruction so the direct branch exposes the same legal action boundary as the live turn.
  for (const name of ["trapped", "maybeTrapped", "maybeDisabled", "maybeLocked"]) {
    if (state[name] === true) pokemon[name] = true;
  }
}

function boundaryRequestsFromBattle(simulatorBattle) {
  const requests = simulatorBattle.getRequests(simulatorBattle.requestState);
  const result = {};
  for (const [index, player] of ["p1", "p2"].entries()) {
    if (requests[index] && typeof requests[index] === "object") result[player] = requests[index];
  }
  return result;
}

async function reseedBattle(command) {
  const battle = requireBattle(command);
  const seed = command.seed;
  if (typeof seed !== "string" || !seed.trim()) {
    throw new Error("Reseed requires a non-empty seed string.");
  }
  await battle.streams.omniscient.write(`>reseed ${seed}`);
  emit({ type: "reseeded", battleId: battle.battleId, seed });
}

// Player options accept either the legacy string form (just a name, used by random battles) or an
// object carrying { name, team }. A custom packed team string is passed straight through to
// Pokemon Showdown's player options; omitting it preserves random-battle behavior.
function normalizePlayerOptions(value, fallbackName) {
  if (value && typeof value === "object") {
    const name = typeof value.name === "string" && value.name ? value.name : fallbackName;
    const options = { name };
    if (typeof value.team === "string" && value.team) {
      options.team = value.team;
    }
    return options;
  }
  return { name: typeof value === "string" && value ? value : fallbackName };
}

function requireBattle(command) {
  const battleId = battleIdOf(command);
  const battle = battles.get(battleId);
  if (!battle || !battle.streams) {
    throw new Error(`No running battle for battleId ${battleId}.`);
  }
  return battle;
}

async function sendChoice(command) {
  const battle = requireBattle(command);
  const player = command.player;
  if (!["p1", "p2"].includes(player)) {
    throw new Error(`Unsupported player: ${player}`);
  }
  const choice = command.choice;
  if (typeof choice !== "string" || !choice.trim()) {
    throw new Error("Choice must be a non-empty string.");
  }
  await battle.streams[player].write(choice);
  emit({ type: "choice_ack", battleId: battle.battleId, player, choice });
}

async function sendChoices(command) {
  const battle = requireBattle(command);
  const choices = command.choices;
  if (!choices || typeof choices !== "object") {
    throw new Error("Choices must be an object keyed by player.");
  }
  battle.boundaryRequests = {};
  battle.readyScheduled = false;
  battle.terminalScheduled = false;
  battle.tRecv = process.hrtime.bigint();
  for (const player of ["p1", "p2"]) {
    if (!Object.prototype.hasOwnProperty.call(choices, player)) continue;
    const choice = choices[player];
    if (typeof choice !== "string" || !choice.trim()) {
      throw new Error(`Choice for ${player} must be a non-empty string.`);
    }
    await battle.streams[player].write(choice);
    emit({ type: "choice_ack", battleId: battle.battleId, player, choice });
  }
}

async function endBattle(command) {
  const battleId = battleIdOf(command);
  const battle = battles.get(battleId);
  if (battle) {
    await teardownBattle(battle);
    battles.delete(battleId);
  }
  emit({ type: "ended", battleId });
}

async function closeAll() {
  for (const battle of battles.values()) {
    await teardownBattle(battle);
  }
  battles.clear();
  searchSnapshots.clear();
  emit({ type: "closed" });
}

async function handleCommand(command) {
  switch (command.type) {
    case "start":
      await startBattle(command);
      break;
    case "choice":
      await sendChoice(command);
      break;
    case "choices":
      await sendChoices(command);
      break;
    case "snapshot":
      snapshotBattle(command);
      break;
    case "snapshot_search":
      snapshotSearchBattle(command);
      break;
    case "restore":
      restoreBattle(command);
      break;
    case "restore_search":
      restoreSearchBattle(command);
      break;
    case "release_search_snapshot":
      releaseSearchSnapshot(command);
      break;
    case "materialize":
      materializeBattle(command);
      break;
    case "reseed":
      await reseedBattle(command);
      break;
    case "end":
      await endBattle(command);
      break;
    case "close":
      await closeAll();
      process.exit(0);
      break;
    default:
      throw new Error(`Unsupported command type: ${command.type}`);
  }
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", line => {
  void (async () => {
    if (!line.trim()) return;
    try {
      await handleCommand(JSON.parse(line));
    } catch (error) {
      emit({ type: "error", message: error.message });
    }
  })();
});

rl.on("close", () => {
  void closeAll().finally(() => process.exit(0));
});
