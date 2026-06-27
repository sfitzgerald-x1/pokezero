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
try {
  ({ BattleStream, getPlayerStreams } = require(path.join(showdownRoot, "dist", "sim", "index.js")));
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

function emit(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function battleIdOf(command) {
  return command.battleId == null ? DEFAULT_BATTLE_ID : String(command.battleId);
}

function newBattleState(battleId) {
  return { battleId, streams: null, boundaryRequests: {}, readyScheduled: false, terminalScheduled: false };
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
  const battleStream = new BattleStream();
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
