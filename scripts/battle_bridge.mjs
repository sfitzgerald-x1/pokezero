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

let streams = null;

function emit(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function emitStreamChunk(stream, chunk) {
  const lines = String(chunk)
    .split("\n")
    .filter(line => line.length > 0);
  if (lines.length > 0) {
    emit({ type: "stream", stream, lines });
  }
}

function listenToStream(name, stream) {
  void (async () => {
    try {
      for await (const chunk of stream) {
        emitStreamChunk(name, chunk);
      }
      emit({ type: "stream_end", stream: name });
    } catch (error) {
      emit({ type: "error", stream: name, message: error.message });
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

async function startBattle(command) {
  if (streams) {
    throw new Error("A battle is already running in this bridge process.");
  }
  const battleStream = new BattleStream();
  streams = getPlayerStreams(battleStream);
  for (const name of ["omniscient", "p1", "p2"]) {
    listenToStream(name, streams[name]);
  }

  const formatid = command.formatid || "gen3randombattle";
  const seed = command.seed;
  const players = command.players || {};
  const startOptions = { formatid, strictChoices: true };
  if (seed) startOptions.seed = seed;
  const p1 = { name: players.p1 || "PokeZero p1" };
  const p2 = { name: players.p2 || "PokeZero p2" };
  if (seed) {
    p1.seed = deriveSeed(seed, "p1");
    p2.seed = deriveSeed(seed, "p2");
  }
  await streams.omniscient.write(
    `>start ${JSON.stringify(startOptions)}\n` +
      `>player p1 ${JSON.stringify(p1)}\n` +
      `>player p2 ${JSON.stringify(p2)}`
  );
  emit({ type: "started", formatid, seed: seed || null });
}

async function sendChoice(command) {
  if (!streams) {
    throw new Error("Cannot submit a choice before starting a battle.");
  }
  const player = command.player;
  if (!["p1", "p2"].includes(player)) {
    throw new Error(`Unsupported player: ${player}`);
  }
  const choice = command.choice;
  if (typeof choice !== "string" || !choice.trim()) {
    throw new Error("Choice must be a non-empty string.");
  }
  await streams[player].write(choice);
  emit({ type: "choice_ack", player, choice });
}

async function closeBattle() {
  if (streams) {
    await streams.omniscient.writeEnd();
    streams = null;
  }
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
    case "close":
      await closeBattle();
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
  void closeBattle().finally(() => process.exit(0));
});
