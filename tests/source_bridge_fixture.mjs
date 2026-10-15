// Disposable authored protocol worker. No simulator, team generation or model.
import fs from "node:fs";
import readline from "node:readline";
import {createRequire} from "node:module";
const root = process.argv[3];
// Positive label fixtures share a frozen toy clone runtime at index.js.
// Existing source-only fixtures retain their original index.js settings.
// Exact collector fixtures may use an existing read-only catalog/Dex while
// pinning authored protocol settings alongside this fixture bridge.
const localSettings = new URL("./source-settings.json", import.meta.url);
const settingsPath = fs.existsSync(localSettings) ? localSettings
  : fs.existsSync(`${root}/dist/sim/source-settings.json`)
    ? `${root}/dist/sim/source-settings.json` : `${root}/dist/sim/index.js`;
const settings = JSON.parse(fs.readFileSync(settingsPath, "utf8"));
const {mode} = settings;
let rounds = settings.rounds;
let actorRequests = settings.actorRequests;
let token, n = 0;
let battleLog = [];
const request = player => {
  const value = structuredClone(actorRequests[player]);
  if (value.active?.[0]?.moves?.[1])
    value.active[0].moves[1].pp = Math.max(1, value.active[0].moves[1].pp - n);
  return value;
};
const actorMessages = protocol => {
  battleLog.push(...protocol);
  emit({type:"stream",battleId:token,stream:"omniscient",lines:protocol});
  for (const player of ["p1","p2"])
    emit({type:"stream",battleId:token,stream:player,
      lines:[...protocol, `|request|${JSON.stringify(request(player))}`]});
};
const emit = value => process.stdout.write(`${JSON.stringify(value)}\n`);
const requested = () => actorRequests
  ? ["p1","p2"].filter(player => !request(player).wait) : ["p1","p2"];
const ready = () => emit({type:"ready", battleId:token, requested:requested()});
// Optional authored clone transport only. Admission, token tombstones,
// snapshots, stream draining and release are the ORIGINAL strict helper,
// never fabricated acknowledgements. The runtime explicitly forbids fresh
// Battle construction/team generation and is pinned by each fixture owner.
const cloneBattles = new Map();
let owned;
if (settings.ownedContinuationTransport === true) {
  const require = createRequire(import.meta.url);
  const {BattleStream, getPlayerStreams} = require("./authored-clones/index.js");
  const {State} = require("./authored-clones/state.js");
  const {createOwnedContinuations} = await import("./battle_bridge_owned_continuation.mjs");
  owned = createOwnedContinuations({BattleStream, getPlayerStreams, State, battles:cloneBattles,
    newBattleState:battleId => ({battleId, boundaryRequests:{}, boundaryGeneration:0,
      boundaryRequestGeneration:null, readyEpoch:0, readyScheduled:false, terminalScheduled:false}),
    actionableRequestedPlayers:battle => ["p1","p2"].filter(player => {
      const value = battle.boundaryRequests[player];
      return value && !value.wait && !value.teamPreview &&
        (value.forceSwitch?.some(Boolean) || value.active?.length > 0);
    }),
    listenToStream:(battle, stream, channel) => (async () => {
      for await (const chunk of channel) {
        const lines = String(chunk).split("\n").filter(Boolean);
        emit({type:"stream",battleId:battle.battleId,stream,lines});
        if (stream === "omniscient" && lines.some(line => line.startsWith("|win|") || line === "|tie|")) {
          const epoch = ++battle.readyEpoch;
          battle.terminalScheduled = true;
          setImmediate(() => {
            if (battle.readyEpoch === epoch && battle.terminalScheduled)
              emit({type:"terminal",battleId:battle.battleId,nodeProcMs:.25});
          });
        }
      }
      emit({type:"stream_end",battleId:battle.battleId,stream});
    })(), emit});
}

async function dispatch(command) {
  const clone = cloneBattles.get(command.battleId);
  if (clone) {
    if (command.type === "reseed") {
      if (typeof command.seed !== "string" || !command.seed.trim())
        throw new Error("authored clone requires original seed string");
      await clone.streams.omniscient.write(">reseed " + command.seed);
      emit({type:"reseeded",battleId:clone.battleId,seed:command.seed});
    } else if (command.type === "choices") {
      clone.readyEpoch++; clone.boundaryGeneration++;
      clone.boundaryRequests = {}; clone.boundaryRequestGeneration = null;
      for (const player of ["p1","p2"]) {
        if (!Object.hasOwn(command.choices, player)) continue;
        const choice = command.choices[player];
        await clone.streams[player].write(choice);
        emit({type:"choice_ack",battleId:clone.battleId,player,choice});
      }
    } else throw new Error("forbidden authored clone command");
    return;
  }
  if (command.type === "start") {
    if (token) throw new Error("authored source restart");
    token = command.battleId;
    rounds = settings.roundsBySeed?.[JSON.stringify(command.seed)] ?? settings.rounds;
    actorRequests = settings.requestsBySeed?.[JSON.stringify(command.seed)] ?? settings.actorRequests;
    emit({type:"started",battleId:token,formatid:command.formatid,seed:command.seed});
    if (actorRequests) actorMessages(settings.actorPrefix);
    else emit({type:"stream",battleId:token,stream:"omniscient",lines:[
      "|player|p1|PokeZero p1|", "|player|p2|PokeZero p2|"]});
    ready();
  } else if (command.type === "choices") {
    if (mode === "bridge_failure") process.exit(9);
    if (mode === "bridge_hang") await new Promise(() => {});
    if (command.battleId !== token) throw new Error("authored token drift");
    for (const [player, choice] of Object.entries(command.choices))
      emit({type:"choice_ack",battleId:token,player,choice});
    n++;
    if (n === rounds) {
      emit({type:"stream",battleId:token,stream:"omniscient",lines:["|win|PokeZero p1"]});
      emit({type:"terminal",battleId:token});
    } else {
      if (actorRequests) actorMessages([`|turn|${n+1}`]);
      ready();
    }
  } else if (command.type === "end") {
    for (const stream of ["omniscient","p1","p2"])
      if (!(mode === "missing_end" && stream === "p2")) emit({type:"stream_end",battleId:token,stream});
    emit({type:"ended",battleId:token});
  } else if (command.type === "snapshot_actionable_boundary") {
    if (!settings.snapshotTransport || command.battleId !== token)
      throw new Error("unreviewed authored snapshot command");
    emit({type:"snapshot_actionable_boundary",battleId:token,available:true,requested:requested(),
      snapshot:{battle:{...(settings.ownedContinuationTransport === true ? {
        // The strict deserializer copies Pokemon.set and log aliases. Supply
        // an explicitly authored serialized party; this toy runtime neither
        // constructs a real Battle nor claims simulator correspondence.
        sides:["p1","p2"].map(player => ({pokemon:
          ((actorRequests ? request(player) : null)?.side?.pokemon ?? [{moves:["tackle"]}]).map(mon => ({
            set:{species:mon.details?.split(",")[0] ?? "Charizard",
              moves:structuredClone(mon.moves ?? ["tackle"]),
              ability:mon.baseAbility ?? "blaze", item:mon.item ?? ""},
          })),
        })),
      } : {}), log:actorRequests ? battleLog : ["|player|p1|PokeZero p1|", "|player|p2|PokeZero p2|", "|start",
        "|split|p1", "|debug|SOURCE_SECRET_P1", "|message|public-p1",
        "|split|p2", "|debug|SOURCE_SECRET_P2", "|message|public-p2", `|turn|${n+1}`],
        privatePokemon:"SOURCE_PRIVATE_POKEMON"},
        boundaryRequests:actorRequests ? {p1:request("p1"),p2:request("p2")} : {p1:{active:[{moves:[{id:"tackle",pp:15}]}]},
          p2:{active:[{moves:[{id:"SOURCE_PRIVATE_MOVE",pp:7}]}]}},terminalScheduled:false},nodeProcMs:.25});
  } else if (command.type === "close") {
    if (owned) await owned.shutdown();
    emit({type:mode === "bad_close" ? "not_closed" : "closed"});
    process.exit(mode === "nonzero_close" ? 7 : 0);
  } else throw new Error("forbidden authored command");
}

const lines = readline.createInterface({input:process.stdin});
for await (const line of lines) {
  const command = JSON.parse(line);
  if (owned) await owned.execute(command, () => dispatch(command));
  else await dispatch(command);
}
