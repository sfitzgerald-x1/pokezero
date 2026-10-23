// Source/target binding ONLY, after all party reordering and timer restoration.
// The opt-in joint sampler supplies qualified assignments; legacy paths do not.
// The caller must derive origins from public chronology, not private simulator
// state. This module does not infer provenance, timers, abilities or a sampler.

function referenceIdentity(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      Object.keys(value).sort().join(",") !== "side,species" ||
      !["p1", "p2"].includes(value.side) ||
      typeof value.species !== "string" || !value.species.trim()) {
    throw new Error("Reference sleep binding needs a resolved side and species.");
  }
  return value;
}

function speciesId(value) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, "");
}

export function restoreReferenceSleep(snapshot, assignments) {
  if (!Array.isArray(assignments)) throw new Error("Reference joint sleep needs an assignment roster.");
  // Validate in a cheap shadow (only parties/status objects copied), so a bad
  // late source/timer never partially changes the supplied snapshot.
  const shadow = {...snapshot, sides: snapshot.sides.map(side => ({...side,
    pokemon: side.pokemon.map(pokemon => ({...pokemon, statusState: {...pokemon.statusState}}))}))};
  const origins = [];
  for (const row of assignments) {
    if (!row || Object.keys(row).sort().join(",") !== "kind,skippedTime,sleeper,source,time" ||
        !Number.isSafeInteger(row.time) || !Number.isSafeInteger(row.skippedTime)) {
      throw new Error("Reference joint sleep requires explicit integer timer assignments.");
    }
    referenceIdentity(row.sleeper);
    const side = shadow.sides.find(side => side.id === row.sleeper.side);
    const matches = side?.pokemon.filter(p => speciesId(p.set.species) === speciesId(row.sleeper.species)) ?? [];
    if (matches.length !== 1) throw new Error("Reference joint sleep cannot resolve its sleeper.");
    matches[0].statusState = {id: "slp", effectOrder: 0, time: row.time, skippedTime: row.skippedTime,
      ...(row.kind === "rest" ? {startTime: 3} : {})};
    origins.push({kind: row.kind, sleeper: row.sleeper, source: row.source});
  }
  bindReferenceSleepSources(shadow, origins);
  for (const [s, side] of shadow.sides.entries()) {
    for (const [p, pokemon] of side.pokemon.entries()) {
      if (pokemon.status === "slp") snapshot.sides[s].pokemon[p].statusState = pokemon.statusState;
    }
  }
}

export function bindReferenceSleepSources(snapshot, assignments) {
  if (!snapshot || !Array.isArray(snapshot.sides) || snapshot.sides.length !== 2 ||
      !["gen3randombattle", "gen3customgame"].includes(snapshot.formatid) ||
      !Array.isArray(assignments) || assignments.length > 12) {
    throw new Error("Reference sleep binding needs both serialized parties.");
  }
  const entries = new Map();
  const sleepers = new Set();
  for (const [sideIndex, side] of snapshot.sides.entries()) {
    const sideId = ["p1", "p2"][sideIndex];
    if (!side || side.id !== sideId || !Array.isArray(side.pokemon) ||
        !side.pokemon.length || side.pokemon.length > 6) {
      throw new Error("Reference sleep binding received an invalid party.");
    }
    for (const [index, pokemon] of side.pokemon.entries()) {
      if (!pokemon || pokemon.position !== index ||
          typeof pokemon.set?.species !== "string" || !speciesId(pokemon.set.species)) {
        throw new Error("Reference sleep binding requires finalized party positions.");
      }
      const key = `${sideId}:${speciesId(pokemon.set.species)}`;
      if (entries.has(key)) throw new Error("Reference sleep binding has ambiguous species.");
      entries.set(key, {pokemon, sideId, reference: `[Pokemon:${sideId}${"abcdef"[index]}]`});
      if (pokemon.status === "slp") {
        if (pokemon.fainted || !Number.isFinite(pokemon.hp) || pokemon.hp <= 0) {
          throw new Error("Reference sleep binding received a fainted sleeper.");
        }
        sleepers.add(key);
      }
    }
  }
  const resolve = value => {
    referenceIdentity(value);
    const key = `${value.side}:${speciesId(value.species)}`;
    const entry = entries.get(key);
    if (!entry) throw new Error("Reference sleep source is absent from the sampled party.");
    return [key, entry];
  };
  const seen = new Set();
  const pending = [];
  for (const row of assignments) {
    if (!row || typeof row !== "object" || Array.isArray(row) ||
        Object.keys(row).sort().join(",") !== "kind,sleeper,source" ||
        !["rest", "induced"].includes(row.kind)) {
      throw new Error("Reference sleep binding has invalid origin provenance.");
    }
    const [key, sleeper] = resolve(row.sleeper);
    const [, source] = resolve(row.source);
    if (!sleepers.has(key) || seen.has(key)) {
      throw new Error("Reference sleep assignments must cover each sleeper exactly once.");
    }
    if ((row.kind === "rest" && source !== sleeper) ||
        (row.kind === "induced" && source.sideId === sleeper.sideId)) {
      throw new Error("Reference sleep origin contradicts its source.");
    }
    const state = sleeper.pokemon.statusState;
    if (!state || typeof state !== "object" || Array.isArray(state) || state.id !== "slp" ||
        !Number.isSafeInteger(state.time) || state.time < 1 || state.time > 5 ||
        !Number.isSafeInteger(state.skippedTime) || state.skippedTime < 0 ||
        state.time + state.skippedTime > 5) {
      throw new Error("Reference sleep binding requires an already restored Gen 3 timer.");
    }
    if ((row.kind === "rest" && (state.startTime !== 3 || state.time + state.skippedTime > 3)) ||
        (row.kind === "induced" && state.startTime !== undefined)) {
      throw new Error("Reference sleep timer contradicts its declared origin.");
    }
    seen.add(key);
    pending.push([state, source.reference, sleeper.reference]);
  }
  if (seen.size !== sleepers.size) {
    throw new Error("Reference sleep binding is missing a sleeping party member.");
  }
  // Validate the entire roster before mutating anything. In particular a late
  // unknown source cannot leave earlier sleepers half rebound.
  for (const [state, source, target] of pending) {
    state.source = source;
    state.target = target;
  }
  return pending.length;
}
