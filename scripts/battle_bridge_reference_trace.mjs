// Opt-in public certificates, after FINAL party ordering. No events are run.
// Only these four Gen 3 copies are stateless and lack ordering-sensitive
// SwitchIn/redirection callbacks. This is behavioral restoration, NOT an
// assertion that a fresh effectOrder equals the historical serialized number.
const STATELESS = new Set(["waterabsorb", "voltabsorb", "levitate", "pressure"]);
const id = value => typeof value === "string" ? value.toLowerCase().replace(/[^a-z0-9]/g, "") : "";

export function restoreReferenceTrace(snapshot, assignments) {
  const fail = () => { throw new Error("Reference Trace needs a complete stateless public certificate."); };
  if (!snapshot || !["gen3randombattle", "gen3customgame"].includes(snapshot.formatid) ||
      !Array.isArray(snapshot.sides) || snapshot.sides.length !== 2 ||
      !Number.isSafeInteger(snapshot.effectOrder) || snapshot.effectOrder < 0 ||
      !Array.isArray(assignments) || assignments.length > 12) fail();
  const entries = new Map();
  for (const [s, side] of snapshot.sides.entries()) {
    if (side?.id !== ["p1", "p2"][s] || !Array.isArray(side.pokemon) ||
        !side.pokemon.length || side.pokemon.length > 6) fail();
    const species = new Set();
    for (const [p, pokemon] of side.pokemon.entries()) {
      const name = id(pokemon?.set?.species);
      if (!name || species.has(name) || pokemon.position !== p) fail();
      species.add(name);
      if (id(pokemon.set.ability) !== "trace") continue;
      if (pokemon.baseAbility !== "trace" || typeof pokemon.isActive !== "boolean" ||
          typeof pokemon.fainted !== "boolean" || !Number.isFinite(pokemon.hp)) fail();
      const reference = `[Pokemon:${side.id}${"abcdef"[p]}]`;
      const active = pokemon.isActive && !pokemon.fainted && pokemon.hp > 0;
      if (active && (p !== 0 || side.active?.length !== 1 || side.active[0] !== reference)) fail();
      entries.set(`${side.id}:${name}`, {pokemon, active, reference});
    }
  }
  if (entries.size !== assignments.length) fail();
  const seen = new Set(), pending = [];
  let order = snapshot.effectOrder;
  for (const row of assignments) {
    if (!row || typeof row !== "object" || Array.isArray(row) ||
        Object.keys(row).sort().join(",") !== "ability,active,side,species" ||
        !["p1", "p2"].includes(row.side) || typeof row.species !== "string" ||
        row.species !== id(row.species) || !row.species ||
        typeof row.active !== "boolean" ||
        (row.ability !== "trace" && !STATELESS.has(row.ability))) fail();
    const key = `${row.side}:${row.species}`, entry = entries.get(key);
    if (!entry || seen.has(key) || row.active !== entry.active ||
        (!row.active && row.ability !== "trace")) fail();
    seen.add(key);
    // A sampled warm lead may have copied Truant from a different opponent.
    // The Python proof admits only holders whose entire source history never
    // copied it; Pokemon initializes this independent persistent field false.
    // Overwrite it deliberately, not with arbitrary warm-template state.
    pending.push({pokemon: entry.pokemon, ability: row.ability,
      abilityState: {id: row.ability, target: entry.reference, effectOrder: row.active ? order++ : 0}});
  }
  if (!Number.isSafeInteger(order)) fail();
  // Atomic: every roster row and snapshot identity validated before changes.
  for (const row of pending) {
    row.pokemon.ability = row.ability;
    row.pokemon.abilityState = row.abilityState;
    row.pokemon.truantTurn = false;
  }
  snapshot.effectOrder = order;
  return pending.length;
}
