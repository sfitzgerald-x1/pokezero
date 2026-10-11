// State.deserializeWithRefs recursively creates fresh objects and arrays. The
// pinned simulator's exceptions are Pokemon.set and Battle.log: they are used
// directly, including by the Battle constructor. Copy those aliases before
// deserializing instead of copying the entire serialized graph twice.
// Only JSON-normalized, bridge-owned hypothetical snapshots use this helper.
// Requests remain separately copied by restoreSerializedBattle.
export function copyDeserializationAliases(state) {
  if (!state || typeof state !== "object" || !Array.isArray(state.sides)) {
    throw new TypeError("Owned search restore requires a serialized battle.");
  }
  const copy = {
    ...state,
    sides: state.sides.map(side => ({
      ...side,
      pokemon: side.pokemon.map(pokemon => ({
        ...pokemon,
        set: structuredClone(pokemon.set),
      })),
    })),
  };
  if (Object.hasOwn(state, "log")) copy.log = structuredClone(state.log);
  return copy;
}
