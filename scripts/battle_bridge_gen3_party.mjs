// Reuse only immutable Gen 3 randbat setup in stateless membership rejection.
// Each complete party still runs the ORIGINAL randomTeam with a fresh PRNG and
// fresh battle-wide flags. No species, sets, proposal order or seeds are forced.
export function createIsolatedGen3PartyGenerator(Teams) {
  let generator;
  return function generate(parts) {
    if (!Array.isArray(parts) || parts.length !== 4 ||
        parts.some(part => !Number.isInteger(part) || part < 0 || part > 0xffff)) {
      throw new Error("Invalid original Gen 3 party seed.");
    }
    if (!generator) {
      generator = Teams.getGenerator("gen3randombattle", parts);
      if (generator.constructor.name !== "RandomGen3Teams" ||
          generator.gen !== 3 || generator.format.id !== "gen3randombattle" ||
          generator.maxTeamSize !== 6 || generator.forceMonotype ||
          generator.battleHasDitto !== false || generator.battleHasWobbuffet !== false ||
          typeof generator.getPokemonPool !== "function" ||
          typeof generator.setSeed !== "function") {
        throw new Error("Unsupported isolated Gen 3 party generator.");
      }
      const list = Object.keys(generator.randomSets);
      const before = generator.prng.getSeed();
      const original = generator.getPokemonPool;
      const [pool, bases] = original.call(generator, "", [], false, list);
      if (generator.prng.getSeed() !== before) {
        throw new Error("Gen 3 species pool construction consumed random state.");
      }
      for (const value of Object.values(pool)) Object.freeze(value);
      Object.freeze(pool);
      Object.freeze(bases);
      generator.getPokemonPool = function (type, excluded = [], monotype = false, supplied) {
        // The original non-monotype, empty-exclusion path ignores type. Only
        // that exact deterministic pool is reusable; unexpected inputs refuse.
        if (monotype !== false || excluded.length !== 0 ||
            !Array.isArray(supplied) || supplied.length !== list.length ||
            supplied.some((species, index) => species !== list[index])) {
          throw new Error("Isolated Gen 3 party pool contract drift.");
        }
        // randomTeam mutates the base-species sampling array, not the read-only
        // species map. Never share that mutable array between proposals.
        return [pool, bases.slice()];
      };
    }
    generator.setSeed(parts);
    generator.battleHasDitto = false;
    generator.battleHasWobbuffet = false;
    return generator.getTeam({seed: parts});
  };
}
