// The original ordered unknown-completion kernel, batched without changing seeds.
// No battle/private state, forced species, reordered proposals, or set mutation.
export function canonicalPartySpecies(value) {
  const id = String(value).toLowerCase().replace(/[^a-z0-9]/g, "");
  return /^unown(?:[a-z]|exclamation|question)$/.test(id) ? "unown" : id;
}

export function rejectUnknownMembership(command, generate, now = () => performance.now()) {
  const {seeds, known, required, maxProposals, budgetMs} = command;
  if (!Array.isArray(seeds) || seeds.length < 10 || seeds.length > 160
      || !seeds.every(s => Number.isInteger(s) && s >= 0 && s <= 0xffffffff)
      || !Array.isArray(known) || known.length > 6 || new Set(known).size !== known.length
      || !Array.isArray(required) || !required.length || required.length > 6
      || [...known, ...required].some(s => typeof s !== "string" || !s || canonicalPartySpecies(s) !== s)
      || !Number.isInteger(maxProposals) || maxProposals < 1 || maxProposals > 16
      || (budgetMs !== null && (!Number.isFinite(budgetMs) || budgetMs <= 0))) {
    throw new Error("Invalid native unknown-membership command.");
  }
  const began = now();
  const result = {status: "EXHAUSTED_BATCH", consumed: 0, completeProposals: 0,
    membershipRejections: 0, unknown: [], partySeeds: []};
  const expired = () => budgetMs !== null && now() - began >= budgetMs;
  for (let proposal = 0; proposal < maxProposals; proposal++) {
    // Reserve the ORIGINAL ten-party safety cap; unused seeds are not consumed.
    if (seeds.length - result.consumed < 10) break;
    const seen = new Set(known), unknown = [], partySeeds = [];
    while (seen.size < 6) {
      if (expired()) return {...result, status: "DEADLINE_CANCELLED"};
      if (partySeeds.length >= 10) {
        throw new Error("unknown-species fresh-party rejection safety cap exceeded");
      }
      const seed = seeds[result.consumed++];
      partySeeds.push(seed);
      const party = generate(seed);
      if (!Array.isArray(party) || party.length !== 6) {
        throw new Error("unknown-species generator returned a partial party");
      }
      for (const row of party) {
        if (!row || typeof row.species !== "string" || !canonicalPartySpecies(row.species)) {
          throw new Error("unknown-species generator returned invalid species");
        }
        const species = canonicalPartySpecies(row.species);
        if (!seen.has(species)) {seen.add(species); unknown.push(row);}
        if (seen.size === 6) break;
      }
    }
    result.completeProposals++;
    if (expired()) return {...result, status: "DEADLINE_CANCELLED"};
    if (required.every(s => seen.has(s))) {
      return {...result, status: "ACCEPTED", unknown, partySeeds};
    }
    result.membershipRejections++;
  }
  return result;
}
