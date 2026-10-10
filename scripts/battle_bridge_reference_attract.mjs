// Gen 3 Attract: public source ownership, not a guessed timer or chance roll.
const normalize = value => String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '');
export function bindReferenceAttract(snapshot, publicState, gen) {
  if (gen !== 3) throw new Error('Reference Attract requires Gen 3.');
  for (const sideId of ['p1', 'p2']) {
    const publicSide = publicState.sides[sideId];
    const certificate = publicSide.referenceAttract;
    const present = publicSide.volatiles.some(value => normalize(value) === 'attract');
    if (!present) {
      if (certificate != null) throw new Error('Reference Attract has a certificate without its volatile.');
      continue;
    }
    const other = sideId === 'p1' ? 'p2' : 'p1';
    if (!certificate || certificate.sourceSide !== other ||
        !['attract', 'cutecharm'].includes(certificate.cause) ||
        typeof certificate.sourceIdent !== 'string' || !certificate.sourceIdent.startsWith(other + 'a: ') ||
        typeof certificate.targetIdent !== 'string' || !certificate.targetIdent.startsWith(sideId + 'a: ')) {
      throw new Error('Reference Attract lacks certified opposing source ownership.');
    }
    const bind = (owner, species) => {
      const side = snapshot.sides.find(row => row.id === owner);
      const known = publicState.sides[owner].pokemon.filter(row => row.active);
      const matches = side.pokemon.map((pokemon, index) => ({pokemon, index})).filter(({pokemon}) =>
        pokemon.isActive && normalize(pokemon.set.species || pokemon.set.name) === normalize(species));
      if (typeof species !== 'string' || !species.trim() || known.length !== 1 ||
          normalize(known[0].species) !== normalize(species) || matches.length !== 1) {
        throw new Error('Reference Attract source/target does not match the public active.');
      }
      if (matches[0].pokemon.fainted || !Number.isFinite(matches[0].pokemon.hp) ||
          matches[0].pokemon.hp <= 0) {
        throw new Error('Reference Attract refuses an uncertified fainted endpoint frontier.');
      }
      return {...matches[0], ref: `[Pokemon:${owner}${'abcdef'[matches[0].index]}]`};
    };
    const target = bind(sideId, certificate.targetSpecies);
    const source = bind(other, certificate.sourceSpecies);
    if (!((target.pokemon.gender === 'M' && source.pokemon.gender === 'F') ||
          (target.pokemon.gender === 'F' && source.pokemon.gender === 'M'))) {
      throw new Error('Reference Attract sampled genders contradict its public start.');
    }
    target.pokemon.volatiles.attract = {id: 'attract', effectOrder: 0,
      target: target.ref, source: source.ref, sourceSlot: other + 'a'};
  }
}
