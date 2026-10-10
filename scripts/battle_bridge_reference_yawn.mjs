// Fixed Gen 3 public Yawn clock. Binding follows BOTH sampled party permutations.
const normalize = value => String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '');
export function bindReferenceYawn(snapshot, publicState, gen) {
  if (gen !== 3) throw new Error('Reference Yawn requires Gen 3.');
  for (const sideId of ['p1', 'p2']) {
    const pub = publicState.sides[sideId];
    const c = pub.referenceYawn;
    const present = pub.volatiles.some(v => normalize(v) === 'yawn');
    if (!present) {
      if (c != null) throw new Error('Reference Yawn has a certificate without its volatile.');
      continue;
    }
    const other = sideId === 'p1' ? 'p2' : 'p1';
    if (publicState.selfRequestKind !== 'move' || snapshot.midTurn) {
      throw new Error('Reference Yawn requires a Gen 3 ordinary move boundary.');
    }
    if (!c || c.sourceSide !== other || c.duration !== 1 || !Number.isInteger(c.start_turn) ||
        c.start_turn < 1 || publicState.turn !== c.start_turn + 1 ||
        !Array.isArray(c.residual_turns) || c.residual_turns.length !== 1 ||
        c.residual_turns[0] !== c.start_turn ||
        typeof c.sourceIdent !== 'string' || !c.sourceIdent.startsWith(other + 'a: ') ||
        typeof c.targetIdent !== 'string' || !c.targetIdent.startsWith(sideId + 'a: ')) {
      throw new Error('Reference Yawn lacks certified public clock/source ownership.');
    }
    const bind = (owner, species, target) => {
      if (typeof species !== 'string' || !species.trim()) throw new Error('Reference Yawn lacks species.');
      const known = publicState.sides[owner].pokemon.filter(p => normalize(p.species) === normalize(species));
      const side = snapshot.sides.find(s => s.id === owner);
      const matches = side.pokemon.map((p, index) => ({p, index})).filter(({p}) =>
        normalize(p.set.species || p.set.name) === normalize(species));
      if (known.length !== 1 || matches.length !== 1 || (target && !known[0].active)) {
        throw new Error('Reference Yawn source/target does not uniquely match public party.');
      }
      const {p, index} = matches[0];
      if (target && (!p.isActive || p.fainted || !Number.isFinite(p.hp) || p.hp <= 0)) {
        throw new Error('Reference Yawn target cannot carry pending delayed sleep.');
      }
      return {p, ref: `[Pokemon:${owner}${'abcdef'[index]}]`};
    };
    const target = bind(sideId, c.targetSpecies, true);
    const source = bind(other, c.sourceSpecies, false);
    target.p.volatiles.yawn = {id: 'yawn', name: 'Yawn', effectOrder: 0, duration: 1,
      sourceEffect: {hit: 0, move: '[Move:yawn]'},
      target: target.ref, source: source.ref, sourceSlot: other + 'a'};
  }
}
