// Gen 3 clocks disclosed by the public protocol, never by the true opponent.
export function applyReferenceTurnClocks(pokemon, publicSide, sideId, gen) {
  if (gen !== 3 || !['p1', 'p2'].includes(sideId) || typeof publicSide.mustRecharge !== 'boolean') {
    throw new Error('Reference turn clocks require explicit Gen 3 public recharge provenance.');
  }
  if (publicSide.mustRecharge) {
    if (pokemon.fainted) throw new Error('Reference recharge cannot belong to a fainted active.');
    // Hyper Beam's onStart creates duration 2. At the next actionable request
    // one residual has already consumed a tick; the owed move attempt is next.
    pokemon.volatiles.mustrecharge = {id: 'mustrecharge', effectOrder: 0, duration: 1,
      target: `[Pokemon:${sideId}a]`};
  }
  const ability = String(pokemon.ability).toLowerCase().replace(/[^a-z0-9]/g, '');
  if (ability === 'truant') {
    if (typeof publicSide.truantPhase !== 'boolean') {
      throw new Error('Reference Truant requires a publicly identifiable residual phase.');
    }
    // Gen 3 uses Pokemon.truantTurn, not the later-generation truant volatile.
    // Its residual toggles even on sleep, recharge and other skipped actions.
    pokemon.truantTurn = publicSide.truantPhase;
  }
  // Original Choice Band hypotheses must keep their public last-move lock
  // AFTER recharge too. Copied disabled flags alone expire on nextTurn and
  // otherwise permit impossible multi-move continuations. Post-Trick acquisition
  // needs a separate chronology certificate, so never guess that case here.
  if (String(pokemon.item).toLowerCase() === 'choiceband') {
    const row = publicSide.pokemon?.find(row => row.active);
    if (row?.currentItem !== undefined) {
      throw new Error('Reference Choice Band acquisition requires an item-mutation chronology certificate.');
    }
    const move = String(publicSide.lastUsedMove || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    if (move && move !== 'switch') {
      if (!pokemon.moveSlots.some(slot => slot.id === move)) {
        throw new Error('Reference Choice Band public last move is absent from sampled move slots.');
      }
      pokemon.volatiles.choicelock = {id: 'choicelock', effectOrder: 0,
        target: `[Pokemon:${sideId}a]`, move};
    }
  }
}

export function applyReferenceRechargePP(snapshot, publicState) {
  const sideId = publicState.selfPlayer;
  if (!publicState.sides[sideId].mustRecharge) return;
  const charge = publicState.selfRechargePPCharge;
  if (!charge || charge.refusal || charge.move !== 'hyperbeam' ||
      !['p1', 'p2'].includes(charge.targetSide) || charge.targetSide === sideId ||
      typeof charge.targetSpecies !== 'string') {
    throw new Error(charge?.refusal || 'Reference recharge lacks a certified actor PP charge.');
  }
  const normalized = value => String(value).toLowerCase().replace(/[^a-z0-9]/g, '');
  const actorSide = snapshot.sides[sideId === 'p1' ? 0 : 1];
  const targetSide = snapshot.sides[charge.targetSide === 'p1' ? 0 : 1];
  const targets = targetSide.pokemon.filter(mon => normalized(mon.set.species) === normalized(charge.targetSpecies));
  if (targets.length !== 1) throw new Error('Reference recharge target must have a unique sampled identity.');
  const actor = actorSide.pokemon.find(mon => mon.isActive);
  const slot = actor?.moveSlots.find(move => move.id === 'hyperbeam');
  if (!slot || !Number.isInteger(slot.pp) || slot.pp < 1) {
    throw new Error('Reference recharge lacks the pre-use actor Hyper Beam PP bank.');
  }
  const cost = normalized(targets[0].ability) === 'pressure' ? 2 : 1;
  slot.pp = Math.max(0, slot.pp - cost);
  slot.used = true;
  // The native serialized original bank can be a separate array. Keep both
  // consistent; do not debit twice when serialization aliases the same slot.
  const original = actor.baseMoveSlots?.find(move => move.id === 'hyperbeam');
  if (original && original !== slot) { original.pp = slot.pp; original.used = true; }
}
