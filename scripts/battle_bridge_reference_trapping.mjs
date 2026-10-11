// Refresh only request-derived trapping caches, not a turn, residual or chance draw.
// The packed-team opening lead can differ from the PUBLIC active at the branch.
// Copying that opening lead's cache lets a benched Shadow Tag user trap a root.
export function refreshReferenceTrapping(battle, publicState) {
  if (battle.gen !== 3) throw new Error('Reference trapping requires Gen 3.');
  const seed = JSON.stringify(battle.prng.getSeed());
  for (const pokemon of battle.getAllActive()) {
    pokemon.trapped = pokemon.maybeTrapped = false;
    // This is the trapping portion of the pinned simulator's nextTurn, without
    // advancing its clocks, PP, residuals, activeTurns or request boundary.
    battle.runEvent('TrapPokemon', pokemon);
    if (!pokemon.knownType || battle.dex.getImmunity('trapped', pokemon)) {
      battle.runEvent('MaybeTrapPokemon', pokemon);
    }
    for (const source of pokemon.foes()) {
      const species = (source.illusion || source).species;
      if (!species.abilities) continue;
      for (const abilitySlot in species.abilities) {
        const name = species.abilities[abilitySlot];
        if (name === source.ability) continue;
        const rules = battle.ruleTable;
        if ((rules.has('+hackmons') || !rules.has('obtainableabilities')) && !battle.format.team) continue;
        if (abilitySlot === 'H' && species.unreleasedHidden) continue;
        const ability = battle.dex.abilities.get(name);
        if (rules.has('-ability:' + ability.id)) continue;
        if (pokemon.knownType && !battle.dex.getImmunity('trapped', pokemon)) continue;
        battle.singleEvent('FoeMaybeTrapPokemon', ability, {}, pokemon, source);
      }
    }
  }
  // A cancelled switch may have disclosed an otherwise hidden ability trap.
  // Preserve that PUBLIC disclosure only when the rebuilt world supplies the
  // cause; never manufacture a trap from the retained actor request itself.
  const actor = battle.sides[publicState.selfPlayer === 'p1' ? 0 : 1].active[0];
  if (publicState.selfActiveRequestState?.trapped === true && actor.trapped === 'hidden') {
    actor.trapped = true;
  }
  // The Gen 3 trap handlers are deterministic. A new/random handler must refuse
  // here rather than silently spend or reset the continuation's chance stream.
  if (JSON.stringify(battle.prng.getSeed()) !== seed) {
    throw new Error('Reference trap reconstruction consumed chance RNG.');
  }
}
