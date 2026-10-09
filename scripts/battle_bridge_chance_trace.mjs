// Observational instrumentation of one ORIGINAL hypothetical transition.
// No random values, abilities, choices, or outcomes are substituted.
export function installChanceTrace(battle) {
  const undo = [], records = [];
  let chance = null, range = null, damage = null, damageMove = null, damageTarget = null, ordinal = 0;
  let volatile = null;
  let pendingAccuracy = null, accuracy = null;
  function wrap(owner, name, wrapper) {
    const descriptor = Object.getOwnPropertyDescriptor(owner, name);
    const original = owner[name];
    if (typeof original !== 'function') throw new Error(`Missing native trace method ${name}.`);
    owner[name] = wrapper(original);
    undo.push(() => descriptor ? Object.defineProperty(owner, name, descriptor) : delete owner[name]);
  }
  try {
    const prng = battle.prng;
    wrap(prng.rng, 'next', original => function (...args) {
      // An intervening random draw invalidates an unconsumed Accuracy event.
      pendingAccuracy = null;
      const value = original.apply(this, args);
      records.push({ordinal: ++ordinal, value, chance, range, damage: damage ? {...damage} : null,
        volatile: volatile ? {...volatile} : null, accuracy: accuracy ? {...accuracy} : null});
      return value;
    });
    wrap(prng, 'random', original => function (...args) {
      pendingAccuracy = null;
      const before = range; range = args;
      try { return original.apply(this, args); } finally { range = before; }
    });
    wrap(prng, 'randomChance', original => function (numerator, denominator) {
      const before = chance; chance = {numerator, denominator};
      const beforeAccuracy = accuracy;
      accuracy = pendingAccuracy?.numerator === numerator && denominator === 100 ? pendingAccuracy : null;
      pendingAccuracy = null;
      try { return original.call(this, numerator, denominator); } finally {
        chance = before; accuracy = beforeAccuracy;
      }
    });
    // Bind the returned native event threshold to its immediately following
    // coin, never to a generic denominator-100 secondary-effect roll. Native
    // TryHit (including Substitute) and public logging still run unchanged.
    wrap(battle, 'runEvent', original => function (event, target, source, move, ...args) {
      pendingAccuracy = null;
      try {
        const result = original.call(this, event, target, source, move, ...args);
        // An outer unrelated event must not leak its inner Accuracy event.
        pendingAccuracy = null;
        if (event === 'Accuracy' && typeof result === 'number' && Number.isFinite(result)
            && source?.side?.id && target?.side?.id && move?.id) {
          pendingAccuracy = {source: source.side.id, target: target.side.id, move: move.id,
            source_ident: source.toString(), target_ident: target.toString(),
            numerator: result, denominator: 100};
        }
        return result;
      } catch (error) {
        pendingAccuracy = null;
        throw error;
      }
    });
    wrap(battle.actions, 'tryMoveHit', original => function (...args) {
      pendingAccuracy = null;
      try { return original.apply(this, args); } finally { pendingAccuracy = null; }
    });
    // Context only: native addVolatile still owns immunity, duration and Start.
    for (const side of battle.sides) for (const pokemon of side.pokemon) {
      wrap(pokemon, 'addVolatile', original => function (status, ...args) {
        const before = volatile;
        volatile = {id: typeof status === 'string' ? status : status.id,
          target: this.side.id, ident: this.toString(),
          after_target_acted: !battle.queue.willMove(this)};
        try { return original.call(this, status, ...args); } finally { volatile = before; }
      });
    }
    wrap(battle.actions, 'getDamage', original => function (source, target, move, ...args) {
      const before = damage;
      const beforeMove = damageMove, beforeTarget = damageTarget;
      damageMove = typeof move === 'object' ? move : battle.activeMove;
      damageTarget = target;
      damage = {source: source.side.id, target: target.side.id,
        move: typeof move === 'string' ? move : move.id, target_hp: target.hp, target_max_hp: target.maxhp};
      try { return original.call(this, source, target, move, ...args); } finally {
        damage = before; damageMove = beforeMove; damageTarget = beforeTarget;
      }
    });
    wrap(battle, 'randomizer', original => function (baseDamage) {
      const before = damage;
      // The active move-hit data already exists in native getDamage. Read only
      // whether its native critical event was accepted; never force that event.
      if (damage) {
        const hit = damageMove?.moveHitData?.[damageTarget?.getSlot()];
        damage = {...damage, base_damage: baseDamage, critical: Boolean(hit?.crit)};
      }
      try { return original.call(this, baseDamage); } finally { damage = before; }
    });
    return {records, close() { for (const restore of undo.reverse()) restore(); }};
  } catch (error) {
    for (const restore of undo.reverse()) restore();
    throw error;
  }
}
