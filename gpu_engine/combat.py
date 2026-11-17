"""Selected Gen 3 combat and ability-trapping rules for the sampled spike.

Move traps, Baton Pass, ability replacement/suppression and the wider ability
surface remain separate mechanics. IDs come from the accepted pool vocabulary.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import mechanics as m


class CombatTable(NamedTuple):
    guts: jax.Array
    thick_fat: jax.Array
    arena_trap: jax.Array
    magnet_pull: jax.Array
    shadow_tag: jax.Array
    levitate: jax.Array
    flying_type: jax.Array
    steel_type: jax.Array
    fire_type: jax.Array
    ice_type: jax.Array
    doubled_attack: jax.Array
    hustle: jax.Array
    physical_type: jax.Array
    compound_eyes: jax.Array
    # Fidelity tier D (appended; every field above keeps its position and value).
    wonder_guard: jax.Array      # [A], gen4 wonderguard onTryHit (inherited by Gen 3)
    pinch_type: jax.Array        # [A], the type Blaze/Torrent/Overgrow/Swarm boost (0: none)
    status_move: jax.Array       # [M], category Status (exempt from Wonder Guard)
    # Damage modifiers (appended): Marvel Scale (ModifyDef) and Gen 3's Plus/Minus (ModifySpA).
    marvel_scale: jax.Array      # [A], base abilities.ts marvelscale onModifyDef (no gen3-8 override)
    plus: jax.Array              # [A], data/mods/gen3/abilities.ts plus onModifySpA
    minus: jax.Array             # [A], data/mods/gen3/abilities.ts minus onModifySpA


PINCH_TYPES = {"blaze": "fire", "torrent": "water", "overgrow": "grass", "swarm": "bug"}


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    flag = lambda name:jnp.asarray([value == name for value in vocab["abilities"]])
    typ = lambda name:jnp.asarray(vocab["types"].index(name))
    # Wonder Guard exempts Status moves by category, so a vocabulary with Wonder Guard needs the move rows; without
    # Wonder Guard (ability-only fixtures, or a table with no moves) status_move is never read and stays all False.
    names = vocab.get("moves", ())
    rows = metadata.get("moves")
    if rows is not None and len(rows) != len(names):
        raise ValueError("move metadata/vocabulary mismatch")
    if rows is None and "wonderguard" in vocab["abilities"] and names:
        raise ValueError("Wonder Guard needs the move rows (category) to exempt Status moves")
    # (At least one row: a table with no move vocabulary still gathers status_move[move] in the step.)
    status_move = ([bool(row) and row.get("category") == "Status" for row in rows] if rows is not None
        else [False]*max(1, len(names)))
    return CombatTable(flag("guts"),flag("thickfat"),flag("arenatrap"),flag("magnetpull"),
        flag("shadowtag"),flag("levitate"),typ("flying"),typ("steel"),typ("fire"),typ("ice"),
        jnp.asarray([name in {"hugepower","purepower"} for name in vocab["abilities"]]),
        flag("hustle"),jnp.asarray([name in {"normal","fighting","flying","poison",
            "ground","rock","bug","ghost","steel"} for name in vocab["types"]]),flag("compoundeyes"),
        flag("wonderguard"),
        jnp.asarray([vocab["types"].index(PINCH_TYPES[name]) if name in PINCH_TYPES else 0
            for name in vocab["abilities"]],jnp.int32),
        jnp.asarray(status_move,bool),
        flag("marvelscale"),flag("plus"),flag("minus"))


def attack_modifier(table,ability,status,physical,*,item_num=1,item_den=1,foe_ability=None,foe_hp=None):
    """One ModifyAtk/ModifySpA event chain, not sequentially rounded ability/item damage. Returns (q, 4096).

    runEvent sorts the handlers by priority (sim/battle.ts comparePriority): the ability (Guts, Huge Power, Pure Power:
    onModifyAtkPriority 5) before the item (priority 1); on SpA the item (Light Ball, Soul Dew: priority 1) before Plus/Minus
    (data/mods/gen4/abilities.ts clears their priority: 0). Each handler's chainModify folds into the event modifier as
    (prev*next + 2048) >> 12 (sim/battle.ts chainModify), and runEvent applies it once to the stat. At most one ability
    factor applies (Guts/Huge Power are physical, Plus/Minus special), so the chain is (ability_q*item_q + 2048) >> 12.
    For the exact factors (4096, 6144, 8192) it is the rational product; Silk Scarf's 4505 is not exact, so the item
    fraction (items.attack_modifier) is taken to Q12 first (num*4096//den, as chainModify truncates it) and chained.

    Damage modifiers: Gen 3's Plus/Minus (data/mods/gen3/abilities.ts:115-134) chainModify(1.5) on SpA when any active
    in getAllActive() (both sides; !fainted) has the partner ability: in singles the OPPOSING active. foe_ability is the
    foe active's current ability (a Trace copy included) and foe_hp its HP (> 0: not fainted); without them Plus/Minus
    do not apply (the confusion self-hit, which is physical)."""
    physical = jnp.asarray(physical)
    activated = table.guts[ability] & (status != m.NONE) & physical
    doubled = table.doubled_attack[ability] & physical
    partner = jnp.zeros_like(activated)
    if foe_ability is not None:
        partner = (~physical & (jnp.asarray(foe_hp) > 0)
            & ((table.plus[ability] & table.minus[foe_ability]) | (table.minus[ability] & table.plus[foe_ability])))
    ability_q = jnp.where(activated | partner,6144,jnp.where(doubled,8192,4096))
    item_q = jnp.asarray(item_num)*4096//jnp.asarray(item_den)
    return (ability_q*item_q+2048)>>12,jnp.full_like(ability_q,4096)


def defense_modifier(table,ability,status,physical,*,item_num=1,item_den=1):
    """Damage modifiers: Marvel Scale (base abilities.ts:2488-2498, no gen3-8 override): onModifyDefPriority 6,
    `if (pokemon.status) return this.chainModify(1.5)`: any major status (par, brn, psn, tox, slp, frz), Def only, so a
    physical hit (Gen 3's category by type). The holder's own confusion self-hit runs getDamage(pokemon, pokemon, 40)
    (data/mods/gen4/conditions.ts confusion), so ModifyDef applies there too. It composes with the item's (Soul Dew SpD)
    fraction, which never shares the stat, so the product stays exact. Explosion's halving follows (gen3_damage)."""
    scaled = table.marvel_scale[ability] & (status != m.NONE) & jnp.asarray(physical)
    return item_num*jnp.where(scaled,3,1),item_den*jnp.where(scaled,2,1)


def base_power(table,ability,move_type,power,*,attacker_ability=None,hp=None,max_hp=None):
    """The Gen 3 BasePower event's ability handlers: one Q12 chain, applied once (sim/battle.ts runEvent).

    Thick Fat (gen4 onSourceBasePower, priority 1) halves Fire and Ice: it modifies BasePower, not Atk/SpA or
    final damage. Fidelity tier D: Blaze, Torrent, Overgrow and Swarm (data/mods/gen4/abilities.ts blaze 25-35,
    overgrow, swarm, torrent; Gen 3 inherits them and drops the Atk/SpA handlers) are onBasePowerPriority 2,
    chainModify(1.5) when the move's (resolved: Hidden Power's own) type matches and attacker.hp <= attacker.maxhp/3,
    i.e. 3*hp <= maxhp. Both handlers chain (chainModify: (prev*next+2048)>>12) before runEvent's single modify:
    1.5 then 0.5 is 3072/4096, which rounds differently from two separate modifies (a 69 power HP Fire: 52, not
    51). Without the attacker's operands only Thick Fat applies (the pre-tier-D call)."""
    halved = table.thick_fat[ability] & ((move_type == table.fire_type) | (move_type == table.ice_type))
    if attacker_ability is None:
        return m.modify(power,1,jnp.where(halved,2,1))
    pinch = pinch_boost(table,attacker_ability,move_type,hp,max_hp)
    chained = (jnp.where(pinch,6144,4096)*jnp.where(halved,2048,4096)+2048)>>12
    return m.modify(power,chained,4096)


def pinch_boost(table,ability,move_type,hp,max_hp):
    """[B] Fidelity tier D: the attacker's pinch ability boosts this move (see base_power)."""
    boosted = table.pinch_type[ability]
    return (boosted > 0) & (move_type == boosted) & (3*jnp.asarray(hp) <= jnp.asarray(max_hp))


def wonder_guard_blocks(table,ability,move,move_type,num,den):
    """[B] Fidelity tier D: Wonder Guard on the target blocks this hit (data/mods/gen4/abilities.ts wonderguard
    onTryHit 530-543, inherited by Gen 3).

    It returns early for a Status move and for the typeless '???' (Struggle's onModifyMove; resolved type 0 here);
    otherwise `target.runEffectiveness(move) <= 0 || !target.runImmunity(move)` prints
    -immune|TARGET|[from] ability: Wonder Guard and returns null. runEffectiveness sums the log2 type factors and
    counts an immune type as 0, and runImmunity catches the immunity, so with the engine's exact product num/den it
    blocks exactly when num <= den (an immune product is 0). Fixed-damage moves (Seismic Toss) and Counter/Mirror
    Coat are Physical/Special, so they are blocked by type like any other hit. The caller resolves the type
    (Hidden Power's own) and reads the target's current ability (a Traced Wonder Guard included)."""
    return (table.wonder_guard[ability] & ~table.status_move[move] & (jnp.asarray(move_type) != 0)
        & (jnp.asarray(num) <= jnp.asarray(den)))


def switch_trapped(table,ability,types,hp):
    """Resolved active singles ability traps, shaped [B,2].

    Gen 3 Shadow Tag traps other Shadow Tag holders too; Ghost immunity to
    trapping is later-generation behavior. The caller permits forced switches.
    """
    foe = ability[:,::-1]
    grounded = ~jnp.any(types == table.flying_type,axis=-1) & ~table.levitate[ability]
    steel = jnp.any(types == table.steel_type,axis=-1)
    affected = (table.shadow_tag[foe] | (table.magnet_pull[foe] & steel)
        | (table.arena_trap[foe] & grounded))
    return affected & (hp > 0) & (hp[:,::-1] > 0)
