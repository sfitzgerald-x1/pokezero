"""Vectorized Gen 3 arithmetic/status components for the lockstep transition.

Inputs are arrays or broadcastable scalars. Chance draws are supplied by the
per-battle stream, not drawn conditionally here. These components are not a full
battle step: move effects, event ordering, switching and ability/item resolution
must be integrated before any move or game-fidelity coverage is claimed.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

NONE, PAR, BRN, PSN, TOX, SLP, FRZ = range(7)


def modify(value, numerator=1, denominator=1):
    """Pinned Showdown Q12 multiply, including unsigned-32 product truncation.

    Nonnegative values and positive denominators are required. Split the wrapped
    product's quotient/remainder so rounding does not introduce another wrap.
    """
    value = jnp.asarray(value, jnp.uint32)
    factor = (jnp.asarray(numerator, jnp.uint32) * 4096
              // jnp.asarray(denominator, jnp.uint32))
    product = value * factor
    return ((product >> 12) + (((product & 4095) + 2047) >> 12)).astype(jnp.int32)


def boosted_stat(stat, stage):
    stage = jnp.clip(jnp.asarray(stage, jnp.int32), -6, 6)
    numerator = jnp.where(stage >= 0, 2 + stage, 2)
    denominator = jnp.where(stage >= 0, 2, 2 - stage)
    return modify(jnp.asarray(stat, jnp.int32) * numerator // denominator)


def gen3_damage(*, level, power, attack, defense, attack_stage=0, defense_stage=0,
                physical=True, critical=False, stab=False, typeless=False, burned=False, guts=False,
                attack_num=1, attack_den=1, defense_num=1, defense_den=1,
                weather_num=1, weather_den=1, phase1_num=1, phase1_den=1, screen=False,
                effectiveness_num=1, effectiveness_den=1, immune=False, roll=100):
    """Singles damage with resolved stat/weather/type modifiers and roll 85..100.

    Caller resolves move category (by type in Gen 3), item/ability stat modifiers,
    type immunity and special move rules. Fixed-damage moves do not use this
    function. Crits ignore only adverse offense and favorable defense stages.
    Mirrors gen3/scripts.ts: burn/screens/weather BEFORE +2, then crit, STAB,
    net type effectiveness and finally the random roll.
    """
    atk_stage = jnp.where(critical & (jnp.asarray(attack_stage) < 0), 0, attack_stage)
    def_stage = jnp.where(critical & (jnp.asarray(defense_stage) > 0), 0, defense_stage)
    atk = modify(boosted_stat(attack, atk_stage), attack_num, attack_den)
    defe = jnp.maximum(1, modify(boosted_stat(defense, def_stage), defense_num, defense_den))
    base = (((jnp.asarray(level, jnp.uint32) * 2 // 5 + 2)
             * jnp.asarray(power, jnp.uint32) * atk.astype(jnp.uint32))
            // defe.astype(jnp.uint32) // 50).astype(jnp.int32)
    base = jnp.where(physical & burned & ~jnp.asarray(guts), modify(base, 1, 2), base)
    # Screen and Flash Fire share ONE ModifyDamagePhase1 event. Compose the
    # Q12 chain first; rounding each handler's damage separately is incorrect.
    screen_active = screen & ~jnp.asarray(critical)
    phase_factor = (jnp.asarray(phase1_num,jnp.uint32)*4096
                    // jnp.asarray(phase1_den,jnp.uint32))
    screen_factor = jnp.where(screen_active,jnp.uint32(2048),jnp.uint32(4096))
    phase_factor = (screen_factor*phase_factor + jnp.uint32(2048)) >> 12
    phase_active = screen_active | (jnp.asarray(phase1_num) != 1) | (jnp.asarray(phase1_den) != 1)
    base = jnp.where(phase_active,modify(base,phase_factor,4096),base)
    base = jnp.where((jnp.asarray(weather_num) != 1) | (jnp.asarray(weather_den) != 1),
                     modify(base, weather_num, weather_den), base)
    base = jnp.where(physical, jnp.maximum(1, base), base) + 2
    base = jnp.where(critical, modify(base, 2), base)
    # Typed moves run the Q12 STAB event even when its multiplier is identity.
    # Confusion/Struggle's ??? type skips that event altogether.
    base = jnp.where(typeless, base, modify(base, jnp.where(stab, 3, 1), jnp.where(stab, 2, 1)))
    base = base * effectiveness_num // effectiveness_den
    damage = jnp.maximum(1, (base.astype(jnp.uint32)
                            * jnp.asarray(roll, jnp.uint32) // 100).astype(jnp.int32))
    return jnp.where(immune | (jnp.asarray(power) <= 0), 0, damage)


def accuracy_hit(accuracy, accuracy_stage, evasion_stage, roll, *, always_hit=False):
    """Ordinary Gen 3 accuracy: supplied integer roll 0..99, no OHKO rules.

    Stages are applied independently, not netted and clamped as a single stage.
    Rational comparison avoids float32 threshold drift. Ability/item modifiers
    and semi-invulnerability are resolved by the caller.
    """
    a = jnp.clip(jnp.asarray(accuracy_stage, jnp.int32), -6, 6)
    e = jnp.clip(jnp.asarray(evasion_stage, jnp.int32), -6, 6)
    numerator = jnp.where(a >= 0, 3 + a, 3) * jnp.where(e >= 0, 3, 3 - e)
    denominator = jnp.where(a >= 0, 3, 3 - a) * jnp.where(e >= 0, 3 + e, 3)
    return jnp.asarray(always_hit) | (jnp.asarray(roll) * denominator < accuracy * numerator)


def paralyzed_speed(speed, paralyzed):
    return jnp.where(paralyzed, modify(speed, 1, 4), speed)


class BeforeMove(NamedTuple):
    status: jax.Array
    counter: jax.Array
    confusion_counter: jax.Array
    can_act: jax.Array
    confusion_self_hit: jax.Array


def before_move(status, counter, confusion_counter=0, *, thaw_roll=99,
                paralysis_roll=99, confusion_roll=0, flinched=False,
                sleep_usable=False, defrost_move=False, early_bird=False):
    """Sleep/freeze -> flinch -> confusion -> paralysis, with supplied 0..99 draws.

    Sleep starts at 2..5 (Rest: 3) and decrements before testing whether it ends.
    Confusion similarly starts at 2..5. A defrost move may pass freeze without
    curing it here: the move's later ModifyMove event performs that cure.
    Sleep Talk's switch-out skipped-time bookkeeping belongs to the future step.
    Confusion self-damage (40 BP physical, typeless) is returned as an event flag.
    """
    status, counter = jnp.asarray(status), jnp.asarray(counter)
    sleep = status == SLP
    remaining = jnp.maximum(0, counter - 1 - jnp.asarray(early_bird, jnp.int32))
    wake = sleep & (remaining == 0)
    thaw = (status == FRZ) & (jnp.asarray(thaw_roll) < 20)
    new_status = jnp.where(wake | thaw, NONE, status)
    new_counter = jnp.where(wake | thaw, 0, jnp.where(sleep, remaining, counter))
    blocked = ((sleep & ~wake & ~jnp.asarray(sleep_usable))
               | ((status == FRZ) & ~thaw & ~jnp.asarray(defrost_move))
               | jnp.asarray(flinched))
    confusion = jnp.asarray(confusion_counter)
    confusion = jnp.where(~blocked, jnp.maximum(0, confusion - 1), confusion)
    self_hit = ~blocked & (confusion > 0) & (jnp.asarray(confusion_roll) >= 50)
    paralyzed = (new_status == PAR) & (jnp.asarray(paralysis_roll) < 25)
    return BeforeMove(new_status, new_counter, confusion, ~(blocked | self_hit | paralyzed), self_hit)


class Residual(NamedTuple):
    hp: jax.Array
    counter: jax.Array
    damage: jax.Array


def major_status_residual(hp, max_hp, status, counter, *, active=True):
    """One eligible end-of-turn event, not a complete residual event scheduler.

    Burn/poison: floor(max HP/8), minimum 1. Toxic: floor(max HP/16), minimum 1,
    THEN multiply by stage (capped at 15). The order differs from flooring the
    final product. Switch-in resets toxic stage outside this component.
    """
    hp, max_hp, status, counter = map(jnp.asarray, (hp, max_hp, status, counter))
    eligible = jnp.asarray(active) & (hp > 0)
    toxic = (status == TOX) & eligible
    stage = jnp.minimum(15, counter + 1)
    regular = jnp.maximum(1, max_hp // 8)
    poison = jnp.maximum(1, max_hp // 16) * stage
    raw = jnp.where(status == TOX, poison, jnp.where((status == BRN) | (status == PSN), regular, 0))
    damage = jnp.where(eligible, jnp.minimum(hp, raw), 0)
    return Residual(hp - damage, jnp.where(toxic, stage, counter), damage)
