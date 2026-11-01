"""Named public tracker scalars -> unchanged v4 column meanings.

Inputs are raw *observed* counters/flags, not private remaining timers. Missing
keys stay zero. No source reconstruction, event fold, or engine/world access.
"""
from typing import NamedTuple
import jax.numpy as j
from pokezero import showdown as s
from .public_active import NAMES as VOLATILE_NAMES
from .public_moves import _static_category
from pokezero.dex import normalize_id

# key -> (original numeric name, divisor, active-only). Raw fractions are already
# source public-HP fractions; elapsed counters are never remaining engine clocks.
MON_FIELDS = {
    'sleep_turns': ('SLEEP_TURNS', 5., False), 'rest_sleep': ('REST_SLEEP', 1., False),
    'turns_active': ('TURNS_ACTIVE', 64., True), 'toxic_stage': ('TOXIC_STAGE', 15., True),
    'stall_counter': ('STALL_COUNTER', 8., True), 'confusion_turns': ('CONFUSION_TURNS', 5., True),
    'encore_turns': ('ENCORE_TURNS', 6., True), 'wrap_trap_turns': ('WRAP_TRAP_TURNS', 5., True),
    'meanlook_trap': ('MEANLOOK_TRAP', 1., True), 'choice_locked': ('CHOICE_LOCKED', 1., True),
    'item_swapped': ('ITEM_SWAPPED', 1., True), 'truant_loaf': ('TRUANT_LOAF', 1., True),
    'last_damage_dealt': ('LAST_DAMAGE_DEALT', 1., True), 'last_damage_taken': ('LAST_DAMAGE_TAKEN', 1., True),
    'switched_before_attack': ('MON_SWITCHED_BEFORE_ATTACK', 64., False),
    'stayed_and_attacked': ('MON_STAYED_AND_ATTACKED', 64., False),
    'turns_active_total': ('MON_TURNS_ACTIVE_TOTAL', 64., False),
    'switched_vs_active': ('MON_SWITCHED_VS_ACTIVE', 8., False),
    'stayed_vs_active': ('MON_STAYED_VS_ACTIVE', 8., False),
}
FIELD_FIELDS = {
    'weather_turns': ('WEATHER_TURNS', 5.), 'weather_permanent': ('WEATHER_PERMANENT', 1.),
    'self_wish_pending': ('SELF_WISH_PENDING', 1.), 'opp_wish_pending': ('OPP_WISH_PENDING', 1.),
    'self_wish_turns': ('SELF_WISH_TURNS', 2.), 'opp_wish_turns': ('OPP_WISH_TURNS', 2.),
    'self_sleep_clause': ('SELF_SLEEP_CLAUSE', 1.), 'opp_sleep_clause': ('OPP_SLEEP_CLAUSE', 1.),
    'sleep_clause_blocks_self': ('SLEEP_CLAUSE_BLOCKS_SELF', 1.), 'sleep_clause_blocks_opp': ('SLEEP_CLAUSE_BLOCKS_OPP', 1.),
    'self_hazard_credit': ('SELF_HAZARD_CREDIT', 6.), 'opp_hazard_credit': ('OPP_HAZARD_CREDIT', 6.),
    'self_items_removed_credit': ('SELF_ITEMS_REMOVED_CREDIT', 6.), 'opp_items_removed_credit': ('OPP_ITEMS_REMOVED_CREDIT', 6.),
}
STATS_FIELDS = {
    'opp_switch_count': ('STAT_OPP_SWITCH_COUNT', 64.),
    'opp_decision_opportunities': ('STAT_OPP_DECISION_OPPORTUNITIES', 64.),
    'blocked_on_our_attack': ('STAT_BLOCKED_ON_OUR_ATTACK', 64.),
    'pursuit_intercept_predict': ('STAT_PURSUIT_INTERCEPT_PREDICT', 64.),
    'my_switch_turns': ('STAT_MY_SWITCH_TURNS', 64.),
}


class Tables(NamedTuple):
    volatile_categories: object
    last_move_categories: object
    switch_categories: object
    traced_abilities: object
    stats_role: object


def compile_tables(metadata, catalog):
    category = lambda text: _static_category(catalog.vocabulary, text) if text else 0
    return Tables(j.asarray([category('volatile:' + n) if n else 0 for n in VOLATILE_NAMES], j.int32),
        j.asarray([category('move:' + ('hiddenpower' if normalize_id(n).startswith('hiddenpower') else n)) if i else 0
            for i, n in enumerate(metadata['vocabulary']['moves'])], j.int32),
        j.asarray([category(s.LAST_USED_MOVE_SWITCH_SENTINEL), category(s.LAST_USED_MOVE_BATON_PASS_SENTINEL)], j.int32),
        j.asarray([category('ability:' + n) if i else 0 for i, n in enumerate(metadata['vocabulary']['abilities'])], j.int32),
        j.asarray(category('stats'), j.int32))


def _column(name):return s.v4_numeric_index(getattr(s, 'NUMERIC_' + name))


def project_mon(numeric, categorical, facts, *, active, present, tables, last_moves=None,
                borrowed_ability=None, volatiles=None, substitute_fraction=None):
    facts = {} if facts is None else facts
    bad = j.zeros((numeric.shape[0],), bool)
    for key, (name, divisor, active_only) in MON_FIELDS.items():
        if key not in facts:continue
        value = facts[key].astype(j.float32)
        if value.shape != present.shape:raise ValueError('public mon fact shape: ' + key)
        bad |= j.any(~j.isfinite(value) | (value < 0), axis=-1)
        use = present & active if active_only else present
        value = j.where(use, j.clip(j.nan_to_num(value) / divisor, 0., 1.), 0)
        # Sleep counters are zero whenever this public observer says not asleep.
        if key in ('sleep_turns', 'rest_sleep') and 'sleeping' in facts:
            value = j.where(facts['sleeping'], value, 0)
        numeric = numeric.at[..., _column(name)].set(value)
    if substitute_fraction is not None:
        numeric = numeric.at[..., _column('SUB_HP_FRACTION')].set(j.where(present & active, substitute_fraction, 0))
    if last_moves is not None:
        valid = (last_moves >= -2) & (last_moves < tables.last_move_categories.shape[0])
        bad |= j.any(~valid, axis=-1)
        at = j.clip(last_moves, 0, tables.last_move_categories.shape[0] - 1)
        value = j.where(last_moves < 0, tables.switch_categories[j.clip(-last_moves - 1, 0, 1)], tables.last_move_categories[at])
        categorical = categorical.at[..., s.CATEGORY_LAST_USED_MOVE].set(j.where(present & active & valid, value, 0))
    if borrowed_ability is not None:
        valid = (borrowed_ability >= 0) & (borrowed_ability < tables.traced_abilities.shape[0])
        bad |= j.any(~valid, axis=-1)
        value = tables.traced_abilities[j.clip(borrowed_ability, 0, tables.traced_abilities.shape[0] - 1)]
        categorical = categorical.at[..., s.CATEGORY_TRACED_ABILITY].set(j.where(present & active & valid, value, 0))
    if volatiles is not None:
        if volatiles.shape != present.shape + (len(VOLATILE_NAMES),):raise ValueError('named public volatile shape')
        at = j.sort(j.where(volatiles, j.arange(len(VOLATILE_NAMES)), len(VOLATILE_NAMES)), axis=-1)[..., :6]
        values = j.where(at < len(VOLATILE_NAMES), tables.volatile_categories[j.clip(at, 0, len(VOLATILE_NAMES) - 1)], 0)
        bad |= j.any(volatiles.sum(axis=-1) > 6, axis=-1)
        categorical = categorical.at[..., s.CATEGORY_VOLATILE_OFFSET:s.CATEGORY_VOLATILE_OFFSET + 6].set(j.where((present & active)[..., None], values, 0))
    return numeric, categorical, bad


def project_field(numeric, facts):
    facts = {} if facts is None else facts
    bad = j.zeros((numeric.shape[0],), bool)
    for key, (name, divisor) in FIELD_FIELDS.items():
        if key not in facts:continue
        value = facts[key].astype(j.float32)
        bad |= ~j.isfinite(value) | (value < 0)
        numeric = numeric.at[..., _column(name)].set(j.clip(j.nan_to_num(value) / divisor, 0., 1.))
    return numeric, bad


def project_stats(facts, tables, batch):
    numeric = j.zeros((batch, 132), j.float32); categorical = j.zeros((batch, 41), j.int32)
    bad = j.zeros(batch, bool)
    if facts is None:return numeric, categorical, j.zeros(batch, bool), bad
    for value in facts.values():bad |= ~j.isfinite(value) | (value < 0)
    for key, (name, divisor) in STATS_FIELDS.items():
        if key in facts:numeric = numeric.at[:, _column(name)].set(j.clip(j.nan_to_num(facts[key]) / divisor, 0., 1.))
    for i in range(6):
        key = 'weather_reveal_' + str(i)
        if key in facts:numeric = numeric.at[:, s.v4_numeric_index(s.NUMERIC_STAT_WEATHER_REVEAL_OFFSET + i)].set(j.clip(j.nan_to_num(facts[key]), 0., 1.))
    numeric = numeric.at[:, _column('PRESENT')].set(1)
    categorical = categorical.at[:, s.CATEGORY_ROLE].set(tables.stats_role)
    return numeric, categorical, j.ones(batch, bool), bad
