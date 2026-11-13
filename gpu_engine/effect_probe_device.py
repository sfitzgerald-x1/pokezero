"""Device mode of the effect probe (gpu_engine/effect_probe.py --device): the device public path against B2.

The state probe drives transition.step_with_chance only, so it never sees the order in which transition.py calls the
device public hooks (engine_device_public_hooks, engine_device_public_port). Device mode runs every case a second time
through transition.initialize_with_public / step_with_public (a fresh public tracker, belief_tables None, the same
actions and chance tape), asserts that the public run's engine state is bitwise the state run's, and after every request
step compares the device's public carry with B2's fold of that step's Showdown protocol lines, cumulatively from battle
start.

B2 is the host public-belief fold (v4_encoder.fold_chunk: public_wire.project_line, public_inputs.fold_chunk and its
consumers public_mon_state, public_items, public_moves, public_abilities, public_active, public_sleep, public_toxic,
public_tendency, public_field, public_damage, public_sleep_clause), compiled by v4_encoder.compile_encoder with the
champion's feature masks over a literal set source built from the run's authored sets (every species' distinct authored
sets are its variants). Each case has two owning folds, as the production adapter (pokezero.recorded_encoder_runner
JaxRecordedAdapter.reset/feed) runs them: owner p1 on p1's own channel and owner p2 on p2's, each channel as
sim/battle-stream.ts getPlayerStreams sends it (effect_probe.cjs playerChannels). Side s's public facts are read from the
fold its opponent owns (owner 1-s): that is the public view of side s, and it is what the device writes for side s (the
device's move hook charges Pressure by the observer's own active, which only the owner 1-s fold sees on its private
start line; its install never shares that line).

Format adaptation (B2 admits the Gen 3 Random Battle wire only; the probe battles run [Gen 3] Custom Game for authored
teams): the .cjs turns off Custom Game's debug exact-HP display so shared HP lines carry HP Percentage Mod's percent;
wire_lines below drops |debug| lines (debug-format only, never on a randbats wire), the |raw| custom-rule infobox
(the format's @@@sleepclausemod suffix; Gen 3 Random Battle has Sleep Clause Mod built in, announced by its |rule|
line), Splash's |-nothing| message (the probes' filler move, in no Gen 3 random set: B2's wire has no consumer for
it, and the line carries no public fact) and the start-of-battle EV-limit |bigerror| warning (authored EVs; random
sets keep the limit), and replaces the |tier| header.

The device runs with the production belief tables (engine_device_observation.compile_tables over the B2 catalog, as
gpu_engine/device_observation_collector.py builds them), so its Pressure PP charge, its hard belief masks and its shown
types follow the production path.

FIELDS is the compared field table (name, scope, device accessor, B2 accessor); LEFT_OUT lists the device public fields
deliberately not compared, with the reason. ALLOWLIST registers explained mismatches, keyed by (scenario, case, step,
field), each with a one-line reason; there is no per-field skip.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

STATUS = ('', 'par', 'brn', 'psn', 'tox', 'slp', 'frz')                    # engine status codes (state.py)
B2_STATUS = ('', 'slp', 'par', 'brn', 'frz', 'psn', 'tox')                 # public_mon_state NONE..TOXIC
WEATHER = {0: '', 1: 'raindance', 2: 'sunnyday', 3: 'sandstorm', 4: 'hail'}  # engine weather codes (phase_d_live_worlds)
TIER_LINE = '|tier|[Gen 3] Random Battle'
EV_LIMIT_WARNING = "|bigerror|Warning: One player isn't adhering to a 510 EV limit, and the other player is."

SOURCE_HASH_PREFIX = 'effect-probe-'

# (field, scope, device accessor, B2 accessor). Scope: mon = every present public slot of the side; active = the side's
# active slot; side = per side; field = per battle. Device: engine_device_trackers.Trackers (t), t.mon_features (mf),
# t.field_features (ff), t.tendency (td), t.public_port (port). B2: v4_encoder.State of the opposing owner's fold:
# mons (public_mon_state.MonState), ctx = context (public_context.ContextState). Slots are public encounter slots on both.
FIELDS = (
    ('present', 'mon', 't.present', 'mons.encounter > 0'),
    ('species', 'mon', 'vocab species[t.species]', 'catalog.species.names[mons.species]'),
    ('level', 'mon', 't.level', 'mons.level'),
    ('fainted', 'mon', 't.fainted', 'mons.fainted'),
    ('hp_percent', 'mon', 't.hp_percent', 'ceil(100*mons.hp_numerator/hp_denominator), 99 when below full'),
    ('status', 'mon', 'STATUS[t.status]', 'B2_STATUS[ctx.sleep.status] (phase_d_live_worlds.from_b2)'),
    ('shown_item', 'mon', 'vocab items[t.shown_item]', 'catalog.items.names[ctx.items.revealed]'),
    ('item_removed', 'mon', 't.item_removed', 'ctx.items.removed'),
    ('item_mutated', 'mon', "mf['item_mutated']", 'ctx.items.mutated'),
    ('item_original', 'mon', "vocab items[mf['item_original']]", 'catalog.items.names[ctx.items.original]'),
    ('moves', 'mon', '(vocab moves[t.observed_moves], t.pp_used) in reveal order',
     '(catalog.moves.names[ctx.moves.move_ids], ctx.moves.uses) in reveal order'),
    ('original_ability', 'mon', 'vocab abilities[t.original_ability]', 'catalog.abilities.names[ctx.abilities.original]'),
    ('borrowed_ability', 'mon', 'vocab abilities[t.borrowed_ability]', 'catalog.abilities.names[ctx.abilities.running]'),
    ('shown_ability', 'mon', 'vocab abilities[t.shown_ability]', 'ctx.abilities.running, else ctx.abilities.original'),
    ('types', 'mon', 'vocab types[t.shown_types]', 'TYPE labels of ctx.types.categories when overridden, else '
     'catalog.species.device.type_categories[mons.species]'),
    ('belief_hard', 'mon', 't.belief_mask', 'beliefs.hard[observer 1-side]'),
    ('visible_hard', 'mon', 't.visible_belief.visible_hard', 'beliefs.hard[observer 1-side]'),
    ('belief_pin', 'mon', 't.visible_belief.pin / pin_active', 'beliefs.pin / pin_active[observer 1-side]'),
    ('visible_pin', 'mon', 't.visible_belief.visible_pin / visible_pin_active', 'beliefs.pin / pin_active[observer 1-side]'),
    ('belief_conflicts', 'mon', 't.visible_belief.conflicts', 'beliefs.conflicts[observer 1-side]'),
    # Active-only, as the encoder reads it (engine_device_features.MON_FIELDS turns_active: active-only): B2's age is the
    # current stint's clock (public_sleep: 0 at arrival, +1 per TURN), stale on a benched mon; the device zeroes it at
    # withdraw (engine_device_public_hooks.withdraw).
    ('turns_active', 'active', "mf['turns_active']", 'ctx.sleep.age[active]'),
    ('sleep_turns', 'mon', "mf['sleep_turns']", 'ctx.sleep.count'),
    ('rest_sleep', 'mon', "mf['rest_sleep']", 'ctx.sleep.rest'),
    ('sleep_skipped_turns', 'mon', "mf['sleep_skipped_turns']", 'ctx.sleep.skipped'),
    ('sleep_cant_pending', 'mon', "mf['sleep_cant_pending']", 'ctx.sleep.pending'),
    ('switched_before_attack', 'mon', "mf['switched_before_attack']", 'ctx.tendency.mon[..., 0] (species-joined)'),
    ('stayed_and_attacked', 'mon', "mf['stayed_and_attacked']", 'ctx.tendency.mon[..., 1] (species-joined)'),
    ('turns_active_total', 'mon', "mf['turns_active_total']", 'ctx.tendency.mon[..., 2] (species-joined)'),
    ('active_slot', 'side', 't.active_slot', 'mons.active'),
    ('boosts', 'active', 't.boosts[active]', 'mons.boosts'),
    ('volatiles', 'active', 'NAMES of t.public_volatiles[active]',
     "NAMES of ctx.active.volatiles, plus 'mustrecharge' from ctx.active.recharge"),
    ('sub_present', 'active', "mf['sub_present']", "'substitute' in ctx.active.volatiles"),
    ('stall_counter', 'active', "mf['stall_counter']", 'ctx.active.stall'),
    ('meanlook_trap', 'active', "mf['meanlook_trap']", 'ctx.active.meanlook'),
    ('item_swapped', 'active', "mf['item_swapped']", 'ctx.active.item_swapped'),
    ('choice_public', 'active', "mf['choice_public']", 'ctx.active.choice_public'),
    ('choice_locked', 'active', "mf['choice_locked']", 'ctx.active.choice_locked'),
    ('confusion_turns', 'active', "mf['confusion_turns']", 'ctx.active.elapsed[0]'),
    ('encore_turns', 'active', "mf['encore_turns']", 'ctx.active.elapsed[1]'),
    ('wrap_trap_turns', 'active', "mf['wrap_trap_turns']", 'ctx.active.elapsed[2]'),
    ('toxic_stage', 'active', "mf['toxic_stage']", 'ctx.toxic.stage'),
    ('last_move', 'active', 't.last_moves[active] (move, -1 switch, -2 Baton Pass arrival, 0 none)',
     'ctx.moves.last_kind / last_id (LAST_MOVE, LAST_SWITCH: Baton Pass when mons.arrived_by_baton, LAST_UNKNOWN)'),
    # Previous-round damage sums (settled at TURN): the shown halves from the opposing owner's fold, the exact halves
    # from the side's own owner's fold (B2's owner sees its own HP exactly), as phase_d_live_worlds.from_b2 maps them.
    ('last_damage_taken', 'active', "mf['last_damage_taken'] (shown)", 'sum(ctx.damage.last_taken[side]) in the foe-owned fold'),
    ('last_damage_dealt', 'active', "mf['last_damage_dealt'] (shown)", 'sum(ctx.damage.last_dealt[side]) in the own fold'),
    ('damage_exact_taken', 'active', 't.damage_exact_taken', 'sum(ctx.damage.last_taken[side]) in the own fold'),
    ('damage_exact_dealt', 'active', 't.damage_exact_dealt', 'sum(ctx.damage.last_dealt[side]) in the foe-owned fold'),
    ('hazard_shown', 'side', 'sum(port.hazards[side, 0])', 'sum(ctx.damage.hazard_suffered[side]) in the foe-owned fold'),
    ('hazard_exact', 'side', 'sum(port.hazards[side, 1])', 'sum(ctx.damage.hazard_suffered[side]) in the own fold'),
    ('moved', 'side', 'td.moved', 'ctx.tendency.moved'),
    ('switches', 'side', 'td.switches', 'ctx.tendency.switches'),
    ('opportunities', 'side', 'td.opportunities', 'ctx.tendency.opportunities'),
    ('opportunity_turn', 'side', 'td.opportunity_turn', 'ctx.tendency.opportunity_turn'),
    ('matchup', 'side', 'td.matchup[side, a, b] over present slots', 'ctx.tendency.matchup (species-joined)'),
    ('blocked', 'side', 'port.blocked', 'ctx.tendency.blocked + the open window if blocked (B2\'s own readout, '
     'public_tendency.py:341-342: finalized windows, the open one added provisionally)'),
    ('pursuit', 'side', 'port.pursuit', 'ctx.tendency.pursuits'),
    ('weather_reveals', 'side', 'port.weather', 'ctx.tendency.weather'),
    ('sleep_clause_holder', 'side', 'port.holder', 'ctx.sleep_clause.holder'),
    ('sleep_clause_victims', 'side', 'port.victims', 'ctx.sleep_clause.victims'),
    ('truant_known', 'side', 'port.known', 'ctx.active.truant_known'),
    ('truant_loaf', 'side', 'port.loaf', 'ctx.active.truant_loaf'),
    ('traced_truant', 'side', 'port.traced', 'ctx.active.traced_truant'),
    ('truant_skip_flip', 'side', 'port.skip_flip', 'ctx.active.truant_skip_flip'),
    ('items_removed_credit', 'side', "ff['items_removed_credit']", 'ctx.damage.items_removed'),
    ('wish_set_turn', 'side', "ff['wish_set_turn']", 'ctx.field.wish_set_turn'),
    ('turn', 'field', 't.turn', 'ctx.field.turn'),
    ('weather', 'field', 'WEATHER[t.weather]', 'catalog.field.weather_names[ctx.field.weather]'),
    ('spikes', 'field', 't.spikes', 'ctx.field.spikes'),
    ('weather_upkeeps', 'field', "ff['weather_upkeeps']", 'ctx.field.weather_upkeeps'),
    ('weather_permanent', 'field', "ff['weather_permanent']", 'ctx.field.weather_ability'),
    ('post_upkeep', 'field', 'port.post_upkeep', 'ctx.active.post_upkeep'),
)

# Device public fields deliberately not compared, with the reason.
LEFT_OUT = {
    'gender': 'the probe builds engine lanes with state_from_pool, which has no gender leaf (0); Showdown draws gender '
              'at start as an audited inert roll, so the device gender is 0 by construction, not a public-path fact',
    'visible_belief.excluded_abilities, excluded_items': 'packed engine-ID exclusion ledgers (the device\'s '
              'non-proc and ability-exclusion names); B2 applies the same exclusions straight into beliefs.hard (compared '
              'as belief_hard), with no name ledger (phase_d_live_worlds.from_b2 writes them as zeros)',
    'damage_history': 'the device\'s current-round accumulators and HP baselines; B2\'s current_dealt/current_taken '
              'are compared through their settled previous-round sums (last_damage_*, damage_exact_*) at each TURN',
    'negative_items': 'the device\'s public negative-item (non-proc) updater state; B2\'s public_item_nonproc keeps a '
              'different internal representation with no per-field counterpart',
    'moves_shown, switches (mon_features), tendencies': 'carried counters with no B2 ledger counterpart (no encoder '
              'reads them; tendencies is a never-written zero placeholder)',
    'sleep_usable_pending': 'device-internal staging for the skipped-sleep-turn classification; B2 public_sleep has no '
              'counterpart (its pending is sleep_cant_pending)',
    'own_order, public_order, physical_to_public, active_physical, next_slot': 'private identity cursors and '
              'request-order permutations (never encoded); B2 indexes by public encounter only',
    'error': 'device diagnostic input-contract bits',
    'wish_turns, wish_pending (field_features)': 'zero in the carry; readout derives them from wish_set_turn '
              '(compared)',
}

# Explained mismatches: (scenario move, case_id, step, field) -> one-line reason. step is the request step index or
# 'initial'. field is the full diff key (e.g. '1:0:hp_percent'). An entry that matches no diff is reported as stale.
# Every entry is also PINNED to the exact (B2, device) pair it explains (PINS, effect_probe_device_pins.json, recorded
# from a full run): any other value at that key, on either side, stays an open diff, so an entry never hides a further
# change of the same field. A 'b2_refusal' entry is pinned to the refusal's kind and error code instead.
ALLOWLIST = {}
# OPEN BUG (reported to the coordinator): a sleeping Sleep Talk user that then hits itself in confusion. Gen 3 prints
# |cant|X|slp even for a sleep-usable move (data/mods/gen3/conditions.ts slp onBeforeMove), then confusion's
# onBeforeMove self-hit stops the move. B2 counts the cant line as a decision opportunity (public_tendency.apply:
# decision = ... | (cant & ~recharge), gpu_engine/public_tendency.py:199-202); the device's before_move counts one only
# when ~result.can_act & ~result.confusion_self_hit (gpu_engine/engine_device_public_hooks.py before_move, `opportunity(c,
# side,use&~result.can_act&~result.confusion_self_hit)`), so the self-hit suppresses the cant's opportunity.
_SLEEP_TALK_SELF_HIT = 'OPEN BUG: cant(slp) then a confusion self-hit; B2 counts the cant as an opportunity, the device does not'
for _case in ('selfhit=99,paralysis=0', 'selfhit=99,paralysis=99'):
    for _step in range(3, 9):
        ALLOWLIST[('confusion_status', _case, _step, '1:opportunities')] = _SLEEP_TALK_SELF_HIT
    for _step in range(3, 6):
        ALLOWLIST[('confusion_status', _case, _step, '1:opportunity_turn')] = _SLEEP_TALK_SELF_HIT
# OPEN BUG (reported): Encore ending at the residual. Showdown prints -end|X|Encore from Encore's onResidual
# (data/moves.ts encore condition) and B2 drops the bag entry (public_active.project_line REMOVE); the device's encore
# bit is written only by public_move_state around a move (engine_device_public_hooks.public_move_state), the Encore
# residual (transition.py: work=repeating.residual(...), the encore block of the order-10 list) calls no device hook and
# derived_volatiles covers only perish, yawn and partiallytrapped, so the bit stays until the mon leaves the field.
_ENCORE_END = 'OPEN BUG: Encore ends at the residual (-end|X|Encore); the device keeps the encore bag bit (no residual hook)'
for _case, _first in (('encore_roll=0', 3), ('encore_roll=1', 4), ('encore_roll=2', 5), ('encore_roll=3', 6)):
    for _step in range(_first, 9):ALLOWLIST[('encore', _case, _step, '1:active:volatiles')] = _ENCORE_END
for _step in range(3, 7):ALLOWLIST[('trick_encore', 'base', _step, '1:active:volatiles')] = _ENCORE_END
for _step in (3, 4):ALLOWLIST[('hyperbeam_order_fast', 'base', _step, '0:active:volatiles')] = _ENCORE_END
# OPEN BUG (reported): a fainted active's public stages at its replacement request. B2's faint keeps them (public_mon_state.
# faint writes HP, status and fainted, not boosts; an arrival resets them, public_mon_state.py:111-112), and B2's encoder
# writes the active token's stages from them (v4_assembly.project_own_boosts); the device's observe copies the engine's
# stages, which faintMessages' clearVolatile zeroed (transition._clear_fainted_boosts), so it reads 0 there.
_FAINTED_STAGES = 'OPEN BUG: a fainted active keeps its public stages in B2 until its replacement; the device reads the cleared engine stages'
for _key in (('wrap_tie', 'tie=0', 0, '0:active:boosts'), ('wrap_tie', 'tie=1', 0, '0:active:boosts'),
             ('knockoff_band', 'roll=0', 8, '1:active:boosts'), ('knockoff_band', 'roll=15', 8, '1:active:boosts'),
             ('superpower_ko', 'base', 5, '0:active:boosts'), ('destinybond_m7_gengar_volbeat', 'base', 3, '0:active:boosts'),
             ('sleepclause_faint', 'base', 3, '1:active:boosts'), ('knockoff_midturn_lock', 'base', 1, '0:active:boosts'),
             ('wh_intimidate_explosion', 'base', 0, '0:active:boosts'),
             *(('destinybond_m5_gengar_entei', _case, 3, '0:active:boosts')
               for _case in ('crit=15,roll=0', 'crit=15,roll=15', 'crit=0,roll=0', 'crit=0,roll=15'))):
    ALLOWLIST[_key] = _FAINTED_STAGES
# OPEN BUG (reported): Flash Fire's activation. A Fire move into a Flash Fire holder prints -start|X|ability: Flash Fire,
# which B2 folds into the flashfire bag entry (public_active.NAMES 'flashfire'); no device hook writes that bit
# (engine_device_public_hooks has no flashfire writer), so the device bag stays without it while the holder is active.
_FLASH_FIRE = 'OPEN BUG: -start|X|ability: Flash Fire sets B2\'s flashfire volatile; no device hook writes it'
for _step in (3, 4):ALLOWLIST[('sacredfire_fail', 'base', _step, '1:active:volatiles')] = _FLASH_FIRE
# OPEN BUG (reported; the last-move half is possibly B2's quirk, for the lead to rule): Pursuit's switch intercept prints
# |move|p2a: X|Pursuit|p1a: Y|[from] Pursuit. B2 records a move line's last-used move only without [from]
# (public_moves.record_move: last_kind/last_id gated by ~has_from, gpu_engine/public_moves.py:210-211) and credits the
# line's -damage to the user's dealt sums (public_damage's move window); the device's move hook records Pursuit as the
# last move and its damage history credits the intercept no dealt damage.
_PURSUIT_FROM = 'OPEN BUG: the [from] Pursuit intercept: B2 keeps the prior last move and credits its damage dealt; the device does neither'
ALLOWLIST[('destinybond_pursuit', 'base', 1, '1:active:last_move')] = _PURSUIT_FROM
for _scenario, _steps in (('destinybond_pursuer', (1, 2)), ('wh_pursuit', (0,))):
    for _step in _steps:
        for _field in ('1:active:last_move', '1:active:last_damage_dealt', '1:active:damage_exact_dealt'):
            ALLOWLIST[(_scenario, 'base', _step, _field)] = _PURSUIT_FROM
# OPEN BUG (reported): a failing Counter called by Sleep Talk. Showdown prints |move|p1a: X|Counter|p2a: Y|[from] Sleep
# Talk and B2 reveals Counter (0 PP charged); the device's called-move reveal (transition.py:1008, device_public.move(...,
# acts & (move!=caller_move), pp_use=False, count_turn=False)) is gated by acts, which Counter's failure gate (transition.py:
# 990, acts &= ~(counter kind & (taken kind differs | the called move is not the chosen one))) has already cleared.
_SLEEP_TALK_COUNTER = 'OPEN BUG: Sleep Talk\'s failing called Counter is revealed by B2 ([from] Sleep Talk line), not by the device'
for _step in (1, 2):ALLOWLIST[('counter_sleeptalk', 'talk=0', _step, '0:0:moves')] = _SLEEP_TALK_COUNTER
# OPEN BUG (reported): the upkeep's Lum and pinch-berry non-proc rules. B2's sweep (gpu_engine/public_item_nonproc.py,
# its triggers) and the Python belief (src/pokezero/belief.py _sweep_end_of_turn_non_procs) rule out, for a living
# active with an unrevealed, unmutated item, Lum Berry when a status is kept to |upkeep, and Salac/Petaya/Liechi when the
# pre-slot HP is at or below 25% (Showdown agrees: data/items.ts lumberry, data/mods/gen3/items.ts pinch berries at
# residual). The device's engine_device_negative_items.upkeep runs only the Leftovers rule, so its hard masks keep those
# variants. It shows wherever the fixed catalog holds such a variant of the species: Snorlax's four and Jolteon's one
# Lum Berry sets, Ninjask's Salac Berry set.
_NONPROC_ITEMS = 'OPEN BUG: B2 rules out Lum Berry (status kept to the upkeep) and pinch-berry variants; the device upkeep runs only the Leftovers rule'
for _scenario, _cases, _field, _steps in (
        ('aromatherapy', ('base',), '0:2', range(1, 7)),
        ('confusion_selfhit', ('damage=0', 'damage=15'), '1:1', range(4, 13)),
        ('sleepclause_faint', ('base',), '1:0', range(2, 7)), ('subfaint', ('base',), '0:0', range(0, 7)),
        ('toxic', ('accuracy=0',), '1:0', (0,)), ('yawn', ('sleep=0', 'sleep=1', 'sleep=2', 'sleep=3'), '1:0', range(1, 7)),
        ('yawn_again', ('base',), '1:0', range(1, 4)), ('yawn_sleepclause', ('base',), '1:0', range(0, 5)),
        ('yawn_statused', ('base',), '1:0', (1,))):
    for _case in _cases:
        for _step in _steps:
            for _mask in ('belief_hard', 'visible_hard'):ALLOWLIST[(_scenario, _case, _step, f'{_field}:{_mask}')] = _NONPROC_ITEMS
# B2 limit (explained, not a device fact): B2's fold refuses (PAYLOAD) a side's second encounter of one species, at the
# repeat's arrival line (|switch|p2a: Gengar2|Gengar, L74, M|...: RAW_BEGIN, SWITCH_PRELUDE, MUDSHOT, SWITCH, NONPROC,
# TENDENCY); Gen 3 Random Battle's Species Clause never sends it, the probes' Custom Game does. The case is compared
# up to that step (B2_LIMIT).
_REPEATED_SPECIES = 'B2 limit: a side repeats a species (no Species Clause in Custom Game); B2 refuses the repeat\'s arrival'
for _scenario, _cases, _step in (
        ('destinybond_expire', ('base',), 2), ('destinybond_ko', ('base',), 2),
        ('destinybond_m1_banette_machamp', ('crit=15,roll=0', 'crit=15,roll=15', 'crit=0,roll=0', 'crit=0,roll=15'), 2),
        ('destinybond_m3_qwilfish_dusclops', ('roll=0', 'roll=15'), 4), ('destinybond_m4_gengar_vigoroth', ('base',), 2),
        ('destinybond_m6_gengar_electrode', ('base',), 3), ('destinybond_m7_gengar_volbeat', ('base',), 4),
        ('destinybond_m9_wobbuffet_dusclops', ('base',), 3), ('facade_status', ('base',), 1), ('hyperbeam_roar', ('base',), 1),
        ('perishsong_last_faster', ('base',), 1), ('perishsong_tie', ('tie=0', 'tie=1'), 1), ('wrap_tie', ('tie=0', 'tie=1'), 1),
        ('wrap_trapper_dragged_back', ('base',), 1)):
    for _case in _cases:ALLOWLIST[(_scenario, _case, _step, 'b2_refusal')] = _REPEATED_SPECIES


PINS_PATH = Path(__file__).with_name('effect_probe_device_pins.json')


def _canon(value):
    """The JSON form of a diff value (tuples as lists, numpy scalars as the run output writes them)."""
    return json.loads(json.dumps(value, default=str))


def load_pins(path=PINS_PATH):
    """{(scenario, case, step, field): [b2, device]} or, for 'b2_refusal', {'kind': ..., 'error': ...}."""
    if not Path(path).exists():return {}
    rows = json.loads(Path(path).read_text())['pins']
    return {(r['scenario'], r['case'], r['step'], r['field']): r['value'] for r in rows}


PINS = load_pins()


def refusal_matches(refusal, pin):
    """A B2 refusal explained by its pin: the same kind and error code, nothing broader."""
    return pin is not None and refusal.get('kind') == pin.get('kind') and refusal.get('error') == pin.get('error')


def wire_lines(lines):
    """One channel's lines as B2's wire admits them (see the module docstring)."""
    out = []
    for line in lines:
        if line.startswith('|debug|') or line.startswith('|raw|') or line in ('|-nothing', EV_LIMIT_WARNING):continue
        out.append(TIER_LINE if line.startswith('|tier|') else line)
    return out


def _norm(name):
    return re.sub(r'[^a-z0-9]', '', str(name or '').lower())


# ---------------------------------------------------------------------------
# B2 catalog over the run's authored sets.

def literal_source(set_lists, dex):
    """Gen3RandbatSource whose variants are the authored sets: each species' distinct (moves, ability, item)."""
    from pokezero.randbat import Gen3RandbatSource, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant, RandbatSourceMetadata
    rows = {}
    for sets in set_lists:
        for team in sets:
            for s in team:
                key = _norm(s['species'])
                entry = rows.setdefault(key, dict(name=s['species'], level=s['level'], variants={}))
                variant = (tuple(s['moves']), s['ability'] or '', s['item'] or '', int(s['level']))
                entry['variants'].setdefault(variant, None)
    payload = sorted((k, sorted(v['variants'])) for k, v in rows.items())
    digest = hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()[:16]
    universes, moves, species = {}, set(), {}
    for key, entry in sorted(rows.items()):
        variants = tuple(Gen3RandbatVariant(f'probe-{key}-{i}', 'effect-probe', entry['name'], 'effect-probe', level, m, a, it)
                         for i, (m, a, it, level) in enumerate(sorted(entry['variants'])))
        if len(variants) > 64:raise ValueError(f'{key}: more than 64 authored sets')
        universes[key] = Gen3RandbatSpeciesUniverse(entry['name'], entry['level'], variants)
        moves.update(m for v in variants for m in v.moves)
        info = dex.species_info(key)
        if info is not None:species[key] = dict(baseStats=dict(info.base_stats), types=list(info.types), gender='N')
    metadata = RandbatSourceMetadata('gen3randombattle', 3, None, None, None, SOURCE_HASH_PREFIX + digest)
    move_metadata = {m: {'type': dex.move_info(m).type, 'category': dex.move_info(m).category}
                     for m in sorted(moves) if dex.move_info(m) is not None}
    return Gen3RandbatSource(metadata, universes, move_metadata=move_metadata, species_metadata=species)


def compile_b2(set_lists, showdown_root):
    """The B2 catalog (v4_encoder.compile_encoder, champion masks, the Showdown dex) over the literal source of
    `set_lists`. The device's belief tables are compiled from the same catalog (engine_device_observation.compile_tables,
    as the production collector does), so both narrow over the same source variants."""
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import load_showdown_dex_cached, normalize_id, resolve_move_effect
    from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
    from gpu_engine import public_active, public_belief, v4_encoder
    dex = load_showdown_dex_cached(showdown_root)
    source = literal_source(set_lists, dex)
    # Category labels as tests/test_gpu_engine_v4_encoder.py builds them; B2's fold reads none of them (an absent label
    # only hashes into the OOV band), they fill the encoder catalog's static category tables.
    labels = {'field', 'stats', 'pokemon:opponent'}
    labels |= {'request_kind:' + n for n in ('none', 'wait', 'team_preview', 'force_switch', 'move', 'unknown')}
    labels |= {'weather:' + n for n in ('raindance', 'sunnyday', 'sandstorm', 'hail')}
    labels |= {'volatile:' + n for n in public_active.NAMES if n}
    labels |= {'type:' + n for n in ('Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',
                                     'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark', '???')}
    labels |= {'status:' + n for n in ('none', 'slp', 'par', 'brn', 'frz', 'psn', 'tox', 'fnt')}
    for name, info in dex.moves.items():
        labels |= {'move:' + name, 'move_category:' + info.gen3_category, 'move_priority:' + str(info.priority)}
        effect, _, _ = resolve_move_effect(info, ())
        if effect:labels.add('move_effect:' + effect)
    labels |= {'species:' + normalize_id(n) for n in dex.species}
    for universe in source.universes.values():
        for v in universe.variants:
            labels |= {'belief:possible_move:' + normalize_id(m) for m in v.moves if m}
            if v.ability:labels |= {'ability:' + normalize_id(v.ability), 'belief:possible_ability:' + normalize_id(v.ability)}
            if v.item:labels |= {'item:' + normalize_id(v.item), 'belief:possible_item:' + normalize_id(v.item)}
    catalog = v4_encoder.compile_encoder(public_belief.compile_source(source), dex, CategoryVocabulary(tuple(sorted(labels))),
        checkpoint_sha256=CHAMPION_CONTRACT['checkpoint_sha256'], feature_masks=dict(CHAMPION_CONTRACT['feature_masks']))
    return catalog


class B2Batch:
    """Owning B2 folds, one stream per (case, owner), batched; one compiled fold for the whole run.

    Per stream this is JaxRecordedAdapter.reset/feed: public_wire.project_line over the stream's new lines, then
    public_inputs.pack_chunks and v4_encoder.fold_chunk chunk by chunk. A wire refusal or a nonzero fold error stops
    that stream (its state is then not compared)."""

    def __init__(self, catalog, streams):
        import jax
        from gpu_engine import v4_encoder
        self.jax, self.v4, self.catalog, self.streams = jax, v4_encoder, catalog, streams
        self.fold = jax.jit(lambda state, chunk: v4_encoder.fold_chunk(state, chunk, catalog))
        self.kwargs = v4_encoder.wire_kwargs(catalog)
        self.ids = np.arange(1, streams + 1, dtype=np.int32)

    def reset(self, owners):
        from gpu_engine import public_wire
        owners = np.asarray(owners, np.int32)
        if owners.shape != (self.streams,):raise ValueError('one owner per stream')
        self.state = self.v4.initialize(self.catalog, stream_ids=self.jax.numpy.asarray(self.ids), owner_sides=owners)
        self.hosts = [public_wire.WireState() for _ in range(self.streams)]
        self.refused = [None] * self.streams;self.bindings = None;self.lines = [0] * self.streams

    def feed(self, deltas):
        """deltas[k]: stream k's new raw channel lines (None: nothing). Returns the streams refused by this call."""
        from gpu_engine import public_inputs, public_wire
        events, fresh = [], []
        for k, lines in enumerate(deltas):
            row = []
            if lines and self.refused[k] is None:
                host = self.hosts[k]
                try:
                    for index, line in enumerate(wire_lines(lines)):
                        host, projection = public_wire.project_line(host, line, **self.kwargs)
                        row.extend(projection.events)
                        if self.bindings is None:
                            self.bindings = {name: np.asarray(getattr(projection, name), np.uint32) for name in (
                                'species_binding', 'ability_binding', 'move_binding', 'item_binding', 'context_binding',
                                'field_binding', 'active_binding', 'damage_binding', 'type_binding', 'sleep_binding',
                                'toxic_binding', 'tendency_binding')}
                except ValueError as error:
                    self.refused[k] = dict(kind='WIRE_REFUSED', line=line, offset=self.lines[k] + index, error=str(error)[:300])
                    fresh.append(k);row = []
                else:
                    self.hosts[k] = host;self.lines[k] += len(lines)
            events.append(row)
        if not any(events):return fresh
        for chunk in public_inputs.pack_chunks(self.ids, events, source_binding=self.catalog.source.device.binding,
                                               **self.bindings):
            self.state, error = self.fold(self.state, chunk)
            for k in np.flatnonzero(np.asarray(error) != 0):
                if self.refused[k] is None:
                    self.refused[k] = dict(kind='FOLD_ERROR', error=int(np.asarray(error)[k]));fresh.append(int(k))
        return fresh

    def host(self):
        return self.jax.device_get(self.state)


# ---------------------------------------------------------------------------
# Views (pure: numpy carries) and the comparison.

def _name(names, index):
    index = int(index)
    return '' if index <= 0 else f'?{index}' if index >= len(names) else _norm(names[index])   # a garbage id stays visible


def _words(x):
    return [int(w) for w in np.asarray(x).reshape(-1)]


def _belief(hard, pin, pin_active, conflicts):
    return dict(belief_hard=_words(hard), visible_hard=_words(hard), belief_pin=[_words(pin), bool(pin_active)],
                visible_pin=[_words(pin), bool(pin_active)], belief_conflicts=_words(conflicts))


def device_view(t, lane, vocab):
    """The device public carry of one lane, as {side: dict(mons, active, side), 'field': {...}} (names normalized)."""
    mf, ff, td, port, vb = t.mon_features, t.field_features, t.tendency, t.public_port, t.visible_belief
    from gpu_engine.public_active import NAMES
    view = {}
    for s in range(2):
        mons = []
        for k in range(6):
            if not bool(t.present[lane, s, k]):mons.append(None);continue
            moves = [(_name(vocab['moves'], m), int(n)) for m, n in zip(t.observed_moves[lane, s, k], t.pp_used[lane, s, k]) if int(m)]
            borrowed, original = _name(vocab['abilities'], t.borrowed_ability[lane, s, k]), _name(vocab['abilities'], t.original_ability[lane, s, k])
            mons.append(dict(present=True, species=_name(vocab['species'], t.species[lane, s, k]), level=int(t.level[lane, s, k]),
                fainted=bool(t.fainted[lane, s, k]), hp_percent=int(t.hp_percent[lane, s, k]), status=STATUS[int(t.status[lane, s, k])],
                shown_item=_name(vocab['items'], t.shown_item[lane, s, k]), item_removed=bool(t.item_removed[lane, s, k]),
                item_mutated=bool(mf['item_mutated'][lane, s, k]), item_original=_name(vocab['items'], mf['item_original'][lane, s, k]),
                moves=moves, original_ability=original, borrowed_ability=borrowed,
                shown_ability=_name(vocab['abilities'], t.shown_ability[lane, s, k]),
                types=[_name(vocab['types'], x) for x in t.shown_types[lane, s, k] if int(x)],
                belief_hard=_words(t.belief_mask[lane, s, k]),
                **({} if vb is None else dict(visible_hard=_words(vb.visible_hard[lane, s, k]),
                    belief_pin=[_words(vb.pin[lane, s, k]), bool(vb.pin_active[lane, s, k])],
                    visible_pin=[_words(vb.visible_pin[lane, s, k]), bool(vb.visible_pin_active[lane, s, k])],
                    belief_conflicts=_words(vb.conflicts[lane, s, k]))),
                **{key: int(mf[key][lane, s, k]) for key in ('sleep_turns', 'rest_sleep', 'sleep_skipped_turns',
                    'sleep_cant_pending', 'switched_before_attack', 'stayed_and_attacked', 'turns_active_total')}))
        slot = int(t.active_slot[lane, s]);active = None
        if 0 <= slot < 6:
            bits = t.public_volatiles[lane, s, slot]
            last = int(t.last_moves[lane, s, slot])
            active = dict(boosts=[int(x) for x in t.boosts[lane, s, slot]],
                volatiles=sorted(NAMES[v] for v in np.flatnonzero(bits) if NAMES[v]),
                last_move='<switch>' if last == -1 else '<baton>' if last == -2 else _name(vocab['moves'], last),
                last_damage_taken=float(mf['last_damage_taken'][lane, s, slot]), last_damage_dealt=float(mf['last_damage_dealt'][lane, s, slot]),
                damage_exact_taken=float(t.damage_exact_taken[lane, s, slot]), damage_exact_dealt=float(t.damage_exact_dealt[lane, s, slot]),
                **{key: int(mf[key][lane, s, slot]) for key in ('turns_active', 'sub_present', 'stall_counter', 'meanlook_trap', 'item_swapped',
                    'choice_public', 'choice_locked', 'confusion_turns', 'encore_turns', 'wrap_trap_turns', 'toxic_stage')})
        present = [k for k in range(6) if mons[k] is not None]
        foe = [k for k in range(6) if bool(t.present[lane, 1 - s, k])]
        side = dict(active_slot=slot, moved=bool(td.moved[lane, s]), switches=int(td.switches[lane, s]),
            opportunities=int(td.opportunities[lane, s]), opportunity_turn=int(td.opportunity_turn[lane, s]),
            matchup={(a, b): [int(x) for x in td.matchup[lane, s, a, b]] for a in present for b in foe},
            blocked=int(port.blocked[lane, s]), pursuit=int(port.pursuit[lane, s]),
            weather_reveals=[[bool(x) for x in row] for row in port.weather[lane, s]],
            sleep_clause_holder=int(port.holder[lane, s]), sleep_clause_victims=[bool(x) for x in port.victims[lane, s]],
            truant_known=bool(port.known[lane, s]), truant_loaf=bool(port.loaf[lane, s]), traced_truant=bool(port.traced[lane, s]),
            truant_skip_flip=bool(port.skip_flip[lane, s]), items_removed_credit=int(ff['items_removed_credit'][lane, s]),
            wish_set_turn=int(ff['wish_set_turn'][lane, s]),
            hazard_shown=float(port.hazards[lane, s, 0].sum()), hazard_exact=float(port.hazards[lane, s, 1].sum()))
        view[s] = dict(mons=mons, active=active, side=side)
    view['field'] = dict(turn=int(t.turn[lane]), weather=WEATHER.get(int(t.weather[lane]), f'?{int(t.weather[lane])}'),
        spikes=[int(x) for x in t.spikes[lane]], weather_upkeeps=int(ff['weather_upkeeps'][lane]),
        weather_permanent=bool(ff['weather_permanent'][lane]), post_upkeep=bool(port.post_upkeep[lane]))
    return view


def _percent(numerator, denominator):
    if denominator <= 0:return 0
    shown = -(-100 * numerator // denominator)
    return 99 if shown == 100 and numerator < denominator else shown


def b2_side(b2, q, own, s, catalog):
    """Side s's public facts in B2 stream q (the fold owned by side 1-s); the exact damage halves from stream own
    (the fold owned by side s)."""
    from gpu_engine.public_active import NAMES
    mons, ctx, beliefs = b2.mons, b2.context, b2.beliefs
    td = ctx.tendency
    tokens = catalog.vocabulary.tokens
    static_types = np.asarray(catalog.species.device.type_categories)
    def type_names(categories):
        out = []
        for c in (int(x) for x in categories):
            if not c:continue
            label = tokens[c - 1] if 0 < c <= len(tokens) else f'?{c}'
            out.append(_norm(label.split(':', 1)[1]) if label.startswith('type:') else label)
        return out
    def join(side, k):
        found = np.flatnonzero(td.species[q, side] == mons.species[q, side, k])
        return int(found[0]) if len(found) == 1 else None
    out = []
    for k in range(6):
        if int(mons.encounter[q, s, k]) <= 0:out.append(None);continue
        ids, uses = ctx.moves.move_ids[q, s, k], ctx.moves.uses[q, s, k]
        running, original = _name(catalog.abilities.names, ctx.abilities.running[q, s, k]), _name(catalog.abilities.names, ctx.abilities.original[q, s, k])
        key = join(s, k)
        tend = [int(x) for x in td.mon[q, s, key]] if key is not None else [None] * 3
        out.append(dict(present=True, species=_name(catalog.species.names, mons.species[q, s, k]), level=int(mons.level[q, s, k]),
            fainted=bool(mons.fainted[q, s, k]), hp_percent=_percent(int(mons.hp_numerator[q, s, k]), int(mons.hp_denominator[q, s, k])),
            status=B2_STATUS[int(ctx.sleep.status[q, s, k])],
            shown_item=_name(catalog.items.names, ctx.items.revealed[q, s, k]), item_removed=bool(ctx.items.removed[q, s, k]),
            item_mutated=bool(ctx.items.mutated[q, s, k]), item_original=_name(catalog.items.names, ctx.items.original[q, s, k]),
            moves=[(_name(catalog.moves.names, m), int(n)) for m, n in zip(ids, uses) if int(m)],
            original_ability=original, borrowed_ability=running, shown_ability=running or original,
            types=type_names(ctx.types.categories[q, s] if bool(ctx.types.overridden[q, s]) and int(ctx.types.encounter[q, s]) == int(mons.encounter[q, s, k])
                             else static_types[int(mons.species[q, s, k])]),
            **_belief(beliefs.hard[q, 1 - s, k], beliefs.pin[q, 1 - s, k], beliefs.pin_active[q, 1 - s, k], beliefs.conflicts[q, 1 - s, k]),
            sleep_turns=int(ctx.sleep.count[q, s, k]), rest_sleep=int(ctx.sleep.rest[q, s, k]),
            sleep_skipped_turns=int(ctx.sleep.skipped[q, s, k]), sleep_cant_pending=int(ctx.sleep.pending[q, s, k]),
            switched_before_attack=tend[0], stayed_and_attacked=tend[1], turns_active_total=tend[2]))
    slot = int(mons.active[q, s]);active = None;a = ctx.active
    if 0 <= slot < 6:
        names = {NAMES[v] for v in np.flatnonzero(a.volatiles[q, s]) if NAMES[v]}
        if bool(a.recharge[q, s]):names.add('mustrecharge')
        kind, last = int(ctx.moves.last_kind[q, s]), int(ctx.moves.last_id[q, s])
        active = dict(boosts=[int(x) for x in mons.boosts[q, s]], volatiles=sorted(names), sub_present=int('substitute' in names),
            turns_active=int(ctx.sleep.age[q, s, slot]),
            stall_counter=int(a.stall[q, s]), meanlook_trap=int(a.meanlook[q, s]), item_swapped=int(a.item_swapped[q, s]),
            choice_public=int(a.choice_public[q, s]), choice_locked=int(a.choice_locked[q, s]),
            confusion_turns=int(a.elapsed[q, s, 0]), encore_turns=int(a.elapsed[q, s, 1]), wrap_trap_turns=int(a.elapsed[q, s, 2]),
            toxic_stage=int(ctx.toxic.stage[q, s]),
            last_move=(_name(catalog.moves.names, last) if kind == 2 else
                       ('<baton>' if bool(mons.arrived_by_baton[q, s]) else '<switch>') if kind == 1 else ''),
            last_damage_taken=float(ctx.damage.last_taken[q, s].sum()), last_damage_dealt=float(ctx.damage.last_dealt[own, s].sum()),
            damage_exact_taken=float(ctx.damage.last_taken[own, s].sum()), damage_exact_dealt=float(ctx.damage.last_dealt[q, s].sum()))
    present = [k for k in range(6) if out[k] is not None]
    foe = [k for k in range(6) if int(mons.encounter[q, 1 - s, k]) > 0]
    matchup = {}
    for x in present:
        for y in foe:
            ja, jb = join(s, x), join(1 - s, y)
            matchup[(x, y)] = None if ja is None or jb is None else [int(v) for v in td.matchup[q, s, ja, jb]]
    clause = ctx.sleep_clause
    side = dict(active_slot=slot, moved=bool(td.moved[q, s]), switches=int(td.switches[q, s]),
        opportunities=int(td.opportunities[q, s]), opportunity_turn=int(td.opportunity_turn[q, s]), matchup=matchup,
        blocked=min(64, int(td.blocked[q, s]) + int(bool(td.window_move[q]) and int(td.window_side[q]) == s and int(td.window_outcome[q]) == 2)),
        pursuit=int(td.pursuits[q, s]), weather_reveals=[[bool(x) for x in row] for row in td.weather[q, s]],
        sleep_clause_holder=int(clause.holder[q, s]), sleep_clause_victims=[bool(x) for x in clause.victims[q, s]],
        truant_known=bool(a.truant_known[q, s]), truant_loaf=bool(a.truant_loaf[q, s]), traced_truant=bool(a.traced_truant[q, s]),
        truant_skip_flip=bool(a.truant_skip_flip[q, s]), items_removed_credit=int(ctx.damage.items_removed[q, s]),
        wish_set_turn=int(ctx.field.wish_set_turn[q, s]),
        hazard_shown=float(ctx.damage.hazard_suffered[q, s].sum()), hazard_exact=float(ctx.damage.hazard_suffered[own, s].sum()))
    return dict(mons=out, active=active, side=side)


def b2_field(b2, q, catalog):
    f, a = b2.context.field, b2.context.active
    return dict(turn=int(f.turn[q]), weather=_name(catalog.field.weather_names, f.weather[q]),
        spikes=[int(x) for x in f.spikes[q]], weather_upkeeps=int(f.weather_upkeeps[q]),
        weather_permanent=bool(f.weather_ability[q]), post_upkeep=bool(a.post_upkeep[q]))


def b2_view(b2, lane, catalog):
    """B2's view of case `lane`: side s from stream 2*lane + (1-s) (owner 1-s), its exact damage halves from stream
    2*lane + s (owner s), the field from owner p1's stream."""
    return {0: b2_side(b2, 2 * lane + 1, 2 * lane, 0, catalog), 1: b2_side(b2, 2 * lane, 2 * lane + 1, 1, catalog),
            'field': b2_field(b2, 2 * lane, catalog)}


FLOAT_FIELDS = ('last_damage_taken', 'last_damage_dealt', 'damage_exact_taken', 'damage_exact_dealt', 'hazard_shown',
                'hazard_exact')
# OPEN BUG (numeric, reported to the coordinator): the device's shown-percent damage fractions divide a float32 by the
# literal 100 (engine_device_damage_history.observe `.../100`, engine_device_public_port.spikes), which XLA compiles to a
# multiply by float32(0.01), one ulp below the correctly rounded quotient for most percents (30 -> 0.29999998, B2
# 0.30000001); B2 (public_damage.apply) divides by its hp_denominator array (100) and rounds correctly, and a round's sum
# of several such fractions drifts by one ulp per line. A float field whose two values differ by less than
# FLOAT_TOLERANCE is recorded in the step's `numeric` list, not as a diff: any real difference is at least one HP point
# of the largest Gen 3 max HP (1/714, ~1.4e-3) or one shown percent (0.01), so the tolerance hides only rounding.
FLOAT_TOLERANCE = 1e-6
NUMERIC_NOTE = ('OPEN BUG: device shown-percent damage fraction divides by the literal 100 (XLA: multiply by '
                'float32(0.01)), float32 ulps from B2\'s division by hp_denominator')


def _within_tolerance(a, b):
    if not isinstance(a, float) or not isinstance(b, float):return False
    return abs(a - b) < FLOAT_TOLERANCE


def compare_views(device, b2, numeric=None):
    """Field-by-field differences, keyed side:slot:field (mon), side:active:field, side:field and field. A FLOAT_FIELDS
    pair closer than FLOAT_TOLERANCE goes to `numeric` (a list), see NUMERIC_NOTE."""
    diffs = []
    for s in range(2):
        for part in ('active', 'side'):
            if device[s][part] is not None and b2[s][part] is not None and set(device[s][part]) != set(b2[s][part]):
                raise ValueError(f'device and B2 {part} views differ in keys: {sorted(set(device[s][part]) ^ set(b2[s][part]))}')
        for k in range(6):
            if device[s]['mons'][k] is not None and b2[s]['mons'][k] is not None and set(device[s]['mons'][k]) != set(b2[s]['mons'][k]):
                raise ValueError(f'device and B2 mon views differ in keys: {sorted(set(device[s]["mons"][k]) ^ set(b2[s]["mons"][k]))}')
    if set(device['field']) != set(b2['field']):
        raise ValueError(f'device and B2 field views differ in keys: {sorted(set(device["field"]) ^ set(b2["field"]))}')
    def check(key, want, got):
        if want == got:return
        if numeric is not None and key.rsplit(':', 1)[-1] in FLOAT_FIELDS and _within_tolerance(want, got):
            numeric.append(dict(field=key, b2=want, device=got));return
        diffs.append(dict(field=key, b2=want, device=got))
    for s in range(2):
        d, b = device[s], b2[s]
        for k in range(6):
            dm, bm = d['mons'][k], b['mons'][k]
            if dm is None or bm is None:
                check(f'{s}:{k}:present', bm is not None, dm is not None);continue
            for name in bm:
                if name != 'present':check(f'{s}:{k}:{name}', bm[name], dm[name])
        if (d['active'] is None) != (b['active'] is None):check(f'{s}:active', b['active'] is not None, d['active'] is not None)
        elif d['active'] is not None:
            for name in b['active']:check(f'{s}:active:{name}', b['active'][name], d['active'][name])
        for name in b['side']:
            if name == 'matchup':
                for pair in sorted(set(b['side']['matchup']) | set(d['side']['matchup'])):
                    check(f'{s}:matchup:{pair[0]}:{pair[1]}', b['side']['matchup'].get(pair), d['side']['matchup'].get(pair))
            else:check(f'{s}:{name}', b['side'][name], d['side'][name])
    for name in b2['field']:check(f'field:{name}', b2['field'][name], device['field'][name])
    return diffs


def field_kind(key):
    """The FIELDS name of a diff key (the last component, or the matchup family)."""
    parts = key.split(':')
    return 'matchup' if len(parts) > 1 and parts[1] == 'matchup' else parts[-1]


def apply_allowlist(move, case_id, step, diffs, used, allowlist=None, pins=None):
    """Split diffs into (open, allowlisted); allowlisted entries carry their reason. `used` collects matched keys. A diff
    at an allowlisted key is explained only if its (b2, device) pair equals the key's pin; otherwise it stays open,
    carrying the reason and the pin it moved from."""
    allowlist = ALLOWLIST if allowlist is None else allowlist
    pins = PINS if pins is None else pins
    open_, allowed = [], []
    for d in diffs:
        key = (move, case_id, step, d['field'])
        if key in allowlist:
            used.add(key)
            if _canon([d['b2'], d['device']]) == pins.get(key):allowed.append(dict(d, reason=allowlist[key]))
            else:open_.append(dict(d, reason=allowlist[key], pin=pins.get(key)))
        else:open_.append(d)
    return open_, allowed


def trees_equal(a, b):
    """Leaf-by-leaf bitwise equality of two engine pytrees (PRNG key arrays by their key data)."""
    import jax
    (left, left_def), (right, right_def) = jax.tree.flatten(a), jax.tree.flatten(b)
    if left_def != right_def or len(left) != len(right):return False
    for x, y in zip(left, right):
        if jax.dtypes.issubdtype(getattr(x, 'dtype', np.int32), jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():return False   # -0.0 and NaN payloads too
    return True
