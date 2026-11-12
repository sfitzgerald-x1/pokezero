#!/usr/bin/env python3
"""Widened carry audit: how often do multi-turn effects the GPU engine does not represent touch the live DepthN search's roots and trees?

usage: depthn_carry_audit.py enumerate [--showdown-root ROOT] [--out DIR]
       depthn_carry_audit.py report --rows DIR_OR_FILE [...] --owning-wire DIR --out DIR [--turn2 DIR_OR_FILE ...]
           [--showdown-root ROOT] [--threshold 0.05] [--decision-share 0.02] [--examples 12]
       depthn_carry_audit.py capture --out DIR [--like HEADER_JSON] [--watch MOVE ...] [--dry-run] [<re-pricer flags> ...]

Why. The live search prices depth-2 trees on the GPU engine (gpu_engine/transition.py and the device hooks). Every effect below is
in the Gen 3 random battle move pool and was absent from that engine, so inside the tree it was mispriced: a recharge turn is free,
a Wrap does not trap or chip, a Perish Song never counts down, and so on. The fidelity patches model some of them in-tree from a
named integration merge (Effect.engine_since: Destiny Bond from A2, 286a5ffe; Hyper Beam's recharge and Trick from M1, 8b212312;
Wrap's partial trap, Perish Song's count and Yawn's drowsy from M2 + R1, f00dfeed);
the live root carry still drops or refuses them (the evidence rows), so they stay audited. The audit counts, per effect, the priced roots it touches
and applies the lead's rule: an effect whose (b) and (c) together cover <= 2% of priced roots is a named residual; otherwise it is
escalated.

Enumeration (`EFFECTS`, `REPRESENTED`, `NOT_IN_POOL`; every row cites file:line at this tree and at the pinned Showdown f76228a).
The pool is the set source the constructor reads: pokezero.randbat.load_gen3_randbat_source_cached(showdown_root) loads
data/random-battles/gen3/sets.json (gpu_engine/phase_d_protocol_constructor.py:129). Its 220 species hold 125 moves; the
pool_species of each row are the species with a set listing the move. With --showdown-root that file is re-read and every row's
pool membership is checked against it (pool_check; a mismatch is reported, never silently accepted).
- Audited (in the pool, not represented): recharge (Hyper Beam), partial_trap (Wrap), perish_count (Perish Song), drowsy (Yawn),
  destiny_bond (Destiny Bond), transform (Transform), trick_lock (Trick: its user always holds Choice Band, so the target ends up
  choice-locked). recharge, partial_trap and transform are also LOCK-class at gpu_engine/engine_request_source.py:380.
  engine_since: destiny_bond 286a5ffe (A2); recharge and trick_lock 8b212312 (M1); partial_trap, perish_count and drowsy f00dfeed (M2 + R1);
  transform none.
- Not in the pool (zero by construction): the locked-move family (Outrage, Thrash, Petal Dance), Uproar, Rollout, Ice Ball, Bide,
  Rage, the two-turn moves other than Solar Beam (Fly, Dig, Dive, Bounce, Skull Bash, Razor Wind, Sky Attack), the other partial
  traps, Future Sight / Doom Desire, Taunt / Disable / Torment.
- Represented (listed, not counted): Solar Beam's charge, Encore, the Choice Band lock, Mean Look / Spider Web, Leech Seed, Wish,
  the Protect/Endure counter, sleep/Rest and Toxic counters, weather, Focus Punch, Sleep Talk, Spikes, Substitute, Baton Pass,
  Truant.

Measures, per audited effect E (its moves M_E), over the replay rows (scripts/depthn_rr_reprice.py rows). A priced root is a row
whose replay.status is COMPLETE (root logits and legal mask and every world's grid were produced).
- (a) legal at turn 1: ours when a legal root lane (root.root_legal) plays a move of M_E (our move per lane: root.own_moves when the
  row has it, else the decision's own request line in the owning-wire capture, request.active[0].moves[i].id); the opponent's when
  some world's peer lane (worlds[w].peer.lanes: move:<id> on a legal lane) plays one.
- (b) turn-2 mass: some world's turn-2 game (a turn-1 cell the depth-2 frame expanded, t1_status 2) puts >= --threshold (5%) of
  either side's prior on M_E. The prior is the pricer's own: at each expanded cell, the masked softmax of the seat's turn-2 logits
  over its enumerated slots (depthn_backups._softmax over t1_own_actions / t1_opp_actions; learner_max averages exactly that
  opponent prior, and the learner's is its prior under the `average` rule). The lane -> move map is read from the turn-1 child state
  (the encoder's: lane i is the active's move slot i, lane 0 the charge move while one charges and Struggle when no move is usable,
  engine_device_observation.py:290-302). The replay rows hold none of this: `capture` records it (below), and the report joins it by
  (ordinal, decision_index). Where a priced root has no turn-2 record the proxy (b1) stands in and the decision is PROVISIONAL.
  (b1) turn-1 mass, the proxy: our root prior (softmax of root.root_logits over root.root_legal) or some world's peer prior
  (softmax of worlds[w].root_peer_logits over worlds[w].kept, the replies turn 1 enumerated: the mass learner_max averages at the
  root) puts >= threshold on M_E. Its gap: it reads the root's actives only. A turn-1 switch brings in a different active, whose
  turn-2 moves the proxy never sees (our bench Slaking switched in, then Hyper Beam at turn 2), and it cannot tell a move the child
  forbids (Choice lock, charge) from a free one.
  (b_max, recorded with b) the learner's turn-2 learner_max row (the own slot whose opponent-prior average of the leaf row is the
  strict max: the row learner_max backs up) plays M_E at some expanded cell.
- (c) search overrides into E at the root: the replay's searched action (replay.action_index) differs from live greedy
  (live.greedy_action_index; argmax of root_logits over root_legal when the row has no live join) and plays a move of M_E while
  greedy's action does not. c_live is the same with the live game's played action (live.played_action_index).
- Guard (the lead's recharge guard): how often our searched action is a recharge move while greedy's is not (= (c) of recharge),
  beside how often each one alone is.
- (d) carried at the root, informational: the decision's own prefix (owning-wire lines) shows E in effect on either side's active
  (carried_effects' marker rules: -mustrecharge, -activate ... move: Wrap, -start ... perishN, -start ... move: Yawn, -singlemove
  ... Destiny Bond, -transform, -item ... Choice Band [from] move: Trick; a switch, drag or faint of that side clears it, as do the
  end markers). The constructor refuses roots after a Transform (phase_d_live_worlds.py:278-279) or a public item swap
  (phase_d_protocol_world.py:105-107), and roots holding a partial trap, a drowsy or a perish count (the rows' live-root evidence),
  so their (d) among priced roots should be zero; the others (a recharge turn, a held Destiny Bond) are refused or single-legal as
  their rows say.
- Coverage: |(b) or (c)| over priced roots; <= --decision-share (2%) is NAMED_RESIDUAL, else ESCALATE. The `extended` line also
  counts (b1) under (b) (the opponent using E at turn 1, which neither (b) nor (c) sees), informational.

capture (one GPU per shard; the re-pricer's exact replay with the turn-2 record added). scripts/depthn_rr_reprice.py is imported and
not modified. While its main runs, three of its module globals are swapped (turn2_hooks; check_reprice_hooks verifies from main's
bytecode that it looks each up at call time): Capture -> Turn2Capture, describe -> a wrapper that returns the original's detail
unchanged and builds the decision's turn-2 record, decision_row -> a wrapper that returns the original's record unchanged and writes
that turn-2 record to turn2.jsonl. Turn2Capture.installed() is rr's own context manager (decide and prepare_roots wrapped) and, for the
same process_many call, also wraps constructor_worlds.price (DepthNLiveSearch._search calls `cw.price(...)` through the module
attribute) to keep the batch and the frame it returns, and DepthNPricer._program (price() calls `self._program(...)`; the class
attribute) so the compiled P1 program's output, the 81 turn-1 children per world row, also yields each child's active, its four
moves, its charge move and usable_moves (gpu_engine.state) through one small jitted read before price() pops them
(check_turn2_call_sites verifies both lookups from bytecode). Every wrapper calls the original with the same objects, returns its
result unchanged and is restored when the call returns or raises; nothing on the decision path changes.
- Per decision, turn2.jsonl holds: address (ordinal, decision_index, battle_id, owner, prefix_sha256), the replay's status and
  action, and per world (block order: block k's world j is world 2k+j, frame root row rows[k]) the expanded cell count, per side
  {move: [[cell, mass], ...]} for the watched moves (--watch; default every audited effect's moves) with mass > 0, own_max
  {move: [cells]}, the distinct turn-2 move labels seen per side (a sanity check on the lane map), and the cells it could not
  label or solve.
- --like HEADER_JSON takes the re-pricer argv from a previous run's header-<shard>.json (e.g. rr-fix-s160/shard-I/header-I.json):
  its --out and --tree-commit are dropped and --accept-tree is set to this checkout's HEAD (the guarded paths must be clean); flags
  after it are appended. The re-pricer writes its own rows/header/summary under <out>/replay; turn2.jsonl, turn2-header.json and
  turn2-summary.json (written on interruption too) go to <out>. --dry-run prints the derived re-pricer argv and exits.

report (CPU, json only): summary.json and summary.md in --out (a new directory): inputs, the enumeration table with the pool check,
per effect (a), (b), (b1), (b_max), (c), c_live, (d), coverage and decision with example roots, the guard counts, the turn-2 join
(rows joined, prefix mismatches, replay-action disagreements between the turn-2 replay and the rows: a different tree replayed),
own-move sources, and the owning-wire read errors.
"""
import argparse
import collections
from contextlib import contextmanager
from dataclasses import dataclass, field
import dis
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))

SCHEMA = 'pokezero.depthn-carry-audit.v1'
TURN2_SCHEMA = 'pokezero.depthn-carry-audit.turn2.v1'
SHOWDOWN_PIN = 'f76228a1354b5d0f307ca2d16101294ad3a2308b'
SETS_PATH = 'data/random-battles/gen3/sets.json'
THRESHOLD = 0.05
DECISION_SHARE = 0.02
LANES = 81


# Enumeration --------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Effect:
    id: str
    moves: tuple
    pool_species: tuple            # species with a gen3 set listing one of `moves` (sets.json at SHOWDOWN_PIN)
    represented: bool
    lock_class: str                # why engine_request_source.py:380 classes it LOCK, or '' when it does not
    mechanism: str
    evidence: tuple
    note: str = ''
    engine_since: str = ''         # the integration merge from which the engine models it in-tree ('' : not modelled)


EFFECTS = (
    Effect('recharge', ('hyperbeam',), ('slaking',), False,
        'request_source_rules.cjs:67 move.flags.recharge -> unsupported_history; engine_request_source.py:261-265 (any last_move) -> :380',
        'the user must recharge the next turn: no move, no switch',
        ('Showdown data/moves.ts:9108-9119: Hyper Beam has flags.recharge and self.volatileStatus mustrecharge',
         'Showdown data/conditions.ts:364-378: mustrecharge cancels the next move (onBeforeMovePriority 11) and is onLockMove recharge; '
         'sim/pokemon.ts:1085-1088: a locked move sets trapped, so the recharge turn has no switch',
         'in-tree since M1 (8b212312): gpu_engine/state.py:87 recharge_counter; transition.py:766-771 the recharge turn',
         'live roots: gpu_engine/constructor_worlds.py:1454 and 1609 an opponent that must recharge is refused '
         '(OPPONENT_RECHARGE_UNREPRESENTED, constructor_worlds.py:467); the owner\'s recharge turn is single-legal (the phase-D '
         'mapper refuses a live mustrecharge too: future_state.py:476, RECHARGE_UNREPRESENTED)'),
        'Only Slaking carries it, and Slaking\'s one ability is Truant: a residual clock in Gen 3 (Showdown data/mods/gen3/abilities.ts:'
        '197-212) the engine models (state.py:84 truant_loaf, transition.py:583-585 and 777-778). In-tree since M1 the recharge turn '
        'is modelled (trapped, no move); before it the engine\'s free turn after Hyper Beam was spent loafing anyway. A Transform '
        'user copying Slaking remains (Transform is itself unrepresented).', engine_since='8b212312'),
    Effect('partial_trap', ('wrap',), ('shuckle',), False,
        'request_source_rules.cjs:57-64 the partiallytrapped condition defines onTrapPokemon (missingCondition) -> engine_request_source.py:380',
        'the target is trapped and chipped each turn for several turns',
        ('Showdown data/moves.ts:21091 (gen4 mod data/mods/gen4/moves.ts:1515, inherited by gen3): Wrap sets volatile partiallytrapped',
         'Showdown data/conditions.ts:222-252: partiallytrapped lasts several turns, damages at each residual and traps (onTrapPokemon)',
         'gpu_engine/linked_traps.py:12-14: the represented move traps are Mean Look, Spider Web and Block only (Wrap is its own volatile)',
         'in-tree since M2 + R1 (f00dfeed): gpu_engine/state.py:91-92 partial_trap_counter (3 + draws[s][7] at Wrap) and '
         'partial_trap_source; transition.py:1844-1849 the Wrap set (the counter and partial_trap_source); transition.py:2409-2429 its residual tick (the trapper read at '
         'the holder\'s own handler), traced by R1 only at residual_order.POSITIONS wrap 4, 8, 9, 11 (gpu_engine/residual_order.py:95)',
         'probes: 12 wrap* scenarios (20 cases), EXACT against Showdown f76228a1 at M2 + R1 (3e0878e4); each gated position fails a '
         'named probe when dropped (#1930 (M2 + R1) PR body, R1 section, position -> probe table: 4 wrap and seven more, 8 and 9 '
         'wrap_tie, 11 wrap_trapper_faint and wrap_trapper_switch)',
         'live roots: a held partial trap is refused: the protocol world is built with approximate_partial_trap_turns=False '
         '(gpu_engine/phase_d_protocol_world.py:306-307), so pokezero.engine_world raises volatile_unsupported'),
        'Shuckle (Staller) only. In-tree since M2 the trap and its residual chip are modelled; a root already holding a partial trap '
        'is refused.', engine_since='f00dfeed'),
    Effect('perish_count', ('perishsong',), ('misdreavus',), False, '',
        'both actives faint at the end of the third following turn unless they switch',
        ('Showdown data/moves.ts:13232 (gen4 mod data/mods/gen4/moves.ts:993): Perish Song starts a 3-turn count on both actives',
         'in-tree since M2 + R1 (f00dfeed): gpu_engine/state.py:89 perish_counter (4 at Perish Song, -1 per residual, faint at 0); '
         'transition.py:1623-1631 the count set (Soundproof exempt); transition.py:2478-2491 its residual, a two-holder pass after the '
         'order-10 list, faster side first, an equal-Speed pair on a drawn effect bit, the count and the faint at 0 (not gated by R1)',
         'probes: 10 perishsong* scenarios (perishsong, _tie, _last_faster, _baton, _switch, _meanlook, _soundproof, _soundproof_user, '
         '_protect_sub, _reuse), EXACT against Showdown f76228a1 at M2 + R1 (3e0878e4)',
         'live roots: a held perish count passes the protocol world (pokezero.engine_world) and is refused by '
         'gpu_engine/v5_hypothetical_state.py:46 (any volatile but leechseed, substitute and truant: volatile:<seat>); '
         'gpu_engine/public_active.py:46, 217-233 keeps the public counts for the encoder'),
        'Misdreavus (Staller), the set that also carries Mean Look (a represented trap).', engine_since='f00dfeed'),
    Effect('drowsy', ('yawn',), ('swalot',), False, '',
        'the target falls asleep at the end of the next turn',
        ('Showdown data/moves.ts:21141 (gen4 mod data/mods/gen4/moves.ts:1527): Yawn puts the target to sleep a turn later',
         'in-tree since M2 + R1 (f00dfeed): gpu_engine/state.py:90 yawn_counter (2 at Yawn, -1 at residual subOrder 19, a sleep attempt '
         'at 0); transition.py:1636-1644 the Yawn set; transition.py:2430-2458 its countdown (2440-2444 Sleep Clause and the '
         'immunities) and the after-status step, '
         'traced by R1 only at residual_order.POSITIONS yawn 6, 12, 13 (gpu_engine/residual_order.py:95)',
         'probes: 11 yawn* scenarios, EXACT against Showdown f76228a1 at M2 + R1 (3e0878e4); each gated position fails a named probe '
         'when dropped (#1930 (M2 + R1) PR body, R1 section, position -> probe table: 6 yawn_faster, added because no earlier probe '
         'reached it; 12 yawn_tie; 13 yawn and six more)',
         'live roots: a held drowsy is refused: the protocol world is built with approximate_hidden_duration_volatiles=False '
         '(gpu_engine/phase_d_protocol_world.py:306-307), so pokezero.engine_world raises volatile_unsupported'),
        'Swalot (Bulky Attacker) only.', engine_since='f00dfeed'),
    Effect('destiny_bond', ('destinybond',), ('banette', 'gengar', 'qwilfish', 'wobbuffet'), False, '',
        'a KO of the user by a move takes its attacker down, until the user moves again (so into the next turn)',
        ('Showdown data/moves.ts:3479-3520: the destinybond volatile ends only at the user\'s next move (onBeforeMove)',
         'in-tree since A2 (286a5ffe): gpu_engine/state.py:21 DESTINY_BOND; effects.py:46; transition.py:758-762 its end at the '
         'holder\'s next move',
         'live roots: gpu_engine/phase_d_live_worlds.py:344-345 a bonded own active is refused (OWN_VOLATILE_OR_LATENT_CLOCK_UNAVAILABLE)'),
        'Gengar, Wobbuffet, Qwilfish, Banette.', engine_since='286a5ffe'),
    Effect('transform', ('transform',), ('ditto', 'mew'), False,
        'request_source_rules.cjs:65-66 transform is a missing origin -> engine_request_source.py:380',
        'the user copies the target\'s species, types, stats, ability, boosts and moves while it stays in',
        ('Showdown data/moves.ts:19836 (gen4 mod data/mods/gen4/moves.ts:1443): Transform',
         'gpu_engine/transition.py:81-113: no move table names transform (no effect in search); public_refresh_stream.py:17 is the '
         'encoder\'s public refresh only',
         'gpu_engine/phase_d_live_worlds.py:278-279: a root with a transformed Pokemon is refused (PUBLIC_TRANSFORM_NOT_MATERIALIZED)'),
        'Ditto and Mew. A transformed root is refused, so the effect reaches the search only as a turn-1 or turn-2 move.'),
    Effect('trick_lock', ('trick',), ('furret', 'kecleon'), False, '',
        'items swap: the user\'s Choice Band choice-locks the target from its next move, the user gets the target\'s item',
        ('Showdown data/random-battles/gen3/teams.ts:473: a set with Trick always holds Choice Band',
         'Showdown data/moves.ts:19875: Trick swaps the two items',
         'in-tree since M1 (8b212312): gpu_engine/items.py:27-28 the Trick table; transition.py:1584-1595 the swap; state.py:85 '
         'item_knocked (live roots carry it since #1935, 3f48093b)',
         'live roots: gpu_engine/phase_d_protocol_world.py:105-107 a root after a public item swap is refused (PUBLIC_ITEM_SWAP_NOT_SUPPORTED)'),
        'Furret and Kecleon. Not LOCK-class at the request source (choice_move and item are read from the state).', engine_since='8b212312'),
)

REPRESENTED = (
    Effect('charge', ('solarbeam',), ('exeggutor', 'sunflora', 'tangela', 'victreebel'), True,
        'engine_request_source.py:341 a mid-charge request (charge_move != 0) is LOCK', 'two-turn charge',
        ('gpu_engine/state.py:49-50 charge_move/charge_counter', 'gpu_engine/solar_beam.py:14-15 (Solar Beam only, solar_beam.py:1)',
         'gpu_engine/transition.py:553-554, 751')),
    Effect('encore', ('encore',), (), True, 'engine_request_source.py:341-342 LOCK while the counters exist and the origin is unknown '
        '(request_source_rules.cjs:64 exempts it from the missing-condition rule)', 'repeat the last move',
        ('gpu_engine/state.py:47-48 encore_move/encore_counter', 'gpu_engine/encore.py:14-20, 33-52')),
    Effect('choice_lock', (), (), True, '', 'Choice Band holder locked into its first move',
        ('gpu_engine/state.py:43 choice_move', 'gpu_engine/state.py:146-148 usable_moves')),
    Effect('move_trap', ('meanlook', 'spiderweb'), (), True, '', 'the target cannot switch while the source stays in',
        ('gpu_engine/state.py:63 trap_source', 'gpu_engine/linked_traps.py:12-25')),
    Effect('seed_and_wish', ('leechseed', 'wish'), (), True, '', 'residual drain; delayed heal',
        ('gpu_engine/state.py:14, 76-77', 'gpu_engine/sustain.py:22-27', 'gpu_engine/transition.py:1799-1823')),
    Effect('stall_counter', ('protect', 'endure'), (), True, '', 'consecutive-use success chance',
        ('gpu_engine/state.py:57 protect_counter', 'gpu_engine/effects.py:173 stall_success')),
    Effect('status_counters', ('rest', 'toxic'), (), True, '', 'sleep turns, Rest origin, Toxic count',
        ('gpu_engine/state.py:36-39',)),
    Effect('weather', ('raindance', 'sunnyday'), (), True, '', 'five-turn weather',
        ('gpu_engine/state.py:78-79 weather/weather_counter', 'gpu_engine/weather.py')),
    Effect('focus_punch', ('focuspunch',), (), True, '', 'charge before the turn, lost on a hit',
        ('gpu_engine/state.py:16-17', 'gpu_engine/focus_punch.py:15-19')),
    Effect('sleep_talk', ('sleeptalk',), (), True, 'request_source_rules.cjs:65-66 a missing caller origin at the request source',
        'calls one of the user\'s moves while asleep', ('gpu_engine/sleep_talk.py:16-22', 'gpu_engine/state.py:38 sleep_skipped')),
    Effect('spikes', ('spikes',), (), True, '', 'entry hazard', ('gpu_engine/state.py:75', 'gpu_engine/entry.py:50')),
    Effect('substitute', ('substitute',), (), True, '', 'HP decoy', ('gpu_engine/state.py:58 substitute_hp',)),
    Effect('baton_pass', ('batonpass',), (), True, '', 'switch keeping boosts and volatiles', ('gpu_engine/baton.py:16',)),
    Effect('truant', (), (), True, '', 'ability: loaf every other turn (Gen 3 residual clock)',
        ('gpu_engine/state.py:84 truant_loaf', 'gpu_engine/transition.py:778, 2551-2555')),
)

NOT_IN_POOL = (
    Effect('locked_move', ('outrage', 'thrash', 'petaldance'), (), False, '', '2-3 turns locked into the move, then confusion',
        ('gpu_engine/state.py:24-92 (BattleState): no locked-move field (future_state.py:155 names lockedmove only as a native duration)',)),
    Effect('uproar', ('uproar',), (), False, 'its condition defines onLockMove', 'locked for several turns, no sleep', ()),
    Effect('rolling', ('rollout', 'iceball'), (), False, '', 'locked for five turns, doubling power', ()),
    Effect('bide', ('bide',), (), False, '', 'stores damage for two turns', ()),
    Effect('rage', ('rage',), (), False, '', 'locked, Attack rises when hit', ()),
    Effect('two_turn', ('fly', 'dig', 'dive', 'bounce', 'skullbash', 'razorwind', 'skyattack'), (), False, '',
        'charge turn (semi-invulnerable for some)', ('gpu_engine/solar_beam.py:1, 14-15: Solar Beam only, not generic charge moves',)),
    Effect('other_partial_trap', ('bind', 'clamp', 'firespin', 'whirlpool', 'sandtomb'), (), False, 'as Wrap', 'as Wrap', ()),
    Effect('future_attack', ('futuresight', 'doomdesire'), (), False, '', 'hits two turns later', ()),
    Effect('request_disable', ('taunt', 'disable', 'torment'), (), False, 'DisableMove handlers (request_source_rules.cjs:59-64)',
        'moves disabled for several turns', ()),
)

AUDITED = {e.id: e for e in EFFECTS}


def normalize(value):
    return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())


def pool_from_sets(path):
    """{move: sorted species} of a gen3 sets.json (every set's movepool)."""
    data = json.loads(Path(path).read_text())
    out = collections.defaultdict(set)
    for species, entry in data.items():
        for item in (entry or {}).get('sets') or ():
            for move in item.get('movepool') or ():out[normalize(move)].add(normalize(species))
    return {m: sorted(s) for m, s in out.items()}


def pool_check(pool):
    """Each row's pool membership against `pool` ({move: species}): mismatches as strings (empty: the snapshot holds)."""
    problems = []
    for e in EFFECTS + REPRESENTED:
        if not e.pool_species:continue
        got = sorted({s for m in e.moves for s in pool.get(m, ())})
        if got != sorted(e.pool_species):problems.append(f'{e.id}: pool species {got}, snapshot {sorted(e.pool_species)}')
    for e in NOT_IN_POOL:
        held = {m: pool[m] for m in e.moves if pool.get(m)}
        if held:problems.append(f'{e.id}: in the pool after all: {held}')
    return problems


def enumeration(showdown_root=None):
    """dict(rows, pool_check): the enumeration table (audited, represented, not in the pool) and, with a Showdown root, the pool
    check against its gen3 sets.json (pool_check None without one)."""
    rows = []
    for group, effects in (('audited', EFFECTS), ('represented', REPRESENTED), ('not_in_pool', NOT_IN_POOL)):
        for e in effects:
            rows.append(dict(group=group, effect=e.id, moves=list(e.moves), pool_species=list(e.pool_species), in_pool=bool(e.pool_species) or group == 'represented',
                represented=e.represented, engine_since=e.engine_since, lock_class=e.lock_class, mechanism=e.mechanism, evidence=list(e.evidence), note=e.note))
    check = None
    if showdown_root is not None:
        path = Path(showdown_root) / SETS_PATH;revision = None
        try:revision = subprocess.check_output(['git', '-C', str(showdown_root), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):pass
        pool = pool_from_sets(path)
        check = dict(path=str(path), revision=revision, pinned=revision == SHOWDOWN_PIN, pool_moves=len(pool), problems=pool_check(pool))
    return dict(rows=rows, pool_check=check)


# Own moves and the root's prefix ---------------------------------------------------------------------------------------------------------
def request_facts(line):
    """dict(moves [4], species) of an own request line: request.active[0].moves[i].id per move lane, the active's species."""
    raw = json.loads(line[len('|request|'):]) if isinstance(line, str) and line.startswith('|request|') else None
    if not isinstance(raw, dict):return dict(moves=[None] * 4, species=None)
    active = raw.get('active');active = active[0] if isinstance(active, list) and active and isinstance(active[0], dict) else {}
    moves = [normalize(m.get('id') or m.get('move')) or None if isinstance(m, dict) else None for m in (active.get('moves') or [])[:4]]
    species = None
    for mon in (raw.get('side') or {}).get('pokemon') or ():
        if isinstance(mon, dict) and mon.get('active'):species = normalize(str(mon.get('details', '')).split(',')[0]) or None;break
    return dict(moves=moves + [None] * (4 - len(moves)), species=species)


def _side(token):
    m = re.match(r'(p[12])[a-z]?:', token or '')
    return m.group(1) if m else None


def carried_effects(lines, owner):
    """{effect: dict(own, opp)}: the audited effects in force on each side's active at the end of `lines` (module docstring (d))."""
    on = {'p1': set(), 'p2': set()}
    for line in lines:
        parts = line.split('|')
        if len(parts) < 3:continue
        kind, side, rest = parts[1], _side(parts[2]), parts[3:]
        if side is None:continue
        first = normalize(rest[0]) if rest else ''
        other = 'p2' if side == 'p1' else 'p1'
        if kind in ('switch', 'drag', 'faint'):
            on[side].clear();on[other].discard('partial_trap')   # the trap ends when its source leaves
            continue
        if kind == '-mustrecharge':on[side].add('recharge')
        elif kind == 'cant' and first == 'recharge':on[side].discard('recharge')
        elif kind == '-activate' and first in ('movewrap', 'wrap'):on[side].add('partial_trap')
        elif kind == '-end' and first in ('movewrap', 'wrap'):on[side].discard('partial_trap')
        elif kind == '-start' and first.startswith('perish'):(on[side].discard if first == 'perish0' else on[side].add)('perish_count')
        elif kind == '-start' and first in ('moveyawn', 'yawn'):on[side].add('drowsy')
        elif (kind == '-end' and first in ('moveyawn', 'yawn')) or (kind == '-status' and first == 'slp'):on[side].discard('drowsy')
        elif kind == '-singlemove' and first == 'destinybond':on[side].add('destiny_bond')
        elif kind == 'move' and first != 'destinybond':on[side].discard('destiny_bond')
        elif kind == '-transform':on[side].add('transform')
        elif kind == '-item' and first == 'choiceband' and any(normalize(x) == 'frommovetrick' for x in rest[1:]):on[side].add('trick_lock')
    them = 'p2' if owner == 'p1' else 'p1'
    return {e: dict(own=e in on.get(owner, ()), opp=e in on.get(them, ())) for e in AUDITED}


class OwningWire:
    """The owning-wire captures (<directory>/<ordinal>.jsonl through phase_d_owning_capture.read_capture), read once per battle:
    per decision_index its own request facts, prefix sha256 and carried effects. Unreadable battles are recorded in `errors`."""
    def __init__(self, directory, *, reader=None):
        self.directory = None if directory is None else Path(directory)
        self.reader = reader;self.cache = {};self.errors = {}

    def _read(self, path):
        if self.reader is not None:return self.reader(path)
        from pokezero import phase_d_owning_capture as owning
        return owning.read_capture(path)

    def get(self, ordinal, decision_index):
        if self.directory is None or ordinal is None:return None
        if ordinal not in self.cache:
            entries = {}
            try:
                for r in self._read(self.directory / f'{ordinal}.jsonl'):
                    entries[r['decision_index']] = dict(request=request_facts(r['own_request_line']), prefix_sha256=r.get('prefix_sha256'),
                        turn=r.get('turn'), carried=carried_effects(r['lines'], r['owner']))
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.errors[ordinal] = f'{type(error).__name__}: {error}'[:300]
            self.cache = {ordinal: entries}   # rows come battle by battle: keep one battle
        return self.cache[ordinal].get(decision_index)


def own_moves_of(row, wire):
    """(moves [4], species, source): root.own_moves when the row has it, else the owning-wire request, else unknown."""
    root = row.get('root') or {}
    given = root.get('own_moves')
    if isinstance(given, list) and given:
        lane = lambda x: None if not isinstance(x, str) or x.startswith('switch:') else normalize(x[len('move:'):] if x.startswith('move:') else x) or None
        moves = [lane(x) for x in given[:4]]
        return moves + [None] * (4 - len(moves)), None, 'row'
    if wire is not None:return list(wire['request']['moves']), wire['request']['species'], 'owning_wire'
    return [None] * 4, None, 'missing'


# Measures -----------------------------------------------------------------------------------------------------------------------------
def mass(logits, mask, lanes):
    """Masked-softmax mass of `lanes` (lane indices) under `mask`; 0 without a legal lane or logits."""
    if not logits or not mask:return 0.
    live = [i for i in range(min(len(logits), len(mask))) if mask[i] and logits[i] is not None and math.isfinite(logits[i])]
    if not live:return 0.
    top = max(float(logits[i]) for i in live);w = {i: math.exp(float(logits[i]) - top) for i in live};total = math.fsum(w.values())
    return math.fsum(w[i] for i in lanes if i in w) / total


def lane_move(label):
    return label[len('move:'):] if isinstance(label, str) and label.startswith('move:') else None


def greedy_of(row):
    """(greedy action, source): live.greedy_action_index, else argmax of root_logits over root_legal (lowest index on ties)."""
    live = row.get('live') or {}
    g = live.get('greedy_action_index')
    if type(g) is int:return g, 'live'
    root = row.get('root') or {};logits, legal = root.get('root_logits'), root.get('root_legal')
    if logits and legal and any(legal):return max((a for a in range(9) if legal[a]), key=lambda a: (logits[a], -a)), 'root_logits'
    return None, None


def turn2_masses(entry, moves):
    """(own max mass, opp max mass, own_max cells) on `moves` over every world and cell of a turn-2 record."""
    own = opp = 0.;argmax = 0
    for w in entry.get('worlds') or ():
        for side in ('own', 'opp'):
            per = collections.defaultdict(float)
            for m in moves:
                for cell, p in ((w.get('mass') or {}).get(side) or {}).get(m, ()):per[cell] += p
            top = max(per.values(), default=0.)
            if side == 'own':own = max(own, top)
            else:opp = max(opp, top)
        argmax += sum(len((w.get('own_max') or {}).get(m, ())) for m in moves)
    return own, opp, argmax


def root_measures(row, wire, turn2, *, threshold=THRESHOLD):
    """Per audited effect, one root's flags and masses (module docstring), plus the root's own-move source and greedy."""
    replay = row.get('replay') or {};root = row.get('root') or {}
    priced = replay.get('status') == 'COMPLETE' and bool(root)
    legal = root.get('root_legal') or [False] * 9;logits = root.get('root_logits')
    own_moves, own_species, own_source = own_moves_of(row, wire)
    greedy, greedy_source = greedy_of(row)
    search = replay.get('action_index') if priced else None
    played = (row.get('live') or {}).get('played_action_index')
    move_at = lambda a: own_moves[a] if type(a) is int and 0 <= a < 4 else None
    # Cells the turn-2 capture expanded but could not label (no P1 child for them): a CAPTURED record with any cannot say (b) is
    # false, only that it is true; without a hit it falls back to the proxy and says so (b_source 'proxy_unlabelled').
    unlabelled = None if turn2 is None else sum(int(w.get('unlabelled') or 0) for w in turn2.get('worlds') or ())
    out = dict(priced=priced, own_source=own_source, greedy_source=greedy_source, own_species=own_species, unlabelled_cells=unlabelled, effects={})
    for e in EFFECTS:
        M = set(e.moves);own_lanes = [i for i in range(4) if own_moves[i] in M]
        a_own = any(legal[i] for i in own_lanes)
        opp_worlds, b1_opp, opp_species = [], 0., set()
        for w in row.get('worlds') or ():
            lanes = (w.get('peer') or {}).get('lanes') or []
            hit = [b for b, label in enumerate(lanes) if lane_move(label) in M]
            if not hit:continue
            opp_worlds.append(w.get('world'));opp_species.add(((w.get('opponent') or {}).get('active_species')))
            if w.get('priced') and w.get('opponent_requested'):b1_opp = max(b1_opp, mass(w.get('root_peer_logits'), w.get('kept'), hit))
        b1_own = mass(logits, legal, own_lanes) if priced else 0.
        flags = dict(a_own=a_own, a_opp=bool(opp_worlds), opp_worlds=opp_worlds, opp_species=sorted(s for s in opp_species if s),
            b1_own=b1_own, b1_opp=b1_opp, b1=priced and max(b1_own, b1_opp) >= threshold)
        if turn2 is not None:
            own2, opp2, argmax = turn2_masses(turn2, e.moves);b2 = priced and max(own2, opp2) >= threshold
            fallback = bool(unlabelled) and not b2   # unlabelled cells may hide the move: no turn-2 verdict, the proxy decides
            flags.update(b2_own=own2, b2_opp=opp2, b2=None if fallback else b2, b_max=priced and argmax > 0, b_max_cells=argmax, b2_unlabelled_fallback=fallback)
        else:flags.update(b2_own=None, b2_opp=None, b2=None, b_max=None, b_max_cells=None, b2_unlabelled_fallback=False)
        flags['b'] = flags['b2'] if flags['b2'] is not None else flags['b1']
        flags['b_source'] = 'turn2' if flags['b2'] is not None else 'proxy_unlabelled' if flags['b2_unlabelled_fallback'] else 'proxy'
        into = lambda chosen: (priced and type(chosen) is int and type(greedy) is int and chosen != greedy and move_at(chosen) in M
            and move_at(greedy) not in M)
        flags.update(c=into(search), c_live=type(played) is int and type(greedy) is int and played != greedy and move_at(played) in M
            and move_at(greedy) not in M, search_is=priced and move_at(search) in M, greedy_is=priced and move_at(greedy) in M)
        carried = (wire or {}).get('carried', {}).get(e.id) if wire else None
        flags.update(d_own=None if carried is None else carried['own'], d_opp=None if carried is None else carried['opp'])
        out['effects'][e.id] = flags
    return out


# Report ---------------------------------------------------------------------------------------------------------------------------------
def jsonl_files(paths, pattern):
    files = []
    for p in map(Path, paths):files += sorted(p.glob(pattern)) if p.is_dir() else [p]
    return files


def read_jsonl(path):
    with Path(path).open() as stream:
        for raw in stream:
            if raw.strip():yield json.loads(raw)


def turn2_index(paths):
    """{(ordinal, decision_index): turn-2 entry} of the capture directories (their turn2.jsonl) or files; ValueError on a duplicate."""
    out = {}
    for f in jsonl_files(paths, 'turn2*.jsonl'):
        for entry in read_jsonl(f):
            key = (entry.get('ordinal'), entry.get('decision_index'))
            if key in out:raise ValueError(f'decision {key} is in two turn-2 records')
            out[key] = entry
    return out


def share(n, d):
    return None if not d else n / d


def audit(rows, wire, turn2=None, *, threshold=THRESHOLD, decision_share=DECISION_SHARE, examples=12):
    """The summary dict over the replay rows (module docstring)."""
    turn2 = turn2 or {}
    counts = collections.Counter();sources = collections.Counter();greedy_sources = collections.Counter()
    tally = {e.id: collections.Counter() for e in EFFECTS};ex = {e.id: collections.defaultdict(list) for e in EFFECTS}
    join = collections.Counter();seen = set();union = collections.Counter()
    for row in rows:
        key = (row.get('ordinal'), row.get('decision_index'))
        if key in seen:counts['duplicate_rows'] += 1;continue
        seen.add(key);counts['rows'] += 1
        w = wire.get(*key) if wire is not None else None
        if w is not None and row.get('prefix_sha256') and w['prefix_sha256'] != row.get('prefix_sha256'):counts['owning_wire_prefix_mismatch'] += 1;w = None
        entry = turn2.get(key)
        if entry is not None:
            if entry.get('prefix_sha256') != row.get('prefix_sha256'):join['prefix_mismatch'] += 1;entry = None
            elif entry.get('turn2') is None or (entry['turn2'] or {}).get('status') != 'CAPTURED':join['not_captured'] += 1;entry = None
        m = root_measures(row, w, None if entry is None else entry['turn2'], threshold=threshold)
        if not m['priced']:continue
        counts['priced'] += 1;sources[m['own_source']] += 1;greedy_sources[m['greedy_source']] += 1
        if entry is not None:
            join['joined'] += 1
            if m['unlabelled_cells']:join['unlabelled_rows'] += 1;join['unlabelled_cells'] += m['unlabelled_cells']
            if entry.get('replay_action_index') != (row.get('replay') or {}).get('action_index'):join['replay_action_differs'] += 1
        else:join['proxy'] += 1
        g = (row.get('live') or {}).get('greedy_action_index');root = row.get('root') or {}
        if type(g) is int and root.get('root_logits'):
            counts['greedy_live_rows'] += 1
            legal = root.get('root_legal');top = max((a for a in range(9) if legal[a]), key=lambda a: (root['root_logits'][a], -a))
            counts['greedy_live_equals_prior_argmax'] += top == g
        address = dict(ordinal=key[0], decision_index=key[1], turn=(row.get('replay') or {}).get('turn'))
        any_cover = any_ext = False
        for e in EFFECTS:
            f = m['effects'][e.id];t = tally[e.id]
            for name in ('a_own', 'a_opp', 'b1', 'c', 'c_live', 'search_is', 'greedy_is', 'd_own', 'd_opp'):t[name] += bool(f[name])
            t['a'] += f['a_own'] or f['a_opp'];t['d'] += bool(f['d_own'] or f['d_opp']);t['d_known'] += f['d_own'] is not None;t['b'] += bool(f['b'])
            t['b_turn2_rows'] += f['b2'] is not None;t['b2_unlabelled_fallback'] += f['b2_unlabelled_fallback']
            if f['b2'] is not None:
                t['b2'] += f['b2'];t['b2_own'] += f['b2_own'] >= threshold;t['b2_opp'] += f['b2_opp'] >= threshold;t['b_max'] += bool(f['b_max'])
            t['b1_own'] += f['b1_own'] >= threshold;t['b1_opp'] += f['b1_opp'] >= threshold
            cover = bool(f['b'] or f['c']);ext = bool(cover or f['b1']);t['cover'] += cover;t['extended'] += ext
            t['cover_proxy'] += cover and f['b2'] is None
            any_cover |= cover;any_ext |= ext
            for name in ('a', 'b', 'b1', 'c', 'c_live', 'd'):
                hit = (f['a_own'] or f['a_opp']) if name == 'a' else (f['d_own'] or f['d_opp']) if name == 'd' else f[name]
                if hit and len(ex[e.id][name]) < examples:
                    ex[e.id][name].append(dict(address, own=dict(species=m['own_species'], legal=f['a_own'], b1=f['b1_own'], b2=f['b2_own'],
                        carried=f['d_own']), opp=dict(species=f['opp_species'], worlds=f['opp_worlds'][:4], b1=f['b1_opp'], b2=f['b2_opp'], carried=f['d_opp']),
                        search=(row.get('replay') or {}).get('action_index'), greedy=greedy_of(row)[0], played=(row.get('live') or {}).get('played_action_index')))
        union['cover'] += any_cover;union['extended'] += any_ext
    priced = counts['priced'];effects = {}
    verdict = lambda n: 'NAMED_RESIDUAL' if priced and n / priced <= decision_share else 'ESCALATE' if priced else 'NO_PRICED_ROOTS'
    for e in EFFECTS:
        t = tally[e.id]
        effects[e.id] = dict(moves=list(e.moves), pool_species=list(e.pool_species),
            a=dict(roots=t['a'], own=t['a_own'], opp=t['a_opp'], share=share(t['a'], priced)),
            b=dict(roots=t['b'], share=share(t['b'], priced), turn2_rows=t['b_turn2_rows'], proxy_rows=priced - t['b_turn2_rows'],
                unlabelled_fallback_rows=t['b2_unlabelled_fallback'],
                source='turn2' if t['b_turn2_rows'] == priced else 'proxy' if not t['b_turn2_rows'] else 'turn2+proxy'),
            b_turn2=None if not t['b_turn2_rows'] else dict(roots=t['b2'], own=t['b2_own'], opp=t['b2_opp'], rows=t['b_turn2_rows'],
                share=share(t['b2'], t['b_turn2_rows']), learner_max_rows=t['b_max']),
            b_turn1=dict(roots=t['b1'], own=t['b1_own'], opp=t['b1_opp'], share=share(t['b1'], priced)),
            c=dict(roots=t['c'], share=share(t['c'], priced), live=t['c_live'], search_is=t['search_is'], greedy_is=t['greedy_is']),
            d=dict(roots=t['d'], own=t['d_own'], opp=t['d_opp'], share=share(t['d'], priced), known=t['d_known']),
            coverage=dict(roots=t['cover'], share=share(t['cover'], priced), rule=f'(b) or (c) <= {decision_share:g} of priced roots'),
            decision=verdict(t['cover']), provisional=bool(t['cover_proxy']) or t['b_turn2_rows'] < priced,
            extended=dict(roots=t['extended'], share=share(t['extended'], priced), decision=verdict(t['extended'])),
            examples={k: v for k, v in ex[e.id].items()})
    r = effects['recharge']['c']
    guard = dict(search_recharge_not_greedy=r['roots'], search_recharge=r['search_is'], greedy_recharge=r['greedy_is'], live_played_recharge_not_greedy=r['live'],
        moves=list(AUDITED['recharge'].moves), note='replay searched action (replay.action_index) against live greedy over priced roots')
    return dict(schema=SCHEMA, threshold=threshold, decision_share=decision_share, counts=dict(counts), priced_roots=priced,
        own_move_sources=dict(sources), greedy_sources=dict(greedy_sources), turn2_join=dict(join, records=len(turn2)),
        owning_wire_errors=dict(wire.errors) if wire is not None else {}, effects=effects,
        any_effect=dict(coverage=dict(roots=union['cover'], share=share(union['cover'], priced)), extended=dict(roots=union['extended'],
            share=share(union['extended'], priced))), recharge_guard=guard)


def fmt(x, digits=4):
    return '-' if x is None else f'{x:.{digits}f}' if isinstance(x, float) else str(x)


def markdown(summary):
    s = summary;lines = ['# Widened carry audit: unrepresented multi-turn effects in the live DepthN search', '']
    c = s['counts'];j = s['turn2_join']
    lines += [f"Rows {c.get('rows', 0)}, priced roots (replay COMPLETE) {s['priced_roots']}. Own moves: {s['own_move_sources']}. "
        f"Greedy: {s['greedy_sources']} (live greedy = root-prior argmax on {c.get('greedy_live_equals_prior_argmax', 0)}/{c.get('greedy_live_rows', 0)}). "
        f"Turn-2 records: {j.get('records', 0)}, joined {j.get('joined', 0)}, proxy {j.get('proxy', 0)}, prefix mismatches {j.get('prefix_mismatch', 0)}, "
        f"not captured {j.get('not_captured', 0)}, replay action differs {j.get('replay_action_differs', 0)}; joined with unlabelled turn-2 cells "
        f"{j.get('unlabelled_rows', 0)} ({j.get('unlabelled_cells', 0)} cells: where those records find no hit, (b) falls back to the proxy, "
        "source proxy_unlabelled).", '']
    if s['owning_wire_errors']:lines += [f"Owning-wire read errors: {len(s['owning_wire_errors'])} battles ({sorted(s['owning_wire_errors'])[:10]}).", '']
    e = s.get('enumeration')
    if e:
        lines += ['## Enumeration', '', '| group | effect | moves | pool species | represented | LOCK class | evidence |', '| --- | --- | --- | --- | --- | --- | --- |']
        for r in e['rows']:
            lines.append(f"| {r['group']} | {r['effect']} | {', '.join(r['moves']) or '-'} | {', '.join(r['pool_species']) or '-'} | {'yes' if r['represented'] else 'no'} | "
                f"{r['lock_class'] or '-'} | {'; '.join(r['evidence']) or '-'} |")
        notes = [r for r in e['rows'] if r['note']]
        if notes:lines += [''] + [f"- {r['effect']}: {r['note']}" for r in notes]
        pc = e.get('pool_check')
        lines += ['', f"Pool check: {'not run (no --showdown-root)' if pc is None else ('holds' if not pc['problems'] else 'MISMATCH: ' + '; '.join(pc['problems']))}"
            + ('' if pc is None else f" ({pc['path']}, revision {pc['revision']}, pinned {pc['pinned']}, {pc['pool_moves']} moves)"), '']
    lines += ['## Measures per audited effect (counts of priced roots)', '',
        f"(b) needs >= {s['threshold']:g} of a side's turn-2 prior on the effect at some expanded cell of some world; its source is the turn-2 capture "
        "where joined, else the turn-1 proxy (b1). Decision: (b) or (c) over priced roots <= " + f"{s['decision_share']:g}.", '',
        '| effect | (a) own / opp / any | (b) roots [source] | (b) turn 2 own / opp [rows] | b_max | (b1) own / opp / any | (c) replay [live] | '
        'search / greedy on it | coverage | decision | extended | (d) own / opp [roots with a prefix] |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, x in s['effects'].items():
        b2 = x['b_turn2']
        lines.append(f"| {name} | {x['a']['own']} / {x['a']['opp']} / {x['a']['roots']} | {x['b']['roots']} [{x['b']['source']}] | "
            f"{'-' if b2 is None else str(b2['own']) + ' / ' + str(b2['opp']) + ' [' + str(b2['rows']) + ']'} | {'-' if b2 is None else b2['learner_max_rows']} | "
            f"{x['b_turn1']['own']} / {x['b_turn1']['opp']} / {x['b_turn1']['roots']} | {x['c']['roots']} [{x['c']['live']}] | {x['c']['search_is']} / {x['c']['greedy_is']} | "
            f"{x['coverage']['roots']} ({fmt(x['coverage']['share'])}) | {x['decision']}{' (PROVISIONAL)' if x['provisional'] else ''} | "
            f"{x['extended']['roots']} ({fmt(x['extended']['share'])}) {x['extended']['decision']} | {x['d']['own']} / {x['d']['opp']} [{x['d']['known']}] |")
    a = s['any_effect']
    lines += ['', f"Any audited effect: coverage {a['coverage']['roots']} ({fmt(a['coverage']['share'])}), extended {a['extended']['roots']} ({fmt(a['extended']['share'])}).", '']
    g = s['recharge_guard']
    lines += ['## Recharge guard', '', f"Our searched action is a recharge move ({', '.join(g['moves'])}) while greedy's is not: {g['search_recharge_not_greedy']} priced roots "
        f"(searched recharge {g['search_recharge']}, greedy recharge {g['greedy_recharge']}; live played recharge while greedy not: {g['live_played_recharge_not_greedy']}).", '']
    lines += ['## How (b) is measured', '', 'At every turn-1 cell the depth-2 frame expanded (t1_status 2), each requested seat\'s turn-2 prior is the masked softmax '
        'of its turn-2 logits over its enumerated slots (depthn_backups._softmax over t1_own_actions / t1_opp_actions: the opponent prior learner_max averages). '
        'A move lane is labelled from the turn-1 child state (its active\'s move slot; the charge move or Struggle on lane 0). The proxy (b1) is the root prior '
        '(ours over root_legal, the opponent\'s per world over kept) on the effect\'s lanes; it misses moves of actives that switch in at turn 1.', '']
    lines += ['## Examples', '']
    for name, x in s['effects'].items():
        for kind, rows in x['examples'].items():
            for r in rows[:5]:lines.append(f"- {name} {kind}: battle {r['ordinal']} decision {r['decision_index']} turn {r['turn']}: own {r['own']}, opp {r['opp']}, "
                f"search {r['search']} greedy {r['greedy']} played {r['played']}")
    return '\n'.join(lines) + '\n'


def write_json(path, value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value, indent=1, sort_keys=True, default=repr) + '\n')


def report_main(argv):
    p = argparse.ArgumentParser(prog='depthn_carry_audit.py report', allow_abbrev=False)
    p.add_argument('--rows', nargs='+', required=True, help='re-pricer rows: rows-*.jsonl files or directories holding them (every shard)')
    p.add_argument('--owning-wire', help='the live run\'s owning-wire directory (<ordinal>.jsonl): our move per lane and the root\'s prefix')
    p.add_argument('--turn2', nargs='+', default=(), help='turn-2 capture directories (their turn2.jsonl) or files')
    p.add_argument('--showdown-root', help='the pinned Showdown checkout: re-read gen3 sets.json for the pool check')
    p.add_argument('--threshold', type=float, default=THRESHOLD);p.add_argument('--decision-share', type=float, default=DECISION_SHARE)
    p.add_argument('--examples', type=int, default=12);p.add_argument('--out', required=True, help='a new directory')
    args = p.parse_args(argv)
    files = jsonl_files(args.rows, 'rows-*.jsonl')
    if not files:p.error('no rows-*.jsonl under --rows')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    wire = OwningWire(args.owning_wire);turn2 = turn2_index(args.turn2) if args.turn2 else {}
    rows = (r for f in files for r in read_jsonl(f))
    summary = audit(rows, wire, turn2, threshold=args.threshold, decision_share=args.decision_share, examples=args.examples)
    summary['enumeration'] = enumeration(args.showdown_root)
    summary['inputs'] = dict(rows=[str(f) for f in files], owning_wire=args.owning_wire, turn2=[str(x) for x in args.turn2], showdown_root=args.showdown_root,
        argv=list(argv))
    write_json(out / 'summary.json', summary);(out / 'summary.md').write_text(markdown(summary))
    print(json.dumps(dict(done=str(out), priced=summary['priced_roots'], decisions={k: v['decision'] + ('*' if v['provisional'] else '') for k, v in summary['effects'].items()})))
    return 0


def enumerate_main(argv):
    p = argparse.ArgumentParser(prog='depthn_carry_audit.py enumerate', allow_abbrev=False)
    p.add_argument('--showdown-root');p.add_argument('--out', help='a new directory for enumeration.json')
    args = p.parse_args(argv);e = enumeration(args.showdown_root)
    if args.out:Path(args.out).mkdir(parents=True, exist_ok=False);write_json(Path(args.out) / 'enumeration.json', e)
    print(json.dumps(e, indent=1))
    return 1 if e['pool_check'] and e['pool_check']['problems'] else 0


# Turn-2 capture --------------------------------------------------------------------------------------------------------------------------
@dataclass
class Children:
    """The turn-1 children of one P1 call (host arrays): lane_cell [n, 81] and, per flat child row (world row * 81 + lane), the
    active [N, 2], its four move ids [N, 2, 4], its charge move [N, 2] and usable_moves [N, 2, 4]."""
    lane_cell: object
    active: object
    moves: object
    charge: object
    usable: object
    rows: dict = field(default_factory=dict)

    def row(self, world_row, cell):
        """The flat child row of (world row, turn-1 cell), -1 when P1 did not enumerate it."""
        if not self.rows:
            import numpy as np
            lane_cell = np.asarray(self.lane_cell);lanes = lane_cell.shape[1]
            for w, lane in zip(*np.nonzero(lane_cell < lanes)):self.rows[(int(w), int(lane_cell[w, lane]))] = int(w) * lanes + int(lane)
        return self.rows.get((int(world_row), int(cell)), -1)


def _child_arrays(states):
    import jax
    import jax.numpy as j
    from gpu_engine.state import usable_moves
    s = jax.tree.map(lambda x: x[:, 0], states)   # P1's flat children are [n*81, 1, ...]: one-battle states
    act = j.clip(s.active, 0, 5)
    moves = j.take_along_axis(s.moves, act[:, :, None, None], axis=2)[:, :, 0]
    charge = j.take_along_axis(s.charge_move, act[:, :, None], axis=2)[:, :, 0]
    return act, moves, charge, usable_moves(s)


_CHILD = {}


def child_facts(out):
    """Children of a P1 output dict (its 'states' and 'lane_cell'), read through one jitted gather (cached per process)."""
    import jax
    import numpy as np
    if 'program' not in _CHILD:_CHILD['program'] = jax.jit(_child_arrays)
    (active, moves, charge, usable), lane_cell = jax.device_get((_CHILD['program'](out['states']), out['lane_cell']))
    return Children(np.asarray(lane_cell), np.asarray(active), np.asarray(moves), np.asarray(charge), np.asarray(usable))


def child_lanes(children, child, seat, name):
    """The four move-lane labels of one seat at a child, as the encoder fills its lanes (module docstring)."""
    charge = int(children.charge[child, seat])
    if charge > 0:return [name(charge), None, None, None]
    if not bool(children.usable[child, seat].any()):return ['struggle', None, None, None]
    return [name(int(m)) for m in children.moves[child, seat]]


def learner_max_action(frame, r, j, a, b):
    """The own action of the learner_max row at a depth-2 turn-1 cell (strict max, the lower slot on ties), None when the frame is
    deeper, the learner waits, or a leaf under the cell is not priced."""
    from gpu_engine import depthn_backups as backups
    if frame.depth != 2 or not bool(frame.t1_own_requested[r, j, a, b]):return None
    c = a * 9 + b
    own = [int(x) for x in frame.t1_own_actions[r, j, a, b]];opp = [int(x) for x in frame.t1_opp_actions[r, j, a, b]]
    own_slots = [i for i, x in enumerate(own) if x >= 0];opp_slots = [i for i, x in enumerate(opp) if x >= 0]
    if not own_slots or not opp_slots:return None
    pi = backups._softmax(frame.t1_peer_logits[r, j, a, b], [opp[x] for x in opp_slots])
    best = best_i = None
    for i in own_slots:
        row = []
        for x in opp_slots:
            if int(frame.leaf_status[r, j, c, i, x]) not in (1, 4):return None
            row.append(float(frame.leaf_value[r, j, c, i, x]))
        q = math.fsum(p * v for p, v in zip(pi, row))
        if best is None or q > best:best, best_i = q, i
    return own[best_i]


def world_turn2(frame, r, j, world, owner, children, name, watch):
    """One world's turn-2 record (module docstring): frame root row r, world j; children the P1 children (None: unlabelled)."""
    from gpu_engine import depthn_backups as backups
    out = dict(world=world, frame_root=int(r), frame_world=int(j), pre=int(frame.pre[r, j]), expanded=0, unlabelled=0, unsolved=0)
    mass = dict(own=collections.defaultdict(dict), opp=collections.defaultdict(dict));own_max = collections.defaultdict(list)
    labels = dict(own=set(), opp=set())
    if out['pre'] == 0:
        status = frame.t1_status[r, j]
        for a in range(9):
            for b in range(9):
                if int(status[a, b]) != 2:continue
                out['expanded'] += 1;c = a * 9 + b
                child = -1 if children is None else children.row(2 * r + j, c)
                if child < 0:out['unlabelled'] += 1;continue
                lanes = dict(own=child_lanes(children, child, owner, name), opp=child_lanes(children, child, 1 - owner, name))
                for side, requested, actions, logits in (('own', frame.t1_own_requested, frame.t1_own_actions, frame.t1_own_logits),
                        ('opp', frame.t1_peer_requested, frame.t1_opp_actions, frame.t1_peer_logits)):
                    if not bool(requested[r, j, a, b]):continue
                    slots = [int(x) for x in actions[r, j, a, b] if int(x) >= 0]
                    if not slots:continue
                    for act, p in zip(slots, backups._softmax(logits[r, j, a, b], slots)):
                        label = lanes[side][act] if act < 4 else None
                        if label is None:continue
                        labels[side].add(label)
                        if label in watch and p > 0:mass[side][label][c] = mass[side][label].get(c, 0.) + p
                best = learner_max_action(frame, r, j, a, b)
                if best is None:out['unsolved'] += bool(frame.t1_own_requested[r, j, a, b])
                elif best < 4 and lanes['own'][best] in watch:own_max[lanes['own'][best]].append(c)
    out.update(mass={side: {m: [[c, p] for c, p in sorted(cells.items())] for m, cells in per.items()} for side, per in mass.items()},
        own_max=dict(own_max), labels={side: sorted(v) for side, v in labels.items()})
    return out


def turn2_record(capture, owner, vocabulary, watch, *, name=None):
    """The decision's turn-2 record from a Turn2Capture (never raises: problems are its status and notes)."""
    notes = list(capture.turn2_notes)
    try:
        if not capture.priced:return dict(schema=TURN2_SCHEMA, status='NOT_PRICED', notes=notes)
        if len(capture.priced) > 1 or len(capture.children) != len(capture.priced):
            notes.append(f'{len(capture.priced)} price calls, {len(capture.children)} P1 captures: the first of each is used')
        batch, frame = capture.priced[0]
        if frame is None:return dict(schema=TURN2_SCHEMA, status='NO_FRAME', notes=notes)
        roots = [x for x in batch.roots if x.get('status') == 'PRICED']
        if len(roots) != 1:return dict(schema=TURN2_SCHEMA, status='PRICED_ROOTS', priced_roots=len(roots), notes=notes)
        if owner not in (0, 1):return dict(schema=TURN2_SCHEMA, status='NO_OWNER', notes=notes)
        if name is None:
            from gpu_engine.phase_d_live_worlds import _source_move
            name = lambda i: None if i <= 0 else _source_move(vocabulary['moves'][i])
        children = capture.children[0] if capture.children else None
        if children is None:notes.append('no P1 capture: no cell can be labelled')
        worlds = [world_turn2(frame, r, j, 2 * k + j, owner, children, name, set(watch)) for k, r in enumerate(roots[0]['rows']) for j in (0, 1)]
        return dict(schema=TURN2_SCHEMA, status='CAPTURED', depth=int(frame.depth), owner=owner, watch=sorted(watch), worlds=worlds, notes=notes)
    except Exception as error:   # a capture problem never stops the replay
        return dict(schema=TURN2_SCHEMA, status='ERROR', error=f'{type(error).__name__}: {error}'[:500], notes=notes)


def check_turn2_call_sites(search=None, price=None):
    """True when DepthNLiveSearch._search calls cw.price through the module attribute and DepthNPricer.price calls self._program
    through the attribute (so the class-level wrapper applies); RuntimeError otherwise."""
    from gpu_engine import constructor_worlds as cw, depthn_live_search as live, device_depthn_pricer as dn
    search = live.DepthNLiveSearch._search if search is None else search
    price = dn.DepthNPricer.price if price is None else price
    found = dict(price=False, program=False);previous = None
    for i in dis.get_instructions(search):
        if (i.opname in ('LOAD_ATTR', 'LOAD_METHOD') and i.argval == 'price' and previous is not None and previous.opname == 'LOAD_GLOBAL'
                and previous.argval == 'cw'):found['price'] = True
        previous = i
    previous = None
    for i in dis.get_instructions(price):   # self is a local or, captured by price's lambdas, a cell
        if (i.opname in ('LOAD_ATTR', 'LOAD_METHOD') and i.argval == '_program' and previous is not None
                and (previous.opname.startswith('LOAD_FAST') or previous.opname == 'LOAD_DEREF')
                and (previous.argval == 'self' or isinstance(previous.argval, tuple) and previous.argval[-1] == 'self')):found['program'] = True
        previous = i
    if not all(found.values()) or search.__globals__ is not vars(live) or live.cw is not cw:
        raise RuntimeError(f'the search does not reach cw.price / self._program at call time: {found}')
    return True


def check_reprice_hooks(function=None):
    """True when the re-pricer's main reads Capture, describe and decision_row as module globals at call time."""
    from scripts import depthn_rr_reprice as rr
    function = rr.main if function is None else function
    loaded = {i.argval for i in dis.get_instructions(function) if i.opname == 'LOAD_GLOBAL'}
    missing = {'Capture', 'describe', 'decision_row'} - loaded
    if missing or function.__globals__ is not vars(rr):raise RuntimeError(f'the re-pricer main does not look {sorted(missing)} up at call time')
    return True


class Recorder:
    """turn2.jsonl of a capture run: the describe wrapper builds the decision's record, the decision_row wrapper writes it."""
    def __init__(self, out, watch):
        self.out, self.watch = Path(out), tuple(watch)
        self.stream = (self.out / 'turn2.jsonl').open('x');self.current = self.pending = None;self.counts = collections.Counter()

    def described(self, capture, detail, vocabulary):
        owner = (detail.get('prepared') or {}).get('owner') if isinstance(detail, dict) else None
        self.pending = turn2_record(capture, owner, vocabulary, self.watch) if hasattr(capture, 'priced') else None

    def written(self, record):
        turn2, self.pending = self.pending, None
        replay = record.get('replay') or {}
        entry = dict(schema=TURN2_SCHEMA, ordinal=record.get('ordinal'), decision_index=record.get('decision_index'), battle_id=record.get('battle_id'),
            owner=record.get('owner'), prefix_sha256=record.get('prefix_sha256'), replay_status=replay.get('status'), replay_action_index=replay.get('action_index'),
            turn=replay.get('turn'), turn2=turn2)
        self.counts[(turn2 or {}).get('status', 'NONE')] += 1
        self.stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n');self.stream.flush();os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def capture_class(recorder):
    """rr.Capture whose installed() also wraps constructor_worlds.price and DepthNPricer._program (module docstring)."""
    from scripts import depthn_rr_reprice as rr
    from gpu_engine import constructor_worlds as cw, device_depthn_pricer as dn

    class Turn2Capture(rr.Capture):
        def __init__(self):
            super().__init__();self.priced, self.children, self.turn2_notes = [], [], [];recorder.current = self

        @contextmanager
        def installed(self):
            with super().installed():
                price, program, capture = cw.price, dn.DepthNPricer.__dict__['_program'], self

                def price_wrapper(*arguments, **keywords):
                    out = price(*arguments, **keywords)
                    batch = arguments[1] if len(arguments) > 1 else keywords.get('batch')
                    capture.priced.append((batch, out[0] if isinstance(out, tuple) and out else None))
                    return out

                def program_wrapper(pricer, key, build, *arguments):
                    compiled = program(pricer, key, build, *arguments)
                    if not (type(key) is tuple and key and key[0] == 'P1'):return compiled

                    def run(*call):
                        out = compiled(*call)
                        try:capture.children.append(child_facts(out))
                        except Exception as error:capture.turn2_notes.append(f'P1 children: {type(error).__name__}: {error}'[:300])
                        return out
                    return run
                cw.price = price_wrapper;dn.DepthNPricer._program = program_wrapper
                try:yield self
                finally:cw.price = price;dn.DepthNPricer._program = program
    return Turn2Capture


@contextmanager
def turn2_hooks(recorder):
    """rr.Capture -> Turn2Capture, rr.describe and rr.decision_row -> pass-through wrappers feeding `recorder`, while main runs."""
    from scripts import depthn_rr_reprice as rr
    check_reprice_hooks();check_turn2_call_sites()
    capture, describe, decision_row = rr.Capture, rr.describe, rr.decision_row

    def describe_wrapper(capture_object, peer_lanes, vocabulary, *arguments, **keywords):
        # Every further argument passes through unchanged: the re-pricer's describe also takes (peer_inputs, (ordinal, decision_index)).
        detail = describe(capture_object, peer_lanes, vocabulary, *arguments, **keywords)
        recorder.described(capture_object, detail, vocabulary)
        return detail

    def row_wrapper(*arguments, **keywords):
        record = decision_row(*arguments, **keywords)
        recorder.written(record)
        return record
    rr.Capture, rr.describe, rr.decision_row = capture_class(recorder), describe_wrapper, row_wrapper
    try:yield
    finally:rr.Capture, rr.describe, rr.decision_row = capture, describe, decision_row


def _drop(argv, flag):
    """argv without `flag VALUE` / `flag=VALUE`."""
    out, skip = [], False
    for x in argv:
        if skip:skip = False;continue
        if x == flag:skip = True;continue
        if x.startswith(flag + '='):continue
        out.append(x)
    return out


def reprice_argv(like, rest, out, head):
    """The re-pricer argv of a capture run: a previous run's header argv (--like) without --out / --tree-commit / --accept-tree,
    then `rest`, then --accept-tree HEAD (unless `rest` names one) and --out <out>/replay."""
    argv = []
    if like is not None:
        argv = list(json.loads(Path(like).read_text())['argv'])
        for flag in ('--out', '--tree-commit', '--accept-tree'):argv = _drop(argv, flag)
    argv += list(rest)
    if any(x == '--out' or x.startswith('--out=') for x in argv):raise ValueError('--out is the capture\'s (<out>/replay), not passed to the re-pricer')
    if head is not None and not any(x == '--accept-tree' or x.startswith('--accept-tree=') for x in argv):argv += ['--accept-tree', head]
    return argv + ['--out', str(Path(out) / 'replay')]


def capture_main(argv):
    p = argparse.ArgumentParser(prog='depthn_carry_audit.py capture', allow_abbrev=False, description='The re-pricer\'s replay with the turn-2 '
        'record (module docstring). Every other argument goes to scripts/depthn_rr_reprice.py.')
    p.add_argument('--out', required=True, help='a new directory: turn2.jsonl, turn2-header.json, turn2-summary.json and replay/')
    p.add_argument('--like', metavar='HEADER_JSON', help='a previous re-pricer run\'s header-<shard>.json: its argv, re-targeted (module docstring)')
    p.add_argument('--watch', nargs='+', help='move ids whose turn-2 mass is recorded (default: every audited effect\'s moves)')
    p.add_argument('--dry-run', action='store_true', help='print the re-pricer argv and exit')
    args, rest = p.parse_known_args(argv)
    watch = tuple(sorted({normalize(m) for m in args.watch})) if args.watch else tuple(sorted({m for e in EFFECTS for m in e.moves}))
    head = None
    try:head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):pass
    try:reprice = reprice_argv(args.like, rest, args.out, head)
    except (ValueError, OSError, KeyError) as error:p.error(str(error))
    if args.dry_run:print(json.dumps(dict(reprice_argv=reprice, watch=watch, head=head)));return 0
    from scripts import depthn_rr_reprice as rr
    try:check_reprice_hooks();check_turn2_call_sites()
    except RuntimeError as error:raise SystemExit(f'depthn_carry_audit: {error}')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'turn2-header.json', dict(argv=list(argv), reprice_argv=reprice, like=args.like, head=head, watch=watch, schema=TURN2_SCHEMA,
        pid=os.getpid(), python=sys.version))
    recorder = Recorder(out, watch);started = time.perf_counter();code = None;finished = False
    try:
        with turn2_hooks(recorder):code = rr.main(reprice)
        finished = True
    finally:
        recorder.close()
        write_json(out / 'turn2-summary.json', dict(statuses=dict(recorder.counts), reprice_exit=code, interrupted=not finished,
            seconds=time.perf_counter() - started))
    print(json.dumps(dict(done=str(out), statuses=dict(recorder.counts), reprice_exit=code)), flush=True)
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = dict(report=report_main, capture=capture_main, enumerate=enumerate_main)
    if not argv or argv[0] not in commands:
        print(__doc__.split('\n\n')[1], file=sys.stderr);return 2
    return commands[argv[0]](argv[1:])


if __name__ == '__main__':raise SystemExit(main())
