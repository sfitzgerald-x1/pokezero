"""Behavioral reweighting test (design gate 5): does weighting our candidate sets for Foul Play's mons by π_FP's likelihood of
Foul Play's public choices raise the probability of each mon's true set?

usage: python -m gpu_engine.fp_reply_reweight COLLECTION OUT --pi-fp PT [--pi-fp-data DIR] [--shard I --shards N] [--limit N]
       (--pi-fp-data defaults to the π_FP checkpoint's fp_reply.data marker; its shards must name COLLECTION)
       python -m gpu_engine.fp_reply_reweight --report OUT

Each game is replayed exactly as fp_reply_data replays it (Foul Play's journaled choices decoded on its own state, our seat's
requests equal to the capture, the winner). At each Foul Play decision t that is not a forced switch:
  - C_t(m) is our seat's candidate variants for each Foul Play mon m we have seen (the env's belief engine,
    resolved_player_view(ours)): what the world sampler draws from, uniformly.
  - m's true set is read off Foul Play's first request (moves as a set, ability and item by id).
  - For the ACTIVE mon only, each distinct variant v in C_t gets a world: Foul Play's request with the active mon's moves, ability
    and item replaced by v's. Shared moves keep their true slot, PP and disabled flag. Hidden Power is one move family (the
    protocol's |move| line names no type): a different type takes the true Hidden Power's slot, PP and disabled flag, and a
    chosen Hidden Power maps to it. Other new moves take full PP. A decision where some world puts a new move into a disabled
    slot is excluded (counted): the flag's cause (Taunt, Imprison, a choice lock) depends on the set. An item the battle has
    publicly removed or swapped stays as it is in every world. The world is encoded by env.observe(fp) through a substituted
    replay snapshot -- the encoder π_FP was trained on; a π_FP trained on a redacted view (fp_reply.redaction) is scored on the
    same redaction -- and π_FP scores the choice: log q(a_t | v), -inf when a_t is a move v lacks.
  - Evidence: a decision counts only if its choice became public that step (Foul Play's |move| line for the chosen move, not a
    called move, or a |switch| / |drag| line). L_m(v) sums log q over m's public decisions as the active mon.
  - Our seat observes first in every round it is requested, as live play does, so its trackers narrow C_t on time.
  - Before adding decision t's own evidence, every seen mon m with |C_t(m)| > 1 is measured: prior uniform over C_t(m),
    posterior ∝ prior * exp(beta * L_m(v)) for beta in BETAS; delta = log P_post(true) - log P_prior(true).
  - Posterior-predictive NLL of a_t: the prior mixture over the active mon's worlds (today's uniform sampler), the posterior
    mixture, and π_FP on Foul Play's actual view (the true set: an oracle no search has).
Gate 5 (part 2 amended by the lead after the first result): the mean delta (beta 1) over measured (decision, mon) rows whose true
set is in C_t, that have evidence and whose every candidate has seen all of it, is positive with a 90% interval clustered on
battle seed excluding 0; and the posterior-predictive NLL is no worse than the prior mixture's. The original part 2, against π_FP
on the true view (an oracle), is still reported.
Stats: each world's attack, defense, special attack, special defense and speed are the ones the Gen 3 generator gives that set
(85 EVs, 31 IVs or the Hidden Power type's IVs, and attack dumped when the set has no physical damaging move -- the type decides
a Hidden Power's class), from the species' base stats and level, but only when the same formula reproduces Foul Play's true
stats exactly (otherwise every world keeps the true stats, counted, and rows that relied on such evidence are flagged).
Approximations, an optimistic context for a live sampler: only the active mon's set varies in a world (the bench is Foul Play's
true team, which Foul Play knows); HP and shared moves' PP and disabled flags stay the true set's; the Tier-2 and investment
annotations read Foul Play's first request, not the substituted one; new Hidden Power moves are named with power 70 and
Return / Frustration with 102.
Checks, counted per shard: the unchanged request through the substituted snapshot encodes exactly as env.observe(fp); the true
set's world encodes exactly as env.observe(fp); env.observe(fp) is unchanged after the worlds; fresh move entries equal the
request's own entries for unused moves.
"""
import argparse
import collections
import contextlib
import copy
import dataclasses
import glob
import json
import re
import time
from pathlib import Path

import numpy as np

from gpu_engine import fp_reply_data as data_mod
from gpu_engine import fp_reply_probe as probe_mod
from gpu_engine import fp_reply_train as train_mod

BETAS = (0.25, 0.5, 1.0)
# moves whose request id and name carry a power: Hidden Power's name (its side id carries the type only), Return / Frustration
# both (randbats sets happiness so either is at 102)
POWERED = {'hiddenpower': 70, 'return': 102, 'frustration': 102}
OBS_DTYPES = dict(categorical_ids=np.int32, numeric_features=np.float32, token_type_ids=np.int16, attention_mask=bool)


def toid(value):
    return re.sub(r'[^a-z0-9]', '', str(value or '').lower())


def move_key(move):
    """A move id without any power suffix ('return102' -> 'return', 'hiddenpowersteel70' -> 'hiddenpowersteel')."""
    return re.sub(r'\d+$', '', toid(move))


def request_move_id(move):
    """A move key in a request's side.pokemon[].moves form ('return' -> 'return102'; Hidden Power keeps its type only)."""
    return f'{move}{POWERED[move]}' if move in POWERED and move != 'hiddenpower' else move


def family(move):
    """Moves a |move| line can't tell apart: every Hidden Power type is 'Hidden Power'."""
    return 'hiddenpower' if move.startswith('hiddenpower') else move


def variant_key(variant):
    return frozenset(move_key(m) for m in variant.get('moves') or ()), toid(variant.get('ability')), toid(variant.get('item'))


def family_key(key):
    """A set key with Hidden Power's type dropped: candidates equal under it differ only in the Hidden Power type."""
    return frozenset(family(m) for m in key[0]), key[1], key[2]


def request_key(row):
    return frozenset(move_key(m) for m in row.get('moves') or ()), toid(row.get('baseAbility') or row.get('ability')), toid(row.get('item'))


def species_of(row):
    return toid(str(row.get('details') or '').split(',')[0])


GEN3_TABLES_SCRIPT = r"""
const root = process.argv[1];
const {Moves} = require(root + '/dist/data/moves.js');
const {Dex} = require(root + '/dist/sim');
const gen3 = Dex.forGen(3);
const out = {fixed_damage: [], hp_ivs: {}};
for (const id of Object.keys(Moves)) { const m = gen3.moves.get(id); if (m && m.exists && (m.damage || m.damageCallback)) out.fixed_damage.push(m.id); }
for (const t of gen3.types.all()) out.hp_ivs[t.id] = t.HPivs || {};
console.log(JSON.stringify(out));
"""
STATS = ('atk', 'def', 'spa', 'spd', 'spe')


def generator_tables(showdown_root):
    """From Showdown's Gen 3 dex: fixed-damage moves (the generator counts them as neither physical nor special) and each type's
    Hidden Power IVs."""
    import subprocess
    out = subprocess.run(['node', '-e', GEN3_TABLES_SCRIPT, str(Path(showdown_root).resolve())], check=True, capture_output=True,
                         text=True, timeout=60)
    tables = json.loads(out.stdout)
    return dict(fixed_damage=frozenset(tables['fixed_damage']), hp_ivs={k: dict(v) for k, v in tables['hp_ivs'].items()})


def generator_stats(base, level, moves, dex, tables):
    """The Gen 3 random generator's battle stats (atk, def, spa, spd, spe) for a set: 85 EVs and 31 IVs, a Hidden Power type's
    IVs, and attack dumped (EV 0; IV 0, or the Hidden Power IV - 28) when no move is a physical damaging move; neutral nature."""
    ivs = {k: 31 for k in STATS};evs = {k: 85 for k in STATS};hidden = [m for m in moves if m.startswith('hiddenpower') and m != 'hiddenpower']
    if hidden:ivs.update({k: v for k, v in tables['hp_ivs'].get(hidden[0][len('hiddenpower'):], {}).items() if k in ivs})
    physical = any((info := dex.move_info(m)) is not None and info.gen3_category == 'Physical' and m not in tables['fixed_damage']
                   for m in moves)
    if not physical and 'transform' not in moves:evs['atk'] = 0;ivs['atk'] = (ivs['atk'] or 31) - 28 if hidden else 0
    return {k: (2 * int(base[k]) + ivs[k] + evs[k] // 4) * level // 100 + 5 for k in STATS}


def attack_dumped(moves, dex, tables):
    return not any((info := dex.move_info(m)) is not None and info.gen3_category == 'Physical' and m not in tables['fixed_damage']
                   for m in moves) and 'transform' not in moves


def level_of(row):
    match = re.search(r',\s*L(\d+)', str(row.get('details') or ''))
    return int(match.group(1)) if match else 100


def fresh_move_entry(dex, move, disabled=False):
    """A request's active-move entry for a move the mon has never used (full PP)."""
    if move.startswith('hiddenpower') and move != 'hiddenpower':
        info = dex.move_info('hiddenpower')
        name, ident = f'Hidden Power {move[len("hiddenpower"):].capitalize()} {POWERED["hiddenpower"]}', 'hiddenpower'
    else:
        info = dex.move_info(move);name, ident = (info.name if info else move), move
        if move in POWERED:name = f'{name} {POWERED[move]}'
    pp = info.max_pp if info else 0
    return {'move': name, 'id': ident, 'pp': pp, 'maxpp': pp, 'target': info.target if info else 'normal', 'disabled': bool(disabled)}


def world_request(request, first_row, variant, dex, stats=None):
    """Foul Play's request with its active mon's set replaced by `variant`. Returns (request, label map, replaced_disabled):
    label map[a] is the world's action index for true action a (None: the world lacks that move); replaced_disabled is True when
    a new (non-family) move went into a slot whose true entry is disabled. stats, when given, replaces the active row's battle
    stats (atk/def/spa/spd/spe)."""
    req = copy.deepcopy(request);rows = req['side']['pokemon']
    row = next(r for r in rows if r.get('active'))
    raw = list(row.get('moves') or ());true_moves = [move_key(m) for m in raw];want = [move_key(m) for m in variant.get('moves') or ()]
    extra = sorted(set(want) - set(true_moves));slots, kin = [], {}
    for m in true_moves:   # shared moves keep their slot; a different Hidden Power type takes the true Hidden Power's slot
        if m in want:slots.append(m);continue
        same = next((x for x in extra if family(x) == family(m) != x), None)
        if same is not None:extra.remove(same);kin[same] = m
        slots.append(same)
    for j, m in enumerate(slots):
        if m is None and extra:slots[j] = extra.pop(0)
    slots = [m for m in slots if m is not None] + extra
    raw_of = dict(zip(true_moves, raw));row['moves'] = [raw_of.get(m) or request_move_id(m) for m in slots]
    active = (req.get('active') or [None])[0]
    locked = not active or len(active.get('moves') or ()) != len(true_moves);replaced_disabled = False
    if not locked:
        by_move = dict(zip(true_moves, active['moves']));entries = []
        for j, m in enumerate(slots):
            if m in by_move:entries.append(by_move[m])
            elif m in kin:entries.append(dict(by_move[kin[m]], move=fresh_move_entry(dex, m)['move']))
            else:
                disabled = bool((active['moves'][j] if j < len(active['moves']) else {}).get('disabled'))   # a new, non-family move
                replaced_disabled |= disabled;entries.append(fresh_move_entry(dex, m, disabled))
        active['moves'] = entries
    if toid(row.get('item')) == toid(first_row.get('item')):row['item'] = toid(variant.get('item'))
    if stats is not None:row['stats'] = dict(row.get('stats') or {}, **stats)
    ability = toid(variant.get('ability'))
    if ability:
        if row.get('ability') and row.get('ability') == row.get('baseAbility'):row['ability'] = ability
        row['baseAbility'] = ability
    slot_of = {m: j for j, m in enumerate(slots)};slot_of.update({kin[m]: j for j, m in enumerate(slots) if m in kin})
    label_map = [a if locked else slot_of.get(true_moves[a]) if a < len(true_moves) else None for a in range(data_mod.MOVES)]
    return req, label_map + list(range(data_mod.MOVES, data_mod.ACTIONS)), replaced_disabled


def chosen_move(request, action):
    """The move id Foul Play chose (None for a switch)."""
    if action >= data_mod.MOVES:return None
    row = next(r for r in request['side']['pokemon'] if r.get('active'));moves = row.get('moves') or ()
    entries = (request.get('active') or [{}])[0].get('moves') or ()
    if len(entries) == len(moves):return move_key(moves[action])
    return move_key(entries[action].get('id')) if action < len(entries) else None


def public_choice(lines, side, action, move):
    """Whether Foul Play's choice is visible in one step's protocol lines."""
    base = family
    for line in lines:
        parts = line.split('|')
        if len(parts) < 4 or not parts[2].startswith(f'{side}a'):continue
        if action >= data_mod.MOVES and parts[1] in ('switch', 'drag'):return True
        if action < data_mod.MOVES and parts[1] == 'move' and '[from]' not in line and base(move_key(parts[3])) == base(move or ''):return True
    return False


def logsumexp(x):
    x = np.asarray(x, float);top = np.max(x) if len(x) else -np.inf
    return top if not np.isfinite(top) else top + np.log(np.exp(x - top).sum())


def set_posterior(keys, evidence, beta):
    """keys: each candidate variant's key (prior uniform over variants); evidence: key -> summed log q. Returns (distinct keys,
    log prior, log posterior) with posterior ∝ prior * exp(beta * L); falls back to the prior if no key is finite."""
    distinct = list(dict.fromkeys(keys));counts = collections.Counter(keys)
    log_prior = np.log(np.array([counts[k] for k in distinct], float) / len(keys))
    z = log_prior + beta * np.array([evidence.get(k, 0.) for k in distinct])
    if not np.isfinite(z).any():z = log_prior.copy()
    return distinct, log_prior, z - logsumexp(z)


@contextlib.contextmanager
def substituted_request(env, player, request):
    """env.observe(player) sees `request` in place of the player's request (the replay snapshot's requests[player])."""
    parser = env._parser;original = parser.snapshot

    def snapshot():
        state = original()
        return dataclasses.replace(state, requests={**state.requests, player: request})
    parser.snapshot = snapshot
    try:
        yield
    finally:
        del parser.snapshot


def same_observation(a, b):
    return all(np.array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k))) for k in (*OBS_DTYPES, 'legal_action_mask'))


def score(model, observations, labels, torch, redaction=None):
    """log q(label | observation) under π_FP per observation (-inf for a None or illegal label), on π_FP's own redaction."""
    arrays = {k: np.stack([np.asarray(getattr(o, k), t) for o in observations]) for k, t in OBS_DTYPES.items()}
    if redaction == 'peer-view-v1':train_mod.redact_peer_view(arrays['numeric_features'])
    elif redaction is not None:raise SystemExit(f'unknown redaction {redaction!r}')
    arrays['legal'] = np.stack([np.asarray(o.legal_action_mask, bool) for o in observations])
    arrays['label'] = np.array([0 if y is None else y for y in labels])
    rows = np.arange(len(observations))
    with torch.no_grad():
        logits = train_mod.forward(model, train_mod.batch(arrays, rows[:, None], rows, torch, 'cpu'))
        logq = torch.log_softmax(logits, -1).numpy()[rows, arrays['label']].astype(float)
    bad = np.array([y is None or not arrays['legal'][i, y] for i, y in enumerate(labels)])
    logq[bad] = -np.inf
    return logq


def replay_game(env, model, game, dex, torch, tables, redaction=None):
    """One game: (mon rows, decision rows, calibration rows, check counts)."""
    fp, ours_seat = game['fp'], game['ours'];env.reset(seed=game['seed']);checks = collections.Counter();misaligned = set()
    journal = dict(game['journal']);ours = 0;round_ = 0
    evidence = collections.defaultdict(dict);evidence_n = collections.Counter();mons, decisions, calib = [], [], []
    first = None
    while env.terminal() is None:
        requested = env.requested_players()
        if not requested:raise data_mod.Mismatch('no requested players before terminal')
        actions = {};pending = None
        if ours_seat in requested:env.observe(ours_seat)   # live order: our seat's trackers narrow beliefs before Foul Play's turn is read
        for player in requested:
            if player != fp:
                if ours >= len(game['our_actions']):raise data_mod.Mismatch(f'our seat requested beyond its {ours} recorded decisions')
                captured = game['our_requests'][ours]
                if captured is None or data_mod._request_body(captured) != data_mod._request_body(json.dumps(env._latest_requests.get(player))):
                    raise data_mod.Mismatch(f'our request differs from the capture at decision {ours}')
                actions[player] = int(game['our_actions'][ours]);ours += 1;continue
            entry = journal.pop(round_, None)
            if entry is None:raise data_mod.Mismatch(f'Foul Play requested in round {round_} without a journal entry')
            action = int(entry['action'])
            if data_mod.choice_action(env, player, entry.get('choice')) != action:
                raise data_mod.Mismatch(f'Foul Play choice {entry.get("choice")!r} does not decode to {action}, round {round_}')
            actions[player] = action
            request = env._latest_requests.get(fp)
            if first is None:first = env._first_requests.get(fp) or request
            if request.get('forceSwitch'):checks['forced_skipped'] += 1;continue
            pending = measure(env, model, game, dex, torch, checks, request, first, action, round_, evidence, evidence_n, mons, decisions, calib,
                              tables, misaligned, redaction)
        before = len(env._lines);env.step(actions)
        if pending is not None:
            mon, keys, logq, act, move, aligned = pending
            observed = public_choice(env._lines[before:], fp, act, move);decisions[-1]['observed'] = observed
            if observed:
                evidence_n[mon] += 1
                if not aligned:misaligned.add(mon)
                for k, lq in zip(keys, logq):evidence[mon][k] = evidence[mon].get(k, 0.) + lq
        round_ += 1
        if round_ > data_mod.MAX_ROUNDS:raise data_mod.Mismatch('runaway replay')
    if journal:raise data_mod.Mismatch(f'{len(journal)} journal entries never requested')
    terminal = env.terminal()
    if (terminal.winner or None) != (game['winner'] or None):raise data_mod.Mismatch(f'winner {terminal.winner} != {game["winner"]}')
    return mons, decisions, calib, checks


def measure(env, model, game, dex, torch, checks, request, first, action, round_, evidence, evidence_n, mons, decisions, calib,
            tables, misaligned, redaction=None):
    fp = game['fp'];turn = int(env._latest_turn)
    baseline = env.observe(fp)   # also syncs the belief engine with this round's protocol lines
    truth = {species_of(r): (r.get('ident'), request_key(r)) for r in first['side']['pokemon']}
    view = env._belief_engine.resolved_player_view(game['ours'])
    active_row = next(r for r in request['side']['pokemon'] if r.get('active'))
    active = species_of(active_row);first_row = next((r for r in first['side']['pokemon'] if r.get('ident') == active_row.get('ident')), None)
    # every seen Foul Play mon: prior vs posterior on its true set, before this decision's evidence
    for belief in view.opponent_pokemon:
        mon = toid(belief.species)
        if mon not in truth or belief.transformed or not belief.candidate_variants:continue
        keys = [variant_key(v) for v in belief.candidate_variants];true_key = truth[mon][1]
        row = dict(seed=game['seed'], ordinal=game['ordinal'], turn=turn, round=round_, active=mon == active, variants=len(keys),
                   distinct=len(set(keys)), hidden_power_only=len(set(keys)) > 1 and len({family_key(k) for k in keys}) == 1,
                   attack_dump_varies=len({attack_dumped(list(k[0]), dex, tables) for k in set(keys)}) > 1,
                   stats_aligned=mon not in misaligned, truth_in=true_key in keys, evidence=evidence_n[mon],
                   incomplete=bool(evidence_n[mon]) and any(k not in evidence[mon] for k in set(keys)))
        for beta in BETAS:
            distinct, log_prior, log_post = set_posterior(keys, evidence[mon], beta)
            hit = [i for i, k in enumerate(distinct) if k == true_key]
            row[f'log_post_{beta}'] = float(logsumexp(log_post[hit])) if hit else -np.inf
            if beta == 1.:
                row['log_prior'] = float(logsumexp(log_prior[hit])) if hit else -np.inf
                if len(distinct) > 1 and row['truth_in']:
                    calib.extend(dict(seed=game['seed'], evidence=evidence_n[mon], p_prior=float(np.exp(lp)), p_post=float(np.exp(lq)),
                                      is_true=k == true_key) for k, lp, lq in zip(distinct, log_prior, log_post))
        mons.append(row)
    # the active mon's worlds: q(a_t | v) per distinct candidate set
    belief = next((b for b in view.opponent_pokemon if toid(b.species) == active), None)
    with substituted_request(env, fp, copy.deepcopy(request)):identity = env.observe(fp)
    if not same_observation(identity, baseline):checks['identity_world_differs'] += 1
    if first_row is None or belief is None or belief.transformed or not belief.candidate_variants \
            or request_key(active_row)[0] != request_key(first_row)[0]:
        checks['active_unmeasured'] += 1;return None
    for entry, m in zip((request.get('active') or [{}])[0].get('moves') or (), active_row.get('moves') or ()):
        if entry.get('pp') == entry.get('maxpp') and len(request['active'][0]['moves']) == len(active_row['moves']):
            checks['fresh_entries'] += 1;checks['fresh_entry_differs'] += fresh_move_entry(dex, move_key(m), entry.get('disabled')) != entry
    keys = [variant_key(v) for v in belief.candidate_variants];distinct = list(dict.fromkeys(keys))
    representative = {variant_key(v): v for v in belief.candidate_variants};true_key = truth[active][1]
    species = dex.species_info(active);level = level_of(active_row)
    true_stats = generator_stats(species.base_stats, level, [move_key(m) for m in first_row.get('moves') or ()], dex, tables) if species else None
    aligned = true_stats is not None and all(true_stats[k] == (active_row.get('stats') or {}).get(k) for k in STATS)
    checks['stats_decisions'] += 1;checks['stats_formula_mismatch'] += not aligned
    observations, labels, replaced_disabled = [], [], False
    for k in distinct:
        stats = generator_stats(species.base_stats, level, list(k[0]), dex, tables) if aligned else None
        world, label_map, disabled = world_request(request, first_row, representative[k], dex, stats);replaced_disabled |= disabled
        with substituted_request(env, fp, world):observations.append(env.observe(fp))
        labels.append(label_map[action])
        if k == true_key:checks['true_world_differs'] += not same_observation(observations[-1], baseline)
    if not same_observation(env.observe(fp), baseline):checks['observe_changed_by_worlds'] += 1
    if replaced_disabled:checks['replaced_disabled_slot_skipped'] += 1;return None
    logq = score(model, observations + [baseline], labels + [action], torch, redaction);true_logq = logq[-1];logq = logq[:-1]
    by_key = dict(zip(distinct, logq));checks['worlds'] += len(distinct)
    row = dict(seed=game['seed'], ordinal=game['ordinal'], turn=turn, round=round_, worlds=len(distinct), truth_in=true_key in by_key,
               evidence=evidence_n[active], incomplete=bool(evidence_n[active]) and any(k not in evidence[active] for k in distinct),
               stats_aligned=aligned and active not in misaligned,
               nll_true=float(-true_logq), observed=False)
    for beta in BETAS:
        d, log_prior, log_post = set_posterior(keys, evidence[active], beta)
        q = np.array([by_key[k] for k in d])
        if beta == 1.:row['nll_prior'] = float(-logsumexp(log_prior + q))
        row[f'nll_post_{beta}'] = float(-logsumexp(log_post + q))
    decisions.append(row)
    return active, distinct, logq, action, chosen_move(request, action), aligned


def run(args):
    import torch
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
    from pokezero.neural_policy import (category_vocab_from_model_config, checkpoint_file_sha256, feature_masks_from_model_config,
                                        load_transformer_checkpoint, observation_spec_from_model_config)
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False);torch.set_float32_matmul_precision('highest')
    collection, out = Path(args.collection), Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    paths = sorted(glob.glob(str(collection / 'results' / '*.json')), key=lambda p: int(Path(p).stem))
    paths = [p for p in paths if int(Path(p).stem) % args.shards == args.shard]
    marker = torch.load(args.pi_fp, map_location='cpu', weights_only=True).get('fp_reply') or {}
    pi_fp_data = args.pi_fp_data or marker.get('data')
    if not pi_fp_data:raise SystemExit("π_FP's training dataset is unknown: pass --pi-fp-data (the checkpoint has no fp_reply.data)")
    data_collection = probe_mod.dataset_collection(pi_fp_data)
    if Path(data_collection).resolve() != collection.resolve():
        raise SystemExit(f'{pi_fp_data} was built from {data_collection}, not {collection}: its ordinals mean other games')
    seen = probe_mod.seen_ordinals(pi_fp_data);redaction = marker.get('redaction')
    model, result = load_transformer_checkpoint(args.pi_fp, map_location='cpu');model.eval()
    if int(result.model_config.window_size) != 1:raise SystemExit(f'π_FP has window {result.model_config.window_size}; this test feeds window 1')
    checks = collections.Counter();mismatches = collections.Counter();examples = {};env = None;config_key = None;dex = None;tables = None
    rows = dict(mons=[], decisions=[], calibration=[]);games = 0
    for path in paths:
        if args.limit and games >= args.limit:break
        if int(Path(path).stem) in seen:checks['skipped_seen_by_pi_fp'] += 1;continue
        game, skip = data_mod.load_game(collection, path)
        if game is None:checks[f'skipped_{skip}'] += 1;continue
        base = game['base_config'];key = (base['showdown_root'], base['node_binary'], bool(base['belief_set_source']))
        if not base['belief_set_source']:checks['skipped_no_belief_source'] += 1;continue
        if config_key is None:config_key = key
        elif key != config_key:checks['skipped_other_runtime'] += 1;continue
        if dex is None:dex = load_showdown_dex_cached(base['showdown_root']);tables = generator_tables(base['showdown_root'])
        if env is None:
            env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=base['showdown_root'], node_binary=base['node_binary'],
                observation_spec=observation_spec_from_model_config(result.model_config),
                category_vocab=category_vocab_from_model_config(result.model_config, base['showdown_root']),
                feature_masks=feature_masks_from_model_config(result.model_config), set_belief_source=True))
        try:mons, decisions, calib, game_checks = replay_game(env, model, game, dex, torch, tables, redaction)
        except data_mod.Mismatch as error:
            reason = re.sub(r"'[^']*'|\"[^\"]*\"|\d+", '#', str(error));mismatches[reason] += 1
            examples.setdefault(reason, dict(ordinal=game['ordinal'], detail=str(error)));continue
        except (LocalShowdownError, ValueError, KeyError, StopIteration) as error:
            reason = f'env error {type(error).__name__}';mismatches[reason] += 1
            examples.setdefault(reason, dict(ordinal=game['ordinal'], detail=str(error)[:500]));env.close();env = None;continue
        games += 1;checks.update(game_checks);rows['mons'] += mons;rows['decisions'] += decisions;rows['calibration'] += calib
        if games % 10 == 0:
            print(json.dumps(dict(event='PROGRESS', shard=args.shard, games=games, decisions=len(rows['decisions']),
                                  worlds=checks['worlds'], seconds=round(time.time() - started, 1))), flush=True)
    if env is not None:env.close()
    for name, table in rows.items():
        np.savez(out / f'{name}-{args.shard}.npz', **({k: np.array([r[k] for r in table]) for k in table[0]} if table else {}))
    summary = dict(schema='v2s-fp-reply-reweight.v2', collection=str(collection), pi_fp=str(args.pi_fp),
                   pi_fp_sha256=checkpoint_file_sha256(args.pi_fp), pi_fp_data=str(pi_fp_data), pi_fp_data_collection=data_collection,
                   pi_fp_redaction=redaction, shard=args.shard, shards=args.shards, games=games,
                   decisions=len(rows['decisions']), mon_rows=len(rows['mons']), checks=dict(checks), mismatched=sum(mismatches.values()),
                   mismatch_reasons=dict(mismatches), mismatch_examples=examples, seconds=round(time.time() - started, 1))
    (out / f'shard-{args.shard}.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(dict(event='SHARD_DONE', **{k: summary[k] for k in ('shard', 'games', 'decisions', 'mismatched', 'checks', 'seconds')})), flush=True)


def _load(out, name):
    parts = [dict(np.load(p)) for p in sorted(glob.glob(str(Path(out) / f'{name}-*.npz')))]
    parts = [p for p in parts if p]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}


def summarize(mons, decisions, calib):
    """Gate 5 on one set of rows (all clustered on battle seed)."""
    out = dict(mon_rows=int(len(mons['seed'])), decisions=int(len(decisions['seed'])))
    multi = mons['distinct'] > 1
    out['coverage'] = dict(truth_in_candidates=float(mons['truth_in'].mean()), truth_in_when_ambiguous=float(mons['truth_in'][multi].mean()),
                           ambiguous_share=float(multi.mean()), mean_distinct_when_ambiguous=float(mons['distinct'][multi].mean()),
                           incomplete_evidence_share=float(mons['incomplete'][multi].mean()))
    measured = multi & mons['truth_in'] & (mons['evidence'] > 0) & ~mons['incomplete'];every = np.ones(int(measured.sum()), bool)
    out['measured_rows'] = int(measured.sum());out['measured_seeds'] = int(len(np.unique(mons['seed'][measured])))
    out['excluded_incomplete_rows'] = int((multi & mons['truth_in'] & (mons['evidence'] > 0) & mons['incomplete']).sum())
    out['delta_log_p_true'] = {}
    if not measured.any():out['gate5'] = None;return out
    for beta in BETAS:
        delta = mons[f'log_post_{beta}'][measured] - mons['log_prior'][measured]
        out['delta_log_p_true'][str(beta)] = dict(**probe_mod.cluster_bootstrap(delta, mons['seed'][measured], [every], lambda a: a),
                                                  improved_share=float((delta > 1e-9).mean()), worsened_share=float((delta < -1e-9).mean()),
                                                  mean_p_prior=float(np.exp(mons['log_prior'][measured]).mean()),
                                                  mean_p_post=float(np.exp(mons[f'log_post_{beta}'][measured]).mean()))
    by_evidence = {}
    for lo, hi in ((1, 1), (2, 3), (4, 7), (8, 10**9)):
        m = measured & (mons['evidence'] >= lo) & (mons['evidence'] <= hi)
        if m.any():by_evidence[f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'] = dict(
            rows=int(m.sum()), delta=float((mons['log_post_1.0'][m] - mons['log_prior'][m]).mean()))
    out['delta_by_evidence_count'] = by_evidence
    for name, mask in (('candidates_differ_only_in_hidden_power_type', mons['hidden_power_only']),
                       ('candidates_differ_beyond_hidden_power_type', ~mons['hidden_power_only']),
                       ('attack_dump_varies', mons['attack_dump_varies']), ('attack_dump_constant', ~mons['attack_dump_varies']),
                       ('stats_aligned', mons['stats_aligned']), ('stats_not_aligned', ~mons['stats_aligned'])):
        m = measured & mask
        out[f'delta_{name}'] = dict(rows=int(m.sum()), **probe_mod.cluster_bootstrap(
            mons['log_post_1.0'][m] - mons['log_prior'][m], mons['seed'][m], [np.ones(int(m.sum()), bool)], lambda a: a))
    measured_active = measured & mons['active'];out['delta_active_mon_only'] = probe_mod.cluster_bootstrap(
        mons['log_post_1.0'][measured_active] - mons['log_prior'][measured_active], mons['seed'][measured_active],
        [np.ones(int(measured_active.sum()), bool)], lambda a: a)
    if calib:
        out['set_calibration'] = {}
        for name in ('p_prior', 'p_post'):
            p = calib[name];y = calib['is_true'].astype(float)
            out['set_calibration'][name] = dict(ece=data_mod.calibration_error(p, y.astype(bool)), brier=float(((p - y) ** 2).mean()),
                                                mean_p=float(p.mean()), true_share=float(y.mean()))
    d = decisions;finite = np.isfinite(d['nll_prior']) & np.isfinite(d['nll_post_1.0']) & np.isfinite(d['nll_true'])
    has = (d['evidence'] > 0) & ~d['incomplete'] & finite;one = np.ones(int(has.sum()), bool)
    pred = dict(rows=int(has.sum()), excluded_infinite_rows=int((~finite).sum()), excluded_incomplete_rows=int((d['incomplete'] & finite).sum()),
                nll_true_view=float(d['nll_true'][has].mean()), nll_prior_mixture=float(d['nll_prior'][has].mean()))
    for beta in BETAS:pred[f'nll_post_mixture_{beta}'] = float(d[f'nll_post_{beta}'][has].mean())
    pred['post_minus_prior'] = probe_mod.cluster_bootstrap(d['nll_post_1.0'][has] - d['nll_prior'][has], d['seed'][has], [one], lambda a: a)
    pred['post_minus_true_view'] = probe_mod.cluster_bootstrap(d['nll_post_1.0'][has] - d['nll_true'][has], d['seed'][has], [one], lambda a: a)
    aligned = has & d['stats_aligned']
    pred['post_minus_prior_stats_aligned'] = dict(rows=int(aligned.sum()), **probe_mod.cluster_bootstrap(
        d['nll_post_1.0'][aligned] - d['nll_prior'][aligned], d['seed'][aligned], [np.ones(int(aligned.sum()), bool)], lambda a: a))
    pred['all_decisions'] = dict(rows=int(finite.sum()), nll_true_view=float(d['nll_true'][finite].mean()),
                                 nll_prior_mixture=float(d['nll_prior'][finite].mean()), nll_post_mixture=float(d['nll_post_1.0'][finite].mean()))
    out['predictive'] = pred
    ci = out['delta_log_p_true']['1.0']['interval_90'];prior = pred['post_minus_prior']['interval_90']
    oracle = pred['post_minus_true_view']['interval_90']
    # "not worse" is the design note's convention (the interval includes 0 or is negative); "better" needs the whole interval below 0
    out['gate5'] = dict(part1_delta_positive=bool(ci and ci[0] > 0),
                        part2_amended_not_worse_than_prior_mixture=bool(prior and prior[0] <= 0),
                        part2_amended_better_than_prior_mixture=bool(prior and prior[1] < 0),
                        part2_original_not_worse_than_true_view=bool(oracle and oracle[0] <= 0))
    return out


def report(out):
    mons, decisions, calib = _load(out, 'mons'), _load(out, 'decisions'), _load(out, 'calibration')
    shards = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(Path(out) / 'shard-*.json')))]
    if not shards or any(not s.get('pi_fp_data') for s in shards) or len({s['pi_fp_sha256'] for s in shards}) != 1:
        raise SystemExit('every shard must come from one π_FP with its training dataset excluded (schema v2)')
    checks = collections.Counter()
    for s in shards:checks.update(s['checks'])
    result = dict(schema='v2s-fp-reply-reweight-report.v1', shards=len(shards), games=sum(s['games'] for s in shards),
                  mismatched=sum(s['mismatched'] for s in shards), checks=dict(checks), betas=list(BETAS),
                  pi_fp_sha256=shards[0]['pi_fp_sha256'], pi_fp_data=shards[0]['pi_fp_data'], pi_fp_redaction=shards[0].get('pi_fp_redaction'))
    pick = lambda table, mask: {k: v[mask] for k, v in table.items()}
    split_m, split_d = data_mod.split_of(mons['seed']), data_mod.split_of(decisions['seed'])
    split_c = data_mod.split_of(calib['seed']) if calib else None
    result['unseen_by_pi_fp'] = summarize(mons, decisions, calib)
    result['test_only'] = summarize(pick(mons, split_m == 'test'), pick(decisions, split_d == 'test'),
                                    pick(calib, split_c == 'test') if calib else {})
    (Path(out) / 'report.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(event='REWEIGHT_REPORT', games=result['games'], checks=result['checks'],
                          gate5=result['unseen_by_pi_fp'].get('gate5'), delta=result['unseen_by_pi_fp']['delta_log_p_true'].get('1.0'),
                          predictive=(result['unseen_by_pi_fp'].get('predictive') or {}).get('post_minus_prior'))), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('collection', nargs='?');parser.add_argument('out')
    parser.add_argument('--pi-fp');parser.add_argument('--pi-fp-data', help="π_FP's training dataset (default: the checkpoint's fp_reply.data); "
                                                                     "games in its train split are skipped")
    parser.add_argument('--shard', type=int, default=0);parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0);parser.add_argument('--report', action='store_true')
    args = parser.parse_args(argv)
    if args.report:report(args.out);return 0
    if not args.collection or not args.pi_fp:parser.error('COLLECTION and --pi-fp are required unless --report')
    run(args);return 0


if __name__ == '__main__':
    raise SystemExit(main())
