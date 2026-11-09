"""Foul Play reply dataset: replay each collected game through the training env and record Foul Play's own view at its decisions.

usage: python -m gpu_engine.fp_reply_data COLLECTION OUT_DIR --checkpoint PT [--shard I --shards N] [--limit GAMES]
       python -m gpu_engine.fp_reply_data --report OUT_DIR [--split all|test]

Input, read-only: a Foul Play choice collection (the engine session's fp_collect.py, RAW champion vs Foul Play with
opponent_journal 'full'):
  results/<ordinal>.json     source_game.opponent_moves, Foul Play's journal (round, seat, choice, action), and the winner;
  telemetry/<ordinal>.jsonl  our seat's played_action_index and greedy action per decision_index;
  owning-wire/<ordinal>.jsonl our seat's ROOT own_request_line per decision_index;
  requests/<ordinal>.json    the cell's base_config (Showdown root, node binary, belief set source).
Replay: LocalShowdownEnv.reset(seed) uses the bridge's own seeding (showdown_seed_from_int), then each round steps our seat's
recorded action and Foul Play's journal action for that round. Showdown is deterministic given the seed and both choices, so
Foul Play's player view is rebuilt exactly, without an input log. The env config latches spec, vocabulary and masks from the
checkpoint, as the bridge does.
Checks per game (a failing game is dropped and counted, never patched):
  - each of our seat's requests equals the captured own_request_line (rqid and the side's username ignored);
  - Foul Play has a journal entry exactly in the rounds it is requested, its action is legal in its own view, and its journaled
    choice string decodes to that action on its own replayed state (teacher_capture.action_index_from_choice_string), which
    constrains Foul Play's side of the replay directly;
  - our seat's decisions are exactly the telemetry's (and the result's pokezero_decisions), and the terminal winner is the result's.
An env error in one game (LocalShowdownError, ValueError, KeyError) drops that game, counted by type, and the env is rebuilt;
MAX_CONSECUTIVE_ENV_ERRORS in a row abort the shard.
Reported, not a drop: encoder parity, the champion's argmax on our seat's env observation against the recorded greedy action.
Games whose battle seed is in the screen band (2103400000 to 2103400999) are skipped: RR scores operators on those roots.
Output per shard: shard-<I>.npz, one row per Foul Play decision (the observation arrays, legal mask, label, the champion's
probabilities and value on Foul Play's view, round, turn, request kind, game ordinal and seed, and fp_opp_hazard_credit: the
OPP_HAZARD_CREDIT Foul Play itself sees, fp_view_opp_hazard_credit, which peer-view-v2 writes over the env's exact value), and
shard-<I>.json (counts; spikes_unknown_before counts rows where one of our Spikes hits had no previous HP on the lines, credit 0;
hazard_credit_rows_differ and hazard_credit_max_abs_gap_vs_env compare fp_opp_hazard_credit with the env's own column).
"""
import argparse
import collections
import glob
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

SCREEN_BAND = (2103400000, 2103401000)
MOVES = 4
ACTIONS = 9
TURN_BANDS = ((1, 3), (4, 9), (10, 19), (20, 10**9))
MAX_ROUNDS = 400
MAX_CONSECUTIVE_ENV_ERRORS = 5   # a systematic config or encoder bug fails the shard loudly, not game by game


class Mismatch(Exception):
    pass


def band(turn):
    for low, high in TURN_BANDS:
        if low <= turn <= high:return f'{low}-{high}' if high < 10**9 else f'{low}+'
    return '0'


def _request_body(line):
    body = json.loads(line[len('|request|'):] if line.startswith('|request|') else line)
    body = dict(body);body.pop('rqid', None)
    if isinstance(body.get('side'), dict):body['side'] = {k: v for k, v in body['side'].items() if k != 'name'}   # usernames differ
    return body


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_game(collection, result_path):
    """Everything the replay needs for one collected game, or None when the game isn't usable (counted by the caller)."""
    result = json.loads(Path(result_path).read_bytes())
    if result.get('status') != 'COMPLETE':return None, 'not_complete'
    game = result.get('source_game') or {}
    journal = game.get('opponent_moves') or []
    if not journal:return None, 'no_journal'
    ordinal = int(Path(result_path).stem);cell = result.get('cell') or {}
    bench = result.get('benchmark') or {}
    fp = bench.get('foulplay_player');ours = {'p1': 'p2', 'p2': 'p1'}.get(fp)
    if ours is None:return None, 'unknown_seats'
    seed = int(game['seed'])
    if SCREEN_BAND[0] <= seed < SCREEN_BAND[1]:return None, 'screen_band'
    telemetry = sorted((r for r in _rows(collection / 'telemetry' / f'{ordinal}.jsonl') if r.get('owner') == ours),
                       key=lambda r: r['decision_index'])
    roots = {r['decision_index']: r['own_request_line'] for r in _rows(collection / 'owning-wire' / f'{ordinal}.jsonl')
             if r.get('event') == 'ROOT'}
    if [r['decision_index'] for r in telemetry] != list(range(len(telemetry))):return None, 'telemetry_gap'
    by_round = {}
    for entry in journal:
        if entry.get('seat') != fp or entry['round'] in by_round:return None, 'journal_shape'
        by_round[entry['round']] = entry
    winner = (game.get('terminal_witness') or {}).get('winner_player')
    request = json.loads((collection / 'requests' / f'{ordinal}.json').read_bytes())
    return dict(ordinal=ordinal, seed=seed, fp_seed=cell.get('fp_seed'), seat=cell.get('seat'), fp=fp, ours=ours,
                journal=by_round, our_actions=[r['played_action_index'] for r in telemetry],
                our_greedy=[(r.get('greedy') or {}).get('action_index') for r in telemetry],
                our_requests=[roots.get(i) for i in range(len(telemetry))], winner=winner,
                decisions=game.get('pokezero_decisions'), base_config=request['base_config'],
                checkpoint_sha256=request.get('checkpoint_sha256')), None


def choice_action(env, player, choice):
    """Foul Play's journaled /choose body decoded to our action index on its own replayed state (None if undecodable)."""
    from pokezero.teacher_capture import action_index_from_choice_string
    return action_index_from_choice_string(env._state_for_player(player), choice or '')


DAMAGING, STATUS, SWITCH, ILLEGAL = 0, 1, 2, -1


def hp_fraction(request):
    """The active mon's HP fraction from a request's side ('cur/max status'; 'fnt' or missing is 0)."""
    for mon in (request.get('side') or {}).get('pokemon') or ():
        if mon.get('active'):
            head = str(mon.get('condition') or '0').split()[0]
            if '/' in head:
                cur, top = head.split('/', 1)
                return float(cur) / max(1., float(top))
            return 0.
    return float('nan')


# Foul Play's view of our Spikes losses (peer-view-v2's OPP_HAZARD_CREDIT) ------------------------------------------------------------
# The env's OPP_HAZARD_CREDIT in Foul Play's view sums our exact HP fractions lost to Spikes (showdown.py's hazard_damage_suffered,
# read off the omniscient stream). Foul Play sees our HP only as Showdown's public percent, and the search's peer view (the
# constructor's fill_owner_spikes, the device's engine_device_public_port.spikes) sums each hit's public-percent drop / 100 in float32
# with a compensated (TwoSum) sum. fp_view_opp_hazard_credit is that rule on the env's own lines, written as the feature value,
# with the device's jitted float32 arithmetic (the GPU-built peer views, e.g. the 48-cell capture). The live root's
# fill_owner_spikes runs its per-hit / 100 eagerly on the CPU, an exactly rounded division: on about 1% of rows that is one
# float32 ulp (~1e-9) away from this value, immaterial for the reply model.
SWITCH_KINDS = ('switch', 'drag', 'replace')
HAZARD_CREDIT_DIVISOR = 6.   # engine_device_features.FIELD_FIELDS['opp_hazard_credit'] and showdown's / _TEAM_SIZE
OPP_HAZARD_CREDIT_COLUMN = 116   # field token 0's numeric column: showdown.v4_numeric_index(NUMERIC_OPP_HAZARD_CREDIT)


def shown_percent(hp, maximum):
    """Showdown's public HP percent (sim/pokemon.ts getHealth with HP percentages): ceil(100 hp / max), 99 for a damaged mon that
    would show 100, 0 when fainted; integer arithmetic, as engine_device_trackers._percent."""
    top = max(int(maximum), 1);hp = min(max(int(hp), 0), top)
    shown = (100 * hp + top - 1) // top
    return 99 if hp < top and shown == 100 else shown


def _condition(text):
    """(hp, max HP or None) of a protocol condition ('263/300', '263/300 par', '0 fnt'); None when unreadable."""
    head = (str(text).split() or [''])[0]
    if head == '0':return 0, None
    hp, sep, top = head.partition('/')
    try:return (int(hp), int(top)) if sep else None
    except ValueError:return None


def spikes_hits(lines, seat):
    """seat's ('p1' / 'p2') Spikes hits in protocol order, each (hp before, hp after, max HP), from exact-HP lines (the env's
    omniscient stream, or the seat's own); the previous HP is the mon's last |switch| / |drag| / |replace| / |-damage| / |-heal| /
    |-sethp| condition. None when a hit's previous HP or max HP is not on the lines (the search side then leaves the credit 0)."""
    hp = {};hits = []
    for line in lines:
        parts = str(line).split('|')
        if len(parts) < 4 or parts[0] != '':continue
        kind = parts[1];side, _, name = parts[2].partition(':')
        if side.strip()[:2] != seat:continue
        name = name.strip()
        if kind in SWITCH_KINDS:
            if len(parts) > 4:
                condition = _condition(parts[4])
                if condition is None:hp.pop(name, None)
                else:hp[name] = condition
        elif kind in ('-damage', '-heal', '-sethp'):
            condition = _condition(parts[3])
            if condition is None:hp.pop(name, None);continue
            new, maximum = condition;before = hp.get(name)
            if maximum is None:maximum = before[1] if before else None
            if kind == '-damage' and any(x.strip().lower() == '[from] spikes' for x in parts[4:]):
                if before is None or maximum is None:return None
                hits.append((before[0], new, maximum))
            hp[name] = (new, maximum)
    return hits


def two_sum_add(high, low, value):
    """engine_device_damage_history._add on one float32 (high, low) pair: TwoSum, each operation rounded to float32; value <= 0 is a
    no-op."""
    f = np.float32;high, low, value = f(high), f(low), f(value)
    if not value > 0:return high, low
    total = f(high + value);recovered = f(total - high)
    error = f(f(high - f(total - recovered)) + f(value - recovered))
    correction = f(low + error);new_high = f(total + correction)
    return new_high, f(correction - f(new_high - total))


def fp_view_opp_hazard_credit(lines, victim):
    """Foul Play's OPP_HAZARD_CREDIT feature value (v4 field token 0, numeric column 116) for victim = our seat ('p1' / 'p2'):
    clip((high + low) * float32(1/6), 0, 1) in float32 of the TwoSum over our Spikes hits of
    (shown_percent(before) - shown_percent(after)) * float32(1/100); 0 when a hit's previous HP is unknown, as the search side
    leaves it. Both divisions are the float32 reciprocal multiplications XLA compiles the device's jitted `/ 100` and `/ 6.`
    into (10 * float32(0.01) rounds to 0.099999994, not 0.1): a correctly rounded division is one ulp off on 21 of the 48-cell
    check's 1,622 rows. The live root's fill_owner_spikes divides the per-hit drop eagerly on the CPU (exactly rounded), so
    there this value can be one ulp off (module comment above)."""
    hits = spikes_hits(lines, victim);high = low = np.float32(0)
    for before, after, maximum in hits or ():
        drop = max(0, shown_percent(before, maximum) - shown_percent(after, maximum))
        high, low = two_sum_add(high, low, np.float32(np.float32(drop) * np.float32(1. / 100)))
    value = np.float32(np.float32(high + low) * np.float32(1. / HAZARD_CREDIT_DIVISOR))
    return np.float32(min(max(value, np.float32(0)), np.float32(1)))


def action_classes(request, legal, dex):
    """Per action index: DAMAGING or STATUS for the active's moves (by dex category), SWITCH for switches, ILLEGAL otherwise."""
    moves = [m.get('id') for m in ((request.get('active') or [{}])[0].get('moves') or ())]
    out = []
    for a in range(ACTIONS):
        if not legal[a]:out.append(ILLEGAL)
        elif a >= MOVES:out.append(SWITCH)
        else:
            info = dex.move_info(moves[a]) if a < len(moves) and moves[a] else None
            out.append(STATUS if info is not None and str(info.category).lower() == 'status' else DAMAGING)
    return out


def opponent_head_logits(model, observation, torch):
    """9375's opponent_action_head on one observation (our seat's view): logits over the opponent's own action indices."""
    from pokezero.neural_policy import observation_window_to_torch
    tensors = observation_window_to_torch([observation], window_size=1, device='cpu')
    with torch.no_grad():
        output = model(categorical_ids=tensors['categorical_ids'], numeric_features=tensors['numeric_features'],
                       token_type_ids=tensors['token_type_ids'], attention_mask=tensors['attention_mask'], history_mask=tensors['history_mask'])
    return [float(x) for x in output.opponent_action_logits[0]]


def replay(env, policy, game, rng, extras=None):
    """Replay one game; returns (Foul Play rows, our-seat parity hits, our-seat decisions). Raises Mismatch on any check.
    extras (dict(dex, model, torch)) adds per Foul Play row: its action classes and label class, both actives' HP fractions, and
    9375's opponent_action_head logits from our seat's view in the same round (NaN when our seat isn't requested)."""
    env.reset(seed=game['seed']);policy.reset()
    journal = dict(game['journal']);rows = [];ours = 0;parity = 0;round_ = 0
    while env.terminal() is None:
        requested = env.requested_players()
        if not requested:raise Mismatch('no requested players before terminal')
        actions = {};fp_row = None;our_logits = None
        for player in requested:
            observation = env.observe(player)
            decision = policy.select_action(observation, rng=rng)
            probs = decision.metadata['policy_distribution']
            if player == game['fp']:
                entry = journal.pop(round_, None)
                if entry is None:raise Mismatch(f'Foul Play requested in round {round_} without a journal entry')
                action = int(entry['action'])
                if not 0 <= action < ACTIONS or not observation.legal_action_mask[action]:
                    raise Mismatch(f'Foul Play action {action} illegal in its own view, round {round_}')
                decoded = choice_action(env, player, entry.get('choice'))
                if decoded != action:
                    raise Mismatch(f'Foul Play choice {entry.get("choice")!r} decodes to {decoded}, not its action {action}, round {round_}')
                request = env._latest_requests.get(player) or {}
                lines = env.protocol_lines
                fp_row = dict(observation=observation, label=action, probs=probs, value=decision.value_estimate,
                              round=round_, turn=int(env._latest_turn), forced=bool(request.get('forceSwitch')), choice=entry.get('choice'),
                              fp_opp_hazard_credit=fp_view_opp_hazard_credit(lines, game['ours']),
                              spikes_unknown_before=spikes_hits(lines, game['ours']) is None)
                if extras is not None:
                    classes = action_classes(request, observation.legal_action_mask, extras['dex'])
                    fp_row.update(candidate_class=classes, label_class=classes[action], own_hp=hp_fraction(request),
                                  opp_hp=hp_fraction(env._latest_requests.get(game['ours']) or {}))
                rows.append(fp_row)
                actions[player] = action
            else:
                if ours >= len(game['our_actions']):raise Mismatch(f'our seat requested beyond its {ours} recorded decisions')
                captured = game['our_requests'][ours]
                if captured is None or _request_body(captured) != _request_body(json.dumps(env._latest_requests.get(player))):
                    raise Mismatch(f'our request differs from the capture at decision {ours}')
                parity += decision.action_index == game['our_greedy'][ours]
                if extras is not None:our_logits = opponent_head_logits(extras['model'], observation, extras['torch'])
                actions[player] = int(game['our_actions'][ours]);ours += 1
        if extras is not None and fp_row is not None:fp_row['opp_head_logits'] = our_logits if our_logits is not None else [float('nan')] * ACTIONS
        env.step(actions);round_ += 1
        if round_ > MAX_ROUNDS:raise Mismatch('runaway replay')
    if journal:raise Mismatch(f'{len(journal)} journal entries never requested, first round {min(journal)}')
    if ours != len(game['our_actions']):raise Mismatch(f'our seat made {ours} of {len(game["our_actions"])} decisions')
    if game.get('decisions') is not None and game['decisions'] != len(game['our_actions']):
        raise Mismatch(f'telemetry has {len(game["our_actions"])} decisions, the result {game["decisions"]}')
    terminal = env.terminal()
    if (terminal.winner or None) != (game['winner'] or None):raise Mismatch(f'winner {terminal.winner} != {game["winner"]}')
    return rows, parity, ours


def build(args):
    import torch
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
    from pokezero.neural_policy import (TransformerSoftmaxPolicy, category_vocab_from_model_config, checkpoint_file_sha256,
                                        feature_masks_from_model_config, load_transformer_checkpoint,
                                        observation_spec_from_model_config)
    torch.set_num_threads(1)
    collection, out = Path(args.collection), Path(args.out);out.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(collection / 'results' / '*.json')), key=lambda p: int(Path(p).stem))
    paths = [p for p in paths if int(Path(p).stem) % args.shards == args.shard]
    if args.limit:paths = paths[:args.limit]
    sha = checkpoint_file_sha256(args.checkpoint)
    model, result = load_transformer_checkpoint(args.checkpoint, map_location='cpu');model.eval()
    policy = TransformerSoftmaxPolicy(model=model, result=result, deterministic=True, device='cpu',
                                      record_policy_distribution=True, checkpoint_path=str(args.checkpoint), weights_sha256=sha)
    extras = None
    if args.extras:
        from pokezero.dex import load_showdown_dex_cached
        extras = dict(model=model, torch=torch, dex=None)
    env = None;config_key = None;rng = random.Random(0)
    counts = collections.Counter();mismatches = collections.Counter();examples = {}
    columns = collections.defaultdict(list);parity_hits = 0;parity_total = 0;started = time.time()
    seats = collections.Counter();consecutive_errors = 0;hazard_gap = 0.
    for path in paths:
        game, skip = load_game(collection, path)
        if game is None:counts[f'skipped_{skip}'] += 1;continue
        if game['checkpoint_sha256'] != sha:counts['skipped_other_checkpoint'] += 1;continue
        base = game['base_config'];key = (base['showdown_root'], base['node_binary'], bool(base['belief_set_source']))
        if config_key is None:config_key = key
        elif key != config_key:counts['skipped_other_runtime'] += 1;continue
        if extras is not None and extras['dex'] is None:extras['dex'] = load_showdown_dex_cached(base['showdown_root'])
        if env is None:
            env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=base['showdown_root'], node_binary=base['node_binary'],
                observation_spec=observation_spec_from_model_config(result.model_config),
                category_vocab=category_vocab_from_model_config(result.model_config, base['showdown_root']),
                feature_masks=feature_masks_from_model_config(result.model_config),
                set_belief_source=bool(base['belief_set_source'])))
        try:rows, parity, decided = replay(env, policy, game, rng, extras)
        except Mismatch as error:
            reason = re.sub(r"'[^']*'|\"[^\"]*\"|\d+", '#', str(error));consecutive_errors = 0
            mismatches[reason] += 1;examples.setdefault(reason, dict(ordinal=game['ordinal'], detail=str(error)));continue
        except (LocalShowdownError, ValueError, KeyError) as error:   # one bad game, not the shard; the env may be wedged
            reason = f'env error {type(error).__name__}'
            mismatches[reason] += 1;examples.setdefault(reason, dict(ordinal=game['ordinal'], detail=str(error)[:500]))
            env.close();env = None;consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ENV_ERRORS:raise SystemExit(f'{consecutive_errors} consecutive env errors: {examples[reason]}')
            continue
        consecutive_errors = 0;seats[game['fp']] += 1
        counts['games'] += 1;parity_hits += parity;parity_total += decided
        for row in rows:
            observation = row['observation']
            columns['categorical_ids'].append(np.asarray(observation.categorical_ids, np.int32))
            columns['numeric_features'].append(np.asarray(observation.numeric_features, np.float32))
            columns['token_type_ids'].append(np.asarray(observation.token_type_ids, np.int16))
            columns['attention_mask'].append(np.asarray(observation.attention_mask, bool))
            columns['legal'].append(np.asarray(observation.legal_action_mask, bool))
            columns['probs'].append(np.asarray(row['probs'], np.float32))
            columns['value'].append(np.float32(row['value'] if row['value'] is not None else np.nan))
            for key_ in ('label', 'round', 'turn', 'forced'):columns[key_].append(row[key_])
            columns['fp_opp_hazard_credit'].append(np.float32(row['fp_opp_hazard_credit']))
            counts['spikes_unknown_before'] += row['spikes_unknown_before']
            gap = abs(float(row['fp_opp_hazard_credit']) - float(columns['numeric_features'][-1][0, OPP_HAZARD_CREDIT_COLUMN]))
            counts['hazard_credit_rows_differ'] += gap > 0;hazard_gap = max(hazard_gap, gap)
            if extras is not None:
                columns['candidate_class'].append(np.asarray(row['candidate_class'], np.int8));columns['label_class'].append(row['label_class'])
                columns['own_hp'].append(np.float32(row['own_hp']));columns['opp_hp'].append(np.float32(row['opp_hp']))
                columns['opp_head_logits'].append(np.asarray(row['opp_head_logits'], np.float32))
            columns['ordinal'].append(game['ordinal']);columns['seed'].append(game['seed']);columns['seat'].append(game['seat'])
        counts['decisions'] += len(rows)
        if counts['games'] % 25 == 0:
            print(json.dumps(dict(event='PROGRESS', shard=args.shard, games=counts['games'], decisions=counts['decisions'],
                                  mismatched=sum(mismatches.values()), seconds=round(time.time() - started, 1))), flush=True)
    if env is not None:env.close()
    arrays = {k: np.stack(v) if k in ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'legal', 'probs',
                                      'candidate_class', 'opp_head_logits')
              else np.asarray(v) for k, v in columns.items()}
    np.savez(out / f'shard-{args.shard}.npz', **arrays)
    summary = dict(schema='v2s-fp-reply-data.v1', collection=str(collection), checkpoint=str(args.checkpoint), checkpoint_sha256=sha,
                   shard=args.shard, shards=args.shards, results_seen=len(paths), **counts, mismatched=sum(mismatches.values()),
                   mismatch_reasons=dict(mismatches), mismatch_examples=examples, foul_play_seat_games=dict(seats),
                   encoder_parity=dict(hits=parity_hits, decisions=parity_total, share=parity_hits / parity_total if parity_total else None),
                   hazard_credit_max_abs_gap_vs_env=hazard_gap,
                   shapes={k: list(v.shape) for k, v in arrays.items()}, seconds=round(time.time() - started, 1))
    (out / f'shard-{args.shard}.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(dict(event='SHARD_DONE', **{k: summary[k] for k in ('shard', 'games', 'decisions', 'mismatched', 'seconds')
                                                if k in summary}, encoder_parity=summary['encoder_parity'])), flush=True)


def split_of(seed):
    """Held-out split by game: battle seed % 10 == 0 is test, == 1 is validation, the rest train."""
    return np.where(seed % 10 == 0, 'test', np.where(seed % 10 == 1, 'val', 'train'))


ECE_BINS = 15


def calibration_error(confidence, correct, bins=ECE_BINS):
    """Expected calibration error of the top-1 prediction: equal-width confidence bins, |accuracy - confidence| weighted by bin size."""
    edges = np.linspace(0., 1., bins + 1);which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[which == b].mean() - confidence[which == b].mean()) * (which == b).mean()
                     for b in range(bins) if (which == b).any()))


def metrics(probs, labels, legal, *, uniform=False):
    """top-1 (argmax over legal), NLL of the label, the share of labels under 1% and 5%, top-1 ECE, and switch mass (predicted vs
    actual), on one set of rows. uniform=True (probs uniform over legal): top-1 is a uniform guess's expected accuracy, mean
    1/|legal|, not the argmax's lowest-index tie-break, and ECE is not reported (every legal action ties)."""
    if not len(labels):return dict(decisions=0)
    masked = np.where(legal, probs, -1.);top = masked.argmax(1)
    raw = probs[np.arange(len(labels)), labels];p = np.clip(raw, 1e-12, 1.)
    predicted_switch = np.where(legal[:, MOVES:], probs[:, MOVES:], 0.).sum(1)
    top1 = float((1. / legal.sum(1)).mean()) if uniform else float((top == labels).mean())
    return dict(decisions=int(len(labels)), top1=top1, nll=float(-np.log(p).mean()),
                label_below_1pct=float((raw < .01).mean()), label_below_5pct=float((raw < .05).mean()),
                ece=None if uniform else calibration_error(masked.max(1), (top == labels).astype(float)),
                predicted_switch_mass=float(predicted_switch.mean()), actual_switch_share=float((labels >= MOVES).mean()),
                mean_label_probability=float(p.mean()))


def report(out, split):
    shards = sorted(glob.glob(str(Path(out) / 'shard-*.npz')))
    cols = collections.defaultdict(list)
    for path in shards:
        data = np.load(path)
        for key in ('probs', 'label', 'legal', 'turn', 'forced', 'seed', 'ordinal'):cols[key].append(data[key])
    data = {k: np.concatenate(v) for k, v in cols.items()}
    keep = np.ones(len(data['label']), bool) if split == 'all' else split_of(data['seed']) == split
    data = {k: v[keep] for k, v in data.items()}
    probs, labels, legal = data['probs'], data['label'], data['legal']
    can_switch = legal[:, MOVES:].any(1);can_move = legal[:, :MOVES].any(1)
    groups = dict(all=np.ones(len(labels), bool), move_request=~data['forced'], forced_switch=data['forced'],
                  free_choice=can_move & can_switch & ~data['forced'], label_move=labels < MOVES, label_switch=labels >= MOVES)
    for low, high in TURN_BANDS:
        name = f'turn {low}-{high}' if high < 10**9 else f'turn {low}+'
        groups[name] = (data['turn'] >= low) & (data['turn'] <= high)
    out_report = dict(schema='v2s-fp-reply-baseline.v1', split=split, shards=len(shards), games=int(len(np.unique(data['ordinal']))),
                      groups={name: metrics(probs[m], labels[m], legal[m]) for name, m in groups.items()})
    print(json.dumps(out_report, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('collection', nargs='?');parser.add_argument('out', nargs='?')
    parser.add_argument('--checkpoint');parser.add_argument('--shard', type=int, default=0);parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--extras', action='store_true', help="also record action classes, HP and our view's opponent-head logits")
    parser.add_argument('--report', metavar='OUT_DIR');parser.add_argument('--split', choices=('all', 'test', 'val', 'train'), default='all')
    args = parser.parse_args(argv)
    if args.report:report(args.report, args.split);return 0
    if not (args.collection and args.out and args.checkpoint):parser.error('COLLECTION, OUT_DIR and --checkpoint are required')
    if not 0 <= args.shard < args.shards:parser.error('0 <= shard < shards required')
    build(args);return 0


if __name__ == '__main__':
    sys.exit(main())
