"""Owner captures for the ladder's self roots, from the actual wire, validated against the labeller's own observations.

Foul Play gate roots have real owner captures (the collection's owning-wire). Self roots (mc_root_labeller self-play from a battle
seed) have none, so this module writes them the way the collection did:
- the battle is replayed exactly as mc_root_labeller.self_root plays it (same env configuration, the champion's probabilities on
  env.observe, the same deterministic draws), in an env that keeps Showdown's actual getPlayerStreams chunks for each seat
  (ChunkedOwningEnv: finite_stream_comparison.OwningShowdownEnv's rule, kept per bridge event);
- our seat's chunks are fed, in order, to an original LiveProtocolActor (the production Python owning backend; the policy is a
  capture-only stub whose choice is never used) wrapped by phase_d_owning_capture.capture_actor inside capture_to, so every
  WIRE and ROOT row is the capture module's own;
- at each of our decisions (every actionable own request, as the live actor decides) the actor's B2 observation (what its owning
  backend encoded from the wire) is compared with env.observe(ours) cell by cell; env.observe encodes the omniscient log, so the
  known redactions differ (known_difference: opponent HP, damage and Spikes losses at the public percent, an opponent ability
  revealed only by [silent] lines and that reveal's effect on the set belief (attribute_silent_reveals), float32 rounding) and
  everything else counts as unexplained;
- the replay stops at the root (our target-th decision), whose legal actions and champion probabilities must equal the bank row's.

usage: python -m gpu_engine.mc_ladder_captures BANK_DIR OUT_DIR --collection DIR --checkpoint CHAMPION.pt [--sample N] [--seed S]
Writes OUT_DIR/owning-wire/<ordinal>.jsonl, OUT_DIR/manifest.jsonl (ordinal -> seed, ours, target) and OUT_DIR/validation.json.
"""
import argparse
import collections
import glob
import hashlib
import json
import random
from pathlib import Path

import numpy as np

FIELDS = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'legal_action_mask')
DECLARATION = hashlib.sha256(b'mc-ladder self-root owner captures v1: actual getPlayerStreams chunks, original capture_actor').hexdigest()


def chunked_owning_env_class():
    from pokezero.local_showdown import LocalShowdownEnv

    class ChunkedOwningEnv(LocalShowdownEnv):
        """Keeps each seat's actual getPlayerStreams chunks of the current battle (one tuple per bridge event)."""
        def __init__(self, config):
            self.owner_chunks = [[], []]
            super().__init__(config)

        def reset(self, *args, **kwargs):
            self.owner_chunks = [[], []]
            return super().reset(*args, **kwargs)

        def _apply_event(self, event):
            if (event.get('type') == 'stream' and event.get('stream') in ('p1', 'p2')
                    and self._battle_token is not None and event.get('battleId') == self._battle_token):
                lines = event.get('lines')
                if not isinstance(lines, list) or not lines or any(type(line) is not str for line in lines):
                    raise ValueError('literal actual owning bridge lines required')
                self.owner_chunks[int(event['stream'][1]) - 1].append(tuple(lines))
            return super()._apply_event(event)
    return ChunkedOwningEnv


def owning_env(base, result):
    """mc_root_labeller.make_env's configuration, in a ChunkedOwningEnv."""
    from pokezero.local_showdown import LocalShowdownConfig
    from pokezero.neural_policy import category_vocab_from_model_config, feature_masks_from_model_config, observation_spec_from_model_config
    return chunked_owning_env_class()(LocalShowdownConfig(showdown_root=base['showdown_root'], node_binary=base['node_binary'],
        observation_spec=observation_spec_from_model_config(result.model_config),
        category_vocab=category_vocab_from_model_config(result.model_config, base['showdown_root']),
        feature_masks=feature_masks_from_model_config(result.model_config), set_belief_source=bool(base['belief_set_source'])))


# The known differences between the owner's wire (the B2 observation) and env.observe, which encodes the omniscient log with our
# own request. Tokens: 0 field, 1-6 our side (1 the active), 7-12 the opponent's. v4 physical numeric indices.
SELF_TOKENS, OPP_TOKENS = range(1, 7), range(7, 13)
HP_FRACTION, LAST_DAMAGE_DEALT, LAST_DAMAGE_TAKEN, REVEALED_ABILITY, OPP_HAZARD_CREDIT = 0, 38, 39, 48, 116
FIELD_TOKEN = 0
PRECISION = 1e-6     # B2 numeric features are float32, env.observe's float64
# The damage features (the active only, summed since the turn began, reset on switch-in) are read off the wire as differences of
# public percents (ceil, and 99 for a damaged Pokemon that would show 100). Back-to-back damage lines cancel each other's
# rounding, so the error grows only with the runs a heal or HP reset separates: each run is off by less than EVENT_ERROR, and
# the sum is a whole number of percent. OPP_HAZARD_CREDIT is checked exactly against fp_reply_data.fp_view_opp_hazard_credit
# (the opponent's Spikes losses at the public percent, / team size), recomputed from the omniscient log.
EVENT_ERROR = 0.02


def public_percent(x):
    """Showdown's public HP percent of an exact fraction (fp_reply_train.public_percent, scalar)."""
    p = float(np.ceil(np.round(100. * float(x), 3)))
    if x < 1 and p >= 100:p = 99.
    return min(max(p, 0.), 100.) / 100.


def whole_percent(x):
    """Whether a fraction is a whole number of percent (float32 noise allowed)."""
    return abs(100. * x - round(100. * x)) < 1e-3


def decision_context(env, idx, ours):
    """The log facts the value checks need at one decision: the opponent active's heal or HP-reset runs this turn, the public-
    percent Spikes credit of the opponent's losses, the opponent team's names in token order and the omniscient [silent]
    opponent ability lines our wire never carried."""
    from gpu_engine import fp_reply_data as fpd
    opp = 'p2' if ours == 'p1' else 'p1';lines = env.protocol_lines
    last = max((i for i, line in enumerate(lines) if line.startswith('|turn|')), default=-1)
    heals = sum(1 for line in lines[last + 1:] if line.startswith(('|-heal|', '|-sethp|')) and line.split('|')[2].startswith(opp + 'a'))
    names = [p.ident.split(': ', 1)[-1] for p in env._state_for_player(ours).opponent_team]
    wire = {line for chunk in env.owner_chunks[idx] for line in chunk}
    silent = [line for line in lines if line.startswith('|-ability|') and line.endswith('|[silent]')
              and line.split('|')[2].startswith(opp + 'a') and line not in wire]
    return dict(heals=heals, credit=float(fpd.fp_view_opp_hazard_credit(lines, opp)), names=names, silent=silent)


def known_difference(token, feature, b2, env_value, ctx):
    """The known redaction a differing numeric cell belongs to, checked by location AND value against the log (ctx:
    decision_context), or None."""
    d = abs(b2 - env_value)
    if d <= PRECISION:return 'float32'
    if token in OPP_TOKENS and feature == HP_FRACTION and abs(b2 - public_percent(env_value)) <= PRECISION:
        return 'opponent_hp_public_percent'
    runs = whole_percent(b2) and d < EVENT_ERROR * (1 + ctx['heals'])   # public-percent readings, one rounding per run
    if token in OPP_TOKENS and feature == LAST_DAMAGE_TAKEN and runs:return 'opponent_damage_public_percent'
    if token in SELF_TOKENS and feature == LAST_DAMAGE_DEALT and runs:return 'damage_dealt_public_percent'
    if token == FIELD_TOKEN and feature == OPP_HAZARD_CREDIT and abs(b2 - ctx['credit']) <= PRECISION:
        return 'opponent_hazard_public_percent'
    return None


def compare(a, b, ctx):
    """How the B2 observation `a` differs from env.observe `b`: per known class its cells [token, feature, b2, env], and every
    other differing cell (numeric or categorical, located) under 'unexplained'. A REVEALED_ABILITY cell (b2 0, env 1) on an
    opponent token is held under 'ability_reveal_candidates' for attribute_silent_reveals to check against the log."""
    out, unexplained = {}, {}
    for name in FIELDS:
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.shape != y.shape:unexplained[name] = f'shape {x.shape} != {y.shape}';continue
        if np.array_equal(x, y):continue
        x, y = x.astype(float), y.astype(float)
        cells = [(int(i[0]), int(i[1])) if x.ndim == 2 else (-1, int(i[0])) for i in zip(*np.nonzero(x != y))]
        for t, f in cells:
            xb, ye = (x[t, f], y[t, f]) if x.ndim == 2 else (x[f], y[f]);cell = [t, f, float(xb), float(ye)]
            kind = known_difference(t, f, xb, ye, ctx) if name == 'numeric_features' else None
            if name == 'numeric_features' and kind is None and t in OPP_TOKENS and f == REVEALED_ABILITY and xb == 0. and ye == 1.:
                kind = 'ability_reveal_candidates'
            if kind is None:unexplained.setdefault(name, []).append(cell)
            elif kind == 'float32':out['float32'] = out.get('float32', 0) + 1   # counted, not listed (millions of cells)
            else:out.setdefault(kind, []).append(cell)
    if unexplained:out['unexplained'] = unexplained
    return out


def attribute_silent_reveals(diff, ctx):
    """An opponent ability revealed only by a [silent] line of the omniscient log (never on our wire) shows in env.observe as that
    Pokemon's REVEALED_ABILITY and narrows env.observe's set belief for that Pokemon. A candidate REVEALED_ABILITY cell on
    token t is attributed ('silent_ability_reveal') only when a silent line before this decision names the Pokemon on t (the
    opponent team in token order, ctx['names']); then the unexplained cells on that SAME token, numeric or categorical, are its
    belief consequence ('silent_ability_reveal_belief'). Cells on any other token stay unexplained, and so do candidates whose
    Pokemon no silent line names."""
    out = {k: v for k, v in diff.items() if k not in ('ability_reveal_candidates', 'unexplained')}
    candidates = diff.get('ability_reveal_candidates', []);unexplained = dict(diff.get('unexplained', {}))
    named = {line.split('|')[2].split(': ', 1)[-1] for line in ctx['silent']}
    of = lambda t: ctx['names'][t - 7] if 0 <= t - 7 < len(ctx['names']) else None
    kept = [c for c in candidates if of(c[0]) in named];orphan = [c for c in candidates if of(c[0]) not in named]
    if orphan:unexplained.setdefault('numeric_features', []).extend(orphan)
    if kept:
        out['silent_ability_reveal'] = kept;out['silent_ability_lines'] = ctx['silent']
        tokens = {t for t, *_ in kept}
        for name in list(unexplained):
            cells = unexplained[name]
            if type(cells) is not list:continue
            mine = [c for c in cells if c[0] in tokens];rest = [c for c in cells if c[0] not in tokens]
            if mine:out.setdefault('silent_ability_reveal_belief', []).extend([name] + c for c in mine)
            if rest:unexplained[name] = rest
            else:del unexplained[name]
    if unexplained:out['unexplained'] = unexplained
    return out


def capture_self_root(env, champion, out_dir, row, ordinal):
    """Write one self root's owner capture; returns the per-decision comparisons and the root check."""
    from gpu_engine.mc_root_labeller import draw, legal_of, sample
    from pokezero.live_protocol_actor import ActorContract, LiveProtocolActor, PythonBackend, RawPolicy
    from pokezero.phase_d_owning_capture import capture_actor, capture_to
    from pokezero.policy import PolicyDecision
    seed, ours, target = int(row['seed']), row['ours'], int(row['target']);idx = int(ours[1]) - 1;config = env.config
    contract = ActorContract.champion()
    stub = lambda observation, rng: PolicyDecision(action_index=int(np.flatnonzero(observation.legal_action_mask)[0]), policy_id='capture-only')
    actor = LiveProtocolActor(contract, backend_factory=lambda c, **identity: PythonBackend(c, config=config, **identity),
                              policy_factory=lambda c: RawPolicy(c, select_action=stub, reset=lambda: None))
    checks, root = [], None
    with capture_to(Path(out_dir) / 'owning-wire', cell=dict(ordinal=ordinal, kind='self', seed=seed, ours=ours, target=target),
                    declaration_sha256=DECLARATION):
        actor = capture_actor(actor)
        env.reset(seed=seed)
        actor.begin_game(battle_id=f'mc-self-{seed}-{ours}', owner=ours, seed=seed, stream_id=ordinal + 1)
        fed, count, ply, close = 0, 0, 0, None
        try:
          while env.terminal() is None:
            requested = list(env.requested_players())
            if ours in requested:   # the wire our seat has received so far, then the actor's decision boundary
                for chunk in env.owner_chunks[idx][fed:]:
                    actor.accept_owner_stream(battle_id=actor.battle_id, owner=ours, generation=actor.generation,
                                              prefix_sha256=actor.prefix_sha256, lines=chunk)
                fed = len(env.owner_chunks[idx]);snapshot = actor.snapshot_boundary()
                actor.decide(snapshot, rng=random.Random(0))
                own = env.observe(ours);ctx = decision_context(env, idx, ours)
                diff = attribute_silent_reveals(compare(actor.decision_observation(snapshot), own, ctx), ctx)
                checks.append(dict(decision=count, diff=diff))
                if count == target:
                    actions = [int(a) for a in np.flatnonzero(legal_of(own))]
                    probs, values = champion.evaluate([own])
                    root = dict(actions_equal=actions == row['actions'],
                                policy_max_abs=float(np.max(np.abs(np.asarray(probs[0]) - np.asarray(row['policy'])))),
                                value_abs=None if row.get('root_value') is None else abs(float(values[0]) - float(row['root_value'])))
                    break
            obs = [env.observe(p) for p in requested];probs = champion.probabilities(obs)
            actions = {p: sample(probs[i], legal_of(o), draw(seed, 10**6, ply, i)) for i, (p, o) in enumerate(zip(requested, obs))}
            count += ours in requested;env.step(actions);ply += 1
        finally:   # a CLOSE row and the capture file closed (ABORTED_UNQUALIFIED: the replay stops at the root), whatever happened
            try:close = actor.close_game().get('status')
            except Exception as error:close = f'close_error:{type(error).__name__}'

    known = {}
    for c in checks:
        for kind, cells in c['diff'].items():
            if kind not in ('unexplained', 'silent_ability_lines'):known[kind] = known.get(kind, 0) + (cells if type(cells) is int else len(cells))
    attributed = [dict(decision=c['decision'], **{k: c['diff'][k] for k in ('silent_ability_reveal', 'silent_ability_reveal_belief',
                  'silent_ability_lines') if k in c['diff']}) for c in checks if 'silent_ability_reveal' in c['diff']]
    return dict(ordinal=ordinal, seed=seed, ours=ours, target=target, decisions=len(checks), known=known, attributed=attributed,
                mismatched=[c for c in checks if 'unexplained' in c['diff']], root=root, close=close)


def root_ok(root):
    """The replayed root is the bank's: the same legal actions, the champion's policy within 1e-6 and its value within 1e-5."""
    return bool(root and root['actions_equal'] and root['policy_max_abs'] < 1e-6 and root['value_abs'] is not None and root['value_abs'] < 1e-5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank');parser.add_argument('out');parser.add_argument('--collection', required=True)
    parser.add_argument('--checkpoint', required=True);parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker', type=int, default=0);parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 <= args.worker < args.workers:raise SystemExit('--worker W --workers N with 0 <= W < N required')
    suffix = '' if args.workers == 1 else f'-{args.worker}'   # a worker's manifest and validation; ordinals stay global
    import torch
    torch.set_num_threads(1)
    from gpu_engine.mc_root_labeller import Model
    rows = sorted((json.loads(l) for p in glob.glob(str(Path(args.bank) / 'roots-self-*.jsonl')) for l in open(p) if l.strip()),
                  key=lambda r: (r['seed'], r['ours'], r['target']))
    if args.sample:rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    out = Path(args.out);(out / 'owning-wire').mkdir(parents=True, exist_ok=args.workers > 1)
    collection = Path(args.collection);first = sorted(glob.glob(str(collection / 'requests' / '*.json')))[0]
    base = json.loads(Path(first).read_bytes())['base_config'];champion = Model(args.checkpoint, torch)
    env = owning_env(base, champion.result);results = []
    try:
        with open(out / f'manifest{suffix}.jsonl', 'x') as manifest:
            for ordinal, row in enumerate(rows):
                if ordinal % args.workers != args.worker:continue
                try:result = capture_self_root(env, champion, out, row, ordinal)
                except Exception as error:result = dict(ordinal=ordinal, seed=row['seed'], ours=row['ours'], target=row['target'],
                                                         error=f'{type(error).__name__}: {error}'[:400])
                results.append(result)
                result['status'] = ('error' if 'error' in result or str(result.get('close') or '').startswith('close_error')
                                    else 'root_mismatch' if not root_ok(result['root'])
                                    else 'unexplained' if result['mismatched'] else 'ok')
                manifest.write(json.dumps({k: result.get(k) for k in ('ordinal', 'seed', 'ours', 'target', 'status')}) + '\n');manifest.flush()
                print(json.dumps({k: result.get(k) for k in ('ordinal', 'decisions', 'root', 'error')} | {'mismatched': len(result.get('mismatched') or [])}), flush=True)
    finally:
        env.close()
    decided = [r for r in results if 'error' not in r]
    summary = dict(schema='mc-ladder-self-captures.v1', roots=len(results), errors=len(results) - len(decided),
                   decisions=sum(r['decisions'] for r in decided),
                   decisions_mismatched=sum(len(r['mismatched']) for r in decided),
                   roots_with_a_mismatch=sum(bool(r['mismatched']) for r in decided),
                   known_cells={k: sum(r['known'].get(k, 0) for r in decided) for k in sorted({k for r in decided for k in r['known']})},
                   unexplained_fields=sorted({f for r in decided for c in r['mismatched'] for f in c['diff']['unexplained']}),
                   roots_matching_bank=sum(root_ok(r['root']) for r in decided),
                   statuses=dict(collections.Counter(r['status'] for r in results)),
                   declaration_sha256=DECLARATION, results=results)
    (out / f'validation{suffix}.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)
    return 0 if summary['errors'] == 0 and summary['decisions_mismatched'] == 0 and summary['roots_matching_bank'] == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
