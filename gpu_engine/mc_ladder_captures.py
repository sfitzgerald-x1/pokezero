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
  revealed only by [silent] lines, float32 rounding) and everything else counts as unexplained;
- the replay stops at the root (our target-th decision), whose legal actions and champion probabilities must equal the bank row's.

usage: python -m gpu_engine.mc_ladder_captures BANK_DIR OUT_DIR --collection DIR --checkpoint CHAMPION.pt [--sample N] [--seed S]
Writes OUT_DIR/owning-wire/<ordinal>.jsonl, OUT_DIR/manifest.jsonl (ordinal -> seed, ours, target) and OUT_DIR/validation.json.
"""
import argparse
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
# own request: (v4 physical numeric index, token range, rule). Tokens: 0 field, 1-6 our side (1 the active), 7-12 the opponent's.
SELF_TOKENS, OPP_TOKENS = range(1, 7), range(7, 13)
HP_FRACTION, LAST_DAMAGE_DEALT, LAST_DAMAGE_TAKEN, REVEALED_ABILITY, OPP_HAZARD_CREDIT = 0, 38, 39, 48, 116
FIELD_TOKEN = 0
PRECISION = 1e-6   # B2 numeric features are float32, env.observe's float64


def known_difference(token, feature, b2, env_value):
    """The known redaction a differing numeric cell belongs to, or None."""
    if abs(b2 - env_value) <= PRECISION:return 'float32'
    if token in OPP_TOKENS and feature in (HP_FRACTION, LAST_DAMAGE_TAKEN):return 'opponent_hp_public_percent'
    if token in SELF_TOKENS and feature == LAST_DAMAGE_DEALT:return 'damage_dealt_public_percent'
    if token == FIELD_TOKEN and feature == OPP_HAZARD_CREDIT:return 'opponent_hazard_public_percent'   # Spikes losses on the foe
    if token in OPP_TOKENS and feature == REVEALED_ABILITY and b2 == 0. and env_value == 1.:return 'silent_ability_reveal'
    return None


def compare(a, b):
    """How the B2 observation `a` differs from env.observe `b`: per known redaction class the cell count, and every other
    difference (non-numeric fields, or numeric cells outside the known classes) under 'unexplained'."""
    out, unexplained = {}, {}
    for name in FIELDS:
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.shape != y.shape:unexplained[name] = f'shape {x.shape} != {y.shape}';continue
        if np.array_equal(x, y):continue
        if name != 'numeric_features':unexplained[name] = int(np.sum(x != y));continue
        x, y = x.astype(float), y.astype(float)
        for t, f in zip(*np.nonzero(x != y)):
            kind = known_difference(int(t), int(f), x[t, f], y[t, f])
            if kind is None:unexplained.setdefault('numeric_features', []).append([int(t), int(f), float(x[t, f]), float(y[t, f])])
            else:out[kind] = out.get(kind, 0) + 1
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
        fed, count, ply = 0, 0, 0
        while env.terminal() is None:
            requested = list(env.requested_players())
            if ours in requested:   # the wire our seat has received so far, then the actor's decision boundary
                for chunk in env.owner_chunks[idx][fed:]:
                    actor.accept_owner_stream(battle_id=actor.battle_id, owner=ours, generation=actor.generation,
                                              prefix_sha256=actor.prefix_sha256, lines=chunk)
                fed = len(env.owner_chunks[idx]);snapshot = actor.snapshot_boundary()
                actor.decide(snapshot, rng=random.Random(0))
                own = env.observe(ours);checks.append(dict(decision=count, diff=compare(actor._decision_observation, own)))
                if count == target:
                    actions = [int(a) for a in np.flatnonzero(legal_of(own))]
                    probs = champion.probabilities([own])[0]
                    root = dict(actions_equal=actions == row['actions'],
                                policy_max_abs=float(np.max(np.abs(np.asarray(probs) - np.asarray(row['policy'])))))
                    break
            obs = [env.observe(p) for p in requested];probs = champion.probabilities(obs)
            actions = {p: sample(probs[i], legal_of(o), draw(seed, 10**6, ply, i)) for i, (p, o) in enumerate(zip(requested, obs))}
            count += ours in requested;env.step(actions);ply += 1
    known = {}
    for c in checks:
        for kind, n in c['diff'].items():
            if kind != 'unexplained':known[kind] = known.get(kind, 0) + n
    return dict(ordinal=ordinal, seed=seed, ours=ours, target=target, decisions=len(checks), known=known,
                mismatched=[c for c in checks if 'unexplained' in c['diff']], root=root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank');parser.add_argument('out');parser.add_argument('--collection', required=True)
    parser.add_argument('--checkpoint', required=True);parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from gpu_engine.mc_root_labeller import Model
    rows = sorted((json.loads(l) for p in glob.glob(str(Path(args.bank) / 'roots-self-*.jsonl')) for l in open(p) if l.strip()),
                  key=lambda r: (r['seed'], r['ours'], r['target']))
    if args.sample:rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    out = Path(args.out);(out / 'owning-wire').mkdir(parents=True, exist_ok=False)
    collection = Path(args.collection);first = sorted(glob.glob(str(collection / 'requests' / '*.json')))[0]
    base = json.loads(Path(first).read_bytes())['base_config'];champion = Model(args.checkpoint, torch)
    env = owning_env(base, champion.result);results = []
    try:
        with open(out / 'manifest.jsonl', 'w') as manifest:
            for ordinal, row in enumerate(rows):
                try:result = capture_self_root(env, champion, out, row, ordinal)
                except Exception as error:result = dict(ordinal=ordinal, seed=row['seed'], ours=row['ours'], target=row['target'],
                                                         error=f'{type(error).__name__}: {error}'[:400])
                results.append(result)
                manifest.write(json.dumps({k: result.get(k) for k in ('ordinal', 'seed', 'ours', 'target')}) + '\n');manifest.flush()
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
                   roots_matching_bank=sum(bool(r['root'] and r['root']['actions_equal'] and r['root']['policy_max_abs'] < 1e-6) for r in decided),
                   declaration_sha256=DECLARATION, results=results)
    (out / 'validation.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)
    return 0 if summary['errors'] == 0 and summary['decisions_mismatched'] == 0 and summary['roots_matching_bank'] == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
