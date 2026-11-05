"""U1: does completed-Q from the D recipe beat the raw head on the fixed held-out bank's top-two pairs?

Bank parents are the rollout-truth producer's reconstructed physical roots: one prefix replay under the
bank's original source, published only where world_id, root_id and the literal public29 observation all
matched the bank (988 SAVED, 36 MISSING on the fixed bank). Each is loaded with device_root_snapshot.load
as a one-row RootContext, re-encoded for both seats and forwarded by the evaluator head exactly as
device_d_cost.run builds a saved root's frames, detached with worlds.detach_roots and priced by the D3
Provider's FramePricer (frame_batch, chunk_batch, vectorized, batched_prep): two worlds x every legal own
action x every legal reply, value leaves, the head's own reply policy, reduce_completed_q. Only the
recipe's quarter selection (Recipe.selected) is bypassed, so that every parent is priced; world seeds,
kernel seeds, pricing order, worlds, branches and the reduction are Recipe(bank cost_recipe_seed)'s own.

W is not a recipe parameter (plan_starts, the pricer and reduce_completed_q fix two worlds). --world-blocks K
composes W = 2K from K independent recipe draws without changing shared code: block 0 is the declared
recipe, block k > 0 uses Recipe(seed_k) with seed_k derived from the declared seed. Q_W is the equal-weight
mean over all 2K worlds of the same reply-weighted values reduce_completed_q averages, refused if any
block refuses (no survivor renormalization), so the declared W=2 worlds are the prefix of every larger W.

Scoring imports pokezero.rollout_truth (the reviewed scorer bytes, checked by digest) and calls its
metrics() on the same truth/bank join as device_rollout_truth_score.evaluate. Every predictor's child
prediction is masked to the raw scorer's child availability (a stored head observation), so its eligible
top-two pairs are the raw head's pairs minus the roots D refuses. All paths are command-line arguments;
the output directory must be new.
"""
import argparse
import collections
import hashlib
import json
import math
import time
from pathlib import Path

REVIEWED_RULES_SHA256 = 'bd148149f5bedc182a15138070027ad169b6f41cb8fc0383e3e6f176da5c996f'
KEY = ('stream', 'frame', 'owner')
BLOCK_DOMAIN = 'pokezero.d3-u1.world-block.v1'
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
LEVEL = 0.90
NEGATIVE_UPPER = 0.02


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


def key(selector):
    return tuple(selector[k] for k in KEY)


def read_bytes(path, sha256):
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha256:raise ValueError('bound artifact digest differs: ' + str(path))
    return blob


def read_json(reference):
    return json.loads(read_bytes(reference['path'], reference['sha256']))


def read_jsonl(reference):
    return [json.loads(line) for line in read_bytes(reference['path'], reference['sha256']).splitlines()]


def rules():
    """The reviewed rollout-truth scorer, by digest: the same bytes every shared-bank head was scored with."""
    from pokezero import rollout_truth as module
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != REVIEWED_RULES_SHA256:
        raise ValueError('reviewed rollout-truth scorer bytes required')
    return module


def world_block_seeds(seed, blocks):
    """Block 0 is the declared recipe seed; later blocks are fixed derivations of it (no outcome input)."""
    if type(seed) is not int or type(blocks) is not int or blocks < 1:raise ValueError('declared seed and block count required')
    return (seed,) + tuple(int.from_bytes(hashlib.sha256(json.dumps([BLOCK_DOMAIN, seed, k]).encode()).digest()[:8], 'big')
        for k in range(1, blocks))


def world_prices(result):
    """The two WorldPrices a LABELED pricer result reduced, rebuilt from its world reports; None if refused."""
    from pokezero import v5_gumbel as original
    if result.status != 'LABELED':return None
    return tuple(original.WorldPrices(r['opponent_requested'], r['reply_legal'], r['reply_logits'], r['joint_values'])
        for r in result.world_reports)


def reduce_worlds(root_logits, root_legal, worlds):
    """reduce_completed_q over len(worlds) equally weighted worlds; for two worlds the identical arithmetic.

    Every legal reply of every legal own action in every world is required; any unknown refuses (None).
    """
    from pokezero import v5_gumbel as original
    legal = original._mask(root_legal);root_logits = tuple(root_logits);original._probabilities(root_logits, legal)
    worlds = tuple(worlds);count = len(worlds)
    if count < 2 or count % 2 or any(type(w) is not original.WorldPrices for w in worlds):
        raise ValueError('an even number of declared worlds required')
    weighted = [[] for _ in range(9)];unknown = 0
    for world in worlds:
        replies = tuple(i for i, active in enumerate(world.opponent_legal) if active) if world.opponent_requested else (0,)
        probabilities = original._probabilities(world.opponent_logits, world.opponent_legal) if world.opponent_requested else (1.,) + (0.,) * 8
        for action, active in enumerate(legal):
            if not active:continue
            for reply in replies:
                value = world.joint_values[action][reply]
                if value is None:unknown += 1
                else:weighted[action].append(probabilities[reply] * float(value) / count)
    if unknown:return None
    q = tuple(math.fsum(weighted[i]) if active else None for i, active in enumerate(legal))
    target = original._probabilities(tuple(float(logit) + (q[i] or 0.) for i, logit in enumerate(root_logits)), legal)
    value = math.fsum(p * v for p, v in zip(target, q) if v is not None)
    return dict(completed_q=q, policy_target=target, root_value=value, worlds=count)


def softmax_legal(logits, legal):
    from pokezero import v5_gumbel as original
    return original._probabilities(tuple(logits), original._mask(tuple(legal)))


def child_rows(records, values, available):
    """Scorer child rows: values[(key, action)] masked to the raw scorer's availability (else None)."""
    rows = []
    for r in records:
        k = (key(r['selector']), r['action'])
        rows.append(dict(selector=r['selector'], action=r['action'], rollouts=r['rollouts'],
            prediction=values.get(k) if available.get(k) else None))
    return rows


def pair_outcomes(pairs, values):
    """The scorer's rule on fixed pairs: correct = prediction(best) > prediction(second); None if unknown."""
    result = []
    for pair in pairs:
        a, b = (values.get((key(pair['selector']), x)) for x in pair['actions'])
        result.append(None if a is None or b is None else int(a > b))
    return result


def _interval(sums, counts, draws, rng):
    import numpy as np
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    index = rng.integers(0, len(sums), size=(draws, len(sums)))
    stats = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    low, high = np.quantile(stats, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])
    return [float(low), float(high)]


def paired_difference(pairs, raw, other, *, mode, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """(other correct - raw correct) / pairs on the same fixed pairs, with game-cluster and pair bootstrap CIs.

    mode: fallback (a refused root keeps the raw head's ranking: no search label, diff 0), conditional
    (only pairs the predictor resolves), refused_as_wrong (a refused pair counts as not correct).
    """
    import numpy as np
    if len(pairs) != len(raw) or len(raw) != len(other) or any(x not in (0, 1) for x in raw):raise ValueError('same fixed pairs required')
    if mode == 'fallback':rows = [(p, r, r if o is None else o) for p, r, o in zip(pairs, raw, other)]
    elif mode == 'refused_as_wrong':rows = [(p, r, 0 if o is None else o) for p, r, o in zip(pairs, raw, other)]
    elif mode == 'conditional':rows = [(p, r, o) for p, r, o in zip(pairs, raw, other) if o is not None]
    else:raise ValueError('declared pairing mode required')
    if not rows:return dict(mode=mode, pairs=0, difference=None)
    clusters = collections.OrderedDict()
    for p, r, o in rows:
        c = clusters.setdefault(p['selector']['stream'], [0, 0]);c[0] += o - r;c[1] += 1
    differences = [o - r for _, r, o in rows]
    rng = np.random.default_rng(seed)
    return dict(mode=mode, pairs=len(rows), games=len(clusters), raw_correct=sum(r for _, r, _ in rows),
        other_correct=sum(o for _, _, o in rows), difference=sum(differences) / len(rows),
        ci90_game_cluster=_interval([c[0] for c in clusters.values()], [c[1] for c in clusters.values()], draws, rng),
        ci90_pair=_interval(differences, [1] * len(differences), draws, rng),
        bootstrap=dict(draws=draws, seed=seed, level=LEVEL, statistic='ratio of resampled difference sum to resampled pair count',
            cluster='game (selector stream): every frame/owner pair of one game resampled together'))


def decision_class(intervals):
    """positive: some W's lower bound > 0; negative: every W's upper bound < +2 points; else ambiguous."""
    if not intervals:return None
    if any(low > 0 for low, _ in intervals):return 'positive'
    if all(high < NEGATIVE_UPPER for _, high in intervals):return 'negative'
    return 'ambiguous'


def score(R, roots, records, values, available):
    return R.metrics(roots, child_rows(records, values, available))


def summary_row(metrics):
    t = metrics['top_two'];c = metrics['child_calibration']
    return dict(eligible_pairs=t['eligible_known'], correct=t['correct'], resolution=t['resolution'], unknown_groups=t['unknown_groups'],
        below_gap_groups=t['below_gap_groups'], full_denominator_bounds=t['full_denominator_resolution_bounds'],
        child_known=metrics['child_known'], child_brier=None if c is None else c['brier'], child_ece=None if c is None else c['ece'],
        child_spearman=metrics['child_spearman'])


def predict(params, config, docs, *, batch=256):
    """device_rollout_truth_score.load_predictor's public-v5 forward, unchanged: eager BF16, 256-row batches.

    Returns (value or None, logits) per doc; value None unless finite, valid and within [-1, 1].
    """
    import jax
    import jax.numpy as j
    import numpy as np
    from gpu_engine import v5_model
    result = []
    for start in range(0, len(docs), batch):
        rows = docs[start:start + batch]
        inputs = {k: j.asarray([d[k] for d in rows], dtype)[:, None] for k, dtype in [('numeric_features', j.float32),
            ('categorical_ids', j.int32), ('token_type_ids', j.int32), ('attention_mask', bool)]}
        inputs['history_mask'] = j.ones((len(rows), 1), bool)
        out = v5_model.forward(params, inputs, config, compute_dtype='bfloat16')
        values, valid, logits = jax.block_until_ready((out.value, out.valid & j.isfinite(out.value) & j.all(j.isfinite(out.policy_logits), axis=-1),
            out.policy_logits))
        for v, ok, lg in zip(np.asarray(values), np.asarray(valid), np.asarray(logits), strict=True):
            result.append((float(v) if bool(ok) and np.isfinite(v) and abs(v) <= 1 else None, tuple(map(float, lg))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--truth-summary', required=True, help='rollout-truth summary.json (binds bank declaration, positions, siblings, runtime, truth)')
    parser.add_argument('--truth-summary-sha256', required=True)
    parser.add_argument('--reconstructed-index', required=True, help="the truth producer's reconstructed-root-index.json")
    parser.add_argument('--reconstructed-index-sha256', required=True)
    parser.add_argument('--bank-root', required=True, help='the held-out bank attempt directory (declaration.json, prespec.json)')
    parser.add_argument('--head', required=True, help='public-v5 head json (kind, archive path/sha256, checkpoint_sha256)')
    parser.add_argument('--raw-score', help="the shared scorer's stored score json for this head (raw baseline and reproduction check)")
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--world-blocks', type=int, default=1, help='K independent declared-recipe blocks; reports W=2 and W=8/32 when K>=4/16')
    parser.add_argument('--chunk', type=int, default=256, help='roots per pricer call (packing only)')
    parser.add_argument('--limit-roots', type=int, help='smoke only: price the first N available roots')
    parser.add_argument('--child-replay', action='store_true', help="replay every bank child in this tree's engine and compare to the bank")
    args = parser.parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter();R = rules();phases = {}
    def phase(name, since):phases[name] = time.perf_counter() - since;return time.perf_counter()

    # Declared bank, truth and reconstruction: every input digest-bound by the truth summary.
    clock = time.perf_counter()
    from pokezero import v5_device_evaluation_bank as schema
    summary = read_json(dict(path=args.truth_summary, sha256=args.truth_summary_sha256));manifest = summary['manifest']
    bank_root = Path(args.bank_root)
    if Path(manifest['declaration']['path']).resolve() != (bank_root / 'declaration.json').resolve():raise ValueError('bank root is not the truth declaration')
    decl, batches, games, labels = schema.declaration(manifest['declaration'])
    _, positions = schema._records(manifest['positions'], schema.POSITIONS)
    _, groups = schema._records(manifest['siblings'], schema.SIBLINGS)
    records = read_jsonl(summary['truth']);header = records.pop(0)
    if header != manifest or len(records) != manifest['declared_children'] or len(positions) != manifest['declared_slots']:
        raise ValueError('complete original truth denominator required')
    if [(r['selector'], r['action']) for r in records] != [(g['selector'], b['action']) for g in groups for b in g['branches']]:
        raise ValueError('truth selector/action order changed')
    for record, group in zip(records, [g for g in groups for _ in g['branches']], strict=True):
        if any(record[k] != group[k] for k in ('reply', 'chance_id', 'root_id', 'world_id')):raise ValueError('same recorded child world/reply/chance required')
    index = read_json(dict(path=args.reconstructed_index, sha256=args.reconstructed_index_sha256))
    original_index = read_json(manifest['root_index'])
    if index['namespace'] != original_index['namespace'] or [e['selector'] for e in index['cost_roots']] != [p['selector'] for p in positions]:
        raise ValueError('reconstructed index differs from the bank namespace/order')
    runtime = read_json(manifest['runtime']);prespec = json.loads((bank_root / 'prespec.json').read_text())
    head = json.loads(Path(args.head).read_text())
    if head.get('kind') != 'public-v5':raise ValueError('a public-v5 head is required: the D pricer evaluates v5 actors')
    stored = None
    if args.raw_score:
        score_doc = json.loads(Path(args.raw_score).read_text())
        if score_doc['head'] != head or score_doc['truth_summary']['sha256'] != args.truth_summary_sha256:raise ValueError('raw score is another head or truth')
        stored = read_json(score_doc['predictions'])
        if [(c['selector'], c['action']) for c in stored['children']] != [(r['selector'], r['action']) for r in records]:
            raise ValueError('stored predictions order differs from truth')
    log('BANK', positions=len(positions), children=len(records), saved_roots=sum(e['status'] == 'SAVED' for e in index['cost_roots']))
    clock = phase('bank_and_truth_load', clock)

    # The D3 Provider for this bank, as d3_saved_root_gate rebuilds it (caller's memory environment).
    from types import SimpleNamespace
    from dataclasses import replace
    from unittest.mock import patch
    from pokezero import v5_worker as worker, v5_gumbel as original, v5_contract as contract
    import jax
    import jax.numpy as j
    import numpy as np
    try:worker._probe(runtime)
    except ValueError as error:
        log('PROBE_DIFFERS', error=str(error));jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    profile, vocabulary, gender, category = worker._runtime_metadata(runtime, expected_pool_sha256=decl['pool_sha256'])
    prepared, pool, _, _ = worker.prepare_device_runtime(SimpleNamespace(runtime=runtime, vocabulary=vocabulary, plan=SimpleNamespace(pool_sha256=decl['pool_sha256'])))
    if (prepared.source_sha256 != decl['source_sha256'] or prepared.source_sha256 != index['namespace']['source_sha256']
            or prepared.pool_sha256 != decl['pool_sha256'] or prepared.category_sha256 != decl['category_vocabulary_sha256']):
        raise ValueError('prepared source/pool/category differs from the bank namespace')
    from gpu_engine import device_d_cost as cost, v5_device_teacher as teacher, v5_model, device_public_worlds as worlds
    from gpu_engine import device_root_snapshot as snapshot, v5_device_evaluation_bank as bank, engine_device_observation as encoder
    from gpu_engine import transition
    from gpu_engine.device_public_teacher_batch import Request
    from gpu_engine.benchmark_device_selfplay import request_seats
    plan = cost.plan_from_bank(profile=profile, labels=labels, category=category, batches=batches, pool_sha256=decl['pool_sha256'],
        selection_seed=decl['cost_recipe_seed'], run_id=prespec['cost_run_id'], max_requests=decl['max_requests'])
    provider = teacher.Provider(prepared, plan, frame_batch=True, chunk_batch=True, vectorized=True, batched_prep=True)
    params, head_config, meta = v5_model.load(head['archive']['path'], expected_archive_sha256=head['archive']['sha256'],
        expected_checkpoint_sha256=head['checkpoint_sha256'], expected_category_vocabulary_sha256=decl['category_vocabulary_sha256'],
        expected_label_vocabulary_sha256=labels.sha256, enable_v5=True)
    if replace(head_config, arm='D', replicate=0) != plan.model:raise ValueError('head geometry differs from the bank D model')
    actors = (params,)
    log('PROVIDER', arm=plan.model.arm, head_arm=head_config.arm, recipe_seed=provider.recipe.seed, parameter_count=meta['parameter_count'],
        devices=[d.device_kind for d in jax.devices()])
    clock = phase('runtime_prepare', clock)

    # Bank parents: reconstructed roots -> both-seat frames (device_d_cost.run's algebra) -> detached roots.
    observe = cost._observer(provider)
    forward_d = jax.jit(lambda p, inputs: v5_model.forward(p, inputs, plan.model, compute_dtype='bfloat16'))
    forward_h = jax.jit(lambda p, inputs: v5_model.forward(p, inputs, head_config, compute_dtype='bfloat16'))
    bank_executor = bank.make_executor(prepared, plan.model)
    roots = [];loaded = {};diagnostics = collections.Counter();arm_checked = False
    for ordinal, (entry, position, group) in enumerate(zip(index['cost_roots'], positions, groups, strict=True)):
        row = dict(ordinal=ordinal, selector=position['selector'], status='MISSING', reason=entry['reason'])
        roots.append(row)
        if entry['status'] != 'SAVED':continue
        if args.limit_roots is not None and len(loaded) >= args.limit_roots:row.update(status='NOT_IN_SMOKE');continue
        owner = position['selector']['owner']
        context, fields = snapshot.load(entry, index)
        root_view = jax.block_until_ready(bank_executor.child_view(context.state, context.public))[owner]
        row['root_public29_equal'] = bank._observation(root_view) == position['observation']
        diagnostics['root_public29_equal' if row['root_public29_equal'] else 'root_public29_differs'] += 1
        observations = observe(context.state, context.public)
        outputs = tuple(jax.block_until_ready(forward_d(params, encoder.actor_inputs(o))) for o in observations)
        if not arm_checked:
            other = jax.block_until_ready(forward_h(params, encoder.actor_inputs(observations[owner])))
            if not (np.array_equal(np.asarray(other.value), np.asarray(outputs[owner].value))
                    and np.array_equal(np.asarray(other.policy_logits), np.asarray(outputs[owner].policy_logits))):
                raise ValueError('D-arm and head-arm forwards differ')
            arm_checked = True
        requested = np.asarray(request_seats(context.state))
        legal = np.asarray(j.stack([o.legal_action_mask for o in observations], axis=1))
        errors = np.asarray(j.stack([o.error for o in observations], axis=1))
        finite = np.asarray(j.stack([o.valid & j.isfinite(o.value) & (j.abs(o.value) <= 1) & j.all(j.isfinite(o.policy_logits), axis=-1)
            & (j.all(j.isfinite(o.value_logits), axis=-1) if o.value_logits is not None else True) for o in outputs], axis=1))
        valid = requested & finite & (errors == 0) & legal.any(axis=-1)
        if not requested[0, owner]:row.update(status='NOT_REQUESTED', reason='selected owner has no actual PRE request');continue
        if not valid[0, owner]:row.update(status='REFUSED', reason='invalid-actual-root-input');continue
        inputs = {name: np.asarray(getattr(observations[owner], name))[0][None, None] for name in teacher.INPUTS}
        inputs['history_mask'] = np.ones((1, 1), bool)
        detached = worlds.detach_roots(context, provider.source, rows=[0], owners=[owner], enable_device_worlds=True)[0]
        if detached.address != contract.DecisionAddress(**position['selector']):raise ValueError('detached root address differs')
        loaded[ordinal] = dict(root=detached, logits=tuple(map(float, np.asarray(outputs[owner].policy_logits)[0])),
            legal=tuple(map(bool, legal[0, owner])), inputs=inputs, context=context, fields=fields)
        row.update(status='PRICED', reason=None)
        if len(loaded) % 100 == 0:log('ROOTS_LOADED', loaded=len(loaded), ordinal=ordinal)
    log('ROOTS_READY', priced=len(loaded), statuses=collections.Counter(r['status'] for r in roots), **diagnostics)
    clock = phase('root_load_encode_detach', clock)

    # Price every loaded parent under each recipe block. Only the quarter selection is bypassed.
    seeds = world_block_seeds(provider.recipe.seed, args.world_blocks);blocks = [];timings = []
    order = sorted(loaded)
    with patch.object(original.Recipe, 'selected', lambda self, address: True):
        for block, seed in enumerate(seeds):
            recipe = provider.recipe if block == 0 else original.Recipe(seed)
            requests = [Request(worlds.plan_starts(loaded[o]['root'], provider.source, recipe), loaded[o]['logits'], loaded[o]['legal'],
                loaded[o]['inputs'], 0, 0) for o in order]
            results = []
            for offset in range(0, len(requests), args.chunk):
                results.extend(provider.frame_pricer.price(requests[offset:offset + args.chunk], provider.materializer, recipe, actors))
                timings.append(dict(block=block, **{k: v for k, v in provider.frame_pricer.last_timing.items() if not isinstance(v, (list, dict))}))
            if len(results) != len(order):raise ValueError('pricer dropped a root')
            blocks.append(dict(zip(order, results)))
            log('BLOCK_PRICED', block=block, seed=seed, labeled=sum(r.status == 'LABELED' for r in results),
                refused=sum(r.status != 'LABELED' for r in results), device_seconds=sum(t['device_seconds'] for t in timings if t['block'] == block),
                compile_seconds=sum(t['compile_seconds'] for t in timings if t['block'] == block))
    clock = phase('pricing', clock)

    # Per-root Q at each W (prefix of blocks); W=2 must reproduce the pricer's own reduction exactly.
    widths = [w for w in (2, 8, 32) if w // 2 <= len(seeds)]
    q_values = {w: {} for w in widths};improved = {w: {} for w in widths};root_values = {w: {} for w in widths}
    refusals = collections.Counter()
    for o in order:
        k = key(roots[o]['selector']);prices = [world_prices(b[o]) for b in blocks]
        first = blocks[0][o]
        roots[o]['block_status'] = [b[o].status for b in blocks]
        if first.status != 'LABELED':
            reason = first.reason or 'unknown';refusals[reason] += 1;roots[o]['refusal'] = reason
            roots[o]['world_reasons'] = [r.get('reason') for r in first.world_reports]
        for w in widths:
            used = prices[:w // 2]
            if any(p is None for p in used):continue
            reduced = reduce_worlds(loaded[o]['logits'], loaded[o]['legal'], [x for p in used for x in p])
            if reduced is None:raise ValueError('a labeled block cannot hold an unknown branch')
            if w == 2:
                target = first.reduction.improvement
                if (reduced['completed_q'] != target.completed_q or reduced['policy_target'] != target.policy_target
                        or reduced['root_value'] != target.root_value):raise ValueError('W=2 reduction differs from the pricer reduction')
            for a, (qa, pa) in enumerate(zip(reduced['completed_q'], reduced['policy_target'])):
                if qa is not None:q_values[w][(k, a)] = qa;improved[w][(k, a)] = 2 * pa - 1
            root_values[w][k] = reduced['root_value']
            roots[o]['q_w%d' % w] = list(reduced['completed_q'])
    log('Q_READY', widths=widths, labeled={w: len(root_values[w]) for w in widths}, refusals=dict(refusals))
    clock = phase('reduce', clock)

    # Raw head on the bank's stored observations (the shared scorer's forward), and its stored predictions.
    observed = [(p, p['observation']) for p in positions if p['observation'] is not None]
    root_predictions = dict(zip([key(p['selector']) for p, _ in observed], predict(params, head_config, [d for _, d in observed])))
    child_docs = [(key(g['selector']), b['action'], b['observation']) for g in groups for b in g['branches'] if b['status'] == 'OBSERVED']
    recomputed = dict(zip([(k, a) for k, a, _ in child_docs], (v for v, _ in predict(params, head_config, [d for _, _, d in child_docs]))))
    raw_policy = {}
    for p, _ in observed:
        k = key(p['selector']);logits = root_predictions[k][1];legal_mask = tuple(map(bool, p['observation']['legal_action_mask']))
        if any(legal_mask):
            for a, prob in enumerate(softmax_legal(logits, legal_mask)):
                if legal_mask[a]:raw_policy[(k, a)] = 2 * prob - 1
    clock = phase('raw_forward', clock)

    def outcome(p):
        if p['terminal'] is None or p['terminal']['status'] != 'COMPLETE':return None
        winner = p['terminal']['winner'];return 0 if winner == 2 else 1 if winner == p['selector']['owner'] else -1
    def root_rows(prediction):
        return [dict(selector=p['selector'], prediction=prediction(key(p['selector'])), outcome=outcome(p), backup=None) for p in positions]
    if stored is not None:
        raw_values = {(key(c['selector']), c['action']): c['prediction'] for c in stored['children']}
        raw_roots = stored['roots']
    else:
        raw_values = recomputed;raw_roots = root_rows(lambda k: root_predictions.get(k, (None,))[0])
    available = {k: v is not None for k, v in raw_values.items()}
    reproduction = dict(stored=stored is not None, recomputed_children=len(recomputed))
    if stored is not None:
        both = [(raw_values[k], v) for k, v in recomputed.items() if raw_values.get(k) is not None and v is not None]
        reproduction.update(children_compared=len(both), children_bitwise_equal=sum(a == b for a, b in both),
            children_max_abs_difference=max((abs(a - b) for a, b in both), default=None),
            children_availability_equal=all((raw_values.get(k) is not None) == (v is not None) for k, v in recomputed.items()))
        stored_roots = {key(r['selector']): r['prediction'] for r in stored['roots']}
        both = [(stored_roots[k], v[0]) for k, v in root_predictions.items() if stored_roots.get(k) is not None and v[0] is not None]
        reproduction.update(roots_compared=len(both), roots_max_abs_difference=max((abs(a - b) for a, b in both), default=None))

    # Score every predictor with the reviewed scorer on the same join; fixed pairs from the raw head.
    raw_metrics = R.metrics(raw_roots, child_rows(records, raw_values, available))
    if stored is not None:
        published = score_doc['metrics']
        same = all(raw_metrics['top_two'][k] == published['top_two'][k] for k in ('eligible_known', 'correct', 'unknown_groups', 'below_gap_groups'))
        same = same and raw_metrics['child_calibration']['brier'] == published['child_calibration']['brier']
        reproduction['published_metrics_reproduced'] = same
        if not same:raise ValueError('stored raw predictions do not reproduce the published score')
    pairs = raw_metrics['top_two']['rows'];raw_correct = [int(r['correct']) for r in pairs]
    if pair_outcomes(pairs, raw_values) != raw_correct:raise ValueError('fixed-pair rule differs from the scorer')
    predictors = {'raw_value': (raw_values, raw_roots), 'raw_value_recomputed': (recomputed, root_rows(lambda k: root_predictions.get(k, (None,))[0])),
        'raw_policy': (raw_policy, root_rows(lambda k: None))}
    for w in widths:
        predictors['completed_q_w%d' % w] = (q_values[w], root_rows(lambda k, w=w: root_values[w].get(k)))
        predictors['improved_policy_w%d' % w] = (improved[w], root_rows(lambda k: None))
    table = {};paired = {};metrics_out = {}
    for name, (values, root_list) in predictors.items():
        m = R.metrics(root_list, child_rows(records, values, available));metrics_out[name] = m
        row = summary_row(m);outcomes = pair_outcomes(pairs, values)
        row.update(fixed_pairs=len(pairs), fixed_pairs_correct=sum(x == 1 for x in outcomes), fixed_pairs_unknown=sum(x is None for x in outcomes),
            fixed_pairs_resolution_known=None if all(x is None for x in outcomes) else sum(x == 1 for x in outcomes) / sum(x is not None for x in outcomes))
        resolved = {(key(r['selector']), tuple(r['actions'])): int(r['correct']) for r in m['top_two']['rows']}
        fixed = {(key(p['selector']), tuple(p['actions'])): x for p, x in zip(pairs, outcomes) if x is not None}
        if resolved != fixed:raise ValueError(name + ': scorer pairs are not the fixed raw pairs minus unknowns')
        if name.startswith('completed_q'):
            # Brier on the children Q resolves, raw head on exactly those children.
            matched = {k: raw_values.get(k) if values.get(k) is not None else None for k in raw_values}
            mc = R.metrics(raw_roots, child_rows(records, matched, available))['child_calibration']
            row.update(matched_raw_child_brier=None if mc is None else mc['brier'], matched_raw_child_ece=None if mc is None else mc['ece'],
                matched_children=None if mc is None else mc['positions'])
        table[name] = row
        if name != 'raw_value':
            paired[name] = {mode: paired_difference(pairs, raw_correct, outcomes, mode=mode) for mode in ('fallback', 'conditional', 'refused_as_wrong')}
    q_names = ['completed_q_w%d' % w for w in widths]
    decision = dict(rule='positive if any W lower 90% bound > 0; negative if every W upper bound < +2 points; else ambiguous',
        primary_mode='fallback', interval='ci90_game_cluster',
        by_width={n: paired[n]['fallback']['ci90_game_cluster'] for n in q_names},
        decision_class=decision_class([paired[n]['fallback']['ci90_game_cluster'] for n in q_names]),
        decision_class_w2_only=decision_class([paired['completed_q_w2']['fallback']['ci90_game_cluster']]) if 'completed_q_w2' in paired else None,
        conditional_mode_class=decision_class([paired[n]['conditional']['ci90_game_cluster'] for n in q_names]))
    clock = phase('scoring', clock)

    # Optional engine-drift check: this tree's one-ply step of every bank child versus the bank's bytes.
    replay = None
    if args.child_replay:
        step = jax.jit(lambda s, p, o0, o1, acts, chance: bank_executor.advance(s, p, j.stack([encoder.map_actions(o0, s, j.asarray([0], j.int32), acts[:, 0]),
            encoder.map_actions(o1, s, j.asarray([1], j.int32), acts[:, 1])], axis=1), chance))
        replay = collections.Counter()
        for o in order:
            group = groups[o];owner = group['selector']['owner'];context, fields = loaded[o]['context'], loaded[o]['fields']
            if int(fields['actions'][0, 1 - owner]) != group['reply']:replay['reply_differs'] += len(group['branches']);continue
            if set(group['chance']) != set(transition.ChanceInputs._fields):replay['chance_fields_differ'] += len(group['branches']);continue
            chance = transition.ChanceInputs(*(j.asarray(group['chance'][k], j.int32) for k in transition.ChanceInputs._fields))
            for branch in group['branches']:
                acts = fields['actions'].at[0, owner].set(branch['action']).at[0, 1 - owner].set(group['reply'])
                after = jax.block_until_ready(step(context.state, context.public, fields['observations'][0], fields['observations'][1], acts, chance))
                if bool(after.step.invalid_actions[0]):replay[branch['status'] + ':engine_refusal'] += 1;continue
                if branch['status'] == 'COMPLETE':
                    actual = dict(done=bool(after.step.state.done[0]), winner=int(after.step.state.winner[0]), state_id=schema.digest(bank._host(after.step.state)))
                    replay['COMPLETE:' + ('match' if actual == branch['terminal'] else 'differs')] += 1
                elif branch['status'] == 'OBSERVED':
                    if bool(after.step.state.done[0]):replay['OBSERVED:now_terminal'] += 1;continue
                    view = jax.block_until_ready(bank_executor.child_view(after.step.state, after.trackers))[owner]
                    replay['OBSERVED:' + ('match' if int(view.error[0]) == 0 and bank._observation(view) == branch['observation'] else 'differs')] += 1
                else:replay[branch['status'] + ':not_compared'] += 1
        replay = dict(replay);log('CHILD_REPLAY', **replay)
        clock = phase('child_replay', clock)

    total = time.perf_counter() - started
    result = dict(schema='pokezero.d3-u1-completed-q.v1', smoke=args.limit_roots is not None,
        inputs=dict(truth_summary=dict(path=args.truth_summary, sha256=args.truth_summary_sha256),
            reconstructed_index=dict(path=args.reconstructed_index, sha256=args.reconstructed_index_sha256), bank_root=str(bank_root),
            head=head, raw_score=args.raw_score, rules_sha256=REVIEWED_RULES_SHA256),
        recipe=dict(declared_seed=provider.recipe.seed, block_seeds=list(seeds), widths=widths, selection='BYPASSED_EVERY_PARENT_PRICED',
            provider_flags=dict(frame_batch=True, chunk_batch=True, vectorized=True, batched_prep=True), chunk=args.chunk,
            composition='W=2K is the equal-weight mean over K independent declared-recipe blocks of two worlds; block 0 is the declared recipe'),
        reconstruction=dict(slots=len(roots), statuses=dict(collections.Counter(r['status'] for r in roots)), **diagnostics),
        pricing=dict(labeled_w2=len(root_values.get(2, {})), refused_w2=dict(refusals),
            labeled_by_width={w: len(root_values[w]) for w in widths}, calls=timings),
        reproduction=reproduction, table=table, paired_vs_raw_value=paired, decision=decision, child_replay=replay,
        environment=dict(jax=jax.__version__, devices=[d.device_kind for d in jax.devices()]),
        phases_seconds=phases, total_seconds=total,
        scope='Offline fixed-bank diagnostic; no games; raw-champion rollout truth; not a strength result')
    (out / 'result.json').write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + '\n')
    (out / 'metrics.json').write_text(json.dumps(metrics_out, sort_keys=True, allow_nan=False) + '\n')
    with (out / 'roots.jsonl').open('x') as stream:
        for row in roots:stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    log('U1_COMPLETE', total_seconds=total, decision=decision['decision_class'],
        table={n: {k: table[n][k] for k in ('eligible_pairs', 'correct', 'resolution', 'fixed_pairs_correct', 'fixed_pairs_unknown', 'child_brier')} for n in table})
    return 0


if __name__ == '__main__':raise SystemExit(main())
