"""Champion 9375 vs V2-B1 forward microbenchmark: warm ms per call and rows per second at the pricers' batch shapes.

Each evaluator is timed exactly as the pricers call it, forward(params, inputs) on V2 public29 encoder inputs [rows, 1, 29, ...]
with its params closed over (as the pricers' programs hold them): the champion through champion_evaluator.ChampionEvaluator
(its 23-token slice, validity rule and NaN poisoning included; --champion-precision, default bf16), V2 through
device_depth2_pricer.default_forward (v5_model.forward(..., model, compute_dtype='bfloat16'), the pricers' default). Per
(evaluator, rows): one jax.jit, lowered and compiled ONCE (lower and compile seconds reported apart, never in a sample), then
--warmup untimed calls and --repeats timed calls, each dispatched and synced with block_until_ready on the outputs; reported:
median, min, p90 and mean ms per call, rows per second = rows / median seconds and microseconds per row. Inputs are
deterministic synthetic V2 rows (--seed): finite numerics, categories in [0, the smallest evaluator vocabulary), the exact
TOKEN_ROLES, every token attended, history True, so every row is valid for both evaluators; neither forward has data-dependent
control flow, so the time does not depend on the input values, nor on the weight values. Default --rows: 2187 (the pricers'
turn-2 block: Depth2Pricer child_block 27 x 81, DepthNPricer block_lanes 2187), 4096 (A3's champion shape) and 34992 (16
blocks of 2187, one DepthNPricer call at blocks_per_call 16). The result is bench.json in a new --out directory (JSON lines
on stdout as it goes). The champion's archive is bound by champion_evaluator.load (archive and checkpoint sha256, the checkpoint
9375's: champion_evaluator.check_checkpoint); the V2
archive by v5_model.load (archive and checkpoint sha256; its category and label namespaces are the archive's own recorded
ones, which the archive sha binds, unless --v2-category-sha256 / --v2-label-sha256 pin them).
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

SCHEMA = 'pokezero.champion-forward-bench.v1'
DEFAULT_ROWS = (2187, 4096, 34992)
SCOPE = ('warm forward only, as the pricers call it (params closed over, V2 inputs [rows,1,29,...]); compile excluded and reported apart; '
    'synthetic valid rows; no encoder, stepping or pricer program around it')


def ints(text):
    return tuple(int(x) for x in str(text).split(','))


def synthetic_inputs(rows, *, vocabulary, seed=20261107):
    """Deterministic V2 public29 inputs [rows,1,29,...] every evaluator accepts as valid (host numpy)."""
    from pokezero.v5_model import TOKEN_ROLES
    rng = np.random.default_rng([seed, rows]);tokens = len(TOKEN_ROLES)
    return dict(numeric_features=rng.normal(0, 1, (rows, 1, tokens, 132)).astype(np.float32),
        categorical_ids=rng.integers(0, vocabulary, (rows, 1, tokens, 41)).astype(np.int32),
        token_type_ids=np.broadcast_to(np.asarray(TOKEN_ROLES, np.int32), (rows, 1, tokens)).copy(),
        attention_mask=np.ones((rows, 1, tokens), bool), history_mask=np.ones((rows, 1), bool))


def percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q)) if values else None


def time_forward(forward, params, inputs, *, warmup, repeats, timer=time.perf_counter):
    """One evaluator at one batch: (compile record, samples in seconds, last output). Lower and compile are timed apart and never
    sampled; every sample is one call to block_until_ready."""
    import jax
    program = jax.jit(lambda x: forward(params, x))
    clock = timer();lowered = program.lower(inputs);lower = timer() - clock
    clock = timer();compiled = lowered.compile();compile_ = timer() - clock
    out = None
    for _ in range(warmup):out = jax.block_until_ready(compiled(inputs))
    samples = []
    for _ in range(repeats):
        clock = timer();out = jax.block_until_ready(compiled(inputs));samples.append(timer() - clock)
    return dict(lower_seconds=lower, compile_seconds=compile_), samples, out


def summarize(rows, samples):
    median = statistics.median(samples)
    return dict(rows=rows, repeats=len(samples), median_ms=1e3 * median, min_ms=1e3 * min(samples), p90_ms=1e3 * percentile(samples, 90),
        mean_ms=1e3 * statistics.fmean(samples), rows_per_second=rows / median if median > 0 else None,
        us_per_row=1e6 * median / rows, samples_ms=[1e3 * s for s in samples])


def run(evaluators, rows, *, warmup=3, repeats=20, seed=20261107, progress=None, timer=time.perf_counter):
    """evaluators: [(name, forward, params, header, vocabulary)]; every evaluator at every rows value, the same inputs per rows.
    Returns the report's `results` ({name: {rows: record}}) and `inputs` sections; progress(record) after each (name, rows)."""
    import jax
    vocabulary = min(v for *_, v in evaluators);results = {name: {} for name, *_ in evaluators};shapes = {}
    for count in rows:
        inputs = jax.device_put(synthetic_inputs(count, vocabulary=vocabulary, seed=seed));shapes[str(count)] = {k: list(x.shape) for k, x in inputs.items()}
        for name, forward, params, header, _ in evaluators:
            compiled, samples, out = time_forward(forward, params, inputs, warmup=warmup, repeats=repeats, timer=timer)
            valid = np.asarray(out.valid);value = np.asarray(out.value);logits = np.asarray(out.policy_logits)
            record = dict(summarize(count, samples), **compiled, valid_rows=int(valid.sum()), finite=bool(np.isfinite(value).all() and np.isfinite(logits).all()),
                abs_value_p50=float(np.median(np.abs(value))) if value.size else None, output_shapes=dict(policy_logits=list(logits.shape), value=list(value.shape)))
            if record['valid_rows'] != count or not record['finite']:raise RuntimeError(f'{name} at {count} rows: synthetic rows must all be valid and finite')
            results[name][str(count)] = record
            if progress is not None:progress(dict(event='FORWARD_TIMED', evaluator=name, **{k: v for k, v in record.items() if k != 'samples_ms'}))
    ratios = {}
    names = [name for name, *_ in evaluators]
    if 'champion' in names and 'v2' in names:
        ratios = {str(c): results['champion'][str(c)]['median_ms'] / results['v2'][str(c)]['median_ms'] for c in rows}
    return dict(results=results, ratio_champion_over_v2_median=ratios or None, inputs=dict(seed=seed, vocabulary=vocabulary, shapes=shapes,
        rule='synthetic V2 rows, all valid: finite numerics, categories in [0, vocabulary), exact TOKEN_ROLES, every token attended, history True'),
        evaluators={name: header for name, _, _, header, _ in evaluators})


def load_v2(args):
    """(forward, params, header, vocabulary) of --v2-archive: v5_model.load under the archive and checkpoint sha256s, the namespaces
    the archive records (bound by its sha) unless pinned, timed through the pricers' default forward (bf16)."""
    import io
    from gpu_engine import v5_model, champion
    from gpu_engine.device_depth2_pricer import default_forward
    from pokezero.v5_model import V5ModelConfig
    blob, digest = champion.snapshot(args.v2_archive)
    if digest != args.v2_archive_sha256:raise ValueError(f'--v2-archive sha256 {digest} differs from --v2-archive-sha256')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:model = json.loads(archive['metadata_json'].tobytes())['model']
    recorded = V5ModelConfig.from_metadata(model)
    params, config, meta = v5_model.load(args.v2_archive, expected_archive_sha256=args.v2_archive_sha256, expected_checkpoint_sha256=args.v2_checkpoint_sha256,
        expected_category_vocabulary_sha256=args.v2_category_sha256 or recorded.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=args.v2_label_sha256 or recorded.label_vocabulary.sha256, enable_v5=True)
    header = dict(kind='v2', archive=args.v2_archive, archive_sha256=args.v2_archive_sha256, checkpoint_sha256=args.v2_checkpoint_sha256, arm=config.arm,
        replicate=config.replicate, precision='bfloat16', parameter_count=meta.get('parameter_count'), forward='device_depth2_pricer.default_forward(model)',
        category_vocabulary_sha256=config.category_vocabulary_sha256, label_vocabulary_sha256=config.label_vocabulary.sha256)
    return default_forward(config), params, header, config.category_vocab_size


def load_champion(args):
    """(forward, params, header, vocabulary) of --champion-archive: champion_evaluator.load's checks, and the checkpoint must be 9375's
    (champion_evaluator.check_checkpoint), as the tools' check_binding requires; there is no bank catalog to bind here."""
    from gpu_engine import champion_evaluator as ce
    evaluator = ce.load_from_arguments(args);ce.check_checkpoint(evaluator)
    return evaluator, evaluator.params, dict(evaluator.header(), archive=args.champion_archive, forward='champion_evaluator.ChampionEvaluator'), evaluator.config.vocabulary


def parse(argv=None):
    from gpu_engine import champion_evaluator as ce
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', required=True, help='new output directory (bench.json)')
    parser.add_argument('--rows', default=','.join(map(str, DEFAULT_ROWS)), help='batch rows per call, comma-separated (default %(default)s)')
    parser.add_argument('--warmup', type=int, default=3, help='untimed warm calls per (evaluator, rows) after the compile')
    parser.add_argument('--repeats', type=int, default=20, help='timed calls per (evaluator, rows); the median is reported')
    parser.add_argument('--seed', type=int, default=20261107, help='synthetic input seed')
    ce.add_arguments(parser)
    parser.add_argument('--v2-archive', help='the V2 actor archive (e.g. B1-i99), timed through the pricers\' default bf16 forward')
    parser.add_argument('--v2-archive-sha256', help='sha256 of --v2-archive (required with it)')
    parser.add_argument('--v2-checkpoint-sha256', help='its checkpoint sha256 (required with --v2-archive)')
    parser.add_argument('--v2-category-sha256', help='optional: pin the archive\'s category vocabulary sha256')
    parser.add_argument('--v2-label-sha256', help='optional: pin the archive\'s label vocabulary sha256')
    args = parser.parse_args(argv)
    ce.check_arguments(parser, args)
    given = [args.v2_archive, args.v2_archive_sha256, args.v2_checkpoint_sha256]
    if any(given) and not all(given):parser.error('--v2-archive, --v2-archive-sha256 and --v2-checkpoint-sha256 go together')
    if (args.v2_category_sha256 or args.v2_label_sha256) and not args.v2_archive:parser.error('--v2-category/label-sha256 need --v2-archive')
    if not args.champion_archive and not args.v2_archive:parser.error('at least one of --champion-archive or --v2-archive required')
    try:args.row_values = ints(args.rows)
    except ValueError:parser.error('--rows: comma-separated integers')
    if any(r < 1 for r in args.row_values) or args.warmup < 0 or args.repeats < 1:parser.error('positive --rows and --repeats, --warmup >= 0')
    if Path(args.out).exists():parser.error(f'--out {args.out} exists: never overwritten')
    return args


def log(record):
    print(json.dumps(record, default=str), flush=True)


def main(argv=None):
    args = parse(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    import jax
    evaluators = []
    if args.champion_archive:evaluators.append(('champion',) + load_champion(args))
    if args.v2_archive:evaluators.append(('v2',) + load_v2(args))
    devices = [d.device_kind for d in jax.devices()]
    log(dict(event='BENCH_START', rows=list(args.row_values), warmup=args.warmup, repeats=args.repeats, devices=devices, evaluators=[e[0] for e in evaluators]))
    report = dict(schema=SCHEMA, devices=devices, jax=jax.__version__, rows=list(args.row_values), warmup=args.warmup, repeats=args.repeats, scope=SCOPE,
        **run(evaluators, args.row_values, warmup=args.warmup, repeats=args.repeats, seed=args.seed, progress=log))
    (out / 'bench.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')
    log(dict(event='BENCH_COMPLETE', out=str(out / 'bench.json'), ratio_champion_over_v2_median=report['ratio_champion_over_v2_median'],
        median_ms={name: {r: x['median_ms'] for r, x in by.items()} for name, by in report['results'].items()}))
    return 0


if __name__ == '__main__':raise SystemExit(main())
