"""Reanalyse immutable gameplay captures without Native or engine advancement.

Only the original process-free owning PythonBackend/public writer is rerun.
Native failures and unresolved chance calls remain UNKNOWN; no resynchronization,
retry, new chance operand, model, or scientific root is created.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from . import finite_gameplay_comparison as comparison
from . import engine_gameplay_trace as trace

SCHEMA = 'pokezero.B4-saved-gameplay-reference-reanalysis.v1'


def _read_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('caller-bound original artifact bytes changed')
    return json.loads(raw)


def _records(path):
    with Path(path).open('rb') as source:
        while raw := source.readline(comparison.MAX_RECORD_BYTES + 1):
            if len(raw) > comparison.MAX_RECORD_BYTES or not raw.endswith(b'\n'):
                raise ValueError('bounded complete saved Native JSONL required')
            yield json.loads(raw)


def _differences(boundary, evidence, parties, vocabulary, layout):
    """Data-only adapters around the ORIGINAL represented-state predicate."""
    snapshot = SimpleNamespace(bridge_snapshot={'battle': evidence['battle']})
    terminal = evidence['terminal']
    env = SimpleNamespace(requested_players=lambda: tuple(evidence['requested_players']),
        terminal=lambda: None if terminal is None else SimpleNamespace(**terminal))
    return comparison.state_differences(boundary, snapshot, env, parties, vocabulary, layout)


def reanalyse(*, measurement_path, expected_measurement_sha256, trace_path, expected_trace_sha256,
              native_result_path, expected_native_result_sha256, native_decisions_path,
              expected_native_decisions_sha256, pool, catalog, env_config, out_path, record_bad_cells=False):
    """Stream the same complete ordered captures to an exclusive NEW artifact.

    The pool is read only for its bound original engine vocabulary; no sets are
    generated, packed, sampled or constructed. Original INITIAL Native order
    supplies physical party identity. Original state judgments must reproduce
    from the saved literal fields before any row is accepted into the reducer.
    """
    start = time.perf_counter_ns()
    measured = _read_json(measurement_path, expected_measurement_sha256)
    native = _read_json(native_result_path, expected_native_result_sha256)
    if trace.digest_file(native_decisions_path) != expected_native_decisions_sha256:
        raise ValueError('caller-bound original Native decision bytes changed')
    ids = [cell['stream_id'] for cell in measured['cohort']]
    roots = native['roots']
    receipt = comparison.original.source_provenance._pool(pool, rehash=True)
    vocabulary = pool.metadata()['vocabulary']
    if (native.get('schema') != comparison.SCHEMA
            or native.get('measurement_sha256') != expected_measurement_sha256
            or native.get('trace_sha256') != expected_trace_sha256
            or native.get('runtime_binding') != measured['runtime_binding']
            or native.get('pool_sha256') != measured['pool_receipt']['pool_sha256']
            or receipt['pool_sha256'] != native['pool_sha256'] or vocabulary is None
            or [row['stream_id'] for row in roots] != ids
            or [row['native_seed'] for row in roots] != list(range(native['native_seed_start'], native['native_seed_start'] + len(ids)))):
        raise ValueError('same saved source/runtime/pool and full ordered Native roster required')
    if native['decisions_artifact']['sha256'] != expected_native_decisions_sha256:
        raise ValueError('original Native summary/decision artifact join changed')
    out_path = Path(out_path)
    if out_path.resolve() in {Path(p).resolve() for p in
            (measurement_path, trace_path, native_result_path, native_decisions_path)}:
        raise ValueError('original artifacts are immutable')
    reducer = comparison.Reducer(ids)
    originals = comparison.Reducer(ids)
    records = iter(_records(native_decisions_path))
    with tempfile.TemporaryDirectory(prefix='saved-gameplay-spool-', dir=out_path.parent) as temp:
        header, counts = comparison._spool(trace_path, measured, expected_trace_sha256, Path(temp))
        with out_path.open('x') as output:
            for cell, root in zip(measured['cohort'], roots):
                stream = cell['stream_id']
                if (root['engine_status'] != cell['status'] or root['trace_decisions'] != counts[stream]
                        or not root['native_started']):
                    raise ValueError('original Native root/engine count binding changed')
                refs, owner_errors = [None, None], [None, None]
                try:
                    with (Path(temp) / str(stream)).open() as gpu:
                        initial = json.loads(next(gpu))['payload']['boundary']
                        saved = next(records)
                        if (saved.get('kind') != 'INITIAL' or saved['stream_id'] != stream
                                or saved['native_seed'] != root['native_seed']):
                            raise ValueError('missing/reordered original Native INITIAL')
                        evidence = saved['native_compared_state']
                        parties = [[comparison.fidelity.toid(mon['set']['species']) for mon in side['pokemon']]
                            for side in evidence['battle']['sides']]
                        differences = _differences(initial, evidence, parties, vocabulary, header['state_layout'])
                        initial_unknown = any(item['kind'] in ('UNRESOLVED', 'INITIAL_RANDOM')
                            for item in root['initial_chance_audit'])
                        continuous = not differences and not initial_unknown
                        status = 'UNKNOWN' if initial_unknown else 'AGREE' if continuous else 'DIVERGE'
                        if differences != root['initial_state_differences'] or status != root['initial_state_status']:
                            raise ValueError('saved INITIAL physical snapshot judgment changed')
                        for side in (0, 1):
                            try:
                                refs[side] = comparison.OwnerReference(catalog, env_config, owner=f'p{side+1}',
                                    root_id=stream, seed=root['native_seed'])
                                refs[side].append(saved['native_owner_lines'][side], constructor=True)
                            except Exception as error:
                                owner_errors[side] = str(error)
                        stopped = False
                        for raw in gpu:
                            captured = json.loads(raw)['payload']
                            old = next(records)
                            if (old.get('kind') is not None or old['stream_id'] != stream
                                    or old['request'] != captured['request']
                                    or old.get('engine_raw') != captured['after']['raw']):
                                raise ValueError('same original Native/GPU decision order/raw binding required')
                            originals.add(old)
                            row = deepcopy(old)
                            row['original_row_sha256'] = comparison.original._sha(old)
                            row['original_owners_sha256'] = comparison.original._sha(old['owners'])
                            if 'error' in old:
                                # A failed actual call has no fabricated post-state.
                                if old['state_status'] != 'UNKNOWN':
                                    raise ValueError('failed Native call must remain UNKNOWN')
                                stopped, continuous = True, False
                            else:
                                if stopped:
                                    raise ValueError('saved Native continuation after global refusal')
                                if old['native_before_state_sha256'] != comparison.original._sha(evidence):
                                    raise ValueError('saved Native before/after snapshot chain changed')
                                pre = _differences(captured['before'], evidence, parties, vocabulary, header['state_layout'])
                                evidence = old['native_compared_state']
                                post = _differences(captured['after'], evidence, parties, vocabulary, header['state_layout'])
                                unresolved = any(item['kind'] in ('UNRESOLVED', 'INITIAL_RANDOM') for item in old['chance_audit'])
                                eligible = continuous and not pre and not post and captured['executed'] and not unresolved
                                status = 'UNKNOWN' if unresolved else 'AGREE' if eligible else 'DIVERGE'
                                if (pre != old['before_state_differences'] or post != old['after_state_differences']
                                        or continuous != old['prior_state_agreement'] or status != old['state_status']):
                                    raise ValueError('saved decision physical snapshot judgment changed')
                                continuous = continuous and not pre and not post and not unresolved
                                row['owners'] = []
                                for side in (0, 1):
                                    lines = old['native_owner_lines'][side]
                                    try:
                                        if owner_errors[side] is not None:
                                            raise ValueError('PRIOR_OWNER_REFERENCE_REFUSAL:' + owner_errors[side])
                                        projected = refs[side].append(lines)
                                        owner = comparison.compare_owner(captured['after'], side, projected, refs[side], eligible=eligible,record_bad_cells=record_bad_cells)
                                    except Exception as error:
                                        if owner_errors[side] is None: owner_errors[side] = str(error)
                                        owner = comparison.owner_refusal(lines, error)
                                    if record_bad_cells and owner.get('columns',{}).get('status')=='COMPARED' and owner['columns'].get('bad_cells',{}).get('status')!='AVAILABLE':
                                        raise ValueError('diagnostic operands unavailable; original metrics not retagged')
                                    if unresolved:
                                        owner['refusals'].append(dict(kind='UNRESOLVED_CHANCE', reason='original coupling audit'))
                                        owner['columns'] = dict(status='UNKNOWN', reason='UNRESOLVED_CHANCE')
                                    row['owners'].append(owner)
                            reducer.add(row)
                            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                        missing = int(measured['gameplay_trace']['attempts'].get(str(stream),
                            measured['gameplay_trace']['attempts'].get(stream, 0))) - counts[stream]
                        for request in range(counts[stream], counts[stream] + missing):
                            old = next(records)
                            if (old['stream_id'], old['request'], old['state_status']) != (stream, request, 'UNKNOWN'):
                                raise ValueError('missing engine capture denominator changed')
                            originals.add(old); reducer.add(old)
                            output.write(json.dumps(dict(old, original_row_sha256=comparison.original._sha(old)), sort_keys=True) + '\n')
                finally:
                    for ref in refs:
                        if ref is not None: ref.close()
            if next(records, None) is not None:
                raise ValueError('extra/foreign original Native decisions')
    if originals.summary() != native['summary']:
        raise ValueError('original full-roster reduction changed')
    if record_bad_cells and reducer.summary()!=native['summary']:
        raise ValueError('diagnostic reference drift: original full-roster metrics must reproduce')
    if trace.digest_file(native_decisions_path) != expected_native_decisions_sha256:
        raise ValueError('original Native decision artifact changed during read')
    return dict(schema=SCHEMA, measurement_sha256=expected_measurement_sha256,
        original_native_result_sha256=expected_native_result_sha256,
        original_native_decisions_sha256=expected_native_decisions_sha256,
        trace_sha256=expected_trace_sha256, runtime_binding=measured['runtime_binding'],
        source=measured['source'], pool_sha256=native['pool_sha256'], roots=deepcopy(roots),
        original_summary=deepcopy(native['summary']), summary=reducer.summary(),
        decisions_artifact=dict(path=str(out_path), sha256=trace.digest_file(out_path)),
        reanalysis_elapsed_seconds=(time.perf_counter_ns()-start)/1e9,
        new_native_calls=0, new_engine_calls=0, new_chance_samples=0,
        full_B4_qualified=False, fidelity_qualified=False,
        scope='SAVED_LITERAL_OWNER_REFERENCE_REANALYSIS_ONLY; original physical judgments rechecked from literal snapshots; unresolved/failed Native calls unchanged; no retry/resync/new root; analysis wall is not throughput.')


def main(argv=None):
    import argparse
    from .finite_stream_comparison_saved import champion_env_config
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
    from pokezero.category_vocab import CategoryVocabulary
    from . import public_belief, source_provenance, v4_encoder
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('measurement', 'trace', 'native-result', 'native-decisions', 'team-pool',
                 'vocabulary', 'showdown-root', 'out', 'decisions-out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('measurement', 'trace', 'native-result', 'native-decisions'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--record-bad-cells',action='store_true')
    parser.add_argument('--node', default='node')
    args = parser.parse_args(argv)
    if args.out.exists() or args.decisions_out.exists(): raise FileExistsError('exclusive NEW outputs required')
    # Static original source/category tables only; no Native environment exists.
    values = json.loads(args.vocabulary.read_text())
    vocabulary = CategoryVocabulary(tuple(values['tokens']), values['oov_buckets'], values['aliases'])
    masks = CHAMPION_CONTRACT['feature_masks']
    catalog = v4_encoder.compile_encoder(public_belief.compile_source(load_gen3_randbat_source_cached(args.showdown_root)),
        load_showdown_dex_cached(args.showdown_root), vocabulary,
        checkpoint_sha256=values['checkpoint_sha256'], feature_masks=masks)
    config = champion_env_config(showdown_root=args.showdown_root, node=args.node, vocabulary=vocabulary, masks=masks)
    result = reanalyse(measurement_path=args.measurement, expected_measurement_sha256=args.measurement_sha256,
        trace_path=args.trace, expected_trace_sha256=args.trace_sha256,
        native_result_path=args.native_result, expected_native_result_sha256=args.native_result_sha256,
        native_decisions_path=args.native_decisions, expected_native_decisions_sha256=args.native_decisions_sha256,
        pool=source_provenance.load_constructor_pool(args.team_pool), catalog=catalog, env_config=config,
        out_path=args.decisions_out,record_bad_cells=args.record_bad_cells)
    with args.out.open('x') as output: json.dump(result, output, sort_keys=True, allow_nan=False)
    print(json.dumps(result['summary'], sort_keys=True))
    return 0


if __name__ == '__main__': raise SystemExit(main())
