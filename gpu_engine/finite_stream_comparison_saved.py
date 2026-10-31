"""Invoke the reviewed constructor comparator on actual saved GPU packets.

No state constructor/transition, checkpoint or model is called. JSON arrays are
decoded from the original initial packet. Missing packets stay UNKNOWN. Native
Showdown execution occurs only when the explicit command is run by its owner.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import finite_stream_comparison as comparison
from .state import BattleState


def champion_feature_masks(masks):
    """Translate the checkpoint metadata alias to the public encoder field."""
    from pokezero.observation import ObservationFeatureMasks
    return ObservationFeatureMasks(**{
        'opponent_tendency_stats_block' if name == 'stats_block' else name: value
        for name, value in masks.items()})


def champion_env_config(*, showdown_root, node, vocabulary, masks):
    from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC
    return comparison.LocalShowdownConfig(showdown_root=showdown_root, node_binary=node,
        observation_spec=V4_REPLAY_OBSERVATION_SPEC, category_vocab=vocabulary,
        feature_masks=champion_feature_masks(masks), set_belief_source=True)


def captured_observations(measured, index, count):
    result = [None, None]
    owners = measured.get('initial_owner_observations')
    if isinstance(owners, list) and len(owners) == 2:
        for side, owner in enumerate(owners):
            admitted = np.asarray(owner.get('admitted', []))
            if admitted.shape != (count,) or admitted.dtype != np.bool_ or not admitted[index]:
                continue
            try:
                result[side] = {name:np.asarray(owner[name], dtype=np.float32 if name == 'numeric_features'
                    else np.bool_ if name.endswith('mask') else np.int32)[index:index+1] for name in comparison.SHAPES}
            except (KeyError, ValueError, TypeError):
                pass
    return result


def reprocess_saved(measurement_bytes, native_result_bytes, catalog, *, expected_measurement_sha256,
                    expected_native_result_sha256, env_config):
    """Correct references using immutable saved evidence; never invoke Native.

    Original state-agreement judgments and team/root order are reused. This is
    offline reprocessing of literal owning captures, not another attempted cell.
    """
    for payload, expected in ((measurement_bytes, expected_measurement_sha256),
            (native_result_bytes, expected_native_result_sha256)):
        if type(payload) is not bytes or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('original saved artifact SHA256 changed')
    measured, native = json.loads(measurement_bytes), json.loads(native_result_bytes)
    cohort, originals = measured['cohort'], native['rows']
    ids = [row['stream_id'] for row in cohort]
    if (not ids or any(type(value) is not int for value in ids) or len(ids) != len(set(ids))
            or len(ids) != measured['original_denominator'] or len(originals) != len(ids)
            or type(measured.get('runtime_binding')) is not str or not measured['runtime_binding']
            or native.get('schema') != comparison.SCHEMA
            or native.get('measurement_file_sha256') != expected_measurement_sha256
            or native.get('source_measurement_binding') != measured.get('runtime_binding')
            or [row.get('root_id') for row in originals] != ids
            or any(row.get('team_indices') != selected['team_indices'] for row, selected in zip(originals, cohort))):
        raise ValueError('same original measurement, Native artifact and complete ordered roster required')
    captures = measured['initial_frame_input_witness']
    capture = tuple(np.asarray(captures[name]) for name in ('owner_full_words', 'owner_full_valid', 'owner_full_more'))
    if any(value.shape[0] != len(ids) for value in capture):
        raise ValueError('original complete engine FULL capture required')
    rows = []
    for index, original in enumerate(originals):
        lines = original.get('native_owner_lines')
        if (type(lines) is not list or len(lines) != 2 or any(type(owner) is not list or not owner
                or any(type(line) is not str for line in owner) for owner in lines)
                or original.get('state_status') not in ('AGREE', 'DIVERGE', 'UNKNOWN')):
            raise ValueError('literal preserved original two-owner Native captures and state judgment required')
        if type(original.get('native_seed')) is not int or not 0 <= original['native_seed'] < 2**31:
            raise ValueError('original recorded Native seed required; never invent one')
        row = deepcopy(original)
        row['original_row_sha256'] = comparison._sha(original)
        row['original_owners_sha256'] = comparison._sha(row.pop('owners', []))
        row['owners'] = []
        actual = captured_observations(measured, index, len(ids))
        for owner in range(2):
            reference = receipt = None
            reference_error = None
            if actual[owner] is not None:
                try:
                    reference, receipt = comparison.owning_python_reference.observe_owner(env_config,
                        lines[owner], owner=('p1','p2')[owner], battle_id=f'saved-constructor-{ids[index]}',
                        stream_id=ids[index], seed=original['native_seed'])
                except Exception as error:
                    reference_error = dict(type=type(error).__name__, message=str(error))
            try:
                compared = comparison.compare_owner(lines[owner], tuple(part[index, owner] for part in capture),
                    catalog, constructor=True, actual_observation=None if actual[owner] is None else
                    {name:value[0] for name,value in actual[owner].items()}, reference_observation=reference)
            except Exception as error:
                compared = dict(status='UNKNOWN', error=dict(type=type(error).__name__, message=str(error)),
                    columns=dict(status='UNKNOWN', reason='REFERENCE_COMPARISON_ERROR'))
            if reference_error is not None:
                compared['columns'] = dict(status='UNKNOWN', reason='OWNING_PYTHON_REFERENCE_REFUSED', detail=reference_error)
            if receipt is not None:
                compared['python_reference'] = receipt
            row['owners'].append(compared)
        rows.append(row)
    return dict(schema=comparison.SCHEMA, scope='SAVED_ACTUAL_OWNING_REFERENCE_REPROCESS_NO_NEW_NATIVE',
        source_measurement_binding=measured.get('runtime_binding'), measurement_file_sha256=expected_measurement_sha256,
        original_native_result_sha256=expected_native_result_sha256,
        original_native_summary_sha256=comparison._sha(native['summary']),
        original_state_judgments_reused=True, new_native_calls=0, rows=rows,
        summary=comparison.reduce_cohort(rows, root_ids=ids))


def compare_saved(measured, pool, catalog, *, showdown_root, node, native_seed_start, env_config=None,
                  original_ivs=None):
    if original_ivs is not None and original_ivs.pool_sha256 != measured.get('pool_receipt', {}).get('pool_sha256'):
        raise ValueError('saved measurement original pool binding differs')
    cohort = measured['cohort']
    ids = [row['stream_id'] for row in cohort]
    if (len(cohort) != measured['original_denominator'] or len(set(ids)) != len(ids)
            or any(type(value) is not int for value in ids)):
        raise ValueError('complete original finite cohort required')
    state_packet = measured.get('initial_engine_state')
    witness = measured.get('initial_frame_input_witness')
    required = ('owner_full_words', 'owner_full_valid', 'owner_full_more')
    if (not isinstance(state_packet, dict) or set(state_packet) != set(BattleState._fields)
            or not isinstance(witness, dict) or any(name not in witness for name in required)):
        rows = [dict(root_id=value, state_status='UNKNOWN', owners=[],
            error=dict(type='MISSING_ACTUAL_CONSTRUCTOR_CAPTURE',
                message='Original state and both owner FULL captures are required; no replay or inference.')) for value in ids]
        return dict(schema=comparison.SCHEMA, rows=rows, summary=comparison.reduce_cohort(rows, root_ids=ids))
    # Direct decode only: never empty_state/state_from_pool/initialize/step.
    state = BattleState(**{name:np.asarray(state_packet[name]) for name in BattleState._fields})
    capture = tuple(np.asarray(witness[name]) for name in required)
    if any(value.shape[0] != len(ids) for value in (*state, *capture)):
        raise ValueError('original saved constructor batch shape required')
    rows = []
    for index, root_id in enumerate(ids):
        actual = captured_observations(measured, index, len(ids))
        # B=1 slicing preserves the actual packet and allows each refused owner
        # to remain absent independently. Native seeds retain original ordinal.
        result = comparison.collect_constructors(pool, [cohort[index]['team_indices']],
            BattleState(*(value[index:index+1] for value in state)),
            tuple(value[index:index+1] for value in capture), catalog, root_ids=[root_id],
            showdown_root=showdown_root, node=node, native_seed_start=native_seed_start+index,
            env_config=env_config, engine_observations=actual, original_ivs=original_ivs)
        rows.extend(result['rows'])
    return dict(schema=comparison.SCHEMA, scope='SAVED_ACTUAL_CONSTRUCTOR_PACKETS_NO_ENGINE_REPLAY',
        source_measurement_binding=measured.get('runtime_binding'), rows=rows,
        summary=comparison.reduce_cohort(rows, root_ids=ids))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('measurement','vocabulary','showdown-root','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--team-pool',type=Path)
    parser.add_argument('--native-result',type=Path,help='Reprocess preserved actual Native lines; no Native call')
    parser.add_argument('--measurement-sha256')
    parser.add_argument('--native-result-sha256')
    parser.add_argument('--node',default='node')
    parser.add_argument('--native-seed-start',type=int)
    args=parser.parse_args(argv)
    if args.out.exists():raise FileExistsError(args.out)
    if args.native_result is not None:
        if args.measurement_sha256 is None or args.native_result_sha256 is None:
            parser.error('saved reprocessing requires both original artifact SHA256s')
    elif args.team_pool is None or args.native_seed_start is None:
        parser.error('Native constructor comparison requires original team pool and declared Native seed start')
    # Rebuild only the same original public tables from immutable source/vocab;
    # no checkpoint archive or champion graph is loaded/executed.
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
    from pokezero.category_vocab import CategoryVocabulary
    from . import public_belief, source_provenance, v4_encoder
    raw=args.measurement.read_bytes();measured=json.loads(raw)
    vocabulary=json.loads(args.vocabulary.read_text())
    vocab=CategoryVocabulary(tuple(vocabulary['tokens']),vocabulary['oov_buckets'],vocabulary['aliases'])
    source=load_gen3_randbat_source_cached(args.showdown_root)
    masks=CHAMPION_CONTRACT['feature_masks']
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),
        load_showdown_dex_cached(args.showdown_root),vocab,
        checkpoint_sha256=vocabulary['checkpoint_sha256'],feature_masks=masks)
    config=champion_env_config(showdown_root=args.showdown_root,node=args.node,vocabulary=vocab,masks=masks)
    if args.native_result is not None:
        result=reprocess_saved(raw,args.native_result.read_bytes(),catalog,env_config=config,
            expected_measurement_sha256=args.measurement_sha256,expected_native_result_sha256=args.native_result_sha256)
    else:
        pool=source_provenance.load_constructor_pool(args.team_pool)
        original_ivs=comparison.load_original_ivs(args.team_pool,pool,
            expected_pool_sha256=measured['pool_receipt']['pool_sha256'])
        result=compare_saved(measured,pool,catalog,
            showdown_root=args.showdown_root,node=args.node,native_seed_start=args.native_seed_start,
            env_config=config,original_ivs=original_ivs)
    result['measurement_file_sha256']=hashlib.sha256(raw).hexdigest()
    with args.out.open('x') as stream:json.dump(result,stream,sort_keys=True,allow_nan=False)
    print(json.dumps(result['summary'],sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
