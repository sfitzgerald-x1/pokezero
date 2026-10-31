"""Invoke the reviewed constructor comparator on actual saved GPU packets.

No state constructor/transition, checkpoint or model is called. JSON arrays are
decoded from the original initial packet. Missing packets stay UNKNOWN. Native
Showdown execution occurs only when the explicit command is run by its owner.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import finite_stream_comparison as comparison
from .state import BattleState


def compare_saved(measured, pool, catalog, *, showdown_root, node, native_seed_start, env_config=None):
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
    observations = measured.get('initial_owner_observations')
    rows = []
    for index, root_id in enumerate(ids):
        actual = [None, None]
        if isinstance(observations, list) and len(observations) == 2:
            for side, owner in enumerate(observations):
                # Captured false admission is an UNKNOWN column view. Do not
                # compare zero-masked or stale tensors as scientific mismatches.
                admitted = np.asarray(owner.get('admitted', []))
                if admitted.shape != (len(ids),) or admitted.dtype != np.bool_ or not admitted[index]:
                    continue
                try:
                    actual[side] = {name:np.asarray(owner[name], dtype=
                        np.float32 if name == 'numeric_features' else
                        np.bool_ if name.endswith('mask') else np.int32)[index:index+1]
                        for name in comparison.SHAPES}
                except (KeyError, ValueError, TypeError):
                    actual[side] = None
        # B=1 slicing preserves the actual packet and allows each refused owner
        # to remain absent independently. Native seeds retain original ordinal.
        result = comparison.collect_constructors(pool, [cohort[index]['team_indices']],
            BattleState(*(value[index:index+1] for value in state)),
            tuple(value[index:index+1] for value in capture), catalog, root_ids=[root_id],
            showdown_root=showdown_root, node=node, native_seed_start=native_seed_start+index,
            env_config=env_config, engine_observations=actual)
        rows.extend(result['rows'])
    return dict(schema=comparison.SCHEMA, scope='SAVED_ACTUAL_CONSTRUCTOR_PACKETS_NO_ENGINE_REPLAY',
        source_measurement_binding=measured.get('runtime_binding'), rows=rows,
        summary=comparison.reduce_cohort(rows, root_ids=ids))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('measurement','team-pool','vocabulary','showdown-root','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--node',default='node')
    parser.add_argument('--native-seed-start',required=True,type=int)
    args=parser.parse_args(argv)
    if args.out.exists():raise FileExistsError(args.out)
    # Rebuild only the same original public tables from immutable source/vocab;
    # no checkpoint archive or champion graph is loaded/executed.
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.observation import ObservationFeatureMasks
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
    config=comparison.LocalShowdownConfig(showdown_root=args.showdown_root,node_binary=args.node,
        category_vocab=vocab,feature_masks=ObservationFeatureMasks(**dict(masks)))
    result=compare_saved(measured,source_provenance.load_constructor_pool(args.team_pool),catalog,
        showdown_root=args.showdown_root,node=args.node,native_seed_start=args.native_seed_start,
        env_config=config)
    result['measurement_file_sha256']=hashlib.sha256(raw).hexdigest()
    with args.out.open('x') as stream:json.dump(result,stream,sort_keys=True,allow_nan=False)
    print(json.dumps(result['summary'],sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
