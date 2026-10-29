"""Compare the complete fixed corpus using regenerated player references.

Run under a separately attested remote execution owner. This command starts no
game and modifies no original source/reference. A source-derived observation
projection is never labeled captured player wire or full encoder admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

if __package__:
    from .compare_recorded_v4_encoder import snapshot_checkpoint, publish_comparison
else:
    from compare_recorded_v4_encoder import snapshot_checkpoint, publish_comparison


def build_runtime(args):
    """Load one frozen model/catalog and create two independent JAX histories."""
    os.environ['JAX_PLATFORMS'] = 'cpu'
    node = Path(args.node).resolve(strict=True)
    if node.name != 'node' or not node.is_file() or not os.access(node, os.X_OK):
        raise ValueError('bound Node executable required')
    os.environ['PATH'] = str(node.parent) + os.pathsep + os.environ.get('PATH', '')
    if Path(shutil.which('node')).resolve() != node:
        raise ValueError('Dex metadata Node differs from bound runtime')
    from pokezero.mirror_value_probe import load_runtime, CHAMPION_SHA
    from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
    checkpoint = snapshot_checkpoint(args.checkpoint, args.scratch_checkpoint, CHAMPION_SHA)
    model, _, vocab, reference_config, contract = load_runtime(
        args.scratch_checkpoint, args.showdown, str(node), args.threads)
    if contract != CHAMPION_CONTRACT:
        raise ValueError('exact recorded champion contract required')
    import jax
    from pokezero.neural_policy import require_torch
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.recorded_encoder_runner import JaxRecordedAdapter
    from gpu_engine import public_belief, v4_encoder
    torch = require_torch()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision('highest')
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('CPU-only projected comparison required')
    source = load_gen3_randbat_source_cached(args.showdown)
    if source.metadata.source_hash != contract['belief_set_source_hash']:
        raise ValueError('source hash differs from recorded contract')
    catalog = v4_encoder.compile_encoder(public_belief.compile_source(source),
        load_showdown_dex_cached(args.showdown), vocab,
        checkpoint_sha256=CHAMPION_SHA, feature_masks=contract['feature_masks'])
    adapters = {seat: JaxRecordedAdapter(catalog, model=model if args.forward else None,
        torch_module=torch if args.forward else None) for seat in ('p1', 'p2')}
    receipt = dict(checkpoint=checkpoint, contract=contract,
        devices=[str(device) for device in jax.devices()], torch_version=torch.__version__,
        jax_version=jax.__version__, mha_fastpath=False,
        torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
        jax_enable_x64=False, jax_default_matmul_precision='highest', threads=args.threads,
        forwards_requested=args.forward, source_hash=source.metadata.source_hash,
        node_sha256=hashlib.sha256(node.read_bytes()).hexdigest(), independent_player_adapters=2)
    return adapters, reference_config, receipt


def validate_result(result, *, with_forward):
    """Retain complete negative measurements, refuse partial or overclaimed ones."""
    from pokezero.recorded_player_encoder_runner import VISIBILITY
    from pokezero.saved_player_projection import (
        PROJECTION_SCOPE, STRUGGLE_WARNING_POLICY, PINNED_SHOWDOWN_REVISION)
    metrics = result['metrics']
    identity = dict(visibility=VISIBILITY, scope=PROJECTION_SCOPE,
        struggle_warning_policy=STRUGGLE_WARNING_POLICY,
        showdown_revision=PINNED_SHOWDOWN_REVISION)
    for payload, schema, status in (
            (result, 'pokezero.B2-projected-player-collection-comparison.v1',
             'COMPLETE_FIXED_PROJECTED_PLAYER_METRICS_ONLY'),
            (metrics, 'pokezero.B2-projected-player-comparison-metrics.v1',
             'PROJECTED_PLAYER_METRICS_ONLY_REQUIRES_RUNTIME_ATTESTATION')):
        if (payload.get('schema') != schema or payload.get('status') != status
                or any(payload.get(key) != value for key, value in identity.items())
                or any(payload.get(key) is not False for key in (
                    'public_wire_qualified', 'actual_captured_wire', 'captured_player_wire',
                    'owning_stream_bytes_reconstructed', 'original_reference_values_compared',
                    'fullencoder', 'full_encoder_admitted'))
                or any(key.startswith('recorded_') for key in payload)):
            raise ValueError('exact unqualified projected-player report identity required')
    if (any(type(metrics.get(key)) is not bool for key in (
            'projected_input_numeric_criteria_met', 'projected_forward_numeric_criteria_met',
            'projected_comparison_numeric_criteria_met'))
            or type(result.get('new_games')) is not int or result['new_games'] != 0
            or type(result.get('simulator_steps')) is not int or result['simulator_steps'] != 0
            or type(metrics.get('views')) is not int or metrics['views'] <= 0
            or type(metrics.get('expected_views')) is not int
            or metrics['views'] != metrics['expected_views']
            or metrics.get('complete_view_count') is not True):
        raise ValueError('complete unqualified projected-player metrics required')
    if with_forward and (metrics.get('model_forward_complete') is not True
            or type(metrics.get('model_forward_views')) is not int
            or metrics['model_forward_views'] != metrics['views']):
        raise ValueError('all requested paired forwards required')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'scratch-checkpoint', 'showdown', 'node', 'collection',
                 'collection-sha256', 'cohort-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--mode', choices=('inputs-only', 'paired-forward'), required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8:
        parser.error('bounded CPU thread count1..8 required')
    args.forward = args.mode == 'paired-forward'
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = dict(schema='pokezero.B2-projected-player-CPU-attempt.v1', status='FAILED',
        actual_exit=1, new_games=0, simulator_steps=0, public_wire_qualified=False,
        actual_captured_wire=False, full_encoder_admitted=False)
    status = 1
    try:
        adapters, reference_config, runtime = build_runtime(args)
        receipt['runtime'] = runtime
        from pokezero.projected_encoder_collection import compare_projected_collection
        result = compare_projected_collection(args.collection,
            expected_collection_sha256=args.collection_sha256,
            expected_cohort_sha256=args.cohort_sha256, adapters=adapters,
            reference_config=reference_config, with_forward=args.forward)
        validate_result(result, with_forward=args.forward)
        blob = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        publish_comparison(output, blob)
        receipt.update(status='COMPLETE_FIXED_PROJECTED_PLAYER_METRICS_ONLY', actual_exit=0,
            comparison_sha256=hashlib.sha256(blob).hexdigest(), comparison_bytes=len(blob),
            views=result['metrics']['views'])
        status = 0
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error)[:4000])
    finally:
        receipt['seconds'] = time.monotonic() - started
        with (output / 'attempt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
