"""Compare the prespecified A2 TRAIN subset's actual owning wires/references.

Offline only: no simulator, extra games, reference regeneration or head fitting.
Scientific code/results require independent review before execution/admission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

if __package__:
    from .compare_projected_v4_encoder import build_runtime
else:
    from compare_projected_v4_encoder import build_runtime


def atomic_json(path, value):
    """Publish one durable checkpoint; failed serialization retains its predecessor."""
    path = Path(path)
    blob = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def column_table(path, metrics):
    fields = ('type', 'column', 'expected_cells', 'compared_cells', 'mismatched_cells',
        'refused_decisions', 'refused_cells', 'unprocessed_cells', 'maximum_absolute_error',
        'compared_cell_mismatch_rate', 'refusal_scope')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        count = 0
        for family, label in (('numeric_columns', 'numeric_features'),
                               ('categorical_columns', 'categorical_ids')):
            for row in metrics[family]:
                writer.writerow(dict(type=label, column=row['index'], **{
                    key: row.get(key, '') for key in fields if key not in ('type', 'column')}))
                count += 1
        if count != 173:
            raise ValueError('complete 173-column table required')
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'scratch-checkpoint', 'showdown', 'node', 'collection',
                 'collection-sha256', 'selection', 'selection-sha256',
                 'selection-metadata', 'selection-metadata-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8:
        parser.error('CPU thread count 1..8 required')
    args.forward = True
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    attempt = dict(schema='pokezero.B2-actual-saved-wire-attempt.v1', status='FAILED',
        actual_exit=1, new_games=0, simulator_steps=0, full_encoder_admitted=False,
        actual_captured_wire=False, raw_bridge_chunk_bytes_captured=False,
        original_reference_values_compared=False,
        selection_sha256=args.selection_sha256, collection_sha256=args.collection_sha256)
    try:
        from pokezero.actual_saved_wire import (inventory_saved_collection,
            compare_saved_collection, SavedWireJaxAdapter)
        inventory = inventory_saved_collection(args.collection, selection=args.selection,
            expected_selection_sha256=args.selection_sha256,
            selection_metadata=args.selection_metadata,
            expected_selection_metadata_sha256=args.selection_metadata_sha256,
            expected_collection_sha256=args.collection_sha256)
        attempt['actual_captured_wire'] = True
        adapters, _, runtime = build_runtime(args)
        from pokezero.mirror_value_probe import model_digest
        frozen = model_digest(adapters['p1'].model)
        if frozen != inventory['battles'][0]['frozen_model_sha256']:
            raise ValueError('loaded champion differs from actual producer model state')
        runtime['frozen_model_sha256'] = frozen
        adapters = {seat: SavedWireJaxAdapter(adapter, seat) for seat, adapter in adapters.items()}
        result = compare_saved_collection(inventory, adapters=adapters, with_forward=True,
            checkpoint=lambda row: atomic_json(output / 'progress.json', row),
            progress=lambda row: print(json.dumps(dict(event='actual_saved_wire_progress', **row)), flush=True))
        result['runtime'] = runtime
        atomic_json(output / 'comparison.json', result)
        column_table(output / 'column-table.csv', result['metrics'])
        metrics = result['metrics']
        attempt.update(status=result['status'], actual_exit=0,
            original_reference_values_compared=result['original_reference_values_compared'],
            comparison_sha256=hashlib.sha256((output / 'comparison.json').read_bytes()).hexdigest(),
            column_table_sha256=hashlib.sha256((output / 'column-table.csv').read_bytes()).hexdigest(),
            **{key: metrics[key] for key in ('expected_views', 'processed_views', 'views',
                'refused_input_views', 'refused_forward_views', 'model_forward_views',
                'policy_argmax_agreement', 'legal_policy_argmax_agreement',
                'policy_argmax_full_roster_identification_bounds',
                'legal_policy_argmax_full_roster_identification_bounds',
                'maximum_value_difference', 'mean_absolute_value_difference',
                'minimum10000_views_met', 'input_numeric_criteria_met',
                'forward_numeric_criteria_met', 'refusals_by_line_kind')})
    except Exception as error:
        # No raw requests/tensors or potentially private exception text in stdout.
        attempt['error_type'] = type(error).__name__
    finally:
        attempt['seconds'] = time.monotonic() - started
        atomic_json(output / 'attempt.json', attempt)
        print(json.dumps(attempt, sort_keys=True, allow_nan=False), flush=True)
    return attempt['actual_exit']


if __name__ == '__main__':
    raise SystemExit(main())
