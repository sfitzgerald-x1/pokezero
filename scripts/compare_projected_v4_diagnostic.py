"""Produce the complete 173-column table; keep decision refusals in its roster."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

if __package__:
    from .compare_projected_v4_encoder import build_runtime
else:
    from compare_projected_v4_encoder import build_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'scratch-checkpoint', 'showdown', 'node', 'collection',
                 'collection-sha256', 'cohort-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8:
        parser.error('CPU thread count1..8 required')
    args.forward = True
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    attempt = dict(status='FAILED', actual_exit=1, full_encoder_admitted=False,
                   actual_captured_wire=False, new_games=0)
    try:
        adapters, config, runtime = build_runtime(args)
        from pokezero.projected_encoder_diagnostic import compare_diagnostic_collection
        result = compare_diagnostic_collection(args.collection,
            expected_collection_sha256=args.collection_sha256,
            expected_cohort_sha256=args.cohort_sha256, adapters=adapters,
            reference_config=config, with_forward=True,
            progress=lambda row: print(json.dumps(dict(event='diagnostic_progress', **row)), flush=True))
        result['runtime'] = runtime
        data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
        with (output/'comparison.json').open('xb') as f:
            f.write(data)
        m = result['metrics']
        fields = ['type', 'column', 'expected_cells', 'compared_cells', 'mismatched_cells',
                  'refused_decisions', 'refused_cells', 'maximum_absolute_error', 'refusal_scope']
        with (output/'column-table.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for family, label in [('numeric_columns', 'numeric_features'),
                                  ('categorical_columns', 'categorical_ids')]:
                for row in m[family]:
                    writer.writerow(dict(type=label, column=row['index'], **{
                        k: row.get(k, '') for k in fields if k not in ('type', 'column')}))
        attempt.update(status=result['status'], actual_exit=0,
            comparison_sha256=hashlib.sha256(data).hexdigest(),
            expected_views=m['expected_views'], processed_views=m['processed_views'],
            compared_views=m['views'], refused_input_views=m['refused_input_views'],
            refused_forward_views=m['refused_forward_views'],
            policy_argmax_agreement=m['policy_argmax_agreement'],
            legal_policy_argmax_agreement=m['legal_policy_argmax_agreement'],
            maximum_value_difference=m['maximum_value_difference'],
            mean_absolute_value_difference=m['mean_absolute_value_difference'],
            refusals_by_line_kind=m['refusals_by_line_kind'])
    except Exception as error:
        # The remote report retains data; stdout never contains raw requests or arrays.
        attempt.update(error_type=type(error).__name__)
    finally:
        attempt['seconds'] = time.monotonic()-start
        with (output/'attempt.json').open('x') as f:
            json.dump(attempt, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
        print(json.dumps(attempt, sort_keys=True, allow_nan=False), flush=True)
    return attempt['actual_exit']


if __name__ == '__main__':
    raise SystemExit(main())
