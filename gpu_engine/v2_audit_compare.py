"""Compare a V2 audit block (compressed row arrays) with file-pipeline v5 caches.

Rows are keyed by (stream, frame, owner). For every field of
`v5_batch_to_torch(cache, training=True)` reports rows equal and, for floats,
the maximum absolute difference. Use on blocks whose games are identical in
both pipelines (same declaration, same actor), e.g. iteration 0. CPU only.

All paths are command-line arguments.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, help='audit-block-*.npz written by v2_fused')
    parser.add_argument('--caches', required=True, help='glob of reference v5 cache directories')
    parser.add_argument('--checkpoint', required=True, help='a v5/V2 checkpoint of the same model (for the model config)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    import torch
    from pokezero import v5_data
    from pokezero.v5_loss import v5_batch_to_torch, INTEGER_FIELDS
    from gpu_engine.v2_bench_trainer import configurations
    model, _, _, _, _ = configurations(args.checkpoint)
    ours = dict(np.load(args.audit))
    keys = np.stack([ours['streams'], ours['frames'], ours['owners']], axis=1).astype(np.int64)
    lo, hi = int(keys[:, 0].min()), int(keys[:, 0].max())
    index = {tuple(k): i for i, k in enumerate(keys)}
    fields = {};matched = 0;caches = []
    for path in sorted(glob.glob(args.caches)):
        batch = next(v5_data.VerifiedCache(Path(path)).batches(batch_size=1 << 30))
        ref_keys = np.asarray([(a['stream'], a['frame'], a['owner']) for a in batch.v5_columns['addresses']], np.int64)
        inside = (ref_keys[:, 0] >= lo) & (ref_keys[:, 0] <= hi)
        if not inside.any():continue
        caches.append(path)
        reference = v5_batch_to_torch(batch, model, training=True)
        r = np.flatnonzero(inside);o = np.asarray([index.get(tuple(k), -1) for k in ref_keys[r]])
        present = o >= 0;r, o = r[present], o[present];matched += len(r)
        for name, tensor in reference.items():
            if name in ('schema', 'observation_schema', 'category_vocabulary_sha256', 'label_vocabulary_sha256', 'terminal_status'):continue
            if name not in ours:continue
            want = tensor.numpy()[r];got = ours[name][o]
            if name in INTEGER_FIELDS:got = got.astype(np.int64)
            eq = np.all((got == want).reshape(len(r), -1), axis=1) if len(r) else np.zeros(0, bool)
            entry = fields.setdefault(name, dict(rows=0, rows_equal=0, max_abs_diff=0.0))
            entry['rows'] += int(len(r));entry['rows_equal'] += int(eq.sum())
            if want.dtype.kind == 'f' and len(r):
                entry['max_abs_diff'] = max(entry['max_abs_diff'], float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))))
    report = dict(audit=args.audit, audit_rows=int(len(keys)), reference_caches=caches, matched_rows=int(matched),
        unmatched_audit_rows=int(len(keys) - matched), fields=fields,
        all_integer_fields_equal=all(v['rows'] == v['rows_equal'] for k, v in fields.items() if k in INTEGER_FIELDS or 'mask' in k))
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    (out / (Path(args.audit).stem + '-compare.json')).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(matched_rows=report['matched_rows'], unmatched=report['unmatched_audit_rows'],
        integers_equal=report['all_integer_fields_equal'],
        worst=max(((k, v['max_abs_diff']) for k, v in fields.items()), key=lambda x: x[1], default=None))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
