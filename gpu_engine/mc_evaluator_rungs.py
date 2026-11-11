"""Model A on the MC ladder: rung files of the depth-1 completed-Q choice under model A (D1A) and under the champion's own value head
on the identical children (D1old), from mc_evaluator_features' shards of the gate bank.

Per root: Q(a) = sum_b prior(b) mean_c v(child(a, b, c)), exactly the trainer's q_values (a terminal child's value is its
outcome). The choice is argmax Q over the root's actions, lowest index on ties. D1old uses the old_value stored per observation, so
the pair (D1A, D1old) differs only in the leaf value. Rows are mc_ladder_score rung rows: the root's identity (Foul Play: ordinal
and target; self: seed, ours and target), action, actions and q.

Without --model only D1old is written (the old head's rung needs no trained model: a check of the features against the ladder).
A residual model (mc_evaluator_train --head residual) gives D1A's child value as old_value + delta, the trainer's own predict().
With a model, every scored root's game (its kind and battle seed) must be absent from the model's training features (the index rows
of the directories it records); any overlap is refused, since a model that saw a gate game would be scored against the old head
unfairly.

usage: python -m gpu_engine.mc_evaluator_rungs --features DIR [DIR ...] [--model OUT/model-a.pt] --out NEW_DIR [--device cuda]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', nargs='+', required=True);parser.add_argument('--model')
    parser.add_argument('--out', required=True);parser.add_argument('--device', default='cuda');parser.add_argument('--batch-roots', type=int, default=64)
    args = parser.parse_args(argv)
    import torch
    from gpu_engine import mc_evaluator_train as T
    device = torch.device(args.device);head = None;kind = None
    if args.model:
        saved = torch.load(args.model, map_location=args.device);kind = saved.get('head', 'hl_gauss')
        head, _ = T.make_heads(kind, saved['width'], saved['bins'], torch);head.to(device);head.load_state_dict(saved['value_head']);head.eval()
    roots = T.load(args.features)
    if head is not None:
        seen = {(r['kind'], r['seed']) for d in saved['features'] for p in sorted(Path(d).glob('index-*.jsonl')) for r in map(json.loads, filter(str.strip, open(p)))}
        overlap = sorted({(r['kind'], r['seed']) for r, _ in roots} & seen)
        if not seen or overlap:raise SystemExit(f"the model's training games must be readable and disjoint from the scored roots' ({len(overlap)} shared, e.g. {overlap[:3]})")
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False);rows = dict(D1A=[], D1old=[]) if head else dict(D1old=[])
    with torch.no_grad():
        for start in range(0, len(roots), args.batch_roots):
            batch = T.Batch(roots[start:start + args.batch_roots], torch, device)
            mean = T.predict(kind, head, batch)[0] if head is not None and batch.n else None
            for root in batch.roots:
                row = root['row'];identity = (dict(kind='foulplay', ordinal=row['ordinal'], target=row['target']) if row['kind'] == 'foulplay'
                                              else dict(kind='self', seed=row['seed'], ours=row['ours'], target=row['target']))
                for name, v in ((('D1A', mean),) if head is not None else ()) + (('D1old', None if not batch.n else batch.old),):
                    if v is None and any(c['slot'] is not None for c in root['children']):continue
                    q = T.q_values(root, v if v is not None else torch.zeros(0, device=device), torch).cpu().numpy().astype(float)
                    action = int(row['actions'][int(np.argmax(q))])
                    rows[name].append(dict(identity, action=action, actions=row['actions'], q=[round(float(x), 6) for x in q]))
    for name, value in rows.items():(out / f'rung-{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in value))
    (out / 'summary.json').write_text(json.dumps(dict(model=args.model, head=kind, roots=len(roots), written={k: len(v) for k, v in rows.items()},
                                                      differ=sum(a['action'] != b['action'] for a, b in zip(rows['D1A'], rows['D1old'])) if head else None)) + '\n')
    print(json.dumps(dict(event='DONE', roots=len(roots), written={k: len(v) for k, v in rows.items()})), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
