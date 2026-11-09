"""The definitive RR's reply prior: a reply model's logits on each world's stored peer-view inputs.

usage: python scripts/depthn_rr_reply_prior.py --peer-inputs RR/shard-*/peer-inputs-*.npz --out REPLY_PRIOR.jsonl \
           --champion-archive A --champion-archive-sha256 H --champion-checkpoint-sha256 H \
           --reply-archive B --reply-archive-sha256 H --reply-checkpoint-sha256 H [--precision bf16]

The re-pricer (scripts/depthn_rr_reprice.py, peer-inputs sidecar schema pokezero.depthn-rr-reprice.peer-inputs.v1) keeps,
per constructed world of every decided root, the champion adapter's 23-token view slice of the opponent's depth-0 view
(numeric_features [K,23,132], categorical_ids [K,23,41], token_type_ids [K,23], attention_mask [K,23], history_mask [K,1]),
its legal mask, the recorded root_peer_logits and the re-pricer's own self_check. Each root's worlds are evaluated as one
batch in sidecar order, as the re-pricer's self-check stacks them, through ChampionEvaluator's own adapter under jax.jit
(the slice is padded back to the V2 29-token layout; the padding is never read).

1. Champion self-check: on every world whose self_check is not -1 (priced, opponent requested), the champion archive's
   logits must equal root_peer_logits bit for bit, so the stored inputs are the ones the search priced. Any failure
   writes nothing (exit 1) and lists the failures with their max |difference|.
2. The reply model (π_FP-red2's champion_evaluator export) on the same batches: one JSONL row per world whose opponent is
   requested, {ordinal, decision_index, world, logits[9]} (depthn_rr_score.py --reply-prior's format; a waiting world
   needs none). OUT.summary.json records the archives, the precision, the self-check counts and every sidecar's header.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

SCHEMA = 'v2s.depthn-rr-reply-prior.v1'
SIDECAR_SCHEMA = 'pokezero.depthn-rr-reprice.peer-inputs.v1'
INPUT_KEYS = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask')
V2_TOKENS, VIEW_TOKENS = 29, 23


def read_sidecar(path):
    """(header dict, arrays) of one peer-inputs sidecar; ValueError on another schema or a misshapen array."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != 'header'};header = json.loads(str(z['header']))
    if header.get('schema') != SIDECAR_SCHEMA:raise ValueError(f'{path}: sidecar schema {header.get("schema")!r}, not {SIDECAR_SCHEMA}')
    k = len(arrays['world'])
    shapes = dict(numeric_features=(k, VIEW_TOKENS, 132), categorical_ids=(k, VIEW_TOKENS, 41), token_type_ids=(k, VIEW_TOKENS),
                  attention_mask=(k, VIEW_TOKENS), history_mask=(k, 1), legal=(k, 9), root_peer_logits=(k, 9))
    for name, shape in shapes.items():
        if arrays[name].shape != shape:raise ValueError(f'{path}: {name} {arrays[name].shape}, expected {shape}')
    return header, arrays


def v2_inputs(arrays, rows):
    """The V2 encoder layout [n,1,29,...] for sidecar rows `rows` (tokens 23..28 zero: ChampionEvaluator.view drops them)."""
    out = {}
    for key in INPUT_KEYS:
        x = arrays[key][rows]
        if key == 'history_mask':out[key] = x.reshape(len(rows), 1);continue
        pad = [(0, 0), (0, V2_TOKENS - VIEW_TOKENS)] + [(0, 0)] * (x.ndim - 2)
        out[key] = np.pad(x, pad)[:, None]
    return out


def roots(arrays):
    """Sidecar row indices grouped by (ordinal, decision_index), each group in sidecar (world) order."""
    groups = collections.OrderedDict()
    for i, key in enumerate(zip(arrays['ordinal'].tolist(), arrays['decision_index'].tolist())):groups.setdefault(key, []).append(i)
    return groups


def reply_prior(sidecars, champion_forward, reply_forward):
    """(JSONL rows, summary) over (header, arrays) pairs; champion_forward / reply_forward map V2 inputs -> logits [n, 9] f32."""
    rows, checked, equal, failures, headers, seen = [], 0, 0, [], [], set()
    for header, arrays in sidecars:
        headers.append(header)
        for key, members in roots(arrays).items():
            if key in seen:raise ValueError(f'root {key} appears in more than one sidecar')
            seen.add(key);members = np.asarray(members);inputs = v2_inputs(arrays, members)
            champion = np.asarray(champion_forward(inputs), np.float32);reply = np.asarray(reply_forward(inputs), np.float32)
            for j, i in enumerate(members.tolist()):
                if int(arrays['self_check'][i]) != -1:
                    checked += 1
                    if np.array_equal(champion[j], arrays['root_peer_logits'][i]):equal += 1
                    else:failures.append(dict(ordinal=key[0], decision_index=key[1], world=int(arrays['world'][i]),
                                              max_abs_diff=float(np.max(np.abs(champion[j].astype(np.float64) - arrays['root_peer_logits'][i])))))
                if bool(arrays['opponent_requested'][i]):
                    logits = [float(x) for x in reply[j]]
                    if not all(math.isfinite(x) for x in logits):raise ValueError(f'non-finite reply logits at {key}, world {int(arrays["world"][i])}')
                    rows.append(dict(ordinal=int(key[0]), decision_index=int(key[1]), world=int(arrays['world'][i]), logits=logits))
    summary = dict(schema=SCHEMA, roots=len(seen), reply_rows=len(rows), self_check=dict(checked=checked, equal=equal, failed=len(failures),
                   passed=checked > 0 and not failures, failures=failures[:50]), sidecar_headers=headers)
    return rows, summary


def jax_forward(evaluator):
    """The re-pricer's self-check program: the evaluator's own adapter under jax.jit, logits as float32 numpy."""
    import jax
    program = jax.jit(lambda p, x: evaluator(p, x))
    return lambda inputs: np.asarray(jax.device_get(program(evaluator.params, inputs)).policy_logits, np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--peer-inputs', nargs='+', required=True);p.add_argument('--out', required=True)
    for who in ('champion', 'reply'):
        p.add_argument(f'--{who}-archive', required=True);p.add_argument(f'--{who}-archive-sha256', required=True)
        p.add_argument(f'--{who}-checkpoint-sha256', required=True)
    p.add_argument('--precision', choices=('bf16', 'f32'), default='bf16', help="both archives' forward precision (the live champion's is bf16)")
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():p.error(f'{out} exists')
    root = Path(__file__).resolve().parents[1]
    for path in (str(root), str(root / 'src')):
        if path not in sys.path:sys.path.insert(0, path)
    from gpu_engine import champion_evaluator as ce
    precision = ce.PRECISIONS[args.precision]
    champion = ce.load(args.champion_archive, expected_archive_sha256=args.champion_archive_sha256,
                       expected_checkpoint_sha256=args.champion_checkpoint_sha256, precision=precision)
    ce.check_checkpoint(champion)
    reply = ce.load(args.reply_archive, expected_archive_sha256=args.reply_archive_sha256, expected_checkpoint_sha256=args.reply_checkpoint_sha256,
                    precision=precision)
    rows, summary = reply_prior((read_sidecar(x) for x in sorted(args.peer_inputs)), jax_forward(champion), jax_forward(reply))
    summary.update(peer_inputs=sorted(str(Path(x).resolve()) for x in args.peer_inputs), precision=args.precision,
                   champion=dict(archive=args.champion_archive, archive_sha256=args.champion_archive_sha256, checkpoint_sha256=args.champion_checkpoint_sha256),
                   reply=dict(archive=args.reply_archive, archive_sha256=args.reply_archive_sha256, checkpoint_sha256=args.reply_checkpoint_sha256,
                              marker=reply.meta.get('extra')))
    Path(str(out) + '.summary.json').write_text(json.dumps(summary, indent=1, default=repr) + '\n')
    if not summary['self_check']['passed']:
        print(json.dumps(dict(event='SELF_CHECK_FAILED', **{k: v for k, v in summary['self_check'].items() if k != 'failures'})));return 1
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps(dict(event='WRITTEN', out=str(out.resolve()), roots=summary['roots'], reply_rows=summary['reply_rows'],
                          self_check={k: v for k, v in summary['self_check'].items() if k != 'failures'})))
    return 0


if __name__ == '__main__':sys.exit(main())
