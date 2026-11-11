"""Model A: value, uncertainty and sibling-ranking heads on the champion's frozen encoder tokens (mc_evaluator_features' shards).

Data: one or more feature directories (features-<w>/{tokens,valid,owner,old_value}.npy, memory-mapped, and index-<w>.jsonl). A child
with tokens is a training position. Its target is the outcomes of the labeller's rollouts whose action and root reply were the
child's (a, b), so the value estimates Q(a, b), what a depth-1 completed-Q consumes.

Heads (pokezero.value_precision_probe.make_value_head), --head:
- hl_gauss (model A): value = attention HL-Gauss over --bins bins on [-1, 1];
- residual: value = the champion's own value on the child (old_value) + delta, delta a scalar attention MLP whose output layer starts
  at zero, so the untrained model is the old head exactly and moves only where the losses give evidence (with a small --lr);
- uncertainty (both): an attention MLP with softplus output (its own --lr-unc).

Losses:
- value: hl_gauss, the HL-Gauss cross-entropy against the mixture of the child's rollout outcomes (each outcome Gaussian-smoothed,
  sigma = 0.75 bin widths); residual, E[(v - z)^2] over the child's rollout outcomes;
- ranking: per root, Q_V(a) = sum_b prior(b) mean_c v(child(a, b, c)) (a terminal child's value is its outcome; the prior is the
  row's opponent root policy renormalized over its support, or 1 for a waiting opponent). On action pairs whose CRN-paired z
  (the paired mean difference d_pair over its SE se_pair, both on the rollouts where neither action is capped) exceeds --rank-c:
  softplus(-sign(z) (Q_V(i) - Q_V(j)) / --rank-tau), weighted by min(|z|, 5). Rows without d_pair are refused
  (gpu_engine.mc_evaluator_pairs backfills older feature directories);
- uncertainty: Huber of u against the stop-gradient E[(v - z)^2] under the child's outcomes.
- Weights: --w-value 1, --w-rank 0.5, --w-unc 0.1.

Split by game: a Foul Play root by ordinal and a self root by battle seed, hashed; --val-fraction of games is validation.
Validation metrics, for model A and for the champion's own value head on identical children (old_value):
- value cross-entropy (hl_gauss only) and MSE against the children's mean outcomes;
- pairwise accuracy on the significant pairs;
- top-1 agreement of argmax Q_V with argmax Q_MC;
- the Monte Carlo value of argmax Q_V minus that of argmax over the policy-free Q_MC.

Early stopping (--early-stop-patience P > 0): training stops after P epochs without a better validation regret, and the saved
model is the best epoch's (epoch 0, the untrained model, included: for the residual head that is the old head).

Output: OUT/model-a.pt (both heads' state_dict, the head kind, the saved epoch and the config), OUT/metrics.json (per epoch),
OUT/config.json.

usage: python -m gpu_engine.mc_evaluator_train --features DIR [DIR ...] --out NEW_DIR [--head hl_gauss|residual] [--epochs 8]
           [--batch-roots 64] [--lr 3e-4] [--lr-unc LR] [--early-stop-patience P] [--bins 51] [--device cuda]
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def game_key(row):
    return f"{row['kind']}:{row['ordinal'] if row['kind'] == 'foulplay' else row['seed']}"


def is_validation(row, fraction, salt='mc-evaluator-a'):
    return int(hashlib.sha256(f'{salt}:{game_key(row)}'.encode()).hexdigest()[:8], 16) / 2**32 < fraction


def load(directories):
    """[(index row, tokens memmap, valid memmap, old_value memmap)] over every shard."""
    roots = []
    for directory in directories:
        for index in sorted(Path(directory).glob('index-*.jsonl')):
            w = index.stem.split('-')[-1];shard = Path(directory) / f'features-{w}'
            if not (shard / 'tokens.npy').exists():continue
            arrays = {n: np.load(shard / f'{n}.npy', mmap_mode='r') for n in ('tokens', 'valid', 'old_value')}
            for line in open(index):
                if line.strip():roots.append((json.loads(line), arrays))
    return roots


def outcome_counts(outcomes):
    """(n_loss, n_draw, n_win) of a list of outcomes in {-1, 0, 1}."""
    o = np.asarray(outcomes, float);return int(np.sum(o < -.5)), int(np.sum(np.abs(o) <= .5)), int(np.sum(o > .5))


class Batch:
    """The children with tokens of a set of roots, flattened, with their targets and per-root ranking structure."""

    def __init__(self, items, torch, device):
        tokens, valid, old, counts, slots = [], [], [], [], [];self.roots = []
        for row, arrays in items:
            base = row['obs_offset'];prior = row['reply_prior'] or [1.]
            prior = np.asarray(prior, float);prior = prior / prior.sum() if prior.sum() > 0 else prior
            children = []
            for c in row['children']:
                outs = row['reply_outcomes'][str(c['a'])][str(c['b'])]
                if c['obs'] is None:children.append(dict(c, slot=None));continue
                slot = len(tokens);tokens.append(np.asarray(arrays['tokens'][base + c['obs']], np.float32))
                valid.append(np.asarray(arrays['valid'][base + c['obs']], bool));old.append(float(arrays['old_value'][base + c['obs']]))
                counts.append(outcome_counts(outs));children.append(dict(c, slot=slot))
            self.roots.append(dict(row=row, prior=prior, children=children))
        self.n = len(tokens)
        if not self.n:return
        self.tokens = torch.tensor(np.stack(tokens), device=device);self.valid = torch.tensor(np.stack(valid), device=device)
        self.old = torch.tensor(old, device=device, dtype=torch.float32)
        self.counts = torch.tensor(counts, device=device, dtype=torch.float32)   # [n, 3]: loss, draw, win


def hl_gauss_rows(torch, values, bins, sigma_ratio=.75, device='cpu'):
    """[len(values), bins] HL-Gauss target rows of scalar values on [-1, 1]."""
    edges = torch.linspace(-1, 1, bins + 1, device=device);sigma = sigma_ratio * 2 / bins
    v = torch.tensor(values, device=device, dtype=torch.float32)[:, None]
    cdf = 0.5 * (1 + torch.erf((edges[None, :] - v) / (sigma * math.sqrt(2))))
    p = cdf[:, 1:] - cdf[:, :-1];return p / p.sum(-1, keepdim=True)


def q_values(root, v, torch):
    """Q_V per action of one root under the head's child values v [n] (a terminal child: its outcome)."""
    row = root['row'];support = row['replies'];prior = root['prior'];draws = row['draws'];out = []
    for a in row['actions']:
        total = 0.
        for j, b in enumerate(support):
            vals = [v[c['slot']] if c['slot'] is not None else torch.tensor(float(c['terminal']), device=v.device) for c in root['children']
                    if c['a'] == a and c['b'] == b]
            total = total + float(prior[j]) * torch.stack(vals).mean()
        out.append(total)
    return torch.stack(out)


def significant_pairs(row, c):
    """[(i, j, sign, weight)] of the action pairs whose CRN-paired z exceeds c: the paired mean difference over the rollouts where
    neither action is capped, over its standard error on those same rollouts (d_pair / se_pair; a row without d_pair is refused,
    since the marginal means' difference over the paired SE is not that test under selective censoring)."""
    if 'd_pair' not in row:raise ValueError('feature row without d_pair: run gpu_engine.mc_evaluator_pairs (or re-extract) first')
    d, se = row['d_pair'], row['se_pair'];out = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if d[i][j] is None or se[i][j] is None or se[i][j] <= 0:continue
            z = d[i][j] / se[i][j]
            if abs(z) > c:out.append((i, j, math.copysign(1., z), min(abs(z), 5.)))
    return out


def make_heads(kind, width, bins, torch):
    """(value head, uncertainty head). 'hl_gauss': model A's HL-Gauss value head. 'residual': a scalar attention MLP whose output
    layer starts at zero, read as a correction delta on the champion's own value (v = old_value + delta), so the untrained model is
    the old head exactly."""
    from pokezero.value_precision_probe import make_value_head
    if kind == 'hl_gauss':value_head = make_value_head('attention_hl_gauss', width, bins=bins)
    elif kind == 'residual':
        value_head = make_value_head('attention_mlp', width);last = value_head.network[-1]
        with torch.no_grad():last.weight.zero_();last.bias.zero_()
    else:raise ValueError(f'unknown head kind {kind}')
    return value_head, make_value_head('attention_mlp', width)


def predict(kind, value_head, batch):
    """(child values [n], HL-Gauss logits or None) of a batch under the value head of this kind."""
    if kind == 'residual':
        delta, _ = value_head(batch.tokens, batch.valid);return batch.old + delta, None
    return value_head(batch.tokens, batch.valid)


def evaluate(heads, roots, torch, device, args, targets):
    """Validation metrics for model A and the old head (module docstring)."""
    value_head, _ = heads;stats = {k: [] for k in ('ce', 'mse_new', 'mse_old', 'pair_new', 'pair_old', 'top_new', 'top_old', 'regret_new', 'regret_old')}
    with torch.no_grad():
        for start in range(0, len(roots), args.batch_roots):
            batch = Batch(roots[start:start + args.batch_roots], torch, device)
            if not batch.n:continue
            mean, logits = predict(args.head, value_head, batch);target = batch.counts @ targets / batch.counts.sum(-1, keepdim=True).clamp(min=1)
            has = batch.counts.sum(-1) > 0
            if logits is not None:stats['ce'].append(float((-(target * torch.log_softmax(logits, -1)).sum(-1))[has].mean()) if has.any() else math.nan)
            z = (batch.counts[:, 2] - batch.counts[:, 0]) / batch.counts.sum(-1).clamp(min=1)
            stats['mse_new'].append(float(((mean - z) ** 2)[has].mean()) if has.any() else math.nan)
            stats['mse_old'].append(float(((batch.old - z) ** 2)[has].mean()) if has.any() else math.nan)
            for root in batch.roots:
                row = root['row'];qmc = row['q_mc']
                if any(x is None for x in qmc):continue
                for name, v in (('new', mean), ('old', batch.old)):
                    qv = q_values(root, v, torch).cpu().numpy();best = int(np.argmax(qmc));pick = int(np.argmax(qv))
                    stats[f'top_{name}'].append(float(pick == best));stats[f'regret_{name}'].append(float(qmc[pick] - qmc[best]))
                    for i, j, sign, w in significant_pairs(row, args.rank_c):stats[f'pair_{name}'].append(float(np.sign(qv[i] - qv[j]) == sign))
    return {k: (None if not x else round(float(np.nanmean(x)), 5)) for k, x in stats.items()} | dict(
        pairs=len(stats['pair_new']), roots=len(stats['top_new']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', nargs='+', required=True);parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=8);parser.add_argument('--batch-roots', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4);parser.add_argument('--bins', type=int, default=51)
    parser.add_argument('--w-value', type=float, default=1.);parser.add_argument('--w-rank', type=float, default=.5)
    parser.add_argument('--w-unc', type=float, default=.1);parser.add_argument('--rank-c', type=float, default=2.)
    parser.add_argument('--rank-tau', type=float, default=.02);parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=0);parser.add_argument('--device', default='cuda')
    parser.add_argument('--head', choices=('hl_gauss', 'residual'), default='hl_gauss')
    parser.add_argument('--lr-unc', type=float, help='the uncertainty head\'s learning rate (default: --lr)')
    parser.add_argument('--early-stop-patience', type=int, default=0, help='stop after this many epochs without a better validation '
                        'regret and save the best epoch (epoch 0 included); 0: train every epoch and save the last')
    args = parser.parse_args(argv)
    import copy
    import torch
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False);(out / 'config.json').write_text(json.dumps(vars(args), indent=1) + '\n')
    torch.manual_seed(args.seed);rng = np.random.default_rng(args.seed);device = torch.device(args.device)
    roots = load(args.features);train = [r for r in roots if not is_validation(r[0], args.val_fraction)]
    val = [r for r in roots if is_validation(r[0], args.val_fraction)]
    width = int(roots[0][1]['tokens'].shape[-1])
    value_head, unc_head = make_heads(args.head, width, args.bins, torch);value_head.to(device);unc_head.to(device)
    optimizer = torch.optim.AdamW([dict(params=list(value_head.parameters()), lr=args.lr),
                                   dict(params=list(unc_head.parameters()), lr=args.lr if args.lr_unc is None else args.lr_unc)], weight_decay=1e-4)
    targets = hl_gauss_rows(torch, [-1., 0., 1.], args.bins, device=device)   # rows for a loss, a draw, a win
    metrics = [dict(epoch=0, train_roots=len(train), val_roots=len(val), val=evaluate((value_head, unc_head), val, torch, device, args, targets))]
    print(json.dumps(metrics[-1]), flush=True)
    best = dict(epoch=0, regret=metrics[0]['val']['regret_new'], state=copy.deepcopy((value_head.state_dict(), unc_head.state_dict())))
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train));losses = []
        for start in range(0, len(order), args.batch_roots):
            batch = Batch([train[i] for i in order[start:start + args.batch_roots]], torch, device)
            if not batch.n:continue
            mean, logits = predict(args.head, value_head, batch);has = batch.counts.sum(-1) > 0
            n = batch.counts.sum(-1).clamp(min=1);ez = (batch.counts[:, 2] - batch.counts[:, 0]) / n;ez2 = (batch.counts[:, 2] + batch.counts[:, 0]) / n
            if logits is not None:
                target = batch.counts @ targets / batch.counts.sum(-1, keepdim=True).clamp(min=1)
                value_loss = (-(target * torch.log_softmax(logits, -1)).sum(-1))[has].mean() if has.any() else mean.sum() * 0
            else:value_loss = (mean ** 2 - 2 * mean * ez + ez2)[has].mean() if has.any() else mean.sum() * 0   # E[(v - z)^2] over the child's outcomes
            u, _ = unc_head(batch.tokens, batch.valid);u = torch.nn.functional.softplus(u)
            m = mean.detach();unc_target = m ** 2 - 2 * m * ez + ez2
            unc_loss = torch.nn.functional.huber_loss(u[has], unc_target[has]) if has.any() else u.sum() * 0
            rank_terms = []
            for root in batch.roots:
                pairs = significant_pairs(root['row'], args.rank_c)
                if not pairs:continue
                qv = q_values(root, mean, torch)
                for i, j, sign, w in pairs:rank_terms.append(w * torch.nn.functional.softplus(-sign * (qv[i] - qv[j]) / args.rank_tau))
            rank_loss = torch.stack(rank_terms).mean() if rank_terms else mean.sum() * 0
            loss = args.w_value * value_loss + args.w_rank * rank_loss + args.w_unc * unc_loss
            optimizer.zero_grad();loss.backward();optimizer.step()
            losses.append([float(value_loss), float(rank_loss), float(unc_loss)])
        metrics.append(dict(epoch=epoch, train_loss=np.mean(losses, 0).round(5).tolist() if losses else None,
                            val=evaluate((value_head, unc_head), val, torch, device, args, targets)))
        print(json.dumps(metrics[-1]), flush=True);(out / 'metrics.json').write_text(json.dumps(metrics, indent=1) + '\n')
        regret = metrics[-1]['val']['regret_new']
        if regret is not None and (best['regret'] is None or regret > best['regret']):
            best = dict(epoch=epoch, regret=regret, state=copy.deepcopy((value_head.state_dict(), unc_head.state_dict())))
        elif args.early_stop_patience and epoch - best['epoch'] >= args.early_stop_patience:break
    if args.early_stop_patience:value_head.load_state_dict(best['state'][0]);unc_head.load_state_dict(best['state'][1])
    saved_epoch = best['epoch'] if args.early_stop_patience else metrics[-1]['epoch']
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1) + '\n')
    torch.save(dict(value_head=value_head.state_dict(), uncertainty_head=unc_head.state_dict(), bins=args.bins, width=width, head=args.head,
                    epoch=saved_epoch, features=args.features, config=vars(args)), out / 'model-a.pt')
    print(json.dumps(dict(event='SAVED', head=args.head, epoch=saved_epoch, val_regret=metrics[saved_epoch]['val']['regret_new'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
