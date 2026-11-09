"""Foul Play reply probes on fp_reply_data --extras shards: 9375's own opponent head, and the option C behavior-summary probe.

usage: python -m gpu_engine.fp_reply_probe DATA_DIR OUT_DIR --base PT --pi-fp PT [--pi-fp-data DIR] [--device cuda] [--epochs 30]
       [--seed 0]

Splits are fp_reply_data.split_of (battle seed % 10: 0 test, 1 validation, else train); intervals are fp_reply_train's 90% bootstrap
clustered on battle seed.

1. opponent_action_head: 9375's reply head reads OUR seat's observation (window 1) and outputs logits over the opponent's own action
   indices; it is scored with Foul Play's legal mask, raw and recalibrated (temperature and switch bias fitted on train, as for π), on
   the rows where our seat is requested in the same round (otherwise there is no our-view observation). Compared on those rows with
   uniform, π, recalibrated π and π_FP.
2. Option C: a GRU over Foul Play's own past decisions in the game (per step: the chosen action class, which classes were available,
   both actives' HP, forced) conditions π_FP through one zero-initialised linear head: three class biases (damaging move, status
   move, switch) and a log-temperature, logits' = log π_FP / T + b[class]. At initialisation it is π_FP exactly (gate 3). Gates:
   (1) test NLL minus π_FP's, upper bound < 0; (2) the improvement late (turn 18+) minus early (turns 1-3), upper bound < 0, and the
   turn-1 improvement not worse than 0 (lower bound <= 0).
   With --pi-fp-data (the dataset π_FP was trained on) the probe fits only on games π_FP did not train on: every non-test game
   whose ordinal is absent from that dataset's train split, divided by seed ((seed // 10) % 4 == 0 for early stopping, the rest to
   fit). Otherwise it fits on train and early-stops on validation, which overstates π_FP wherever π_FP fit its own training games.
   Control: the same head with no history (one temperature and three class biases, LBFGS on the probe's fit rows). probe - static is
   what the game history adds beyond recalibrating π_FP; reported with intervals, not a gate.
"""
import glob
import argparse
import json
import time
from pathlib import Path

import numpy as np

from gpu_engine import fp_reply_data as data_mod
from gpu_engine import fp_reply_train as train_mod

CLASSES = 3
FEATURES = 2 * CLASSES + 3   # chosen class one-hot, available classes, own HP, opponent HP, forced
LATE, EARLY = (18, 10**9), (1, 3)


def cluster_bootstrap(values, clusters, masks, combine, draws=train_mod.BOOTSTRAP_DRAWS, seed=0):
    """90% interval of combine(*means) where means[i] is the mean of `values` over rows in masks[i], resampling clusters with
    replacement; None below MIN_CLUSTERS clusters."""
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = [np.bincount(inverse, np.where(m, values, 0.), len(unique)) for m in masks]
    counts = [np.bincount(inverse, m.astype(float), len(unique)) for m in masks]
    point = combine(*[s.sum() / max(c.sum(), 1e-12) for s, c in zip(sums, counts)])
    if len(unique) < train_mod.MIN_CLUSTERS:return dict(mean=float(point), interval_90=None, seeds=int(len(unique)))
    rng = np.random.default_rng(seed);stats = []
    for _ in range(draws):
        pick = rng.integers(0, len(unique), len(unique))
        stats.append(combine(*[s[pick].sum() / max(c[pick].sum(), 1e-12) for s, c in zip(sums, counts)]))
    return dict(mean=float(point), interval_90=[float(q) for q in np.quantile(stats, [.05, .95])], seeds=int(len(unique)))


def seen_ordinals(data_dir):
    """Game ordinals in the train split of a fp_reply_data dataset (the games a π_FP trained on it has fit)."""
    seen = set()
    for path in glob.glob(str(Path(data_dir) / 'shard-*.npz')):
        with np.load(path) as shard:
            if 'ordinal' in shard.files and len(shard['ordinal']):
                seen.update(shard['ordinal'][data_mod.split_of(shard['seed']) == 'train'].tolist())
    if not seen:raise SystemExit(f'{data_dir} has no train-split rows')
    return seen


def fit_rows(split, ordinal, seed, seen=None):
    """(probe fit rows, probe early-stopping rows): train and validation, or with `seen` the non-test rows of games outside it,
    split by (seed // 10) % 4 == 0 so both seats of a battle seed land on one side."""
    if seen is None:return np.flatnonzero(split == 'train'), np.flatnonzero(split == 'val')
    pool = (split != 'test') & ~np.isin(ordinal, np.fromiter(seen, np.int64, len(seen)));inner = (seed // 10) % 4 == 0
    return np.flatnonzero(pool & ~inner), np.flatnonzero(pool & inner)


def masked_softmax(logits, legal):
    z = np.where(legal, logits, -np.inf);z = z - z.max(1, keepdims=True);e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def step_features(arrays):
    """Per row: the features of that decision, fed to the GRU at the NEXT decision of the same game."""
    n = len(arrays['label']);x = np.zeros((n, FEATURES), np.float32)
    x[np.arange(n), arrays['label_class'].clip(0, CLASSES - 1)] = 1.
    for c in range(CLASSES):x[:, CLASSES + c] = (arrays['candidate_class'] == c).any(1)
    x[:, 2 * CLASSES] = np.nan_to_num(arrays['own_hp']);x[:, 2 * CLASSES + 1] = np.nan_to_num(arrays['opp_hp'])
    x[:, 2 * CLASSES + 2] = arrays['forced']
    return x


def game_sequences(ordinal, rows):
    """Contiguous runs of `rows` (sorted, one game each) as index arrays, in decision order."""
    rows = np.sort(rows);breaks = np.flatnonzero(np.diff(ordinal[rows]) != 0) + 1
    return np.split(rows, breaks)


def build_probe(torch, hidden=32):
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(FEATURES, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden, CLASSES + 1)
            torch.nn.init.zeros_(self.head.weight);torch.nn.init.zeros_(self.head.bias)

        def forward(self, features, log_pi, classes, legal):
            """features [G, L, F] (decision s's features at s, shifted inside), log_pi/classes/legal [G, L, 9]."""
            shifted = torch.cat([torch.zeros_like(features[:, :1]), features[:, :-1]], 1)   # decision t sees only t' < t
            out = self.head(self.gru(shifted)[0])
            bias = out[..., :CLASSES];inv_temperature = torch.exp(-out[..., CLASSES:])
            per_action = torch.gather(bias, -1, classes.clamp(min=0).long())
            logits = log_pi * inv_temperature + per_action
            return logits.masked_fill(~legal, -1e9)
    return Probe()


def static_fit(logp, classes, legal, labels, rows, torch):
    """The probe's head without history: logits = log π_FP / T + b[class], T and b fitted by LBFGS on `rows`. Returns (params, probs
    for every row)."""
    lp = torch.as_tensor(logp, dtype=torch.float64);cl = torch.as_tensor(np.clip(classes, 0, CLASSES - 1), dtype=torch.long)
    lg = torch.as_tensor(legal, dtype=torch.bool);y = torch.as_tensor(labels, dtype=torch.long);r = torch.as_tensor(rows, dtype=torch.long)
    w = torch.zeros(CLASSES + 1, dtype=torch.float64, requires_grad=True)
    logits = lambda i: (lp[i] * torch.exp(-w[CLASSES]) + w[:CLASSES][cl[i]]).masked_fill(~lg[i], -1e9)
    optimizer = torch.optim.LBFGS([w], max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad();loss = torch.nn.functional.cross_entropy(logits(r), y[r]);loss.backward();return loss
    optimizer.step(closure)
    with torch.no_grad():probs = torch.softmax(logits(torch.arange(len(labels))), -1).numpy().astype(np.float32)
    return dict(temperature=float(torch.exp(w[CLASSES])), class_bias=w[:CLASSES].tolist()), probs


def pad(sequences, arrays, extra, torch, device):
    """Pad per-game index runs into [G, L, ...] tensors; `extra` holds per-row log π_FP [n, 9]."""
    length = max(len(s) for s in sequences);g = len(sequences)
    feats = np.zeros((g, length, FEATURES), np.float32);logp = np.zeros((g, length, 9), np.float32)
    classes = np.zeros((g, length, 9), np.int64);legal = np.zeros((g, length, 9), bool);labels = np.zeros((g, length), np.int64)
    valid = np.zeros((g, length), bool)
    for i, s in enumerate(sequences):
        k = len(s);feats[i, :k] = arrays['_features'][s];logp[i, :k] = extra[s];classes[i, :k] = arrays['candidate_class'][s]
        legal[i, :k] = arrays['legal'][s];labels[i, :k] = arrays['label'][s];valid[i, :k] = True
    legal[~valid] = True   # padded steps: any finite softmax, excluded from the loss
    t = lambda a: torch.as_tensor(a, device=device)
    return t(feats), t(logp), t(classes), t(legal), t(labels), t(valid)


def probe_probs(model, sequences, arrays, logp, torch, device, size=64):
    out = np.zeros((len(arrays['label']), 9), np.float32);model.eval()
    with torch.no_grad():
        for lo in range(0, len(sequences), size):
            chunk = sequences[lo:lo + size];f, lp, c, lg, _, valid = pad(chunk, arrays, logp, torch, device)
            probs = torch.softmax(model(f, lp, c, lg), -1).cpu().numpy()
            for i, s in enumerate(chunk):out[s] = probs[i, :len(s)]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('data');parser.add_argument('out');parser.add_argument('--base', required=True);parser.add_argument('--pi-fp', required=True)
    parser.add_argument('--pi-fp-data', help="the dataset π_FP was trained on; the probe then fits only on games outside its train split")
    parser.add_argument('--device', default='cuda');parser.add_argument('--epochs', type=int, default=30);parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3);parser.add_argument('--games-per-batch', type=int, default=16)
    args = parser.parse_args(argv)
    import torch
    from pokezero.neural_policy import checkpoint_file_sha256, load_transformer_checkpoint
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest');torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(args.seed)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time();rng = np.random.default_rng(args.seed)
    arrays = train_mod.load(args.data)
    for key in ('candidate_class', 'label_class', 'own_hp', 'opp_hp', 'opp_head_logits'):
        if key not in arrays:raise SystemExit(f'{args.data} lacks {key}: rebuild it with fp_reply_data --extras')
    split = data_mod.split_of(arrays['seed']);rows = {n: np.flatnonzero(split == n) for n in ('train', 'val', 'test')}
    labels, legal = arrays['label'], arrays['legal'];test = rows['test']
    pi_fp_model, _ = load_transformer_checkpoint(args.pi_fp, map_location=args.device)
    index = train_mod.windows(arrays['ordinal'], 1)
    pi_fp = train_mod.predict(pi_fp_model, arrays, index, np.arange(len(labels)), torch, args.device)
    del pi_fp_model
    _, calibrated = train_mod.recalibrate(arrays['probs'][rows['train']], labels[rows['train']], legal[rows['train']], torch)
    report = dict(schema='v2s-fp-reply-probe.v1', data=str(args.data), base=str(args.base), base_sha256=checkpoint_file_sha256(args.base),
                  pi_fp=str(args.pi_fp), pi_fp_sha256=checkpoint_file_sha256(args.pi_fp), rows={k: int(len(v)) for k, v in rows.items()},
                  games={k: int(len(np.unique(arrays['ordinal'][v]))) for k, v in rows.items()},
                  seeds={k: int(len(np.unique(arrays['seed'][v]))) for k, v in rows.items()}, args=vars(args))

    # 1. 9375's opponent head, on rows where our seat decided in the same round
    head = arrays['opp_head_logits'];has = ~np.isnan(head).any(1)
    head_probs = np.where(has[:, None], masked_softmax(np.nan_to_num(head), legal), 1. / 9)
    train_has = rows['train'][has[rows['train']]]
    head_fit, head_calibrated = train_mod.recalibrate(head_probs[train_has], labels[train_has], legal[train_has], torch)
    t1 = test[has[test]]
    models = dict(uniform_legal=legal[t1] / legal[t1].sum(1, keepdims=True), champion_pi=arrays['probs'][t1],
                  champion_pi_recalibrated=calibrated(arrays['probs'][t1], legal[t1]), opponent_head=head_probs[t1],
                  opponent_head_recalibrated=head_calibrated(head_probs[t1], legal[t1]), pi_fp=pi_fp[t1])
    report['opponent_head'] = dict(coverage=dict(test_rows=int(len(test)), with_our_view=int(len(t1))), recalibration=head_fit,
        test={name: train_mod.grouped(p, arrays, t1, uniform=name == 'uniform_legal') for name, p in models.items()},
        bootstrap=train_mod.game_bootstrap(models, arrays, t1, (('opponent_head', 'champion_pi_recalibrated'),
            ('opponent_head_recalibrated', 'champion_pi_recalibrated'), ('opponent_head_recalibrated', 'uniform_legal'),
            ('pi_fp', 'opponent_head_recalibrated'))))
    print(json.dumps(dict(event='OPPONENT_HEAD', **{k: report['opponent_head']['test'][k]['all'] for k in models},
                          bootstrap=report['opponent_head']['bootstrap'])), flush=True)

    # 2. option C probe on π_FP
    arrays['_features'] = step_features(arrays);logp = np.log(np.clip(pi_fp, 1e-12, 1.)).astype(np.float32)
    seen = seen_ordinals(args.pi_fp_data) if args.pi_fp_data else None
    fit_train, fit_val = fit_rows(split, arrays['ordinal'], arrays['seed'], seen)
    for name, r in (('fit', fit_train), ('early stopping', fit_val)):
        if len(np.unique(arrays['seed'][r])) < train_mod.MIN_CLUSTERS:raise SystemExit(f'too few seeds for the probe {name} rows')
    fitting = dict(source='unseen by π_FP' if seen is not None else 'train/validation',
                   rows=dict(fit=int(len(fit_train)), early_stopping=int(len(fit_val))),
                   games=dict(fit=int(len(np.unique(arrays['ordinal'][fit_train]))), early_stopping=int(len(np.unique(arrays['ordinal'][fit_val])))),
                   pi_fp_nll=dict(fit=train_mod.nll(pi_fp[fit_train], labels[fit_train]), early_stopping=train_mod.nll(pi_fp[fit_val], labels[fit_val]),
                                  train=train_mod.nll(pi_fp[rows['train']], labels[rows['train']]), test=train_mod.nll(pi_fp[test], labels[test])))
    print(json.dumps(dict(event='PROBE_FIT_ROWS', **fitting)), flush=True)
    seqs = dict(train=game_sequences(arrays['ordinal'], fit_train), val=game_sequences(arrays['ordinal'], fit_val),
                test=game_sequences(arrays['ordinal'], test))
    model = build_probe(torch).to(args.device)
    identity = probe_probs(model, seqs['test'], arrays, logp, torch, args.device)
    identity_gap = float(np.abs(identity[test] - pi_fp[test]).max())
    print(json.dumps(dict(event='PROBE_IDENTITY', max_abs_probability_difference=identity_gap)), flush=True)
    if not identity_gap <= 1e-5:raise SystemExit(f'gate 3 failed: the zero-initialised probe differs from π_FP by {identity_gap}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    val_nll = lambda: train_mod.nll(probe_probs(model, seqs['val'], arrays, logp, torch, args.device)[fit_val], labels[fit_val])
    best = dict(epoch=0, val_nll=val_nll());state = {k: v.detach().clone() for k, v in model.state_dict().items()};history = [dict(best)]
    for epoch in range(1, args.epochs + 1):
        model.train();order = rng.permutation(len(seqs['train']));losses = []
        for lo in range(0, len(order), args.games_per_batch):
            f, lp, c, lg, y, valid = pad([seqs['train'][i] for i in order[lo:lo + args.games_per_batch]], arrays, logp, torch, args.device)
            logits = model(f, lp, c, lg)
            loss = torch.nn.functional.cross_entropy(logits[valid], y[valid])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step();losses.append(float(loss.detach()))
        val = val_nll();history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_nll=val))
        print(json.dumps(dict(event='PROBE_EPOCH', **history[-1])), flush=True)
        if val < best['val_nll']:best = dict(epoch=epoch, val_nll=val);state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    probe = probe_probs(model, seqs['test'], arrays, logp, torch, args.device)
    d = train_mod.label_nll(probe[test], labels[test]) - train_mod.label_nll(pi_fp[test], labels[test])
    turn = arrays['turn'][test];clusters = arrays['seed'][test];every = np.ones(len(test), bool)
    late = (turn >= LATE[0]) & (turn <= LATE[1]);early = (turn >= EARLY[0]) & (turn <= EARLY[1]);first = turn == 1
    gates = dict(gate1_nll_probe_minus_pi_fp=cluster_bootstrap(d, clusters, [every], lambda a: a),
                 gate2_late_minus_early=cluster_bootstrap(d, clusters, [late, early], lambda a, b: a - b),
                 gate2_turn1=cluster_bootstrap(d, clusters, [first], lambda a: a),
                 gate3_identity_max_abs_difference=identity_gap)
    static_params, static = static_fit(logp, arrays['candidate_class'], legal, labels, fit_train, torch)
    nll_static = train_mod.label_nll(static[test], labels[test])
    ds = nll_static - train_mod.label_nll(pi_fp[test], labels[test]);dh = train_mod.label_nll(probe[test], labels[test]) - nll_static
    control = dict(static=static_params, static_minus_pi_fp=cluster_bootstrap(ds, clusters, [every], lambda a: a),
                   probe_minus_static=cluster_bootstrap(dh, clusters, [every], lambda a: a),
                   probe_minus_static_late_minus_early=cluster_bootstrap(dh, clusters, [late, early], lambda a, b: a - b))
    print(json.dumps(dict(event='PROBE_STATIC_CONTROL', **control)), flush=True)
    gates['pass'] = dict(gate1=bool(gates['gate1_nll_probe_minus_pi_fp']['interval_90'] and gates['gate1_nll_probe_minus_pi_fp']['interval_90'][1] < 0),
                         gate2=bool(gates['gate2_late_minus_early']['interval_90'] and gates['gate2_late_minus_early']['interval_90'][1] < 0
                                    and gates['gate2_turn1']['interval_90'] and gates['gate2_turn1']['interval_90'][0] <= 0),
                         gate3=identity_gap <= 1e-5)
    with torch.no_grad():
        f, lp, c, lg, _, valid = pad(seqs['test'], arrays, logp, torch, args.device)
        out_head = model.head(model.gru(torch.cat([torch.zeros_like(f[:, :1]), f[:, :-1]], 1))[0])[valid].cpu().numpy()
    report['probe'] = dict(fitting=fitting, history=history, best_epoch=best['epoch'], gates=gates, static_control=control,
                           test=dict(pi_fp=train_mod.grouped(pi_fp[test], arrays, test), static=train_mod.grouped(static[test], arrays, test),
                                     probe=train_mod.grouped(probe[test], arrays, test)),
                           in_context_curve=train_mod.in_context_curve(dict(pi_fp=pi_fp[test], static=static[test], probe=probe[test]), arrays, test),
                           head_output=dict(bias_mean=out_head[:, :CLASSES].mean(0).tolist(), bias_sd=out_head[:, :CLASSES].std(0).tolist(),
                                            temperature_mean=float(np.exp(out_head[:, CLASSES]).mean())))
    torch.save(dict(state_dict=state, features=FEATURES, classes=CLASSES, pi_fp_sha256=report['pi_fp_sha256']), out / 'probe.pt')
    report['seconds'] = round(time.time() - started, 1)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(event='PROBE_DONE', best_epoch=best['epoch'], gates=gates, test_all={k: v['all'] for k, v in report['probe']['test'].items()})), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
