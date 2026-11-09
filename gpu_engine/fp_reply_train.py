"""Train π_FP: fine-tune the champion's policy on Foul Play's own decisions, and report it against the champion's π.

usage: python -m gpu_engine.fp_reply_train DATA_DIR OUT_DIR --checkpoint PT [--epochs 8] [--lr 2e-5] [--batch 256]
       [--weight-decay 0.01] [--device cuda] [--seed 0] [--redact peer-view-v1]

DATA_DIR holds fp_reply_data shards (one row per Foul Play decision, games in decision order). Splits are by game, from the
battle seed: seed % 10 == 0 is test, == 1 is validation, the rest train (fp_reply_data.split_of). Each row's input is the
champion's own window: the last window_size Foul Play observations of that game, front-padded as observation_window_to_torch
does. The loss is cross-entropy of Foul Play's action under the legal-masked policy logits; the value head gets no loss. The
best epoch by validation NLL is kept and saved in the champion's checkpoint format, marked policy-only: a top-level `fp_reply`
key, value_calibration_transform cleared, and policy_id renamed. The value head gets no gradient but the shared trunk moves, so
its value output is meaningless.
Reported on the test split, by the fp_reply_data groups: uniform over legal (top-1 is a uniform guess's expectation), the
champion's π zero-shot (its stored probabilities), π recalibrated (a temperature and a switch bias fitted on train, from the
stored float32 probabilities clipped at 1e-12, so tail actions share one floor; a slight approximation), and π_FP.
Checks: TF32 and the MHA eval fast path are off, so CUDA runs exact fp32 (the fast path alone moved probabilities by up to 5.5e-3). Before training, the base model's validation probabilities through windows/batch
must reproduce the stored π to 1e-3. --export-archive recomputes π_FP on CPU in fp32 and requires the JAX archive forward to
match it to 1e-4.
--redact peer-view-v1 trains and evaluates on Foul Play's view redacted the way the search's constructor builds the opponent's
(peer) view (redact_peer_view); the input-parity check runs on the unredacted tensors first, and the checkpoint's fp_reply marker
records the redaction.
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

from gpu_engine import fp_reply_data as data_mod

OBS_KEYS = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask')


def load(data_dir):
    shards = sorted(glob.glob(str(Path(data_dir) / 'shard-*.npz')), key=lambda p: int(Path(p).stem.split('-')[1]))
    parts = [dict(np.load(p)) for p in shards]
    parts = [p for p in parts if len(p.get('label', ()))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def windows(ordinal, window):
    """index[i, j] = row of the j-th window slot for decision i (-1 = padding); rows of one game are contiguous and in order."""
    index = np.full((len(ordinal), window), -1, np.int64)
    start = 0
    for i in range(len(ordinal)):
        if i and ordinal[i] != ordinal[i - 1]:start = i
        lo = max(start, i - window + 1);rows = np.arange(lo, i + 1)
        index[i, window - len(rows):] = rows
    return index


def batch(arrays, index, rows, torch, device):
    idx = index[rows];pad = idx < 0;safe = np.where(pad, 0, idx)
    out = {}
    for key, dtype in zip(OBS_KEYS, (torch.long, torch.float32, torch.long, torch.bool)):
        values = arrays[key][safe]
        values[pad] = 0
        out[key] = torch.as_tensor(values, device=device).to(dtype)
    out['history_mask'] = torch.as_tensor(~pad, device=device)
    out['legal'] = torch.as_tensor(arrays['legal'][rows], device=device)
    out['label'] = torch.as_tensor(arrays['label'][rows], device=device).long()
    return out


def forward(model, b):
    output = model(categorical_ids=b['categorical_ids'], numeric_features=b['numeric_features'], token_type_ids=b['token_type_ids'],
                   attention_mask=b['attention_mask'], history_mask=b['history_mask'])
    return output.policy_logits.float().masked_fill(~b['legal'], -1e9)


def predict(model, arrays, index, rows, torch, device, size=512):
    model.eval();out = []
    with torch.no_grad():
        for lo in range(0, len(rows), size):
            b = batch(arrays, index, rows[lo:lo + size], torch, device)
            out.append(torch.softmax(forward(model, b), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, data_mod.ACTIONS), np.float32)


def nll(probs, labels):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.)).mean())


def recalibrate(probs, labels, legal, torch):
    """Fit log-temperature and a switch-action bias on train: p' ∝ exp(log p / T + b·[a is a switch]) over legal actions."""
    logp = torch.as_tensor(np.log(np.clip(probs, 1e-12, 1.)), dtype=torch.float64)
    mask = torch.as_tensor(legal);y = torch.as_tensor(labels).long()
    switch = torch.zeros(data_mod.ACTIONS, dtype=torch.float64);switch[data_mod.MOVES:] = 1.
    params = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], max_iter=200, line_search_fn='strong_wolfe')

    def apply(p, lp, m):
        return torch.log_softmax((lp / torch.exp(p[0]) + p[1] * switch).masked_fill(~m, -1e9), -1)

    def closure():
        optimizer.zero_grad();loss = -apply(params, logp, mask)[torch.arange(len(y)), y].mean();loss.backward();return loss
    optimizer.step(closure)
    p = params.detach()
    return dict(temperature=float(torch.exp(p[0])), switch_bias=float(p[1])), \
        lambda pr, lg: torch.exp(apply(p, torch.as_tensor(np.log(np.clip(pr, 1e-12, 1.)), dtype=torch.float64), torch.as_tensor(lg))).numpy()


def grouped(probs, arrays, rows, uniform=False):
    labels, legal, turn, forced = (arrays[k][rows] for k in ('label', 'legal', 'turn', 'forced'))
    can_switch = legal[:, data_mod.MOVES:].any(1);can_move = legal[:, :data_mod.MOVES].any(1)
    groups = dict(all=np.ones(len(rows), bool), move_request=~forced, forced_switch=forced, free_choice=can_move & can_switch & ~forced,
                  label_move=labels < data_mod.MOVES, label_switch=labels >= data_mod.MOVES)
    for low, high in data_mod.TURN_BANDS:
        groups[f'turn {low}-{high}' if high < 10**9 else f'turn {low}+'] = (turn >= low) & (turn <= high)
    return {name: data_mod.metrics(probs[m], labels[m], legal[m], uniform=uniform) for name, m in groups.items()}


TURN_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 17), (18, 24), (25, 34), (35, 49), (50, 10**9))
BOOTSTRAP_DRAWS = 2000
MIN_CLUSTERS = 5


def label_nll(probs, labels):
    return -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.))


def in_context_curve(models, arrays, rows):
    """NLL by turn bucket for each reply model on the given rows: a history model should improve as turns accumulate; a state-only
    one shouldn't (any slope it shows comes from the state, e.g. revealed sets)."""
    turn, labels = arrays['turn'][rows], arrays['label'][rows];curve = []
    for low, high in TURN_BUCKETS:
        m = (turn >= low) & (turn <= high)
        if m.any():curve.append(dict(turns=f'{low}' if low == high else (f'{low}-{high}' if high < 10**9 else f'{low}+'), decisions=int(m.sum()),
                                     **{name: float(label_nll(p[m], labels[m]).mean()) for name, p in models.items()}))
    if sum(c['decisions'] for c in curve) != len(rows):raise ValueError('turn buckets must cover every row')
    return curve


def game_bootstrap(models, arrays, rows, pairs, draws=BOOTSTRAP_DRAWS, seed=0):
    """90% cluster bootstrap intervals for mean-NLL differences (a − b) on the given rows. The cluster is the battle seed, the split
    unit: the collector plays each seed once from each seat, so the two games on a seed are correlated and resampled together.
    Fewer than MIN_CLUSTERS seeds gives no interval."""
    labels, clusters = arrays['label'][rows], arrays['seed'][rows]
    unique, inverse = np.unique(clusters, return_inverse=True);rng = np.random.default_rng(seed)
    per = {name: label_nll(p, labels) for name, p in models.items()};out = {}
    for a, b in pairs:
        d = per[a] - per[b];sums = np.bincount(inverse, d, len(unique));counts = np.bincount(inverse, None, len(unique))
        stats = [];idx = np.arange(len(unique))
        for _ in range(draws if len(unique) >= MIN_CLUSTERS else 0):
            pick = rng.choice(idx, len(idx));stats.append(sums[pick].sum() / counts[pick].sum())
        out[f'{a} - {b}'] = dict(mean=float(d.mean()), seeds=int(len(unique)), games=int(len(np.unique(arrays['ordinal'][rows]))),
                                 interval_90=[float(q) for q in np.quantile(stats, [.05, .95])] if len(unique) >= MIN_CLUSTERS else None)
    return out


def test_report(arrays, test, pi_fp_probs, calibrated, compare=None):
    """The four reply models on the test rows: uniform over legal actions, the champion's π (its stored probabilities), π recalibrated,
    and π_FP; by group, by turn (the in-context curve), and game-bootstrap intervals for the NLL differences. compare, when given,
    is another run's π_FP on the same test rows (e.g. window 1 for a window-k run), reported as pi_fp_compare."""
    legal = arrays['legal'][test]
    models = dict(uniform_legal=legal / legal.sum(1, keepdims=True), champion_pi=arrays['probs'][test],
                  champion_pi_recalibrated=calibrated(arrays['probs'][test], legal), pi_fp=pi_fp_probs)
    pairs = [('pi_fp', 'champion_pi'), ('pi_fp', 'champion_pi_recalibrated'), ('pi_fp', 'uniform_legal'), ('champion_pi', 'uniform_legal')]
    if compare is not None:models['pi_fp_compare'] = compare;pairs.insert(0, ('pi_fp', 'pi_fp_compare'))
    report = {name: grouped(p, arrays, test, uniform=name == 'uniform_legal') for name, p in models.items()}
    report['in_context_curve'] = in_context_curve(models, arrays, test)
    report['game_bootstrap_nll'] = game_bootstrap(models, arrays, test, pairs)
    return report


def widen(checkpoint, window, out, torch):
    """The base checkpoint at a longer window. The model takes window_size from its config and attends over every frame's tokens
    jointly; the one parameter sized by the window is history_position_embedding (window_size × width). Its rows all start as the
    base's single row, so the current frame (index window − 1) keeps the base's embedding and, with earlier frames masked, the
    widened model reproduces the base exactly. No other parameter changes. champion.export (window 1 only) can't take the result."""
    payload = dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    config = dict(payload['model_config']);base = int(config['window_size'])
    if base != 1:raise SystemExit(f'widen expects a window-1 base, got {base}')
    state = dict(payload['state_dict']);row = state['history_position_embedding.weight']
    if row.shape[0] != 1:raise SystemExit(f'unexpected history_position_embedding shape {tuple(row.shape)}')
    state['history_position_embedding.weight'] = row.repeat(window, 1).clone()
    payload['model_config'] = dict(config, window_size=window);payload['state_dict'] = state
    if isinstance(payload.get('training_config'), dict) and 'window_size' in payload['training_config']:
        payload['training_config'] = dict(payload['training_config'], window_size=window)
    path = Path(out) / f'base-window-{window}.pt'
    torch.save(payload, path)
    return path


def current_only(index):
    """The same rows with every earlier frame masked: only the current observation, at the window's last slot."""
    only = np.full_like(index, -1);only[:, -1] = index[:, -1]
    return only


def step_cost(model, arrays, index, rows, torch, device, single=200, batched=4096):
    """Inference cost per decision: batch-1 latency (median over `single` decisions after warm-up) on the device and on CPU, and
    batch-512 throughput on the device."""
    import copy
    sync = (lambda: torch.cuda.synchronize(device)) if str(device) != 'cpu' else (lambda: None)
    def latency(m, dev, count):
        m.eval();times = []
        with torch.no_grad():
            for i, r in enumerate(rows[:count + 10]):
                b = batch(arrays, index, np.array([r]), torch, dev)
                if dev != 'cpu':sync()
                started = time.perf_counter();forward(m, b)
                if dev != 'cpu':sync()
                if i >= 10:times.append(time.perf_counter() - started)
        return 1e3 * float(np.median(times))
    out = dict(device=str(device), note='for comparing windows only: exact-fp32 kernels (MHA fast path off), host gathering included '
               'in the batched figure, CPU latency on one thread', batch1_ms_device=latency(model, device, single))
    with torch.no_grad():
        chunk = rows[:batched];predict(model, arrays, index, chunk[:512], torch, device);sync()
        started = time.perf_counter();predict(model, arrays, index, chunk, torch, device);sync()
        out['batch512_ms_per_decision_device'] = 1e3 * (time.perf_counter() - started) / max(1, len(chunk))
    threads = torch.get_num_threads();torch.set_num_threads(1)
    try:out['batch1_ms_cpu_1thread'] = latency(copy.deepcopy(model).to('cpu'), 'cpu', min(single, 50))
    finally:torch.set_num_threads(threads)
    return out


ARCHIVE_TOLERANCE = 1e-4
INPUT_TOLERANCE = 1e-3

# The constructor's peer view (gpu_engine/phase_d_live_worlds.py) with Foul Play as the observer differs from env.observe(Foul Play)
# by design in exactly these v4 columns (the input-parity session's 48-cell true-world run, November 9): Foul Play's own damage
# history is unavailable to the reply policy, and our HP reaches it at Showdown's public percent precision.
REDACTIONS = ('peer-view-v1',)
SELF_ACTIVE_TOKEN, OPPONENT_TOKENS = 1, slice(7, 13)
HP_FRACTION, LAST_DAMAGE_DEALT, LAST_DAMAGE_TAKEN = 0, 38, 39
V4_SHAPE = (23, 132)


def public_percent(x):
    """Showdown's public HP percent of exact fractions x, as a fraction: ceil(100x), 99 for a damaged mon that would show 100, 0 for
    fainted or absent. 100x is rounded to 3 decimals first: float32 noise lands just above an exact integer percent (2/5 -> 41 when
    rounded to 6 decimals), while a non-integer percent is at least 1/maxhp (>= 0.00125 for maxhp <= 800) from one."""
    p = np.ceil(np.round(100. * np.asarray(x, np.float64), 3))
    p = np.where((np.asarray(x) < 1) & (p >= 100), 99., p)
    return (np.clip(p, 0., 100.) / 100.).astype(np.asarray(x).dtype)


def redact_peer_view(numeric):
    """peer-view-v1, in place on v4 numeric features [..., 23, 132]: Foul Play's own active LAST_DAMAGE_TAKEN -> 0; our six
    Pokémon's LAST_DAMAGE_DEALT -> 0 (it is Foul Play's exact HP loss); our six Pokémon's HP_FRACTION -> the public percent."""
    if tuple(numeric.shape[-2:]) != V4_SHAPE:raise SystemExit(f'peer-view-v1 is defined on v4 {V4_SHAPE} numeric features, not {numeric.shape[-2:]}')
    numeric[..., SELF_ACTIVE_TOKEN, LAST_DAMAGE_TAKEN] = 0.
    numeric[..., OPPONENT_TOKENS, LAST_DAMAGE_DEALT] = 0.
    numeric[..., OPPONENT_TOKENS, HP_FRACTION] = public_percent(numeric[..., OPPONENT_TOKENS, HP_FRACTION])
    return numeric


def archive_parity(checkpoint, archive, arrays, index, rows, torch, extra, size=512):
    """Export π_FP to the champion_evaluator weight archive and require the JAX fp32 forward to reproduce π_FP's torch probabilities,
    recomputed on CPU in fp32, to ARCHIVE_TOLERANCE (an export bug that size can't hide behind kernel noise)."""
    import jax
    import jax.numpy as jnp
    from gpu_engine import champion
    from pokezero.neural_policy import load_transformer_checkpoint
    cpu_model, _ = load_transformer_checkpoint(checkpoint, map_location='cpu')
    torch_probs = predict(cpu_model, arrays, index, rows, torch, 'cpu')
    meta = champion.export(checkpoint, archive, extra=extra)
    params, config, loaded = champion.load(archive)
    worst = 0.;agree = 0
    for lo in range(0, len(rows), size):
        r = rows[lo:lo + size]
        output = champion.forward(params, jnp.asarray(arrays['numeric_features'][r]), jnp.asarray(arrays['categorical_ids'][r]),
                                  jnp.asarray(arrays['token_type_ids'][r].astype(np.int32)), jnp.asarray(arrays['attention_mask'][r]), config)
        logits = jnp.where(jnp.asarray(arrays['legal'][r]), output.policy_logits, -1e9)
        probs = np.asarray(jax.nn.softmax(logits, axis=-1));reference = torch_probs[lo:lo + size]
        worst = max(worst, float(np.abs(probs - reference).max()))
        agree += int((np.where(arrays['legal'][r], probs, -1).argmax(1) == np.where(arrays['legal'][r], reference, -1).argmax(1)).sum())
    result = dict(archive=str(archive), weights_archive_sha256=loaded['weights_archive_sha256'], checkpoint_sha256=meta['checkpoint_sha256'],
                  extra=loaded.get('extra'), reference='torch fp32 on CPU', tolerance=ARCHIVE_TOLERANCE, rows=int(len(rows)),
                  max_abs_probability_difference=worst, argmax_agreement=agree / len(rows) if len(rows) else None)
    if not worst <= ARCHIVE_TOLERANCE:
        Path(archive).replace(str(archive) + '.failed-parity')
        raise SystemExit(f'archive parity failed (archive moved to {archive}.failed-parity): {json.dumps(result)}')
    return result


def seats_by_split(arrays, rows):
    """Rows per cell seat in each split (the collector may tie seat to seed parity, which would make a split single-seat)."""
    return {name: {str(int(s)): int((arrays['seat'][r] == s).sum()) for s in np.unique(arrays['seat'])} for name, r in rows.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('data');parser.add_argument('out');parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epochs', type=int, default=8);parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch', type=int, default=256);parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--device', default='cuda');parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--evaluate', metavar='PI_FP_PT', help='skip training: report this trained π_FP on test')
    parser.add_argument('--export-archive', metavar='NPZ', help='with --evaluate: export π_FP to a champion_evaluator archive and check parity')
    parser.add_argument('--window', type=int, default=None, help='train at this window (the base is widened; see widen)')
    parser.add_argument('--compare-to', metavar='RUN_DIR', help="another run's test-probs.npz to report beside this π_FP (same test rows)")
    parser.add_argument('--redact', choices=REDACTIONS, help="train/evaluate on Foul Play's view redacted as the constructor's peer view")
    args = parser.parse_args(argv)
    if args.evaluate and (args.window or args.compare_to):parser.error('--window and --compare-to apply to training, not --evaluate')
    if args.export_archive and not args.evaluate:parser.error('--export-archive needs --evaluate')
    import torch
    from pokezero.neural_policy import checkpoint_file_sha256, load_transformer_checkpoint
    # Exact fp32 on CUDA: TF32 off, highest matmul precision, and no MHA fast path. In eval mode nn.TransformerEncoder's fused fast
    # path on CUDA moved π's probabilities by up to 5.5e-3 against CPU (median 6e-4); with it off the difference is under 4e-6.
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest');torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(args.seed);rng = np.random.default_rng(args.seed)
    if args.export_archive and Path(args.export_archive).exists():raise SystemExit(f'{args.export_archive} exists; champion.export writes exclusively')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    arrays = load(args.data)
    base_path = args.checkpoint
    if args.window and not args.evaluate:base_path = widen(args.checkpoint, args.window, out, torch)
    model, result = load_transformer_checkpoint(args.evaluate or base_path, map_location=args.device)
    window = int(result.model_config.window_size)
    index = windows(arrays['ordinal'], window)
    split = data_mod.split_of(arrays['seed'])
    rows = {name: np.flatnonzero(split == name) for name in ('train', 'val', 'test')}
    games = {name: int(len(np.unique(arrays['ordinal'][r]))) for name, r in rows.items()}
    seats = seats_by_split(arrays, rows)
    print(json.dumps(dict(event='DATA', rows={k: int(len(v)) for k, v in rows.items()}, games=games, seats=seats, window=window)), flush=True)
    labels = arrays['label'];compare = None
    if args.compare_to:   # validated before training, so a bad comparison can't waste a run
        other = np.load(Path(args.compare_to) / 'test-probs.npz')
        for key in ('ordinal', 'seed', 'label', 'round'):
            if not np.array_equal(other[key], arrays[key][rows['test']]):raise SystemExit(f'--compare-to run has different test rows ({key})')
        compare = other['probs']
    if args.evaluate and args.redact:redact_peer_view(arrays['numeric_features'])
    calibration, calibrated = recalibrate(arrays['probs'][rows['train']], labels[rows['train']], arrays['legal'][rows['train']], torch)
    if args.evaluate:
        test = rows['test'];probs = predict(model, arrays, index, test, torch, args.device)
        report = dict(schema='v2s-fp-reply-model-eval.v1', data=str(args.data), redaction=args.redact, base_checkpoint=str(args.checkpoint),
                      base_checkpoint_sha256=checkpoint_file_sha256(args.checkpoint), pi_fp=str(args.evaluate),
                      pi_fp_sha256=checkpoint_file_sha256(args.evaluate), window=window, rows={k: int(len(v)) for k, v in rows.items()},
                      games=games, seats=seats, split_rule='battle seed % 10: 0 test, 1 validation, else train', recalibration=calibration,
                      ece_bins=data_mod.ECE_BINS, test=test_report(arrays, test, probs, calibrated))
        (out / 'report-eval.json').write_text(json.dumps(report, indent=1) + '\n')
        if args.export_archive:
            if window != 1:raise SystemExit('champion_evaluator archives need window 1')
            payload = torch.load(args.evaluate, map_location='cpu', weights_only=True)
            report['archive'] = archive_parity(args.evaluate, args.export_archive, arrays, index, test, torch, payload.get('fp_reply'))
            (out / 'report-eval.json').write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps(dict(event='EVALUATED', test_all={k: v['all'] for k, v in report['test'].items() if isinstance(v, dict) and 'all' in v}, bootstrap=report['test']['game_bootstrap_nll'], archive=report.get('archive'))), flush=True)
        return 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    base_val = predict(model, arrays, current_only(index), rows['val'], torch, args.device)
    input_parity = float(np.abs(base_val - arrays['probs'][rows['val']]).max()) if len(rows['val']) else 0.
    print(json.dumps(dict(event='INPUT_PARITY', max_abs_probability_difference=input_parity, tolerance=INPUT_TOLERANCE)), flush=True)
    if not input_parity <= INPUT_TOLERANCE:raise SystemExit(f'base model through windows/batch differs from the stored π by {input_parity}')
    if args.redact:
        before = arrays['numeric_features'].copy();redact_peer_view(arrays['numeric_features'])
        changed = (before != arrays['numeric_features']).any(-1);del before
        redaction = dict(name=args.redact, rows_changed=float(changed.any(-1).mean()),
                         token_rows_changed={int(t): float(changed[:, t].mean()) for t in range(changed.shape[1]) if changed[:, t].any()})
        print(json.dumps(dict(event='REDACTED', **redaction)), flush=True)
    else:
        redaction = None
    best = dict(epoch=0, val_nll=nll(predict(model, arrays, index, rows['val'], torch, args.device), labels[rows['val']]))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [dict(epoch=0, val_nll=best['val_nll'])]
    print(json.dumps(dict(event='EPOCH', **history[-1])), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train();order = rng.permutation(rows['train']);losses = []
        for lo in range(0, len(order), args.batch):
            b = batch(arrays, index, order[lo:lo + args.batch], torch, args.device)
            loss = torch.nn.functional.cross_entropy(forward(model, b), b['label'])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step()
            losses.append(float(loss))
        val = nll(predict(model, arrays, index, rows['val'], torch, args.device), labels[rows['val']])
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_nll=val));print(json.dumps(dict(event='EPOCH', **history[-1])), flush=True)
        if val < best['val_nll']:
            best = dict(epoch=epoch, val_nll=val);best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = rows['test'];test_probs = predict(model, arrays, index, test, torch, args.device)
    np.savez(out / 'test-probs.npz', rows=test, ordinal=arrays['ordinal'][test], seed=arrays['seed'][test], label=labels[test],
             round=arrays['round'][test], probs=test_probs)
    payload = dict(torch.load(base_path, map_location='cpu', weights_only=True))   # saved before the optional extras below
    base_sha = checkpoint_file_sha256(args.checkpoint)
    payload['state_dict'] = best_state;payload['value_calibration_transform'] = None
    payload['value_calibration_source_checkpoint_sha256'] = None
    if isinstance(payload.get('model_config'), dict) and 'policy_id' in payload['model_config']:
        payload['model_config'] = dict(payload['model_config'], policy_id=f"fp-reply-{payload['model_config']['policy_id']}")
    payload['fp_reply'] = dict(policy_only=True, base_checkpoint_sha256=base_sha, best_epoch=best['epoch'], data=str(args.data), window=window,
                               redaction=args.redact,
                               note='policy fine-tuned on Foul Play decisions; the value output is not trained and must not be used')
    torch.save(payload, out / 'pi-fp.pt')
    cost = step_cost(model, arrays, index, test, torch, args.device)
    report = dict(schema='v2s-fp-reply-model.v1', data=str(args.data), checkpoint=str(args.checkpoint),
                  checkpoint_sha256=checkpoint_file_sha256(args.checkpoint), window=window, rows={k: int(len(v)) for k, v in rows.items()},
                  games=games, seats=seats, input_parity=input_parity, redaction=redaction, split_rule='battle seed % 10: 0 test, 1 validation, else train',
                  args=vars(args), history=history,
                  best_epoch=best['epoch'], recalibration=calibration, ece_bins=data_mod.ECE_BINS,
                  test=test_report(arrays, test, test_probs, calibrated, compare), compare_to=args.compare_to, inference_cost=cost,
                  seconds=round(time.time() - started, 1))
    report['saved'] = dict(path=str(out / 'pi-fp.pt'), sha256=checkpoint_file_sha256(out / 'pi-fp.pt'),
                           note='champion checkpoint format; policy fine-tuned on Foul Play decisions; the value head output is not trained')
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(event='DONE', best_epoch=best['epoch'], test_all={k: v['all'] for k, v in report['test'].items() if isinstance(v, dict) and 'all' in v}, bootstrap=report['test']['game_bootstrap_nll'], inference_cost=cost)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
