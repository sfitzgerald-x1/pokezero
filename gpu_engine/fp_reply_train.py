"""Train π_FP: fine-tune the champion's policy on Foul Play's own decisions, and report it against the champion's π.

usage: python -m gpu_engine.fp_reply_train DATA_DIR OUT_DIR --checkpoint PT [--epochs 8] [--lr 2e-5] [--batch 256]
       [--weight-decay 0.01] [--device cuda] [--seed 0]

DATA_DIR holds fp_reply_data shards (one row per Foul Play decision, games in decision order). Splits are by game, from the
battle seed: seed % 10 == 0 is test, == 1 is validation, the rest train (fp_reply_data.split_of). Each row's input is the
champion's own window: the last window_size Foul Play observations of that game, front-padded as observation_window_to_torch
does. The loss is cross-entropy of Foul Play's action under the legal-masked policy logits; the value head gets no loss. The
best epoch by validation NLL is kept and saved in the champion's checkpoint format (policy only; its value output is not
trained).
Reported on the test split, by the fp_reply_data groups: the champion's π zero-shot (its stored probabilities), π recalibrated
(a temperature and a switch bias fitted on train), and π_FP.
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


def grouped(probs, arrays, rows):
    labels, legal, turn, forced = (arrays[k][rows] for k in ('label', 'legal', 'turn', 'forced'))
    can_switch = legal[:, data_mod.MOVES:].any(1);can_move = legal[:, :data_mod.MOVES].any(1)
    groups = dict(all=np.ones(len(rows), bool), move_request=~forced, forced_switch=forced, free_choice=can_move & can_switch & ~forced,
                  label_move=labels < data_mod.MOVES, label_switch=labels >= data_mod.MOVES)
    for low, high in data_mod.TURN_BANDS:
        groups[f'turn {low}-{high}' if high < 10**9 else f'turn {low}+'] = (turn >= low) & (turn <= high)
    return {name: data_mod.metrics(probs[m], labels[m], legal[m]) for name, m in groups.items()}


def test_report(arrays, test, pi_fp_probs, calibrated):
    """The four reply models on the test rows: uniform over legal actions, the champion's π (its stored probabilities), π recalibrated,
    and π_FP."""
    legal = arrays['legal'][test]
    return dict(uniform_legal=grouped(legal / legal.sum(1, keepdims=True), arrays, test),
                champion_pi=grouped(arrays['probs'][test], arrays, test),
                champion_pi_recalibrated=grouped(calibrated(arrays['probs'][test], legal), arrays, test),
                pi_fp=grouped(pi_fp_probs, arrays, test))


def archive_parity(checkpoint, archive, arrays, rows, torch_probs, size=512):
    """Export π_FP to the champion_evaluator weight archive and check the JAX forward reproduces the torch probabilities."""
    import jax
    import jax.numpy as jnp
    from gpu_engine import champion
    meta = champion.export(checkpoint, archive)
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
    return dict(archive=str(archive), weights_archive_sha256=loaded['weights_archive_sha256'], checkpoint_sha256=meta['checkpoint_sha256'],
                rows=int(len(rows)), max_abs_probability_difference=worst, argmax_agreement=agree / len(rows) if len(rows) else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('data');parser.add_argument('out');parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epochs', type=int, default=8);parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch', type=int, default=256);parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--device', default='cuda');parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--evaluate', metavar='PI_FP_PT', help='skip training: report this trained π_FP on test')
    parser.add_argument('--export-archive', metavar='NPZ', help='with --evaluate: export π_FP to a champion_evaluator archive and check parity')
    args = parser.parse_args(argv)
    import torch
    from pokezero.neural_policy import checkpoint_file_sha256, load_transformer_checkpoint
    torch.manual_seed(args.seed);rng = np.random.default_rng(args.seed)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True);started = time.time()
    arrays = load(args.data)
    model, result = load_transformer_checkpoint(args.evaluate or args.checkpoint, map_location=args.device)
    window = int(result.model_config.window_size)
    index = windows(arrays['ordinal'], window)
    split = data_mod.split_of(arrays['seed'])
    rows = {name: np.flatnonzero(split == name) for name in ('train', 'val', 'test')}
    games = {name: int(len(np.unique(arrays['ordinal'][r]))) for name, r in rows.items()}
    print(json.dumps(dict(event='DATA', rows={k: int(len(v)) for k, v in rows.items()}, games=games, window=window)), flush=True)
    labels = arrays['label']
    calibration, calibrated = recalibrate(arrays['probs'][rows['train']], labels[rows['train']], arrays['legal'][rows['train']], torch)
    if args.evaluate:
        test = rows['test'];probs = predict(model, arrays, index, test, torch, args.device)
        report = dict(schema='v2s-fp-reply-model-eval.v1', data=str(args.data), base_checkpoint=str(args.checkpoint),
                      base_checkpoint_sha256=checkpoint_file_sha256(args.checkpoint), pi_fp=str(args.evaluate),
                      pi_fp_sha256=checkpoint_file_sha256(args.evaluate), window=window, rows={k: int(len(v)) for k, v in rows.items()},
                      games=games, split_rule='battle seed % 10: 0 test, 1 validation, else train', recalibration=calibration,
                      ece_bins=data_mod.ECE_BINS, test=test_report(arrays, test, probs, calibrated))
        if args.export_archive:
            if window != 1:raise SystemExit('champion_evaluator archives need window 1')
            report['archive'] = archive_parity(args.evaluate, args.export_archive, arrays, test, probs)
        (out / 'report-eval.json').write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps(dict(event='EVALUATED', test_all={k: v['all'] for k, v in report['test'].items()}, archive=report.get('archive'))), flush=True)
        return 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    test = rows['test']
    report = dict(schema='v2s-fp-reply-model.v1', data=str(args.data), checkpoint=str(args.checkpoint),
                  checkpoint_sha256=checkpoint_file_sha256(args.checkpoint), window=window, rows={k: int(len(v)) for k, v in rows.items()},
                  games=games, split_rule='battle seed % 10: 0 test, 1 validation, else train', args=vars(args), history=history,
                  best_epoch=best['epoch'], recalibration=calibration, ece_bins=data_mod.ECE_BINS,
                  test=test_report(arrays, test, predict(model, arrays, index, test, torch, args.device), calibrated),
                  seconds=round(time.time() - started, 1))
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    payload = dict(payload);payload['state_dict'] = best_state
    torch.save(payload, out / 'pi-fp.pt')
    report['saved'] = dict(path=str(out / 'pi-fp.pt'), sha256=checkpoint_file_sha256(out / 'pi-fp.pt'),
                           note='champion checkpoint format; policy fine-tuned on Foul Play decisions; the value head output is not trained')
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(event='DONE', best_epoch=best['epoch'], test_all={k: report['test'][k]['all'] for k in report['test']})), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
