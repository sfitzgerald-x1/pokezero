"""Update a small local Showdown progress table and SVG from compact summaries."""
from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
import tempfile
import time

from .v5_progress_evaluation import INTERVAL_GAMES, OPPONENTS, SCHEMA

LABELS = dict(foul_play='Foul Play', champion_9375='Champion 9375', max_damage='Max damage')
COLORS = dict(foul_play='#2053a0', champion_9375='#a03651', max_damage='#1d7558')


def _atomic(path, data):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _validate(row):
    if (row['schema'] != SCHEMA or type(row['declared_games']) is not int
            or row['declared_games'] <= 0 or row['declared_games'] % INTERVAL_GAMES
            or set(row['panel']) != set(OPPONENTS)):
        raise ValueError('invalid progress milestone summary')
    for panel in row['panel'].values():
        counts = [panel[k] for k in ('games_completed', 'unknown', 'pending')]
        if any(type(x) is not int or x < 0 for x in counts) or sum(counts) != panel['games_declared']:
            raise ValueError('progress panel denominator differs')
        rate, ci = panel['win_rate'], panel['ci95']
        if counts[0] == 0:
            if rate is not None or ci is not None:
                raise ValueError('unobserved progress scores must be empty')
        elif (not 0 <= rate <= 1 or len(ci) != 2 or not 0 <= ci[0] <= ci[1] <= 1):
            raise ValueError('invalid progress proportion or interval')


def _format(panel):
    rate, bounds = panel['win_rate'], panel['ci95']
    score = '—' if rate is None else f'{rate:.1%} [{bounds[0]:.1%}, {bounds[1]:.1%}]'
    return (f"{score} · {panel['games_completed']} complete, {panel['unknown']} unknown, "
            f"{panel['pending']} pending")


def _table(rows, training_games, now):
    lines = ['# Real Showdown progress', '', 'Win rates use completed games. Ties are reported as non-wins.',
        '95% pointwise intervals group the two seats by seed. Unknown and pending games widen the interval.',
        'Lag runs from milestone publication to the last result or, for a pending milestone, the report time.',
        '', '| Milestone | Foul Play | Champion 9375 | Max damage | Queue wait | Eval lag |',
        '|---:|---|---|---|---:|---:|']
    expected = list(range(INTERVAL_GAMES, training_games+1, INTERVAL_GAMES))
    for games in sorted(set(expected) | set(rows)):
        row = rows.get(games)
        if row is None:
            lines.append(f'| {games:,} | Queued | Queued | Queued | — | — |')
        else:
            summary = ' | '.join(_format(row['panel'][name]) for name in OPPONENTS)
            lag = row['eval_lag_seconds'] if row['complete'] else max(0, now-row['published_unix'])
            wait = f"{row['queue_wait_seconds']/60:.1f} min" if row['work_started'] else '—'
            lines.append(f"| {games:,} | {summary} | {wait} | {lag/60:.1f} min |")
    if not expected and not rows:
        lines += ['', 'No milestones have been published yet.']
    completed = max((games for games, row in rows.items() if row['complete']), default=0)
    lines += ['', f'Training at {training_games:,} declared games. Latest fully reported milestone at {completed:,}.',
              f'{max(0, (training_games-completed)//INTERVAL_GAMES)} milestones behind the training frontier.',
              '', '[Win-rate plot](showdown-progress.svg)', '']
    return '\n'.join(lines)


def _svg(rows, training_games):
    max_games = max(training_games, max(rows, default=0), INTERVAL_GAMES)
    x = lambda games: 66 + 596 * games / max_games
    y = lambda rate: 208 - 155 * rate
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 254" role="img" '
             'aria-label="Win rates by declared training games with 95 percent confidence intervals">',
             '<rect width="700" height="254" fill="white"/>',
             '<g font-family="system-ui,sans-serif" fill="#17252e" font-size="12">',
             '<text x="66" y="22" font-size="16" font-weight="bold">Showdown win rate</text>']
    for rate in (0, .25, .5, .75, 1):
        parts += [f'<text x="58" y="{y(rate)+4:.2f}" text-anchor="end">{rate:.0%}</text>',
                  f'<path d="M66 {y(rate):.2f}H662" stroke="#d7dee3"/>']
    for i, name in enumerate(OPPONENTS):
        color = COLORS[name]
        points = [(games, row['panel'][name]) for games, row in sorted(rows.items())
                  if row['panel'][name]['win_rate'] is not None]
        if points:
            coords = ' '.join(f'{x(games):.2f},{y(panel["win_rate"]):.2f}' for games, panel in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}"/>')
        for games, panel in points:
            xx = x(games)
            low, high = panel['ci95']
            parts += [f'<path d="M{xx:.2f} {y(low):.2f}V{y(high):.2f}" stroke="{color}" opacity=".54"/>',
                      f'<circle cx="{xx:.2f}" cy="{y(panel["win_rate"]):.2f}" r="3" fill="{color}"/>']
        label_x = 66 + 145*i
        parts.append(f'<text x="{label_x}" y="241" fill="{color}">{escape(LABELS[name])}</text>')
    if not any(p['win_rate'] is not None for row in rows.values() for p in row['panel'].values()):
        parts.append('<text x="364" y="136" text-anchor="middle">No completed cells</text>')
    parts += ['<text x="66" y="224">0</text>',
              f'<text x="662" y="224" text-anchor="end">{max_games/INTERVAL_GAMES:g}M games</text>',
              '</g></svg>', '']
    return '\n'.join(parts)


def update_report(directory, summaries, *, training_games, now=None):
    if type(training_games) is not int or training_games < 0:
        raise ValueError('nonnegative declared training progress required')
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / 'showdown-progress.json'
    prior = json.loads(state.read_bytes()) if state.exists() else {'rows': [], 'training_games': 0}
    rows = {row['declared_games']: row for row in prior['rows']}
    for row in summaries:
        _validate(row)
        games = row['declared_games']
        if rows:
            first = next(iter(rows.values()))
            for k in ('protocol_sha256', 'declaration_sha256'):
                if row[k] != first[k]:
                    raise ValueError('report cannot mix run declarations or bank protocols')
        if games in rows and rows[games] != row:
            old = rows[games]
            if (old['dispatch_sha256'] != row['dispatch_sha256'] or old['complete']
                    or old['observed_unix'] >= row['observed_unix']):
                raise ValueError('older or changed milestone summary')
            if any(row['panel'][n][k] < old['panel'][n][k]
                   for n in OPPONENTS for k in ('games_completed', 'unknown')):
                raise ValueError('milestone outcomes cannot regress')
        rows[games] = row
    training_games = max(training_games, prior['training_games'], max(rows, default=0))
    data = dict(rows=[rows[games] for games in sorted(rows)], training_games=training_games)
    _atomic(state, json.dumps(data, indent=2, allow_nan=False) + '\n')
    _atomic(folder/'showdown-progress.md', _table(rows, training_games, time.time() if now is None else now))
    _atomic(folder/'showdown-progress.svg', _svg(rows, training_games))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--training-games', type=int, required=True)
    parser.add_argument('summaries', nargs='*', type=Path)
    args = parser.parse_args(argv)
    update_report(args.directory, [json.loads(path.read_bytes()) for path in args.summaries],
                  training_games=args.training_games)


if __name__ == '__main__':
    main()
