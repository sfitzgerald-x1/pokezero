"""Arm fairness for the DepthN Foul Play panel: Foul Play's realized search throughput by arm (pure functions).

Foul Play's 1,000 ms is wall clock, so its strength depends on CPU contention. The bridge's opponent_think rows record, per
Foul Play move, the MCTS work it realized within its budget (iterations_per_budget_second), the budget stratum it ran under
(4x500ms early, 2x1000ms later at the bridge's --search-parallelism 1: foulplay_bridge._foulplay_think_stratum) and the decision round. Waits are not a contention
reading: the bridge starts a wait only after our own decision returns, so a search arm's waits are shorter by construction,
and Foul Play stops at its budget whatever work it did. The metric is therefore iterations_per_budget_second only, and it
is measurable only when the bridge's iterations_observable header is true (declared up front: fairness_observable).
- Strata: budget stratum x turn band (1-3, 4-9, 10-19, 20+, phase_d_saved_roots.TURN_BANDS). A move's turn is the battle
  turn at its round, from the owner's capture (round r is the owner stream's r-th request line, actionable or wait; turn 0
  counts as 1); without a capture it is round + 1 (counted in band_from_round).
- Sparse cells: a stratum x band cell with fewer than min_cell_moves moves in either arm merges into the adjacent band of the
  same stratum (the next band; the previous one for the last), repeatedly, until every cell has enough moves per arm or the
  stratum is a single cell. The merge map is reported.
- Reading: per cell, the search/RAW ratio of mean iterations_per_budget_second; the cells combine with weights from the
  pooled (both-arm) move counts (the weighted ratio, difference = ratio - 1); the unstratified ratio beside it; each arm's
  iteration coverage (moves with a rate over Foul Play moves, telemetry failures included).
- Flag: |difference| > ratio_tolerance, or the coverages more than coverage_difference_max apart. After a benign
  acknowledgement of a flag with difference d0: |d| > max(ratio_tolerance, |d0| + rearm_margin), or a sign flip with
  |d| > ratio_tolerance; the coverage gap re-flags above max(coverage_difference_max, the acknowledged gap + rearm_margin).
"""
import json
import math
from pathlib import Path
from statistics import fmean

from .phase_d_saved_roots import TURN_BANDS, band as turn_band

RULE = dict(metric='ITERATIONS_PER_BUDGET_SECOND', strata='FOULPLAY_BUDGET_STRATUM_X_TURN_BAND',
            combine='POOLED_COUNT_WEIGHTED_RATIO_OF_MEANS', reference='RAW', ratio_tolerance=.05, coverage_difference_max=.05,
            min_moves_per_arm=1000, min_pairs=50, checkpoint_pairs=50, rearm_margin=.025, min_cell_moves=30)
# Foul Play's own schedule (fp/search/main.py search_time_num_battles_randombattles) at the bridge's --search-parallelism 1 and
# with no Showdown timer (no time pressure): 4 sampled battles x 500 ms while <= 3 opponent Pokemon are revealed and its active one
# has shown no move, else 2 x 1000 ms. (A bridge comment's '8x500ms' assumed parallelism 2.) The lead's per-stratum calibration
# floors: the early stratum is rare (~3 moves a game) but the one most exposed to starvation, so it stays gated at 60 moves.
CALIBRATION_STRATA = ('4x500ms', '2x1000ms')
CALIBRATION_FLOORS = {'4x500ms': 60, '2x1000ms': 300}
BANDS = tuple(f'{lo}+' if hi is None else f'{lo}-{hi}' for lo, hi in TURN_BANDS)


def round_turns(capture_path):
    """The battle turn at each round of one game (index = round), from the owner's owning-wire capture; None without one."""
    if capture_path is None or not Path(capture_path).exists():return None
    turns, turn = [], 0
    for raw in Path(capture_path).read_bytes().split(b'\n'):
        try:row = json.loads(raw)
        except ValueError:continue   # an interrupted last line
        if type(row) is not dict or row.get('event') != 'WIRE':continue
        for line in row.get('lines') or ():
            if line.startswith('|turn|'):
                value = line.split('|')[2]
                if value.isascii() and value.isdigit():turn = int(value)
            elif line.startswith('|request|'):turns.append(turn)
    return turns


def moves(result, turns=None):
    """One cell result's Foul Play moves: rated (stratum, band, iterations_per_budget_second), all moves (telemetry
    failures included), moves banded by round for want of a capture, and wait + overlap seconds (descriptive only)."""
    game = result.get('source_game') or {};rows = game.get('opponent_think') or ()
    rated, waits, from_round = [], [], 0
    for row in rows:
        if type(row.get('wait_seconds')) in (int, float) and type(row.get('overlap_seconds')) in (int, float):
            waits.append(row['wait_seconds'] + row['overlap_seconds'])
        rate, stratum, number = row.get('iterations_per_budget_second'), row.get('stratum'), row.get('round')
        if type(rate) not in (int, float) or isinstance(rate, bool) or not math.isfinite(rate) or type(stratum) is not str:continue
        if turns is not None and type(number) is int and 0 <= number < len(turns):turn = turns[number]
        else:turn = number + 1 if type(number) is int and number >= 0 else 1;from_round += 1
        rated.append((stratum, turn_band(max(1, turn)), float(rate)))
    failures = game.get('opponent_think_failures') or 0
    return dict(rated=rated, moves=len(rows) + (failures if type(failures) is int else 0), band_from_round=from_round, waits=waits)


def pool(items):
    """Several cells' moves() pooled into one arm's."""
    return dict(rated=[m for x in items for m in x['rated']], moves=sum(x['moves'] for x in items),
                band_from_round=sum(x['band_from_round'] for x in items), waits=[w for x in items for w in x['waits']])


def _merge(bands, least):
    """Bands of one stratum (each {'bands', 'raw', 'search'}, in band order) merged until every group has `least` moves
    per arm or one group is left."""
    groups = [dict(g) for g in bands]
    while len(groups) > 1:
        sparse = next((i for i, g in enumerate(groups) if min(len(g['raw']), len(g['search'])) < least), None)
        if sparse is None:break
        j = sparse + 1 if sparse + 1 < len(groups) else sparse - 1;lo, hi = sorted((sparse, j))
        groups[lo:hi + 1] = [dict(bands=groups[lo]['bands'] + groups[hi]['bands'], raw=groups[lo]['raw'] + groups[hi]['raw'],
                                  search=groups[lo]['search'] + groups[hi]['search'])]
    return groups


def reading(raw, search, rule=RULE):
    """The stratified, pooled-count-weighted search/RAW ratio of mean iterations_per_budget_second, with the merge map, the
    unstratified ratio and each arm's coverage."""
    strata = {}
    for arm, pooled in (('raw', raw), ('search', search)):
        for stratum, band_index, rate in pooled['rated']:
            cell = strata.setdefault(stratum, {}).setdefault(band_index, dict(bands=[band_index], raw=[], search=[]))
            cell[arm].append(rate)
    cells, merge_map, weight, total, excluded = [], {}, 0, 0., 0
    for stratum in sorted(strata):
        groups = _merge([strata[stratum][b] for b in sorted(strata[stratum])], rule['min_cell_moves'])
        merge_map[stratum] = [[BANDS[b] for b in g['bands']] for g in groups]
        for g in groups:
            n_raw, n_search = len(g['raw']), len(g['search'])
            mean_raw = fmean(g['raw']) if n_raw else None;mean_search = fmean(g['search']) if n_search else None
            ratio = mean_search / mean_raw if n_raw and n_search and mean_raw > 0 else None
            cells.append(dict(stratum=stratum, bands=[BANDS[b] for b in g['bands']], raw_moves=n_raw, search_moves=n_search,
                              raw_mean=mean_raw, search_mean=mean_search, ratio=ratio))
            if ratio is None:excluded += 1;continue
            weight += n_raw + n_search;total += (n_raw + n_search) * ratio
    rates = {arm: [m[2] for m in pooled['rated']] for arm, pooled in (('raw', raw), ('search', search))}
    weighted = None if not weight else total / weight
    unstratified = (fmean(rates['search']) / fmean(rates['raw']) if rates['raw'] and rates['search'] and fmean(rates['raw']) > 0 else None)
    coverage = {arm: (len(pooled['rated']) / pooled['moves'] if pooled['moves'] else None) for arm, pooled in (('RAW', raw), ('search', search))}
    gap = None if None in coverage.values() else abs(coverage['RAW'] - coverage['search'])
    return dict(metric=rule['metric'], weighted_ratio=weighted, difference=None if weighted is None else weighted - 1,
                unstratified_ratio=unstratified, coverage=coverage, coverage_difference=gap,
                moves=dict(RAW=len(raw['rated']), search=len(search['rated'])), all_moves=dict(RAW=raw['moves'], search=search['moves']),
                band_from_round=dict(RAW=raw['band_from_round'], search=search['band_from_round']), cells=cells, merge_map=merge_map,
                excluded_cells=excluded)


def flag(value, rule=RULE, acknowledged=None):
    """None, or why this reading flags (baseline, or re-armed after a benign acknowledgement)."""
    if value['weighted_ratio'] is None:return None
    d, gap = value['difference'], value['coverage_difference']
    if acknowledged is None:
        if gap is not None and gap > rule['coverage_difference_max']:return 'coverage'
        return 'ratio' if abs(d) > rule['ratio_tolerance'] else None
    prior, prior_gap = acknowledged['difference'] or 0., acknowledged.get('coverage_difference') or 0.
    if gap is not None and gap > max(rule['coverage_difference_max'], prior_gap + rule['rearm_margin']):return 'coverage'
    if abs(d) > max(rule['ratio_tolerance'], abs(prior) + rule['rearm_margin']):return 'drift'
    if d * prior < 0 and abs(d) > rule['ratio_tolerance']:return 'sign'
    return None


def descriptive_waits(raw, search):
    """When iterations are not observable: Foul Play's request-to-answer seconds (wait + overlap) per arm, descriptive only."""
    return {arm: dict(moves=len(p['waits']), mean_seconds=fmean(p['waits']) if p['waits'] else None) for arm, p in (('RAW', raw), ('search', search))}
