"""S0 PROGRESS scheduling and scoring for the V2 engine evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist


PANEL_PAIRS = {'champion_9375': 4000, 'max_damage': 500}
CHAMPION_SHARDS = 36
INTERIM_SHARDS = (12, 28, 4, 24, 9, 1, 13, 21, 0, 10)
SEATS = ('p1', 'p2')
Z90 = NormalDist().inv_cdf(.95)


@dataclass(frozen=True)
class Cell:
    offset: int
    seed: int
    seat: int
    teams: tuple[str, str]


@dataclass(frozen=True)
class Terminal:
    seed: int
    seat: int
    requests: int
    capped: bool
    winner: int | None


def schedule(bank):
    """Keep the Showdown physical teams and seed fixed across the two seats."""
    if not bank:
        raise ValueError('a nonempty ordered team bank is required')
    cells = []
    start = bank[0]['seed']
    for offset, row in enumerate(bank):
        seed, teams = row['seed'], row['teams']
        if (type(seed) is not int or not 0 <= seed < 2**32 or seed != start + offset
                or type(row['offset']) is not int or row['offset'] != offset
                or set(teams) != set(SEATS) or any(type(t) is not str or not t for t in teams.values())):
            raise ValueError(f'invalid registered team bank row at offset {offset}')
        packed = tuple(teams[owner] for owner in SEATS)
        cells.extend(Cell(offset, seed, seat, packed) for seat in range(2))
    return cells


def disposition(*, live, refused, done, requests, cap):
    """Classify one tick for both scalar fixtures and batched device runs."""
    refused = live & refused
    eligible = live ^ refused
    completed = eligible & done
    capped = (eligible ^ completed) & (requests >= cap)
    return refused, completed, capped


def _summary(blocks):
    wins = sum(win for block in blocks for win, _ in block)
    caps = sum(cap for block in blocks for _, cap in block)
    games = sum(map(len, blocks))
    scores = [sum(win for win, _ in block) / len(block) for block in blocks]
    n = len(scores)
    center = (sum(scores) + .5) / (n + 1)
    variance = (sum((x - center)**2 for x in scores) + .5 * center**2 + .5 * (1 - center)**2) / n
    margin = Z90 * sqrt(variance / (n + 1))
    return dict(games=games, wins=wins, caps=caps, win_rate=wins / games,
                turn_cap_rate=caps / games, ci90=[max(0., center - margin), min(1., center + margin)])


def score(cells, outcomes, *, max_requests, interim=False):
    """Verify one terminal per cell, then score with one cluster per mirrored seed."""
    if type(max_requests) is not int or max_requests <= 0:
        raise ValueError('a positive request cap is required')
    expected = {(cell.seed, cell.seat) for cell in cells}
    if not cells or len(expected) != len(cells):
        raise ValueError('evaluation schedule must contain distinct cells')
    recorded = {}
    for terminal in outcomes:
        key = terminal.seed, terminal.seat
        if key not in expected or key in recorded:
            raise ValueError(f'unexpected or duplicate terminal for {key}')
        if (not 1 <= terminal.requests <= max_requests
                or (terminal.capped and (terminal.winner is not None or terminal.requests != max_requests))
                or (not terminal.capped and terminal.winner not in (0, 1, 2))):
            raise ValueError(f'invalid terminal for {key}')
        recorded[key] = terminal
    if recorded.keys() != expected:
        raise ValueError('evaluation has missing terminals')

    pairs = {}
    for cell in cells:
        terminal = recorded[cell.seed, cell.seat]
        pair = pairs.setdefault(cell.offset, {})
        if cell.seat in pair:
            raise ValueError('evaluation schedule must contain one cell per seat and offset')
        pair[cell.seat] = (cell, terminal)
    blocks = {}
    for offset, pair in pairs.items():
        if (set(pair) != {0, 1} or pair[0][0].seed != pair[1][0].seed
                or pair[0][0].teams != pair[1][0].teams):
            raise ValueError(f'invalid mirrored pair at offset {offset}')
        blocks[offset] = [(terminal.winner == seat, terminal.capped)
                          for seat, (_, terminal) in pair.items()]

    result = _summary(blocks.values())
    if interim:
        if blocks.keys() != set(range(PANEL_PAIRS['champion_9375'])):
            raise ValueError('registered interim subset requires the complete champion panel')
        subset = [block for offset, block in blocks.items() if offset % CHAMPION_SHARDS in INTERIM_SHARDS]
        result['registered_subset'] = _summary(subset)
    return result
