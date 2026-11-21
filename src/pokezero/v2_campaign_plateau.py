"""Off-loop plateau evidence on frozen, complete progress and self-anchor banks."""
from dataclasses import dataclass
import math
from statistics import median

import numpy as np

from .v2_campaign_schedule import MILLION


@dataclass(frozen=True)
class PlateauLook:
    declared_games: int
    regime_game: int
    pooled_slope_pp: float
    upper_80_pp: float
    anchor_wins: int

    @property
    def qualifies(self):
        return self.upper_80_pp < 2 and self.anchor_wins < 2120  # 53% of the fixed 4,000 games


def score_look(milestones, h2h_blocks, fp_blocks, *, regime_game, anchor_wins, anchor_games, bootstrap_seed,
               bootstrap_replicates=20_000):
    """Rows are matched seed/team blocks; H2H rows average the two seats.

    The caller verifies roster, outcome receipts and the exact 10M-earlier
    checkpoint identity before supplying the complete 4,000-game anchor.
    """
    games = tuple(milestones)
    if (len(games) != 10 or any(type(game) is not int for game in games)
            or type(regime_game) is not int or regime_game < 0 or games[0] <= regime_game
            or games[-1] % (5 * MILLION) or games != tuple(range(games[-1] - 9 * MILLION, games[-1] + 1, MILLION))):
        raise ValueError('ten complete consecutive 1M looks in one regime ending on a 5M boundary required')
    if (type(anchor_games) is not int or anchor_games != 4000 or type(anchor_wins) is not int
            or not 0 <= anchor_wins <= anchor_games):
        raise ValueError('complete registered 4,000-game self-anchor required')
    if (type(bootstrap_seed) is not int or bootstrap_seed < 0
            or type(bootstrap_replicates) is not int or bootstrap_replicates < 1):
        raise ValueError('explicit bootstrap seed and positive replicate count required')
    panel_slopes = []
    x = np.arange(10) - 4.5
    weights = 1000 * x / np.dot(x, x)  # probability units per million -> pp per 10M
    for values, roster in ((h2h_blocks, (0., .5, 1.)), (fp_blocks, (0., 1.))):
        panel = np.asarray(values)
        if panel.shape != (1000, 10) or not np.isin(panel, roster).all():
            raise ValueError('complete observations for each frozen 1,000-block roster required')
        # OLS is linear: resampling these slopes is exactly resampling each
        # block's full trajectory, with all ten observations kept together.
        panel_slopes.append(panel @ weights)
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_replicates)
    for offset in range(0, bootstrap_replicates, 256):
        size = min(256, bootstrap_replicates - offset)
        draws[offset:offset+size] = .5 * sum(
            panel[rng.integers(0, len(panel), size=(size, len(panel)))].mean(axis=1)
            for panel in panel_slopes)
    return PlateauLook(games[-1], regime_game, float(.5 * sum(x.mean() for x in panel_slopes)),
                       float(np.quantile(draws, .8)), anchor_wins)


def supports_advance(previous: PlateauLook, current: PlateauLook):
    """Two distinct consecutive qualifying looks can support a recorded decision."""
    return (current.declared_games - previous.declared_games == 5 * MILLION
            and current.regime_game == previous.regime_game and previous.qualifies and current.qualifies)


def churn_without_progress(kl_looks, pooled_slope_pp):
    """Check one complete look against its previous five KLs, with no stop effect."""
    values = tuple(kl_looks)
    if (len(values) != 6 or not math.isfinite(pooled_slope_pp)
            or any(not math.isfinite(value) or value < 0 for value in values)):
        raise ValueError('six valid consecutive KL looks and a finite pooled progress slope required')
    return pooled_slope_pp <= 0 and values[-1] > 2 * median(values[:-1])
