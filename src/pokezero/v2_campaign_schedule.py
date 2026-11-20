"""Horizon-free policy for the stable and cooldown phases of V2 training."""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Protocol

from .v5_contract import _binding


class LRStart(Protocol):
    """The LR-start owner resolves scaling and computes the warmup rate."""

    @property
    def warmup_updates(self) -> int: ...

    @property
    def scaling(self) -> Literal['linear', 'sqrt']: ...

    @property
    def initial_lr(self) -> float:
        """The already-scaled peak LR, before applying warmup."""
        ...

    def rate_at(self, completed_updates: int) -> float:
        """Rate of the next update. Warmup update W reaches initial_lr."""
        ...


def _positive_rate(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _validate_start(start):
    if (type(start.warmup_updates) is not int or start.warmup_updates <= 0
            or start.scaling not in ('linear', 'sqrt') or not _positive_rate(start.initial_lr)):
        raise ValueError('positive global warmup, linear/sqrt scaling and finite scaled peak LR required')


@dataclass(frozen=True)
class CheckpointAnchor:
    """Values read from a verified, completed parent checkpoint by its loader."""

    sha256: str
    declared_games: int
    completed_updates: int
    applied_lr: float

    def __post_init__(self):
        _binding(self.sha256)
        if (any(type(x) is not int or x <= 0 for x in (self.declared_games, self.completed_updates))
                or not _positive_rate(self.applied_lr)):
            raise ValueError('positive committed games, completed updates and applied LR required')
        object.__setattr__(self, 'applied_lr', float(self.applied_lr))


@dataclass(frozen=True)
class Cooldown:
    parent: CheckpointAnchor
    end_game: int

    def __post_init__(self):
        parent = CheckpointAnchor(**self.parent) if type(self.parent) is dict else self.parent
        if type(parent) is not CheckpointAnchor or type(self.end_game) is not int or self.end_game <= parent.declared_games:
            raise ValueError('finite cooldown must end after the committed parent boundary')
        object.__setattr__(self, 'parent', parent)

    def rate_at_game(self, game):
        origin = self.parent.declared_games
        if not origin <= game <= self.end_game:
            raise ValueError('game coordinate is outside the finite cooldown')
        fraction = (game - origin) / (self.end_game - origin)
        return self.parent.applied_lr * (1 - .9 * fraction)


@dataclass(frozen=True)
class CampaignSchedule:
    step_down_games: tuple[int, ...] = ()
    cooldown: Cooldown | None = None

    def __post_init__(self):
        if type(self.step_down_games) not in (tuple, list):
            raise ValueError('ordered absolute step-down games required')
        knots = tuple(self.step_down_games)
        if (len(knots) > 2 or any(type(game) is not int or game <= 0 for game in knots)
                or tuple(sorted(set(knots))) != knots):
            raise ValueError('at most two distinct increasing positive step-down games required')
        object.__setattr__(self, 'step_down_games', knots)
        cooldown = Cooldown(**self.cooldown) if type(self.cooldown) is dict else self.cooldown
        if cooldown is not None and (type(cooldown) is not Cooldown
                or any(game >= cooldown.parent.declared_games for game in knots)):
            raise ValueError('cooldown includes only step-downs applied before its parent boundary')
        object.__setattr__(self, 'cooldown', cooldown)

    def fork_cooldown(self, start: LRStart, parent: CheckpointAnchor, *, end_game):
        """Freeze a child schedule without changing the parent's future knots."""
        _validate_start(start)
        if self.cooldown is not None or type(parent) is not CheckpointAnchor:
            raise ValueError('cooldown needs an original completed campaign checkpoint')
        if parent.completed_updates < start.warmup_updates:
            raise ValueError('cooldown cannot start before warmup completes')
        inherited = tuple(game for game in self.step_down_games if game < parent.declared_games)
        if parent.applied_lr != start.initial_lr * .3 ** len(inherited):
            raise ValueError('parent applied LR differs from its stable campaign rate')
        return CampaignSchedule(inherited, Cooldown(parent, end_game))

    def validate_next_window(self, start: LRStart, window, completed_updates):
        """Validate a newly bound window. Resume reuses the original cursor/window."""
        _validate_start(start)
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError('nonnegative global completed-update cursor required')
        if any(window.start_game < game < window.end_game for game in self.step_down_games):
            raise ValueError('campaign step-down must be at a declared iteration boundary')
        if completed_updates < start.warmup_updates and any(game <= window.start_game for game in self.step_down_games):
            raise ValueError('campaign step-down cannot precede completed warmup')
        decay = self.cooldown
        if decay is not None:
            parent = decay.parent
            if (window.start_game < parent.declared_games or window.end_game > decay.end_game
                    or completed_updates < parent.completed_updates
                    or window.start_game == parent.declared_games and completed_updates != parent.completed_updates):
                raise ValueError('new window or update cursor is outside the finite cooldown lineage')

    def window_rate(self, start: LRStart, window, *, completed_updates, offset, window_updates):
        """Rate at an update's left edge in a bound window, independent of total games."""
        decay = self.cooldown
        if decay is not None and window.start_game >= decay.parent.declared_games:
            if (type(offset) is not int or type(window_updates) is not int or not 0 <= offset < window_updates
                    or completed_updates < decay.parent.completed_updates):
                raise ValueError('invalid update cursor within the cooldown window')
            elapsed = (window.end_game - window.start_game) * offset / window_updates
            return decay.rate_at_game(window.start_game + elapsed)
        if completed_updates < start.warmup_updates:
            return start.rate_at(completed_updates)
        steps = sum(game <= window.start_game for game in self.step_down_games)
        return start.initial_lr * .3 ** steps
