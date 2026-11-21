"""Initial learning-rate arithmetic shared by opt-in V2 schedules."""
from dataclasses import dataclass
import math


BASE_LEARNING_RATE = 5.9e-5
SCALING_RULES = ('linear', 'sqrt')


def scaled_peak_rate(global_batch_size, scaling):
    """Return the peak LR for a declared global batch, independent of epochs."""
    if type(global_batch_size) is not int or global_batch_size <= 0 or scaling not in SCALING_RULES:
        raise ValueError('positive global batch and linear or sqrt LR scaling required')
    factor = global_batch_size / 512
    return BASE_LEARNING_RATE * (factor if scaling == 'linear' else math.sqrt(factor))


def warmup_multiplier(completed_updates, warmup_updates):
    """Factor for the next global optimizer update on a validated clock."""
    if completed_updates >= warmup_updates:
        return 1.
    return (completed_updates + 1) / warmup_updates


@dataclass(frozen=True)
class LRStart:
    scaling: str
    warmup_updates: int = 0

    def __post_init__(self):
        if self.scaling not in SCALING_RULES or type(self.warmup_updates) is not int or self.warmup_updates < 0:
            raise ValueError('LR start requires linear or sqrt scaling and a nonnegative update count')

    def rate(self, peak, completed_updates):
        return peak * warmup_multiplier(completed_updates, self.warmup_updates)

    def validate_training(self, config):
        if (config.learning_rate_schedule != 'constant'
                or config.learning_rate != scaled_peak_rate(config.batch_size, self.scaling)):
            raise ValueError('LR start requires its scaled peak and constant base schedule')
