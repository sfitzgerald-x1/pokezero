"""Finite semantic-input accounting for the full-state fidelity panel.

This module proves cardinality and mass for a DECLARED observation partition.
It cannot prove that the declaration includes every use in a transition kernel.
That source/domain obligation belongs to the prospectively reviewed panel.
In particular, a passing sampled trace is not an inactivity certificate.

Each physical input coordinate occurs once. Multiple consumers of a reused
draw refine one partition; they are never multiplied as independent events.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
from typing import Callable, Hashable, Iterator, Mapping


DRAW_BOUNDS = (2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2)
# Names follow ChanceInputs, without importing JAX or mutating a native flag.
COORDINATES = tuple(
    [(f"draws.{side}.{column}", bound)
     for side in range(2) for column, bound in enumerate(DRAW_BOUNDS)]
    + [(f"{field}.{side}", bound)
       for field, bound in (("phazing", 60), ("secondary", 100),
                            ("sleep_talk", 60), ("entry_ties", 2))
       for side in range(2)]
    + [("initial_tie", 2)]
)


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Observation:
    """A primitive value or predicate actually consumed by a source rule."""
    kind: str
    argument: int | Fraction = 0

    def __post_init__(self):
        if self.kind not in {"identity", "less_than", "modulo"}:
            raise ValueError("unrecognized observation")
        if self.kind == "modulo":
            _positive_integer(self.argument, "modulus")
        elif self.kind == "less_than":
            if type(self.argument) not in {int, Fraction}:
                raise ValueError("threshold must be an exact integer or Fraction")
        elif self.argument != 0:
            raise ValueError("identity takes no argument")

    def value(self, draw: int):
        if self.kind == "identity":
            return draw
        if self.kind == "less_than":
            return draw < self.argument
        return draw % self.argument


@dataclass(frozen=True)
class Cell:
    representative: int
    members: tuple[int, ...]
    mass: Fraction


def partition(bound: int, observations: tuple[Observation, ...]) -> tuple[Cell, ...]:
    """Exhaust all primitive values, grouping only equal declared observations."""
    _positive_integer(bound, "bound")
    if not isinstance(observations, tuple) or any(not isinstance(o, Observation) for o in observations):
        raise ValueError("observations must be an immutable tuple of Observation")
    groups = defaultdict(list)
    for value in range(bound):
        groups[tuple(observer.value(value) for observer in observations)].append(value)
    return tuple(Cell(values[0], tuple(values), Fraction(len(values), bound))
                 for values in groups.values())


@dataclass(frozen=True)
class Axis:
    coordinate: str
    observations: tuple[Observation, ...]
    source_uses: tuple[str, ...] = ()
    inactive_reason: str | None = None

    def __post_init__(self):
        if self.coordinate not in dict(COORDINATES):
            raise ValueError(f"unknown coordinate: {self.coordinate}")
        if self.observations:
            if not self.source_uses or self.inactive_reason is not None:
                raise ValueError("active coordinates require source uses, not inactivity")
        elif not self.inactive_reason or self.source_uses:
            raise ValueError("inactive coordinates require an explicit domain reason")
        if not isinstance(self.source_uses, tuple) or any(not isinstance(s, str) or not s.strip() for s in self.source_uses):
            raise ValueError("source uses must be nonempty text references")
        partition(dict(COORDINATES)[self.coordinate], self.observations)


class CardinalityRefusal(ValueError):
    def __init__(self, required: int, cap: int):
        self.required, self.cap = required, cap
        super().__init__(f"REFUSED_CARDINALITY_CAP: required={required}, cap={cap}; no cases emitted")


@dataclass(frozen=True)
class Case:
    values: tuple[tuple[str, int], ...]
    mass: Fraction


@dataclass(frozen=True)
class Plan:
    axes: tuple[Axis, ...]

    def __post_init__(self):
        expected = [name for name, _ in COORDINATES]
        if not isinstance(self.axes, tuple) or any(not isinstance(a, Axis) for a in self.axes):
            raise ValueError("immutable complete coordinate plan required")
        if [axis.coordinate for axis in self.axes] != expected:
            raise ValueError("all physical coordinates required once in canonical order")

    def cells(self):
        return tuple(partition(bound, axis.observations)
                     for axis, (_, bound) in zip(self.axes, COORDINATES, strict=True))

    @property
    def cardinality(self):
        return math.prod(len(cells) for cells in self.cells())

    def bounded_cases(self, cap: int) -> Iterator[Case]:
        """Admission happens before returning an iterator, not after a prefix."""
        _positive_integer(cap, "case cap")
        cells = self.cells()
        count = math.prod(map(len, cells))
        if count > cap:
            raise CardinalityRefusal(count, cap)
        names = tuple(name for name, _ in COORDINATES)
        return (Case(tuple(zip(names, (cell.representative for cell in row), strict=True)),
                     math.prod((cell.mass for cell in row), start=Fraction(1)))
                for row in itertools.product(*cells))


def pushforward(plan: Plan, execute: Callable[[Case], Hashable], *, cap: int):
    """Complete declared law only; any execution/canonicalization refusal raises.

    A failed case is not dropped, rerolled, or renormalized. Batch execution can
    consume bounded_cases directly and apply the same mass accounting.
    """
    result = defaultdict(Fraction)
    mass = Fraction()
    for case in plan.bounded_cases(cap):
        result[execute(case)] += case.mass
        mass += case.mass
    if mass != 1:
        raise ValueError("incomplete input law")
    return dict(result)


def compare_laws(exact: Mapping[Hashable, Fraction], native: Mapping[Hashable, float], *, mass_tolerance: float):
    """Compare full canonical supports without normalizing native f32 mass."""
    if not exact or any(type(v) is not Fraction or v <= 0 for v in exact.values()) or sum(exact.values()) != 1:
        raise ValueError("positive exact rational unit-mass law required")
    if type(mass_tolerance) is not float or not math.isfinite(mass_tolerance) or mass_tolerance < 0:
        raise ValueError("explicit finite nonnegative mass tolerance required")
    if not native or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in native.values()):
        raise ValueError("positive finite native masses required; remove zero branches explicitly")
    native_mass = math.fsum(native.values())
    if abs(native_mass - 1) > mass_tolerance:
        raise ValueError(f"native mass is not within declared tolerance: {native_mass}")
    support = set(exact) | set(native)
    differences = [abs(float(exact.get(key, Fraction())) - native.get(key, 0.0)) for key in support]
    return dict(exact_only=frozenset(set(exact) - set(native)),
                native_only=frozenset(set(native) - set(exact)),
                max_probability_difference=max(differences, default=0.0),
                half_l1=math.fsum(differences) / 2,
                native_mass=native_mass, native_mass_residual=native_mass - 1)
