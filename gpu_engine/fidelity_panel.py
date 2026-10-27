"""Whole-slot exact-law collection for a prospectively fixed fidelity panel.

This is an implementation comparator, not independent simulator ground truth.
Constructor/cap/positive-successor refusals retain UNKNOWN unit mass for the
entire fixed slot. A callback may never filter or normalize its outcomes.
The owned run supervisor, not this module, enforces the operation-time deadline.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict
from fractions import Fraction
import itertools
import math
import os

from . import future_state as mapping
from .finite_law import CardinalityRefusal, compare_laws


class WholeSlotUnknown(Exception):
    def __init__(self, category, detail):
        self.category, self.detail = category, detail
        super().__init__(category)


def require_future(value):
    # Refused is hashable. Accepting it as a key would silently forge a law.
    if isinstance(value, mapping.Refused):
        raise WholeSlotUnknown("UNSUPPORTED_POSITIVE_SUCCESSOR", asdict(value))
    if not isinstance(value, mapping.CanonicalFuture):
        raise TypeError("complete canonical successor or explicit refusal required")
    return value


def chunks(iterator, size):
    if type(size) is not int or size <= 0:
        raise ValueError("positive integer execution batch required")
    while batch := tuple(itertools.islice(iterator, size)):
        yield batch


def table_snapshot(value):
    if hasattr(value, "_asdict"):
        value = value._asdict()
    if isinstance(value, dict):
        return {name: table_snapshot(item) for name, item in value.items()}
    return value.tolist() if hasattr(value, "tolist") else value


def compare_slot(plan, execute_cases, execute_native, *, batch_size,
                 mass_tolerance, probability_tolerance):
    """Collect one COMPLETE law or a named whole-slot unknown.

    execute_cases must return one unfiltered successor per supplied Case.
    execute_native yields every (unmodified probability, canonical successor).
    Zero-mass native branches are counted explicitly, never canonicalized.
    Unexpected implementation exceptions propagate and stop the owned run.
    """
    for name, value in (("mass", mass_tolerance), ("probability", probability_tolerance)):
        if type(value) is not float or not math.isfinite(value) or value < 0:
            raise ValueError(f"explicit finite nonnegative {name} tolerance required")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("positive integer execution batch required")
    if isinstance(plan, mapping.Refused):
        return dict(status="UNKNOWN", category="INPUT_OR_PLAN_REFUSED",
                    detail=asdict(plan), identified_mass="0", unknown_mass="1")
    exact, native = defaultdict(Fraction), defaultdict(float)
    representatives = {}
    completed_cases = 0
    zero_native = 0
    try:
        # Admission happens before either engine executes any transition.
        iterator = plan.bounded_cases()
        for batch in chunks(iterator, batch_size):
            results = tuple(execute_cases(batch))
            if len(results) != len(batch):
                raise RuntimeError("case executor dropped or added successors")
            for case, result in zip(batch, results, strict=True):
                future = require_future(result)
                exact[future.key] += case.mass
                representative = representatives.setdefault(future.key, {})
                if "jax" not in representative:
                    representative["jax"] = asdict(future)
                completed_cases += 1
        if completed_cases != plan.cardinality or sum(exact.values()) != 1:
            raise RuntimeError("incomplete exact law; no prefix may be accepted")
        for probability, result in execute_native():
            if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                    or not math.isfinite(probability) or probability < 0):
                raise WholeSlotUnknown("INVALID_NATIVE_PROBABILITY", repr(probability))
            if probability == 0:
                zero_native += 1
                continue
            future = require_future(result)
            native[future.key] += probability
            representative = representatives.setdefault(future.key, {})
            if "native" not in representative:
                representative["native"] = asdict(future)
        if not native or abs(math.fsum(native.values()) - 1) > mass_tolerance:
            raise WholeSlotUnknown("INVALID_NATIVE_MASS", dict(
                mass=math.fsum(native.values()), tolerance=mass_tolerance))
        comparison = compare_laws(exact, native, mass_tolerance=mass_tolerance)
    except CardinalityRefusal as error:
        return dict(status="UNKNOWN", category="CASE_CAP", required=error.required,
                    cap=error.cap, completed_cases=0, identified_mass="0", unknown_mass="1")
    except WholeSlotUnknown as error:
        return dict(status="UNKNOWN", category=error.category, detail=error.detail,
                    completed_cases=completed_cases, identified_mass="0", unknown_mass="1",
                    partial_laws_accepted=False)
    support_equal = not comparison["exact_only"] and not comparison["native_only"]
    agrees = support_equal and comparison["max_probability_difference"] <= probability_tolerance
    return dict(status="AGREE" if agrees else "DIVERGE", identified_mass="1", unknown_mass="0",
        exact_cases=completed_cases, native_zero_branches=zero_native,
        support_equal=support_equal, probability_tolerance=probability_tolerance,
        probability_criterion="approximate per-key tolerance, NOT exact arithmetic equality",
        **{k: v for k, v in comparison.items() if k not in {"exact_only", "native_only"}},
        outcomes=[dict(key=key, jax_probability=str(exact.get(key, Fraction())),
            native_probability=native.get(key, 0.0), raw_representatives=representatives[key])
            for key in sorted(set(exact) | set(native), key=repr)])


class _PhysicalSuccessorCache:
    """Root-local canonicalization memo; no cases or probabilities are removed.

    A key includes every returned state field's name, dtype, shape and exact
    bytes. The bound mapping context is checked once per batch; changing it
    clears the memo so existing correspondence/refusal checks run again.
    """
    def __init__(self, fields, batch_size, *, limit=1024):
        if type(limit) is not int or limit <= 0:
            raise ValueError("positive canonical successor cache bound required")
        self.fields, self.size, self.limit = tuple(fields), batch_size, limit
        self.context = None
        self.values = OrderedDict()

    def canonicalize(self, host, count, plan):
        import numpy as np
        data = host._asdict()
        if tuple(data) != self.fields or not 0 < count <= self.size:
            raise ValueError("complete fixed-size successor state required")
        context = (mapping._metadata_sha(plan.metadata), plan.native_input.certificate)
        if context != self.context:
            self.values.clear()
            self.context = context
        arrays = tuple(np.asarray(data[name]) for name in self.fields)
        if any(a.ndim == 0 or a.shape[0] != self.size or a.dtype.hasobject for a in arrays):
            raise ValueError("complete non-object successor arrays required")
        header = tuple((name, a.dtype.str, a.shape[1:])
                       for name, a in zip(self.fields, arrays, strict=True))
        rows = np.concatenate(tuple(np.ascontiguousarray(a).reshape(self.size, -1)
            .view(np.uint8).reshape(self.size, -1) for a in arrays), axis=1)
        results = []
        for index in range(count):
            key = (header, rows[index].tobytes())
            if key in self.values:
                future = self.values.pop(key)
            else:
                future = plan.canonical_successor(host, index)
            self.values[key] = future
            if len(self.values) > self.limit:
                self.values.popitem(last=False)
            results.append(future)
        return tuple(results)


class JaxCaseExecutor:
    """Fixed-size padded execution; padding never enters the probability law."""
    def __init__(self, state, actions, table, plan, *, batch_size):
        import jax
        import jax.numpy as jnp
        from .transition import step_with_chance
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("positive batch size required")
        if state.active.shape != (1, 2):
            raise ValueError("one exact bound root required")
        if mapping._freeze(mapping._sample(state, 0)) != plan.native_actions.raw_jax:
            raise ValueError("JAX root differs from the bound native/planner input")
        if mapping._freeze(actions) != plan.native_actions.raw_actions:
            raise ValueError("JAX actions differ from the bound native/planner actions")
        if mapping._freeze(table_snapshot(table)) != plan.table_certificate:
            raise ValueError("JAX table differs from the source consumer certificate")
        self.plan, self.size, self.jax, self.jnp = plan, batch_size, jax, jnp
        self.state = jax.tree.map(lambda x: jnp.repeat(x, batch_size, axis=0), state)
        self.actions = jnp.repeat(jnp.asarray([actions], jnp.int32), batch_size, axis=0)
        self.table = table
        self.compiled = jax.jit(step_with_chance)
        self.canonical_cache = _PhysicalSuccessorCache(self.state._fields, batch_size)

    def __call__(self, cases):
        import numpy as np
        from .transition import ChanceInputs
        if not cases or len(cases) > self.size:
            raise ValueError("nonempty case batch within declared size required")
        values = [dict(case.values) for case in cases]
        values += [values[0]] * (self.size - len(values))
        fields = {}
        fields["draws"] = self.jnp.asarray([[[v[f"draws.{s}.{c}"] for c in range(12)]
            for s in range(2)] for v in values], self.jnp.int32)
        for field in ("phazing", "secondary", "sleep_talk", "entry_ties"):
            fields[field] = self.jnp.asarray([[v[f"{field}.{s}"] for s in range(2)]
                for v in values], self.jnp.int32)
        fields["initial_tie"] = self.jnp.asarray([v["initial_tie"] for v in values], self.jnp.int32)
        result = self.jax.block_until_ready(self.compiled(
            self.state, self.actions, self.table, ChanceInputs(**fields)))
        if np.asarray(result.invalid_actions)[:len(cases)].any():
            raise WholeSlotUnknown("INVALID_JAX_ACTION", "bound requested action refused")
        host = self.jax.device_get(result.state)
        return self.canonical_cache.canonicalize(host, len(cases), self.plan)


def native_executor(plan):
    """No normalization, parent mutation or silent unsupported branch filtering."""
    if os.environ.get("POKEZERO_ENUMERATE_ROLLS") != "1":
        raise ValueError("isolated worker must enable exact rolls before native import")
    import poke_engine
    state = plan.native_input.state
    before = state.to_string()
    branches = poke_engine.generate_instructions(state, *plan.native_actions.choices)
    if state.to_string() != before:
        raise RuntimeError("native instruction generation mutated bound parent")
    results = []
    for branch in branches:
        probability = float(branch.percentage) / 100
        future = (mapping.canonical_native_future(state.apply_instructions(branch),
                  plan.native_input.certificate) if probability > 0 else None)
        results.append((probability, future))
    if state.to_string() != before:
        raise RuntimeError("native branch application mutated bound parent")
    return results
