"""Whole-slot semantic chance laws for explicitly bound suspended roots.

Native cases are evaluated and cached after source/action/cap admission and
before JAX execution. This is native evaluation, not validation-only preflight.
The caller must freeze its prospective inputs first and attest both extensions.
This adapter selects no action and changes no collector or historical panel.
"""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from . import conditioned_future as conditioned
from . import future_state as mapping
from .fidelity_panel import WholeSlotUnknown, table_snapshot
from .finite_law import Axis, Case, COORDINATES, Observation, Plan


KERNEL_MODULES = tuple("absorption baton combat critical damage_moves effects encore entry focus_punch items linked_traps mechanics phazing pressure pursuit secondary self_destruct sleep_talk solar_beam state sustain transition variable_moves weather".split())
KERNEL_SHA256 = "2bb90563e06a858ba9a6cba991ad5bdc9ef06dcdb1af4c112da341503c6b2bfe"
MAPPING_PINS = {
    "future_state.py": "7203851b21944d9e742be32c38be65d088c3414673ea1a97a7b862f3e406c3a2",
    "conditioned_future.py": "274d8c20aa93eca9228b65e9d9b0eec172b302daaa91d9e969b93b67f2fe65a7",
    "finite_law.py": "d999cdbaddebc33cf3f84583e584e2110d7ee79f37bf365f715d0eb7f424ae8c",
    "fidelity_panel.py": "8bb00fde83082211c013e44ffe89865642d7b2d1d2e318649e39a41991901611",
}


def _source_certificate():
    root = Path(mapping.__file__).parent
    if Path(conditioned.__file__).parent != root:
        raise ValueError("CONSUMER_SOURCE_ROOT_MISMATCH")
    domain = hashlib.sha256(b"".join(name.encode()+b"\0"+(root/(name+".py")).read_bytes()+b"\0"
        for name in KERNEL_MODULES)).hexdigest()
    if domain != KERNEL_SHA256:
        raise ValueError("CONSUMER_DOMAIN_DRIFT")
    rows = []
    for name, expected in MAPPING_PINS.items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("CONSUMER_MAPPING_DRIFT: " + name)
        rows.append((name, actual))
    rows.extend((name+".py", hashlib.sha256((root/(name+".py")).read_bytes()).hexdigest())
        for name in KERNEL_MODULES)
    rows.append(("conditional_law.py", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return tuple(rows)


def _law():
    axes = []
    for name, _ in COORDINATES:
        if name.startswith("entry_ties."):
            axes.append(Axis(name, (Observation("identity"),),
                ("transition.step_with_chance: resumed switch installation/runSwitch ordering",)))
        else:
            reason = ("No unresolved initial entry; initial_tie is not consumed" if name == "initial_tie"
                else "A legal live pending-residual request uses its saved capsule, overriding this fresh coordinate")
            axes.append(Axis(name, (), inactive_reason=reason))
    return Plan(tuple(axes))


@dataclass(frozen=True)
class ConditionalActions:
    replacement: int
    replacing: int
    raw_actions: tuple
    raw_jax: tuple
    raw_native: str


@dataclass(frozen=True)
class NativeCase:
    case: Case
    branches: tuple


def _json_default(value):
    if type(value) is Fraction:
        return {"$fraction": [value.numerator, value.denominator]}
    raise TypeError("unbound conditional-plan value: " + type(value).__name__)


def _digest(plan):
    payload = asdict(plan)
    payload.pop("integrity_sha256")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=_json_default).encode()).hexdigest()


@dataclass(frozen=True)
class ConditionalConsumerPlan:
    law: Plan
    native_input: conditioned.SuspendedNativeInput
    native_actions: ConditionalActions
    source_certificate: tuple
    table_certificate: tuple
    metadata: dict
    cap: int
    native_cases: tuple[NativeCase, ...]
    integrity_sha256: str = ""

    @property
    def cardinality(self):
        return self.law.cardinality

    @property
    def axes(self):
        return self.law.axes

    def _verify(self):
        if self.integrity_sha256 != _digest(self):
            raise WholeSlotUnknown("CONDITIONAL_PLAN_BINDING_CHANGED", "Frozen root/action/table/provenance/case cache changed")
        conditioned._check_binding(self.native_input)
        expected = tuple(self.law.bounded_cases(self.cap))
        if len(expected) != 4 or tuple(row.case for row in self.native_cases) != expected:
            raise WholeSlotUnknown("INCOMPLETE_CONDITIONAL_NATIVE_CASES", "All four native cases must be retained in canonical order")
        if sum((row.case.mass for row in self.native_cases), Fraction()) != 1:
            raise WholeSlotUnknown("INVALID_NATIVE_OUTER_MASS", "No native outer-case normalization is allowed")

    def bounded_cases(self):
        self._verify()
        return self.law.bounded_cases(self.cap)

    def canonical_successor(self, state, index):
        return conditioned.canonical_resumed_jax(state, index, self.metadata, native_input=self.native_input)

    def native_results(self):
        self._verify()
        return tuple((float(row.case.mass)*probability, future)
            for row in self.native_cases for probability, future in row.branches)

    def native_evidence(self):
        """Retain each case and its native carry even when outcome keys merge."""
        self._verify()
        return tuple(asdict(row) for row in self.native_cases)


def plan_for_suspension(state, index, actions, table, metadata, *, certificate,
                        cap, native_module=None, search_module=None):
    """Bind legality and all four fresh entry-tie cases before JAX executes.

    A batched kernel state and explicit nonnegative sample index are required.
    Saved ordinary/auxiliary draws remain active in the immutable capsule;
    they are not new independent observations and are never marginalized out.
    Unsupported native cases return a whole-slot refusal, not a partial law.
    """
    raw, original = (), ()
    completed = []
    def refuse(code, detail):
        evidence = dict(reason=detail, completed_native_cases=len(completed),
            native_case_evidence=[asdict(row) for row in completed])
        return mapping.Refused((mapping.Gap("conditional_plan", code,
            json.dumps(evidence, sort_keys=True, default=_json_default)),), raw,
            raw_actions=original)
    try:
        if type(cap) is not int or cap <= 0:
            return refuse("INVALID_EXPLICIT_CASE_CAP", "Positive explicit per-slot case cap required")
        if type(index) is not int or index < 0:
            return refuse("INVALID_EXPLICIT_SAMPLE_INDEX", "Batched kernel state and explicit nonnegative sample index required")
        data = mapping._sample(state, index)
        raw, original = mapping._freeze(data), mapping._freeze(actions)
        law = _law()
        if law.cardinality > cap:
            return refuse("CHANCE_CARDINALITY_CAP", dict(required=law.cardinality, cap=cap))
        sources = _source_certificate()
        if not isinstance(original, tuple) or len(original) != 2 or any(type(a) is not int or not 0 <= a <= 8 for a in original):
            return refuse("INVALID_NINE_ACTION_REQUEST", "Two exact integer action slots required")
        if not data["pending_residual"] or data["done"]:
            return refuse("NO_LIVE_SUSPENSION", "Legal live resumption is required for the saved-draw proof")
        from .transition import action_mask
        # The actual supplied table participates in legality. No native-value
        # equality test is used to hide deterministic table divergences.
        mask = action_mask(state, table)
        legal = mask.tolist() if index is None else mask[index].tolist()
        forced = tuple(bool(data["forced_switch"][s]) or data["hp"][s][data["active"][s]] <= 0 for s in range(2))
        if sum(forced) != 1:
            return refuse("UNSUPPORTED_REPLACEMENT_REQUEST", "Exactly one requested replacement required")
        replacing = forced.index(True)
        action = original[replacing]
        if action < 4 or not legal[replacing][action]:
            return refuse("ILLEGAL_REPLACEMENT_REQUEST", repr(original))
        replacement = [p for p in range(6) if p != data["active"][replacing]][action-4]
        native = conditioned.suspended_input_from_jax(state, index, metadata,
            certificate=certificate, module=native_module)
        if isinstance(native, mapping.Refused):
            return native
        bound_actions = ConditionalActions(replacement, replacing, original, raw, native.raw_native)
        frozen_table = mapping._freeze(table_snapshot(table))
        for case in law.bounded_cases(cap):
            values = dict(case.values)
            branches = conditioned.native_successors(native, replacement,
                (values["entry_ties.0"], values["entry_ties.1"]), module=search_module)
            if isinstance(branches, mapping.Refused):
                return refuse("CONDITIONAL_NATIVE_CASE_REFUSED", dict(case=asdict(case), refusal=asdict(branches)))
            completed.append(NativeCase(case, branches))
        plan = ConditionalConsumerPlan(law, native, bound_actions, sources, frozen_table,
            json.loads(json.dumps(metadata)), cap, tuple(completed))
        plan = replace(plan, integrity_sha256=_digest(plan))
        plan._verify()
        return plan
    except OSError as error:
        return refuse("CONSUMER_SOURCE_REFUSED", str(error))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        return refuse("CONDITIONAL_PLAN_REFUSED", str(error))
