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


KERNEL_MODULES = tuple("absorption action_history baton combat critical damage_moves effects encore entry focus_punch forecast infatuation items linked_traps mechanics phazing pressure pursuit residual_order secondary self_destruct sleep_talk solar_beam state sustain transform transition variable_moves weather".split())
# One whole-file sha256 per kernel module, in KERNEL_MODULES order (it replaced a single KERNEL_SHA256 over them).
KERNEL_PINS = {
    'absorption.py': '2d684602eb9aa69bc470b215c188533b0e1c4d8495dd5b71ca2508dc38f20623',
    'action_history.py': 'a47bdb2712f7eadaed12d91ede70be2ebe9f1c1ae3f70fe56495437997212407',
    'baton.py': 'f86ad12ae0d59c1c8f89eb3f84f2d0519e1214ae4a4506d5cb700c6e0c20fa01',
    'combat.py': '2d3ad693aebf879dc4b26cd992e9357fcc0e7a9fcf22bf1274e33869cd684aa7',
    'critical.py': '7f8be9394e5a5473df5e6e9e9dcd242947ac0970cf5f46d50e27b27658e58f72',
    'damage_moves.py': 'c2427b88dc7f7b643625659f12498e1c06412c6875c60d548fa573f6e7a45e72',
    'effects.py': '47f549d608b60d46e8cc4db7bad5be249ae728a4aab2a0ef28f105f3614e40e2',
    'encore.py': '63da39bbd181a8c6371b3bca853322291a03674c60f535ad0eacdbe1a440e0dd',
    'entry.py': '688cc4eaa0266f71faf22c78fa2d91f2ee0fd8b875171e88fe4c2304c66aa285',
    'focus_punch.py': '881bd17a84dc22576aa618256ada2b0f67bee29224dbb50508925b55e119585f',
    'forecast.py': '3c08324c4012ab60b8aa33f1a3d217c820783c4b85e0e0f5b0bdc2feade97bcf',
    'infatuation.py': '6d25d2cfe1f45e0cee4334e89b66ae2ff3ca92de141b2825d9345aec04bc3ffa',
    'items.py': '4a936d5d4e92036155bce8a1ae19005ca7555996cb23a6359cdd4c128fa93490',
    'linked_traps.py': '743c01ee69509162e7829ae555b85f069f1c20ec9bd19cec0779d066894bade4',
    'mechanics.py': 'fa63643594e8e01cae573c324fbd442fd82ddf63e4514bc23c2e3a4c337e3444',
    'phazing.py': '8004d4880991a074480b6a7e9be6f6cb980577a61d2388c4c8c56d9a1ab2f26a',
    'pressure.py': '73eefb16678fedf1885a4e11cbc8954d15706592742caa6b0b673fb4d63d698c',
    'pursuit.py': '31a6805d9a6b0032c849877efcb2b5a2447b9abc29fbe5170e687b0abe1fbb55',
    'residual_order.py': 'd49fbd0f47ffe4fa1c98bba49eeaacfb5e16518c2ef938aa7ff311d996a0df94',
    'secondary.py': 'a3058e796b8affc38f5aa0d188bf52cc554d6c07cc8b4e83ac792ed70734001a',
    'self_destruct.py': 'b3c9df0ea093eca7c0960b4c4ffd4277e289a0ded6f32f914466c0752fa9b739',
    'sleep_talk.py': '968a78ddde87158b13f68dfcacc2158f07336d82c1be472d2283421e8583c154',
    'solar_beam.py': '5463e103392e24df7cc560003740b7e425aba766a0d1bba90ff4428606849155',
    'state.py': '4cb9d910e8eb8cd70e80185fe4352132fd8933321a9fb00132623077b5993387',
    'sustain.py': '0b575a55f091312679058e0ab990116b1f2348f163654ae777e293c1e7b51b24',
    'transform.py': '5ff5f4613ec84b6ca7a72efec3ffd6b64dcf28d1f1672c633428a8e5dc45e20b',
    'transition.py': 'e86226e51f4ec2941d8ce793c6eaadae6deafaf591e2bd057eac1f03ce0e0a97',
    'variable_moves.py': '0d3655aac5b547782457be46ebfee0b8b66ece3a360d05d4ec34cebbf801ea6d',
    'weather.py': '3c861027e16d27917fa47fa5cf5d6dc0c68d3a74af282108dfb94369d4db57e5',
}
MAPPING_PINS = {
    'conditioned_future.py': 'bee60d69d535f4d9982c027f75ea6c49906787fc6c1754a9ac4d1a4548c2b8d6',
    'fidelity_panel.py': '728c1d14c3cc5e95de63d6d3770d143263d58c0d5d7dec8a444b64c5b829099f',
    'finite_law.py': '6adba87e46cc027d0b45fe4e4d5602dd36ac3bb00e66156774e0705fcabb8231',
    'future_state.py': 'a79bc6e0b9e18bf679404444e7d6ac5a148bd4e3a893b5628c909e307ed30bd2',
}


def _source_certificate():
    root = Path(mapping.__file__).parent
    if Path(conditioned.__file__).parent != root:
        raise ValueError("CONSUMER_SOURCE_ROOT_MISMATCH")
    if tuple(KERNEL_PINS) != tuple(name+".py" for name in KERNEL_MODULES):
        raise ValueError("CONSUMER_DOMAIN_DRIFT: KERNEL_MODULES")
    data = {name: (root/(name+".py")).read_bytes() for name in KERNEL_MODULES}
    for name in KERNEL_MODULES:
        if hashlib.sha256(data[name]).hexdigest() != KERNEL_PINS[name+".py"]:
            raise ValueError("CONSUMER_DOMAIN_DRIFT: " + name + ".py")
    rows = []
    for name, expected in MAPPING_PINS.items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("CONSUMER_MAPPING_DRIFT: " + name)
        rows.append((name, actual))
    rows.extend((name+".py", hashlib.sha256(data[name]).hexdigest())
        for name in KERNEL_MODULES)
    rows.append(("conditional_law.py", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return tuple(rows)


def _law():
    axes = []
    for name, _ in COORDINATES:
        if name.startswith("entry_ties."):
            axes.append(Axis(name, (Observation("identity"),),
                ("transition.step_with_chance: resumed switch installation/runSwitch ordering",)))
        elif name.startswith("effect."):
            # §6: the resumed step reads the saved block, never this fresh coordinate (transition.py:583-585,
            # effect_draws=jnp.where(resuming,state.effect_draws,..)). The block is bound with the root's raw input
            # and joins every suspended successor's key (conditioned_future.SAVED_BLOCK).
            axes.append(Axis(name, (), inactive_reason="A legal live pending-residual request replays its saved "
                "effect block (state.effect_draws), overriding this fresh coordinate"))
        else:
            reason = ("No unresolved initial entry; initial_tie is not consumed" if name == "initial_tie"
                else "A legal live pending-residual request uses its saved capsule, overriding this fresh coordinate")
            axes.append(Axis(name, (), inactive_reason=reason))
    return Plan(tuple(axes))


@dataclass(frozen=True)
class ConditionalActions:
    replacement: int | tuple
    replacing: int | tuple
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
        if sum(forced) not in (1, 2):
            return refuse("UNSUPPORTED_REPLACEMENT_REQUEST", "One or two requested replacements required")
        replacements = []
        for side, requested in enumerate(forced):
            action = original[side]
            if requested and (action < 4 or not legal[side][action]):
                return refuse("ILLEGAL_REPLACEMENT_REQUEST", repr(original))
            replacements.append([p for p in range(6) if p != data["active"][side]][action-4] if requested else None)
        replacing = forced.index(True) if sum(forced) == 1 else forced
        replacement = replacements[replacing] if sum(forced) == 1 else tuple(replacements)
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
