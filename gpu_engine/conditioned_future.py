"""Explicit suspended-state correspondence for the opt-in continuation oracle.

No collection, action selection or historical-panel admission. The original
construction certificate is required. A native refusal invalidates the whole
comparison, including any earlier positive branches; no fallback or filtering.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib import import_module
import json
import math

from . import future_state as mapping


CAPSULE_FIELDS = (
    "pending_residual", "forced_switch", "baton_pass", "queued_actor",
    "queued_actions", "queued_stall", "residual_draws", "phazing_draws",
    "secondary_draws", "sleep_talk_draws", "replacement_count", "turn", "keys",
)
_BOOLEANS = {"pending_residual", "forced_switch", "baton_pass"}
_DRAW_BOUNDS = (2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2)


@dataclass(frozen=True)
class SuspendedNativeInput:
    raw_native: str
    raw_jax: tuple
    capsule: tuple
    certificate: mapping.Certificate
    source_certificate: mapping.Certificate
    integrity_sha256: str


@dataclass(frozen=True)
class ConditionedFuture(mapping.CanonicalFuture):
    capsule: tuple = ()


def _binding_digest(raw_native, raw_jax, capsule, certificate, source_certificate):
    payload = [raw_native, raw_jax, capsule, asdict(certificate), asdict(source_certificate)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _check_binding(value):
    if (value.integrity_sha256 != _binding_digest(value.raw_native, value.raw_jax,
            value.capsule, value.certificate, value.source_certificate)
            or value.certificate.input_fields != value.raw_jax
            or _from_data(dict(value.raw_jax)) != value.capsule):
        raise ValueError("SUSPENSION_BINDING_CHANGED")


def _leaves(value):
    if isinstance(value, tuple):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _capsule(values):
    """Validate every typed coordinate without casting or replacing its value."""
    if set(values) != {*CAPSULE_FIELDS, "version", "schema"}:
        raise ValueError("CAPSULE_FIELDS_MISMATCH")
    if type(values["version"]) is not int or values["version"] != 1 or values["schema"] != "pokezero.suspended-turn":
        raise ValueError("CAPSULE_SCHEMA_MISMATCH")
    result = {}
    for field in CAPSULE_FIELDS:
        value = mapping._freeze(values[field])
        if mapping._shape(value) != mapping.SHAPES[field]:
            raise ValueError("CAPSULE_FIELD_SHAPE: " + field)
        wanted = bool if field in _BOOLEANS else int
        if any(type(v) is not wanted for v in _leaves(value)):
            raise ValueError("CAPSULE_FIELD_TYPE: " + field)
        result[field] = value
    for field in ("replacement_count", "turn", "keys"):
        if any(not 0 <= v < 2**32 for v in _leaves(result[field])):
            raise ValueError("CAPSULE_FIELD_RANGE: " + field)
    if result["queued_actor"] not in (-1, 0, 1):
        raise ValueError("INVALID_QUEUED_ACTOR")
    if any(not 0 <= v < 4 for v in result["queued_actions"]):
        raise ValueError("INVALID_QUEUED_SLOT")
    if result["queued_stall"] != (0, 0):
        raise ValueError("UNSUPPORTED_QUEUED_STALL")
    if any(not 0 <= v < bound for row in result["residual_draws"] for v, bound in zip(row, _DRAW_BOUNDS, strict=True)):
        raise ValueError("INVALID_SAVED_ORDINARY_DRAW")
    for field, bound in (("phazing_draws", 60), ("secondary_draws", 100), ("sleep_talk_draws", 60)):
        if any(not 0 <= v < bound for v in result[field]):
            raise ValueError("INVALID_SAVED_DRAW: " + field)
    return tuple((name, result[name]) for name in CAPSULE_FIELDS)


def _from_data(data):
    return _capsule({**{name: data[name] for name in CAPSULE_FIELDS},
                     "version": 1, "schema": "pokezero.suspended-turn"})


def suspended_input_from_jax(state, index, metadata, *, certificate, module=None, native_successor=None):
    """Bind an explicit suspension to retained original constructor provenance.

    This verifies physical construction and capsule types. The native API then
    checks its narrow mechanic/replacement domain before executing a transition.
    The ordinary constructor and ordinary canonicalizers continue to refuse
    suspended states. No pending field is cleared to get through their guards.
    """
    raw = ()
    native_wire = None
    try:
        data = mapping._sample(state, index)
        raw = mapping._freeze(data)
        if not isinstance(certificate, mapping.Certificate):
            raise ValueError("SUSPENSION_PROVENANCE_REQUIRED")
        required_static = {f"{side}.pokemon[{p}]" for side in mapping.SIDE_NAMES for p in range(6)}
        if set(dict(certificate.static_native)) != required_static or not certificate.input_fields:
            raise ValueError("INCOMPLETE_SUSPENSION_PROVENANCE")
        expected, gaps, static, last, damage = mapping._prepare(
            data, metadata, certificate, boundary="suspended")
        if (last, damage) != (certificate.conditional_last, certificate.conditional_damage):
            gaps.append(mapping.Gap("moves", "CONDITIONAL_MECHANICS_CHANGED", "Original constructor tracking context differs"))
        if gaps:
            return mapping.Refused(tuple(gaps), raw)
        capsule = _from_data(data)
        module = import_module("poke_engine") if module is None else module
        if native_successor is not None:
            if not isinstance(native_successor, ConditionedFuture) or native_successor.capsule != capsule:
                raise ValueError("NATIVE_SUCCESSOR_CAPSULE_MISMATCH")
            if (native_successor.certificate.static_native != certificate.static_native
                    or native_successor.certificate.metadata_sha256 != certificate.metadata_sha256):
                raise ValueError("NATIVE_SUCCESSOR_PROVENANCE_MISMATCH")
            native_wire = native_successor.raw_native
        else:
            native_wire = mapping._construct_native(expected, module).to_string()
        if type(native_wire) is not str:
            raise ValueError("NATIVE_WIRE_REQUIRED")
        physical = mapping._native_wire(native_wire)
        _pending_context(physical, capsule)
        alive = [any(mon["hp"] > 0 for mon in physical[side]["pokemon"]) for side in mapping.SIDE_NAMES]
        done = not all(alive)
        physical.update(derived_done=done,
            derived_winner=-1 if not done else 0 if alive[0] else 1 if alive[1] else 2,
            conditional_last=last, conditional_damage=damage)
        differences = mapping.field_differences(tuple(mapping._flat(expected)), tuple(mapping._flat(physical)))
        if differences:
            return mapping.Refused((mapping.Gap("native", "CONSTRUCTION_NOT_LOSSLESS", repr(differences)),), raw, native_wire)
        policies = dict(certificate.field_policies)
        paths = dict(certificate.field_native_paths)
        for name in CAPSULE_FIELDS:
            policies[name] = "active suspended-turn capsule; retained exactly"
            paths[name] = ("CONDITIONED_CAPSULE:" + name,)
        bound_certificate = replace(certificate, field_policies=tuple(sorted(policies.items())),
            field_native_paths=tuple(sorted(paths.items())), input_fields=raw,
            static_native=static, inactive_addresses=())
        if native_successor is not None:
            complete = _complete_key(mapping.CanonicalFuture(tuple(mapping._flat(expected)),
                bound_certificate, raw, native_wire), capsule)
            if complete.key != native_successor.key:
                raise ValueError("NATIVE_SUCCESSOR_KEY_MISMATCH")
        digest = _binding_digest(native_wire, raw, capsule, bound_certificate, certificate)
        return SuspendedNativeInput(native_wire, raw, capsule, bound_certificate, certificate, digest)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        return mapping.Refused((mapping.Gap("suspension", "SUSPENSION_INPUT_REFUSED", str(error)),), raw, native_wire)


def _complete_key(physical, capsule):
    if isinstance(physical, mapping.Refused):
        return physical
    # Complete physical state and all future-reused/external carry participate.
    # No capsule field can disappear merely because physical HP agrees.
    return ConditionedFuture(tuple(sorted((*physical.key,
        *mapping._flat({"capsule": dict(capsule)})))), physical.certificate,
        physical.raw_jax, physical.raw_native, capsule)


def _pending_context(physical, capsule):
    """Validate physical pending flags and exact queued identity together."""
    c = dict(capsule)
    sides = [physical[side] for side in mapping.SIDE_NAMES]
    forced = tuple(side["force_switch"] for side in sides)
    baton = tuple(side["baton_passing"] for side in sides)
    if not c["pending_residual"] or forced != c["forced_switch"] or baton != c["baton_pass"] or sum(forced) != 1:
        raise ValueError("SUSPENSION_FLAGS_MISMATCH")
    replacing = forced.index(True)
    active = sides[replacing]["pokemon"][int(sides[replacing]["active_index"])]
    direct = baton == forced and c["replacement_count"] == 0 and active["hp"] > 0
    chain = (not any(baton) and 1 <= c["replacement_count"] <= 5 and active["hp"] == 0
        and c["queued_actor"] == -1 and any(p["hp"] > 0 for p in sides[replacing]["pokemon"]))
    if not (direct or chain):
        raise ValueError("UNSUPPORTED_SUSPENSION_QUEUE")
    if c["queued_actor"] == -1:
        if c["queued_actions"] != (0, 0) or any(side["switch_out_move_second_saved_move"] != "none" for side in sides):
            raise ValueError("UNSUPPORTED_EMPTY_QUEUE_CONTEXT")
    else:
        actor = c["queued_actor"]
        if actor != 1-replacing or sides[replacing]["switch_out_move_second_saved_move"] != "none":
            raise ValueError("UNSUPPORTED_SUSPENSION_QUEUE")
        waiter = sides[actor]
        move = waiter["pokemon"][int(waiter["active_index"])]["moves"][c["queued_actions"][actor]]["id"]
        if waiter["switch_out_move_second_saved_move"] != move:
            raise ValueError("QUEUED_COMMITMENT_MISMATCH")


def canonical_resumed_jax(state, index, metadata, *, native_input):
    if not isinstance(native_input, SuspendedNativeInput):
        return mapping.Refused((mapping.Gap("suspension", "INPUT_NOT_BOUND", "Suspended input required"),))
    try:
        _check_binding(native_input)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return mapping.Refused((mapping.Gap("suspension", "SUSPENSION_BINDING_CHANGED", str(error)),),
            native_input.raw_jax, native_input.raw_native)
    try:
        data = mapping._sample(state, index)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return mapping.Refused((mapping.Gap("suspension", "INVALID_SUCCESSOR", str(error)),))
    if data["pending_residual"]:
        try:
            expected, gaps, _, last, damage = mapping._prepare(data, metadata,
                native_input.certificate, boundary="suspended")
            if (last, damage) != (native_input.certificate.conditional_last, native_input.certificate.conditional_damage):
                raise ValueError("CONDITIONAL_MECHANICS_CHANGED")
            if gaps:
                return mapping.Refused(tuple(gaps), mapping._freeze(data))
            physical = mapping.CanonicalFuture(tuple(mapping._flat(expected)),
                native_input.certificate, mapping._freeze(data))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            return mapping.Refused((mapping.Gap("suspension", "UNSUPPORTED_PENDING_SUCCESSOR", str(error)),),mapping._freeze(data))
    else:
        physical = mapping.canonical_jax_future(state, index, metadata, certificate=native_input.certificate)
    try:
        return _complete_key(physical, _from_data(mapping._sample(state, index)))
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return mapping.Refused((mapping.Gap("capsule", "INVALID_SUCCESSOR_CAPSULE", str(error)),),
            physical.raw_jax)


def native_successors(native_input, replacement, entry_ties, *, module=None):
    """Return every positive successor, or a refusal for the entire request.

    The replacement is supplied by the caller; this function chooses no action.
    Unsupported native context is never retried with ordinary enumeration.
    """
    if not isinstance(native_input, SuspendedNativeInput):
        return mapping.Refused((mapping.Gap("suspension", "INPUT_NOT_BOUND", "Suspended input required"),))
    try:
        request = mapping._freeze({"replacement_index": replacement, "entry_ties": entry_ties})
    except (ValueError, TypeError):
        request = (("replacement_index", repr(replacement)), ("entry_ties", repr(entry_ties)))
    offending = None
    def refuse(code, detail):
        if offending is not None:
            detail = json.dumps({"reason": detail, "offending_successor": offending}, sort_keys=True)
        return mapping.Refused((mapping.Gap("suspension", code, detail),),
            native_input.raw_jax, native_input.raw_native, raw_actions=request)
    try:
        if type(replacement) is not int or not 0 <= replacement < 6:
            raise ValueError("INVALID_REPLACEMENT_INDEX")
        ties = mapping._freeze(entry_ties)
        if not isinstance(ties, tuple) or len(ties) != 2 or any(type(v) is not int or v not in (0, 1) for v in ties):
            raise ValueError("INVALID_ENTRY_TIES")
        _check_binding(native_input)
        module = import_module("pokezero_search") if module is None else module
        if getattr(module, "CONDITIONED_ORACLE_ENABLED", False) is not True:
            raise ValueError("CONDITIONED_ORACLE_REQUIRED")
        capsule = dict(native_input.capsule, version=1, schema="pokezero.suspended-turn")
        report = json.loads(module.conditioned_resume(native_input.raw_native, json.dumps(capsule), replacement, ties))
        if set(report) != {"version", "branches"} or report["version"] != 1 or not isinstance(report["branches"], list):
            raise ValueError("INVALID_CONDITIONED_REPORT")
        results = []
        for branch in report["branches"]:
            offending = branch
            mass = branch["percentage"]
            if type(mass) not in (int, float) or not math.isfinite(mass) or mass <= 0:
                raise ValueError("INVALID_CONDITIONAL_MASS")
            try:
                carry = _capsule(branch["capsule"])
                if dict(carry)["pending_residual"]:
                    _pending_context(mapping._native_wire(branch["state"]), carry)
                    physical = mapping._canonical_native_future(branch["state"],
                        native_input.certificate, boundary="suspended")
                else:
                    physical = mapping.canonical_native_future(branch["state"], native_input.certificate)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                return refuse("UNSUPPORTED_POSITIVE_SUCCESSOR", str(error))
            if isinstance(physical, mapping.Refused):
                return refuse("UNSUPPORTED_POSITIVE_SUCCESSOR", repr(physical.gaps))
            if any(dict(carry)["baton_pass"]) and not dict(carry)["pending_residual"]:
                return refuse("UNSUPPORTED_POSITIVE_SUCCESSOR", "Successor is still suspended")
            results.append((mass / 100, _complete_key(physical, carry)))
        if not results or abs(math.fsum(mass for mass, _ in results) - 1) > 1e-6:
            raise ValueError("CONDITIONAL_MASS_NOT_ONE")
        return tuple(results)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        return refuse("CONDITIONED_CONTINUATION_REFUSED", str(error))
