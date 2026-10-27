"""Original constructor provenance, separate from state correspondence.

Capture after gender assignment and before entry callbacks. This record does
not initialize a state, clear entry_pending, or create a native certificate.
The ordinary post-entry builder certifies physical construction separately.
Pool/catalog/runtime identities must come from verified cohort admission.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import os
from types import MappingProxyType
from typing import NamedTuple

from . import future_state as mapping


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _hex(value, length):
    return type(value) is str and re.fullmatch(f"[a-f0-9]{{{length}}}", value) is not None


@dataclass(frozen=True)
class ConstructorPool:
    arrays: object
    metadata_json: str
    receipt_json: str
    receipt_sha256: str

    def metadata(self):
        return json.loads(self.metadata_json)


def load_constructor_pool(path):
    """Load one byte snapshot, retaining Nature/EVs and read-only constructor arrays."""
    import numpy as np
    with Path(path).open("rb") as stream:
        before=os.fstat(stream.fileno());payload=stream.read();after=os.fstat(stream.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if identity(before)!=identity(after) or len(payload)!=after.st_size:
        raise ValueError("CONSTRUCTOR_POOL_CHANGED_DURING_READ")
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        metadata=json.loads(archive["metadata_utf8"].tobytes())
        fields=(*_POOL_STATE_FIELDS,"nature","evs")
        if not set(fields)<=set(archive.files):raise ValueError("POOL_CONSTRUCTOR_FIELDS_MISSING")
        arrays={name:archive[name] for name in fields}
        for name in ("gender","happiness"):
            if name in archive.files:arrays[name]=archive[name]
    descriptions={}
    for name,array in arrays.items():
        if array.dtype.kind not in "iu":raise ValueError("INTEGER_CONSTRUCTOR_POOL_REQUIRED: "+name)
        # A flag on an owning ndarray can be reversed. An immutable bytes
        # backing prevents resetting writeable even through its base view.
        array=np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)
        arrays[name]=array
        descriptions[name]=dict(shape=list(array.shape),dtype=str(array.dtype),
            sha256=hashlib.sha256(array.tobytes()).hexdigest())
    receipt=dict(schema="pokezero.constructor-pool.v1",pool_sha256=hashlib.sha256(payload).hexdigest(),
        metadata_sha256=mapping._metadata_sha(metadata),arrays=descriptions)
    return ConstructorPool(MappingProxyType(arrays),_json(metadata),_json(receipt),_sha(receipt))


def _pool(pool,*,rehash=False):
    if not isinstance(pool,ConstructorPool):raise ValueError("VERIFIED_CONSTRUCTOR_POOL_REQUIRED")
    receipt=json.loads(pool.receipt_json)
    if _sha(receipt)!=pool.receipt_sha256 or mapping._metadata_sha(pool.metadata())!=receipt["metadata_sha256"]:
        raise ValueError("CONSTRUCTOR_POOL_RECEIPT_CHANGED")
    if set(pool.arrays)!=set(receipt["arrays"]):raise ValueError("CONSTRUCTOR_POOL_FIELDS_CHANGED")
    for name,array in pool.arrays.items():
        description=receipt["arrays"][name]
        if array.flags.writeable or list(array.shape)!=description["shape"] or str(array.dtype)!=description["dtype"]:
            raise ValueError("CONSTRUCTOR_POOL_STORAGE_CHANGED")
        if rehash and hashlib.sha256(array.tobytes()).hexdigest()!=description["sha256"]:
            raise ValueError("CONSTRUCTOR_POOL_CONTENT_CHANGED")
    return receipt


@dataclass(frozen=True)
class RuntimeAttestation:
    payload_json: str
    sha256: str


def attest_native_runtime():
    """Read both native consumers once at cohort admission, not per root."""
    consumers = {}
    for name in ("poke_engine", "pokezero_search"):
        package = importlib.import_module(name)
        binary = importlib.import_module(name+"."+name)
        location = Path(binary.__file__)
        wrappers = {str(p.relative_to(Path(package.__file__).parent)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(package.__file__).parent.rglob("*.py"))}
        consumers[name] = dict(binary_sha256=hashlib.sha256(location.read_bytes()).hexdigest(),
            python_sources=wrappers, version=importlib.metadata.version(name.replace("_", "-")))
    search = importlib.import_module("pokezero_search")
    payload = dict(schema="pokezero.native-runtime.v1", python=platform.python_version(),
        consumers=consumers, conditioned_oracle_enabled=getattr(search,"CONDITIONED_ORACLE_ENABLED",False),
        dependencies={name:importlib.metadata.version(name) for name in ("jax","jaxlib","numpy")})
    return RuntimeAttestation(_json(payload), _sha(payload))


def _runtime(attestation):
    if not isinstance(attestation, RuntimeAttestation) or _sha(json.loads(attestation.payload_json)) != attestation.sha256:
        raise ValueError("NATIVE_RUNTIME_ATTESTATION_CHANGED")
    payload = json.loads(attestation.payload_json)
    if set(payload.get("consumers", {})) != {"poke_engine","pokezero_search"}:
        raise ValueError("BOTH_NATIVE_CONSUMERS_REQUIRED")
    if payload.get("conditioned_oracle_enabled") is not True:
        raise ValueError("CONDITIONED_ORACLE_REQUIRED")
    return payload


def verify_native_runtime(attestation):
    _runtime(attestation)
    if attest_native_runtime() != attestation:
        raise ValueError("NATIVE_RUNTIME_IDENTITY_CHANGED")


@dataclass(frozen=True)
class ConstructorContext:
    cohort_id: str
    pool_sha256: str
    pool_receipt_sha256: str
    pool_metadata_sha256: str
    mapping_metadata_json: str
    mapping_metadata_sha256: str
    catalog_sha256: str
    runtime_sha256: str
    gender_policy_json: str | None
    gender_initializer_sha256: str
    sha256: str

    def metadata(self):
        return json.loads(self.mapping_metadata_json)


def constructor_context(pool, catalog, *, runtime, cohort_id, gender_policy=None):
    """Bind an immutable pool header to an independently captured weight catalog."""
    verify_native_runtime(runtime)
    pool_receipt=_pool(pool,rehash=True)
    pool_sha256=pool_receipt["pool_sha256"];metadata=pool.metadata()
    if type(cohort_id) is not str or not cohort_id:
        raise ValueError("EXPLICIT_CONSTRUCTOR_COHORT_REQUIRED")
    if catalog.get("schema") != "pokezero.gen3-constructor-catalog.v1":
        raise ValueError("CONSTRUCTOR_CATALOG_REQUIRED")
    provenance = catalog["provenance"]
    if (provenance["pool_sha256"] != pool_sha256
            or provenance["compiled_tree"] != metadata["compiled_tree"]
            or provenance["showdown_commit"] != metadata["source_commit_label"]
            or not _hex(provenance["showdown_commit"],40)
            or not _hex(provenance["compiled_tree"]["sha256"],64)):
        raise ValueError("CONSTRUCTOR_CATALOG_SOURCE_MISMATCH")
    for name in ("constructor_catalog.cjs", "gender_policy.cjs"):
        if provenance["generator_sources"][name] != hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest():
            raise ValueError("CONSTRUCTOR_CATALOG_GENERATOR_CHANGED")
    if catalog["species"] != metadata["vocabulary"]["species"] or len(catalog["weights"]) != len(catalog["species"]):
        raise ValueError("CONSTRUCTOR_CATALOG_VOCABULARY_MISMATCH")
    unresolved=[name for name,weight in zip(catalog["species"][1:],catalog["weights"][1:],strict=True) if weight is None]
    if catalog.get("unresolved")!=unresolved or any(w is not None and (type(w) not in (int,float)
            or not math.isfinite(w) or w<=0) for w in catalog["weights"][1:]):
        raise ValueError("INVALID_CONSTRUCTOR_WEIGHT_CATALOG")
    enriched = json.loads(_json(metadata))
    enriched["oracle_species"] = {name:dict(weight_kg=weight) for name,weight
        in zip(catalog["species"][1:],catalog["weights"][1:],strict=True)}
    if gender_policy is not None:
        from .gender import validate_policy
        validate_policy(gender_policy,metadata["vocabulary"]["species"])
        source=gender_policy["provenance"]
        if (source["pool_sha256"]!=pool_sha256 or source["compiled_tree"]!=metadata["compiled_tree"]
                or source["showdown_commit"]!=metadata["source_commit_label"]):
            raise ValueError("GENDER_POLICY_SOURCE_MISMATCH")
        if set(source["generator_sources"])!={"gender_policy.cjs","generate_teams.cjs"} or any(
                value!=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name,value in source["generator_sources"].items()):
            raise ValueError("GENDER_POLICY_GENERATOR_CHANGED")
    pool_meta_sha = mapping._metadata_sha(metadata)
    values = dict(cohort_id=cohort_id,pool_sha256=pool_sha256,pool_receipt_sha256=pool.receipt_sha256,
        pool_metadata_sha256=pool_meta_sha,
        mapping_metadata_json=_json(enriched),mapping_metadata_sha256=mapping._metadata_sha(enriched),
        catalog_sha256=_sha(catalog),runtime_sha256=runtime.sha256,
        gender_policy_json=None if gender_policy is None else _json(gender_policy),
        gender_initializer_sha256=hashlib.sha256(Path(__file__).with_name("gender.py").read_bytes()).hexdigest())
    return ConstructorContext(**values,sha256=_sha(values))


def _context(context, runtime):
    _runtime(runtime)
    if not isinstance(context,ConstructorContext):
        raise ValueError("CONSTRUCTOR_CONTEXT_REQUIRED")
    values=asdict(context);digest=values.pop("sha256")
    if (_sha(values)!=digest or context.runtime_sha256!=runtime.sha256
            or mapping._metadata_sha(context.metadata())!=context.mapping_metadata_sha256
            or context.gender_initializer_sha256!=hashlib.sha256(Path(__file__).with_name("gender.py").read_bytes()).hexdigest()):
        raise ValueError("CONSTRUCTOR_CONTEXT_CHANGED")


@dataclass(frozen=True)
class ConstructorProvenance:
    context_sha256: str
    root_id: int
    team_indices: tuple[int,int]
    initial_fields: tuple
    pool_constructor_fields: tuple
    gender_assignment: tuple
    static_native: tuple
    sha256: str


_POOL_STATE_FIELDS = ("species","level","types","stats","moves","pp","item","ability",
    "hidden_power_type","hidden_power_power")
_IMMUTABLE_FIELDS = ("species","level","stats","max_hp","moves","hidden_power_type",
    "hidden_power_power","happiness","gender","base_ability")


class _GenderInput(NamedTuple):
    species: object
    gender: object


def capture_before_entry(state,index,pool,team_indices,*,context,runtime,root_id,gender_assignment):
    """Capture originals; return a named refusal instead of fabricating fields."""
    raw=()
    try:
        _context(context,runtime)
        _pool(pool)
        if pool.receipt_sha256!=context.pool_receipt_sha256:raise ValueError("CONSTRUCTOR_POOL_CONTEXT_MISMATCH")
        pool=pool.arrays
        metadata=context.metadata();data=mapping._sample(state,index);raw=mapping._freeze(data)
        teams=mapping._freeze(team_indices)
        if type(root_id) is not int or not 0<=root_id<2**32 or len(teams)!=2 or any(type(t) is not int or t<0 for t in teams):
            raise ValueError("EXPLICIT_CONSTRUCTOR_ROOT_REQUIRED")
        if (data["entry_pending"]!=(True,True) or data["turn"]!=0 or data["active"]!=(0,0)
                or data["pending_residual"] or any(data["baton_pass"]) or any(data["forced_switch"])):
            raise ValueError("PRE_ENTRY_CONSTRUCTOR_STAGE_REQUIRED")
        if data['last_action_kind'] != (0,0) or data['last_action_index'] != (-1,-1):
            raise ValueError('PRE_ENTRY_ACTION_HISTORY_MUST_BE_NONE')
        zero_fields=("status","status_counter","sleep_skipped","disabled","choice_move","last_move",
            "encore_move","encore_counter","charge_move","charge_counter","boosts","volatile",
            "confusion_counter","protect_counter","substitute_hp","active_turns","trapped",
            "queued_actions","queued_stall","residual_draws","phazing_draws","secondary_draws",
            "sleep_talk_draws","replacement_count","spikes","wish_counter","weather","weather_counter","truant_loaf")
        if any(not mapping._all(data[name]) for name in zero_fields) or data["queued_actor"]!=-1 or data["done"] or data["winner"]!=-1:
            raise ValueError("PRE_ENTRY_CONSTRUCTOR_STATE_CHANGED")
        if not isinstance(gender_assignment,dict) or gender_assignment.get("kind") not in ("explicit_pool","generated"):
            raise ValueError("GENDER_ASSIGNMENT_PROVENANCE_REQUIRED")
        if gender_assignment["kind"]=="generated" and (not {"policy_sha256","seed","domain","root_id"} <= gender_assignment.keys()
                or not _hex(gender_assignment["policy_sha256"],64) or gender_assignment["root_id"]!=root_id
                or any(type(gender_assignment[k]) is not int or not 0<=gender_assignment[k]<2**32 for k in ("seed","domain","root_id"))):
            raise ValueError("GENERATED_GENDER_BINDING_REQUIRED")
        selected={name:tuple(mapping._freeze(pool[name][t]) for t in teams)
            for name in (*_POOL_STATE_FIELDS,"nature","evs")}
        selected["happiness"]=(tuple(mapping._freeze(pool["happiness"][t]) for t in teams)
            if "happiness" in pool else ((255,)*6,)*2)
        if any(data[name]!=selected[name] for name in _POOL_STATE_FIELDS):
            raise ValueError("CONSTRUCTOR_POOL_IDENTITY_CHANGED")
        if data["base_ability"]!=selected["ability"]:
            raise ValueError("CONSTRUCTOR_BASE_ABILITY_CHANGED")
        if data["happiness"]!=selected["happiness"] or any(type(v) is not int or not 0<=v<=255
                for party in selected["happiness"] for v in party):
            raise ValueError("CONSTRUCTOR_HAPPINESS_CHANGED")
        if data["hp"]!=data["max_hp"] or data["max_hp"]!=tuple(tuple(row[0] for row in party) for party in data["stats"]):
            raise ValueError("CONSTRUCTOR_HP_CHANGED")
        genders=data["gender"]
        if any(type(g) is not int or g not in (0,1,2) for party in genders for g in party):
            raise ValueError("ORIGINAL_GENDER_NOT_RETAINED")
        if gender_assignment["kind"]=="explicit_pool":
            if "gender" not in pool or genders!=tuple(mapping._freeze(pool["gender"][t]) for t in teams):
                raise ValueError("POOL_GENDER_BINDING_CHANGED")
        else:
            import numpy as np
            from .gender import initialize_genders
            if context.gender_policy_json is None:raise ValueError("BOUND_GENDER_POLICY_REQUIRED")
            policy=json.loads(context.gender_policy_json)
            if gender_assignment["policy_sha256"]!=_sha(policy):raise ValueError("GENDER_POLICY_BINDING_CHANGED")
            # Replay only the constructor field on a separate, minimal proof
            # input. This neither initializes the live state nor consumes keys.
            proof=_GenderInput(np.asarray([data["species"]]),np.full((1,2,6),-1,np.int32))
            expected=initialize_genders(proof,policy,metadata["vocabulary"]["species"],
                seed=gender_assignment["seed"],domain=gender_assignment["domain"],root_ids=[root_id],
                explicit_genders=np.asarray([[pool["gender"][t] for t in teams]]) if "gender" in pool else None)
            if genders!=mapping._freeze(np.asarray(expected.gender)[0]):
                raise ValueError("GENERATED_GENDER_VALUE_CHANGED")
        static=[]
        for side in range(2):
            for p in range(6):
                species=mapping._vocab(metadata,"species",data["species"][side][p])
                nature=mapping._vocab(metadata,"natures",selected["nature"][side][p])
                evs=selected["evs"][side][p]
                if nature=="none" or len(evs)!=6 or any(type(v) is not int or not 0<=v<=255 for v in evs) or sum(evs)>510:
                    raise ValueError("ORIGINAL_NATURE_EVS_UNRESOLVED")
                original_weight=metadata["oracle_species"][species]["weight_kg"]
                if original_weight is None:raise ValueError("ORIGINAL_WEIGHT_UNRESOLVED: "+species)
                weight=mapping._f32(original_weight)
                if not math.isfinite(weight) or weight<=0:raise ValueError("ORIGINAL_WEIGHT_UNRESOLVED")
                types=tuple("typeless" if (name:=mapping._vocab(metadata,"types",v))=="none" else name for v in data["types"][side][p])
                ability=mapping._vocab(metadata,"abilities",data["ability"][side][p])
                static.append((f"{mapping.SIDE_NAMES[side]}.pokemon[{p}]",
                    (nature,evs,{0:"none",1:"male",2:"female"}[genders[side][p]],weight,types,ability)))
        values=dict(context_sha256=context.sha256,root_id=root_id,team_indices=teams,
            initial_fields=raw,pool_constructor_fields=mapping._freeze(selected),
            gender_assignment=mapping._freeze(gender_assignment),static_native=tuple(static))
        return ConstructorProvenance(**values,sha256=_sha(values))
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,OSError) as error:
        return mapping.Refused((mapping.Gap("constructor","CONSTRUCTOR_PROVENANCE_REFUSED",str(error)),),raw)


def native_input_with_provenance(state,index,metadata,*,provenance,context,runtime,
                                  expected_root_id,expected_pool_sha256,module=None):
    """Build a full post-entry ordinary correspondence using explicit originals."""
    raw=()
    try:
        _context(context,runtime)
        data=mapping._sample(state,index);raw=mapping._freeze(data)
        if not isinstance(provenance,ConstructorProvenance):raise ValueError("CONSTRUCTOR_PROVENANCE_REQUIRED")
        values=asdict(provenance);digest=values.pop("sha256")
        if _sha(values)!=digest or provenance.context_sha256!=context.sha256:
            raise ValueError("CONSTRUCTOR_PROVENANCE_CHANGED")
        if provenance.root_id!=expected_root_id or context.pool_sha256!=expected_pool_sha256:
            raise ValueError("CONSTRUCTOR_ROOT_BINDING_CHANGED")
        if mapping._metadata_sha(metadata)!=mapping._metadata_sha(context.metadata()):
            raise ValueError("CONSTRUCTOR_MAPPING_METADATA_CHANGED")
        initial=dict(provenance.initial_fields)
        if any(data[name]!=initial[name] for name in _IMMUTABLE_FIELDS):
            raise ValueError("IMMUTABLE_CONSTRUCTOR_IDENTITY_CHANGED")
        # No pending-entry flag is cleared. The ordinary mapper must prove the
        # complete current physical state before it creates a Certificate.
        return mapping._native_input_from_data(data,metadata,module=module,
            original_static=provenance.static_native,original_genders=initial["gender"])
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,OSError) as error:
        return mapping.Refused((mapping.Gap("constructor","CONSTRUCTOR_INPUT_REFUSED",str(error)),),raw)
