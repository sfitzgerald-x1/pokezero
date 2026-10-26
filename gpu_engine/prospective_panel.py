"""Opt-in original constructor binding for a NEW prospectively admitted panel.

This module neither launches a roster nor recovers historical constructors.
Capture precedes entry callbacks. Native construction/comparison is permitted
only after the collector closes its complete source artifact.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import tempfile

from . import future_state as mapping
from . import source_provenance as source


SCHEMA = "pokezero.prospective-constructors.v1"


@dataclass(frozen=True)
class Sources:
    pool: source.ConstructorPool
    context: source.ConstructorContext
    runtime: source.RuntimeAttestation
    root_start: int

    def root_id(self,slot):
        if type(slot) is not int or slot<0 or self.root_start+slot>=2**32:
            raise ValueError("bounded original root occurrence required")
        return self.root_start+slot

    def capture(self,state,index,teams,entry):
        gender=entry.get("gender_initialization")
        assignment=(dict(kind="explicit_pool") if gender is None else dict(kind="generated",
            policy_sha256=source._sha(gender["policy"]),seed=gender["seed"],domain=gender["domain"],
            root_id=self.root_id(index)))
        return source.capture_before_entry(state,index,self.pool,teams,context=self.context,
            runtime=self.runtime,root_id=self.root_id(index),gender_assignment=assignment)

    def frozen_context(self):
        return dict(schema=SCHEMA,context=asdict(self.context),runtime=asdict(self.runtime),
            pool_receipt=self.pool.receipt_json,root_start=self.root_start)


def admit(entry,pool_path):
    """Verify actual runtime/pool against the separately source-reviewed entry."""
    config=entry["constructor_provenance"]
    if (set(config)!={"schema","catalog","runtime","root_start","showdown_root"} or config["schema"]!=SCHEMA
            or type(config["root_start"]) is not int or config["root_start"]<0
            or config["root_start"]+entry["slots"]>2**32
            or type(config["showdown_root"]) is not str or not config["showdown_root"]):
        raise ValueError("complete bounded prospective constructor entry required")
    pool=source.load_constructor_pool(pool_path)
    if source._pool(pool)["pool_sha256"]!=entry["pool_sha256"]:
        raise ValueError("prospective immutable pool digest mismatch")
    gender=entry.get("gender_initialization")
    if gender is None:
        if "gender" not in pool.arrays:
            raise ValueError("explicit pool genders or prospective gender policy required")
    elif gender["root_start"]!=config["root_start"]:
        raise ValueError("gender and constructor root occurrences must agree")
    # Regenerate against the actual checkout/compiled tree, not an artifact's
    # self-reported source labels. No battle, action selection or search occurs.
    header=pool.metadata();header["pool_sha256"]=entry["pool_sha256"]
    with tempfile.TemporaryDirectory(prefix="pokezero-constructor-admission-") as directory:
        path=Path(directory)/"pool-metadata.json";path.write_text(source._json(header))
        artifacts=[("constructor_catalog.cjs",config["catalog"])]
        if gender is not None:artifacts.append(("gender_policy.cjs",gender["policy"]))
        for name,expected in artifacts:
            result=subprocess.run(["node",str(Path(__file__).with_name(name)),config["showdown_root"],
                str(path),header["source_commit_label"]],check=True,capture_output=True,text=True,timeout=60)
            if json.loads(result.stdout)!=expected:
                raise ValueError("ACTUAL_CONSTRUCTOR_SOURCE_MISMATCH: "+name)
    runtime=source.RuntimeAttestation(**config["runtime"])
    context=source.constructor_context(pool,config["catalog"],runtime=runtime,
        cohort_id=entry["source_commit"]+":"+entry["pool_sha256"]+":"+str(config["root_start"]),
        gender_policy=None if gender is None else gender["policy"])
    return Sources(pool,context,runtime,config["root_start"])


def plan_for_record(root,post_entry,record,original,sources,table,entry,*,record_sha256):
    """After source freeze, bind original statics to the exact current root."""
    raw=mapping._freeze(mapping._sample(root,0))
    def refuse(code,detail):
        return mapping.Refused((mapping.Gap("prospective_source",code,detail),),raw)
    if not isinstance(original,(source.ConstructorProvenance,mapping.Refused)):
        return refuse("MISSING_CONSTRUCTOR_PROVENANCE","Complete original record or explicit source refusal required")
    try:
        required={"state","constructor_provenance","post_entry_state","constructor_context_sha256","slot","teams","actions"}
        if type(record) is not dict or not required<=record.keys():
            return refuse("INCOMPLETE_FROZEN_CONSTRUCTOR_SOURCE","Complete frozen source fields required")
        if (source._sha(record)!=record_sha256 or mapping._freeze(record["state"])!=raw
                or mapping._freeze(record["constructor_provenance"])!=mapping._freeze(asdict(original))
                or mapping._freeze(record["post_entry_state"])!=mapping._freeze(mapping._sample(post_entry,0))
                or record["constructor_context_sha256"]!=sources.context.sha256):
            return refuse("FROZEN_CONSTRUCTOR_SOURCE_CHANGED","Complete root/original/post-entry/context binding required")
        if isinstance(original,mapping.Refused):return original
        root_id=sources.root_id(record["slot"])
        if tuple(record["teams"])!=original.team_indices or original.root_id!=root_id:
            return refuse("FROZEN_ROOT_OCCURRENCE_CHANGED","Exact original team pair/root occurrence required")
        initial_fields=dict(original.initial_fields);current_fields=dict(raw)
        if any(current_fields[name]!=initial_fields[name] for name in source._IMMUTABLE_FIELDS):
            return refuse("IMMUTABLE_CONSTRUCTOR_IDENTITY_CHANGED","Current root differs from its original constructor")
    except (KeyError,TypeError,ValueError,AttributeError) as error:
        return refuse("INVALID_FROZEN_CONSTRUCTOR_SOURCE",str(error))
    args=dict(provenance=original,context=sources.context,runtime=sources.runtime,
        expected_root_id=root_id,expected_pool_sha256=entry["pool_sha256"])
    metadata=sources.context.metadata()
    actions=tuple(record["actions"])
    if dict(raw)["pending_residual"]:
        from .conditional_law import plan_for_suspension
        initial=source.native_input_with_provenance(post_entry,0,metadata,**args)
        if isinstance(initial,mapping.Refused):return initial
        return plan_for_suspension(root,0,actions,table,metadata,
            certificate=initial.certificate,cap=entry["max_cases"])
    from .chance_consumers import plan_for_state
    current=source.native_input_with_provenance(root,0,metadata,**args)
    if isinstance(current,mapping.Refused):return current
    return plan_for_state(root,0,actions,table,metadata,cap=entry["max_cases"],native_input=current)
