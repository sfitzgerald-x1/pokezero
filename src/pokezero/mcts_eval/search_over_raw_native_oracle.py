"""Explicit A1 native team consumption, never public deployment or admission.

Actual override, constructor output and serialized native state are joined to
the existing dispatch/atomic-prefix observer. Zero attempts prove preparation
only. This module does not launch, retry, qualify physics or authorize science.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib

from .search_over_raw import digest,require
from .search_over_raw_outcomes import _sha
from .search_over_raw_oracle import (TeamOracle,DiagnosticTeamOracleSearchAdapter,
    opening_team,team_sha256)
from .search_over_raw_world_inputs import (NativeWorldInputs,_WorldInputAdapterMixin,
    _state_payload,public_request_witness)

SCHEMA="pokezero.search-over-raw.native-oracle-consumption.v1"
NATIVE_ORACLE_CAPTURE=dict(schema=SCHEMA,original_opponent_team_only=True,
    at_actual_preparation=True,at_each_actual_sampling_attempt=True,
    constructor_and_native_dispatch_join_required=True,includes_zero_attempt_preparation=True,
    no_sampled_fallback=True,deployable=False,runtime_qualified=False)


class NativeOracleWorldInputs(NativeWorldInputs):
    def __init__(self,*args,oracle,oracle_request,oracle_source,oracle_policy,**kwargs):
        require(type(oracle) is TeamOracle and kwargs.get("record_timed_atomic_work") is True
            and kwargs.get("record_atomic_work") is True
            and kwargs.get("expected_public_request") is not None,
            "A1 oracle requires actual timed/full-public observation")
        kwargs["record_construction"]=True
        super().__init__(*args,**kwargs)
        self.oracle,self.oracle_request,self.oracle_source,self.oracle_policy=(
            oracle,oracle_request,oracle_source,oracle_policy)
        from ..env import BattleStartOverride
        from ..showdown_fixture import pack_team
        oracle.validate(oracle_request,oracle_source)
        subject=oracle.subject;opponent="p2" if subject == "p1" else "p1"
        own=opening_team(oracle_request.state.self_initial_request,subject,oracle_source)
        self.expected_override=BattleStartOverride(player_teams={subject:pack_team(own),
            opponent:pack_team(oracle.opponent_team)},observation_format_id="gen3randombattle")
        self.override_sha=digest(asdict(self.expected_override))
        self.oracle_sha=digest(_state_payload(oracle))
        self.catalog_sha=digest(_state_payload(oracle_source))
        self.installed_override=oracle_policy._fixed_override
        self.oracle_attempts=[]
        self.preparation=self._actual_preparation()

    def _actual_preparation(self):
        from ..env import BattleStartOverride
        self.oracle.validate(self.oracle_request,self.oracle_source)
        require(digest(_state_payload(self.oracle)) == self.oracle_sha
            and digest(_state_payload(self.oracle_source)) == self.catalog_sha
            and digest(public_request_witness(self.oracle_request)) == digest(self.expected_public_request)
            and self.oracle_policy._set_source is self.oracle_source
            and self.oracle_policy._fixed_override is self.installed_override
            and type(self.installed_override) is BattleStartOverride
            and digest(asdict(self.installed_override)) == self.override_sha
            and digest(asdict(self.expected_override)) == self.override_sha,
            "actual native oracle preparation changed")
        return dict(schema=SCHEMA,oracle=self.oracle.receipt(),
            public_request_witness=deepcopy(self.expected_public_request),
            override_sha256=self.override_sha,reconstruction_catalog_sha256=self.catalog_sha,
            source_snapshot_transported=False,committed_opponent_action_transported=False,
            deployable=False,runtime_qualified=False)

    def attempt(self,context,ordinal,override):
        try:
            self._actual_preparation()
            require(override is self.installed_override and digest(asdict(override)) == self.override_sha,
                "actual native oracle attempt used a sampled or different team")
            super().attempt(context,ordinal,override)
            require(self.rows[-1]["draw_sha256"] == self.override_sha,
                "actual oracle draw/constructor binding changed")
            self.oracle_attempts.append(dict(ordinal=ordinal,override_sha256=self.override_sha,
                own_team_sha256=self.oracle.own_team_sha256,
                opponent_team_sha256=self.oracle.opponent_team_sha256))
            self._actual_preparation()
        except BaseException as error:
            self.errors.append(type(error).__name__)
            raise

    def finish(self,*args,**kwargs):
        require(digest(self._actual_preparation()) == digest(self.preparation)
            and len(self.oracle_attempts) == len(self.rows),"native oracle preparation/attempt roster drift")
        result=super().finish(*args,**kwargs)
        result["native_team_oracle"]=dict(preparation=deepcopy(self.preparation),
            attempts=deepcopy(self.oracle_attempts),prepared=True,
            team_consumed=any(row["consumed"] for row in self.rows),
            deployable=False,runtime_qualified=False)
        validate_native_oracle_inputs(result,oracle_receipt=self.oracle.receipt(),
            public_request=self.expected_public_request,expected_override_sha256=self.override_sha,
            expected_reconstruction_catalog_sha256=self.catalog_sha)
        require(digest(self._actual_preparation()) == digest(self.preparation),
            "native oracle changed during final input validation")
        return result

    def _checked_hook(self,hook,*args):
        try:
            self._actual_preparation()
            result=hook(*args)
            self._actual_preparation()
            return result
        except BaseException as error:
            self.errors.append(type(error).__name__)
            raise

    def world_built(self,value):
        return self._checked_hook(super().world_built,value)

    def native_built(self,value):
        return self._checked_hook(super().native_built,value)

    def constructed(self,context,world,state):
        return self._checked_hook(super().constructed,context,world,state)

    def dispatch(self,args):
        return self._checked_hook(super().dispatch,args)


def validate_native_oracle_inputs(manifest,*,oracle_receipt,public_request,expected_override_sha256,
                                 expected_reconstruction_catalog_sha256):
    """Independent team→construction→actual-call join, no fixed completed quota."""
    from .search_over_raw_construction import read_native_construction,construction_source_hashes
    before=digest(dict(manifest=manifest,oracle=oracle_receipt,public=public_request,
        override=expected_override_sha256,catalog=expected_reconstruction_catalog_sha256))
    saved=deepcopy(manifest);observation=saved.get("native_team_oracle")
    binding=public_request["consumed_binding"]
    require(type(oracle_receipt) is dict and set(oracle_receipt) == {
        "schema","information_scope","root_binding","own_team_sha256","opponent_team_sha256",
        "set_source_hash","deployable","source_snapshot_transported","committed_opponent_action_transported"}
        and oracle_receipt.get("schema") == "pokezero.search-over-raw.team-oracle.v1"
        and oracle_receipt.get("information_scope") == "original_opponent_team_only"
        and oracle_receipt.get("root_binding") == binding["root_binding"]
        and oracle_receipt.get("set_source_hash") == binding["set_source_hash"]
        and all(_sha(oracle_receipt.get(k)) for k in ("own_team_sha256","opponent_team_sha256"))
        and all(oracle_receipt.get(k) is False for k in ("deployable","source_snapshot_transported",
            "committed_opponent_action_transported"))
        and _sha(expected_override_sha256)
        and _sha(expected_reconstruction_catalog_sha256)
        and digest({k:saved.get(k) for k in binding}) == digest(binding),
        "native oracle prospective team/public root binding drift")
    require(type(observation) is dict and set(observation) == {
        "preparation","attempts","prepared","team_consumed","deployable","runtime_qualified"}
        and observation["prepared"] is True and observation["deployable"] is False
        and observation["runtime_qualified"] is False
        and saved.get("construction_observed") is True,
        "actual native oracle construction observation required")
    prep=observation["preparation"]
    require(type(prep) is dict and set(prep) == {"schema","oracle","public_request_witness",
        "override_sha256","reconstruction_catalog_sha256","source_snapshot_transported",
        "committed_opponent_action_transported","deployable","runtime_qualified"}
        and prep.get("schema") == SCHEMA
        and digest(prep.get("oracle")) == digest(oracle_receipt)
        and digest(prep.get("public_request_witness")) == digest(public_request)
        and prep.get("override_sha256") == expected_override_sha256
        and _sha(prep.get("reconstruction_catalog_sha256"))
        and prep["reconstruction_catalog_sha256"] == expected_reconstruction_catalog_sha256
        and all(prep.get(k) is False for k in ("source_snapshot_transported",
            "committed_opponent_action_transported","deployable","runtime_qualified")),
        "native oracle preparation differs from prospective original-team binding")
    rows,attempts,calls=saved.get("rows"),observation["attempts"],saved.get("calls")
    require(type(rows) is list and type(attempts) is list and type(calls) is list
        and len(rows) == len(attempts),"native oracle full attempted roster required")
    for ordinal,(row,attempt) in enumerate(zip(rows,attempts)):
        require(type(row) is dict and type(row.get("ordinal")) is int and row["ordinal"] == ordinal
            and type(row.get("worker")) is int and row["worker"] == 0
            and row.get("status") == "CONSTRUCTED" and row.get("draw_sha256") == expected_override_sha256
            and digest(attempt) == digest(dict(ordinal=ordinal,override_sha256=expected_override_sha256,
                own_team_sha256=oracle_receipt["own_team_sha256"],
                opponent_team_sha256=oracle_receipt["opponent_team_sha256"])),
            "native oracle attempt/truth/constructor roster drift")
        read_native_construction(row.get("construction_receipt"),binding=binding,ordinal=ordinal,
            draw_sha256=expected_override_sha256,constructed_sha256=row["constructed_sha256"],
            state_utf8_sha256=row["native_state_utf8_sha256"],source_hashes=construction_source_hashes())
        draw=row["construction_receipt"]["events"][0]["value"]["override"]
        subject=binding["subject"];opponent="p2" if subject == "p1" else "p1"
        require(hashlib.sha256(draw["player_teams"][subject].encode()).hexdigest() == oracle_receipt["own_team_sha256"]
            and hashlib.sha256(draw["player_teams"][opponent].encode()).hexdigest() == oracle_receipt["opponent_team_sha256"],
            "constructed override differs from original teams")
    covered=[]
    for ordinal,call in enumerate(calls):
        indices=call.get("construction_ordinals")
        require(type(indices) is list and indices and type(call.get("call_ordinal")) is int
            and call["call_ordinal"] == ordinal and call.get("status") == "RETURNED"
            and all(type(i) is int and 0 <= i < len(rows) and i not in covered for i in indices)
            and len(set(indices)) == len(indices) and type(call.get("multiplicity")) is int
            and call["multiplicity"] == len(indices)
            and all(rows[i].get("consumed") is True and rows[i].get("call_ordinal") == ordinal
                and type(rows[i].get("call_ordinal")) is int
                and rows[i].get("input_sha256") == call.get("input_sha256")
                and rows[i]["native_state_utf8_sha256"] == call["native_call_parameters"]["state_sha256"]
                for i in indices),"oracle constructed-state/actual dispatch join drift")
        covered.extend(indices)
    require(all(type(row.get("consumed")) is bool and row["consumed"] is (i in covered)
            for i,row in enumerate(rows))
        and observation["team_consumed"] is bool(covered)
        and before == digest(dict(manifest=manifest,oracle=oracle_receipt,public=public_request,
            override=expected_override_sha256,catalog=expected_reconstruction_catalog_sha256)),
        "native oracle consumed roster or late mutation drift")
    return dict(schema=SCHEMA,prepared=True,attempted_worlds=len(rows),
        constructed_worlds=len(rows),dispatched_worlds=len(covered),native_calls=len(calls),
        team_consumed=bool(covered),original_team_construction_join_validated=True,
        atomic_prefix_validated=False,owning_process_validated=False,deployable=False,
        runtime_qualified=False,scientific_collection_authorized=False)


class NativeTimedOracleSearchAdapter(_WorldInputAdapterMixin,DiagnosticTeamOracleSearchAdapter):
    """A1 model/one-second diagnostic only; public adapter laws stay unchanged."""
    _native_input_type=NativeOracleWorldInputs

    def _check_world_input_scope(self,configuration):
        require(configuration.arm == "incumbent" and configuration.belief == "oracle"
            and configuration.leaf == "model" and configuration.seconds == 1.
            and configuration.workers == 1,"A1 native oracle requires model/one-second/single-worker scope")

    def __init__(self,configuration,*,expected_public_request_sha256,**kwargs):
        require(kwargs.get("value_diagnostics") is None and kwargs.get("native_ancestry_contract") is None
            and kwargs.get("timed_reference_atomic_work",False) is False,
            "A1 oracle is not a leaf-label, reference or ancestry lane")
        super().__init__(configuration,expected_public_request_sha256=expected_public_request_sha256,
            timed_native_atomic_work=True,**kwargs)
        self.runtime_configuration["native_team_oracle_observation"]=deepcopy(NATIVE_ORACLE_CAPTURE)
        from .search_over_raw_construction import SCHEMA as CONSTRUCTION_SCHEMA,construction_source_hashes
        self.runtime_configuration["native_oracle_construction"]=dict(schema=CONSTRUCTION_SCHEMA,
            source_hashes=construction_source_hashes(),private_controller_only=True,
            instrumentation_can_change_deadlines=True,runtime_qualified=False)
        self.runtime_sha256=digest(self.runtime_configuration)

    def _native_input_options(self,request):
        return dict(oracle=self.oracle,oracle_request=request,
            oracle_source=self._oracle_source,oracle_policy=self._native)

    def _zero_work_unqualification(self,error,context,arguments):
        result=super()._zero_work_unqualification(error,context,arguments)
        if result is not None:
            result["evidence"]["team_oracle"]=self.oracle.receipt()
            self.last_unqualified=deepcopy(result)
        return result
