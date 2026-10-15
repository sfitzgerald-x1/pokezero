"""Prospective owned native replay with an independently checked uncertain suffix.

V1 remains all-or-nothing. V2 may stop only at a valid observed boundary where
the next original action is demonstrably unavailable. Every declared coordinate
remains required; no reached prefix supplies a missing evaluated AFTER endpoint.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_boundary_transport import BoundaryTransportCapture, read_owned_boundary_transport, SCHEMA as TRANSPORT_V1
from .search_over_raw_boundary_run import (
    BoundaryRunContract, BoundaryProcessOwner, OwnedNativePathBoundaryJob, RawBoundaryRunValidator,
    NATIVE_PATH_INPUT_KIND, _validate_native_path_inputs, _join_raw_result, raw_root_json,
)
from .search_over_raw_showdown_boundary import commands_for_native_actions, NativeReplayUnavailable, _mapped_sides
from .search_over_raw_supervision import _supervise_owned_job

INPUT_KIND = "pokezero.search-over-raw.native-action-boundary-input.v2"
CAPTURE_SCHEMA = "pokezero.search-over-raw.owned-native-prefix-transport.v2"


def prefix_law():
    return dict(schema=INPUT_KIND, unavailable_suffix="all_original_unreached_coordinates_uncertain",
        stopping_witness="rederive_next_original_action_at_last_actual_owned_snapshot",
        fallback_permitted=False, retries_permitted=False, substitute_after_endpoint_permitted=False,
        terminal_observation_is_scored_label=False)


def _validate_inputs(value):
    require(type(value) is dict and value.get("input_kind") == INPUT_KIND
        and digest(value.get("prefix_law")) == digest(prefix_law())
        and digest(value.get("prefix_derivation_sources")) == digest(prefix_source_hashes()),
        "prospective source-bound native prefix input law required")
    strict = deepcopy(value)
    strict.pop("prefix_law")
    strict.pop("prefix_derivation_sources")
    strict["input_kind"] = NATIVE_PATH_INPUT_KIND
    _validate_native_path_inputs(strict)


def prefix_source_hashes():
    paths = (Path(__file__), Path(__file__).with_name("search_over_raw_boundary_transport.py"))
    return {str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def termination_witness(current, boundaries, mapping, observed):
    # Keep the mapper explicit through the entire derivation; never use a
    # caller's reported status as authority to shorten a transport roster.
    required = len(boundaries)
    require(type(observed) is int and 1 <= observed <= required, "original observed prefix count required")
    reason, actors, action_sha = None, [], None
    if observed < required:
        actions = boundaries[observed]["native"]["action_identity"]
        action_sha = digest(actions)
        try:
            commands_for_native_actions(current, mapping, actions)
        except NativeReplayUnavailable as unavailable:
            reason, actors = unavailable.reason, list(unavailable.actors)
        else:
            raise ValueError("executable next original action cannot be declared unavailable")
    return dict(status="FULL_PATH_OBSERVED" if reason is None else "UNAVAILABLE_SUFFIX",
        required_boundary_records=required, observed_boundary_records=observed,
        unavailable_boundary_records=required-observed, next_original_ordinal=None if reason is None else observed,
        reason=reason, unavailable_actors=actors, intended_actions_sha256=action_sha,
        last_actual_snapshot_sha256=digest(current))


def strict_transport_view(capture):
    """Transport parser adapter only; does not validate a stopping witness."""
    require(type(capture) is dict and set(capture) == {"schema", "clone_binding", "request_ids", "records",
        "snapshots", "private_controller_only", "termination"} and capture["schema"] == CAPTURE_SCHEMA
        and type(capture["snapshots"]) is list and type(capture["request_ids"]) is list
        and 1 <= len(capture["snapshots"]) <= len(capture["request_ids"]), "bounded original native prefix required")
    value = deepcopy(capture)
    value.pop("termination")
    value["schema"] = TRANSPORT_V1
    value["request_ids"] = value["request_ids"][:len(value["snapshots"])]
    return value


def read_native_prefix_transport(capture, *, process_records, raw_root, clone_binding, request_ids,
        native_boundaries, party_mapping):
    """Reconstruct acknowledged prefix and its stop; retain the full denominator."""
    before = digest(dict(capture=capture, records=process_records, root=raw_root, clone=clone_binding,
        requests=request_ids, boundaries=native_boundaries, mapping=party_mapping))
    view = strict_transport_view(capture)
    require(capture["request_ids"] == request_ids and len(native_boundaries) == len(request_ids),
        "full original native prefix coordinate roster required")
    trace = read_owned_boundary_transport(view, process_records=process_records,
        initial_snapshot_json=raw_root_json(raw_root), clone_binding=clone_binding, request_ids=view["request_ids"])
    snapshots = [json.loads(row["currentSnapshotJson"]) for row in capture["snapshots"]]
    for current in snapshots: _mapped_sides(current, party_mapping)
    for ordinal, observed in enumerate(trace["observed_snapshots"][1:]):
        expected = commands_for_native_actions(snapshots[ordinal], party_mapping,
            native_boundaries[ordinal+1]["native"]["action_identity"])
        require(digest(observed["incoming_choices"]) == digest(expected),
            "actual prefix command differs from next original native action")
    termination = termination_witness(snapshots[-1], native_boundaries, party_mapping, len(snapshots))
    require(digest(capture["termination"]) == digest(termination), "original native prefix stopping witness drift")
    require(before == digest(dict(capture=capture, records=process_records, root=raw_root, clone=clone_binding,
        requests=request_ids, boundaries=native_boundaries, mapping=party_mapping)), "native prefix evidence changed")
    return dict(trace, schema=CAPTURE_SCHEMA, capture_sha256=digest(capture), termination=termination,
        full_native_path_observed=termination["status"] == "FULL_PATH_OBSERVED")


@dataclass(frozen=True)
class NativePrefixBoundaryRunContract(BoundaryRunContract):
    def receipt(self):
        return dict(super().receipt(), schema="pokezero.search-over-raw.native-prefix-boundary-run-contract.v2",
            input_kind=INPUT_KIND, prefix_law=prefix_law(), python_actor_snapshot_required=False,
            caller_command_roster=False, unavailable_action_fallback=False)


class NativePrefixBoundaryProcessOwner(BoundaryProcessOwner):
    launch_schema = "pokezero.search-over-raw.native-prefix-boundary-bridge-launch.v2"
    close_schema = "pokezero.search-over-raw.native-prefix-boundary-bridge-close.v2"

    def __init__(self, *, raw_root, native_boundaries, party_mapping, **kwargs):
        super().__init__(**kwargs)
        self.raw_root, self.boundaries, self.mapping = deepcopy(raw_root), deepcopy(native_boundaries), deepcopy(party_mapping)

    def _match_capture(self, capture):
        require(getattr(self.env, "_bridge_boundary_observer", None) is None, "prefix tap must be removed before sealing")
        read_native_prefix_transport(capture, process_records=self.records, raw_root=self.raw_root,
            clone_binding=self.clone, request_ids=self.requests, native_boundaries=self.boundaries, party_mapping=self.mapping)


@dataclass(frozen=True)
class OwnedNativePrefixBoundaryJob(OwnedNativePathBoundaryJob):
    process_owner_class = NativePrefixBoundaryProcessOwner

    def input_binding(self):
        value = super().input_binding()
        value.update(input_kind=INPUT_KIND, prefix_law=prefix_law(), prefix_derivation_sources=prefix_source_hashes())
        _validate_inputs(value)
        return value

    def __call__(self, ownership, phase):
        phase("startup")
        before = self.input_binding()
        env = self._components()
        tap = BoundaryTransportCapture(clone_binding=self.clone_binding, request_ids=self.request_ids)
        owner = self.process_owner_class(directory=Path(ownership["directory"]).parent, ownership=ownership,
            runtime=self.runtime, config=self.config, initial_snapshot_json=self.initial_snapshot_json(),
            clone_binding=self.clone_binding, request_ids=self.request_ids, raw_root=self.raw_root,
            native_boundaries=self.native_boundaries, party_mapping=self.party_mapping)
        primary = None
        with owner.installed(env):
            try:
                phase("boundaries")
                with tap.installed(env):
                    env._start_bridge()
                    env._battle_token = self.clone_binding["battleId"]
                    event = env._bridge_request_event(dict(type="clone_snapshot", battleId=self.clone_binding["battleId"],
                        cloneId=self.clone_binding["cloneId"], snapshotJson=self.initial_snapshot_json()), "cloned")
                    require(all(event.get(k) == v for k,v in self.clone_binding.items()), "actual prefix clone drift")
                    current = tap.snapshot()
                    self.check_reached_boundary(current)
                    for ordinal in range(self.transition_count()):
                        try:
                            choices = self.commands_at_boundary(ordinal, current)
                        except NativeReplayUnavailable:
                            break  # The sealed reader independently rederives this category.
                        self._advance(env, choices)
                        current = tap.snapshot()
                        self.check_reached_boundary(current)
                    termination = termination_witness(current, self.native_boundaries, self.party_mapping, len(tap.snapshots))
                    env._bridge_request_event(dict(type="release_clone", **self.clone_binding), "clone_released")
                    env._battle_token = None
                require(digest(self.input_binding()) == digest(before), "prefix inputs changed during replay")
                capture = dict(tap.export(), schema=CAPTURE_SCHEMA, termination=termination)
                owner.seal(capture)
            except BaseException as error:
                primary = error
                raise
            finally:
                phase("cleanup")
                if env._process is not None:
                    try: env.close()
                    except BaseException as cleanup:
                        if primary is None: raise
                        primary.add_note("Native prefix cleanup failed: " + type(cleanup).__name__)
        require(digest(self.input_binding()) == digest(before), "prefix inputs changed after bridge reap")
        from .search_over_raw_boundary_run import UNQUALIFIED
        return dict(**before, capture=capture, capture_sha256=digest(capture),
            synthetic_process_fixture=getattr(self, "synthetic_process_fixture", False) is True,
            **{key:False for key in UNQUALIFIED})


class NativePrefixBoundaryRunValidator(RawBoundaryRunValidator):
    contract_class = NativePrefixBoundaryRunContract
    launch_schema = NativePrefixBoundaryProcessOwner.launch_schema
    close_schema = NativePrefixBoundaryProcessOwner.close_schema
    validate_expected = staticmethod(_validate_inputs)

    def _check_inputs(self):
        super()._check_inputs()
        _mapped_sides(self.raw_root, self.expected["party_mapping"])

    def _match_capture(self, capture, records):
        read_native_prefix_transport(capture, process_records=records, raw_root=self.raw_root,
            clone_binding=self.expected["clone_binding"], request_ids=self.expected["request_ids"],
            native_boundaries=self.expected["native_boundaries"], party_mapping=self.expected["party_mapping"])


def supervise_native_prefix_boundary_run(job, *, directory, binding_sha256, verify_admission,
        validate_result, contract=None, _synthetic_watchdog_seconds=None):
    contract = NativePrefixBoundaryRunContract() if contract is None else contract
    require(type(contract) is NativePrefixBoundaryRunContract and (type(job) is OwnedNativePrefixBoundaryJob or
        isinstance(job, OwnedNativePrefixBoundaryJob) and getattr(job, "synthetic_process_fixture", False) is True)
        and callable(validate_result), "distinct prospective native prefix owner required")
    before = job.input_binding()
    def verify():
        require(digest(job.input_binding()) == digest(before), "native prefix admission inputs changed")
        return verify_admission()
    def validate(result):
        _join_raw_result(result, before, result["capture"])
        frozen = digest(result)
        require(validate_result(result) is not False, "native prefix result refused")
        require(digest(result) == frozen and digest(job.input_binding()) == digest(before), "native prefix acceptance mutation")
    return _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256, verify_admission=verify,
        contract=contract, kind="boundary-run", work_phase="boundaries", validate_result=validate,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)
