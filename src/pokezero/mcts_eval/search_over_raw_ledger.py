"""Create-only Phase A evidence with a one-shot held-out execution claim.

Hash bindings detect accidental drift; they are not an authentication system or
a sandbox against an operator deliberately bypassing the controller. An orphan
claim is an unresolved attempt, never permission to retry the held-out panel.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping

from .search_over_raw import digest, freeze_selection, panel_summary, require, validation_gate


def _write_new(path: Path, value: Mapping) -> None:
    # Serialize before opening: invalid payloads must not leave a seemingly
    # valid zero-length receipt. A crash during writing remains fail-closed.
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "receipt must be a JSON object")
    return value


class PhaseALedger:
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory).resolve()
        envelope = _load(self.directory / "contract.json")
        self.contract = envelope["contract"]
        require(envelope["contract_sha256"] == digest(self.contract), "ledger contract digest mismatch")
        self.contract_sha256 = envelope["contract_sha256"]

    @classmethod
    def create(cls, directory: Path | str, contract: Mapping) -> "PhaseALedger":
        directory = Path(directory).resolve()
        # Validate canonicalizability before creating anything.
        envelope = dict(contract=dict(contract), contract_sha256=digest(contract))
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "exploration").mkdir()
        _write_new(directory / "contract.json", envelope)
        return cls(directory)

    def verify(self, *, execution: bool = False) -> None:
        envelope = _load(self.directory / "contract.json")
        require(envelope["contract_sha256"] == self.contract_sha256
            and digest(envelope["contract"]) == self.contract_sha256, "ledger contract drift")
        for name, expected in self.contract.get("input_hashes", {}).items():
            require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected,
                "bound input drift: " + name)
        if execution:
            require(self.contract.get("execution_ready") is True
                and self.contract.get("remaining_gates") == [],
                "scientific execution admission is incomplete")

    def record_exploration(self, configuration: str, root_intervals: Mapping) -> dict:
        self.verify()
        require(not (self.directory / "selection.json").exists(), "exploration is already frozen")
        summary = panel_summary(self.contract, panel="exploration", configuration=configuration,
            root_intervals=root_intervals)
        artifact = dict(contract_sha256=self.contract_sha256, configuration=configuration,
            root_intervals=dict(root_intervals), summary=summary)
        _write_new(self.directory / "exploration" / f"{configuration}.json", artifact)
        return summary

    def freeze(self, configuration: str) -> dict:
        self.verify()
        require(not (self.directory / "validation-claim.json").exists(), "validation has already been opened")
        artifact = _load(self.directory / "exploration" / f"{configuration}.json")
        require(artifact["contract_sha256"] == self.contract_sha256
            and artifact["configuration"] == configuration, "exploration artifact binding mismatch")
        summary = panel_summary(self.contract, panel="exploration", configuration=configuration,
            root_intervals=artifact["root_intervals"])
        require(summary == artifact["summary"], "exploration summary differs from its evidence")
        selection = freeze_selection(self.contract, configuration, exploration_summary=summary)
        selection["exploration_artifact_sha256"] = digest(artifact)
        _write_new(self.directory / "selection.json", selection)
        return selection

    def _selection(self) -> dict:
        selection = _load(self.directory / "selection.json")
        artifact = _load(self.directory / "exploration" / f"{selection['configuration']}.json")
        require(digest(artifact) == selection["exploration_artifact_sha256"],
            "frozen exploration artifact drift")
        expected = freeze_selection(self.contract, selection["configuration"],
            exploration_summary=artifact["summary"])
        require(all(selection.get(k) == v for k, v in expected.items()), "frozen selection drift")
        return selection

    def validation_once(self, collect: Callable[[Mapping, str], Mapping]) -> dict:
        """Claim before callback, recompute summaries, never reopen after failure.

        The runtime callback must return the full fixed-roster interval evidence;
        omitted roots retain [-1,1]. It must itself enforce public-only inputs,
        qualified runtime/source bindings and durable progress of each worker.
        This controller does not qualify arbitrary callbacks as scientific code.
        """
        self.verify(execution=True)
        selection = self._selection()
        claim = dict(contract_sha256=self.contract_sha256, selection_sha256=digest(selection),
            configuration=selection["configuration"], status="VALIDATION_ATTEMPT_CLAIMED",
            retries_permitted=False)
        # O_EXCL provides one-winner admission even when two controllers race.
        _write_new(self.directory / "validation-claim.json", claim)
        try:
            # Detach objects so a callback cannot mutate this controller's state.
            intervals = collect(json.loads(json.dumps(self.contract)), selection["configuration"])
            self.verify(execution=True)
            require(digest(self._selection()) == claim["selection_sha256"], "selection drift during validation")
            require(_load(self.directory / "validation-claim.json") == claim, "validation claim drift")
            summary = panel_summary(self.contract, panel="validation",
                configuration=selection["configuration"], root_intervals=intervals)
            receipt = dict(contract_sha256=self.contract_sha256, selection_sha256=digest(selection),
                root_intervals=dict(intervals), summary=summary,
                gate=validation_gate(self.contract, selection, summary), status="VALIDATION_ATTEMPT_FINISHED",
                phase_b_authorized=False)
        except Exception as error:
            # Preserve the attempt without logging potentially private exception
            # text. The runtime should write its own sanitized failure witness.
            _write_new(self.directory / "validation-failure.json", dict(
                contract_sha256=self.contract_sha256, selection_sha256=digest(selection),
                status="VALIDATION_ATTEMPT_FAILED", error_type=type(error).__name__,
                all_validation_roots_uncertain=True, phase_b_authorized=False, retries_permitted=False))
            raise
        _write_new(self.directory / "validation-result.json", receipt)
        return receipt
