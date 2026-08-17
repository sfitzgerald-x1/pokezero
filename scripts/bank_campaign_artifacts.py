#!/usr/bin/env python3
"""Bank a campaign's shards into the campaign store with a complete provenance stamp -- or refuse.

The campaign store's contract is that a directory of shards with no provenance stamp is not
banked, it is *dumped*: the two campaigns whose disagreement the search-ceiling program's
consistency gate exists to reconcile differ by an **engine build**, and that only became
knowable because both recorded their provenance. This tool is the only sanctioned writer into
the store, and its single job is to make an incomplete stamp impossible to produce.

WHY IT REFUSES INSTEAD OF WARNING
---------------------------------
Two artifacts in this program silently lost whole sections **because an input was absent
rather than wrong**: the writer received no value, wrote no key (or wrote ``null``), and
nothing complained -- so a partial artifact was published and read as a whole one for days.
A third campaign was nearly lost to a scratch directory. Every guard below therefore exits
non-zero and names the offending input. Nothing here warns, defaults, or fills a ``null``:

* ``absent`` is not ``null`` is not ``empty`` is not ``zero``. Those are four distinct
  refusals with four distinct messages, because "the field was missing" and "the field said
  zero" are different defects with different fixes and must never be reported as one.
* an empty shard list is a refusal, not a zero-shard bank;
* a shard whose recomputed sha256 disagrees with its declared one is a refusal;
* a declared shard file that is not on disk is a refusal;
* banking into a directory that already holds a *differing* artifact is a refusal unless
  overwrite is requested explicitly (an identical artifact is a no-op, not a rewrite);
* banking campaign ``X`` into a directory not named ``X`` is a refusal -- that is the
  scratch-directory loss, made unrepresentable.

ATOMICITY
---------
Validation is *complete before any byte is written*: every scalar, every shard, the
destination check and the optional checkpoint re-hash all run first, and all failures are
collected and reported together. Only then is a sibling temp directory built, filled, fsynced
and ``rename``d into place. If anything fails during that write -- a full disk, a vanished
source -- the temp directory is removed and the refusal is raised, so a refusal never leaves a
partial artifact behind. See ``tests/test_bank_campaign_artifacts.py`` for the injected-failure
demonstration; a guard nobody has watched read False certifies nothing.

THE SCHEMA IS DATA, NOT CONTROL FLOW
------------------------------------
Required fields live in ``ARTIFACT_KINDS`` as declared ``FieldSpec`` tuples per artifact kind.
Adding a required field is one edit in one place and cannot be forgotten on one of several
code paths, because there is only one path: :func:`_validate`. The Phase 1 instrument 2 kind
(``phase1.instrument2.rollout-arbiter.v1``) extends the base list with the arm's own estimand
caveats as first-class fields, so a caller *cannot* bank an arbiter cell whose numbers travel
without them.

PUBLIC-REPO HYGIENE
-------------------
Every path, image reference, pod and node name arrives as an argument or a stamp field.
Nothing about any deployment is hardcoded here, and nothing about one may be added.

Usage:
    python scripts/bank_campaign_artifacts.py \
        --kind phase1.instrument2.rollout-arbiter.v1 \
        --stamp /path/to/stamp.json \
        --source-dir /path/to/harvested/shards \
        --out-dir /path/to/campaign-store/my-campaign-20260817

    python scripts/bank_campaign_artifacts.py --print-required-fields \
        phase1.instrument2.rollout-arbiter.v1

Library use (the launchers import this; they do not shell out to it):
    from bank_campaign_artifacts import BankRefusal, bank_artifact
    try:
        result = bank_artifact(stamp, source_dir, out_dir)
    except BankRefusal as refusal:
        ...  # refusal.codes names every guard that read False
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_NAME = "scripts/bank_campaign_artifacts.py"
TOOL_VERSION = "1"

PROVENANCE_FILENAME = "PROVENANCE.json"
SHA256SUMS_FILENAME = "SHA256SUMS"
PROVENANCE_SCHEMA = "pokezero.campaign-store.provenance.v1"

#: Refusal exits with this status. Not 2: argparse owns 2 for usage errors, and a caller must
#: be able to tell "you invoked me wrongly" from "your artifact is incomplete".
REFUSAL_EXIT = 3

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_CAMPAIGN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# Refusal codes. Each one is a guard, and each one has a demonstrated failing input in
# tests/test_bank_campaign_artifacts.py. A code with no failing-input test does not belong here.
MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
NULL_REQUIRED_FIELD = "NULL_REQUIRED_FIELD"
INVALID_FIELD_VALUE = "INVALID_FIELD_VALUE"
MISSING_REQUIRED_ALTERNATIVE = "MISSING_REQUIRED_ALTERNATIVE"
EMPTY_SHARD_LIST = "EMPTY_SHARD_LIST"
SHARD_NOT_A_RECORD = "SHARD_NOT_A_RECORD"
SHARD_FILE_ABSENT = "SHARD_FILE_ABSENT"
SHARD_SHA256_MISMATCH = "SHARD_SHA256_MISMATCH"
DUPLICATE_SHARD_FILE = "DUPLICATE_SHARD_FILE"
UNSAFE_SHARD_PATH = "UNSAFE_SHARD_PATH"
UNACKNOWLEDGED_SHARD_FAILURE = "UNACKNOWLEDGED_SHARD_FAILURE"
UNDECLARED_ESTIMAND = "UNDECLARED_ESTIMAND"
UNACKNOWLEDGED_CPU_BUDGET = "UNACKNOWLEDGED_CPU_BUDGET"
DERIVED_FIELD_SUPPLIED = "DERIVED_FIELD_SUPPLIED"
UNKNOWN_ARTIFACT_KIND = "UNKNOWN_ARTIFACT_KIND"
ARTIFACT_KIND_MISMATCH = "ARTIFACT_KIND_MISMATCH"
STAMP_NOT_A_MAPPING = "STAMP_NOT_A_MAPPING"
STAMP_NOT_JSON = "STAMP_NOT_JSON"
SOURCE_DIR_ABSENT = "SOURCE_DIR_ABSENT"
CHECKPOINT_SHA256_MISMATCH = "CHECKPOINT_SHA256_MISMATCH"
CAMPAIGN_ID_DIRECTORY_MISMATCH = "CAMPAIGN_ID_DIRECTORY_MISMATCH"
DESTINATION_HOLDS_DIFFERENT_ARTIFACT = "DESTINATION_HOLDS_DIFFERENT_ARTIFACT"
WRITE_FAILED = "WRITE_FAILED"


# ----------------------------------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Refusal:
    """One guard that read False, with the input that made it read False."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class BankRefusal(Exception):
    """Raised instead of writing a partial artifact. Carries every guard that read False."""

    def __init__(self, reasons: Sequence[Refusal]) -> None:
        self.reasons = tuple(reasons)
        super().__init__(self._render())

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)

    def messages_for(self, code: str) -> tuple[str, ...]:
        return tuple(reason.message for reason in self.reasons if reason.code == code)

    def _render(self) -> str:
        plural = "" if len(self.reasons) == 1 else "s"
        head = (
            f"REFUSED to bank: {len(self.reasons)} unmet requirement{plural}; "
            f"nothing was written."
        )
        return "\n".join([head, *(f"  - {reason}" for reason in self.reasons)])


# ----------------------------------------------------------------------------------------------
# The schema, as data
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """One required provenance field.

    ``accepts`` is prose naming the accepted values and goes verbatim into the refusal, so a
    caller learns the contract from the failure rather than from this file. ``why`` is the
    reason the field is load-bearing, so nobody deletes it as noise later.
    """

    name: str
    accepts: str
    why: str
    check: Callable[[Any], bool]


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_campaign_id(value: Any) -> bool:
    return _is_nonempty_str(value) and bool(_CAMPAIGN_ID_RE.fullmatch(value))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _int_at_least(minimum: int) -> Callable[[Any], bool]:
    # bool is an int subclass in Python; True must not satisfy "an integer >= 1".
    def check(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= minimum

    return check


def _is_fraction(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _is_bool(value: Any) -> bool:
    # Deliberately not "truthy": the CPU-budget acknowledgement is an assertion about a
    # deployment, and the string "false" is truthy in Python.
    return isinstance(value, bool)


def _is_flag_list(value: Any) -> bool:
    # An empty list is ACCEPTED here and only here: a launcher genuinely may take no flags, and
    # forcing a caller to invent one would teach the habit of inventing provenance. The key
    # itself is still required -- absent stays a refusal.
    return isinstance(value, list) and all(_is_nonempty_str(item) for item in value)


def _is_nonempty_int_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _is_shard_list(value: Any) -> bool:
    # Emptiness is checked separately so it gets its OWN refusal code and message: "you declared
    # no shards" and "you declared something that is not a list of shards" are different defects.
    return isinstance(value, list)


BASE_REQUIRED: tuple[FieldSpec, ...] = (
    FieldSpec(
        "campaign_id", "a non-empty identifier matching the destination directory name",
        "the store is addressed by campaign id; a mismatch is how a campaign gets banked into "
        "a scratch directory and lost",
        _is_campaign_id),
    FieldSpec(
        "cell", "a non-empty string naming the cell this artifact answers",
        "a shard set with no cell cannot be matched back to the question it was run for",
        _is_nonempty_str),
    FieldSpec(
        "checkpoint", "a non-empty path string, as it existed at run time",
        "every downstream number is conditional on the checkpoint that produced it",
        _is_nonempty_str),
    FieldSpec(
        "checkpoint_sha256", "64 hex characters",
        "the path is not an identity: two runs at the same path can be different weights. This "
        "field is what makes a re-download distinguishable from a re-run",
        _is_sha256),
    FieldSpec(
        "rollouts", "an integer >= 1 (R, rollouts per leaf/arm)",
        "the label noise budget is 0.5/sqrt(R * distinct leaves); a number quoted without R "
        "has no stated noise floor",
        _int_at_least(1)),
    FieldSpec(
        "sims", "an integer >= 1 (search simulations per decision)",
        "the search budget is half of what any strength delta is a delta OF",
        _int_at_least(1)),
    FieldSpec(
        "depth", "an integer >= 1 (search depth)",
        "the search budget is half of what any strength delta is a delta OF",
        _int_at_least(1)),
    FieldSpec(
        "arms", "a non-empty string naming the arms setting",
        "top-2-only and all-arms headroom are different estimands with the same column names",
        _is_nonempty_str),
    FieldSpec(
        "image", "a non-empty image reference (digest preferred over tag)",
        "the two campaigns this program must reconcile differ by an engine build and nothing "
        "else; the image is the only field that records it",
        _is_nonempty_str),
    FieldSpec(
        "launcher", "a non-empty string naming the launcher that ran the cell",
        "'what ran this' is not reconstructible from the shards",
        _is_nonempty_str),
    FieldSpec(
        "launcher_flags", "a list of non-empty strings (an empty list is accepted; the key is not)",
        "a launcher plus its flags is the reproduction recipe; the launcher alone is not",
        _is_flag_list),
    FieldSpec(
        "shards", "a non-empty list of shard records",
        "an empty shard list is a zero-evidence bank wearing the shape of evidence",
        _is_shard_list),  # emptiness -> EMPTY_SHARD_LIST, a separate code with its own message
)

#: Validated when present; the group requirement below is what makes one of them mandatory.
SEED_ALTERNATIVES: tuple[FieldSpec, ...] = (
    FieldSpec(
        "seed_band", "a non-empty string describing the seed band",
        "seed reuse across arms is the difference between an independent sample and a paired "
        "one, and it cannot be recovered later",
        _is_nonempty_str),
    FieldSpec(
        "seeds", "a non-empty list of integers",
        "the exact seeds are what make a re-run comparable to this run",
        _is_nonempty_int_list),
)

SHARD_REQUIRED: tuple[FieldSpec, ...] = (
    FieldSpec(
        "file", "a relative path inside the source directory",
        "the shard's name in the bank",
        _is_nonempty_str),
    FieldSpec(
        "sha256", "64 hex characters, recomputed and compared against the file on disk",
        "a report must cite the shard sha256 it was computed from; a declared-but-unverified "
        "sha is a citation to nothing",
        _is_sha256),
    FieldSpec(
        "exit_status", "an integer >= 0 (0 for a clean shard)",
        "the store contract requires the exit status of EVERY shard: a silently-failed shard "
        "is how a partial campaign gets pooled as a whole one",
        _int_at_least(0)),
    FieldSpec(
        "pod", "a non-empty string",
        "the pod name is the only handle on the run's logs afterwards",
        _is_nonempty_str),
    FieldSpec(
        "node", "a non-empty string",
        "node identity is how a hardware-correlated outlier becomes visible instead of "
        "averaging into a result",
        _is_nonempty_str),
)

#: Phase 1 instrument 2 (the oracle-leaf / rollout arbiter). These are the arm's own estimand
#: caveats, and they are REQUIRED rather than optional because they must travel with every
#: number the arm produces. See the program's Phase 1 instrument 2 and the crate's
#: ``leaf_eval="rollout_crate"`` estimand warning.
ROLLOUT_ARBITER_REQUIRED: tuple[FieldSpec, ...] = (
    FieldSpec(
        "rollout_policy", "a policy with a declared estimand caveat: " "'uniform'",
        "the rollout policy IS the estimand. 'uniform' prices P(win | both seats uniform-random "
        "from here), which is NOT the vhprobe shards' policy-continuation true_*",
        _is_nonempty_str),
    FieldSpec(
        "rollout_fallback_fraction", "a number in [0.0, 1.0] (0.0 is a legal value: a pure oracle)",
        "a capped rollout falls back to the handcrafted leaf, so a high fallback fraction means "
        "the 'oracle' was mostly the incumbent evaluator. Without this field a blend can be "
        "reported as an oracle",
        _is_fraction),
    FieldSpec(
        "leaf_batch", "an integer >= 1",
        "leaf_batch=1 is the sequential regime the crate fidelity gate certifies; >1 is a "
        "measured fidelity LOSS and must be visible on the artifact that carries it",
        _int_at_least(1)),
    FieldSpec(
        "rollout_threads", "an integer >= 1",
        "the paired-eval opponent is TIME-budgeted and overlaps this arm on the same pod, so "
        "CPU spent here is CPU stolen from it -- a confound in the flattering direction",
        _int_at_least(1)),
    FieldSpec(
        "rollout_threads_cpu_budget_ack", "a real boolean (not a truthy string)",
        "whether the CPU-budget hazard was checked against the run's shard concurrency is a "
        "property of the deployment, so it must be asserted rather than inferred",
        _is_bool),
)

#: The caveat text that travels with a number produced under each rollout policy. A policy
#: absent from this table is a REFUSAL, not an empty caveat: the tool cannot state an estimand
#: it has never been told, and silence would read as "no caveat applies".
ROLLOUT_POLICY_ESTIMANDS: Mapping[str, Mapping[str, str]] = {
    "uniform": {
        "prices": "P(side one wins | BOTH seats play uniformly at random from here).",
        "is_not": (
            "NOT the vhprobe shards' true_a/true_b, which are terminal win probabilities under "
            "POLICY continuation. The two are different estimands and must not be compared "
            "without a measured bridge."
        ),
        "consequence": (
            "A null on this arm is confounded with rollout-policy bias and does not by itself "
            "convict the search mechanism."
        ),
    },
}

#: Keys this tool computes. A caller supplying one is a refusal, not an override: the estimand
#: caveat above all must not be forgeable by the writer whose numbers it constrains.
DERIVED_KEYS: frozenset[str] = frozenset({
    "schema",
    "banked_at_utc",
    "banked_by",
    "shard_count",
    "per_file",
    "estimand",
    "nonzero_exit_shards_acknowledged",
})


@dataclass(frozen=True)
class ArtifactKind:
    """A declared required-field set. Adding a field is one edit, here, and only here."""

    name: str
    description: str
    required: tuple[FieldSpec, ...]
    required_one_of: tuple[tuple[FieldSpec, ...], ...] = ()
    shard_required: tuple[FieldSpec, ...] = SHARD_REQUIRED
    derives_estimand: bool = False

    def field_names(self) -> tuple[str, ...]:
        alternatives = tuple(
            spec.name for group in self.required_one_of for spec in group
        )
        return tuple(spec.name for spec in self.required) + alternatives


ARTIFACT_KINDS: Mapping[str, ArtifactKind] = {
    "campaign.v1": ArtifactKind(
        name="campaign.v1",
        description="A banked search-cell campaign: shards plus the store contract's stamp.",
        required=BASE_REQUIRED,
        required_one_of=(SEED_ALTERNATIVES,),
    ),
    "phase1.instrument2.rollout-arbiter.v1": ArtifactKind(
        name="phase1.instrument2.rollout-arbiter.v1",
        description=(
            "Phase 1 instrument 2, the oracle-leaf rollout arbiter. Everything campaign.v1 "
            "requires, plus the arm's estimand caveats as first-class fields."
        ),
        required=BASE_REQUIRED + ROLLOUT_ARBITER_REQUIRED,
        required_one_of=(SEED_ALTERNATIVES,),
        derives_estimand=True,
    ),
}


@dataclass(frozen=True)
class BankResult:
    """What was written, so a launcher can log it rather than re-derive it."""

    status: str  # "banked" | "unchanged"
    out_dir: Path
    provenance_path: Path
    sha256sums_path: Path
    shard_count: int
    replaced_path: Path | None = None


# ----------------------------------------------------------------------------------------------
# Hashing helpers
# ----------------------------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file's sha256. Streamed because shard sets outgrow memory, not for speed."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    """Indirection with a purpose: the atomicity test injects a failure here."""

    shutil.copy2(source, destination)


def _describe(value: Any) -> str:
    rendered = json.dumps(value, default=repr)
    return rendered if len(rendered) <= 120 else rendered[:117] + "..."


def _zero_or_empty(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return isinstance(value, (str, list, dict, tuple)) and len(value) == 0


# ----------------------------------------------------------------------------------------------
# Validation -- the single path
# ----------------------------------------------------------------------------------------------
def _check_field(
    container: Mapping[str, Any],
    spec: FieldSpec,
    *,
    where: str,
    kind_name: str,
) -> list[Refusal]:
    """Three outcomes, three codes. Absent, null and invalid are never merged."""

    if spec.name not in container:
        return [Refusal(
            MISSING_REQUIRED_FIELD,
            f"{where}: '{spec.name}' is ABSENT. Artifact kind '{kind_name}' requires it "
            f"({spec.accepts}), because {spec.why}. Absent is not empty and not zero: supply "
            f"the value, or do not bank.",
        )]
    value = container[spec.name]
    if value is None:
        return [Refusal(
            NULL_REQUIRED_FIELD,
            f"{where}: '{spec.name}' is present but NULL. A null provenance field is a refusal, "
            f"not a recorded absence -- it reads downstream as 'this does not apply' when it "
            f"means 'nobody supplied it'. Required: {spec.accepts}, because {spec.why}.",
        )]
    if not spec.check(value):
        tail = ""
        if _zero_or_empty(value):
            tail = (
                " A zero or empty value is a VALUE, not an absence; it is refused on its own "
                "terms, with a different code from the absent case."
            )
        return [Refusal(
            INVALID_FIELD_VALUE,
            f"{where}: '{spec.name}' = {_describe(value)} is not {spec.accepts}. Required "
            f"because {spec.why}.{tail}",
        )]
    return []


def _validate_shards(
    kind: ArtifactKind,
    shards: Sequence[Any],
    source_dir: Path,
    *,
    allow_nonzero_shard_exit: bool,
) -> tuple[list[Refusal], list[dict[str, Any]]]:
    reasons: list[Refusal] = []
    normalized: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for index, shard in enumerate(shards):
        where = f"shards[{index}]"
        if not isinstance(shard, Mapping):
            reasons.append(Refusal(
                SHARD_NOT_A_RECORD,
                f"{where} is {_describe(shard)}, not a record. Every shard carries its own "
                f"sha256, exit status, pod and node; a bare filename cannot.",
            ))
            continue

        field_reasons: list[Refusal] = []
        for spec in kind.shard_required:
            field_reasons.extend(
                _check_field(shard, spec, where=where, kind_name=kind.name)
            )
        if field_reasons:
            reasons.extend(field_reasons)
            continue

        relative = str(shard["file"])
        if (
            os.path.isabs(relative)
            or "\\" in relative
            or relative.startswith("~")
            or any(part in ("..", "") for part in Path(relative).parts)
        ):
            reasons.append(Refusal(
                UNSAFE_SHARD_PATH,
                f"{where}: file {_describe(relative)} is not a relative path inside the source "
                f"directory. A banked shard's name is part of the artifact; an absolute path or "
                f"a '..' component would bank something from outside the harvest.",
            ))
            continue

        if relative in seen:
            reasons.append(Refusal(
                DUPLICATE_SHARD_FILE,
                f"{where}: file {_describe(relative)} was already declared at "
                f"shards[{seen[relative]}]. Two records for one file mean one of them is wrong "
                f"and SHA256SUMS would carry only one of them.",
            ))
            continue
        seen[relative] = index

        exit_status = int(shard["exit_status"])
        if exit_status != 0 and not allow_nonzero_shard_exit:
            reasons.append(Refusal(
                UNACKNOWLEDGED_SHARD_FAILURE,
                f"{where}: file {_describe(relative)} exited {exit_status}. Banking a failed "
                f"shard is allowed but never silent: pass allow_nonzero_shard_exit=True "
                f"(--allow-nonzero-shard-exit) so the stamp records that the failure was seen, "
                f"or drop the shard.",
            ))
            continue

        path = source_dir / relative
        if not path.is_file():
            reasons.append(Refusal(
                SHARD_FILE_ABSENT,
                f"{where}: declared file {_describe(relative)} does not exist under the source "
                f"directory ({source_dir}). A declared-but-absent shard is exactly the partial "
                f"artifact this tool exists to refuse.",
            ))
            continue

        declared = str(shard["sha256"]).lower()
        actual = sha256_file(path)
        if actual != declared:
            reasons.append(Refusal(
                SHARD_SHA256_MISMATCH,
                f"{where}: declared sha256 {declared} for {_describe(relative)} but the file on "
                f"disk hashes to {actual}. Either the shard changed after it was measured or the "
                f"stamp describes a different run; both make every citation of it false.",
            ))
            continue

        record = {key: value for key, value in shard.items()}
        record["file"] = relative
        record["sha256"] = actual
        record["bytes"] = path.stat().st_size
        normalized.append(record)

    return reasons, normalized


def _validate(
    stamp: Mapping[str, Any],
    kind: ArtifactKind,
    source_dir: Path,
    *,
    out_dir: Path,
    allow_nonzero_shard_exit: bool,
    verify_checkpoint: Path | None,
) -> tuple[list[Refusal], list[dict[str, Any]]]:
    reasons: list[Refusal] = []

    supplied_derived = sorted(DERIVED_KEYS & set(stamp))
    if supplied_derived:
        reasons.append(Refusal(
            DERIVED_FIELD_SUPPLIED,
            f"the stamp supplies derived key(s) {supplied_derived}. This tool computes them, "
            f"and the estimand caveat above all must not be forgeable by the writer whose "
            f"numbers it constrains.",
        ))

    for spec in kind.required:
        reasons.extend(_check_field(stamp, spec, where="stamp", kind_name=kind.name))

    for group in kind.required_one_of:
        names = tuple(spec.name for spec in group)
        present = [spec for spec in group if spec.name in stamp and stamp[spec.name] is not None]
        if not present:
            reasons.append(Refusal(
                MISSING_REQUIRED_ALTERNATIVE,
                f"stamp: none of {list(names)} is present with a value. Artifact kind "
                f"'{kind.name}' requires at least one, because {group[0].why}.",
            ))
        for spec in group:
            if spec.name in stamp:
                reasons.extend(_check_field(stamp, spec, where="stamp", kind_name=kind.name))

    if kind.derives_estimand:
        policy = stamp.get("rollout_policy")
        if _is_nonempty_str(policy) and policy not in ROLLOUT_POLICY_ESTIMANDS:
            reasons.append(Refusal(
                UNDECLARED_ESTIMAND,
                f"stamp: rollout_policy {_describe(policy)} has no declared estimand caveat. "
                f"Declared policies: {sorted(ROLLOUT_POLICY_ESTIMANDS)}. The caveat is what the "
                f"arm's numbers mean, so an undeclared policy cannot be banked -- add its "
                f"estimand to ROLLOUT_POLICY_ESTIMANDS first.",
            ))
        threads = stamp.get("rollout_threads")
        ack = stamp.get("rollout_threads_cpu_budget_ack")
        if _int_at_least(2)(threads) and ack is False:
            reasons.append(Refusal(
                UNACKNOWLEDGED_CPU_BUDGET,
                f"stamp: rollout_threads={threads} > 1 with rollout_threads_cpu_budget_ack="
                f"False. The time-budgeted opponent shares the pod, so unacknowledged extra "
                f"threads read as a strength gain for this arm -- a confound in the flattering "
                f"direction on the arm that is supposed to arbitrate.",
            ))

    if not source_dir.is_dir():
        reasons.append(Refusal(
            SOURCE_DIR_ABSENT,
            f"source directory {source_dir} does not exist or is not a directory; there is "
            f"nothing to bank from.",
        ))
        shard_reasons, normalized = [], []
    else:
        shards = stamp.get("shards")
        if isinstance(shards, list) and shards:
            shard_reasons, normalized = _validate_shards(
                kind, shards, source_dir,
                allow_nonzero_shard_exit=allow_nonzero_shard_exit,
            )
        else:
            # An absent or null `shards` is already reported by the field check above; only the
            # EMPTY case needs its own voice, because an empty list is a VALUE that would
            # otherwise sail through as a zero-shard bank.
            shard_reasons, normalized = [], []
            if isinstance(shards, list) and not shards:
                shard_reasons = [Refusal(
                    EMPTY_SHARD_LIST,
                    "stamp: 'shards' is an EMPTY list. Zero shards is not a small bank, it is no "
                    "bank: a directory with a provenance stamp and no evidence reads as banked "
                    "and cites nothing. This is refused separately from 'shards' being absent.",
                )]
        reasons.extend(shard_reasons)

    campaign_id = stamp.get("campaign_id")
    if _is_campaign_id(campaign_id) and out_dir.name != campaign_id:
        reasons.append(Refusal(
            CAMPAIGN_ID_DIRECTORY_MISMATCH,
            f"campaign_id {_describe(campaign_id)} does not match the destination directory "
            f"name {_describe(out_dir.name)}. The store is addressed by campaign id; a campaign "
            f"banked under another name is the scratch-directory loss with extra steps.",
        ))

    if verify_checkpoint is not None:
        declared = stamp.get("checkpoint_sha256")
        if not verify_checkpoint.is_file():
            reasons.append(Refusal(
                CHECKPOINT_SHA256_MISMATCH,
                f"--verify-checkpoint {verify_checkpoint} is not a file, so the declared "
                f"checkpoint sha256 could not be re-derived. An unverifiable verification is "
                f"not a pass.",
            ))
        elif _is_sha256(declared):
            actual = sha256_file(verify_checkpoint)
            if actual != str(declared).lower():
                reasons.append(Refusal(
                    CHECKPOINT_SHA256_MISMATCH,
                    f"declared checkpoint_sha256 {str(declared).lower()} but "
                    f"{verify_checkpoint} hashes to {actual}. The stamp describes different "
                    f"weights than the ones presented.",
                ))

    return reasons, normalized


# ----------------------------------------------------------------------------------------------
# Building the artifact
# ----------------------------------------------------------------------------------------------
def _estimand_block(stamp: Mapping[str, Any]) -> dict[str, Any]:
    policy = str(stamp["rollout_policy"])
    caveat = ROLLOUT_POLICY_ESTIMANDS[policy]
    fallback = float(stamp["rollout_fallback_fraction"])
    leaf_batch = int(stamp["leaf_batch"])
    block: dict[str, Any] = {
        "rollout_policy": policy,
        "prices": caveat["prices"],
        "is_not": caveat["is_not"],
        "consequence": caveat["consequence"],
        "rollout_fallback_fraction": fallback,
        "fallback_note": (
            "0.0: every priced leaf reached a terminal observation."
            if fallback == 0.0
            else f"{fallback}: that share of rollouts hit the ply cap or a dead end and fell "
                 f"back to the handcrafted leaf. This arm is a BLEND to that extent, not an "
                 f"oracle."
        ),
        "leaf_batch": leaf_batch,
        "leaf_batch_note": (
            "1: the sequential selection regime the crate fidelity gate certifies."
            if leaf_batch == 1
            else f"{leaf_batch}: >1 is a measured fidelity LOSS against the sequential regime."
        ),
        "rollout_threads": int(stamp["rollout_threads"]),
        "rollout_threads_cpu_budget_ack": bool(stamp["rollout_threads_cpu_budget_ack"]),
    }
    return block


def build_provenance(
    stamp: Mapping[str, Any],
    kind: ArtifactKind,
    shards: Sequence[Mapping[str, Any]],
    *,
    banked_at_utc: str,
    allow_nonzero_shard_exit: bool,
) -> dict[str, Any]:
    """The stamp as it will be written. Pure, so the destination check can compare against it."""

    provenance: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "artifact_kind": kind.name,
        "banked_at_utc": banked_at_utc,
        "banked_by": f"{TOOL_NAME} v{TOOL_VERSION}",
    }
    for key, value in stamp.items():
        if key in ("shards", "artifact_kind"):
            continue
        provenance[key] = value
    provenance["shard_count"] = len(shards)
    provenance["shards"] = [dict(record) for record in shards]
    # The {file, sha256, bytes} projection the store's existing HARVEST.json readers consume.
    provenance["per_file"] = [
        {"file": record["file"], "sha256": record["sha256"], "bytes": record["bytes"]}
        for record in shards
    ]
    provenance["nonzero_exit_shards_acknowledged"] = (
        sorted(
            str(record["file"]) for record in shards if int(record["exit_status"]) != 0
        )
        if allow_nonzero_shard_exit
        else []
    )
    if kind.derives_estimand:
        provenance["estimand"] = _estimand_block(stamp)
    return provenance


def render_sha256sums(shards: Sequence[Mapping[str, Any]]) -> str:
    """``sha256sum`` output shape: two spaces, paths relative to the banked directory.

    Sorted by PATH, not by digest, matching the store's existing ``SHA256SUMS`` files -- a diff
    between two banks of the same campaign should show the shard that changed, not a reshuffle.
    """

    ordered = sorted(shards, key=lambda record: str(record["file"]))
    lines = [f"{record['sha256']}  {record['file']}" for record in ordered]
    return "\n".join(lines) + "\n"


def _provenance_json(provenance: Mapping[str, Any]) -> str:
    return json.dumps(provenance, indent=2, sort_keys=False) + "\n"


def _comparable(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The stamp with the fields that legitimately differ between two identical banks removed."""

    return {
        key: value
        for key, value in provenance.items()
        if key not in ("banked_at_utc",)
    }


def _existing_matches(
    out_dir: Path, provenance: Mapping[str, Any], sha256sums: str
) -> bool:
    stamp_path = out_dir / PROVENANCE_FILENAME
    sums_path = out_dir / SHA256SUMS_FILENAME
    if not stamp_path.is_file() or not sums_path.is_file():
        return False
    try:
        existing = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(existing, Mapping):
        return False
    if _comparable(existing) != _comparable(provenance):
        return False
    if sums_path.read_text(encoding="utf-8") != sha256sums:
        return False
    for record in provenance["shards"]:
        banked = out_dir / str(record["file"])
        if not banked.is_file() or sha256_file(banked) != record["sha256"]:
            return False
    return True


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems refuse fsync on directories. The rename below is still ordered.
        pass
    finally:
        os.close(fd)


def _write_atomically(
    out_dir: Path,
    source_dir: Path,
    shards: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    sha256sums: str,
    *,
    overwrite: bool,
) -> Path | None:
    """Build the whole artifact in a sibling temp dir, then rename it into place.

    Returns the path the displaced artifact was moved to, if any. Displaced artifacts are
    MOVED, never deleted: an overwrite is still somebody's evidence until they say otherwise.
    """

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".bank-tmp-{out_dir.name}-", dir=out_dir.parent))
    replaced: Path | None = None
    try:
        for record in shards:
            destination = staging / str(record["file"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(source_dir / str(record["file"]), destination)
        (staging / PROVENANCE_FILENAME).write_text(
            _provenance_json(provenance), encoding="utf-8"
        )
        (staging / SHA256SUMS_FILENAME).write_text(sha256sums, encoding="utf-8")
        os.chmod(staging, 0o755)  # mkdtemp is 0700; a banked artifact is readable evidence
        _fsync_dir(staging)

        if out_dir.exists():
            if any(out_dir.iterdir()):
                if not overwrite:  # pragma: no cover - the caller already refused this case
                    raise RuntimeError(f"{out_dir} exists and overwrite was not requested")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                replaced = out_dir.with_name(f"{out_dir.name}.replaced-{stamp}")
                os.rename(out_dir, replaced)
            else:
                # An existing EMPTY directory is not an artifact and needs no overwrite; it is
                # removed so the staging directory can be renamed onto its name atomically.
                out_dir.rmdir()
        try:
            os.rename(staging, out_dir)
        except OSError:
            if replaced is not None:  # put the displaced artifact back before reporting
                os.rename(replaced, out_dir)
                replaced = None
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return replaced


# ----------------------------------------------------------------------------------------------
# The entry point libraries use
# ----------------------------------------------------------------------------------------------
def resolve_kind(name: Any) -> ArtifactKind:
    if not isinstance(name, str) or name not in ARTIFACT_KINDS:
        raise BankRefusal([Refusal(
            UNKNOWN_ARTIFACT_KIND,
            f"artifact kind {_describe(name)} is not declared. Known kinds: "
            f"{sorted(ARTIFACT_KINDS)}. A kind IS its required-field list; banking under an "
            f"undeclared one would validate nothing.",
        )])
    return ARTIFACT_KINDS[name]


def bank_artifact(
    stamp: Mapping[str, Any],
    source_dir: Path | str,
    out_dir: Path | str,
    *,
    kind: str | None = None,
    overwrite: bool = False,
    allow_nonzero_shard_exit: bool = False,
    verify_checkpoint: Path | str | None = None,
    banked_at_utc: str | None = None,
) -> BankResult:
    """Bank ``stamp``'s shards from ``source_dir`` into ``out_dir``, or raise :class:`BankRefusal`.

    Every check runs before any byte is written, and all failures are reported together: a
    caller fixing an incomplete stamp should learn the whole list once, not one field per run.
    """

    if not isinstance(stamp, Mapping):
        raise BankRefusal([Refusal(
            STAMP_NOT_A_MAPPING,
            f"the provenance stamp is {_describe(stamp)}, not an object of fields.",
        )])

    declared_kind = stamp.get("artifact_kind")
    if kind is not None and declared_kind is not None and declared_kind != kind:
        raise BankRefusal([Refusal(
            ARTIFACT_KIND_MISMATCH,
            f"the stamp declares artifact_kind {_describe(declared_kind)} but the caller asked "
            f"for {_describe(kind)}. Two answers to 'what is being validated' is no answer.",
        )])
    artifact_kind = resolve_kind(kind if kind is not None else declared_kind)

    source = Path(source_dir)
    destination = Path(out_dir)
    checkpoint = Path(verify_checkpoint) if verify_checkpoint is not None else None

    reasons, shards = _validate(
        stamp, artifact_kind, source,
        out_dir=destination,
        allow_nonzero_shard_exit=allow_nonzero_shard_exit,
        verify_checkpoint=checkpoint,
    )

    provenance: dict[str, Any] | None = None
    sha256sums = ""
    if not reasons:
        provenance = build_provenance(
            stamp, artifact_kind, shards,
            banked_at_utc=banked_at_utc or datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            allow_nonzero_shard_exit=allow_nonzero_shard_exit,
        )
        sha256sums = render_sha256sums(shards)

        if destination.exists() and any(destination.iterdir()):
            if _existing_matches(destination, provenance, sha256sums):
                return BankResult(
                    status="unchanged",
                    out_dir=destination,
                    provenance_path=destination / PROVENANCE_FILENAME,
                    sha256sums_path=destination / SHA256SUMS_FILENAME,
                    shard_count=len(shards),
                )
            if not overwrite:
                reasons.append(Refusal(
                    DESTINATION_HOLDS_DIFFERENT_ARTIFACT,
                    f"{destination} already holds an artifact that differs from the one being "
                    f"banked. Silently replacing banked evidence destroys the record the store "
                    f"exists to keep: pass overwrite=True (--overwrite) if that is intended.",
                ))

    if reasons:
        raise BankRefusal(reasons)
    assert provenance is not None  # unreachable otherwise; keeps the type narrow

    try:
        replaced = _write_atomically(
            destination, source, shards, provenance, sha256sums, overwrite=overwrite
        )
    except BankRefusal:
        raise
    except Exception as error:
        raise BankRefusal([Refusal(
            WRITE_FAILED,
            f"the artifact could not be written ({type(error).__name__}: {error}). The staging "
            f"directory was removed, so {destination} holds no partial artifact.",
        )]) from error

    return BankResult(
        status="banked",
        out_dir=destination,
        provenance_path=destination / PROVENANCE_FILENAME,
        sha256sums_path=destination / SHA256SUMS_FILENAME,
        shard_count=len(shards),
        replaced_path=replaced,
    )


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def describe_kind(kind: ArtifactKind) -> str:
    lines = [f"{kind.name} -- {kind.description}", "", "REQUIRED stamp fields:"]
    for spec in kind.required:
        lines.append(f"  {spec.name}: {spec.accepts}")
        lines.append(f"      why: {spec.why}")
    for group in kind.required_one_of:
        names = " OR ".join(spec.name for spec in group)
        lines.append(f"  at least one of: {names}")
        for spec in group:
            lines.append(f"      {spec.name}: {spec.accepts}")
    lines.append("")
    lines.append("REQUIRED per-shard fields (every entry of `shards`):")
    for spec in kind.shard_required:
        lines.append(f"  {spec.name}: {spec.accepts}")
    lines.append("")
    lines.append(f"DERIVED by the tool, and refused if supplied: {sorted(DERIVED_KEYS)}")
    if kind.derives_estimand:
        lines.append(
            f"ESTIMAND caveat is derived from rollout_policy; declared policies: "
            f"{sorted(ROLLOUT_POLICY_ESTIMANDS)}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bank a campaign's shards into the campaign store with a complete provenance "
            "stamp, or refuse and write nothing."
        ),
    )
    parser.add_argument(
        "--stamp",
        help="path to the provenance stamp JSON, or '-' for stdin.",
    )
    parser.add_argument("--source-dir", help="directory the declared shard files live in.")
    parser.add_argument("--out-dir", help="destination directory, named after the campaign id.")
    parser.add_argument(
        "--kind",
        help=f"artifact kind; one of {sorted(ARTIFACT_KINDS)}. May instead be declared in the "
             f"stamp as 'artifact_kind'.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace a differing artifact already in the destination (the displaced one is "
             "moved aside, not deleted).",
    )
    parser.add_argument(
        "--allow-nonzero-shard-exit", action="store_true",
        help="bank shards whose exit status is not 0, recording them in the stamp.",
    )
    parser.add_argument(
        "--verify-checkpoint",
        help="re-hash this file and refuse unless it matches the stamp's checkpoint_sha256.",
    )
    parser.add_argument(
        "--print-required-fields", metavar="KIND", nargs="?", const="",
        help="print the declared required-field set for KIND (or all kinds) and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.print_required_fields is not None:
        names: Iterable[str] = (
            [args.print_required_fields] if args.print_required_fields else sorted(ARTIFACT_KINDS)
        )
        try:
            kinds = [resolve_kind(name) for name in names]
        except BankRefusal as refusal:
            print(refusal, file=sys.stderr)
            return REFUSAL_EXIT
        print("\n\n".join(describe_kind(kind) for kind in kinds))
        return 0

    missing = [
        flag for flag, value in (
            ("--stamp", args.stamp),
            ("--source-dir", args.source_dir),
            ("--out-dir", args.out_dir),
        ) if not value
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    try:
        raw = (
            sys.stdin.read() if args.stamp == "-"
            else Path(args.stamp).read_text(encoding="utf-8")
        )
    except OSError as error:
        print(
            Refusal(STAMP_NOT_JSON, f"the stamp could not be read ({error})."),
            file=sys.stderr,
        )
        return REFUSAL_EXIT
    try:
        stamp = json.loads(raw)
    except json.JSONDecodeError as error:
        print(
            Refusal(STAMP_NOT_JSON, f"{args.stamp} is not valid JSON ({error})."),
            file=sys.stderr,
        )
        return REFUSAL_EXIT

    try:
        result = bank_artifact(
            stamp,
            args.source_dir,
            args.out_dir,
            kind=args.kind,
            overwrite=args.overwrite,
            allow_nonzero_shard_exit=args.allow_nonzero_shard_exit,
            verify_checkpoint=args.verify_checkpoint,
        )
    except BankRefusal as refusal:
        print(refusal, file=sys.stderr)
        return REFUSAL_EXIT

    if result.status == "unchanged":
        print(f"unchanged: {result.out_dir} already holds this exact artifact "
              f"({result.shard_count} shards); nothing was rewritten.")
        return 0
    print(f"banked: {result.shard_count} shards -> {result.out_dir}")
    print(f"  {result.provenance_path}")
    print(f"  {result.sha256sums_path}")
    if result.replaced_path is not None:
        print(f"  displaced artifact moved to {result.replaced_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
