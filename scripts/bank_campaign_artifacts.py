#!/usr/bin/env python3
"""Bank a campaign's files into the campaign store with a complete provenance stamp -- or refuse.

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
* an empty file list is a refusal, not a zero-shard bank;
* a file whose recomputed sha256 disagrees with its declared one is a refusal;
* a declared file that is not on disk is a refusal;
* banking into a directory that already holds a *differing* artifact is a refusal unless
  overwrite is requested explicitly (an identical artifact is a no-op, not a rewrite);
* banking campaign ``X`` into a directory not named ``X`` is a refusal -- that is the
  scratch-directory loss, made unrepresentable.

THE BANKED DIRECTORY VERIFIES AGAINST ITS OWN ``SHA256SUMS``, AS AN INVARIANT
----------------------------------------------------------------------------
An independent review of this tool's first version found four inputs that produced a *written*
artifact which should have been refused, two of which banked a directory that failed its own
``shasum -a 256 -c SHA256SUMS``. Both had one root cause: the tool trusted facts it had measured
earlier about a source directory other processes can write to. It no longer does.

* A file declared as ``PROVENANCE.json`` or ``SHA256SUMS`` used to be copied into staging and
  then overwritten by the stamp -- citing a sha256 for bytes the bank had destroyed. Reserved
  names are now refused, which matters because Phase 0 item 1 means pointing ``--source-dir``
  at directories that *already contain both files*.
* The declared sha256 was verified against the SOURCE and never against the copy, so a source
  mutated between hashing and copying banked a self-inconsistent artifact. Every staged byte is
  now re-hashed against the stamp immediately before the ``rename``, so self-consistency is a
  verified invariant of the write rather than an assumption about the source holding still.

ATOMICITY
---------
Validation is *complete before any byte is written*: every scalar, every file, the destination
check and the optional checkpoint re-hash all run first, and all failures are collected and
reported together. Only then is a sibling temp directory built, filled, re-verified, fsynced
and ``rename``d into place. If anything fails during that write -- a full disk, a vanished
source, a ``KeyboardInterrupt`` -- the temp directory is removed, any displaced artifact is
moved back, and the refusal is raised. A refusal never leaves a partial artifact behind, and an
interrupted overwrite never leaves the store without the directory its campaign id addresses.
See ``tests/test_bank_campaign_artifacts.py``; a guard nobody has watched read False certifies
nothing.

THE SCHEMA IS DATA, NOT CONTROL FLOW
------------------------------------
Required fields live in ``ARTIFACT_KINDS`` as declared ``FieldSpec`` tuples per artifact kind.
Adding a required field is one edit in one place and cannot be forgotten on one of several code
paths, because there is only one path: :func:`_validate`. Three kinds are declared:

* ``campaign.v1`` -- a results cell: shards, seeds, search budget, per-shard exit status.
* ``phase1.instrument2.rollout-arbiter.v1`` -- the same shape plus the arbiter arm's own
  estimand caveats as first-class REQUIRED fields, so a caller *cannot* bank a cell whose
  numbers travel without them. Both write ``schema`` = ``pokezero.campaign-store.provenance.v1``:
  the second is a strict superset of the first, not a different document.
* ``staging.v1`` -- an INPUTS cell (staged checkpoint, exported tables, model artifacts): no
  games, no seeds, no results. Its shape is disjoint from the results shape, so it writes a
  DIFFERENT schema string, ``pokezero.campaign-store.staging.v1``. One schema string never
  names two disjoint layouts; ``artifact_kind`` distinguishes refinements within one.

PUBLIC-REPO HYGIENE
-------------------
Every path, image reference, pod and node name arrives as an argument or a stamp field.
Nothing about any deployment is hardcoded here, and nothing about one may be added.

SCOPE
-----
This tool governs artifacts it banks. The store's pre-existing cells are NOT retrofitted: their
``HARVEST.json`` files use ``rollouts_per_arm`` where this requires ``rollouts``, and a single
``launch_record`` string where this requires ``launcher`` + ``launcher_flags``. Aliasing those
names would let a caller satisfy ``rollouts`` with a different quantity's value, which is the
estimand confusion this whole program is trying to undo -- so they are deliberately not aliased.
Existing cells stand as banked; new banks go through here.

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
        for reason in refusal.reasons:
            log(reason.code, reason.field, reason.message)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_NAME = "scripts/bank_campaign_artifacts.py"
TOOL_VERSION = "2"

PROVENANCE_FILENAME = "PROVENANCE.json"
SHA256SUMS_FILENAME = "SHA256SUMS"
STAGING_PREFIX = ".bank-tmp-"

#: A results cell (``campaign.v1`` and its arbiter refinement). One string, one shape.
PROVENANCE_SCHEMA = "pokezero.campaign-store.provenance.v1"
#: An INPUTS cell. Disjoint shape, therefore a different string -- see the module docstring.
STAGING_SCHEMA = "pokezero.campaign-store.staging.v1"

#: Names a banked file may not take: the stamp and the checksum file are written over the top of
#: the copied tree, so a file claiming one of these names loses its bytes and leaves the
#: directory failing its own checksums. Compared case-insensitively, because a case-insensitive
#: filesystem collides on ``provenance.json`` too.
RESERVED_FILENAMES = (PROVENANCE_FILENAME, SHA256SUMS_FILENAME)

#: Refusal exits with this status. Not 2: argparse owns 2 for usage errors, and a caller must
#: be able to tell "you invoked me wrongly" from "your artifact is incomplete".
REFUSAL_EXIT = 3

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_CAMPAIGN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# Refusal codes. Each one is a guard, and each one has a demonstrated failing input in
# tests/test_bank_campaign_artifacts.py. A code with no failing-input test does not belong here.
# "SHARD" in a code name means "a declared evidence file": the results kinds call them shards,
# the staging kind calls them staged files, and one validation path serves both.
MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
NULL_REQUIRED_FIELD = "NULL_REQUIRED_FIELD"
INVALID_FIELD_VALUE = "INVALID_FIELD_VALUE"
MISSING_REQUIRED_ALTERNATIVE = "MISSING_REQUIRED_ALTERNATIVE"
EMPTY_SHARD_LIST = "EMPTY_SHARD_LIST"
SHARD_NOT_A_RECORD = "SHARD_NOT_A_RECORD"
SHARD_FILE_ABSENT = "SHARD_FILE_ABSENT"
SHARD_SHA256_MISMATCH = "SHARD_SHA256_MISMATCH"
SHARD_BYTES_MISMATCH = "SHARD_BYTES_MISMATCH"
EMPTY_SHARD_FILE = "EMPTY_SHARD_FILE"
DUPLICATE_SHARD_FILE = "DUPLICATE_SHARD_FILE"
UNSAFE_SHARD_PATH = "UNSAFE_SHARD_PATH"
NON_CANONICAL_SHARD_PATH = "NON_CANONICAL_SHARD_PATH"
RESERVED_SHARD_NAME = "RESERVED_SHARD_NAME"
SYMLINKED_SHARD = "SYMLINKED_SHARD"
UNACKNOWLEDGED_SHARD_FAILURE = "UNACKNOWLEDGED_SHARD_FAILURE"
UNDECLARED_ESTIMAND = "UNDECLARED_ESTIMAND"
UNACKNOWLEDGED_CPU_BUDGET = "UNACKNOWLEDGED_CPU_BUDGET"
DERIVED_FIELD_SUPPLIED = "DERIVED_FIELD_SUPPLIED"
SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
UNSERIALIZABLE_PROVENANCE = "UNSERIALIZABLE_PROVENANCE"
UNKNOWN_ARTIFACT_KIND = "UNKNOWN_ARTIFACT_KIND"
ARTIFACT_KIND_MISMATCH = "ARTIFACT_KIND_MISMATCH"
STAMP_NOT_A_MAPPING = "STAMP_NOT_A_MAPPING"
STAMP_NOT_JSON = "STAMP_NOT_JSON"
DUPLICATE_STAMP_KEY = "DUPLICATE_STAMP_KEY"
SOURCE_DIR_ABSENT = "SOURCE_DIR_ABSENT"
CHECKPOINT_SHA256_MISMATCH = "CHECKPOINT_SHA256_MISMATCH"
CAMPAIGN_ID_DIRECTORY_MISMATCH = "CAMPAIGN_ID_DIRECTORY_MISMATCH"
DESTINATION_HOLDS_DIFFERENT_ARTIFACT = "DESTINATION_HOLDS_DIFFERENT_ARTIFACT"
DESTINATION_NOT_A_DIRECTORY = "DESTINATION_NOT_A_DIRECTORY"
STAGED_COPY_MISMATCH = "STAGED_COPY_MISMATCH"
WRITE_FAILED = "WRITE_FAILED"


# ----------------------------------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Refusal:
    """One guard that read False, with the input that made it read False.

    ``field`` is the machine-readable half: a launcher branching on which input was wrong should
    not have to parse prose. It is the dotted location in the stamp (``rollouts``,
    ``shards[1].sha256``) or ``None`` for whole-artifact refusals.
    """

    code: str
    message: str
    field: str | None = None

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

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(reason.field for reason in self.reasons if reason.field is not None)

    def messages_for(self, code: str) -> tuple[str, ...]:
        return tuple(reason.message for reason in self.reasons if reason.code == code)

    def fields_for(self, code: str) -> tuple[str, ...]:
        return tuple(
            reason.field for reason in self.reasons
            if reason.code == code and reason.field is not None
        )

    def _render(self) -> str:
        plural = "" if len(self.reasons) == 1 else "s"
        head = (
            f"REFUSED to bank: {len(self.reasons)} unmet requirement{plural}; "
            f"nothing was written."
        )
        return "\n".join([head, *(f"  - {reason}" for reason in self.reasons)])


# ----------------------------------------------------------------------------------------------
# Value predicates
# ----------------------------------------------------------------------------------------------
#: Unicode general categories that occupy no visible width. A required field whose entire value
#: is drawn from these reads as present and displays as blank -- U+200B, U+FEFF and U+00AD all
#: survive ``str.strip()`` untouched, so ``strip()`` alone is not an emptiness test.
_BLANK_CATEGORIES = frozenset({"Cc", "Cf", "Zs", "Zl", "Zp"})


def _has_visible_character(value: str) -> bool:
    return any(
        not char.isspace() and unicodedata.category(char) not in _BLANK_CATEGORIES
        for char in value
    )


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and _has_visible_character(value)


def _is_campaign_id(value: Any) -> bool:
    return _is_nonempty_str(value) and bool(_CAMPAIGN_ID_RE.fullmatch(value))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_int(value: Any) -> bool:
    # bool is an int subclass in Python; True must never satisfy an integer field.
    return isinstance(value, int) and not isinstance(value, bool)


def _int_at_least(minimum: int) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        return _is_int(value) and value >= minimum

    return check


def _is_fraction(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
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
    return isinstance(value, list) and len(value) > 0 and all(_is_int(item) for item in value)


def _is_file_list(value: Any) -> bool:
    # Emptiness is checked separately so it gets its OWN refusal code and message: "you declared
    # no files" and "you declared something that is not a list of files" are different defects.
    return isinstance(value, list)


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


IDENTITY_REQUIRED: tuple[FieldSpec, ...] = (
    FieldSpec(
        "campaign_id", "a non-empty identifier matching the destination directory name",
        "the store is addressed by campaign id; a mismatch is how a campaign gets banked into "
        "a scratch directory and lost",
        _is_campaign_id),
    FieldSpec(
        "cell", "a non-empty string naming the cell this artifact answers",
        "a file set with no cell cannot be matched back to the question it was run for",
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
)

SHARDS_SPEC = FieldSpec(
    "shards", "a non-empty list of shard records",
    "an empty shard list is a zero-evidence bank wearing the shape of evidence",
    _is_file_list)  # emptiness -> EMPTY_SHARD_LIST, a separate code with its own message

STAGED_FILES_SPEC = FieldSpec(
    "staged_files", "a non-empty list of staged-file records",
    "an inputs cell with no inputs banks nothing while reading as staged",
    _is_file_list)

RESULTS_REQUIRED: tuple[FieldSpec, ...] = IDENTITY_REQUIRED + (
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
    SHARDS_SPEC,
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
        "file", "a canonical relative path inside the source directory",
        "the shard's name in the bank",
        _is_nonempty_str),
    FieldSpec(
        "sha256", "64 hex characters, recomputed and compared against the file on disk",
        "a report must cite the shard sha256 it was computed from; a declared-but-unverified "
        "sha is a citation to nothing",
        _is_sha256),
    FieldSpec(
        "exit_status", "an integer: 0 for a clean shard, POSIX 137 or Python -9 for a kill",
        "the store contract requires the exit status of EVERY shard: a silently-failed shard "
        "is how a partial campaign gets pooled as a whole one. Negative values are accepted "
        "because Python's subprocess convention reports a signal as -N, and an OOM-killed "
        "shard is the case where banking-with-acknowledgement matters most",
        _is_int),
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

#: The staging kind's per-file record. No exit status and no per-file pod: staging is one
#: operation on one pod, recorded once at the top level.
STAGED_FILE_REQUIRED: tuple[FieldSpec, ...] = (
    FieldSpec(
        "file", "a canonical relative path inside the source directory",
        "the input's name in the bank",
        _is_nonempty_str),
    FieldSpec(
        "sha256", "64 hex characters, recomputed and compared against the file on disk",
        "an inputs cell exists so a later run can prove it consumed these exact bytes",
        _is_sha256),
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

#: The staging (INPUTS) kind. No seeds, no search budget, no results -- by construction.
#:
#: The volume the inputs were staged on is deliberately NOT required. It is a property of one
#: deployment style, and putting it in a public tool's contract would force every caller to
#: answer a Kubernetes question; ``staged_from`` already records where the bytes came from, and
#: a caller that has a volume name passes it through as an undeclared field.
STAGING_REQUIRED: tuple[FieldSpec, ...] = IDENTITY_REQUIRED + (
    FieldSpec(
        "staged_at_utc", "a non-empty UTC timestamp string",
        "when the inputs were staged is not when they were banked, and a later run needs the "
        "former to line these bytes up against a build",
        _is_nonempty_str),
    FieldSpec(
        "staged_from", "a non-empty path string the inputs were staged from",
        "an inputs cell whose origin is unrecorded cannot be re-staged or audited",
        _is_nonempty_str),
    FieldSpec(
        "staging_pod", "a non-empty string",
        "the pod name is the only handle on the staging job's logs afterwards",
        _is_nonempty_str),
    FieldSpec(
        "staging_node", "a non-empty string",
        "node identity is how a hardware-correlated staging fault becomes visible",
        _is_nonempty_str),
    FieldSpec(
        "image_id", "a non-empty image reference (digest preferred over tag)",
        "these inputs are only meaningful against the build that will consume them",
        _is_nonempty_str),
    FieldSpec(
        "no_results_banked_here", "a non-empty string asserting what is NOT in this directory",
        "the store contract forbids derived numbers here, and an inputs cell sitting beside "
        "results cells must say so in its own words rather than by absence",
        _is_nonempty_str),
    STAGED_FILES_SPEC,
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
#: caveat above all must not be forgeable by the writer whose numbers it constrains. Matched on
#: ``key.strip().casefold()`` and applied to file records too, because ``Estimand``,
#: ``"estimand "`` and ``shards[0].estimand`` are all forgeries of the same field.
DERIVED_KEYS: frozenset[str] = frozenset({
    "banked_at_utc",
    "banked_by",
    "shard_count",
    "staged_file_count",
    "per_file",
    "estimand",
    "nonzero_exit_shards_acknowledged",
})

#: ``schema`` is derived too, but a stamp declaring the SAME value the tool would write is
#: agreeing, not forging, so it is accepted (and rewritten). A different value is a refusal.
SCHEMA_KEY = "schema"

#: Per-file keys this tool computes. ``bytes`` may be declared -- it is then verified against
#: the file on disk rather than trusted.
VERIFIED_FILE_KEYS: frozenset[str] = frozenset({"bytes"})


@dataclass(frozen=True)
class ArtifactKind:
    """A declared required-field set. Adding a field is one edit, here, and only here."""

    name: str
    schema: str
    description: str
    required: tuple[FieldSpec, ...]
    files_field: str = "shards"
    count_field: str = "shard_count"
    file_required: tuple[FieldSpec, ...] = SHARD_REQUIRED
    required_one_of: tuple[tuple[FieldSpec, ...], ...] = ()
    derives_estimand: bool = False

    def field_names(self) -> tuple[str, ...]:
        alternatives = tuple(spec.name for group in self.required_one_of for spec in group)
        return tuple(spec.name for spec in self.required) + alternatives

    def records_exit_status(self) -> bool:
        return any(spec.name == "exit_status" for spec in self.file_required)


ARTIFACT_KINDS: Mapping[str, ArtifactKind] = {
    "campaign.v1": ArtifactKind(
        name="campaign.v1",
        schema=PROVENANCE_SCHEMA,
        description="A banked search-cell campaign: shards plus the store contract's stamp.",
        required=RESULTS_REQUIRED,
        required_one_of=(SEED_ALTERNATIVES,),
    ),
    "phase1.instrument2.rollout-arbiter.v1": ArtifactKind(
        name="phase1.instrument2.rollout-arbiter.v1",
        schema=PROVENANCE_SCHEMA,
        description=(
            "Phase 1 instrument 2, the oracle-leaf rollout arbiter. Everything campaign.v1 "
            "requires, plus the arm's estimand caveats as first-class fields."
        ),
        required=RESULTS_REQUIRED + ROLLOUT_ARBITER_REQUIRED,
        required_one_of=(SEED_ALTERNATIVES,),
        derives_estimand=True,
    ),
    "staging.v1": ArtifactKind(
        name="staging.v1",
        schema=STAGING_SCHEMA,
        description=(
            "An INPUTS cell: staged checkpoints, exported tables and model artifacts, with no "
            "games, no seeds and no results. Disjoint from the results shape, so it carries a "
            "different schema string."
        ),
        required=STAGING_REQUIRED,
        files_field="staged_files",
        count_field="staged_file_count",
        file_required=STAGED_FILE_REQUIRED,
    ),
}


@dataclass(frozen=True)
class BankResult:
    """What was written, so a launcher can log it rather than re-derive it."""

    status: str  # "banked" | "unchanged"
    out_dir: Path
    provenance_path: Path
    sha256sums_path: Path
    file_count: int
    replaced_path: Path | None = None
    stale_staging: tuple[Path, ...] = ()

    @property
    def shard_count(self) -> int:
        """Back-compatible alias; the staging kind calls the same number a file count."""
        return self.file_count


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


def _destination_is_a_real_directory(path: Path) -> bool:
    """Whether ``path`` is safe to read as this store's campaign directory.

    ``Path.is_dir`` follows symlinks. A matching artifact behind one must not be accepted as an
    unchanged artifact in this store, so links (including dangling ones) are always unsafe.
    """

    return not path.is_symlink() and (not path.exists() or path.is_dir())


def _destination_not_a_directory_refusal(path: Path) -> Refusal:
    return Refusal(
        DESTINATION_NOT_A_DIRECTORY,
        f"{path} is a link or is not a directory. A banked campaign IS a directory; a file "
        f"or link sitting at the campaign id's path is a refusal, not something to write "
        f"through -- and reaching `iterdir()` on it raised a bare OSError instead.",
    )


def _describe(value: Any) -> str:
    rendered = json.dumps(value, default=repr)
    return rendered if len(rendered) <= 120 else rendered[:117] + "..."


def _zero_or_empty(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return isinstance(value, (str, list, dict, tuple)) and len(value) == 0


def _normalized_key(key: Any) -> str:
    return str(key).strip().casefold()


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

    field = spec.name if where == "stamp" else f"{where}.{spec.name}"
    if spec.name not in container:
        return [Refusal(
            MISSING_REQUIRED_FIELD,
            f"{where}: '{spec.name}' is ABSENT. Artifact kind '{kind_name}' requires it "
            f"({spec.accepts}), because {spec.why}. Absent is not empty and not zero: supply "
            f"the value, or do not bank.",
            field,
        )]
    value = container[spec.name]
    if value is None:
        return [Refusal(
            NULL_REQUIRED_FIELD,
            f"{where}: '{spec.name}' is present but NULL. A null provenance field is a refusal, "
            f"not a recorded absence -- it reads downstream as 'this does not apply' when it "
            f"means 'nobody supplied it'. Required: {spec.accepts}, because {spec.why}.",
            field,
        )]
    if not spec.check(value):
        tail = ""
        if _zero_or_empty(value):
            tail = (
                " A zero or empty value is a VALUE, not an absence; it is refused on its own "
                "terms, with a different code from the absent case."
            )
        elif isinstance(value, str) and not _has_visible_character(value):
            tail = (
                " Every character of that value is zero-width or non-printing, so the field "
                "reads as present and displays as blank."
            )
        return [Refusal(
            INVALID_FIELD_VALUE,
            f"{where}: '{spec.name}' = {_describe(value)} is not {spec.accepts}. Required "
            f"because {spec.why}.{tail}",
            field,
        )]
    return []


def _forged_derived_keys(container: Mapping[str, Any], where: str) -> list[Refusal]:
    """Refuse a caller-supplied derived key however it is spelled, at any level."""

    reasons: list[Refusal] = []
    for key in container:
        normalized = _normalized_key(key)
        if normalized in DERIVED_KEYS:
            reasons.append(Refusal(
                DERIVED_FIELD_SUPPLIED,
                f"{where}: derived key {_describe(key)} (normalises to '{normalized}') is "
                f"supplied. This tool computes it, and the estimand caveat above all must not be "
                f"forgeable by the writer whose numbers it constrains -- a forged caveat banked "
                f"beside the derived one is two answers to what the numbers mean.",
                str(key) if where == "stamp" else f"{where}.{key}",
            ))
    return reasons


def _non_serializable(value: Any, path: str) -> list[tuple[str, str]]:
    """Find values that make ``PROVENANCE.json`` unreadable, and name where they are.

    ``NaN`` is the sharp case: Python writes it happily, ``JSON.parse`` then rejects the whole
    file, and ``jq`` silently yields ``null`` at exit 0 -- turning a number into exactly the
    "this does not apply" misreading this tool exists to prevent.
    """

    problems: list[tuple[str, str]] = []
    if value is None or isinstance(value, bool):
        return problems
    if isinstance(value, float) and not math.isfinite(value):
        return [(path, f"{value!r} has no JSON representation")]
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append((f"{path}.{key!r}", f"object key {key!r} is not a string"))
            problems.extend(_non_serializable(item, f"{path}.{key}"))
        return problems
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            problems.extend(_non_serializable(item, f"{path}[{index}]"))
        return problems
    if not isinstance(value, (str, int, float)):
        problems.append((path, f"a {type(value).__name__} has no JSON representation"))
    return problems


def _reached_through_symlink(path: Path, source_dir: Path) -> bool:
    """True if the file, or any directory between it and the source root, is a link."""

    if path.is_symlink():
        return True
    current = path.parent
    root = source_dir.resolve()
    while True:
        if current.is_symlink():
            return True
        if current.resolve() == root or current == current.parent:
            return False
        current = current.parent


def _validate_files(
    kind: ArtifactKind,
    records: Sequence[Any],
    source_dir: Path,
    *,
    allow_nonzero_shard_exit: bool,
    allow_empty_shard: bool,
) -> tuple[list[Refusal], list[dict[str, Any]]]:
    reasons: list[Refusal] = []
    normalized: list[dict[str, Any]] = []
    seen_declared: dict[str, int] = {}
    seen_inode: dict[tuple[int, int], int] = {}
    reserved = {name.casefold() for name in RESERVED_FILENAMES}
    source_root = source_dir.resolve()

    for index, record in enumerate(records):
        where = f"{kind.files_field}[{index}]"
        if not isinstance(record, Mapping):
            reasons.append(Refusal(
                SHARD_NOT_A_RECORD,
                f"{where} is {_describe(record)}, not a record. Every file carries its own "
                f"sha256 and provenance; a bare filename cannot.",
                where,
            ))
            continue

        field_reasons: list[Refusal] = []
        for spec in kind.file_required:
            field_reasons.extend(_check_field(record, spec, where=where, kind_name=kind.name))
        field_reasons.extend(_forged_derived_keys(record, where))
        if field_reasons:
            reasons.extend(field_reasons)
            continue

        relative = str(record["file"])
        if (
            os.path.isabs(relative)
            or "\\" in relative
            or relative.startswith("~")
            or any(part in ("..", "") for part in Path(relative).parts)
        ):
            reasons.append(Refusal(
                UNSAFE_SHARD_PATH,
                f"{where}: file {_describe(relative)} is not a relative path inside the source "
                f"directory. A banked file's name is part of the artifact; an absolute path or "
                f"a '..' component would bank something from outside the harvest.",
                f"{where}.file",
            ))
            continue

        if os.path.normpath(relative) != relative:
            # `results/./shard0.json` and `results//shard0.json` both survive the '..' scan
            # above, name the same bytes as `results/shard0.json`, and evade a duplicate check
            # that compares declared strings -- banking one file as two, which doubles N and
            # shrinks the apparent standard error. The declared name IS the banked name, so it
            # must arrive canonical rather than be normalised silently behind the caller.
            reasons.append(Refusal(
                NON_CANONICAL_SHARD_PATH,
                f"{where}: file {_describe(relative)} is not canonical; it names the same bytes "
                f"as {_describe(os.path.normpath(relative))}. The declared name is the banked "
                f"name and is what a duplicate check compares, so declare the canonical form.",
                f"{where}.file",
            ))
            continue

        if relative.casefold() in reserved:
            reasons.append(Refusal(
                RESERVED_SHARD_NAME,
                f"{where}: file {_describe(relative)} collides with a name this tool writes "
                f"({', '.join(RESERVED_FILENAMES)}). The stamp is written over the copied tree, "
                f"so banking it would destroy the file's bytes and leave the directory failing "
                f"its own SHA256SUMS. Rename it in the harvest first.",
                f"{where}.file",
            ))
            continue

        path = source_dir / relative
        if _reached_through_symlink(path, source_dir):
            reasons.append(Refusal(
                SYMLINKED_SHARD,
                f"{where}: file {_describe(relative)} is, or is reached through, a symlink. The "
                f"path check above refuses '..' because it would bank something from outside "
                f"the harvest; a link does exactly that while looking canonical.",
                f"{where}.file",
            ))
            continue

        if not path.is_file():
            reasons.append(Refusal(
                SHARD_FILE_ABSENT,
                f"{where}: declared file {_describe(relative)} does not exist under the source "
                f"directory ({source_dir}). A declared-but-absent file is exactly the partial "
                f"artifact this tool exists to refuse.",
                f"{where}.file",
            ))
            continue

        resolved = str(path.resolve())
        if not Path(resolved).is_relative_to(source_root):
            reasons.append(Refusal(
                SYMLINKED_SHARD,
                f"{where}: file {_describe(relative)} resolves to {resolved}, outside the "
                f"source directory ({source_root}).",
                f"{where}.file",
            ))
            continue

        # Two keys, because two records can name one file in two ways a single key misses:
        #   - the declared string, CASE-FOLDED, because a case-insensitive filesystem collides
        #     on `Results/shard0.json` while the raw strings differ;
        #   - the (device, inode) pair, because a HARDLINK gives one file two unrelated names
        #     that no amount of path comparison can connect -- `Path.resolve()` cannot see it,
        #     the copy duplicates the content, and both SHA256SUMS lines verify.
        # Both spellings inflate the declared count over the distinct evidence actually banked.
        stat = path.stat()
        duplicate_of: int | None = None
        duplicate_why = ""
        for key, table, why in (
            (relative.casefold(), seen_declared, "the same declared name"),
            ((stat.st_dev, stat.st_ino), seen_inode, "the same file on disk (hardlink)"),
        ):
            if key in table:
                duplicate_of, duplicate_why = table[key], why
                break
        if duplicate_of is not None:
            reasons.append(Refusal(
                DUPLICATE_SHARD_FILE,
                f"{where}: file {_describe(relative)} is {duplicate_why} as "
                f"{kind.files_field}[{duplicate_of}]. Two records for one file inflate the "
                f"declared count while banking one copy -- a pooled analysis then counts it "
                f"twice, shrinking the apparent standard error in the flattering direction, and "
                f"`sha256sum -c` passes both lines so nothing downstream catches it.",
                f"{where}.file",
            ))
            continue
        seen_declared[relative.casefold()] = index
        seen_inode[(stat.st_dev, stat.st_ino)] = index

        declared = str(record["sha256"]).lower()
        actual = sha256_file(path)
        size = stat.st_size
        if actual != declared:
            reasons.append(Refusal(
                SHARD_SHA256_MISMATCH,
                f"{where}: declared sha256 {declared} for {_describe(relative)} but the file on "
                f"disk hashes to {actual}. Either the file changed after it was measured or the "
                f"stamp describes a different run; both make every citation of it false.",
                f"{where}.sha256",
            ))
            continue

        declared_size = record.get("bytes")
        if declared_size is not None and declared_size != size:
            reasons.append(Refusal(
                SHARD_BYTES_MISMATCH,
                f"{where}: declared bytes {_describe(declared_size)} for {_describe(relative)} "
                f"but the file on disk is {size} bytes. A declared size the tool does not verify "
                f"is a number nobody checked.",
                f"{where}.bytes",
            ))
            continue

        if size == 0 and not allow_empty_shard:
            reasons.append(Refusal(
                EMPTY_SHARD_FILE,
                f"{where}: file {_describe(relative)} is ZERO bytes, and its sha256 is honestly "
                f"the hash of nothing, so every other guard reads True. A truncated result is "
                f"the 'absent rather than wrong' case in its purest form: pass "
                f"allow_empty_shard=True (--allow-empty-shard) if the emptiness is the finding.",
                f"{where}.file",
            ))
            continue

        if kind.records_exit_status():
            exit_status = int(record["exit_status"])
            if exit_status != 0 and not allow_nonzero_shard_exit:
                reasons.append(Refusal(
                    UNACKNOWLEDGED_SHARD_FAILURE,
                    f"{where}: file {_describe(relative)} exited {exit_status}. Banking a failed "
                    f"shard is allowed but never silent: pass allow_nonzero_shard_exit=True "
                    f"(--allow-nonzero-shard-exit) so the stamp records that the failure was "
                    f"seen, or drop the shard.",
                    f"{where}.exit_status",
                ))
                continue

        entry = dict(record)
        entry["file"] = relative
        entry["sha256"] = actual
        entry["bytes"] = size
        normalized.append(entry)

    return reasons, normalized


def _validate(
    stamp: Mapping[str, Any],
    kind: ArtifactKind,
    source_dir: Path,
    *,
    out_dir: Path,
    allow_nonzero_shard_exit: bool,
    allow_empty_shard: bool,
    verify_checkpoint: Path | None,
) -> tuple[list[Refusal], list[dict[str, Any]]]:
    reasons: list[Refusal] = []

    reasons.extend(_forged_derived_keys(stamp, "stamp"))
    for key, value in stamp.items():
        if _normalized_key(key) == SCHEMA_KEY and value != kind.schema:
            reasons.append(Refusal(
                SCHEMA_MISMATCH,
                f"stamp: {_describe(key)} is {_describe(value)} but artifact kind '{kind.name}' "
                f"writes {_describe(kind.schema)}. A stamp that agrees with the tool is fine and "
                f"is rewritten; one that disagrees would put two shapes behind one schema "
                f"string, which is exactly what makes a reader unable to tell them apart.",
                str(key),
            ))

    for path, why in _non_serializable(stamp, "stamp"):
        reasons.append(Refusal(
            UNSERIALIZABLE_PROVENANCE,
            f"{path}: {why}. PROVENANCE.json must be readable by tools that are not Python: a "
            f"NaN makes JSON.parse reject the whole file, and `jq` turns it into `null` at exit "
            f"0 -- a number silently becoming the absence this tool exists to distinguish.",
            path.removeprefix("stamp."),
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
                names[0],
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
                "rollout_policy",
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
                "rollout_threads_cpu_budget_ack",
            ))

    normalized: list[dict[str, Any]] = []
    if not source_dir.is_dir():
        reasons.append(Refusal(
            SOURCE_DIR_ABSENT,
            f"source directory {source_dir} does not exist or is not a directory; there is "
            f"nothing to bank from.",
        ))
    else:
        records = stamp.get(kind.files_field)
        if isinstance(records, list) and records:
            file_reasons, normalized = _validate_files(
                kind, records, source_dir,
                allow_nonzero_shard_exit=allow_nonzero_shard_exit,
                allow_empty_shard=allow_empty_shard,
            )
            reasons.extend(file_reasons)
        elif isinstance(records, list) and not records:
            # An absent or null list is already reported by the field check above; only the
            # EMPTY case needs its own voice, because an empty list is a VALUE that would
            # otherwise sail through as a zero-shard bank.
            reasons.append(Refusal(
                EMPTY_SHARD_LIST,
                f"stamp: '{kind.files_field}' is an EMPTY list. Zero files is not a small bank, "
                f"it is no bank: a directory with a provenance stamp and no evidence reads as "
                f"banked and cites nothing. This is refused separately from "
                f"'{kind.files_field}' being absent.",
                kind.files_field,
            ))

    campaign_id = stamp.get("campaign_id")
    if _is_campaign_id(campaign_id) and out_dir.name != campaign_id:
        reasons.append(Refusal(
            CAMPAIGN_ID_DIRECTORY_MISMATCH,
            f"campaign_id {_describe(campaign_id)} does not match the destination directory "
            f"name {_describe(out_dir.name)}. The store is addressed by campaign id; a campaign "
            f"banked under another name is the scratch-directory loss with extra steps.",
            "campaign_id",
        ))

    if not _destination_is_a_real_directory(out_dir):
        reasons.append(_destination_not_a_directory_refusal(out_dir))

    if verify_checkpoint is not None:
        declared = stamp.get("checkpoint_sha256")
        if not verify_checkpoint.is_file():
            reasons.append(Refusal(
                CHECKPOINT_SHA256_MISMATCH,
                f"--verify-checkpoint {verify_checkpoint} is not a file, so the declared "
                f"checkpoint sha256 could not be re-derived. An unverifiable verification is "
                f"not a pass.",
                "checkpoint_sha256",
            ))
        elif _is_sha256(declared):
            actual = sha256_file(verify_checkpoint)
            if actual != str(declared).lower():
                reasons.append(Refusal(
                    CHECKPOINT_SHA256_MISMATCH,
                    f"declared checkpoint_sha256 {str(declared).lower()} but "
                    f"{verify_checkpoint} hashes to {actual}. The stamp describes different "
                    f"weights than the ones presented.",
                    "checkpoint_sha256",
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
    return {
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


def build_provenance(
    stamp: Mapping[str, Any],
    kind: ArtifactKind,
    files: Sequence[Mapping[str, Any]],
    *,
    banked_at_utc: str,
    allow_nonzero_shard_exit: bool,
) -> dict[str, Any]:
    """The stamp as it will be written. Pure, so the destination check can compare against it."""

    provenance: dict[str, Any] = {
        "schema": kind.schema,
        "artifact_kind": kind.name,
        "banked_at_utc": banked_at_utc,
        "banked_by": f"{TOOL_NAME} v{TOOL_VERSION}",
    }
    for key, value in stamp.items():
        if key in (kind.files_field, "artifact_kind") or _normalized_key(key) == SCHEMA_KEY:
            continue
        provenance[key] = value
    provenance[kind.count_field] = len(files)
    provenance[kind.files_field] = [dict(record) for record in files]
    # The {file, sha256, bytes} projection the store's existing HARVEST.json readers consume.
    provenance["per_file"] = [
        {"file": record["file"], "sha256": record["sha256"], "bytes": record["bytes"]}
        for record in files
    ]
    if kind.records_exit_status():
        provenance["nonzero_exit_shards_acknowledged"] = (
            sorted(str(r["file"]) for r in files if int(r["exit_status"]) != 0)
            if allow_nonzero_shard_exit
            else []
        )
    if kind.derives_estimand:
        provenance["estimand"] = _estimand_block(stamp)
    return provenance


def render_sha256sums(files: Sequence[Mapping[str, Any]]) -> str:
    """``sha256sum`` output shape: two spaces, paths relative to the banked directory.

    Sorted by PATH, not by digest, matching the store's existing ``SHA256SUMS`` files -- a diff
    between two banks of the same campaign should show the file that changed, not a reshuffle.
    """

    ordered = sorted(files, key=lambda record: str(record["file"]))
    return "\n".join(f"{record['sha256']}  {record['file']}" for record in ordered) + "\n"


def _provenance_json(provenance: Mapping[str, Any]) -> str:
    # allow_nan=False: a NaN written here is valid Python and invalid JSON, and every non-Python
    # reader either rejects the whole file or silently reads `null`. Validation already names the
    # offending key; this makes the guarantee structural instead of dependent on that scan.
    return json.dumps(provenance, indent=2, sort_keys=False, allow_nan=False) + "\n"


def _comparable(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The stamp with the fields that legitimately differ between two identical banks removed."""

    return {key: value for key, value in provenance.items() if key not in ("banked_at_utc",)}


_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _open_directory_no_follow(path: Path, *, parent_fd: int | None = None) -> int:
    """Open a real directory without ever resolving a symlink in its final component."""

    if not _O_NOFOLLOW:
        raise OSError("this platform has no O_NOFOLLOW; cannot safely inspect a bank directory")
    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
    fd = os.open(str(path) if parent_fd is None else path.name, flags, dir_fd=parent_fd)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise NotADirectoryError(f"{path} is not a directory")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_relative_file_no_follow(directory_fd: int, relative: str) -> int:
    """Open a declared regular file, refusing symlinks in every path component."""

    parts = tuple(Path(relative).parts)
    if not parts or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise OSError(f"unsafe relative artifact path {relative!r}")
    current_fd = os.dup(directory_fd)
    try:
        for part in parts[:-1]:
            next_fd = _open_directory_no_follow(Path(part), parent_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | _O_NOFOLLOW, dir_fd=current_fd)
    finally:
        os.close(current_fd)
    if not stat.S_ISREG(os.fstat(file_fd).st_mode):
        os.close(file_fd)
        raise OSError(f"{relative!r} is not a regular file")
    return file_fd


def _read_relative_file_no_follow(directory_fd: int, relative: str) -> bytes:
    file_fd = _open_relative_file_no_follow(directory_fd, relative)
    try:
        handle = os.fdopen(file_fd, "rb")
    except BaseException:
        try:
            os.close(file_fd)
        except OSError:
            pass
        raise
    with handle:
        return handle.read()


def _relative_regular_files_no_follow(directory_fd: int, *, prefix: str = "") -> set[str]:
    """List the artifact tree from directory descriptors, never traversing a link."""

    files: set[str] = set()
    for name in os.listdir(directory_fd):
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        relative = f"{prefix}{name}"
        if stat.S_ISREG(info.st_mode):
            files.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            child_fd = _open_directory_no_follow(Path(name), parent_fd=directory_fd)
            try:
                files.update(_relative_regular_files_no_follow(child_fd, prefix=f"{relative}/"))
            finally:
                os.close(child_fd)
        else:
            # A link, device or FIFO is not part of a banked artifact. Returning a marker makes
            # the exact-file-set comparison fail without ever dereferencing it.
            files.add(f"<non-regular:{relative}>")
    return files


def _sha256_relative_file_no_follow(directory_fd: int, relative: str) -> str:
    digest = hashlib.sha256()
    file_fd = _open_relative_file_no_follow(directory_fd, relative)
    try:
        handle = os.fdopen(file_fd, "rb")
    except BaseException:
        try:
            os.close(file_fd)
        except OSError:
            pass
        raise
    with handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_still_names_directory(path: Path, directory_fd: int) -> bool:
    """Reject a swap observed after the descriptor-bound artifact check finished."""

    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(current.st_mode) and os.path.samestat(current, os.fstat(directory_fd))


def _existing_matches(
    out_dir: Path, provenance: Mapping[str, Any], sha256sums: str, files_field: str
) -> bool:
    """True only if the descriptor-bound destination is byte-for-byte what we would write.

    A no-op is evidence: it must not read a matching artifact through a symlink that a concurrent
    writer inserted after validation. The directory is therefore opened with ``O_NOFOLLOW`` and
    every descendant is read relative to that descriptor with the same rule. If the path changes
    while those reads run, we either keep reading the original opened directory or observe the
    replacement before returning; we never dereference the replacement.
    """

    try:
        directory_fd = _open_directory_no_follow(out_dir)
    except OSError:
        return False
    try:
        try:
            existing = json.loads(
                _read_relative_file_no_follow(directory_fd, PROVENANCE_FILENAME).decode("utf-8")
            )
            actual_sums = _read_relative_file_no_follow(directory_fd, SHA256SUMS_FILENAME).decode(
                "utf-8"
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(existing, Mapping) or _comparable(existing) != _comparable(provenance):
            return False
        if actual_sums != sha256sums:
            return False
        declared = {str(record["file"]) for record in provenance[files_field]}
        if _relative_regular_files_no_follow(directory_fd) != declared | {
            PROVENANCE_FILENAME, SHA256SUMS_FILENAME,
        }:
            return False
        for record in provenance[files_field]:
            if _sha256_relative_file_no_follow(directory_fd, str(record["file"])) != record["sha256"]:
                return False
        return _path_still_names_directory(out_dir, directory_fd)
    except OSError:
        return False
    finally:
        os.close(directory_fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems refuse fsync on directories. The rename below is still ordered.
        pass
    finally:
        os.close(fd)


def _verify_staged_tree(staging: Path, files: Sequence[Mapping[str, Any]]) -> None:
    """Re-hash what was actually written, immediately before the rename.

    The declared sha256 was verified against the SOURCE, and a source directory is not frozen: a
    straggler shard still flushing, or a harvest copy still running, changes the bytes between
    the hash and the copy. Verifying the STAGED bytes makes "this directory verifies against its
    own SHA256SUMS" an invariant of the write instead of an assumption about the source.
    """

    for record in files:
        staged = staging / str(record["file"])
        actual = sha256_file(staged)
        size = staged.stat().st_size
        if actual != record["sha256"] or size != record["bytes"]:
            raise BankRefusal([Refusal(
                STAGED_COPY_MISMATCH,
                f"the staged copy of {record['file']} hashes to {actual} ({size} bytes) but the "
                f"stamp records {record['sha256']} ({record['bytes']} bytes). The source changed "
                f"between being hashed and being copied, so this artifact would have failed its "
                f"own SHA256SUMS. Nothing was banked; re-run once the source has settled.",
                str(record["file"]),
            )])


def _write_atomically(
    out_dir: Path,
    source_dir: Path,
    files: Sequence[Mapping[str, Any]],
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
    staging = Path(tempfile.mkdtemp(prefix=f"{STAGING_PREFIX}{out_dir.name}-", dir=out_dir.parent))
    replaced: Path | None = None
    installed = False
    try:
        for record in files:
            destination = staging / str(record["file"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(source_dir / str(record["file"]), destination)
        _verify_staged_tree(staging, files)
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
        os.rename(staging, out_dir)
        installed = True
    except BaseException:
        # BaseException, not OSError. The displacement window is exactly where a
        # KeyboardInterrupt used to leave the store holding only `<campaign-id>.replaced-<ts>`
        # and nothing at the campaign id -- the scratch-directory loss with a timestamp suffix.
        # Ctrl-C is the likeliest interruption of a long copy, so it must roll back like any
        # other failure.
        shutil.rmtree(staging, ignore_errors=True)
        if replaced is not None and not installed:
            try:
                os.rename(replaced, out_dir)
            except OSError:  # pragma: no cover - nothing better is available at this point
                pass
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
            "artifact_kind",
        )])
    return ARTIFACT_KINDS[name]


def stale_staging_directories(out_dir: Path | str) -> tuple[Path, ...]:
    """Staging directories a previous run was killed before it could clean up.

    SIGKILL cannot be caught, so `.bank-tmp-*` debris in the store is possible. It is REPORTED
    rather than removed: a partial artifact with a dot in front of it is still somebody's bytes,
    and this tool does not delete evidence.
    """

    parent = Path(out_dir).parent
    if not parent.is_dir():
        return ()
    return tuple(sorted(
        entry for entry in parent.iterdir()
        if entry.is_dir() and entry.name.startswith(STAGING_PREFIX)
    ))


def bank_artifact(
    stamp: Mapping[str, Any],
    source_dir: Path | str,
    out_dir: Path | str,
    *,
    kind: str | None = None,
    overwrite: bool = False,
    allow_nonzero_shard_exit: bool = False,
    allow_empty_shard: bool = False,
    verify_checkpoint: Path | str | None = None,
    banked_at_utc: str | None = None,
) -> BankResult:
    """Bank ``stamp``'s files from ``source_dir`` into ``out_dir``, or raise :class:`BankRefusal`.

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
            "artifact_kind",
        )])
    artifact_kind = resolve_kind(kind if kind is not None else declared_kind)

    source = Path(source_dir)
    destination = Path(out_dir)
    checkpoint = Path(verify_checkpoint) if verify_checkpoint is not None else None

    reasons, files = _validate(
        stamp, artifact_kind, source,
        out_dir=destination,
        allow_nonzero_shard_exit=allow_nonzero_shard_exit,
        allow_empty_shard=allow_empty_shard,
        verify_checkpoint=checkpoint,
    )

    stale = stale_staging_directories(destination)
    provenance: dict[str, Any] | None = None
    sha256sums = ""
    if not reasons:
        provenance = build_provenance(
            stamp, artifact_kind, files,
            banked_at_utc=banked_at_utc or datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            allow_nonzero_shard_exit=allow_nonzero_shard_exit,
        )
        sha256sums = render_sha256sums(files)

        # Re-check after provenance construction. `_validate` checked the address before any
        # work, but this branch is about to read it again and another writer may have replaced
        # it with a symlink meanwhile.
        if not _destination_is_a_real_directory(destination):
            reasons.append(_destination_not_a_directory_refusal(destination))
        elif destination.is_dir() and any(destination.iterdir()):
            if _existing_matches(
                destination, provenance, sha256sums, artifact_kind.files_field
            ):
                # `_existing_matches` checks before it reads, but its final file hash can still
                # overlap a replacement. Re-check immediately before certifying this address as
                # an unchanged artifact.
                if not _destination_is_a_real_directory(destination):
                    reasons.append(_destination_not_a_directory_refusal(destination))
                else:
                    return BankResult(
                        status="unchanged",
                        out_dir=destination,
                        provenance_path=destination / PROVENANCE_FILENAME,
                        sha256sums_path=destination / SHA256SUMS_FILENAME,
                        file_count=len(files),
                        stale_staging=stale,
                    )
            if not overwrite and not reasons:
                if not _destination_is_a_real_directory(destination):
                    reasons.append(_destination_not_a_directory_refusal(destination))
                else:
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
            destination, source, files, provenance, sha256sums, overwrite=overwrite
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
        file_count=len(files),
        replaced_path=replaced,
        stale_staging=stale,
    )


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def describe_kind(kind: ArtifactKind) -> str:
    lines = [
        f"{kind.name} -- {kind.description}",
        f"  schema written: {kind.schema}",
        "",
        "REQUIRED stamp fields:",
    ]
    for spec in kind.required:
        lines.append(f"  {spec.name}: {spec.accepts}")
        lines.append(f"      why: {spec.why}")
    for group in kind.required_one_of:
        lines.append(f"  at least one of: {' OR '.join(spec.name for spec in group)}")
        for spec in group:
            lines.append(f"      {spec.name}: {spec.accepts}")
    lines.append("")
    lines.append(f"REQUIRED per-file fields (every entry of `{kind.files_field}`):")
    for spec in kind.file_required:
        lines.append(f"  {spec.name}: {spec.accepts}")
    lines.append("")
    lines.append(f"DERIVED by the tool, and refused if supplied: {sorted(DERIVED_KEYS)}")
    lines.append(f"VERIFIED against disk if supplied per file: {sorted(VERIFIED_FILE_KEYS)}")
    if kind.derives_estimand:
        lines.append(
            f"ESTIMAND caveat is derived from rollout_policy; declared policies: "
            f"{sorted(ROLLOUT_POLICY_ESTIMANDS)}"
        )
    return "\n".join(lines)


def _load_stamp_json(raw: str) -> Any:
    """Parse a stamp, refusing duplicate keys instead of silently keeping the last one.

    ``{"rollouts": 0, "rollouts": 64}`` parses to 64 by default. A stamp that answers a required
    field twice is the same "two answers is no answer" defect ARTIFACT_KIND_MISMATCH refuses, and
    the discarded answer is invisible.
    """

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                raise BankRefusal([Refusal(
                    DUPLICATE_STAMP_KEY,
                    f"the stamp declares {_describe(key)} more than once. JSON parsing keeps the "
                    f"last value silently, so one of the two answers would vanish without anyone "
                    f"seeing which.",
                    key,
                )])
            seen[key] = value
        return seen

    return json.loads(raw, object_pairs_hook=no_duplicates)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bank a campaign's files into the campaign store with a complete provenance "
            "stamp, or refuse and write nothing."
        ),
        epilog=(
            f"exit codes: 0 = banked or unchanged; {REFUSAL_EXIT} = REFUSED, with every unmet "
            f"requirement named on stderr and nothing written; 2 = bad invocation. Branch on "
            f"{REFUSAL_EXIT}. Run --print-required-fields KIND to see what a kind requires "
            f"before running the cell that has to satisfy it."
        ),
    )
    parser.add_argument("--stamp", help="path to the provenance stamp JSON, or '-' for stdin.")
    parser.add_argument("--source-dir", help="directory the declared files live in.")
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
        "--allow-empty-shard", action="store_true",
        help="bank zero-byte files (only when the emptiness is itself the finding).",
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
        stamp = _load_stamp_json(raw)
    except json.JSONDecodeError as error:
        print(
            Refusal(STAMP_NOT_JSON, f"{args.stamp} is not valid JSON ({error})."),
            file=sys.stderr,
        )
        return REFUSAL_EXIT
    except BankRefusal as refusal:
        print(refusal, file=sys.stderr)
        return REFUSAL_EXIT

    try:
        result = bank_artifact(
            stamp,
            args.source_dir,
            args.out_dir,
            kind=args.kind,
            overwrite=args.overwrite,
            allow_nonzero_shard_exit=args.allow_nonzero_shard_exit,
            allow_empty_shard=args.allow_empty_shard,
            verify_checkpoint=args.verify_checkpoint,
        )
    except BankRefusal as refusal:
        print(refusal, file=sys.stderr)
        return REFUSAL_EXIT

    if result.status == "unchanged":
        print(f"unchanged: {result.out_dir} already holds this exact artifact "
              f"({result.file_count} files); nothing was rewritten.")
    else:
        print(f"banked: {result.file_count} files -> {result.out_dir}")
        print(f"  {result.provenance_path}")
        print(f"  {result.sha256sums_path}")
        if result.replaced_path is not None:
            print(f"  displaced artifact moved to {result.replaced_path}")
    for path in result.stale_staging:
        print(f"  NOTE: stale staging directory from an earlier killed run: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
