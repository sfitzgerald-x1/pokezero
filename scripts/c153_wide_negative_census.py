#!/usr/bin/env python
"""C153: re-measure every window-scoped negative claim OUTSIDE the two permitted windows.

WHY THIS EXISTS. `reports/c138_known_gaps_ledger.md` §8 gained a standing rule in
#1200 (C152):

  > A negative measured only inside the two permitted windows is a claim about those
  > windows. Widening the CORPUS cannot find this class of error; only widening the
  > MEASUREMENT can.

The rule landed with no instrument behind it. `tests/test_never_fired_counter_census.py`
re-derives every absence over the committed CORPUS on every run -- which is exactly the
check the rule says cannot find this class of error, and C152 says so in the same
sentence. Nothing re-measured the affected negatives anywhere new, so §3.5's inventory
was still asserted at a scope it had never been measured at.

THE EXPOSURE IS MEASURED, NOT SUPPOSED. Of the 388 committed JSON under `reports/` and
`docs/` that `counter_artifacts()` selected at `7fcd9e19`, 103 carry a top-level seed
span; 83 of those are the two permitted windows, 1 is the burned final-holdout block, 4
are C152's own wide census, and 15 are pre-fingerprint c6-c13 / c26-c27 artifacts on
obsolete engines. So excluding C152's four, EVERY sweep on an engine near the current one
is inside the two windows this program has iterated against for its entire history.

WHAT THIS SCRIPT IS. One bounded measurement, not a programme: a single wide-seed census
that gives every window-scoped negative in §3.5 -- and the row-level ones in G33b, G33c,
G50, H13, H14 and H15 -- a verdict at a NEW scope, stated in the verdict.

Four verdicts, and the last two are deliberately not measurement results:

  * ``FIRED``                  -- a nonzero counter in a committed census shard, with the
                                  shard, the counter key, the value, and the SEEDS, taken
                                  from the per-game checkpoint records and closed against
                                  the shard total.
  * ``NOT_OBSERVED_AT_SCOPE``  -- zero across the whole census, with the scope written
                                  into the sentence. Never a bare "never fired".
  * ``UNREACHABLE_STRUCTURAL`` -- the code cannot emit it on any input, with the
                                  demonstration carried on the entry.
  * ``UNREACHABLE_POOL``       -- the gen3 randbats pool cannot produce the trigger, per
                                  the ledger's §4 rows R1/R7/R8.

THE INVENTORY IS DERIVED, NOT TRANSCRIBED. Every taxonomy comes out of source by AST:

  * the 40 ``EngineWorldUnsupported`` reasons from ``src/pokezero/engine_world.py``;
  * the 19 ``classify_divergence`` return classes and the 8 ``UnmappableChoice`` reasons
    from ``scripts/engine_transition_differential.py``;
  * the whole COUNTER KEY SPACE -- every ``counts[...] += 1`` literal key and f-string
    prefix in the differential -- so a §3.5 name that no longer corresponds to a key the
    harness can emit is a loud failure rather than a silently unmeasurable row.

Regenerate with::

    PYTHONPATH=src python scripts/c153_wide_negative_census.py \\
        --strict-shard reports/artifacts/c153_wide_census_*_sweep.json \\
        --banded-shard reports/artifacts/c153_banded_census_*_sweep.json \\
        --checkpoint-dir <dir with the per-game JSONL> \\
        --write reports/artifacts/c153_wide_negative_census.json

The checkpoints are per-game JSONL from ``--checkpoint``; they are NOT committed (10,000
records) and are only used to attribute a firing to a seed. The attribution is closed
against the committed shard totals in the artifact itself and in the pin, so a lost
checkpoint cannot make a claim unfalsifiable -- it makes the closure fail.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DIFFERENTIAL = REPO_ROOT / "scripts/engine_transition_differential.py"
ENGINE_WORLD = REPO_ROOT / "src/pokezero/engine_world.py"

# ---------------------------------------------------------------------------
# Taxonomies, by AST.
# ---------------------------------------------------------------------------


def world_unsupported_reasons() -> set[str]:
    """Every literal first argument to a ``raise EngineWorldUnsupported(...)``."""

    reasons: set[str] = set()
    for node in ast.walk(ast.parse(ENGINE_WORLD.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        if (getattr(func, "id", None) or getattr(func, "attr", None)) != "EngineWorldUnsupported":
            continue
        if node.exc.args and isinstance(node.exc.args[0], ast.Constant):
            value = node.exc.args[0].value
            if isinstance(value, str):
                reasons.add(value)
    return reasons


def divergence_classes() -> set[str]:
    """Every static class ``classify_divergence`` can return.

    A ``"prefix:" + payload`` return contributes its literal prefix, which is what
    appears in the counter key ahead of the dynamic component list. Identical to the
    derivation in ``tests/test_never_fired_counter_census.py``, deliberately: two
    derivations that disagreed would be a third thing to adjudicate.
    """

    tree = ast.parse(DIFFERENTIAL.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "classify_divergence"
    )
    classes: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            classes.add(value.value)
        elif isinstance(value, ast.BinOp) and isinstance(value.left, ast.Constant):
            classes.add(value.left.value.rstrip(":").split(":%s")[0])
    return classes


def unmappable_choice_reasons() -> set[str]:
    """Every ``raise UnmappableChoice(...)`` reason, literal or f-string prefix.

    The two interpolated ones (``move_not_in_engine_set:{id}`` and
    ``unknown_kind:{kind}``) contribute their literal prefix, because that is what a
    counter key starts with.
    """

    out: set[str] = set()
    for node in ast.walk(ast.parse(DIFFERENTIAL.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        if (getattr(func, "id", None) or getattr(func, "attr", None)) != "UnmappableChoice":
            continue
        if not node.exc.args:
            continue
        arg = node.exc.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.add(arg.value)
        elif isinstance(arg, ast.JoinedStr) and isinstance(arg.values[0], ast.Constant):
            out.add(arg.values[0].value)
    return out


def counter_key_space() -> tuple[frozenset[str], frozenset[str]]:
    """``(exact keys, f-string prefixes)`` the differential's ``counts`` can ever carry.

    Derived so that a §3.5 name which no longer corresponds to an emittable key is a LOUD
    failure. An inventory entry the harness cannot emit is not a negative that survived a
    census; it is a row measuring nothing, and this repo has shipped eight of those.
    """

    exact: set[str] = set()
    prefixes: set[str] = set()
    for node in ast.walk(ast.parse(DIFFERENTIAL.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.AugAssign) or not isinstance(node.target, ast.Subscript):
            continue
        base = node.target.value
        if (getattr(base, "id", None) or getattr(base, "attr", None)) not in {
            "counts",
            "totals",
        }:
            continue
        key = node.target.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            exact.add(key.value)
        elif isinstance(key, ast.JoinedStr) and isinstance(key.values[0], ast.Constant):
            prefixes.add(key.values[0].value)
    return frozenset(exact), frozenset(prefixes)


# ---------------------------------------------------------------------------
# The inventory: §3.5's four verified-negative lists, assembled from the taxonomies.
# ---------------------------------------------------------------------------

# The 10 of 40 `EngineWorldUnsupported` reasons with nonzero recorded evidence in the
# committed corpus, per `tests/test_never_fired_counter_census.py`. Held here as the
# COMPLEMENT operator that turns the AST-derived 40 into §3.5's list, not as a
# transcription of the list itself -- so a reason added to `engine_world.py` joins the
# inventory automatically rather than going unmeasured.
FIRED_IN_CORPUS_WORLD_UNSUPPORTED = frozenset(
    {
        "encore_move_unknown",
        "materialization_blocker",
        "payload_malformed",
        "pending_baton_pass",
        "self_moveset_mismatch",
        "self_request_state_unsupported",
        "status_unsupported",
        "substitute_health_unknown",
        "transform_unexpressible",
        "volatile_unsupported",
    }
)

# §3.5 excludes `future_sight_pending` from its list of 33 (and therefore from the 29)
# and retires it under R1 as UNREACHABLE in this pool: `futuresight` and `doomdesire` are
# each 0 of 220 species. It is carried here anyway, with an UNREACHABLE verdict, because
# an inventory that silently drops a name is how a "closed" row turns out to be a fourth
# category in disguise.
POOL_UNREACHABLE_WORLD_UNSUPPORTED = frozenset(
    {
        "future_sight_pending",  # R1: futuresight/doomdesire 0 of 220 species
        "nature_not_neutral",  # R7: natures unset on 24,000/24,000 generated Pokemon
        "weather_unsupported",  # R8: all four gen3 weathers are in `_WEATHER_IDS`
    }
)

# The two `divergence_class` values §3.5 counts among its static counters, both
# STRUCTURALLY unreachable rather than merely unobserved. Each carries the demonstration
# that makes it structural, so a reader never has to take the word "structural" on trust.
STRUCTURAL_DIVERGENCE_CLASSES = {
    "mapper_lossy": (
        "`evaluate_boundary_strict` returns the verdict `skip_lossy` with the trigger "
        "body `every branch rendered lossy`, and the run loop `continue`s at the "
        "`verdict == \"skip_lossy\"` branch BEFORE the `divergence_class:` line, which "
        "runs only under `verdict == \"diverged\"`. The classifier can therefore never "
        "be handed the body that would return this class."
    ),
    "no_usable_branch": (
        "The trigger body `mapper produced no usable branch` is produced by NO CODE in the "
        "repository: the only occurrence on any execution path is the classifier's own "
        "test of it at `engine_transition_differential.py:1915`, so no input can make "
        "`classify_divergence` return this class. Stated as 'no producer' rather than "
        "'nowhere in the repository', which a first draft wrote and its own commit "
        "falsified -- the phrase now also appears in this docstring and in the census "
        "artifact that records it. A grep-shaped claim has to be scoped to executable "
        "code or it is false the moment it is written down."
    ),
}

# H15's tripartite split of the twelve never-fired `divergence_class` values, held as the
# CLAIM UNDER TEST rather than as a fact. The census adjudicates it; see the report.
#
# H15 says these four are "reachable only through the `--matcher banded` path, which no
# committed artifact used". That is a scope statement, and the banded arm of this census
# is the measurement that discharges it.
H15_CLAIMED_BANDED_REACHABLE_ONLY = frozenset(
    {"boost_delta_support", "status_support", "faint_boundary", "damage_band"}
)

# And H15 says these six are "strict-path classes the program has simply never produced".
H15_CLAIMED_NEVER_PRODUCED = frozenset(
    {
        "component_set_equal_but_unmatched",
        "evidence:crit_in_step",
        "evidence:faint_ply_no_upkeep",
        "evidence:spikes_in_step",
        "no_miss_recorded",
        "unclassified",
    }
)

# The eighth `unmappable_choice` reason: the one that FIRES, and therefore the control
# that makes the other seven mean something. A scanner finding nothing would otherwise
# "verify" all eight.
FIRED_UNMAPPABLE_CHOICE = frozenset({"struggle_not_submittable"})

# §3.5's six never-fired DYNAMIC families, by counter-key prefix. Held explicitly because
# a prefix is a claim about a family, not about a name: the AST gives every emittable
# prefix and this selects the six §3.5 names among them.
DYNAMIC_FAMILY_PREFIXES = (
    "skip:no_materialization:",
    "skip:world_error:",
    "strict:branch_events_error:",
    "engine_error:",
    "engine_error_choice:",
    "world_prestate_mismatch:weather_",
)

# `skip:world_error:no_constructible_candidate` is a STATIC §3.5 counter that also sits
# under the `skip:world_error:` family prefix. Counting a hit on it as a hit for the
# family would let one firing close two inventory entries, which is the arithmetic that
# produced "10 listed, 9 measured" in this document's own history.
DYNAMIC_FAMILY_EXCLUSIONS = {"skip:world_error:": {"skip:world_error:no_constructible_candidate"}}

# WHAT THIS CENSUS CANNOT REACH, and why -- recorded per entry rather than left to be
# inferred from a zero. A zero produced by an instrument that could never have produced a
# one is not the same measurement as a zero produced by an instrument that could, and
# collapsing the two is how a "closed" row turned out to be a fourth category in disguise.
#
# ⚠ EVERY ENTRY BELOW IS TRACED FROM THE RAISE SITE TO THE DIFFERENTIAL'S ACTUAL CALL,
# not to a plausible sentence about it. That rule was added after review, because the
# first revision of this map got one entry outright wrong and two more imprecise:
#
#   * `public_effect_blocked` was filed here on the claim "the differential declares
#     none". FALSE. `engine_transition_differential.py:2662` passes `blocked_slots=blocked`
#     and `blocked` comes from the PRODUCTION `EngineMctsPolicy._public_effect_signals`
#     (:2624) on a live observation, which populates it on two ordinary data-dependent
#     branches -- `engine_search.py:2391` (item mutated with no protocol-confirmed current
#     item) and `:2409` (active transformed into an unnamed species). The metadata early
#     return at `:2363` is demonstrably not always taken here: the same scan's
#     `transformed` output drove SIX `transform_unexpressible` firings in this census. It
#     is REACHABLE and merely not observed, and has been moved out.
#   * `deferred_opponent_action` said the payload "never carries" those fields. It always
#     carries them; they are always EMPTY, which is a different fact and the one that
#     matters.
#   * `rest_sleep_refund_pending_precounts_legacy` said live rows always carry the
#     counts. One live branch does not -- and sets a flag `engine_world` tests first,
#     which is what actually closes the path.
#
# A category that says "the instrument cannot reach this" is doing the same work as a
# never-fired claim, and it earns the same standard of evidence.
CENSUS_CANNOT_REACH = {
    "skip:world_unsupported:rest_sleep_refund_pending_precounts_legacy": (
        "Raised at `engine_world.py:1958` only when a row has `restSleepActiveRefundPending` "
        "and NO `restSleepAttempts`. A live row CAN lack the counts -- "
        "`local_showdown._apply_rest_sleep_provenance` sets the pending flag at :2784 and "
        "then `continue`s at :2800 without writing them -- so the naive reading is wrong. "
        "What closes the path is the ORDER of the tests: that same :2800 branch sets "
        "`restSleepProvenanceUnrepresentable`, and `_hp_and_status` raises on THAT at "
        ":1914, before it ever reaches :1958. So the surviving way in is a row carrying the "
        "pending flag, no counts and no unrepresentable flag, which no live producer emits. "
        "`engine_world.py` calls the neighbouring branch a CANARY whose expected count in a "
        "fresh post-split era is exactly zero; a census zero CONFIRMS that design property "
        "and is not evidence about coverage."
    ),
    "skip:world_unsupported:rest_sleep_refund_pending_unsplit_legacy": (
        "Raised at `engine_world.py:1986` only when a row carries the pre-split "
        "`restSleepRefundPending` flag and NEITHER producer flag -- and `_hp_and_status` "
        "tests both producer flags first, at :1935 and :1943. "
        "`_mark_legacy_rest_refund_pending` has exactly TWO call sites in "
        "`local_showdown.py` (:2755 and :2785) and each is preceded on the same row by a "
        "producer flag (`restSleepAttemptUnsettled` at :2754, "
        "`restSleepActiveRefundPending` at :2784), so a live row always trips an earlier "
        "branch. Reachable only by replaying a pre-split corpus. Same canary: zero is the "
        "designed value, not an unmeasured absence."
    ),
    "skip:world_unsupported:override_side_missing": (
        "Raised at `engine_world.py:490` when `override.player_teams.get(slot)` is falsy. "
        "The differential builds that mapping at "
        "`engine_transition_differential.py:2396` as "
        '`{slot: true_teams[slot]["packed"] for slot in ("p1", "p2")}` -- a comprehension '
        "over exactly the two slots the loop then iterates, so a slot cannot be ABSENT. "
        "The residual way in is an empty packed string from the bridge snapshot, which "
        "would mean a battle started with an empty team; 10,000 games produced none."
    ),
    "skip:world_unsupported:deferred_opponent_action": (
        "Raised at `engine_world.py:922` on `payload.get(\"deferredOpponentActions\") or "
        "payload.get(\"deferredOpponentActionPriors\")`. The payload always CARRIES both "
        "keys -- `local_showdown._public_materialization_payload` emits them at :2350-2352 "
        "-- so 'never carries' would be false. They are always EMPTY: both derive from "
        "keyword-only parameters defaulting to `None` (:2220-2222, `dict(... or {})` at "
        ":2302), and the differential calls that function with neither argument at both of "
        "its call sites (`engine_transition_differential.py:2649` and `:2760`). An empty "
        "dict is falsy, so the guard never fires. The field is for the opponent-action "
        "predictor, which no differential run uses."
    ),
    "divergence_class:mapper_lossy": (
        "Structural, not instrumental -- see the demonstration on the entry itself."
    ),
    "divergence_class:no_usable_branch": (
        "Structural, not instrumental -- see the demonstration on the entry itself."
    ),
}

# §3.5's static counters that are plain counter keys (the other two are the structural
# divergence classes above). `world_prestate_mismatch:side_conditions` is the key
# `_prestate_mismatch`'s "side conditions ..." message produces once the run loop takes
# its first two whitespace-separated tokens.
STATIC_COUNTER_KEYS = (
    "abort:no_legal_action",
    "skip:no_action_candidates",
    "skip:world_error:no_constructible_candidate",
    "strict:no_damage_rolls",
    "engine_error",
    "world_prestate_mismatch:side_conditions",
)


def inventory() -> dict[str, dict[str, Any]]:
    """`name -> {family, key_kind, key, measurement_independent, ...}` for every entry.

    Keys are counter keys as the differential emits them, so a verdict is a lookup rather
    than an interpretation.
    """

    entries: dict[str, dict[str, Any]] = {}

    for name in sorted(STATIC_COUNTER_KEYS):
        entries[name] = {
            "family": "section_3_5_static_counter",
            "key_kind": "exact",
            "key": name,
            "measurement_independent": False,
        }

    for name, demonstration in sorted(STRUCTURAL_DIVERGENCE_CLASSES.items()):
        entries[f"divergence_class:{name}"] = {
            "family": "section_3_5_static_counter",
            "key_kind": "exact",
            "key": f"divergence_class:{name}",
            "measurement_independent": True,
            "structural_demonstration": demonstration,
        }

    for prefix in DYNAMIC_FAMILY_PREFIXES:
        entries[prefix] = {
            "family": "section_3_5_dynamic_family",
            "key_kind": "prefix",
            "key": prefix,
            "excludes": sorted(DYNAMIC_FAMILY_EXCLUSIONS.get(prefix, ())),
            "measurement_independent": False,
        }

    for reason in sorted(unmappable_choice_reasons() - FIRED_UNMAPPABLE_CHOICE):
        key = f"skip:unmappable_choice:{reason}"
        entries[key] = {
            "family": "section_3_5_unmappable_choice",
            "key_kind": "prefix" if reason.endswith(":") else "exact",
            "key": key,
            "measurement_independent": False,
        }

    unobserved = (
        world_unsupported_reasons()
        - FIRED_IN_CORPUS_WORLD_UNSUPPORTED
        - {"future_sight_pending"}
    )
    for reason in sorted(unobserved):
        key = f"skip:world_unsupported:{reason}"
        entries[key] = {
            "family": "section_3_5_world_unsupported",
            "key_kind": "exact",
            "key": key,
            "measurement_independent": reason in POOL_UNREACHABLE_WORLD_UNSUPPORTED,
        }

    # Outside §3.5's 50, and named rather than dropped: R1 retires it as pool-unreachable.
    entries["skip:world_unsupported:future_sight_pending"] = {
        "family": "retired_pool_unreachable_R1",
        "key_kind": "exact",
        "key": "skip:world_unsupported:future_sight_pending",
        "measurement_independent": True,
    }

    # H15's row-level negatives beyond §3.5's two structural ones.
    for name in sorted(H15_CLAIMED_BANDED_REACHABLE_ONLY):
        entries[f"divergence_class:{name}"] = {
            "family": "H15_class_claimed_banded_reachable_only",
            "key_kind": "exact",
            "key": f"divergence_class:{name}",
            "measurement_independent": False,
        }
    for name in sorted(H15_CLAIMED_NEVER_PRODUCED):
        entries[f"divergence_class:{name}"] = {
            "family": "H15_class_claimed_never_produced",
            "key_kind": "exact",
            "key": f"divergence_class:{name}",
            "measurement_independent": False,
        }

    return entries


# ---------------------------------------------------------------------------
# The scan.
# ---------------------------------------------------------------------------


def _counter_hits(entry: dict[str, Any], counters: dict[str, Any]) -> dict[str, int]:
    """Every nonzero counter in one shard that answers to this inventory entry."""

    key = entry["key"]
    excluded = set(entry.get("excludes") or ())
    if entry["key_kind"] == "exact":
        value = counters.get(key)
        return {key: value} if isinstance(value, (int, float)) and value else {}
    return {
        found: value
        for found, value in counters.items()
        if found.startswith(key) and found not in excluded and value
    }


def scan_shards(shards: list[tuple[str, dict]]) -> dict[str, dict[str, dict[str, int]]]:
    """`entry -> {shard -> {counter key -> value}}` over the committed census shards."""

    entries = inventory()
    found: dict[str, dict[str, dict[str, int]]] = {}
    for name, entry in entries.items():
        for shard, report in shards:
            hits = _counter_hits(entry, report.get("counters") or {})
            if hits:
                found.setdefault(name, {})[shard] = hits
    return found


def attribute_seeds(
    checkpoints: Iterable[Path], entries: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, int]]]:
    """`entry -> {counter key -> {seed -> value}}` from the per-game checkpoint records.

    This is the only place a SEED enters the census. The totals it produces are closed
    against the committed shard counters by `closure()` below and by the pin, so a
    checkpoint that went missing breaks the closure rather than quietly weakening a claim.
    """

    out: dict[str, dict[str, dict[str, int]]] = {}
    for path in checkpoints:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            seed = record["seed"]
            counters = record.get("counters") or {}
            for name, entry in entries.items():
                for key, value in _counter_hits(entry, counters).items():
                    out.setdefault(name, {}).setdefault(key, {})[str(seed)] = value
    return out


def closure(
    shard_hits: dict[str, dict[str, dict[str, int]]],
    seed_hits: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    """Per-counter `sum(per-seed) == sum(per-shard)`, reported rather than asserted here.

    Reported so the artifact carries its own falsifier: the pin asserts it, and a reader
    of the JSON can check it without running anything.
    """

    report: dict[str, Any] = {}
    for name, by_shard in shard_hits.items():
        totals: Counter = Counter()
        for hits in by_shard.values():
            totals.update(hits)
        per_seed: Counter = Counter()
        for key, seeds in seed_hits.get(name, {}).items():
            per_seed[key] = sum(seeds.values())
        report[name] = {
            "shard_totals": dict(sorted(totals.items())),
            "checkpoint_totals": dict(sorted(per_seed.items())),
            "agrees": dict(sorted(totals.items())) == dict(sorted(per_seed.items())),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-shard", nargs="+", type=Path, required=True)
    parser.add_argument("--banded-shard", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args(argv)

    def load(paths: list[Path]) -> list[tuple[str, dict]]:
        out = []
        for path in sorted(paths):
            document = json.loads(path.read_text(encoding="utf-8"))
            try:
                label = path.resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                label = path.name
            out.append((label, document))
        return out

    strict = load(args.strict_shard)
    banded = load(args.banded_shard)
    shards = strict + banded
    entries = inventory()

    exact_keys, prefixes = counter_key_space()
    unemittable = sorted(
        name
        for name, entry in entries.items()
        if entry["key_kind"] == "exact"
        and entry["key"] not in exact_keys
        and not any(entry["key"].startswith(p) for p in prefixes)
    )

    shard_hits = scan_shards(shards)
    checkpoints = (
        sorted(args.checkpoint_dir.glob("*.jsonl")) if args.checkpoint_dir else []
    )
    seed_hits = attribute_seeds(checkpoints, entries) if checkpoints else {}

    def span(reports: list[tuple[str, dict]]) -> dict[str, Any]:
        return {
            "seed_min": min(r["seeds"]["min"] for _n, r in reports),
            "seed_max": max(r["seeds"]["max"] for _n, r in reports),
            "games": sum(r["games"] for _n, r in reports),
            "boundaries_full_round": sum(r["boundaries_full_round"] for _n, r in reports),
            "boundaries_measured": sum(r["boundaries_measured"] for _n, r in reports),
            "transitions_matched": sum(r["transitions_matched"] for _n, r in reports),
            "transitions_diverged": sum(r["transitions_diverged"] for _n, r in reports),
            "shards": [n for n, _r in reports],
        }

    strict_span = span(strict)
    banded_span = span(banded)

    provenance = {
        name: json.loads(report["checkpoint_provenance"]["distinct"][0])
        for name, report in shards
    }
    distinct_provenance = sorted({json.dumps(p, sort_keys=True) for p in provenance.values()})
    fingerprint = json.loads(distinct_provenance[0])["engine_fingerprint"]

    verdicts: dict[str, Any] = {}
    banded_shards = set(banded_span["shards"])
    for name, entry in sorted(entries.items()):
        record: dict[str, Any] = dict(entry)
        if name in shard_hits:
            record["verdict"] = "FIRED"
            record["evidence"] = {
                shard: hits for shard, hits in sorted(shard_hits[name].items())
            }
            record["seeds"] = seed_hits.get(name, {})
            # WHICH ARM, and it is never cosmetic. `--matcher banded` reaches the
            # protocol-evidence tail of `classify_divergence` that the strict path
            # cannot, and H15's negative about part of that tail is already scoped to
            # "no committed artifact used that path". A firing that happened only under
            # banded refutes "never produced" and says nothing about the SHIPPING
            # matcher, so the two must not be reported as one.
            record["arms_that_fired"] = sorted(
                {
                    "banded" if shard in banded_shards else "strict"
                    for shard in shard_hits[name]
                }
            )
        elif entry.get("measurement_independent") and "structural_demonstration" in entry:
            record["verdict"] = "UNREACHABLE_STRUCTURAL"
        elif entry.get("measurement_independent"):
            record["verdict"] = "UNREACHABLE_POOL"
        else:
            record["verdict"] = "NOT_OBSERVED_AT_SCOPE"
        # ONE scope sentence, covering BOTH arms, because both arms measure every entry.
        # `classify_divergence` runs on any diverged boundary whatever the matcher, and
        # every world-construction and choice-mapping counter is incremented BEFORE the
        # matcher is consulted at all. What the matcher changes is which miss text the
        # classifier is handed, and that is recorded per entry in `arms_that_fired`
        # rather than smuggled into the denominator.
        record["scope"] = (
            f"{strict_span['games'] + banded_span['games']:,} games on unregistered "
            f"seeds {strict_span['seed_min']:,}-{banded_span['seed_max']:,} "
            f"({strict_span['games']:,} strict + {banded_span['games']:,} banded), "
            f"{strict_span['boundaries_measured'] + banded_span['boundaries_measured']:,}"
            f" measured boundaries, engine fingerprint {fingerprint[:16]}"
        )
        if name in CENSUS_CANNOT_REACH:
            record["census_cannot_reach"] = CENSUS_CANNOT_REACH[name]
        verdicts[name] = record

    document = {
        "_README": (
            "C153. One wide-seed census re-measuring every window-scoped negative in "
            "reports/c138_known_gaps_ledger.md section 3.5, plus the row-level ones in "
            "G33b/G33c/G50/H13/H14/H15, OUTSIDE the two permitted windows. Regenerate "
            "with scripts/c153_wide_negative_census.py. NOT fidelity evidence: these are "
            "unregistered seeds and the divergence rate here must never be quoted as the "
            "program's."
        ),
        "arms": {"strict": strict_span, "banded": banded_span},
        "provenance": {
            "distinct": distinct_provenance,
            "single_build": len(distinct_provenance) == 1,
        },
        "taxonomy_sizes": {
            "world_unsupported_reasons": len(world_unsupported_reasons()),
            "divergence_classes": len(divergence_classes()),
            "unmappable_choice_reasons": len(unmappable_choice_reasons()),
        },
        "inventory_size": {
            "total": len(entries),
            "section_3_5_verified_negatives": sum(
                1 for e in entries.values() if e["family"].startswith("section_3_5")
            ),
            "measurement_independent": sum(
                1 for e in entries.values() if e.get("measurement_independent")
            ),
            "window_scoped": sum(
                1
                for e in entries.values()
                if e["family"].startswith("section_3_5") and not e.get("measurement_independent")
            ),
        },
        "entries_with_no_emittable_counter_key": unemittable,
        # ANTI-VACUITY. Every absence above is a loop over the same shards, and a loop
        # over a shard set that stopped being read passes. These four counters DO fire in
        # the census, so a scanner or a corpus that silently emptied turns the pin red
        # instead of "verifying" all 46.
        "controls": {
            key: sum(
                (report.get("counters") or {}).get(key, 0) for _name, report in shards
            )
            for key in (
                "skip:unmappable_choice:struggle_not_submittable",
                "skip:world_unsupported:volatile_unsupported",
                "skip:world_unsupported:materialization_blocker",
                "world_prestate_mismatch",
            )
        },
        # WHAT THIS SAMPLE CAN AND CANNOT RULE OUT, stated as a bound rather than implied
        # by a headline count. Rule of three: zero events in N independent trials puts the
        # 95 % upper bound on the per-trial rate at 3/N.
        #
        # THREE DENOMINATORS, and the third is the one a reviewer would catch. A
        # `divergence_class` negative is not a claim about boundaries: the classifier only
        # runs on a boundary that already DIVERGED, so the trials for "class X never
        # fired" number `transitions_diverged`, not `boundaries_measured`. Quoting the
        # boundary bound for a class would overstate this census by four orders of
        # magnitude, which is the exact shape of defect the rest of this work is about.
        "statistical_bounds": {
            arm: {
                "games": span_["games"],
                "boundaries_measured": span_["boundaries_measured"],
                "classified_divergences": span_["transitions_diverged"],
                "rule_of_three_per_game_upper_95": round(3 / span_["games"], 8),
                "rule_of_three_per_boundary_upper_95": round(
                    3 / span_["boundaries_measured"], 10
                ),
                "rule_of_three_per_divergence_upper_95": round(
                    3 / span_["transitions_diverged"], 6
                ),
                "one_in_n_games": round(span_["games"] / 3),
                "one_in_n_boundaries": round(span_["boundaries_measured"] / 3),
            }
            for arm, span_ in (
                ("strict", strict_span),
                ("banded", banded_span),
                (
                    "combined",
                    {
                        "games": strict_span["games"] + banded_span["games"],
                        "boundaries_measured": strict_span["boundaries_measured"]
                        + banded_span["boundaries_measured"],
                        "transitions_diverged": strict_span["transitions_diverged"]
                        + banded_span["transitions_diverged"],
                    },
                ),
            )
        },
        # C152's two refutations, RE-MEASURED on the shipping build. C152 found them on a
        # THROWAWAY INSTRUMENTED engine (`89797289...`, the shipping tree plus two
        # `eprintln!` blocks) at an older harness digest, so until now no committed
        # artifact showed either counter firing on an engine anyone can rebuild. These
        # figures are the first that do. Recorded rather than assumed, because a
        # refutation that only reproduces on an unreproducible build is one merge away
        # from being unsupported again.
        "c152_refutations_on_the_shipping_build": {
            key: {
                "strict_arm": sum(
                    (report.get("counters") or {}).get(key, 0) for _n, report in strict
                ),
                "banded_arm": sum(
                    (report.get("counters") or {}).get(key, 0) for _n, report in banded
                ),
            }
            for key in (
                "skip:rump_branch_set",
                "strict:branch_event_legal_error:BranchLegalRollError",
            )
        },
        "verdicts": verdicts,
        "closure": closure(shard_hits, seed_hits),
        "counts": {
            "FIRED": sum(1 for v in verdicts.values() if v["verdict"] == "FIRED"),
            "NOT_OBSERVED_AT_SCOPE": sum(
                1 for v in verdicts.values() if v["verdict"] == "NOT_OBSERVED_AT_SCOPE"
            ),
            "UNREACHABLE_STRUCTURAL": sum(
                1 for v in verdicts.values() if v["verdict"] == "UNREACHABLE_STRUCTURAL"
            ),
            "UNREACHABLE_POOL": sum(
                1 for v in verdicts.values() if v["verdict"] == "UNREACHABLE_POOL"
            ),
        },
    }

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"-> {args.write}")
    else:
        print(json.dumps(document, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
