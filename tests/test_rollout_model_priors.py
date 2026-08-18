"""Rollout leaves composed with MODEL PRIORS, and the fidelity gate re-run on
the model driver.

Search-ceiling program Phase 1 instrument 2 -- the arbiter.
`tests/test_rollout_leaf_arbiter.py` already
certifies the seam on the SEQUENTIAL `puct_search_multi` path. That path is
uniform-priors, and the campaign's surviving search configuration is priors ON,
which lives only on the model path (`_search_ladder` ->
`search_batched_multi_encoded`). §7 of the plan: "The selection-tuning
campaign's surviving config (priors on) is the search configuration under test
everywhere."

So the whole basis for saying "same search, same everything" has to be
RE-ESTABLISHED here. It is not inherited: the sequential gate compares a
different driver against a different production entry point, and a seam that is
faithful in one does not thereby become faithful in the other. This file is
that re-establishment, and it is the reason the composition is claimable at all.

## The control is the gate, and it is not dead code

`rollout_leaf_mode="model_value"` runs leaf values through the identical new
deferred-row plumbing the arm uses while keeping production's leaf value. That
is what lets the gate be an equality rather than an argument: if the plumbing
perturbed selection, expansion, backup, the prior wiring, the early-stop lock,
the collision ledger or the depth occupancy, the control would diverge from
production on that field.

Timing fields are excluded, and that exclusion is named rather than quietly
applied: `elapsed_s`, `iterations_per_s` and the per-phase `*_s` decomposition
are wall-clock measurements of the same work and cannot be equal across two
runs of anything. Every other field production emits is compared, including the
per-arm `side_one`/`side_two` visit and value rows and `root_value` -- the
fields that would actually move if the search changed.

## Every gate here ships a demonstrated failing input

Per the program's rule (§6): a check that cannot read False certifies nothing;
the campaign found three such guards green for months.

| gate | reads False on |
|---|---|
| `test_the_model_value_control_reproduces_production_field_for_field` | `rollout` pricer; a changed `leaf_batch` (both asserted in the two tests below it) |
| `test_the_encode_skip_preserves_every_prior` | the `rollout_skip_all` mutant, which drops `prior_branches` from 97 to 0 |
| `test_rollout_values_are_not_seat_reflected` | the `model_value` control on the same pair of seats, which IS reflected and so does move |
| the boundary rejections | one test per refusal, each constructing the offending input |

## What this file does NOT close

The estimand. `rollout_policy="uniform"` prices
`P(side one wins | both seats play uniformly at random from here)`, which is not
the vhprobe shards' policy-continuation `true_*`. No test here can fix that; it
is a property of the pricer, it travels with every number this arm produces, and
`test_the_report_carries_the_estimand_ledger` pins the fields that let a reader
see how much of the "oracle" was actually the handcrafted fallback.
"""

from __future__ import annotations

import copy
import json
import random
import re
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover
    pokezero_search = None

from test_model_priors_search import _EncodedSearchFixture, _crate_ready

from pokezero import engine_search
from pokezero.engine_search import (
    EngineMctsConfig,
    EngineMctsPolicy,
    EngineMctsStats,
    EngineSearchWitnessError,
    EngineShardSchemaError,
    ROLLOUT_LEAF_QUOTIENT_FIELDS,
    ROLLOUT_LEAF_REPORT_FIELDS,
    ROLLOUT_LEAF_SHARD_FIELDS,
    ROLLOUT_LEAF_SHARD_MARKERS,
    ROLLOUT_LEAF_SHARD_SCHEMA,
    ROLLOUT_LEAF_V1_WORLD_FIELD,
    ROLLOUT_LEAF_WITNESS_FIELDS,
    migrate_rollout_leaf_shard_v1,
    require_rollout_leaf_shard_schema,
    require_rollout_leaf_witness,
)

#: Wall-clock measurements of the same work. Named here, once, so a reader can
#: see the whole exclusion list and check it contains nothing semantic.
TIMING_FIELDS = frozenset(
    {
        "elapsed_s",
        "iterations_per_s",
        "encode_s",
        "model_s",
        "tree_s",
        "fold_clone_s",
        "render_s",
        "fold_advance_s",
        "tensor_s",
        "action_map_s",
        "row_input_s",
        "products_s",
        "row_write_s",
    }
)

#: Counters that measure WORK RATHER THAN A CLAIM ABOUT THE SEARCH. The encode
#: skip is a work reduction by construction, so these three are expected to move
#: between a skipping and a non-skipping run -- and they are the ONLY three
#: allowed to. `prior_branches` is deliberately NOT here: the whole point of the
#: skip gate is that priors survive it.
ENCODE_SKIP_WORK_FIELDS = frozenset(
    {"model_evals", "rollout_encode_skipped", "rollout_leaf_mode"}
)

#: A RATIO-SHAPED LITERAL: `1.8x`, `4.3x`, `1.25-3.98x`. The VALUE CLASS the CPU
#: fence must not assert as the arm's cost, rather than the two specific figures
#: someone happened to enumerate -- a third, unlisted ratio survived the enumerated
#: form, which is the failure mode a value-class check exists to remove.
#:
#: B5. THE CHARACTER CLASS WAS THE HOLE, and it was the hole in the direction that
#: matters: the previous pattern was `...\s*x\b`, ASCII lowercase `x` only, while the
#: headline figure this guard exists to protect is written `7.7x-16.9x` WITH U+00D7
#: MULTIPLICATION SIGNS in the PR body it was copied from. So `7.7<U+00D7>`, `7.7X`,
#: `7.7-fold` and `a factor of 7.7` all evaded a guard built to catch exactly that
#: number. Four spellings, one value class:
#:
#:   * `x` / `X` / U+00D7 (MULTIPLICATION SIGN) / U+2715 (MULTIPLICATION X);
#:   * `-fold` and `fold` after a space;
#:   * the prose form `a factor of N`.
#:
#: `\b` is dropped after the symbol class because U+00D7 is not a word character, so
#: `\b` after it asserts the opposite boundary and never matched.
_COST_RATIO = re.compile(
    r"(?:(?:a|an)\s+factor\s+of\s+)?"
    r"\d+(?:\.\d+)?(?:\s*[-‐-―−]\s*\d+(?:\.\d+)?)?"
    r"(?:\s*(?:[xX×✕](?![a-zA-Z0-9_])|[-‐-―−]?fold\b))"
    r"|(?:a|an)\s+factor\s+of\s+\d+(?:\.\d+)?"
)

#: The paragraph that is ALLOWED to name the contested figures, because its whole
#: content is why neither may be quoted as the cost.
_NO_POINT_RATIO = "NO POINT RATIO, because"

#: EVERY file in scope, not the twenty-one lines of one of them.
#:
#: B5. The previous scope was `_cpu_fence_window(engine_search.py)` -- a 1400-character
#: slice, twenty-one lines, of a single `.py` file, and NO DOCS AT ALL. The figure it
#: protects is a prose figure: it was published in a PR body, it is restated in
#: `docs/`, and a retraction that only holds inside one comment block in one module is
#: not a retraction of anything a reader is likely to find. The tracked prose that
#: discusses this arm's CPU cost is in scope too.
_RETRACTION_SCOPE_DOCS = (
    "docs/mcts_value_gap_findings_20260812.md",
    "docs/selfplay_mcts_roadmap.md",
)


def _cpu_fence_window(source: str) -> str:
    """The `rollout_threads` CPU fence's own prose, read on its own.

    Located rather than counted across the file, for the reason the sibling guard
    records: an occurrence count over the whole module is satisfied by the guard's
    own literal and says nothing about the fence that carried the claim.
    """

    start = source.index("HOW MUCH MORE CPU, MEASURED")
    return source[start : start + 1400]


def _paired_eval_module():
    """`scripts/foulplay_paired_eval.py`, imported as a module.

    B1's READER. `scripts/` is not a package, so the driver is loaded by path -- and
    loaded rather than re-implemented, because the whole point is that the refusal is
    reached from the code that actually reads the shard off disk.
    """

    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "foulplay_paired_eval.py"
    spec = importlib.util.spec_from_file_location("_paired_eval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _power_report_module():
    """`scripts/foulplay_power_report.py`, imported as a module. B1's POOLING reader."""

    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "foulplay_power_report.py"
    spec = importlib.util.spec_from_file_location("_power_report_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script_module(filename: str, alias: str):
    """A `scripts/` writer, imported as a module.

    The two loaders above predate this one and are kept as they are; these are the
    two writer modules the battery's `targets` never listed, so nothing had ever
    imported them under test at all.
    """

    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _acceptance_h2h_module():
    """`scripts/mcts_acceptance_h2h.py` -- THE THIRD WRITER, never a battery target."""

    return _script_module("mcts_acceptance_h2h.py", "_acceptance_h2h_under_test")


def _depth_grid_module():
    """`scripts/hc_depth_grid.py` -- THE FOURTH WRITER, never a battery target."""

    return _script_module("hc_depth_grid.py", "_depth_grid_under_test")


def _sanctioned_span(window: str) -> tuple[int, int]:
    """The one paragraph allowed to quote the contested figures."""

    start = window.index(_NO_POINT_RATIO)
    return start, window.index("\n    #\n", start)


def _stray_cost_ratios(source: str) -> list[str]:
    """Every ratio-shaped literal in the CPU fence OUTSIDE the sanctioned paragraph.

    A function of its input so the guard can be shown to read False on synthetic
    attacks -- including the third shape nobody enumerated, and the four spellings
    that evaded the ASCII-only character class.
    """

    window = _cpu_fence_window(source)
    start, end = _sanctioned_span(window)
    return [
        match.group(0)
        for match in _COST_RATIO.finditer(window)
        if not start <= match.start() < end
    ]


@unittest.skipUnless(_crate_ready, "crate not built with the model feature")
class RolloutModelPriorsTest(_EncodedSearchFixture, unittest.TestCase):
    """Priors ON, leaves priced by rollouts, and the gate that certifies it."""

    # Priors-on everywhere in this file: both the self head (`model_priors`) and
    # the opponent head (`use_opponent_priors`), because that pair is what the
    # campaign's surviving config turns on and what the arbiter must reproduce.
    SIMS = 32
    SEED = 5
    DEPTH = 2
    ROLLOUTS = 8
    # Deliberately short. These are mechanics gates on a random-weights artifact,
    # and a short cap makes the FALLBACK fraction large -- which is the honest
    # state to test the fallback accounting in. It is NOT the arm's setting, and
    # `test_the_report_carries_the_estimand_ledger` asserts the blend is reported
    # as a blend precisely so a short cap can never be mistaken for an oracle.
    MAX_PLIES = 40

    def _native(self):
        return pokezero_search.NativeLeafModel(
            str(self.artifact),
            device="cpu",
            window=1,
            tokens=int(self.layout["token_count"]),
            categorical_features=int(self.layout["categorical_feature_count"]),
            numeric_features=int(self.layout["numeric_feature_count"]),
        )

    def _search(
        self,
        *,
        mode: str | None = None,
        sims: int | None = None,
        batch: int = 1,
        seed: int | None = None,
        depth: int | None = None,
        rollouts: int | None = None,
        max_plies: int | None = None,
        policy: str = "uniform",
        rollout_seed: int = 0,
        threads: int = 1,
        model_priors: bool = True,
        use_opponent_priors: bool = True,
        row_inputs: str | None = None,
        _raw: bool = False,
    ):
        """One encoded search. `mode=None` is PRODUCTION: the seam's positionals
        are not appended at all, so the call is byte for byte the pre-seam one.
        """
        position = self.position
        fold = pokezero_search.FoldState.from_payload(position["fold_state"])
        args = [
            position["state_str"],
            self.SIMS if sims is None else sims,
            batch,
            self.tables_json,
            position["row_inputs"] if row_inputs is None else row_inputs,
            position["ctx"],
            fold,
            self.DEPTH if depth is None else depth,
            1.4,
            self.SEED if seed is None else seed,
            True,  # deep_ko_split
            model_priors,
            0,  # early_stop_min_sims
            True,  # early_stop_side_one
            bool(use_opponent_priors),
            None,  # fpu_reduction
            False,  # arm_priors
        ]
        if mode is not None:
            args += [
                mode,
                self.ROLLOUTS if rollouts is None else rollouts,
                self.MAX_PLIES if max_plies is None else max_plies,
                policy,
                rollout_seed,
                threads,
                False,  # rollout_branch_on_damage
            ]
        raw = self._search_raw_from_args(args)
        return raw if _raw else json.loads(raw)

    def _search_raw(self, **kwargs) -> str:
        """The report as the crate emitted it, before `json.loads` normalises it.

        Exists because `json.loads` silently collapses duplicate keys, so a
        defect in the emitted JSON is invisible to every test that goes through
        the dict. One did ship.
        """
        return self._search(_raw=True, **kwargs)

    def _search_raw_from_args(self, args) -> str:
        return self._native().search_batched_multi_encoded(*args)

    def _differing(self, left: dict, right: dict, *, ignore=frozenset()) -> list[str]:
        """Field names on which two reports disagree, timings excluded.

        Compares the UNION of keys, not the intersection: comparing only shared
        keys would let a field vanish from one report and read as agreement,
        which is the failure mode where an absent input passes for a matching
        one.
        """
        keys = (set(left) | set(right)) - TIMING_FIELDS - set(ignore)
        return sorted(key for key in keys if left.get(key) != right.get(key))

    # ---------------------------------------------------------------- the gate

    def test_the_model_value_control_reproduces_production_field_for_field(self) -> None:
        """THE RE-RUN FIDELITY GATE, on the model driver, at priors ON.

        This is the whole basis for "same search, same everything" on the arm's
        actual path, and it is asserted here rather than inherited from the
        sequential gate.
        """
        production = self._search(mode=None)
        control = self._search(mode="model_value")
        # The seam's own columns are additive: production cannot carry them, so
        # they are compared as "present only on the control" rather than as a
        # disagreement.
        seam_only = set(control) - set(production)
        self.assertEqual(
            self._differing(production, control, ignore=seam_only),
            [],
            "the model_value control must reproduce production on every field "
            "production emits",
        )
        # And the gate must actually be looking at the fields that would move.
        # A gate that compared only scalars nobody reads would pass vacuously.
        for field in ("side_one", "side_two", "root_value", "prior_branches",
                      "depth_occupancy", "expansions", "leaf_evals"):
            self.assertIn(field, production, f"{field} must be in the compared set")

    def test_the_gate_holds_at_the_panels_own_batch(self) -> None:
        """The gate must hold at b64, because b64 IS production here.

        This is where the model path differs from the sequential one and it
        matters for how the arm may be configured. On the sequential
        `puct_search_multi` path `leaf_batch=1` is production and any batching is
        an uncertified fidelity loss. On THIS path the campaign's surviving
        configuration is itself batched -- the priors-on panel ran
        d4-s4096-b64-w4 -- so `search_batch=64` is not a compromise to be
        acknowledged, it is the thing being reproduced.

        Asserted at the panel's batch and depth rather than inferred from the
        b1 result, because "it held at 1 so it holds at 64" is exactly the kind
        of inheritance this whole file exists to refuse.
        """
        production = self._search(mode=None, batch=64, sims=256, depth=4)
        control = self._search(mode="model_value", batch=64, sims=256, depth=4)
        seam_only = set(control) - set(production)
        self.assertEqual(
            self._differing(production, control, ignore=seam_only),
            [],
            "the control must reproduce production at the panel's own batch",
        )
        self.assertEqual(production["batch_size"], 64)
        self.assertEqual(production["max_depth"], 4)
        # The round structure really did batch, so this is not a b1 run wearing
        # a b64 label.
        self.assertLess(production["rounds"], 256)

    def test_the_gate_reads_false_on_the_rollout_pricer(self) -> None:
        """Demonstrated failing input 1: swap the pricer, the gate must fail.

        If this passed, the gate above would be certifying nothing -- it would
        mean the report is blind to the leaf value, and the arm could not be
        distinguished from production by any field.
        """
        production = self._search(mode=None)
        arm = self._search(mode="rollout")
        seam_only = set(arm) - set(production)
        differing = self._differing(production, arm, ignore=seam_only)
        self.assertNotEqual(differing, [], "swapping the pricer must move the report")
        # And it must move the fields that MEAN the search differed, not merely
        # the work counters. Otherwise a reader could not tell a different
        # search from a differently-instrumented one.
        self.assertIn("root_value", differing)
        self.assertTrue(
            {"side_one", "side_two"} & set(differing),
            f"per-arm rows must move under a different leaf value; moved: {differing}",
        )

    def test_the_gate_reads_false_on_a_changed_leaf_batch(self) -> None:
        """Demonstrated failing input 2: batching is a fidelity loss, and the
        gate sees it.

        `leaf_batch > 1` is not an approximation of `leaf_batch = 1`; it changes
        which selections observe which virtual losses. The sequential gate
        demonstrates this on its own path and it must hold here too.
        """
        control = self._search(mode="model_value", batch=1)
        batched = self._search(mode="model_value", batch=8)
        differing = self._differing(control, batched)
        self.assertNotEqual(
            differing, [], "a changed leaf_batch must move the report"
        )
        self.assertNotEqual(
            differing,
            ["batch_size"],
            "the ONLY difference must not be the echoed knob itself -- a gate "
            "that fires on a literal echo of the mutated field covers nothing",
        )

    # ------------------------------------------------------------ encode skip

    def test_the_encode_skip_preserves_every_prior(self) -> None:
        """The encode skip must be a WORK reduction and nothing else.

        Under rollout leaves the model's value is unused, so a leaf that cannot
        host a child decision node needs no forward. The risk is that the skip
        predicate drifts away from the prior-map guard and silently drops
        priors, which would make the arm a different search than the panel's --
        the one thing this arm may not be.

        `rollout_encode_all` is the same rollout values with the skip disabled.
        Same ordinals, same rollout seeds, same leaves, so the trees are
        identical and this is a sharp equality.
        """
        reference = self._search(mode="rollout_encode_all")
        arm = self._search(mode="rollout")
        self.assertEqual(
            self._differing(reference, arm, ignore=ENCODE_SKIP_WORK_FIELDS),
            [],
            "the encode skip must change only the work counters",
        )
        self.assertEqual(
            reference["prior_branches"],
            arm["prior_branches"],
            "priors must survive the skip",
        )
        self.assertEqual(reference["prior_fallbacks"], arm["prior_fallbacks"])
        self.assertGreater(
            arm["rollout_encode_skipped"],
            0,
            "if nothing was skipped this gate is vacuous -- it would be "
            "comparing a run against itself",
        )
        self.assertLess(
            arm["model_evals"],
            reference["model_evals"],
            "the skip must actually remove forwards",
        )
        # The accounting identity, with the term that made it FALSE as first
        # written. Every leaf either got a forward or was counted as skipped,
        # exactly once. The extra forward is the single ROOT forward that prices
        # the root node's priors -- and it lives inside `if model_priors`
        # (`model.rs`), so it is NOT unconditional. Asserted as `+1` flat, this
        # read `lhs=136 rhs=137` on 30 of 242 swept configs, all of them
        # `model_priors=False`. A wrong identity, not an untested one, and it
        # would have failed the moment anyone ran the arm without priors.
        #
        # `model_priors` is now required for this arm, so the arm itself always
        # pays the root forward; the identity is still written conditionally
        # rather than hard-coding `+1`, because the term's EXISTENCE is what the
        # reader needs to understand and a bare `+1` hides it.
        root_forwards = 1 if arm["model_priors"] else 0
        self.assertEqual(
            arm["model_evals"] + arm["rollout_encode_skipped"],
            arm["leaves_priced"] + root_forwards,
            "forwards + skips must account for every leaf plus the root forward",
        )

    def test_the_accounting_identity_holds_with_priors_off_too(self) -> None:
        """The demonstrated failing input for the identity's root-forward term.

        The arm now refuses `model_priors=False`, but the CORE still supports it
        and the gate fixtures reach it, so the identity has to be right there or
        it is right by luck. With priors off there is no root forward, so the
        term is 0 -- and a flat `+1` reads False here, which is precisely how the
        bug was found.
        """
        run = self._search(mode="rollout", model_priors=False,
                           use_opponent_priors=False)
        self.assertFalse(run["model_priors"])
        self.assertEqual(run["model_evals"], 0,
                         "priors off means no forward is needed at all")
        self.assertEqual(
            run["model_evals"] + run["rollout_encode_skipped"],
            run["leaves_priced"],
            "with no root forward the identity carries no +1",
        )
        # And the flat form really is wrong here -- stated as an assertion so the
        # regression cannot come back quietly.
        self.assertNotEqual(
            run["model_evals"] + run["rollout_encode_skipped"],
            run["leaves_priced"] + 1,
        )

    def test_the_report_has_no_duplicate_json_keys(self) -> None:
        """The seam appended a second `"rounds"` key for one commit.

        `to_rollout_only_json_fields` was added to prevent exactly that and then
        not called; `to_json_fields` shipped instead, emitting `"rounds"` beside
        the one the encoded core has reported since before this seam existed.
        Harmless only because the values coincided and `json.loads` keeps the
        last -- which is why no assertion over the parsed dict could see it.

        So this parses the RAW STRING. A dict cannot represent the defect, and a
        gate that cannot represent the defect cannot catch it.
        """
        for mode in (None, "model_value", "rollout", "rollout_encode_all"):
            with self.subTest(mode=mode):
                duplicates = self._duplicate_keys(self._search_raw(mode=mode))
                self.assertEqual(duplicates, {}, f"duplicate report keys: {duplicates}")

    @staticmethod
    def _duplicate_keys(raw: str) -> dict[str, int]:
        """Keys repeated WITHIN a single JSON object, anywhere in the document.

        Per object, not per document. The first draft of this counted across all
        nested objects and reported `{'move': 18, 'visits': 18, 'q': 18}` -- the
        per-arm rows, which legitimately repeat those keys once per arm. That
        version would have failed on a correct report and, worse, a reader would
        have "fixed" it by loosening the check until it passed, which is how a
        gate ends up unable to see the thing it was built for.
        """
        import json as _json

        duplicates: dict[str, int] = {}

        def inspect(pairs):
            counts: dict[str, int] = {}
            for key, _ in pairs:
                counts[key] = counts.get(key, 0) + 1
            for key, n in counts.items():
                if n > 1:
                    duplicates[key] = max(duplicates.get(key, 0), n)
            return dict(pairs)

        _json.loads(raw, object_pairs_hook=inspect)
        return duplicates

    def test_the_duplicate_key_gate_reads_false_on_a_duplicated_key(self) -> None:
        """The demonstrated failing input, and it earned it twice over.

        The defect this catches shipped for one commit (`"rounds"` emitted by both
        the encoded core and the rollout block), and the FIRST version of the
        detector could not have caught it while failing on a correct report. So
        the gate needs its own proof both that it fires on a real duplicate and
        that it does NOT fire on the per-arm rows.
        """
        self.assertEqual(
            self._duplicate_keys('{"rounds":32,"leaf_evals":9,"rounds":32}'),
            {"rounds": 2},
        )
        # Nested, which is where the report's own duplicate lived.
        self.assertEqual(
            self._duplicate_keys('{"a":{"rounds":1,"rounds":2}}'), {"rounds": 2}
        )
        # And the shape that must NOT read False: the same keys once per arm.
        self.assertEqual(
            self._duplicate_keys(
                '{"side_one":[{"move":"a","visits":1,"q":0.5},'
                '{"move":"b","visits":2,"q":0.4}]}'
            ),
            {},
        )

    def test_the_encode_skip_gate_reads_false_on_a_skip_everything_mutant(self) -> None:
        """Demonstrated failing input for the gate above.

        `rollout_skip_all` skips the forward for EVERY leaf, including the ones
        whose prior map needs it -- precisely the defect the gate exists to
        catch. It loses priors silently rather than panicking, because a
        `PanicException` crossing the FFI boundary escapes the caller's `except
        Exception` and kills the whole shard.
        """
        reference = self._search(mode="rollout_encode_all")
        mutant = self._search(mode="rollout_skip_all")
        differing = self._differing(
            reference, mutant, ignore=ENCODE_SKIP_WORK_FIELDS
        )
        self.assertIn(
            "prior_branches",
            differing,
            "the mutant must lose priors and the gate must see it",
        )
        self.assertEqual(
            mutant["prior_branches"],
            0,
            "skipping every forward leaves no branch priors at all",
        )
        self.assertGreater(
            reference["prior_branches"],
            0,
            "the reference must have priors to lose, or the contrast is empty",
        )

    # ------------------------------------------------------- seat orientation

    def _row_inputs_for_slot(self, slot: str) -> str:
        payload = json.loads(self.position["row_inputs"])
        payload["observation_metadata"]["showdown_slot"] = slot
        return json.dumps(payload, sort_keys=True)

    def test_rollout_values_are_not_seat_reflected(self) -> None:
        """Rollout values are side-one-absolute; model values are self-relative.

        `rollout_once` prices a terminal from `battle_is_over()` (`> 0` = side
        ONE won) and the HP-fraction cap fallback is side-one-absolute too --
        the same frame `finalize` and the tree's own terminal branches use. The
        model's value is instead +1 iff the ENCODING seat won, so `model.rs`
        reflects it when the searching seat is side two. Feeding rollout values
        through that reflection would invert every leaf on p2 decisions and
        leave p1 correct: a defect invisible in half of all games.

        With `model_priors` off there is no forward at all on the rollout path,
        so flipping the seat changes nothing the rollout arm can see -- and the
        reports must be identical. The `model_value` half of this test is its
        discriminating power: on the SAME pair of seats the reflected path DOES
        move, so an identical-reports result above is a fact about the frame and
        not about the test being unable to tell seats apart.
        """
        p1 = self._row_inputs_for_slot("p1")
        p2 = self._row_inputs_for_slot("p2")

        rollout_p1 = self._search(mode="rollout", model_priors=False,
                                  use_opponent_priors=False, row_inputs=p1)
        rollout_p2 = self._search(mode="rollout", model_priors=False,
                                  use_opponent_priors=False, row_inputs=p2)
        # `collision_self_side` is excluded because it is a LABEL of which
        # seat searched, not a value the search produced: it is "side_one" for
        # p1 and "side_two" for p2 by definition, and it ships alongside the
        # collision counts precisely so a pooled read cannot average p1
        # decisions against p2 decisions in opposite frames. Excluding a value
        # here would gut the test; excluding the seat's own name does not, and
        # the measured result is that it is the ONLY field that moves.
        self.assertEqual(
            self._differing(rollout_p1, rollout_p2, ignore={"collision_self_side"}),
            [],
            "rollout values are side-one-absolute, so the searching seat must "
            "not change them",
        )

        # Discriminating power, on the same two seats.
        model_p1 = self._search(mode="model_value", model_priors=False,
                                use_opponent_priors=False, row_inputs=p1)
        model_p2 = self._search(mode="model_value", model_priors=False,
                                use_opponent_priors=False, row_inputs=p2)
        model_differing = self._differing(
            model_p1, model_p2, ignore={"collision_self_side"}
        )
        self.assertIn(
            "root_value",
            model_differing,
            "the self-relative model value MUST move with the seat -- if it "
            "does not, this test cannot distinguish the two frames and its "
            "rollout half proves nothing",
        )

    # ------------------------------------------------------- estimand ledger

    def test_the_report_carries_the_estimand_ledger(self) -> None:
        """A blend must be reportable as a blend, never as an oracle."""
        arm = self._search(mode="rollout")
        self.assertEqual(arm["rollout_leaf_mode"], "rollout")
        self.assertEqual(arm["rollout_policy"], "uniform")
        self.assertEqual(arm["rollouts"], self.ROLLOUTS)
        self.assertEqual(arm["rollout_max_plies"], self.MAX_PLIES)
        self.assertEqual(arm["rollouts_run"], arm["leaves_priced"] * self.ROLLOUTS)
        # Terminal + fallback must partition the rollouts. If they did not, a
        # fallback share could be understated and a blend read as an oracle.
        self.assertEqual(
            arm["rollout_terminal_hits"] + arm["rollout_cap_hits"] + arm["rollout_dead_ends"],
            arm["rollouts_run"],
        )
        self.assertAlmostEqual(
            arm["rollout_terminal_fraction"] + arm["rollout_fallback_fraction"],
            1.0,
            places=5,
        )
        # At this deliberately short cap the run IS mostly fallback, and the
        # report says so. That is the property under test: the field moves with
        # the truth rather than sitting at a comfortable constant.
        self.assertGreater(arm["rollout_fallback_fraction"], 0.5)

    def test_a_longer_cap_moves_the_fallback_fraction_down(self) -> None:
        """The honesty field must be a measurement, not a decoration.

        A field that reads the same number whatever the rollouts did could not
        distinguish an oracle from the handcrafted evaluator, which is the one
        thing this arm's readers must be able to do.
        """
        short = self._search(mode="rollout", max_plies=4)
        longer = self._search(mode="rollout", max_plies=400)
        self.assertGreater(short["rollout_fallback_fraction"], longer["rollout_fallback_fraction"])
        self.assertEqual(short["rollout_terminal_hits"], 0,
                         "a 4-ply cap cannot reach a terminal from this root")
        self.assertGreater(longer["rollout_terminal_hits"], 0)

    def test_the_rollout_seed_moves_the_values_and_r_shrinks_the_spread(self) -> None:
        """The rollout RNG is wired, and R does what R is for."""
        a = self._search(mode="rollout", rollout_seed=1)
        b = self._search(mode="rollout", rollout_seed=2)
        self.assertNotEqual(a["root_value"], b["root_value"],
                            "a different rollout seed must reprice the leaves")

        def spread(rollouts: int) -> float:
            values = [
                self._search(mode="rollout", rollouts=rollouts, rollout_seed=s,
                             max_plies=400)["root_value"]
                for s in range(6)
            ]
            return max(values) - min(values)

        self.assertLess(
            spread(32), spread(2),
            "averaging more rollouts must shrink the seed-to-seed spread",
        )

    def test_thread_count_does_not_move_the_priced_values(self) -> None:
        """Threading is throughput only, on this path too."""
        one = self._search(mode="rollout", threads=1)
        many = self._search(mode="rollout", threads=4)
        self.assertEqual(
            self._differing(one, many, ignore={"rollout_threads"}),
            [],
            "the thread count must not change any priced value",
        )

    # --------------------------------------------------- boundary rejections

    def test_an_unknown_leaf_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._search(mode="oracle")
        self.assertIn("unknown rollout_leaf_mode", str(caught.exception))

    def test_zero_rollouts_is_refused(self) -> None:
        # A zero-rollout "Monte-Carlo" leaf is the handcrafted fallback wearing
        # the oracle's name in the report.
        with self.assertRaises(ValueError) as caught:
            self._search(mode="rollout", rollouts=0)
        self.assertIn("rollouts must be > 0", str(caught.exception))

    def test_zero_rollouts_is_refused_on_the_control_too(self) -> None:
        # Refused on BOTH modes, so a control run cannot be configured into a
        # state its own arm would refuse.
        with self.assertRaises(ValueError):
            self._search(mode="model_value", rollouts=0)

    def test_a_zero_ply_cap_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._search(mode="rollout", max_plies=0)
        self.assertIn("rollout_max_plies must be > 0", str(caught.exception))

    def test_zero_threads_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._search(mode="rollout", threads=0)
        self.assertIn("rollout_threads must be > 0", str(caught.exception))

    def test_an_unimplemented_rollout_policy_is_refused(self) -> None:
        # The estimand-faithful pricer is a POLICY continuation, and it is not
        # implemented in-crate. Accepting the name would price uniform play and
        # label it policy play.
        with self.assertRaises(ValueError) as caught:
            self._search(mode="rollout", policy="policy")
        self.assertIn("unknown rollout_policy", str(caught.exception))


class RolloutModelPriorsConfigTest(unittest.TestCase):
    """`EngineMctsConfig` boundary, and the call assembly behind it."""

    @staticmethod
    def _config(**overrides):
        from pokezero.engine_search import EngineMctsConfig

        base = {
            "leaf_eval": "model",
            "model_path": "/tmp/model_ts.pt",
            "checkpoint_path": "/tmp/checkpoint.pt",
            "tables_path": "/tmp/encoder_tables.json",
            "search_sims": 64,
            "search_batch": 8,
            "search_depth": 4,
            "rollout_leaf_eval": True,
            "use_opponent_priors": True,
        }
        base.update(overrides)
        return EngineMctsConfig(**base)

    def test_a_valid_arm_config_is_accepted(self) -> None:
        config = self._config()
        self.assertTrue(config.rollout_leaf_eval)
        self.assertEqual(config.leaf_eval, "model")

    def test_the_seam_is_refused_off_the_model_path(self) -> None:
        # The demonstrated failing input for the refusal that matters most:
        # silently ignoring the flag would bank a cell as "oracle-leaf with
        # model priors" having actually run the handcrafted leaf.
        for leaf_eval in ("hp_fraction", "hp_fraction_crate", "rollout_crate"):
            with self.subTest(leaf_eval=leaf_eval), self.assertRaises(ValueError) as caught:
                self._config(leaf_eval=leaf_eval)
            self.assertIn("requires leaf_eval='model'", str(caught.exception))

    def test_the_arm_is_refused_at_uniform_priors(self) -> None:
        # The demonstrated failing input for the refusal that protects this
        # path's whole reason to exist. `model_priors=False` here would price
        # rollout leaves at UNIFORM priors -- the sequential seam's estimand --
        # while carrying this path's name into the report. It would also make the
        # encode skip cover every leaf, since no leaf needs a prior map, so the
        # arm would run with no model forward at all.
        with self.assertRaises(ValueError) as caught:
            self._config(model_priors=False)
        message = str(caught.exception)
        self.assertIn("requires model_priors=True", message)
        # The refusal must NAME the alternative, or a reader whose cell genuinely
        # wants uniform priors has nowhere to go and will reach for the flag.
        self.assertIn("rollout_crate", message)

    def test_uniform_priors_is_fine_when_the_arm_is_off(self) -> None:
        config = self._config(rollout_leaf_eval=False, model_priors=False)
        self.assertFalse(config.model_priors)

    def test_zero_rollout_count_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._config(rollout_count=0)
        self.assertIn("rollout_count must be > 0", str(caught.exception))

    def test_zero_rollout_max_plies_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._config(rollout_max_plies=0)
        self.assertIn("rollout_max_plies must be > 0", str(caught.exception))

    def test_an_unimplemented_policy_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._config(rollout_policy="policy")
        self.assertIn("rollout_policy must be 'uniform'", str(caught.exception))

    def test_multithreaded_arm_requires_the_cpu_budget_ack(self) -> None:
        # The hazard is repeated on this path rather than inherited: the
        # opponent is time-budgeted and thinks concurrently with this search.
        with self.assertRaises(ValueError) as caught:
            self._config(rollout_threads=4)
        message = str(caught.exception)
        self.assertIn("rollout_threads_cpu_budget_ack=True", message)
        self.assertIn("TIME-budgeted", message)
        # And it is accepted once acknowledged, so the fence is a gate and not a
        # hard cap.
        self._config(rollout_threads=4, rollout_threads_cpu_budget_ack=True)

    def test_the_ack_is_inert_when_the_seam_is_off(self) -> None:
        config = self._config(rollout_leaf_eval=False, rollout_threads=4)
        self.assertFalse(config.rollout_leaf_eval)


class RolloutSeamCallAssemblyTest(unittest.TestCase):
    """The positional contract, which is where a silent search change hides.

    `native_search_args` appends the seam LAST, so asking for it must
    materialize every conditional slot in front of it. If it did not,
    `"rollout"` would land in `early_stop_min_sims` (truncating the budget), or
    in `use_opponent_priors` (a non-empty string is truthy, turning the opponent
    head on by accident), or in `fpu_reduction`.
    """

    #: One shared sentinel for the fold slot. Per-call `object()`s would make
    #: two assembled lists differ on identity alone, which would mask a real
    #: difference behind an artefact of the test.
    FOLD = object()

    @classmethod
    def _args(cls, **overrides) -> list:
        from pokezero.engine_search import EngineMctsConfig, native_search_args

        base = {
            "leaf_eval": "model",
            "model_path": "/tmp/model_ts.pt",
            "checkpoint_path": "/tmp/checkpoint.pt",
            "tables_path": "/tmp/encoder_tables.json",
            "search_sims": 64,
            "search_batch": 8,
            "search_depth": 4,
        }
        base.update(overrides)
        config = EngineMctsConfig(**base)
        record = {
            "state_str": "state",
            "ctx_json": "{}",
            "seed": 11,
            "side_key": "side_one",
        }
        return native_search_args(
            config,
            record,
            tables_json="tables",
            root_inputs="root_inputs",
            rust_fold=cls.FOLD,
            early_stop_min_sims=0,
        )

    def test_the_seam_off_makes_the_pre_seam_call_byte_for_byte(self) -> None:
        self.assertEqual(len(self._args()), 12)

    def test_the_seam_materializes_every_slot_in_front_of_it(self) -> None:
        args = self._args(rollout_leaf_eval=True)
        # 12 leading + early-stop pair + opponent flag + fpu + arm_priors + 7.
        self.assertEqual(len(args), 24)
        self.assertEqual(args[12], 0, "early_stop_min_sims, not the mode string")
        self.assertIs(args[13], True, "early_stop_side_one for a side_one record")
        self.assertIs(args[14], False, "the opponent flag keeps its config value")
        self.assertIsNone(args[15], "fpu_reduction is its own default")
        self.assertIs(args[16], False, "arm_priors keeps its config value, not True")
        self.assertEqual(args[17], "rollout")

    def test_the_seam_does_not_disturb_the_knobs_in_front_of_it(self) -> None:
        with_seam = self._args(
            rollout_leaf_eval=True, use_opponent_priors=True, fpu_reduction=0.3
        )
        without = self._args(use_opponent_priors=True, fpu_reduction=0.3)
        self.assertEqual(with_seam[: len(without)], without)

    def test_the_arm_priors_slot_is_not_forced_true(self) -> None:
        # A regression this assembly nearly shipped: materializing `arm_priors`
        # to reach the slot behind it by writing an unconditional True would
        # switch the arm-name telemetry column on for every rollout cell and
        # change the report a shard is compared against.
        args = self._args(rollout_leaf_eval=True)
        self.assertIs(args[16], False)
        args = self._args(rollout_leaf_eval=True, override_telemetry=True)
        self.assertIs(args[16], True)

    def test_the_seam_knobs_are_passed_in_the_declared_order(self) -> None:
        args = self._args(
            rollout_leaf_eval=True,
            rollout_count=16,
            rollout_max_plies=250,
            rollout_seed=99,
        )
        self.assertEqual(
            args[17:],
            ["rollout", 16, 250, "uniform", 99, 1, False],
        )


# ---------------------------------------------------------------------------
# The witness on the SHIPPING path
# ---------------------------------------------------------------------------
#
# Everything above this line certifies the crate. None of it certified the
# PYTHON BOUNDARY on the path that actually runs the arm.
#
# `_search_rollout_crate` -- the sequential, uniform-priors seam -- read the
# crate's whole rollout ledger. `_search_model`, the path that runs with priors ON
# and therefore the path the arbiter uses, read NONE of the sixteen rollout keys
# the crate emits, and emitted `leaf_eval: "model"` with no rollout column at all.
# An arm decision banked as an ordinary model decision, indistinguishable from the
# raw arm's, with `rollout_fallback_fraction` -- the field that says how much of
# the "oracle" was the handcrafted HP-fraction evaluator -- unavailable exactly
# where the arm runs. At this file's own fixture config that fraction measures
# 0.96503.
#
# THAT FIGURE IS THE FIXTURE'S, NOT THE ARM'S, and the distinction is load-bearing
# because it is a claim about how honest the arm is. 0.96503 is measured at THIS
# FILE'S 40-PLY CAP (`MAX_PLIES = 40`), which is a fixture chosen to make rollouts
# cheap. The arm as LAUNCHED runs the default 200-ply cap, and its own banked
# whole-game run measures `rollout_fallback_fraction = 0.018139547833834504` -- a
# 98.2% oracle. The measured curve over the cap is 40 -> 0.965, 50 -> 0.8678,
# 200 -> 0.0 on the search-only fixture, so `> 0.9` remains the right assertion HERE
# and any prose saying the arm is 96.5% handcrafted is wrong.
#
# THE METHODOLOGICAL POINT, because a mutation-heavy round walked straight past
# this: ABSENCE OF CODE IS NOT MUTATION-TESTABLE. A battery finds code that is
# broken; it can never find code that was never written, so a seam that emitted
# nothing had nothing to mutate and survived intact. The question that finds the
# class is asked of each claim separately -- "is there code that would have to
# exist for this to be true, and does it exist on the path that runs?" -- and the
# durable answer is a runtime requirement, which is what
# `require_rollout_leaf_witness` is.


class _ShippingPathHarness:
    """Drive the REAL `_search_model` over a fake native, as the arm's path.

    A mixin rather than a base `TestCase` so unittest does not collect it as an
    extra empty class, and so the crate-backed subclass below can add its own skip
    decorator (decorators do not inherit).

    The native is fake and the PLUMBING IS REAL: `native_search_args`,
    `run_world`'s absorption, the witness builder and the guard are all
    production's. That is the correct division here, because the defect being
    fixed was entirely on the Python side of the boundary -- the crate was already
    emitting every field.
    """

    #: The crate's rollout columns for a world, as `search_batched_multi_encoded`
    #: emits them. Overridden by the crate-backed subclass with a REAL report.
    ROLLOUT_COLUMNS = {
        "rollout_leaf_mode": "rollout",
        "rollout_encode_skipped": 66,
        "leaves_priced": 168,
        "rollouts_run": 1344,
        "rollout_plies": 53541,
        "rollout_terminal_hits": 47,
        "rollout_cap_hits": 1297,
        "rollout_dead_ends": 0,
    }

    class _Native:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def search_batched_multi_encoded(self, *args):
            self.calls.append(args)
            response = self.responses.pop(0)
            return json.dumps(response)

    def _report(self, *, rollout: bool) -> dict:
        report = {
            "iterations": 32,
            "requested_iterations": 32,
            "remaining_iterations": 0,
            "early_stopped": False,
            "model_evals": 103,
            "lossy_renders": 0,
            "attribution_unsafe_renders": 0,
            "prior_fallbacks": 0,
            "side_one": [
                {"move": "alpha", "visits": 24, "q": 0.5},
                {"move": "beta", "visits": 8, "q": 0.4},
            ],
        }
        if rollout:
            report.update(copy.deepcopy(self.ROLLOUT_COLUMNS))
        return report

    @staticmethod
    def _context():
        observation = SimpleNamespace(
            legal_action_mask=(True, True, False, False),
            metadata={
                "action_candidates": [
                    {"action_index": 0, "kind": "move", "legal": True, "move_id": "alpha"},
                    {"action_index": 1, "kind": "move", "legal": True, "move_id": "beta"},
                ]
            },
        )
        return SimpleNamespace(
            observation=observation,
            public_materialization_state=SimpleNamespace(
                replay=SimpleNamespace(turn_number=1)
            ),
            player_id="p1",
            battle_id="rollout-witness",
            decision_round_index=0,
        )

    @staticmethod
    def _world(label: str):
        return (
            SimpleNamespace(
                party_species={"p1": ("rattata",), "p2": ("chansey",)},
                slot_sides={"p1": "side_one"},
            ),
            SimpleNamespace(to_string=lambda: label),
        )

    def _policy(self, *, arm: bool):
        """`arm=True` is the rollout arm; `arm=False` is the raw arm it is paired
        against -- the same model search with production's leaf value."""
        policy = object.__new__(EngineMctsPolicy)
        policy.policy_id = "rollout-witness"
        policy._config = EngineMctsConfig(
            worlds=2,
            leaf_eval="model",
            model_path="model.pt",
            checkpoint_path="checkpoint.pt",
            tables_path="tables.json",
            search_sims=32,
            search_batch=8,
            model_priors=arm,
            rollout_leaf_eval=arm,
            rollout_count=8,
            rollout_max_plies=40,
        )
        policy._tables_json = "{}"
        policy.stats = EngineMctsStats()
        policy._world_failures_before = {}
        return policy

    def _run(self, policy, native, worlds):
        fake_module = SimpleNamespace(
            FoldState=SimpleNamespace(from_payload=lambda _payload: object())
        )
        live_fold = SimpleNamespace(to_payload=lambda: {})
        with (
            patch.dict(sys.modules, {"pokezero_search": fake_module}),
            patch.object(EngineMctsPolicy, "_native", return_value=native),
            patch.object(
                EngineMctsPolicy, "_validate_model_root_observation", return_value=None
            ),
            patch.object(EngineMctsPolicy, "_root_inputs_json", return_value="{}"),
        ):
            return policy._search_model(
                self._context(), worlds, live_fold, random.Random(7)
            )

    def _decide(self, *, arm: bool, worlds: int = 1, rollout: bool | None = None):
        policy = self._policy(arm=arm)
        native = self._Native(
            [self._report(rollout=arm if rollout is None else rollout)] * worlds
        )
        decision = self._run(policy, native, [self._world(f"w{i}") for i in range(worlds)])
        return policy, decision


class RolloutLeafWitnessTravelsOnTheModelPathTest(_ShippingPathHarness, unittest.TestCase):
    """The witness must reach the decision `_search_model` returns.

    No crate needed: the defect and its fix are both on the Python side of the
    boundary, so these run in any checkout. The crate-backed class below re-runs
    the central one against the crate's OWN measured ledger.
    """

    def test_an_arm_decision_carries_the_witness(self) -> None:
        """THE FIX. Before it, this dict did not exist."""
        _, decision = self._decide(arm=True)
        engine = decision.metadata["engine_mcts"]
        witness = engine["rollout_leaf"]
        for field in ROLLOUT_LEAF_WITNESS_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, witness)
        self.assertEqual(witness["rollout_leaf_modes"], {"rollout": 1})
        self.assertEqual(witness["rollouts_run"], 1344)
        self.assertEqual(witness["leaves_priced"], 168)
        self.assertEqual(witness["rollout_terminal_hits"], 47)
        self.assertEqual(witness["rollout_cap_hits"], 1297)
        self.assertEqual(witness["rollout_dead_ends"], 0)
        self.assertAlmostEqual(witness["rollout_fallback_fraction"], 1297 / 1344, places=12)

    def test_the_fallback_fraction_is_available_where_the_arm_runs(self) -> None:
        """The measurement that made this urgent.

        A mostly-handcrafted evaluator must not be able to bank as an oracle. The
        assertion is on the VALUE, not on the key's presence: a column that exists
        and reads 0 would be worse than a missing one.

        0.96503 is this FIXTURE's number, at its 40-ply cap. The arm's own banked
        whole-game run at the launched 200-ply cap reads 0.018139547833834504.
        """
        policy, decision = self._decide(arm=True)
        witness = decision.metadata["engine_mcts"]["rollout_leaf"]
        self.assertAlmostEqual(witness["rollout_fallback_fraction"], 0.96503, places=5)
        self.assertAlmostEqual(witness["rollout_terminal_fraction"], 0.03497, places=5)
        # And on the SHARD, which is where a campaign sorts cells.
        shard = policy.stats.to_dict()
        self.assertAlmostEqual(shard["rollout_fallback_fraction"], 0.96503, places=5)
        self.assertEqual(shard["rollout_leaf_modes"], {"rollout": 1})

    def test_the_encode_skip_count_reaches_the_decision_and_the_shard(self) -> None:
        """A2. `rollout_encode_skipped` was HALF-CLOSED, and this is the closure.

        Its per-decision value died only under a `skipUnless(_crate_ready)` subtest,
        so it was unguarded wherever the crate is absent -- which is most
        environments and all of CI's non-crate jobs. This class runs with no crate,
        so hard-zeroing the per-decision value or the shard column reads False HERE.

        Asserted on the VALUE and on the SUM over worlds, not on key presence: the
        two surviving mutants were `-> 0` and `-> {}`, both of which keep the key.
        """
        policy, decision = self._decide(arm=True, worlds=2)
        witness = decision.metadata["engine_mcts"]["rollout_leaf"]
        self.assertEqual(witness["rollout_encode_skipped"], 2 * 66)
        shard = policy.stats.to_dict()
        self.assertEqual(shard["rollout_encode_skipped"], 2 * 66)
        # And the accounting identity the field exists FOR, on the shard's own
        # numbers: forwards + skips == leaves + the root forward per world.
        self.assertEqual(
            shard["model_evals"] + shard["rollout_encode_skipped"],
            shard["rollout_leaves_priced"] + 2,
        )

    def test_a_collapsed_draw_counts_the_worlds_it_represents(self) -> None:
        """A6's accumulation rule, and the only place it is observable.

        Two identical belief draws share ONE crate search (the draw cache collapses
        them), so `+= 1` per report would record one world where two were
        represented. `+= weight` records two. This is why v2's accumulation is the
        more informative one -- and it is the mutant a two-distinct-world test cannot
        kill, because every weight there is 1.

        AND THE UNITS ARE NAMED, because they are not the same across the block:
        `rollout_leaf_worlds` is per WORLD (weighted), while the cost ledger --
        `rollouts_run`, `rollout_plies`, `leaves_priced`, `model_evals` -- is per
        INVOCATION and is NOT weighted, because it measures compute that happened
        once. So `rollouts_run / rollout_leaf_worlds` is not rollouts-per-world on a
        run with collapses. Asserted here so a reader finds the asymmetry stated
        rather than deriving a wrong ratio from it.
        """
        policy = self._policy(arm=True)
        native = self._Native([self._report(rollout=True)])
        decision = self._run(policy, native, [self._world("same"), self._world("same")])
        self.assertEqual(len(native.calls), 1, "the draws must actually collapse")
        self.assertEqual(policy.stats.worlds_collapsed, 1)
        witness = decision.metadata["engine_mcts"]["rollout_leaf"]
        self.assertEqual(witness["rollout_leaf_worlds"], 2)
        self.assertEqual(policy.stats.to_dict()["rollout_leaf_worlds"], 2)
        # The unweighted half of the same block, pinned as the asymmetry it is.
        self.assertEqual(witness["rollouts_run"], 1344)
        self.assertEqual(witness["leaves_priced"], 168)

    def test_a_report_without_the_encode_skip_count_refuses_the_run(self) -> None:
        """The report-side half: dropping the key from the crate's report must
        refuse, not default to a zero that reads as "the skip never fired"."""
        policy = self._policy(arm=True)
        report = self._report(rollout=True)
        del report["rollout_encode_skipped"]
        native = self._Native([report])
        with self.assertRaises(EngineSearchWitnessError) as caught:
            self._run(policy, native, [self._world("w0")])
        self.assertIn("rollout_encode_skipped", str(caught.exception))

    def test_a_missing_witness_refuses_the_run(self) -> None:
        """THE DEMONSTRATED FAILING INPUT, positive direction.

        The shipped shape: the arm runs and the decision says nothing about it.
        Reproduced by deleting the builder's output -- i.e. by putting the code
        back the way it was -- which must now raise rather than bank.
        """
        policy = self._policy(arm=True)
        native = self._Native([self._report(rollout=True)])
        with patch.object(EngineMctsPolicy, "_rollout_leaf_witness", return_value=None):
            with self.assertRaises(EngineSearchWitnessError) as caught:
                self._run(policy, native, [self._world("w0")])
        self.assertIn("carries NO rollout witness", str(caught.exception))

    def test_a_partial_witness_refuses_the_run(self) -> None:
        """Deleting ONE field must fail too, or the guard only pins the block's
        existence and every column inside it is deletable."""
        for field in ROLLOUT_LEAF_WITNESS_FIELDS:
            with self.subTest(dropped=field):
                policy = self._policy(arm=True)
                native = self._Native([self._report(rollout=True)])
                real = EngineMctsPolicy._rollout_leaf_witness

                def _short(self_, modes, ledger, _field=field):
                    witness = real(self_, modes, ledger)
                    del witness[_field]
                    return witness

                with patch.object(EngineMctsPolicy, "_rollout_leaf_witness", _short):
                    with self.assertRaises(EngineSearchWitnessError) as caught:
                        self._run(policy, native, [self._world("w0")])
                self.assertIn(field, str(caught.exception))

    def test_a_raw_arm_decision_carries_no_witness(self) -> None:
        """THE DEMONSTRATED FAILING INPUT, negative direction.

        The raw arm is the DENOMINATOR of every paired delta. `--arm raw
        --engine-rollout-leaf` writing `"rollout_leaf": true` onto a raw row was a
        real false witness, so "no witness unless the arm ran" is asserted as
        loudly as its converse.
        """
        _, decision = self._decide(arm=False)
        self.assertNotIn("rollout_leaf", decision.metadata["engine_mcts"])
        self.assertEqual(decision.metadata["engine_mcts"]["leaf_eval"], "model")

    def test_a_false_witness_on_a_raw_arm_decision_refuses(self) -> None:
        """Forged the way the live defect forged it: a bare `"rollout_leaf": true`
        stamped onto a row whose search never ran a rollout."""
        metadata = {"engine_mcts": {"leaf_eval": "model", "rollout_leaf": True}}
        with self.assertRaises(EngineSearchWitnessError) as caught:
            require_rollout_leaf_witness(metadata, rollout_leaf_eval=False)
        self.assertIn("FALSE WITNESS", str(caught.exception))

    def test_a_raw_arm_shard_gains_no_rollout_columns_at_all(self) -> None:
        """Flag-off must be the strong form: not "identical after dropping new
        fields", but a payload with no new fields in it."""
        policy, _ = self._decide(arm=False)
        shard = policy.stats.to_dict()
        rollout_keys = sorted(k for k in shard if "rollout" in k)
        self.assertEqual(rollout_keys, [])

    def test_the_arm_refuses_a_report_that_priced_no_rollouts(self) -> None:
        """The config asked for the arm and the crate did not engage the seam.

        This is the disaster the whole witness exists for: the leaves were priced
        by the MODEL, so every rollout number read off that search would be a
        number about a search that ran no rollouts. It must refuse rather than
        default the missing counters to zero -- a zeroed fallback fraction reads as
        "pure oracle", which is the most flattering possible misreading.
        """
        policy = self._policy(arm=True)
        native = self._Native([self._report(rollout=False)])
        with self.assertRaises(EngineSearchWitnessError) as caught:
            self._run(policy, native, [self._world("w0")])
        message = str(caught.exception)
        self.assertIn("no rollout_leaf_mode", message)
        self.assertIn("priced by the", message)

    def test_a_pricer_the_config_cannot_ask_for_refuses(self) -> None:
        """`model_value` is the fidelity CONTROL: production's leaf value through
        the arm's plumbing. A shipped decision priced that way ran the raw arm."""
        policy = self._policy(arm=True)
        report = self._report(rollout=True)
        report["rollout_leaf_mode"] = "model_value"
        native = self._Native([report])
        with self.assertRaises(EngineSearchWitnessError) as caught:
            self._run(policy, native, [self._world("w0")])
        self.assertIn("model_value", str(caught.exception))

    def test_the_witness_accumulates_over_every_world(self) -> None:
        """Per INVOCATION, like `model_evals`: two worlds priced twice as many
        leaves, and the witness must say so rather than reporting one world's."""
        _, decision = self._decide(arm=True, worlds=2)
        witness = decision.metadata["engine_mcts"]["rollout_leaf"]
        self.assertEqual(witness["rollout_leaf_worlds"], 2)
        self.assertEqual(witness["rollouts_run"], 2 * 1344)
        self.assertEqual(witness["leaves_priced"], 2 * 168)
        self.assertEqual(witness["rollout_leaf_modes"], {"rollout": 2})
        # The fraction is scale-free, so pooling two worlds must not move it.
        self.assertAlmostEqual(
            witness["rollout_fallback_fraction"], 1297 / 1344, places=12
        )


class RolloutLeafWitnessGuardTest(unittest.TestCase):
    """`require_rollout_leaf_witness`'s own failing inputs.

    Each constructs the offending metadata directly, one defect per case: a guard
    whose cases change two things at once pins neither, because the first check to
    fire short-circuits the rest.
    """

    @staticmethod
    def _witness(**overrides) -> dict:
        witness = {
            "rollout_leaf_modes": {"rollout": 1},
            "rollout_leaf_worlds": 1,
            "leaves_priced": 168,
            "rollouts_run": 1344,
            "rollout_plies": 53541,
            "rollout_encode_skipped": 66,
            "rollout_terminal_hits": 47,
            "rollout_cap_hits": 1297,
            "rollout_dead_ends": 0,
            "rollout_terminal_fraction": 47 / 1344,
            "rollout_fallback_fraction": 1297 / 1344,
            "rollout_mean_plies": 53541 / 1344,
        }
        witness.update(overrides)
        return witness

    def _refuse(self, witness, *, expect: str) -> None:
        metadata = {"engine_mcts": {"leaf_eval": "model", "rollout_leaf": witness}}
        with self.assertRaises(EngineSearchWitnessError) as caught:
            require_rollout_leaf_witness(metadata, rollout_leaf_eval=True)
        self.assertIn(expect, str(caught.exception))

    def test_a_complete_witness_is_accepted(self) -> None:
        """The guard's discriminating power: it must pass the good input, or every
        refusal below is satisfied by a check that refuses everything."""
        require_rollout_leaf_witness(
            {"engine_mcts": {"leaf_eval": "model", "rollout_leaf": self._witness()}},
            rollout_leaf_eval=True,
        )

    def test_the_DEAD_END_term_of_the_fallback_numerator_is_pinned(self) -> None:
        """B4. `expected = (cap + dead) -> cap` SURVIVED in this guard.

        Not because the check is wrong but because every fixture in the file had
        `rollout_dead_ends == 0`, where the two rules give the same number -- so the
        `dead` term was asserted by nothing. That is the same "a term that cannot read
        non-zero is not a measurement" objection the sibling branch raised against its
        own cap-only fraction, and its resolution was "#1271 makes the term testable at
        the writer instead". That resolution HOLDS AT TWO OF THREE SITES: the shard
        reader and the shard writer both have a non-zero-dead fixture. This is the
        third -- the per-decision witness guard -- and it did not.

        The fixture sets `dead > 0` and keeps the partition balanced, so the ONLY thing
        that can fail is the numerator rule. A cap-only expectation reads False here.
        """
        # dead > 0, partition balanced, fraction computed over the WHOLE fallback.
        run, terminal, cap, dead = 1344, 40, 1297, 7
        self.assertEqual(terminal + cap + dead, run)
        honest = self._witness(
            rollouts_run=run,
            rollout_terminal_hits=terminal,
            rollout_cap_hits=cap,
            rollout_dead_ends=dead,
            rollout_terminal_fraction=terminal / run,
            rollout_fallback_fraction=(cap + dead) / run,
        )
        require_rollout_leaf_witness(
            {"engine_mcts": {"leaf_eval": "model", "rollout_leaf": honest}},
            rollout_leaf_eval=True,
        )
        # THE v1 NUMERATOR on the same counts: `cap` alone. This is what a shard
        # written by the cap-only rule carries, and it must read False.
        self.assertNotAlmostEqual(cap / run, (cap + dead) / run, places=9)
        self._refuse(
            self._witness(
                rollouts_run=run,
                rollout_terminal_hits=terminal,
                rollout_cap_hits=cap,
                rollout_dead_ends=dead,
                rollout_terminal_fraction=terminal / run,
                rollout_fallback_fraction=cap / run,
            ),
            expect="is not the quotient of its own partition",
        )

    def test_the_witnesss_WEIGHTED_tally_must_sum_to_its_world_count(self) -> None:
        """B4, the per-decision half of the `+= weight` invariant.

        `rollout_modes[mode] += weight` and `rollout_ledger["worlds"] += weight` are
        one datum written twice on one line pair, so their sum is an identity. Without
        it, unweighting the Counter is invisible on BOTH surfaces at once, because the
        modes mapping is otherwise read only for its keys.
        """
        require_rollout_leaf_witness(
            {
                "engine_mcts": {
                    "leaf_eval": "model",
                    "rollout_leaf": self._witness(
                        rollout_leaf_modes={"rollout": 2}, rollout_leaf_worlds=2
                    ),
                }
            },
            rollout_leaf_eval=True,
        )
        self._refuse(
            self._witness(rollout_leaf_modes={"rollout": 1}, rollout_leaf_worlds=2),
            expect="does not sum to its world count",
        )

    def test_an_absent_witness_refuses(self) -> None:
        with self.assertRaises(EngineSearchWitnessError):
            require_rollout_leaf_witness(
                {"engine_mcts": {"leaf_eval": "model"}}, rollout_leaf_eval=True
            )

    def test_an_empty_mode_map_refuses(self) -> None:
        """A witness that names no pricer cannot say the rollout pricer ran."""
        self._refuse(self._witness(rollout_leaf_modes={}), expect="names no pricer")

    def test_a_zeroed_ledger_refuses(self) -> None:
        """The stub shape. Present keys are not a witness: a block of zeros says
        nothing was priced, which is a refusal rather than a decision."""
        self._refuse(
            self._witness(
                rollouts_run=0,
                rollout_terminal_hits=0,
                rollout_cap_hits=0,
                rollout_dead_ends=0,
                rollout_fallback_fraction=0.0,
            ),
            expect="degenerate",
        )

    def test_zero_leaves_priced_refuses(self) -> None:
        self._refuse(self._witness(leaves_priced=0), expect="leaves_priced=0")

    def test_an_unbalanced_partition_refuses(self) -> None:
        """terminal + cap + dead_ends must be every rollout. Otherwise the
        fallback share is read off a denominator that is not the population."""
        self._refuse(self._witness(rollout_terminal_hits=46), expect="every rollout")

    def test_a_fabricated_fallback_fraction_refuses(self) -> None:
        """The flattering-direction defect, and the one a reader cannot see: a
        fallback column that does not follow from the counts printed beside it.

        0.0 specifically, because that is the value that reads as "pure oracle".
        """
        self._refuse(
            self._witness(rollout_fallback_fraction=0.0), expect="not the quotient"
        )

    def test_the_partition_and_the_fraction_are_checked_independently(self) -> None:
        """ONE PERTURBED FIELD PER CASE, and each must trip only its own check.

        A case that moved the counts AND the fraction together would pin neither:
        whichever check fires first short-circuits the other, so the second could be
        deleted and the case would still pass. So each perturbation is checked to
        trip its own check and, by the message, NOT the other one.
        """
        # Counts unbalanced, fraction still consistent with (cap + dead) / run --
        # so only the partition check can fire.
        metadata = {
            "engine_mcts": {"rollout_leaf": self._witness(rollout_terminal_hits=46)}
        }
        with self.assertRaises(EngineSearchWitnessError) as caught:
            require_rollout_leaf_witness(metadata, rollout_leaf_eval=True)
        self.assertIn("every rollout", str(caught.exception))
        self.assertNotIn("quotient", str(caught.exception))
        # Counts balanced, fraction wrong -- so only the quotient check can fire.
        metadata = {
            "engine_mcts": {"rollout_leaf": self._witness(rollout_fallback_fraction=0.5)}
        }
        with self.assertRaises(EngineSearchWitnessError) as caught:
            require_rollout_leaf_witness(metadata, rollout_leaf_eval=True)
        self.assertIn("quotient", str(caught.exception))
        self.assertNotIn("every rollout", str(caught.exception))

    def test_the_required_field_list_is_pinned_by_literal(self) -> None:
        """The classification IS the rule.

        A guard that iterated "whatever the builder produced" could not read False
        when the builder stopped producing something -- the same self-reference
        that let a dropped `LADDER_PER_DECISION_CLAIMS` entry survive a whole
        suite. So membership is a literal, and shrinking it fails HERE.
        """
        self.assertEqual(
            ROLLOUT_LEAF_WITNESS_FIELDS,
            (
                "rollout_leaf_modes",
                "leaves_priced",
                "rollouts_run",
                "rollout_terminal_hits",
                "rollout_cap_hits",
                "rollout_dead_ends",
                "rollout_encode_skipped",
                "rollout_fallback_fraction",
            ),
            "every name here is a field a decision must carry to be allowed to "
            "claim the rollout arm; removing one is a decision about the rule",
        )

    def test_the_crate_report_field_map_is_pinned_by_literal(self) -> None:
        """A2. The required-key list and the accumulation were TWO literals.

        Dropping `rollout_encode_skipped` from the report's required-key list
        SURVIVED, because nothing pinned the list and the sibling copy that fed the
        accumulation was a different literal -- so neither copy's shrinkage was
        visible from the other. They are now one mapping, pinned here, and the
        values are the shard counters each key feeds so a rewire is visible too.
        """
        self.assertEqual(
            dict(ROLLOUT_LEAF_REPORT_FIELDS),
            {
                "rollout_leaf_mode": None,
                "leaves_priced": "rollout_leaves_priced",
                "rollouts_run": "rollouts_run",
                "rollout_plies": "rollout_plies",
                "rollout_terminal_hits": "rollout_terminal_hits",
                "rollout_cap_hits": "rollout_cap_hits",
                "rollout_dead_ends": "rollout_dead_ends",
                "rollout_encode_skipped": "rollout_encode_skipped",
            },
        )

    def test_an_encode_skip_count_above_the_leaves_priced_refuses(self) -> None:
        """A2, the other half: the field must be able to be WRONG.

        Every skip is a leaf whose forward was not needed, so the skips are a subset
        of the priced leaves. `> 0` would be the wrong check -- false at a depth
        where every leaf hosts a child decision -- so the check is the range.
        """
        self._refuse(
            self._witness(rollout_encode_skipped=169), expect="rollout_encode_skipped"
        )
        self._refuse(
            self._witness(rollout_encode_skipped=-1), expect="rollout_encode_skipped"
        )

    def test_a_witness_without_the_encode_skip_count_refuses(self) -> None:
        witness = self._witness()
        del witness["rollout_encode_skipped"]
        self._refuse(witness, expect="missing rollout_encode_skipped")

    def test_the_raw_arm_direction_accepts_an_absent_witness(self) -> None:
        require_rollout_leaf_witness(
            {"engine_mcts": {"leaf_eval": "model"}}, rollout_leaf_eval=False
        )

    def test_the_raw_arm_direction_refuses_any_witness(self) -> None:
        for forged in (True, {}, self._witness()):
            with self.subTest(forged=type(forged).__name__):
                metadata = {"engine_mcts": {"rollout_leaf": forged}}
                with self.assertRaises(EngineSearchWitnessError):
                    require_rollout_leaf_witness(metadata, rollout_leaf_eval=False)


class TheRetractedCostRatioStaysRetractedTest(unittest.TestCase):
    """`engine_search.py` must not both assert and retract the same figure.

    It did. The `rollout_threads` fence retracted "3-4 orders of magnitude" BY
    MEASUREMENT, and seventy lines later the `rollout_leaf_eval` fence this branch
    adds re-asserted it as fact -- so the file contradicted itself and a reader
    landing on either comment had no way to tell which was live. A retraction that
    a later paste can silently undo is not a retraction.

    SCOPED BY LOCATION, not by phrasing. Asserting "the string appears once" would
    be satisfied by the guard's own literal and would say nothing about the fence
    that carried the claim, so the fence's own source is extracted and checked. The
    surviving mention is REQUIRED to be inside the retraction, which is what keeps
    this from being deletable by simply removing both.
    """

    @staticmethod
    def _source() -> str:
        import inspect

        from pokezero import engine_search

        return inspect.getsource(engine_search)

    RETRACTED = "orders of magnitude"

    def test_the_arm_fence_does_not_re_assert_the_retracted_figure(self) -> None:
        """The fence that carried the claim, read on its own."""
        source = self._source()
        marker = "rollout_threads_cpu_budget_ack=True. The paired-eval opponent is"
        # The arm's ack fence is the SECOND occurrence of the shared message; the
        # first belongs to the sequential `leaf_eval="rollout_crate"` path.
        first = source.index(marker)
        second = source.index(marker, first + 1)
        # Everything from the previous blank-comment boundary up to the raise.
        block = source[source.rindex("if self.rollout_threads > 1", 0, second) : second]
        self.assertNotIn(
            self.RETRACTED,
            block,
            "the rollout-arm CPU fence must not restate a figure the sibling fence "
            "retracts by measurement; the file would then assert and deny the same "
            "number in two places",
        )

    def test_the_only_surviving_mention_is_the_retraction_itself(self) -> None:
        source = self._source()
        occurrences = source.count(self.RETRACTED)
        self.assertEqual(
            occurrences,
            1,
            f"{self.RETRACTED!r} appears {occurrences} times; exactly one mention is "
            "allowed and it is the one that retracts the figure",
        )
        index = source.index(self.RETRACTED)
        window = source[max(0, index - 400) : index + 400]
        self.assertIn(
            "asserted",
            window,
            "the surviving mention must be the RETRACTION -- a bare restatement "
            "passes a mere occurrence count",
        )
        self.assertIn("not what the arm costs", window)

    def test_no_unverifiable_point_ratio_is_asserted_as_the_arms_cost(self) -> None:
        """SCOPED BY VALUE-CLASS, not by enumerated literals.

        A5. The previous form enumerated the two contested measurements
        (`"1.8x at R=8"`, `"1.25-3.98x"`) and checked each. It read False on five of
        six attacks -- and a THIRD, unlisted ratio asserted as the arm's cost
        SURVIVED, because a guard that names its own inputs can only refuse the
        shapes someone already thought of. The failure mode is "a ratio a reader
        cannot check", which is a CLASS, so the check is now on the class: every
        ratio-shaped literal in the fence must sit inside the paragraph that
        explains why no point ratio is quoted.
        """
        source = self._source()
        stray = _stray_cost_ratios(source)
        self.assertEqual(
            stray,
            [],
            "the CPU fence quotes ratio-shaped figures outside the paragraph that "
            f"says why no point ratio is quoted: {stray}. Both derivations need the "
            "pinned panel checkpoint and its CPU export, which are not in the "
            "repository, so whichever number is written down is the one a later "
            "reader believes",
        )
        window = _cpu_fence_window(source)
        self.assertIn("SINGLE DIGITS", window)
        # DISCRIMINATING POWER. The regex must actually be finding the ratios that
        # ARE there (inside the sanctioned paragraph) -- otherwise "no strays" is
        # satisfied by a pattern that matches nothing.
        self.assertGreaterEqual(
            len(_COST_RATIO.findall(window)),
            3,
            "the value-class pattern must match the ratios the fence legitimately "
            "quotes inside its retraction, or the emptiness above proves nothing",
        )

    def test_the_value_class_guard_reads_false_on_a_third_shape(self) -> None:
        """THE ATTACK THAT SURVIVED, plus the two the old form already caught.

        Applied to synthetic sources rather than to the tree, so all three shapes
        can be shown against the same checker. The third is the one that matters:
        a ratio nobody enumerated, asserted as the arm's cost, outside the
        paragraph.
        """
        source = self._source()
        window = _cpu_fence_window(source)
        anchor = "    # NO POINT RATIO, because"
        self.assertIn(anchor, source)
        for label, injected in (
            ("shape 1 -- an enumerated literal, restated as fact",
             "    # The arm costs 1.8x at R=8 against the raw arm.\n"),
            ("shape 2 -- the other enumerated literal",
             "    # The arm/raw wall ratio is 1.25-3.98x.\n"),
            ("shape 3 -- UNLISTED, and the one that survived",
             "    # Measured: the arm costs 2.7x the raw arm on one core.\n"),
            # B5. THE FOUR SPELLINGS THAT EVADED THE ASCII-ONLY CHARACTER CLASS. The
            # first of them is not hypothetical: it is how the headline this guard
            # exists to protect is actually written in the PR body the figure came
            # from -- `7.7<U+00D7>-16.9<U+00D7>` with MULTIPLICATION SIGNS. A guard
            # that matches `x` and not `×` protected nothing it was built for.
            ("shape 4 -- U+00D7 MULTIPLICATION SIGN, the headline's own spelling",
             "    # The arm costs 7.7× the raw arm at R=8.\n"),
            ("shape 5 -- uppercase X",
             "    # The arm costs 7.7X the raw arm at R=8.\n"),
            ("shape 6 -- the word form",
             "    # The arm's cost is 7.7-fold the raw arm's at R=8.\n"),
            ("shape 7 -- the prose form, no symbol at all",
             "    # The arm costs a factor of 7.7 more than the raw arm.\n"),
        ):
            with self.subTest(shape=label):
                mutant = source.replace(anchor, injected + anchor, 1)
                self.assertNotEqual(mutant, source, "the injection must apply")
                self.assertNotEqual(
                    _stray_cost_ratios(mutant),
                    [],
                    "a ratio asserted as the arm's cost outside the retraction's own "
                    "paragraph must be found, whether or not anyone enumerated it",
                )
        # And the real fence, unmutated, is clean -- so the pattern is not simply
        # refusing every source it is handed.
        self.assertEqual(_stray_cost_ratios(source), [])
        self.assertIn("1.8x at R=8", window)

    def test_the_character_class_is_not_ASCII_x_only(self) -> None:
        """B5, applied to the CHECKER rather than to the tree.

        The evasions above are shown against the real fence by injection. This shows
        them against the pattern directly, together with the negatives -- because a
        pattern widened until it matches everything refuses every source and proves
        nothing. `x86`, a hex literal and the word `matrix` must all stay clean.
        """
        for spelling in ("7.7x", "7.7X", "7.7×", "7.7✕", "7.7-fold",
                         "7.7 fold", "a factor of 7.7", "1.25-3.98x",
                         "7.7×–16.9×"):
            with self.subTest(matches=spelling):
                self.assertTrue(
                    _COST_RATIO.search(spelling),
                    f"{spelling!r} is a ratio-shaped literal and must be found",
                )
        for benign in ("x86_64", "0x1f", "matrix", "4 examples", "exit code 1",
                       "xfail", "index"):
            with self.subTest(clean=benign):
                self.assertIsNone(
                    _COST_RATIO.search(benign),
                    f"{benign!r} is not a ratio and must not be flagged",
                )

    def test_the_retraction_scope_is_not_one_comment_block(self) -> None:
        """B5. Twenty-one lines of one `.py` file, and no docs at all.

        `_cpu_fence_window` slices 1400 characters out of `engine_search.py`. The
        retracted figure is PROSE -- it was published in a PR body and it is restated
        in `docs/` -- so a guard scoped to one comment block cannot see the places a
        reader is most likely to find it. The tracked prose that discusses this arm's
        CPU cost is now in scope, and the retraction's own wording is required to be
        present in the module so "the scope is wider" is not satisfied by widening it
        to files that say nothing.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        window = _cpu_fence_window(self._source())
        self.assertLess(
            len(window.splitlines()),
            40,
            "the fence window is deliberately small; the point is that it is not the "
            "whole scope",
        )
        checked = 0
        for relative in _RETRACTION_SCOPE_DOCS:
            path = root / relative
            if not path.exists():
                continue
            checked += 1
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "orders of magnitude",
                text,
                f"{relative} restates a figure `engine_search.py` retracts BY "
                "MEASUREMENT; the retraction has to hold in the prose a reader "
                "actually reads, not only inside one comment block",
            )
        self.assertGreater(
            checked, 0, "the doc scope must name at least one file that exists"
        )


@unittest.skipUnless(_crate_ready, "crate not built with the model feature")
class RolloutLeafWitnessCarriesTheCratesOwnLedgerTest(
    _EncodedSearchFixture, _ShippingPathHarness, unittest.TestCase
):
    """The witness, end to end, over the CRATE'S OWN measured rollout ledger.

    The class above proves the plumbing moves numbers. This one proves the numbers
    it moves are the crate's: the rollout columns come from a real
    `search_batched_multi_encoded` at this file's fixture config, and are then
    absorbed by production's `_search_model`. Only the visit rows are the
    harness's, because the fixture root's move names are not this harness's
    candidate list -- and the visit rows are not what the witness reports.

    This is where 0.96503 is a MEASUREMENT rather than a constant: the number is
    read off the crate here and asserted to survive the boundary unchanged.
    """

    SIMS = 32
    SEED = 5
    DEPTH = 2
    ROLLOUTS = 8
    MAX_PLIES = 40

    def _native(self):
        return pokezero_search.NativeLeafModel(
            str(self.artifact),
            device="cpu",
            window=1,
            tokens=int(self.layout["token_count"]),
            categorical_features=int(self.layout["categorical_feature_count"]),
            numeric_features=int(self.layout["numeric_feature_count"]),
        )

    _search = RolloutModelPriorsTest._search
    _search_raw_from_args = RolloutModelPriorsTest._search_raw_from_args

    def test_the_witness_reproduces_the_crates_own_fallback_fraction(self) -> None:
        crate = self._search(mode="rollout")
        self.ROLLOUT_COLUMNS = {
            key: crate[key]
            for key in (
                "rollout_leaf_mode",
                "rollout_encode_skipped",
                "leaves_priced",
                "rollouts_run",
                "rollout_plies",
                "rollout_terminal_hits",
                "rollout_cap_hits",
                "rollout_dead_ends",
            )
        }
        _, decision = self._decide(arm=True)
        witness = decision.metadata["engine_mcts"]["rollout_leaf"]
        for key in (
            "leaves_priced",
            "rollouts_run",
            "rollout_plies",
            "rollout_terminal_hits",
            "rollout_cap_hits",
            "rollout_dead_ends",
            "rollout_encode_skipped",
        ):
            with self.subTest(key=key):
                self.assertEqual(witness[key], crate[key])
        self.assertEqual(witness["rollout_leaf_modes"], {crate["rollout_leaf_mode"]: 1})
        # The crate rounds its own emission to 6 places; the witness does not, so
        # they are compared at the crate's precision.
        self.assertAlmostEqual(
            witness["rollout_fallback_fraction"],
            crate["rollout_fallback_fraction"],
            places=6,
        )
        # AND THE VALUE IS THE ONE THAT MAKES THIS URGENT. Asserted as a number,
        # not just as "equal to the crate", so a crate change that silently made
        # this arm look like an oracle would fail here rather than agree with
        # itself.
        self.assertGreater(
            witness["rollout_fallback_fraction"],
            0.9,
            "at THIS FIXTURE'S 40-ply cap the blend is 96.5% handcrafted; if this "
            "ever reads low, re-derive it rather than assuming the arm improved. "
            "This is not the arm's own figure: at the launched 200-ply cap the "
            "arm's banked whole-game run measures 0.018139547833834504, a 98.2% "
            "oracle. The cap is the variable, not the pricer",
        )


#: The rollout block of the arm's OWN BANKED SHARD, as the sibling bridge branch
#: wrote it (`ceiling-c3-rollout-arbiter-game-20260817/games/rollout-arm-game-seed1-*`,
#: three runs, bit-identical on this block). Copied in as a literal so this gate runs
#: in a bare checkout -- the artifacts live in the campaign store, not the repo.
#:
#: THIS IS THE v1 SCHEMA: `rollout_leaf_world_records` (accumulated `+= 1` per crate
#: report), every key emitted unconditionally, and `rollout_fallback_fraction`
#: computed from `rollout_cap_hits` alone.
#:
#: Note the value of the fallback fraction: 0.0181, a 98.2% ORACLE, at the launched
#: 200-ply cap. It is not 0.965 -- that is this file's 40-ply fixture.
BANKED_V1_ARM_SHARD = {
    "worlds_collapsed": 0,
    "worlds_searched": 65,
    "unique_worlds_searched": 65,
    "model_evals": 8997,
    "rollout_leaf_modes": {"rollout": 65},
    "rollout_leaf_world_records": 65,
    "rollouts_run": 71832,
    "rollout_plies": 3935252,
    "rollout_terminal_hits": 70529,
    "rollout_cap_hits": 1303,
    "rollout_dead_ends": 0,
    "rollout_leaves_priced": 8979,
    "rollout_encode_skipped": 47,
    "rollout_terminal_fraction": 0.9818604521661655,
    "rollout_fallback_fraction": 0.018139547833834504,
    "rollout_mean_plies": 54.78410736162156,
}


class RolloutLeafShardSchemaTest(unittest.TestCase):
    """A6. Two sibling branches, one shard surface, two schemas.

    #1272 owns `--engine-rollout-leaf` and wrote the shard above with
    `rollout_leaf_world_records` (`+= 1`, unconditional, `None` fractions). This
    branch renames it `rollout_leaf_worlds` (`+= weight`, conditional). Whichever
    landed second would SILENTLY RE-SCHEMA the arbiter's own banked artifacts: a v2
    reader on a v1 shard finds no `rollout_leaf_worlds` and reads zero engaged
    worlds, which is a banked arm run reading as no arm at all and is the most
    flattering possible misreading.

    The resolution is not a rename. It is a VERSION STAMP that travels with the data
    plus a refusal that fires when a shard's schema is not the reader's, because a
    silent re-schema of banked data is worse than a crash -- the crash is the only
    version of the event anyone finds out about.
    """

    @staticmethod
    def _v2(**overrides) -> dict:
        stats = EngineMctsStats()
        stats.rollout_leaf_modes["rollout"] = 65
        stats.rollout_leaf_worlds = 65
        stats.rollouts_run = 71832
        stats.rollout_plies = 3935252
        stats.rollout_terminal_hits = 70529
        stats.rollout_cap_hits = 1303
        stats.rollout_dead_ends = 0
        stats.rollout_leaves_priced = 8979
        stats.rollout_encode_skipped = 47
        payload = stats.to_dict()
        payload.update(overrides)
        return payload

    def test_the_writer_stamps_the_schema_it_wrote(self) -> None:
        """Without the stamp the two schemas are indistinguishable after the fact."""
        payload = self._v2()
        self.assertEqual(payload["rollout_leaf_schema"], ROLLOUT_LEAF_SHARD_SCHEMA)
        for name in ROLLOUT_LEAF_SHARD_FIELDS:
            with self.subTest(field=name):
                self.assertIn(name, payload)
        require_rollout_leaf_shard_schema(payload)

    def test_the_schema_VERSION_LITERAL_is_pinned(self) -> None:
        """B4. `ROLLOUT_LEAF_SHARD_SCHEMA = 2 -> = 1` SURVIVED.

        Every assertion about the stamp was self-referential: the writer stamps
        `ROLLOUT_LEAF_SHARD_SCHEMA`, the reader compares against
        `ROLLOUT_LEAF_SHARD_SCHEMA`, and the test above asserts the two are equal --
        so setting the constant to 1 kept the whole file internally consistent while
        every shard on disk got stamped as the schema this reader exists to REFUSE.
        That is the exact defect the constant's own docstring warns about one field
        over ("a reader that derives its expectation from the shard it is reading
        cannot detect a re-schema of that shard"), applied to the version itself.

        Pinned by LITERAL, and pinned to the number the four banked shards would be
        migrated INTO. Bumping the schema is meant to be a decision that shows up in a
        diff of this line.
        """
        self.assertEqual(ROLLOUT_LEAF_SHARD_SCHEMA, 2)
        # AND v1 IS STILL REFUSED BY NAME at that value, so the pin is not satisfied by
        # a constant that happens to equal 2 while the reader accepts anything.
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(self._v2(rollout_leaf_schema=1))
        self.assertIn("schema v1", str(caught.exception))
        self.assertIn("expects v2", str(caught.exception))

    def test_the_WEIGHTED_pricer_tally_must_sum_to_the_world_count(self) -> None:
        """B4. `rollout_leaf_modes[mode] += weight -> += 1` SURVIVED.

        Its sibling scalar `rollout_leaf_worlds += weight` was killed, because the
        world count is asserted. The Counter beside it was read only for its KEYS
        anywhere in the tree, so its VALUES were emitted and unverifiable -- and
        `+= weight` versus `+= 1` is precisely the accumulation difference that
        distinguishes v2 from v1 and is the whole stated reason for choosing v2.

        This is the SEVENTH instance of the drift this module's own docstring
        catalogues: `LADDER_PER_DECISION_CLAIM_HISTOGRAMS` calls
        `root_q_gap_histogram` versus `root_q_gap_sum` the SIXTH, and it is the same
        shape -- a Counter and a scalar incremented from one datum on one line pair,
        with a check on the scalar only.

        The fix is the invariant, not another assertion on one number:
        `sum(rollout_leaf_modes.values()) == rollout_leaf_worlds` is an identity
        because both are `+= weight` in the same pass, so any unweighting of either
        breaks it.
        """
        # THE COLLAPSED DRAW, which is the case the two rules disagree on. Two
        # identical belief draws share one search, so one crate report carries
        # weight 2.
        weighted = self._v2()
        weighted["rollout_leaf_modes"] = {"rollout": 130}
        weighted["rollout_leaf_worlds"] = 130
        require_rollout_leaf_shard_schema(weighted)
        # `+= 1` on the Counter alone: 65 reports, 130 worlds.
        unweighted_counter = self._v2()
        unweighted_counter["rollout_leaf_modes"] = {"rollout": 65}
        unweighted_counter["rollout_leaf_worlds"] = 130
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(unweighted_counter)
        self.assertIn("does not sum to its world count", str(caught.exception))
        self.assertIn("seven times", str(caught.exception))
        # And the other direction, so the invariant is not one-sided.
        unweighted_scalar = self._v2()
        unweighted_scalar["rollout_leaf_modes"] = {"rollout": 130}
        unweighted_scalar["rollout_leaf_worlds"] = 65
        with self.assertRaises(EngineShardSchemaError):
            require_rollout_leaf_shard_schema(unweighted_scalar)
        # Two pricers summing correctly is accepted -- the invariant is on the SUM, not
        # on a single-entry mapping.
        split = self._v2()
        split["rollout_leaf_modes"] = {"model_value": 20, "rollout": 45}
        split["rollout_leaf_worlds"] = 65
        require_rollout_leaf_shard_schema(split)

    def test_BOTH_world_counters_are_accumulated_BY_WEIGHT_at_the_writer(self) -> None:
        """B4. The invariant catches a DISAGREEING shard; this catches the writer.

        `b4_unweight_the_shard_pricer_counter` -- `self.stats.rollout_leaf_modes[mode]
        += weight` -> `+= 1` -- SURVIVED the invariant above, and the reason is worth
        stating rather than patching around: `sum(modes) == worlds` fires on a shard
        where the two DISAGREE, and the only shard that can disagree is one written
        from a COLLAPSED draw. Every fixture in this file, and every search the crate
        gate runs, has weight 1 -- two identical belief draws from the same state,
        context and seat -- so the mutant produces a byte-identical shard and nothing
        can see it.

        Reaching it behaviourally needs a model search that actually collapses a draw,
        which this file's fixture cannot produce. So the pair is pinned at the writer,
        SCOPED BY VALUE: both members accumulate the operand `weight`, and the
        assertion is derived from the counter names rather than from a copied source
        line. A third world counter added to the pair is required to join it.

        Stated as the limitation it is: this is a structural pin, and the invariant is
        the behavioural half. Together they say "the writer weights both" and "a shard
        where they disagree is refused"; neither alone says both.
        """
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "src" / "pokezero" / "engine_search.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        def _is_shard_world_counter(target: ast.AST) -> bool:
            # `self.stats.rollout_leaf_worlds` or `self.stats.rollout_leaf_modes[...]`
            node = target.value if isinstance(target, ast.Subscript) else target
            return (
                isinstance(node, ast.Attribute)
                and node.attr in ("rollout_leaf_worlds", "rollout_leaf_modes")
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "stats"
            )

        found: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign) and _is_shard_world_counter(node.target):
                name = (
                    node.target.value.attr
                    if isinstance(node.target, ast.Subscript)
                    else node.target.attr
                )
                found.setdefault(name, []).append(ast.unparse(node.value))
        self.assertEqual(
            sorted(found),
            ["rollout_leaf_modes", "rollout_leaf_worlds"],
            "both shard-level world counters must be accumulated somewhere; a missing "
            f"one is a surface that stopped being written. Found: {found}",
        )
        for name, operands in sorted(found.items()):
            with self.subTest(counter=name):
                self.assertEqual(
                    operands,
                    ["weight"],
                    f"`self.stats.{name}` must be accumulated `+= weight` exactly "
                    "once. `+= 1` records ONE world where a collapsed draw "
                    "represented two, which is the whole stated reason v2 was chosen "
                    f"over v1. Found: {operands}",
                )

    def test_zero_rollouts_does_not_SHORT_CIRCUIT_the_partition_checks(self) -> None:
        """B5. `rollouts_run == 0` returned before every remaining check.

        Two things got through it, and the second is exactly what the sibling bridge
        branch writes on a FLAG-OFF shard:

          * `rollouts_run: 0` beside `rollout_cap_hits: 5` -- a partition that does not
            balance. The identity is not undefined at zero, it is TRIVIAL at zero, and
            a non-trivial reading is a counter wired to the wrong population.
          * `rollouts_run: 0` beside `rollout_terminal_fraction: None` -- v1's
            unconditional-emission shape, stamped v2. A pooling reader averages a
            `None` as though it were a measurement, and the block asserts the arm
            engaged and priced nothing, which is a refusal shape rather than a shard.
        """
        zero = self._v2()
        for name in (
            "rollouts_run",
            "rollout_plies",
            "rollout_terminal_hits",
            "rollout_cap_hits",
            "rollout_dead_ends",
        ):
            zero[name] = 0
        for name in ROLLOUT_LEAF_QUOTIENT_FIELDS:
            zero.pop(name, None)
        # The legitimate zero-rollout v2 block: counts present, quotients omitted.
        require_rollout_leaf_shard_schema(zero)

        torn = dict(zero)
        torn["rollout_cap_hits"] = 5
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(torn)
        self.assertIn("does not account for every rollout", str(caught.exception))

        for name in ROLLOUT_LEAF_QUOTIENT_FIELDS:
            with self.subTest(v1_shape=name):
                v1_shape = dict(zero)
                v1_shape[name] = None
                with self.assertRaises(EngineShardSchemaError) as caught:
                    require_rollout_leaf_shard_schema(v1_shape)
                self.assertIn(name, str(caught.exception))
                self.assertIn("wearing this schema's stamp", str(caught.exception))

    def test_the_siblings_FLAG_OFF_shard_is_refused_by_this_reader(self) -> None:
        """B2. What #1272 writes if it lands without its emission block deleted.

        Its `to_dict` emits the whole rollout block UNCONDITIONALLY: thirteen rollout
        keys on a shard whose seam never engaged, zeroed counts, `None` fractions, and
        -- since it has adopted this branch's constant -- STAMPED v2. That document
        passed the first revision of this reader, because `rollouts_run == 0`
        short-circuited every check that would have looked at it.

        Reconstructed here as the literal key set that writer emits, so the refusal is
        pinned to the artifact and not to a description of it.
        """
        sibling_flag_off = {
            "worlds_collapsed": 0,
            "model_evals": 8185,
            "rollout_leaf_schema": ROLLOUT_LEAF_SHARD_SCHEMA,
            "rollout_leaf_modes": {},
            "rollout_leaf_worlds": 0,
            "rollouts_run": 0,
            "rollout_plies": 0,
            "rollout_terminal_hits": 0,
            "rollout_cap_hits": 0,
            "rollout_dead_ends": 0,
            "rollout_leaves_priced": 0,
            "rollout_encode_skipped": 0,
            "rollout_terminal_fraction": None,
            "rollout_fallback_fraction": None,
            "rollout_mean_plies": None,
        }
        self.assertEqual(
            len([k for k in sibling_flag_off if str(k).startswith("rollout")]),
            13,
            "thirteen rollout keys on a flag-off shard is the shape being refused",
        )
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(sibling_flag_off)
        self.assertIn("rollout_terminal_fraction", str(caught.exception))

    def test_the_writer_PINS_its_rollout_key_set_so_a_rename_fails_there(self) -> None:
        """B5. The reader cannot see a FULL rename, and this is where that is closed.

        The accept case (`if not present: return`) survives the stated attack -- all
        twelve keys have to go, not one. But the predicate is
        `startswith("rollout")`, so a writer that renamed every column to a
        NON-`rollout`-prefixed name produces a block this reader accepts as flag-off
        while `worlds_collapsed: 7` sits beside it saying a search ran. That residual
        is REAL and it is not closable at the reader: an artifact does not carry its
        launch command, and the reader has no closed set of legitimate non-rollout
        `policy_stats` columns to diff against.

        It IS closable at the writer, where the key set is pinned by literal. This
        asserts the emitted rollout key set is EXACTLY `ROLLOUT_LEAF_SHARD_FIELDS`, so
        a rename in `to_dict` fails here rather than sailing past the reader. The
        residual that remains -- a DIFFERENT writer renaming the columns of an
        artifact this repo did not produce -- is stated rather than papered over.
        """
        emitted = {k for k in self._v2() if str(k).startswith("rollout")}
        self.assertEqual(
            emitted,
            set(ROLLOUT_LEAF_SHARD_FIELDS),
            "the writer's rollout key set must be exactly the pinned literal; a "
            "rename either fails here or is invisible to every reader",
        )
        # And the pin is on a LITERAL tuple, not on the writer's own output.
        self.assertEqual(len(ROLLOUT_LEAF_SHARD_FIELDS), 13)
        self.assertEqual(len(set(ROLLOUT_LEAF_SHARD_FIELDS)), 13)

    def test_a_flag_off_shard_carries_no_rollout_keys_and_is_accepted(self) -> None:
        """The conditional emission, and why the reader keys off KEYS not config.

        A shard is read without its launch command, so the predicate has to be "does
        this block carry rollout keys". Flag-off emits none, which must be accepted
        rather than treated as a dropped field -- otherwise every pre-seam shard in
        the archive refuses.
        """
        payload = EngineMctsStats().to_dict()
        self.assertEqual([k for k in payload if k.startswith("rollout")], [])
        require_rollout_leaf_shard_schema(payload)

    def test_the_banked_v1_shard_is_refused_and_named_as_v1(self) -> None:
        """THE DEMONSTRATED FAILING INPUT, and it is real banked data."""
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(BANKED_V1_ARM_SHARD)
        message = str(caught.exception)
        self.assertIn("schema v1", message)
        self.assertIn(ROLLOUT_LEAF_V1_WORLD_FIELD, message)
        self.assertIn("migrate_rollout_leaf_shard_v1", message)

    def test_the_v1_world_count_is_not_silently_read_as_zero(self) -> None:
        """The precise corruption. Stated as an assertion so it cannot come back.

        A reader that merely `.get`s the v2 name off a v1 shard gets None, and the
        arithmetic downstream treats that as zero engaged worlds -- with the shard
        sitting right there saying 65.
        """
        self.assertIsNone(BANKED_V1_ARM_SHARD.get("rollout_leaf_worlds"))
        self.assertEqual(BANKED_V1_ARM_SHARD[ROLLOUT_LEAF_V1_WORLD_FIELD], 65)

    def test_a_half_migrated_writer_carrying_both_spellings_refuses(self) -> None:
        both = dict(self._v2())
        both[ROLLOUT_LEAF_V1_WORLD_FIELD] = 64
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(both)
        self.assertIn("BOTH", str(caught.exception))

    def test_an_unstamped_v2_shaped_shard_refuses(self) -> None:
        """A writer that emits the v2 key set and forgets the stamp is refused.

        "Written before the stamp existed" and "written by a writer that dropped a
        field" are the same artifact, so the reader cannot resolve it and must not
        guess.
        """
        unstamped = dict(self._v2())
        del unstamped["rollout_leaf_schema"]
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(unstamped)
        self.assertIn("rollout_leaf_schema", str(caught.exception))

    def test_a_future_schema_version_refuses(self) -> None:
        with self.assertRaises(EngineShardSchemaError):
            require_rollout_leaf_shard_schema(
                self._v2(rollout_leaf_schema=ROLLOUT_LEAF_SHARD_SCHEMA + 1)
            )

    def test_a_dropped_column_refuses_rather_than_reading_as_zero(self) -> None:
        """ONE FIELD PER CASE, so each drop trips its own branch."""
        for name in ROLLOUT_LEAF_SHARD_FIELDS:
            if name == "rollout_leaf_schema":
                continue  # its own case above
            with self.subTest(field=name):
                shard = dict(self._v2())
                del shard[name]
                with self.assertRaises(EngineShardSchemaError) as caught:
                    require_rollout_leaf_shard_schema(shard)
                self.assertIn(name, str(caught.exception))

    def test_an_unrecognised_rollout_column_refuses(self) -> None:
        """A new column is a schema change. Silently pooling a shard that has one
        with shards that do not is the same class of defect as the rename."""
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(self._v2(rollout_new_thing=1))
        self.assertIn("rollout_new_thing", str(caught.exception))

    def test_a_v1_fallback_numerator_on_a_v2_shard_refuses(self) -> None:
        """The other divergence, and the one that is NOT a rename.

        v1's numerator is `cap` alone; v2's is `cap + dead_ends`. They agree only
        while dead_ends is zero, and v1 keeps it zero by REFUSING the world rather
        than by arithmetic -- so the identity is a property of a writer version, not
        of the schema. A v1 fraction on a v2-stamped shard reads False here.
        """
        shard = self._v2(rollout_dead_ends=7, rollout_terminal_hits=70522)
        shard["rollout_fallback_fraction"] = 1303 / 71832  # cap only: v1's rule
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(shard)
        self.assertIn("not the quotient", str(caught.exception))

    def test_an_unbalanced_partition_refuses(self) -> None:
        with self.assertRaises(EngineShardSchemaError) as caught:
            require_rollout_leaf_shard_schema(self._v2(rollout_terminal_hits=70528))
        self.assertIn("every rollout", str(caught.exception))

    def test_the_shard_fallback_numerator_is_the_whole_partition(self) -> None:
        """The v1/v2 divergence that is NOT a rename, on the WRITER side.

        Every other fixture in this file has `rollout_dead_ends == 0`, where `cap` and
        `cap + dead` are equal -- so a mutant that drops the `dead` term survived them
        all. This is the one case that separates the two rules, and it exists for that
        reason: `dead == 0` is a property of the sibling writer (which refuses a
        non-zero count upstream), not of the schema.
        """
        stats = EngineMctsStats()
        stats.rollout_leaf_modes["rollout"] = 1
        stats.rollout_leaf_worlds = 1
        stats.rollouts_run = 100
        stats.rollout_plies = 1000
        stats.rollout_terminal_hits = 80
        stats.rollout_cap_hits = 13
        stats.rollout_dead_ends = 7
        stats.rollout_leaves_priced = 10
        stats.rollout_encode_skipped = 3
        payload = stats.to_dict()
        self.assertAlmostEqual(payload["rollout_fallback_fraction"], 0.20, places=12)
        self.assertNotAlmostEqual(
            payload["rollout_fallback_fraction"],
            0.13,
            places=12,
            msg="cap alone is v1's numerator; dropping the dead-end term must move "
            "this number, or the two schemas are indistinguishable here",
        )
        self.assertAlmostEqual(
            payload["rollout_terminal_fraction"] + payload["rollout_fallback_fraction"],
            1.0,
            places=12,
        )
        require_rollout_leaf_shard_schema(payload)

    def test_the_migration_is_value_preserving_on_the_banked_shard(self) -> None:
        """The cost of choosing v2, paid explicitly.

        The rename is sound on this shard because the shard says so: `+= 1` and
        `+= weight` agree exactly when no draw was collapsed, and
        `worlds_collapsed == 0` IS that condition. Measured 0 on all four banked
        shards, so the migrated world count is the banked one and the migrated
        fallback fraction is bit-identical to what was banked.
        """
        migrated = migrate_rollout_leaf_shard_v1(BANKED_V1_ARM_SHARD)
        require_rollout_leaf_shard_schema(migrated)
        self.assertEqual(migrated["rollout_leaf_worlds"], 65)
        self.assertNotIn(ROLLOUT_LEAF_V1_WORLD_FIELD, migrated)
        self.assertEqual(migrated["rollout_leaf_schema"], ROLLOUT_LEAF_SHARD_SCHEMA)
        self.assertEqual(
            migrated["rollout_fallback_fraction"],
            BANKED_V1_ARM_SHARD["rollout_fallback_fraction"],
        )
        self.assertEqual(migrated["rollout_fallback_fraction"], 0.018139547833834504)
        # ... which is 98.2% ORACLE at the launched 200-ply cap. The 0.965 in this
        # file is the 40-ply FIXTURE, and conflating the two is the mislabel this
        # revision corrects.
        self.assertLess(migrated["rollout_fallback_fraction"], 0.02)

    def test_the_migration_refuses_a_collapsed_shard(self) -> None:
        """The DEMONSTRATED FAILING INPUT for the migration's own condition.

        With a collapsed draw the v1 number counts crate REPORTS and the v2 name
        promises WORLDS. Renaming there would publish one under the other's name,
        which is exactly the silent re-schema this whole gate exists to prevent -- so
        the migration refuses rather than being best-effort.
        """
        shard = dict(BANKED_V1_ARM_SHARD, worlds_collapsed=3)
        with self.assertRaises(EngineShardSchemaError) as caught:
            migrate_rollout_leaf_shard_v1(shard)
        self.assertIn("worlds_collapsed=3", str(caught.exception))

    def test_the_migration_refuses_a_nonzero_dead_end_count(self) -> None:
        shard = dict(
            BANKED_V1_ARM_SHARD, rollout_dead_ends=5, rollout_terminal_hits=70524
        )
        with self.assertRaises(EngineShardSchemaError) as caught:
            migrate_rollout_leaf_shard_v1(shard)
        self.assertIn("rollout_dead_ends=5", str(caught.exception))

    def test_the_migration_drops_the_block_from_a_flag_off_v1_shard(self) -> None:
        """v1 emitted the block unconditionally, so its flag-OFF shards carry a
        zeroed one. v2 emits nothing there, and stamping a zeroed block as v2 would
        assert that the arm engaged and priced nothing -- which is a refusal shape,
        not a shard."""
        off = {
            "worlds_collapsed": 0,
            "model_evals": 8185,
            "rollout_leaf_modes": {},
            "rollout_leaf_world_records": 0,
            "rollouts_run": 0,
            # v1 emitted EVERY key unconditionally, `rollout_dead_ends` included. The
            # first version of this fixture omitted it and still passed, because the
            # migration read `shard.get("rollout_dead_ends") or 0` -- absent as zero, in
            # a module that carries an "ABSENT IS NOT FALSE" paragraph. Both conditions
            # the migration's value-preservation proof rests on are now required to be
            # PRESENT, so the fixture has to be what v1 actually writes.
            "rollout_dead_ends": 0,
            "rollout_terminal_fraction": None,
            "rollout_fallback_fraction": None,
            "rollout_mean_plies": None,
        }
        migrated = migrate_rollout_leaf_shard_v1(off)
        self.assertEqual([k for k in migrated if k.startswith("rollout")], [])
        require_rollout_leaf_shard_schema(migrated)

    def test_the_migration_refuses_an_ABSENT_precondition_rather_than_reading_zero(
        self,
    ) -> None:
        """B5. ABSENT IS NOT ZERO, on the two fields the proof rests on.

        `worlds_collapsed == 0` and `rollout_dead_ends == 0` are the ENTIRE conditions
        under which this migration claims to be value-preserving. Read through
        `int(shard.get(name) or 0)`, a shard that never recorded the field passed the
        check -- so the premise of the proof was being read off a field that was not
        measured, and "no collapses happened" and "this writer does not count
        collapses" became the same artifact. One field dropped per subtest, so the
        first check cannot short-circuit the second.
        """
        for name in ("worlds_collapsed", "rollout_dead_ends"):
            with self.subTest(absent=name):
                shard = dict(BANKED_V1_ARM_SHARD)
                self.assertIn(name, shard, "the fixture must carry it to drop it")
                del shard[name]
                with self.assertRaises(EngineShardSchemaError) as caught:
                    migrate_rollout_leaf_shard_v1(shard)
                self.assertIn(name, str(caught.exception))
                self.assertIn("ABSENT IS NOT ZERO", str(caught.exception))
        # And the unmutated fixture migrates, so the refusal is not simply refusing
        # everything.
        migrate_rollout_leaf_shard_v1(dict(BANKED_V1_ARM_SHARD))

    def test_the_migration_does_not_OVERWRITE_a_banked_quotient(self) -> None:
        """B5. "Provably value-preserving" was a claim the code contradicted.

        All three quotients were assigned UNCONDITIONALLY from the shard's counts, so
        a v1 shard carrying `rollout_terminal_fraction: 0.99` and
        `rollout_mean_plies: 999.0` came out the other side as this shard's real 0.8
        and 20.0 with nothing recording that anything had changed -- under a docstring
        calling the rename value-preserving. Only `rollout_fallback_fraction` is a
        DEFINED re-derivation (v1's numerator was `cap`, v2's is `cap + dead_ends`),
        and `dead == 0` is enforced upstream, so even that one cannot legitimately
        move. A disagreement is now named and refused.

        One field perturbed per subtest: a fixture that moves two at once pins
        neither, because the first refusal short-circuits the second.
        """
        for name, forged in (
            ("rollout_terminal_fraction", 0.99),
            ("rollout_mean_plies", 999.0),
            ("rollout_fallback_fraction", 0.5),
        ):
            with self.subTest(overwritten=name):
                shard = dict(BANKED_V1_ARM_SHARD)
                honest = migrate_rollout_leaf_shard_v1(dict(shard))[name]
                self.assertNotAlmostEqual(honest, forged, places=6)
                shard[name] = forged
                with self.assertRaises(EngineShardSchemaError) as caught:
                    migrate_rollout_leaf_shard_v1(shard)
                message = str(caught.exception)
                self.assertIn(name, message)
                self.assertIn(repr(forged), message)
                self.assertIn(repr(honest), message)
        # A shard whose banked quotient AGREES is migrated, unchanged.
        agreeing = dict(BANKED_V1_ARM_SHARD)
        expected = migrate_rollout_leaf_shard_v1(dict(agreeing))
        agreeing["rollout_terminal_fraction"] = expected["rollout_terminal_fraction"]
        self.assertEqual(
            migrate_rollout_leaf_shard_v1(agreeing)["rollout_terminal_fraction"],
            expected["rollout_terminal_fraction"],
        )

    def test_the_migrations_trailing_SELF_CHECK_is_not_deletable(self) -> None:
        """B4. The last line of the migration was a free deletion.

        `require_rollout_leaf_shard_schema(migrated)` closes the loop: the migration
        may only emit a document its own reader accepts. Nothing exercised it, because
        every migration fixture happened to produce a valid v2 block -- so deleting
        the line changed no observable behaviour. Two inputs that pass every
        PRECONDITION and produce an INVALID v2 block:

          * a v1 shard carrying a column v2 does not recognise, which the rename
            copies straight through into a v2-stamped block;
          * a v1 shard whose partition does not balance, which the rename cannot fix
            and must not publish.

        Both are refused by the trailing check and by nothing else in the function.
        """
        unknown = dict(BANKED_V1_ARM_SHARD)
        unknown["rollout_new_column"] = 3
        with self.assertRaises(EngineShardSchemaError) as caught:
            migrate_rollout_leaf_shard_v1(unknown)
        self.assertIn("rollout_new_column", str(caught.exception))
        self.assertIn("unrecognised", str(caught.exception))

        torn = dict(BANKED_V1_ARM_SHARD)
        torn["rollout_terminal_hits"] = int(torn["rollout_terminal_hits"]) + 1
        # The quotient stays consistent with the counts under BOTH rules, so the
        # overwrite refusal above cannot be what catches this one.
        torn["rollout_terminal_fraction"] = (
            int(torn["rollout_terminal_hits"]) / int(torn["rollouts_run"])
        )
        with self.assertRaises(EngineShardSchemaError) as caught:
            migrate_rollout_leaf_shard_v1(torn)
        self.assertIn("does not account for every rollout", str(caught.exception))

    def test_the_migration_refuses_a_shard_that_is_not_v1(self) -> None:
        with self.assertRaises(EngineShardSchemaError):
            migrate_rollout_leaf_shard_v1(self._v2())


class BankedShardCarriesThePooledWitnessTest(unittest.TestCase):
    """A1. The witness was guarded to its last hop, and the last hop was open.

    The pooled witness reaches the arbiter's paired-eval shard by exactly one path:

      crate report -> `_search_model`'s absorb -> `EngineMctsStats`
        -> `EngineMctsStats.to_dict` -> `_engine_policy_stats`
        -> `ControlledFoulPlayBenchmarkResult.policy_stats`
        -> `"policy_stats": self.policy_stats` -> `_write_json` -> disk

    Twelve mutants on the first six hops were all killed. The SEVENTH survived:
    `self.policy_stats` -> `None` and `_engine_policy_stats` -> `{}` both passed 435
    tests across every module that references `policy_stats` -- the exact historical
    silent-empty bug, one frame further out than the tests had followed. The
    existing guards covered a DIFFERENT writer
    (`mcts_acceptance_h2h.build_shard_report`) and hand-written fixtures on the
    READING side, neither of which touches this line.

    WHERE THE WALK STOPS, AND WHY. At `_write_json`, the single funnel through which
    every shard reaches disk -- the last frame in this repository that still holds
    the datum. There is no eighth hop to instrument: the next consumer is a file.
    And the refusal there is not the only thing standing in the way of a silent
    re-schema, which is the answer to "then delete the guard": the reader-side
    `require_rollout_leaf_shard_schema` fires on the same shard from the other side
    of the disk, so a shard written past a deleted writer-side guard is REFUSED ON
    READ rather than pooled. Two frames on opposite sides of the artifact, and both
    would have to go for the corruption to be quiet.
    """

    @staticmethod
    def _bridge():
        from pokezero import foulplay_bridge

        return foulplay_bridge

    def _payload(self, *, policy_stats, **extra) -> dict:
        """A bridge shard payload with the block the guard reads.

        Shaped rather than constructed from a real benchmark run, because the defect
        is one key in one dict and a real run needs a Showdown checkout.
        """
        block = {
            "decisions": 65,
            "fallback_decisions": 0,
            "fallback_rate": 0.0,
            "search_wall_per_searched_decision": 1.25,
            "policy_stats": policy_stats,
        }
        if policy_stats is _MISSING:
            del block["policy_stats"]
        block.update(extra)
        return {"policy_id": "arm", "engine_mcts": block}

    def test_the_emitter_actually_carries_policy_stats_to_the_shard(self) -> None:
        """THE SURVIVING LINE, asserted on the real emitter.

        `ControlledFoulPlayBenchmarkResult.to_dict()` -> the `engine_mcts` block ->
        `policy_stats`. Reads False on `"policy_stats": None`, which is the mutant
        that survived.
        """
        from pathlib import Path

        bridge = self._bridge()
        stats = {"decisions": 65, "search_wall_per_searched_decision": 1.25}
        result = bridge.ControlledFoulPlayBenchmarkResult(
            config=bridge.ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                policy_mode="engine-mcts",
                engine_model_path=Path("model.pt"),
                engine_tables_path=Path("tables.json"),
            ),
            policy_id="arm",
            games=(),
            policy_stats=stats,
        )
        block = result.to_dict()["engine_mcts"]
        self.assertEqual(block["policy_stats"], stats)
        self.assertEqual(block["search_wall_per_searched_decision"], 1.25)

    def test_engine_policy_stats_returns_the_searchers_own_telemetry(self) -> None:
        """THE OTHER SURVIVING MUTANT, and the hop before the last one.

        `_engine_policy_stats` returning `{}` for an engine-mcts run is the historical
        bug verbatim. Its own docstring says it raises rather than "quietly produce an
        empty telemetry block ... the failure mode that left every acceptance shard
        with `policy_stats: {}`" -- and nothing checked that it does. Asserted on
        IDENTITY with the policy's own `to_dict()`, so a stub cannot satisfy it.
        """
        bridge = self._bridge()
        policy = SimpleNamespace(stats=EngineMctsStats())
        stats = bridge._engine_policy_stats(policy, "engine-mcts")
        self.assertTrue(stats, "an engine-mcts run's telemetry block must not be empty")
        self.assertEqual(stats, policy.stats.to_dict())
        self.assertGreater(len(stats), 20)
        # None -- not `{}` -- outside engine-mcts, which is the honest answer there:
        # there is no engine searcher, so there is nothing to report.
        self.assertIsNone(bridge._engine_policy_stats(policy, "root-puct"))

    def test_the_write_refuses_a_none_policy_stats(self) -> None:
        """The exact surviving mutant, at the boundary it escaped through."""
        bridge = self._bridge()
        with self.assertRaises(bridge.BankedShardWitnessError) as caught:
            bridge.require_banked_shard_witness(self._payload(policy_stats=None))
        self.assertIn("SILENT-EMPTY", str(caught.exception))

    def test_the_write_refuses_an_empty_policy_stats(self) -> None:
        """`_engine_policy_stats` -> `{}`: the historical bug verbatim."""
        bridge = self._bridge()
        with self.assertRaises(bridge.BankedShardWitnessError):
            bridge.require_banked_shard_witness(self._payload(policy_stats={}))

    def test_the_write_refuses_a_block_with_the_key_deleted(self) -> None:
        """The mutation one step further: delete the line rather than blank it."""
        bridge = self._bridge()
        with self.assertRaises(bridge.BankedShardWitnessError) as caught:
            bridge.require_banked_shard_witness(self._payload(policy_stats=_MISSING))
        self.assertIn("no 'policy_stats' key", str(caught.exception))

    def test_the_guard_reaches_a_block_nested_under_an_arm(self) -> None:
        """The comparison summaries nest one block per arm. A top-level-only guard
        would pass exactly the payloads where two arms are compared."""
        bridge = self._bridge()
        nested = {
            "comparison": {"arms": [self._payload(policy_stats=None)]},
        }
        with self.assertRaises(bridge.BankedShardWitnessError):
            bridge.require_banked_shard_witness(nested)

    def test_a_populated_block_is_accepted(self) -> None:
        """Discriminating power: without this every refusal above is satisfied by a
        guard that refuses everything."""
        bridge = self._bridge()
        bridge.require_banked_shard_witness(
            self._payload(policy_stats={"decisions": 65})
        )

    def test_a_non_engine_payload_is_untouched(self) -> None:
        """`_write_json` writes more than engine shards; the guard must not fire on
        a payload that has no engine block, and `engine_mcts: None` is what a
        non-engine-mcts run legitimately emits."""
        bridge = self._bridge()
        bridge.require_banked_shard_witness({"engine_mcts": None, "root_puct": {}})
        bridge.require_banked_shard_witness({"comparison": {"sample_size": {}}})

    def test_the_write_refuses_a_v1_schemaed_policy_stats(self) -> None:
        """A6's refusal, reached through the WRITE path -- so a writer that still
        emits v1 cannot bank alongside v2 shards.

        B1. THROUGH `_shard_json_text`, not through `require_banked_shard_witness`.
        The schema refusal used to be called from INSIDE the witness guard, and this
        test asserted it there -- which is what made the two look like one check and
        hid the fact that deleting the witness call removed both. They are separate
        statements at the boundary now, so the boundary is what is tested.
        """
        bridge = self._bridge()
        with self.assertRaises(Exception) as caught:
            bridge._shard_json_text(
                self._payload(policy_stats=dict(BANKED_V1_ARM_SHARD))
            )
        self.assertIn("schema v1", str(caught.exception))
        # AND NOT through the witness guard, which no longer owns it. If this ever
        # raises again the two guards have been re-nested and the independence claim
        # below is false.
        bridge.require_banked_shard_witness(
            self._payload(policy_stats=dict(BANKED_V1_ARM_SHARD))
        )

    def test_the_two_boundary_GUARDS_ARE_INDEPENDENTLY_DELETABLE(self) -> None:
        """B1. The pair claim, made honest.

        The earlier revision claimed a shard written past a deleted writer-side guard
        would be "REFUSED ON READ instead of pooled. Both frames -- opposite sides of
        the artifact -- would have to be deleted together". That was FALSE:
        `require_rollout_leaf_shard_schema` had NO read-path caller, its only two
        non-test callers were both write-side, and one NESTED the other -- so deleting
        the single line `require_banked_shard_witness(payload)` removed BOTH refusals.

        What the topology has to be for the claim to hold, asserted by construction:

          * the witness refusal and the schema refusal are two statements at the
            boundary, so each catches an input the other does not;
          * the schema refusal is also reached from a READ-path caller in a different
            module that does not import `foulplay_bridge`.
        """
        import inspect

        from pokezero import engine_search

        bridge = self._bridge()
        source = inspect.getsource(bridge._shard_json_text)
        self.assertIn("require_banked_shard_witness(payload)", source)
        self.assertIn("require_rollout_leaf_document_schema(payload)", source)
        # NEITHER GUARD IS INSIDE THE OTHER. Checked on the CALL GRAPH and not on the
        # source text: the witness guard's docstring now NAMES the schema refusal in
        # order to explain the correction, so a substring test reads False on the
        # documentation of the fix. `ast` distinguishes a mention from a call.
        import ast
        import textwrap

        witness_ast = ast.parse(
            textwrap.dedent(inspect.getsource(bridge.require_banked_shard_witness))
        )
        called = {
            node.func.id
            for node in ast.walk(witness_ast)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } | {
            node.func.attr
            for node in ast.walk(witness_ast)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for nested in (
            "require_rollout_leaf_shard_schema",
            "require_rollout_leaf_document_schema",
        ):
            self.assertNotIn(
                nested,
                called,
                "the schema refusal must not be CALLED from inside the witness "
                "refusal again; nesting is what made one deletion remove two checks",
            )
        # EACH CATCHES AN INPUT THE OTHER DOES NOT. A v1-schemaed block whose witness
        # is otherwise perfect, and a silent-empty block carrying no rollout keys at
        # all.
        v1_only = self._payload(policy_stats=dict(BANKED_V1_ARM_SHARD))
        bridge.require_banked_shard_witness(v1_only)  # witness guard: passes
        with self.assertRaises(engine_search.EngineShardSchemaError):
            engine_search.require_rollout_leaf_document_schema(v1_only)
        empty_only = self._payload(policy_stats={})
        engine_search.require_rollout_leaf_document_schema(empty_only)  # schema: passes
        with self.assertRaises(bridge.BankedShardWitnessError):
            bridge.require_banked_shard_witness(empty_only)

    def test_the_READ_path_refuses_what_a_deleted_writer_guard_would_have_banked(
        self,
    ) -> None:
        """B1. The read side, driven through the real reader on a real file.

        This is the call that did not exist. The shard is written to disk WITHOUT
        going through the bridge -- i.e. exactly the state the tree is in if every
        writer-side guard is deleted -- and the paired-eval driver's own loader is
        asked to read it.
        """
        import json
        import tempfile
        from pathlib import Path

        paired = _paired_eval_module()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shard-p1.json"
            target.write_text(
                json.dumps(
                    {
                        "policy_id": "arm",
                        "engine_mcts": {
                            "decisions": 65,
                            "policy_stats": dict(BANKED_V1_ARM_SHARD),
                        },
                    }
                ),
                encoding="utf-8",
            )
            document = json.loads(target.read_text(encoding="utf-8"))
            with self.assertRaises(Exception) as caught:
                paired.require_rollout_leaf_document_schema(document)
            self.assertIn("schema v1", str(caught.exception))
        # And the reader is reached from a module that does NOT import the bridge, so
        # the two deletions are in two files and two processes.
        import inspect

        self.assertNotIn("foulplay_bridge", inspect.getsource(paired.run_seat))

    def test_BOTH_reader_call_sites_exist_where_the_shard_is_loaded(self) -> None:
        """B1. The two read-path deletions, each caught on its own.

        The test above proves the refusal WORKS on a document read off disk. It does
        not prove it is CALLED, and the battery said so: `b1_delete_the_paired_eval_
        READER_call` and `b1_delete_the_power_report_READER_call` both SURVIVED,
        because a test that calls the function itself cannot see the call site
        disappear. That is the same "absence of code is not mutation-testable" shape
        the witness guard exists for, one layer out -- so it is asserted on the call
        graph.

        Driving `run_seat` end to end would need a Showdown checkout and a subprocess
        bridge invocation, so the assertion is structural AND it is pinned to the frame
        that matters: the call must sit in the function that LOADS the file, after the
        parse and before the value is returned to be pooled.
        """
        import ast
        import inspect
        import textwrap

        paired = _paired_eval_module()
        power = _power_report_module()
        for label, function, loader in (
            ("scripts/foulplay_paired_eval.py::run_seat", paired.run_seat, "json.loads"),
            (
                "scripts/foulplay_power_report.py::load_shards",
                power.load_shards,
                "json.loads",
            ),
        ):
            with self.subTest(reader=label):
                source = textwrap.dedent(inspect.getsource(function))
                self.assertIn(loader, source, "this must be the frame that parses")
                calls = [
                    node
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "require_rollout_leaf_document_schema"
                ]
                self.assertEqual(
                    len(calls),
                    1,
                    f"{label} must call the reader-side refusal exactly once on the "
                    "document it just parsed; without it the writer-side guard is the "
                    "only copy and deleting one line removes every refusal",
                )
                # ON THE PARSED DOCUMENT, not on some other name -- a call on the wrong
                # argument satisfies a bare occurrence count and checks nothing.
                (call,) = calls
                self.assertEqual(len(call.args), 1)
                self.assertIsInstance(call.args[0], ast.Name)
                self.assertIn(call.args[0].id, ("summary", "payload"))

    def test_BOTH_stdout_writers_render_through_the_guarded_funnel(self) -> None:
        """B1. `_write_json` was not the only funnel, and the print sites were open.

        `b1_stdout_writer_bypasses_the_funnel` SURVIVED: reverting one `print` to
        `json.dumps(payload, indent=2, sort_keys=True)` changed nothing observable,
        because nothing asserted the stdout path is guarded. The text is
        BYTE-IDENTICAL to what `_write_json` writes -- same object, same arguments --
        so with `--json` and a shell redirect but no `--summary-out` it is the only
        copy that reaches disk, and it passed no guard.

        Asserted two ways: the funnel refuses a bad payload (behaviour), and neither
        CLI serializes a shard except through it (call graph, both sites).
        """
        import ast
        import inspect
        import textwrap

        bridge = self._bridge()
        # BEHAVIOUR: the funnel refuses, and returns the same bytes it always did.
        good = self._payload(policy_stats={"decisions": 65})
        self.assertEqual(
            bridge._shard_json_text(good),
            json.dumps(good, indent=2, sort_keys=True),
            "the funnel must not change the bytes -- only refuse them",
        )
        with self.assertRaises(Exception):
            bridge._shard_json_text(
                self._payload(policy_stats=dict(BANKED_V1_ARM_SHARD))
            )
        # CALL GRAPH: no `json.dumps(payload, ...)` survives in either CLI.
        for entry in (bridge.async_main, bridge.async_comparison_main):
            with self.subTest(cli=entry.__name__):
                source = textwrap.dedent(inspect.getsource(entry))
                dumps_on_payload = [
                    node
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "dumps"
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "payload"
                ]
                self.assertEqual(
                    dumps_on_payload,
                    [],
                    "a shard payload must not be serialized outside "
                    "`_shard_json_text`; this is the stdout writer that reached disk "
                    "unguarded",
                )
                self.assertIn("_shard_json_text(payload)", source)

    def test_the_reader_finds_the_block_under_EVERY_writers_spelling(self) -> None:
        """B1. `_write_json` was not the only funnel, and `engine_mcts` was not the
        only spelling.

        Four writers put this mapping on disk under four different parent keys, and
        `require_banked_shard_witness` finds blocks by the PARENT's name -- so it
        walks past three of the four vacuously. The column-keyed walk finds all four,
        and would find a fifth writer's fifth name without being told about it.
        """
        from pokezero import engine_search

        v1 = dict(BANKED_V1_ARM_SHARD)
        layouts = {
            "engine_mcts.policy_stats (foulplay_bridge._write_json)": {
                "engine_mcts": {"decisions": 65, "policy_stats": v1}
            },
            "per_seat[seat].policy_stats (scripts/foulplay_paired_eval.py)": {
                "per_seat": {"p1": {"policy_stats": v1}}
            },
            "root policy_stats (scripts/mcts_acceptance_h2h.py)": {"policy_stats": v1},
            "engine_stats (scripts/hc_depth_grid.py)": {"engine_stats": v1},
            "a list of arms (the comparison summaries)": {"arms": [{"policy_stats": v1}]},
        }
        for label, document in layouts.items():
            with self.subTest(layout=label):
                self.assertEqual(
                    len(engine_search.iter_rollout_leaf_shard_blocks(document)),
                    1,
                    "the block must be found by its own columns, not by its parent",
                )
                with self.assertRaises(engine_search.EngineShardSchemaError):
                    engine_search.require_rollout_leaf_document_schema(document)
        # AND IT DOES NOT FIRE ON THE CONFIG ECHO. The bridge's `engine_mcts` block and
        # the per-decision metadata both carry rollout-PREFIXED knobs; a
        # `startswith("rollout")` walk would refuse every legitimate payload as an
        # unstamped shard.
        echo = {
            "engine_mcts": {
                "decisions": 65,
                "rollout_leaf": True,
                "rollout_count": 8,
                "rollout_max_plies": 200,
                "policy_stats": {"decisions": 65},
            }
        }
        self.assertEqual(engine_search.iter_rollout_leaf_shard_blocks(echo), [])
        engine_search.require_rollout_leaf_document_schema(echo)

    def test_the_write_refuses_an_arm_claim_with_no_pricer(self) -> None:
        """THE ASKED-FOR SIDE AGAINST THE RAN SIDE.

        The sibling branch's own docstring names this hazard and leaves it
        unenforced -- "a shard with `rollout_leaf: true` here and no leaf modes there
        ran the value head". Enforced here, at the write, because a shard is what a
        campaign sorts cells on.
        """
        bridge = self._bridge()
        # The v2 writer's own flag-off shape: conditional emission, so an arm run that
        # never engaged the seam carries NO rollout columns at all.
        with self.assertRaises(bridge.BankedShardWitnessError) as caught:
            bridge.require_banked_shard_witness(
                self._payload(policy_stats={"decisions": 65}, rollout_leaf=True)
            )
        self.assertIn("no pricer ever ran", str(caught.exception))

    def test_the_crosscheck_refuses_a_CONTROL_MODE_shard_claiming_the_arm(self) -> None:
        """The cross-check was UNDER-BROAD, in the case its own error text names.

        `engaged = bool(stats.get("rollout_leaf_modes"))` reads True on
        `{"model_value": 9}`. But `model_value` is the CONTROL mode: it routes leaf
        values through the arm's deferred-row plumbing while keeping PRODUCTION'S LEAF
        VALUE. So a shard with `rollout_leaf: true` and that mapping is a run whose
        "leaves were priced by the MODEL" -- the exact sentence the refusal prints --
        and it passed. It was caught only ONE FRAME IN, by
        `require_rollout_leaf_witness`'s per-decision `wrong` check, which never runs
        on a loaded artifact and therefore cannot see a banked shard at all.

        Checked by VALUE now: engaged means the ARM'S pricer is named.
        """
        bridge = self._bridge()
        for label, modes in (
            ("the control mode", {"model_value": 9}),
            ("a crate gate fixture", {"hp_fraction": 9}),
            ("control and gate together, no arm", {"model_value": 4, "hp_fraction": 5}),
        ):
            with self.subTest(modes=label):
                with self.assertRaises(bridge.BankedShardWitnessError) as caught:
                    bridge.require_banked_shard_witness(
                        self._payload(
                            policy_stats={
                                "decisions": 65,
                                "worlds_searched": 65,
                                "rollout_leaf_modes": modes,
                            },
                            rollout_leaf=True,
                        )
                    )
                self.assertIn("no pricer ever ran", str(caught.exception))
        # And a mapping that DOES name the arm passes, so the check is not refusing
        # every non-empty mapping.
        bridge.require_banked_shard_witness(
            self._payload(
                policy_stats={
                    "decisions": 65,
                    "worlds_searched": 65,
                    "rollout_leaf_modes": {"rollout": 65},
                },
                rollout_leaf=True,
            )
        )

    def test_an_ALL_WORLDS_FAILED_arm_run_is_still_bankable(self) -> None:
        """And the cross-check was narrowly OVER-BROAD, in the other direction.

        If every world failed, no crate report was ever absorbed, so
        `rollout_leaf_modes` is empty for a run that genuinely asked for the arm and
        genuinely ran it. Refusing that made a total-failure run UNBANKABLE, which
        pushes it to be relabelled as a raw-arm run -- the same corruption this guard
        exists to prevent, arrived at from the other side.

        The exemption is scoped to what the shard says in its own counters:
        `worlds_searched == 0` WITH recorded failure reasons. A zero-searched shard
        that records no reason is still refused, so the exemption is not a hole.
        """
        bridge = self._bridge()
        bridge.require_banked_shard_witness(
            self._payload(
                policy_stats={
                    "decisions": 65,
                    "worlds_searched": 0,
                    "worlds_constructed": 260,
                    "world_failure_reasons": {"crate_search: rollout dead end": 260},
                    "rollout_leaf_modes": {},
                },
                rollout_leaf=True,
            )
        )
        # No reason recorded -> still refused. "Nothing was searched" without a cause
        # is the silent shape, not the honest one.
        with self.assertRaises(bridge.BankedShardWitnessError):
            bridge.require_banked_shard_witness(
                self._payload(
                    policy_stats={
                        "decisions": 65,
                        "worlds_searched": 0,
                        "worlds_constructed": 260,
                        "world_failure_reasons": {},
                        "rollout_leaf_modes": {},
                    },
                    rollout_leaf=True,
                )
            )
        # And a shard that DID search worlds is still refused, so the exemption is
        # keyed on the failure and not on the empty mapping.
        with self.assertRaises(bridge.BankedShardWitnessError):
            bridge.require_banked_shard_witness(
                self._payload(
                    policy_stats={
                        "decisions": 65,
                        "worlds_searched": 65,
                        "world_failure_reasons": {"crate_search: something": 1},
                        "rollout_leaf_modes": {},
                    },
                    rollout_leaf=True,
                )
            )
        # And the sibling writer's shape: emitted unconditionally, so the columns are
        # PRESENT and zeroed with an empty mode map. Present keys are not engagement,
        # which is the same distinction `require_rollout_leaf_witness` makes per
        # decision -- so both writers' flag-off shards reach this branch.
        with self.assertRaises(bridge.BankedShardWitnessError) as caught:
            bridge.require_banked_shard_witness(
                self._payload(
                    policy_stats=dict(self._zeroed_v2_stats(), decisions=65),
                    rollout_leaf=True,
                )
            )
        self.assertIn("no pricer ever ran", str(caught.exception))

    def test_the_write_refuses_a_raw_row_carrying_arm_telemetry(self) -> None:
        """The inverse, and the one that corrupts the DENOMINATOR.

        `--arm raw --engine-rollout-leaf` was the live instance. A raw row wearing
        the arm's provenance does not corrupt one arm of the comparison; it corrupts
        the comparison.
        """
        bridge = self._bridge()
        stats = dict(self._v2_stats(), decisions=65)
        with self.assertRaises(bridge.BankedShardWitnessError) as caught:
            bridge.require_banked_shard_witness(
                self._payload(policy_stats=stats, rollout_leaf=False)
            )
        self.assertIn("FALSE WITNESS", str(caught.exception))

    def test_both_agreeing_directions_are_accepted(self) -> None:
        """Discriminating power for the cross-check, and the compatibility statement.

        `rollout_leaf` ABSENT gets no cross-check at all: a writer that does not echo
        the knob is not saying the arm was off. That is what lets this branch (which
        emits no bridge-level flag) and the sibling (which does) both pass.
        """
        bridge = self._bridge()
        engaged = dict(self._v2_stats(), decisions=65)
        bridge.require_banked_shard_witness(
            self._payload(policy_stats=engaged, rollout_leaf=True)
        )
        bridge.require_banked_shard_witness(
            self._payload(policy_stats={"decisions": 57}, rollout_leaf=False)
        )
        bridge.require_banked_shard_witness(
            self._payload(
                policy_stats=dict(self._zeroed_v2_stats(), decisions=57),
                rollout_leaf=False,
            )
        )
        # Absent: no cross-check, either way.
        bridge.require_banked_shard_witness(self._payload(policy_stats=engaged))
        bridge.require_banked_shard_witness(
            self._payload(policy_stats={"decisions": 57})
        )

    @staticmethod
    def _zeroed_v2_stats() -> dict:
        """The SIBLING writer's flag-off shape, in v2 spelling.

        #1272 emits the block unconditionally, so its flag-off shards carry every
        column zeroed with an empty mode map rather than carrying nothing. Stamped,
        because it adopts the schema -- so it is a well-formed v2 block that says the
        seam never engaged, which is exactly what the cross-check has to read.
        """
        return {
            "rollout_leaf_schema": ROLLOUT_LEAF_SHARD_SCHEMA,
            "rollout_leaf_modes": {},
            "rollout_leaf_worlds": 0,
            "rollout_leaves_priced": 0,
            "rollouts_run": 0,
            "rollout_plies": 0,
            "rollout_terminal_hits": 0,
            "rollout_cap_hits": 0,
            "rollout_dead_ends": 0,
            "rollout_encode_skipped": 0,
            "rollout_terminal_fraction": None,
            "rollout_fallback_fraction": None,
            "rollout_mean_plies": None,
        }

    @staticmethod
    def _v2_stats() -> dict:
        """A v2-schema policy_stats block whose seam engaged, from the real writer."""
        stats = EngineMctsStats()
        stats.rollout_leaf_modes["rollout"] = 65
        stats.rollout_leaf_worlds = 65
        stats.rollouts_run = 71832
        stats.rollout_plies = 3935252
        stats.rollout_terminal_hits = 70529
        stats.rollout_cap_hits = 1303
        stats.rollout_dead_ends = 0
        stats.rollout_leaves_priced = 8979
        stats.rollout_encode_skipped = 47
        return stats.to_dict()

    def test_the_boundary_write_calls_the_guard(self) -> None:
        """The guard has to be ON the write, not merely importable.

        Asserted by driving `_write_json` at a temp path with the offending payload:
        a guard that exists and is not called is the shape this whole finding is.
        """
        import tempfile
        from pathlib import Path

        bridge = self._bridge()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shard.json"
            with self.assertRaises(bridge.BankedShardWitnessError):
                bridge._write_json(target, self._payload(policy_stats=None))
            self.assertFalse(
                target.exists(),
                "the refusal must happen BEFORE the bytes land; a shard that is "
                "written and then complained about is a banked shard",
            )
            bridge._write_json(target, self._payload(policy_stats={"decisions": 1}))
            self.assertTrue(target.exists())


class EverySeventhWriterIsPinnedTest(unittest.TestCase):
    """The four guard call sites review measured as FREE DELETIONS, plus the
    seventh writer the enumeration missed.

    THE FINDING, recorded as measured rather than as described. Review deleted the
    `require_rollout_leaf_document_schema` calls at `scripts/foulplay_paired_eval.py`,
    `scripts/mcts_acceptance_h2h.py`, `scripts/hc_depth_grid.py` and
    `src/pokezero/engine_search.py` -- individually and all four together -- and got
    ZERO semantic failures across the whole suite. The only red was the battery's
    sha256 content pin, which pins the bytes and not the behaviour: it fires for a
    whitespace change and says nothing about whether the refusal still runs. Four of
    seven guard sites were therefore decorative, and the battery's `targets` list
    only four FILES, so `hc_depth_grid.py` and `mcts_acceptance_h2h.py` had never
    been mutated at all.

    AND THERE WAS A SEVENTH WRITER. `engine_search.main`'s
    `print(json.dumps(printable, indent=2))` is an unconditional top-level statement;
    the refusal sat nine lines below it inside `if args.out:`. With a shell redirect
    and no `--out` that document reached disk unguarded -- verbatim the shape
    `_shard_json_text`'s own docstring names. Latent (no `add_argument` exposes the
    rollout arm on this CLI), so it was a miscount of the writer surface rather than a
    live corruption path, and it is closed as a miscount: the guard is hoisted above
    both emissions.

    THE FIX IS STRUCTURAL, not another guard call. All four sites now write THROUGH
    `write_guarded_document`, so "the shard is refused before it is written" is a
    property of the call graph rather than of four statements a mutation can lift
    out one at a time.
    """

    def test_the_document_funnel_REFUSES_BEFORE_IT_WRITES(self) -> None:
        """Behaviour, and the reason the refusal precedes the `open`.

        This is the ONE remaining deletable statement for all four writers, so it is
        driven directly rather than asserted structurally. A half-written shard is
        worse than no shard: it is the one a pooling reader might still parse.
        """
        import tempfile
        from pathlib import Path

        v1_block = {
            "rollout_leaf_world_records": 4,
            "rollouts_run": 12,
            "rollout_cap_hits": 1,
            "rollout_dead_ends": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shard.json"
            with self.assertRaises(engine_search.EngineShardSchemaError):
                engine_search.write_guarded_document(
                    target, {"engine_mcts": {"policy_stats": v1_block}}
                )
            self.assertFalse(
                target.exists(),
                "the refusal must happen BEFORE the bytes land; a document that is "
                "written and then complained about is a banked document",
            )
            # And the positive control, so this is not a function that refuses
            # everything.
            engine_search.write_guarded_document(target, {"engine_mcts": {"games": 1}})
            self.assertTrue(target.exists())

        # `guarded_document_text` is the same refusal without the file, and the
        # stdout writers use it; deleting the refusal in EITHER must be caught.
        with self.assertRaises(engine_search.EngineShardSchemaError):
            engine_search.guarded_document_text({"policy_stats": v1_block})
        self.assertEqual(
            engine_search.guarded_document_text({"a": 1}, indent=None),
            '{"a": 1}',
        )

    def test_EVERY_document_writer_writes_THROUGH_the_funnel(self) -> None:
        """The call graph, per site, so reverting one site to a raw write is caught.

        Deleting the refusal is no longer expressible at these four sites; what IS
        still expressible is reverting one of them to `write_text(json.dumps(...))`.
        That is the mutation this asserts against, keyed on the writing frame rather
        than on the file, so a fifth writer added to one of these modules does not
        pass vacuously.
        """
        import ast
        import inspect
        import textwrap

        paired = _paired_eval_module()
        h2h = _acceptance_h2h_module()
        grid = _depth_grid_module()
        for label, function in (
            ("scripts/foulplay_paired_eval.py::main", paired.main),
            ("scripts/mcts_acceptance_h2h.py::main", h2h.main),
            ("scripts/hc_depth_grid.py::main", grid.main),
            ("src/pokezero/engine_search.py::main", engine_search.main),
        ):
            with self.subTest(writer=label):
                source = textwrap.dedent(inspect.getsource(function))
                tree = ast.parse(source)
                funnelled = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "write_guarded_document"
                ]
                self.assertTrue(
                    funnelled,
                    f"{label} must reach disk through `write_guarded_document`; a "
                    "bare guard call beside the write was measured as a free "
                    "deletion at exactly this site",
                )
                # NO RAW ESCAPE HATCH in the same frame. `write_text(json.dumps(...))`
                # and `json.dump(...)` are the two spellings that bypass the funnel.
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    if isinstance(func, ast.Attribute) and func.attr == "write_text":
                        rendered = ast.dump(node)
                        self.assertNotIn(
                            "'dumps'",
                            rendered,
                            f"{label} renders a document to disk without the funnel",
                        )
                    if isinstance(func, ast.Attribute) and func.attr == "dump":
                        self.assertNotEqual(
                            getattr(func.value, "id", None),
                            "json",
                            f"{label} uses `json.dump` to write past the funnel",
                        )

    def test_the_STDOUT_arm_of_the_engine_search_CLI_is_guarded_TOO(self) -> None:
        """THE SEVENTH WRITER. The refusal must dominate the unconditional print.

        Not "the call exists somewhere in `main`" -- that was already true while the
        stdout arm was open, because the call sat inside `if args.out:`. What is
        asserted is DOMINANCE: the refusal is a statement of `main`'s own body (not
        nested in any `if`), and it comes before the `print(json.dumps(...))`.
        """
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(engine_search.main))
        (function,) = [
            node
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        def _top_level_index(predicate) -> int | None:
            for index, statement in enumerate(function.body):
                for node in ast.walk(statement):
                    if predicate(node):
                        # Only if the statement ITSELF is unnested at `main`'s level.
                        return index
            return None

        guard_at = _top_level_index(
            lambda node: isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require_rollout_leaf_document_schema"
        )
        print_at = _top_level_index(
            lambda node: isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and any(
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Attribute)
                and arg.func.attr == "dumps"
                for arg in node.args
            )
        )
        self.assertIsNotNone(guard_at, "the CLI must refuse its own document")
        self.assertIsNotNone(print_at, "expected the unconditional json print")

        # THE GUARD'S OWN STATEMENT MUST BE UNCONDITIONAL. This is the assertion that
        # fails if the refusal is pushed back inside `if args.out:`.
        guard_statement = function.body[guard_at]
        self.assertIsInstance(
            guard_statement,
            ast.Expr,
            "the refusal must be a bare statement of `main`, not nested in a branch; "
            "inside `if args.out:` it leaves the stdout writer unguarded",
        )
        self.assertLess(
            guard_at,
            print_at,
            "the refusal must DOMINATE the unconditional `print(json.dumps(...))`; "
            "with a shell redirect and no `--out` that print is the only copy that "
            "reaches disk",
        )

    def test_the_battery_targets_EVERY_file_it_claims_to_cover(self) -> None:
        """"Nine mutants across every writer, all KILLED" overstated its scope.

        The battery's `targets` named four files, so two of the writer modules were
        never mutated at all and their guard sites could not have been measured. The
        target list must contain every module that carries a guard call site.
        """
        import importlib.util
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        path = root / "scripts" / "mutate_rollout_leaf_witness.py"
        spec = importlib.util.spec_from_file_location("_battery_under_test", path)
        battery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(battery)

        # READ FROM `ALL_TARGETS`, not from the file's text. `ALL_TARGETS` is derived
        # from the mutants' own edit paths, so this reads what the sweep would
        # actually touch; a `grep` for the path string would pass on a module that is
        # merely MENTIONED in a comment and never mutated, which is the failure mode
        # being closed here.
        targets = {Path(target).resolve() for target in battery.ALL_TARGETS}
        for module in (
            "src/pokezero/engine_search.py",
            "src/pokezero/foulplay_bridge.py",
            "scripts/foulplay_paired_eval.py",
            "scripts/foulplay_power_report.py",
            "scripts/mcts_acceptance_h2h.py",
            "scripts/hc_depth_grid.py",
        ):
            with self.subTest(module=module):
                self.assertIn(
                    (root / module).resolve(),
                    targets,
                    f"{module} carries a guard call site and is not a battery "
                    "target, so a mutation there would report NOT APPLIED and the "
                    "sweep would read as clean. '\"Nine mutants across every "
                    "writer, all KILLED\"' was true of the mutants written and "
                    "overstated as to scope.",
                )


class TheUnstampedBlockAndTheDeadEndTermTest(unittest.TestCase):
    """Two survivors from the independent battery, closed at their own sites.

    Both are the same class of gap the body already names once: a term or a set
    whose only exercised value is the one where the mutation makes no difference.
    """

    def test_an_UNSTAMPED_rollout_block_is_found_by_its_OWN_columns(self) -> None:
        """Narrowing `ROLLOUT_LEAF_SHARD_MARKERS` to the stamp survived.

        Every existing fixture's block carries either the stamp
        (`rollout_leaf_schema`) or the v1 world field, so a marker set narrowed to
        just those two finds every fixture and the whole point of a MARKER SET --
        that an UNSTAMPED block is still recognised as a rollout block, by the
        columns it does carry -- rested on nothing.

        That case is the dangerous one: a block with the rollout counts and no stamp
        is exactly what a pre-seam writer, or a writer whose stamp was dropped,
        produces. If it is not FOUND it is not refused, and it pools silently.
        """
        # No `rollout_leaf_schema`, no `rollout_leaf_world_records` -- identified
        # only by the partition columns.
        unstamped = {
            "rollouts_run": 12,
            "rollout_cap_hits": 1,
            "rollout_terminal_hits": 11,
            "rollout_dead_ends": 0,
        }
        found = engine_search.iter_rollout_leaf_shard_blocks(
            {"engine_mcts": {"policy_stats": unstamped}}
        )
        self.assertEqual(
            len(found),
            1,
            "an unstamped rollout block must still be FOUND by its own columns; a "
            "marker set narrowed to the stamp cannot see it, and an unfound block "
            "is an unrefused one",
        )
        with self.assertRaises(engine_search.EngineShardSchemaError) as caught:
            engine_search.require_rollout_leaf_document_schema(
                {"engine_mcts": {"policy_stats": unstamped}}
            )
        self.assertIn("rollout_leaf_schema", str(caught.exception))

        # Per-column, so the set cannot be narrowed to any SINGLE marker and pass.
        for column in (
            "rollout_leaf_modes",
            "rollout_leaf_worlds",
            "rollout_leaves_priced",
            "rollouts_run",
            "rollout_plies",
            "rollout_terminal_hits",
            "rollout_cap_hits",
            "rollout_dead_ends",
            "rollout_encode_skipped",
            "rollout_terminal_fraction",
            "rollout_fallback_fraction",
            "rollout_mean_plies",
        ):
            with self.subTest(marker=column):
                self.assertEqual(
                    len(
                        engine_search.iter_rollout_leaf_shard_blocks(
                            {"engine_mcts": {"policy_stats": {column: 0}}}
                        )
                    ),
                    1,
                    f"{column} is a declared shard field and must identify a block "
                    "on its own",
                )
        # THE NEGATIVE SIDE, so this is not satisfied by `startswith('rollout')`:
        # the config echoes must NOT be read as a shard block.
        for echo in ("rollout_leaf", "rollout_count", "rollout_max_plies"):
            with self.subTest(config_echo=echo):
                self.assertEqual(
                    engine_search.iter_rollout_leaf_shard_blocks({echo: True}),
                    [],
                    f"{echo} is a CONFIG echo, not a shard column; treating it as "
                    "one refuses every legitimate payload",
                )

    def test_the_PER_DECISION_fallback_numerator_counts_DEAD_ENDS(self) -> None:
        """`(cap + dead) / denominator` at the per-decision writer survived.

        This is the FOURTH site carrying the `cap + dead` rule; the body counts
        three, and the AST guard's reversal was justified by "#1271 makes the term
        testable at the writer instead" -- true at the shard reader and the shard
        writer, and NOT true here, because no fixture anywhere in `tests/` drives
        `_rollout_leaf_witness` with a non-zero `rollout_dead_ends`. Every fixture
        sat at `dead == 0`, where `cap` and `cap + dead` are the same number.

        The partition is balanced (`terminal + cap + dead == rollouts_run`), so the
        numerator rule is the only thing that can make this fail.
        """
        modes = {"model": 3}
        ledger = {
            "worlds": 3,
            "leaves_priced": 9,
            "rollouts_run": 20,
            "rollout_plies": 100,
            "rollout_encode_skipped": 0,
            "rollout_terminal_hits": 11,
            "rollout_cap_hits": 4,
            # THE FIELD THAT WAS ALWAYS ZERO. Non-zero here and nowhere else.
            "rollout_dead_ends": 5,
        }
        self.assertEqual(
            ledger["rollout_terminal_hits"]
            + ledger["rollout_cap_hits"]
            + ledger["rollout_dead_ends"],
            ledger["rollouts_run"],
            "the fixture's own partition must balance, or this measures nothing",
        )
        witness = EngineMctsPolicy._rollout_leaf_witness(None, modes, ledger)
        self.assertEqual(witness["rollout_cap_hits"], 4)
        self.assertEqual(witness["rollout_dead_ends"], 5)
        # (4 + 5) / 20 == 0.45; cap alone would read 0.20.
        self.assertEqual(witness["rollout_fallback_fraction"], 0.45)
        self.assertNotEqual(
            witness["rollout_fallback_fraction"],
            ledger["rollout_cap_hits"] / ledger["rollouts_run"],
            "a cap-only numerator must not reproduce this value",
        )
        # And the three quotients still partition to 1.
        self.assertAlmostEqual(
            witness["rollout_terminal_fraction"] + witness["rollout_fallback_fraction"],
            1.0,
        )


#: Sentinel for "the key is absent", distinct from `None` -- the two are different
#: mutants (`-> None` versus deleting the line) and each must be refused on its own.
_MISSING = object()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
