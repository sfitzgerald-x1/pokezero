"""Rollout leaves composed with MODEL PRIORS, and the fidelity gate re-run on
the model driver.

Search-ceiling program (`docs/search-ceiling-program-20260816.md`) Phase 1
instrument 2 -- the arbiter. `tests/test_rollout_leaf_arbiter.py` already
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

import json
import unittest

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover
    pokezero_search = None

from test_model_priors_search import _EncodedSearchFixture, _crate_ready

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
