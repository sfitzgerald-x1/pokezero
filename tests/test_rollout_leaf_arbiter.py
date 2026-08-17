"""Gates for the oracle-leaf arbiter arm (search-ceiling program, Phase 1
instrument 2): the R-rollout leaf pricer and its batching seam.

Two things are under test, and they are different in kind:

1. **Fidelity.** The arm must differ from the production crate search in leaf
   VALUATION and nothing else. The Rust-side gate
   (`rollout::tests::rollout_batch1_matches_sequential_report`) compares the two
   drivers' full reports field for field; this file re-derives the same claim
   across the Python boundary on a REAL adapter-built state, so a wheel that
   was built from different sources than the one `cargo test` ran cannot pass
   silently.

2. **Guards, each with a demonstrated failing input.** Program-wide rule from
   the 2026-08-17 amendment: "a check that cannot read False certifies
   nothing". Every rejection below is asserted twice — once on the input that
   trips it, once on the neighbouring input that must NOT.

Skips unless the native crate imports, matching the conventions in
tests/test_multiply_chance_search.py.
"""

from __future__ import annotations

import json
import unittest

try:  # pragma: no cover - exercised only when the native crate is built
    import pokezero_search
except ImportError:  # pragma: no cover
    pokezero_search = None  # type: ignore[assignment]


HAS_ROLLOUT = pokezero_search is not None and hasattr(
    pokezero_search, "puct_search_multi_rollout"
)

#: Fields allowed to differ between the two drivers on the same search: wall
#: clock, the evaluator label, and the rollout-only accounting.
_VOLATILE = frozenset(
    {
        "elapsed_s",
        "iterations_per_s",
        "evaluator",
        "rollouts",
        "rollout_policy",
        "rollout_max_plies",
        "rollout_seed",
        "rollout_threads",
        "leaf_batch",
        "rounds",
        "leaves_priced",
        "rollouts_run",
        "rollout_plies",
        "rollout_terminal_hits",
        "rollout_cap_hits",
        "rollout_dead_ends",
        "rollout_terminal_fraction",
        "rollout_fallback_fraction",
        "rollout_mean_plies",
    }
)


def _normalize(report: str) -> dict:
    return {k: v for k, v in json.loads(report).items() if k not in _VOLATILE}


def _state() -> str:
    """A gen3 state built through the shared adapter, same recipe as
    tests/test_multiply_chance_search.py's ``_build_state``."""

    from pokezero.poke_engine_adapter import (
        BattleSpec,
        MoveSpec,
        PokemonSpec,
        SideSpec,
        build_poke_engine_state,
    )

    def mon(species, moves, *, hp=100, speed=100):
        return PokemonSpec(
            id=species,
            level=100,
            types=("normal",),
            hp=hp,
            maxhp=100,
            attack=100,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=speed,
            status="none",
            moves=tuple(MoveSpec(id=name, pp=32) for name in moves),
        )

    spec = BattleSpec(
        side_one=SideSpec(pokemon=(mon("rattata", ("toxic", "seismictoss"), speed=200),)),
        side_two=SideSpec(pokemon=(mon("chansey", ("splash", "tackle")),)),
    )
    return build_poke_engine_state(spec).to_string()


@unittest.skipUnless(HAS_ROLLOUT, "pokezero_search crate with the rollout seam not built")
class RolloutSeamFidelityTests(unittest.TestCase):
    def test_batch1_hp_fraction_reproduces_production_search(self) -> None:
        """THE FIDELITY CHECK, across the Python boundary.

        Same state, same sims, same depth, same c_puct, same seed: the rollout
        driver at ``leaf_batch=1`` with the leaf pricer held at production's
        HP-fraction evaluator must reproduce ``puct_search_multi``'s report --
        every visit count, every Q, the depth occupancy, the expansion and
        leaf-eval counters, the root value.
        """

        state = _state()
        for iterations, depth, seed in ((512, 2, 3), (1_024, 4, 11)):
            with self.subTest(iterations=iterations, depth=depth, seed=seed):
                production = pokezero_search.puct_search_multi(
                    state, iterations, max_depth=depth, c_puct=1.4, seed=seed
                )
                arm = pokezero_search.puct_search_multi_rollout(
                    state,
                    iterations,
                    max_depth=depth,
                    c_puct=1.4,
                    seed=seed,
                    # Deliberately non-default rollout knobs: under
                    # leaf_mode="hp_fraction" they must be inert, and this is
                    # what would catch a pricer leaking into the tree.
                    rollouts=7,
                    rollout_max_plies=9,
                    rollout_seed=98765,
                    rollout_threads=3,
                    leaf_batch=1,
                    leaf_mode="hp_fraction",
                )
                self.assertEqual(_normalize(production), _normalize(arm))

    def test_fidelity_check_reads_false_on_batching_and_on_pricer(self) -> None:
        """The fidelity check's DEMONSTRATED FAILING INPUTS.

        Both mutations must break the equality asserted above, or that
        assertion is measuring something insensitive to the thing it claims:

        * ``leaf_batch=8`` -- virtual-loss batching is a real selection change.
        * ``leaf_mode="rollout"`` -- if swapping the leaf pricer left the report
          unchanged, leaf values would not be reaching the tree at all.
        """

        state = _state()
        production = pokezero_search.puct_search_multi(
            state, 1_024, max_depth=4, c_puct=1.4, seed=11
        )
        batched = pokezero_search.puct_search_multi_rollout(
            state,
            1_024,
            max_depth=4,
            c_puct=1.4,
            seed=11,
            rollouts=4,
            rollout_max_plies=40,
            leaf_batch=8,
            leaf_mode="hp_fraction",
        )
        self.assertNotEqual(_normalize(production), _normalize(batched))
        priced = pokezero_search.puct_search_multi_rollout(
            state,
            1_024,
            max_depth=4,
            c_puct=1.4,
            seed=11,
            rollouts=4,
            rollout_max_plies=40,
            leaf_batch=1,
            leaf_mode="rollout",
        )
        self.assertNotEqual(_normalize(production), _normalize(priced))

    def test_thread_count_does_not_move_the_priced_values(self) -> None:
        """Threads are a throughput knob. The report must be identical."""

        state = _state()
        kwargs = dict(
            max_depth=3,
            c_puct=1.4,
            seed=5,
            rollouts=8,
            rollout_max_plies=60,
            rollout_seed=4242,
            leaf_batch=1,
            leaf_mode="rollout",
        )
        one = json.loads(
            pokezero_search.puct_search_multi_rollout(state, 300, rollout_threads=1, **kwargs)
        )
        many = json.loads(
            pokezero_search.puct_search_multi_rollout(state, 300, rollout_threads=6, **kwargs)
        )
        for report in (one, many):
            for key in ("elapsed_s", "iterations_per_s", "rollout_threads"):
                report.pop(key, None)
        self.assertEqual(one, many)

    def test_report_carries_the_cost_and_honesty_ledger(self) -> None:
        """Every estimate quoted off this arm must be reconstructible from the
        artifact: the rollout count, what those rollouts cost, and what
        fraction of them actually reached a terminal rather than falling back
        to the handcrafted evaluator."""

        report = json.loads(
            pokezero_search.puct_search_multi_rollout(
                _state(),
                256,
                max_depth=3,
                seed=1,
                rollouts=8,
                rollout_max_plies=200,
                leaf_mode="rollout",
            )
        )
        for key in (
            "rollouts",
            "rollouts_run",
            "leaves_priced",
            "rollout_plies",
            "rollout_terminal_fraction",
            "rollout_fallback_fraction",
            "rollout_mean_plies",
            "elapsed_s",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["rollouts_run"], report["leaves_priced"] * 8)
        self.assertAlmostEqual(
            report["rollout_terminal_fraction"] + report["rollout_fallback_fraction"],
            1.0,
            places=5,
        )

    def test_a_one_ply_cap_is_reported_as_a_blend_not_an_oracle(self) -> None:
        """DEMONSTRATED FAILING INPUT for the fallback-fraction column: a
        1-ply cap makes essentially every rollout a handcrafted evaluation, and
        the column must say so. A column that read ~0 here would let a blend be
        reported as an oracle."""

        capped = json.loads(
            pokezero_search.puct_search_multi_rollout(
                _state(), 128, max_depth=2, seed=1, rollouts=4,
                rollout_max_plies=1, leaf_mode="rollout",
            )
        )
        deep = json.loads(
            pokezero_search.puct_search_multi_rollout(
                _state(), 128, max_depth=2, seed=1, rollouts=4,
                rollout_max_plies=400, leaf_mode="rollout",
            )
        )
        self.assertGreater(capped["rollout_fallback_fraction"], 0.9)
        self.assertLess(deep["rollout_fallback_fraction"], 0.1)


@unittest.skipUnless(HAS_ROLLOUT, "pokezero_search crate with the rollout seam not built")
class RolloutBoundaryGuardTests(unittest.TestCase):
    """Crate-boundary rejections. Each asserted on the tripping input AND on
    the neighbour that must pass, so none of them is true by construction."""

    def _call(self, **overrides):
        kwargs = dict(
            max_depth=2,
            seed=0,
            rollouts=4,
            rollout_max_plies=20,
            rollout_policy="uniform",
            rollout_threads=1,
            leaf_batch=1,
            leaf_mode="rollout",
        )
        kwargs.update(overrides)
        return pokezero_search.puct_search_multi_rollout(_state(), 16, **kwargs)

    def test_valid_call_is_accepted(self) -> None:
        self.assertTrue(self._call())

    def test_zero_rollouts_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollouts must be > 0"):
            self._call(rollouts=0)

    def test_zero_ply_cap_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollout_max_plies must be > 0"):
            self._call(rollout_max_plies=0)

    def test_unknown_rollout_policy_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown rollout_policy"):
            self._call(rollout_policy="max_damage")

    def test_zero_leaf_batch_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "leaf_batch must be > 0"):
            self._call(leaf_batch=0)

    def test_unknown_leaf_mode_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown leaf_mode"):
            self._call(leaf_mode="oracle")


class RolloutConfigGuardTests(unittest.TestCase):
    """`EngineMctsConfig` validation for the arm. No crate needed."""

    @staticmethod
    def _config(**overrides):
        from pokezero.engine_search import EngineMctsConfig

        kwargs = dict(
            leaf_eval="rollout_crate",
            worlds=1,
            search_sims=64,
            search_depth=2,
        )
        kwargs.update(overrides)
        return EngineMctsConfig(**kwargs)

    def test_valid_arm_config_is_accepted(self) -> None:
        config = self._config()
        self.assertEqual(config.leaf_eval, "rollout_crate")
        self.assertEqual(config.rollout_threads, 1)
        self.assertEqual(config.leaf_batch, 1)

    def test_unknown_leaf_eval_still_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "leaf_eval must be"):
            self._config(leaf_eval="rollout")

    def test_zero_rollout_count_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollout_count must be > 0"):
            self._config(rollout_count=0)

    def test_zero_rollout_max_plies_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollout_max_plies must be > 0"):
            self._config(rollout_max_plies=0)

    def test_unimplemented_rollout_policy_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollout_policy must be 'uniform'"):
            self._config(rollout_policy="model")

    def test_multithreaded_arm_requires_the_cpu_budget_ack(self) -> None:
        """THE TIMING-HAZARD GUARD, with its demonstrated failing input.

        The paired-eval opponent is time-budgeted and thinks concurrently with
        this search on the same host, and the shard schema records only the
        budget it was given, never the work it achieved. So cores taken by this
        arm weaken the opponent in the direction that flatters the arm, and
        undetectably. Raising `rollout_threads` must therefore be an explicit
        act.
        """

        with self.assertRaisesRegex(ValueError, "requires rollout_threads_cpu_budget_ack"):
            self._config(rollout_threads=8)
        # ... and the ack is what makes it pass, so the guard is not simply
        # banning the field.
        acked = self._config(rollout_threads=8, rollout_threads_cpu_budget_ack=True)
        self.assertEqual(acked.rollout_threads, 8)
        # The ack is scoped to the hazard: it does nothing at one thread, and
        # it does not excuse any other invalid value.
        with self.assertRaisesRegex(ValueError, "rollout_count must be > 0"):
            self._config(rollout_threads=8, rollout_threads_cpu_budget_ack=True, rollout_count=0)

    def test_the_ack_is_inert_on_other_leaf_evals(self) -> None:
        """The arm's knobs must not reach any production path. A production
        `leaf_eval` accepts the arm's fields and ignores them, so a stray value
        cannot change what a production cell does."""

        from pokezero.engine_search import EngineMctsConfig

        production = EngineMctsConfig(
            leaf_eval="hp_fraction_crate",
            worlds=1,
            search_sims=64,
            search_depth=2,
            # Values that would be REFUSED on the arm.
            rollout_count=0,
            rollout_threads=32,
            rollout_threads_cpu_budget_ack=False,
        )
        self.assertEqual(production.leaf_eval, "hp_fraction_crate")

    def test_a_batched_arm_requires_the_fidelity_loss_ack(self) -> None:
        # The demonstrated failing input for the guard that was previously on the
        # wrong knob: `leaf_batch > 1` is the arm's ONLY uncertified selection
        # regime and used to be accepted silently, while `rollout_threads` --
        # proven value-invariant -- required an acknowledgement.
        for leaf_batch in (2, 3, 8, 64, 1000, 4096):
            with self.subTest(leaf_batch=leaf_batch), self.assertRaises(ValueError) as caught:
                self._config(leaf_batch=leaf_batch)
            message = str(caught.exception)
            self.assertIn("leaf_batch_fidelity_loss_ack=True", message)
            # The refusal must carry the MAGNITUDE, so a reader deciding whether
            # to set the ack sees what it costs rather than only that it is
            # uncertified. Asserted as the RANGE the sweep measured, not as a
            # single triple: an earlier revision pinned "8.2 pp" from one
            # unrecorded run and independent review could not reproduce it in
            # 9792 crate runs. A literal nobody can re-derive is worse than no
            # number, because it looks like evidence.
            self.assertIn("2.8-11.7 pp", message)

    def test_the_batch_ack_admits_the_regime_rather_than_banning_it(self) -> None:
        # A fence that cannot be opened would make the batching seam untestable;
        # the requirement is an explicit choice, not a prohibition.
        config = self._config(leaf_batch=8, leaf_batch_fidelity_loss_ack=True)
        self.assertEqual(config.leaf_batch, 8)

    def test_the_batch_ack_is_inert_on_other_leaf_evals(self) -> None:
        """A production `leaf_eval` ignores `leaf_batch` entirely, so a stray
        value cannot change what a production cell does."""

        from pokezero.engine_search import EngineMctsConfig

        for leaf_eval in ("hp_fraction", "hp_fraction_crate"):
            with self.subTest(leaf_eval=leaf_eval):
                production = EngineMctsConfig(
                    leaf_eval=leaf_eval,
                    worlds=1,
                    search_sims=64,
                    search_depth=2,
                    # Refused on the arm at any value above 1, accepted here.
                    leaf_batch=64,
                    leaf_batch_fidelity_loss_ack=False,
                )
                self.assertEqual(production.leaf_batch, 64)


class EveryTestInThisModuleHasABodyTest(unittest.TestCase):
    """A meta-gate, added because this module shipped a test that could not fail.

    Inserting three methods between
    `test_the_ack_is_inert_on_other_leaf_evals`'s docstring and its body left the
    named test as `RESUME; RETURN_CONST None` -- a test that passes
    unconditionally -- while the suite stayed green, because the orphaned
    assertions ran inside whichever method absorbed them. Independent review
    caught it by disassembling the bytecode.

    That is the same defect class this program has now found four times: a check
    that cannot read False. The lesson is not "insert methods more carefully",
    it is that a docstring-only test body is mechanically detectable, so it
    should be detected mechanically.

    Scope, stated so this is not mistaken for a general style rule: it applies to
    THIS module only. A deliberately empty test elsewhere is somebody else's
    decision.
    """

    def _empty_test_methods(self, module) -> list[str]:
        """Names of `test_*` methods whose body is only a docstring (or `pass`).

        Detected from the CODE OBJECT rather than the source text, because the
        failure was invisible in a diff -- the source read as two well-formed
        methods -- and only the bytecode showed the body was gone.
        """
        import dis
        import inspect

        def is_stub(func) -> bool:
            code = getattr(func, "__code__", None)
            if code is None:
                return False
            # Read the INSTRUCTIONS, not `co_consts`. A docstring-only body keeps
            # its docstring in `co_consts`, so a "no constants" test never fires
            # on the very shape this gate exists to catch -- the first draft of
            # this detector made exactly that mistake and its own demonstrated
            # failing input caught it, which is the argument for shipping one.
            #
            # What both stub spellings reduce to is: no operation except
            # returning a constant. `RESUME`/`NOP`/`CACHE` are bookkeeping.
            for instruction in dis.get_instructions(code):
                name = instruction.opname
                if name in ("RESUME", "NOP", "CACHE", "RETURN_CONST"):
                    continue
                if name == "LOAD_CONST":
                    continue
                if name == "RETURN_VALUE":
                    continue
                return False
            return True

        empty = []
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, unittest.TestCase) or cls is unittest.TestCase:
                continue
            if cls.__module__ != module.__name__:
                continue
            for name, func in vars(cls).items():
                if not name.startswith("test_") or not callable(func):
                    continue
                if is_stub(func):
                    empty.append(f"{cls.__name__}.{name}")
        return empty

    def test_no_test_in_this_module_is_a_docstring_only_stub(self) -> None:
        import sys

        module = sys.modules[type(self).__module__]
        self.assertEqual(
            self._empty_test_methods(module),
            [],
            "a test whose body is only a docstring passes unconditionally and "
            "certifies nothing; its assertions were probably absorbed by the "
            "method below it",
        )

    def test_the_meta_gate_reads_false_on_a_severed_test(self) -> None:
        """The demonstrated failing input, which is the whole point.

        Reconstructs the exact shape the review found -- a `test_*` method
        holding only a docstring -- and asserts the detector names it. Without
        this, the meta-gate above would be one more check nobody had shown
        capable of failing.
        """

        import types

        module = types.ModuleType("severed_fixture")

        class Severed(unittest.TestCase):
            def test_looks_fine_but_has_no_body(self) -> None:
                """Docstring only -- exactly what the review disassembled."""

            def test_really_asserts_something(self) -> None:
                self.assertEqual(1, 1)

        Severed.__module__ = "severed_fixture"
        module.Severed = Severed

        found = self._empty_test_methods(module)
        self.assertIn("Severed.test_looks_fine_but_has_no_body", found)
        self.assertNotIn("Severed.test_really_asserts_something", found)

    def test_the_meta_gate_does_not_flag_a_pass_only_test_as_clean(self) -> None:
        """`pass` is the other spelling of the same stub, so it must also read
        False -- otherwise the fix for a flagged test is to write `pass`."""

        import types

        module = types.ModuleType("pass_fixture")

        class PassOnly(unittest.TestCase):
            def test_pass_only(self) -> None:
                pass

        PassOnly.__module__ = "pass_fixture"
        module.PassOnly = PassOnly
        self.assertIn("PassOnly.test_pass_only", self._empty_test_methods(module))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
