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
HAS_UNIFORM_ROW_PRICER = pokezero_search is not None and hasattr(
    pokezero_search, "price_uniform_rollout_rows"
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


@unittest.skipUnless(
    HAS_UNIFORM_ROW_PRICER,
    "pokezero_search crate with the direct uniform row pricer not built",
)
class UniformRowPricerTests(unittest.TestCase):
    """The public writer's native seam, separate from tree traversal.

    This is not a small variation on ``puct_search_multi_rollout``: it must
    price exactly the supplied successor states, return their one-for-one
    values, and expose the terminal/fallback ledger used by the artifact.
    """

    @staticmethod
    def _price(*, threads: int = 1, **overrides):
        kwargs = dict(
            rollouts=4,
            rollout_max_plies=60,
            rollout_seed=1234,
            rollout_threads=threads,
            rollout_branch_on_damage=False,
        )
        kwargs.update(overrides)
        return json.loads(
            pokezero_search.price_uniform_rollout_rows(
                [_state(), _state()], [17, 29], **kwargs
            )
        )

    def test_prices_only_the_supplied_rows_and_reports_the_complete_ledger(self) -> None:
        report = self._price()

        self.assertEqual(report["schema"], "pokezero.uniform-rollout-row-prices.v1")
        self.assertEqual(report["value_frame"], "side_one_absolute")
        self.assertEqual(report["rollout_policy"], "uniform")
        self.assertEqual(report["leaves_priced"], 2)
        self.assertEqual(report["rollouts_run"], 8)
        self.assertEqual(len(report["values"]), 2)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in report["values"]))
        self.assertEqual(
            report["rollout_terminal_hits"]
            + report["rollout_cap_hits"]
            + report["rollout_dead_ends"],
            report["rollouts_run"],
        )

    def test_thread_count_does_not_change_direct_row_values(self) -> None:
        one = self._price(threads=1)
        many = self._price(threads=6)
        one.pop("rollout_threads")
        many.pop("rollout_threads")
        self.assertEqual(one, many)

    def test_refuses_wrong_length_duplicate_ordinals_and_degenerate_config(self) -> None:
        state = _state()
        with self.assertRaisesRegex(ValueError, "every state needs one stable ordinal"):
            pokezero_search.price_uniform_rollout_rows([state], [1, 2])
        with self.assertRaisesRegex(ValueError, "duplicate 17"):
            pokezero_search.price_uniform_rollout_rows([state, state], [17, 17])
        with self.assertRaisesRegex(ValueError, "rollouts must be > 0"):
            pokezero_search.price_uniform_rollout_rows([state], [17], rollouts=0)
        with self.assertRaisesRegex(ValueError, "rollout_max_plies must be > 0"):
            pokezero_search.price_uniform_rollout_rows([state], [17], rollout_max_plies=0)
        with self.assertRaisesRegex(ValueError, "rollout_threads must be > 0"):
            pokezero_search.price_uniform_rollout_rows([state], [17], rollout_threads=0)


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

    def test_a_batched_arm_requires_the_fidelity_loss_ack(self) -> None:
        # The demonstrated failing input for the guard that was previously on the
        # wrong knob: `leaf_batch > 1` is the arm's ONLY uncertified selection
        # regime and used to be accepted silently, while `rollout_threads` --
        # proven value-invariant -- required an acknowledgement.
        for leaf_batch in (2, 8, 64, 4096):
            with self.subTest(leaf_batch=leaf_batch), self.assertRaises(ValueError) as caught:
                self._config(leaf_batch=leaf_batch)
            message = str(caught.exception)
            self.assertIn("leaf_batch_fidelity_loss_ack=True", message)
            # The refusal must carry the MAGNITUDE, so a reader deciding whether
            # to set the ack sees what it costs rather than only that it is
            # uncertified.
            self.assertIn("8.2 pp", message)

    def test_the_batch_ack_admits_the_regime_rather_than_banning_it(self) -> None:
        # A fence that cannot be opened would make the batching seam untestable;
        # the requirement is an explicit choice, not a prohibition.
        config = self._config(leaf_batch=8, leaf_batch_fidelity_loss_ack=True)
        self.assertEqual(config.leaf_batch, 8)

    def test_the_batch_ack_is_inert_on_other_leaf_evals(self) -> None:
        from pokezero.engine_search import EngineMctsConfig

        EngineMctsConfig(leaf_eval="hp_fraction", leaf_batch=64)


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
