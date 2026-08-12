"""Crate-free existence pins for the two opponent-prior mutation fixtures.

WHY THIS FILE EXISTS, and it is not belt-and-braces. Two mutants in
`rust/pokezero-search/src/model.rs` are killed by fixtures in
`tests/test_model_priors_search.py`:

  * M9  -- `branch: opponent_prefix() -> self_prefix()`, which freezes the
    opponent's request order at its root value;
  * M9' -- the CHAINING arm, `Some(key) => rec.opponent_order.clone()` replaced
    by the root order, which is the same defect one ply further down and the one
    that bites at `--depth 4`, the depth `foulplay_paired_eval.py` defaults to.

Both are recorded KILLED in `priors.rs`'s mutation census. But NOTHING IN CI RUNS
EITHER FIXTURE: they need the `model` feature (a libtorch and a built wheel), no
workflow builds it, and `engine-fidelity-gates.yml`'s `cargo test` step omits
`--features model`, so even the Rust-side `priors::` tests are not compiled
there. A future commit could delete or defang both fixtures and every workflow
would stay green, while the census still read "killed".

These pins need no crate. They read the test module as TEXT and as an AST and
assert the two fixtures still exist, are not skipped, and still carry the
assertion that does the killing. That reddens on deletion or defanging without a
libtorch, which is the cheapest available guard until torch is in CI.

They deliberately do NOT try to verify the fixtures still PASS -- only a real run
can do that. A pin that cannot fail for the right reason is worse than none, so
the docstrings here say exactly what is and is not covered.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

_MODULE = pathlib.Path(__file__).with_name("test_model_priors_search.py")

M9_FIXTURE = "test_a_branch_that_switches_the_opponent_evolves_its_request_order"
M9_PRIME_FIXTURE = "test_a_deeper_seam_chains_the_parent_branch_order_not_the_root"


def _functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(_MODULE.read_text())
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


class MutationFixturesStillExistTests(unittest.TestCase):
    """Deletion or renaming of either fixture reddens here, with no crate."""

    def test_both_mutation_fixtures_are_present(self) -> None:
        names = _functions()
        for fixture in (M9_FIXTURE, M9_PRIME_FIXTURE):
            self.assertIn(
                fixture,
                names,
                f"{fixture} is gone from {_MODULE.name}; priors.rs's census "
                "still records the mutant it kills as KILLED, so the census "
                "would now be claiming coverage that does not exist",
            )

    def test_neither_fixture_is_skipped(self) -> None:
        """A `@skip` would leave the census true-looking and the mutant alive."""
        for fixture, node in _functions().items():
            if fixture not in (M9_FIXTURE, M9_PRIME_FIXTURE):
                continue
            decorators = [ast.unparse(d) for d in node.decorator_list]
            self.assertFalse(
                [d for d in decorators if "skip" in d.lower()],
                f"{fixture} carries a skip decorator: {decorators}",
            )


class MutationFixturesStillAssertTests(unittest.TestCase):
    """The killing assertion, not merely the function, must survive.

    Each fixture has ONE assertion that does the discriminating work. A
    refactor that keeps the name and drops that line would leave a green test
    that kills nothing -- the exact failure the surrounding programme calls
    "the instrument that cannot move".
    """

    def test_the_m9_fixture_still_compares_against_the_root_order(self) -> None:
        body = ast.unparse(_functions()[M9_FIXTURE])
        self.assertIn(
            "!= root_order",
            body,
            "the M9 fixture no longer compares a ply-2 order against the root; "
            "without that comparison a frozen order passes",
        )
        self.assertIn("assertGreater", body)

    def test_the_m9_prime_fixture_still_counts_swaps(self) -> None:
        body = ast.unparse(_functions()[M9_PRIME_FIXTURE])
        # The DEDUCED value, not merely the helper's name: an earlier version of
        # this pin looked for "swaps_from_root" anywhere in the body and stayed
        # green when the killing line was replaced by `0 for order in orders`,
        # because the helper is also referenced by a nearby assertion. Pin the
        # reduction that actually produces the number being compared.
        # Regex, not a literal: `ast.unparse` parenthesises a bare generator
        # argument as `max((... for ... in ...))`, so a literal match here
        # failed on an UNMODIFIED tree -- a pin that reddens on correct code is
        # as useless as one that stays green on broken code.
        self.assertRegex(
            body,
            r"deepest\s*=\s*max\(+\s*swaps_from_root\(order\)\s+for\s+order\s+in\s+orders",
            "the M9' fixture no longer reduces the recorded orders to a maximum "
            "swap distance; swap distance is the only thing separating a chained "
            "order from a frozen one",
        )
        # And that the maximum is compared against 2, which is the threshold a
        # frozen chain cannot reach.
        self.assertRegex(
            body,
            r"assertGreaterEqual\(\s*deepest\s*,\s*2\b",
            "the >= 2 swap assertion is what fails under the frozen-chain "
            "mutant; a different threshold or subject does not kill it",
        )
        self.assertIn(
            "max_depth=3",
            body,
            "at max_depth=2 the chaining arm cannot execute and the fixture is "
            "vacuous -- the depth is part of the kill",
        )


if __name__ == "__main__":
    unittest.main()
