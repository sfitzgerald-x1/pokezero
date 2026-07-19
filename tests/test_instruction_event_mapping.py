"""Gates for the instruction→event mapping (rust/pokezero-search/src/events.rs).

Covers the search-leaf seam of the engine-swap plan (docs/
test_time_search_plan_v3.md, search-tree contract item 2): a chance branch's
engine instruction list + context renders as Showdown protocol lines, and a
CLONE of the root fold state advances over those lines to produce per-outcome
fold products — REAL history tokens at leaves, no freezing, no stale history.

Mirrors tests/test_multiply_chance_search.py conventions: every test skips
unless the built native module imports.
"""

from __future__ import annotations

import json
import unittest

try:  # pragma: no cover - exercised only when the native crate is built
    import pokezero_search
except ImportError:  # pragma: no cover
    pokezero_search = None  # type: ignore[assignment]


def _build_state(side_one_moves, side_two_moves, *, s1_speed=200, s2_hp=100):
    from pokezero.poke_engine_adapter import (
        BattleSpec,
        MoveSpec,
        PokemonSpec,
        SideSpec,
        build_poke_engine_state,
    )

    def mon(species, moves, *, hp=100, maxhp=100, speed=100):
        return PokemonSpec(
            id=species,
            level=100,
            types=("normal",),
            hp=hp,
            maxhp=maxhp,
            attack=100,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=speed,
            status="none",
            moves=tuple(MoveSpec(id=m, pp=32) for m in moves),
        )

    spec = BattleSpec(
        side_one=SideSpec(pokemon=(mon("rattata", side_one_moves, speed=s1_speed),)),
        side_two=SideSpec(pokemon=(mon("chansey", side_two_moves, hp=s2_hp),)),
    )
    return build_poke_engine_state(spec).to_string()


CTX = json.dumps({"p1": ["Rattata"], "p2": ["Chansey"], "turn": 1})

# The lead lines that put the fold at the turn-1 decision boundary (the root
# prefix every branch shares).
LEAD_LINES = [
    "|switch|p1a: Rattata|Rattata, L100|100/100",
    "|switch|p2a: Chansey|Chansey, L100|100/100",
    "|turn|1",
]


@unittest.skipIf(pokezero_search is None, "pokezero_search native module not built")
class BranchEventsTest(unittest.TestCase):
    """The mapper: instruction list + context -> protocol lines."""

    def setUp(self) -> None:
        try:
            self.analytic = _build_state(("toxic", "seismictoss"), ("splash",))
        except Exception as exc:  # engine binding missing/broken
            self.skipTest(f"poke_engine fixture unavailable: {exc}")

    def branches(self, s1: str, s2: str):
        report = json.loads(
            pokezero_search.branch_events(self.analytic, s1, s2, CTX, True, False)
        )
        return report

    def test_enumerates_and_renders_all_outcomes(self) -> None:
        report = self.branches("toxic", "splash")
        self.assertTrue(report["end_of_turn"])
        branches = report["branches"]
        # gen3 toxic on a splash-locked target: 85% hit / 15% miss.
        self.assertEqual(len(branches), 2)
        by_pct = {round(b["percentage"]): b for b in branches}
        hit, miss = by_pct[85], by_pct[15]
        for branch in branches:
            self.assertEqual(branch["lossy"], [], branch)
            self.assertTrue(branch["turn_completed"], branch)
            self.assertIn("|upkeep", branch["events"])
            self.assertIn("|turn|2", branch["events"])
        hit_text = "\n".join(hit["events"])
        self.assertIn("|move|p1a: Rattata|toxic|p2a: Chansey", hit_text)
        self.assertIn("|-status|p2a: Chansey|tox", hit_text)
        self.assertIn("[from] psn", hit_text)  # end-of-turn residual, tagged
        miss_text = "\n".join(miss["events"])
        self.assertIn("|[miss]", miss_text)
        self.assertIn("|-miss|p1a: Rattata|p2a: Chansey", miss_text)

    def test_fold_input_contract_ascii_integers(self) -> None:
        # fold.rs input contract: hp fields are plain ASCII integers.
        for branch in self.branches("seismictoss", "splash")["branches"]:
            for line in branch["events"]:
                if line.startswith("|-damage|") or line.startswith("|-heal|"):
                    hp_field = line.split("|")[3].split(" ")[0]
                    if hp_field == "0":
                        continue
                    numerator, _, denominator = hp_field.partition("/")
                    self.assertTrue(
                        numerator.isascii() and numerator.isdigit(), line
                    )
                    self.assertTrue(
                        denominator.isascii() and denominator.isdigit(), line
                    )

    def test_ko_branch_renders_faint(self) -> None:
        # Seismic toss (level 100) KOs the 100-HP Chansey: terminal branch.
        branches = self.branches("seismictoss", "splash")["branches"]
        ko = "\n".join(branches[0]["events"])
        self.assertIn("|-damage|p2a: Chansey|0 fnt", ko)
        self.assertIn("|faint|p2a: Chansey", ko)


@unittest.skipIf(pokezero_search is None, "pokezero_search native module not built")
class LeafFoldAdvanceTest(unittest.TestCase):
    """End-to-end leaf demo: root fold state -> branch -> synthesized events
    -> Rust fold advance -> per-outcome products (the exact flow the in-crate
    encoder integration will run at the batch-row write)."""

    def setUp(self) -> None:
        try:
            self.state = _build_state(("toxic", "seismictoss"), ("splash",))
        except Exception as exc:
            self.skipTest(f"poke_engine fixture unavailable: {exc}")
        self.root_fold = pokezero_search.FoldState.initial("p1")
        self.root_fold.advance_in_place(LEAD_LINES)
        self.root_products = self.root_fold.products_payload()

    def test_per_outcome_fold_products(self) -> None:
        report = json.loads(
            pokezero_search.branch_events(self.state, "toxic", "splash", CTX, True, False)
        )
        products_by_branch = []
        for branch in report["branches"]:
            leaf_fold = self.root_fold.clone_state()
            leaf_fold.advance_in_place(branch["events"])
            products_by_branch.append(leaf_fold.products_payload())

        root_total = self.root_products["transition_token_total"]
        for products in products_by_branch:
            # The leaf's history extends the shared root prefix with the
            # simulated turn (owner contract: no freezing, no stale history).
            self.assertGreater(products["transition_token_total"], root_total)
            self.assertEqual(
                products["transition_tokens"][: root_total],
                self.root_products["transition_tokens"],
            )
            last_turn_tokens = [
                token
                for token in products["transition_tokens"][root_total:]
                if token["kind"] == "move" and token["actor_slot"] == "p1"
            ]
            self.assertEqual(len(last_turn_tokens), 1)
            self.assertEqual(last_turn_tokens[0]["action"], "toxic")

        # Per-outcome histories DIFFER: the miss branch's token shows the
        # miss, the hit branch's shows the inflicted status (the exact
        # internal consistency the search-tree contract demands).
        def p1_toxic_token(products):
            return next(
                token
                for token in products["transition_tokens"][root_total:]
                if token["kind"] == "move" and token["actor_slot"] == "p1"
            )

        outcomes = {
            (p1_toxic_token(p)["miss"], p1_toxic_token(p)["side_effect"])
            for p in products_by_branch
        }
        self.assertEqual(
            outcomes, {(True, "none"), (False, "status-inflicted")}
        )
        # The root fold state itself is untouched (branches advance CLONES).
        self.assertEqual(
            self.root_fold.products_payload(), self.root_products
        )

    def test_terminal_branch_products(self) -> None:
        report = json.loads(
            pokezero_search.branch_events(
                self.state, "seismictoss", "splash", CTX, True, False
            )
        )
        (branch,) = report["branches"]
        leaf_fold = self.root_fold.clone_state()
        leaf_fold.advance_in_place(branch["events"])
        products = leaf_fold.products_payload()
        toss = next(
            token
            for token in products["transition_tokens"]
            if token["kind"] == "move" and token["action"] == "seismictoss"
        )
        self.assertTrue(toss["ko"])
        self.assertAlmostEqual(toss["damage_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
