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


def _build_state(
    side_one_moves,
    side_two_moves,
    *,
    s1_speed=200,
    s2_hp=100,
    s1_status="none",
    s2_ability=None,
    s2_maxhp=100,
    s1_reserve=False,
    s2_volatiles=(),
    s2_volatile_durations=None,
):
    from pokezero.poke_engine_adapter import (
        BattleSpec,
        MoveSpec,
        PokemonSpec,
        SideSpec,
        build_poke_engine_state,
    )

    def mon(species, moves, *, hp=100, maxhp=100, speed=100, status="none", ability=None):
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
            status=status,
            ability=ability,
            moves=tuple(MoveSpec(id=m, pp=32) for m in moves),
        )

    side_one = [mon("rattata", side_one_moves, speed=s1_speed, status=s1_status)]
    if s1_reserve:
        side_one.append(mon("pikachu", ("splash",), speed=s1_speed - 1))
    spec = BattleSpec(
        side_one=SideSpec(pokemon=tuple(side_one)),
        side_two=SideSpec(
            pokemon=(
                mon(
                    "chansey",
                    side_two_moves,
                    hp=s2_hp,
                    maxhp=s2_maxhp,
                    ability=s2_ability,
                ),
            ),
            volatile_statuses=s2_volatiles,
            volatile_status_durations=s2_volatile_durations or {},
        ),
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

    def test_full_hp_absorb_keeps_public_hit_and_miss_histories(self) -> None:
        state = _build_state(
            ("hydropump",),
            ("splash",),
            s2_ability="waterabsorb",
            s2_hp=100,
            s2_maxhp=100,
        )
        report = json.loads(
            pokezero_search.branch_events(state, "hydropump", "splash", CTX, True, False)
        )
        by_pct = {round(branch["percentage"]): branch for branch in report["branches"]}
        self.assertIn("|-immune|p2a: Chansey|[from] ability: Water Absorb", by_pct[80]["events"])
        self.assertIn("|-miss|p1a: Rattata|p2a: Chansey", by_pct[20]["events"])

    def test_inaccurate_move_into_protect_keeps_public_hit_and_miss_histories(self) -> None:
        state = _build_state(
            ("hydropump",),
            ("protect",),
            s2_ability="waterabsorb",
            s2_hp=100,
            s2_maxhp=100,
        )
        report = json.loads(
            pokezero_search.branch_events(state, "hydropump", "protect", CTX, True, False)
        )
        by_pct = {round(branch["percentage"]): branch for branch in report["branches"]}
        self.assertIn("|-activate|p2a: Chansey|Protect", by_pct[80]["events"])
        self.assertIn("|-miss|p1a: Rattata|p2a: Chansey", by_pct[20]["events"])

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

    def test_rough_skin_contact_damage_is_not_a_self_cost(self) -> None:
        # Contact-ability punishment (Rough Skin) must carry its [from]
        # attribution: a bare attacker-side |-damage| would be read by the
        # fold as self_hp_cost (PR #727 review, LOW-1).
        state = _build_state(
            ("tackle",), ("splash",), s2_ability="roughskin", s2_hp=400, s2_maxhp=400
        )
        report = json.loads(
            pokezero_search.branch_events(state, "tackle", "splash", CTX, True, False)
        )
        contact_branches = 0
        for branch in report["branches"]:
            self.assertEqual(branch["lossy"], [], branch)
            attacker_damage = [
                line for line in branch["events"] if line.startswith("|-damage|p1a: Rattata|")
            ]
            if not attacker_damage:
                continue  # miss branch
            contact_branches += 1
            for line in attacker_damage:
                self.assertIn(
                    "|[from] ability: Rough Skin|[of] p2a: Chansey", line, line
                )
        self.assertGreater(contact_branches, 0)
        # And the fold reads it as opponent-inflicted, not a self cost.
        fold = pokezero_search.FoldState.initial("p1")
        fold.advance_in_place(LEAD_LINES)
        hit = next(
            b
            for b in report["branches"]
            if any(l.startswith("|-damage|p1a: Rattata|") for l in b["events"])
        )
        fold.advance_in_place(hit["events"])
        tackle = next(
            token
            for token in fold.products_payload()["transition_tokens"]
            if token["kind"] == "move" and token["action"] == "tackle"
        )
        self.assertEqual(tackle["self_hp_cost"], 0.0)

    def test_switch_prefixed_confusion_self_hit_is_canonical_and_not_lossy(self) -> None:
        # This is the retained certification shape: p1 switches, then p2's
        # confused active hits itself before it can use Splash. The mapper must
        # preserve the switch prefix while rendering the standard confusion
        # activation/damage pair rather than dropping into its generic
        # attacker-side-damage fallback.
        state = _build_state(
            ("splash",),
            ("splash",),
            s1_reserve=True,
            s2_volatiles=("confusion",),
            s2_volatile_durations={"confusion": 0},
        )
        context = json.dumps({"p1": ["Rattata", "Pikachu"], "p2": ["Chansey"], "turn": 1})
        report = json.loads(
            pokezero_search.branch_events(state, "pikachu", "splash", context, True, False)
        )
        self_hit = next(
            branch
            for branch in report["branches"]
            if any(line.endswith("|[from] confusion") for line in branch["events"])
        )
        self.assertEqual(self_hit["lossy"], [], self_hit)
        switch = next(
            index
            for index, line in enumerate(self_hit["events"])
            if line.startswith("|switch|p1a: Pikachu|")
        )
        activate = self_hit["events"].index("|-activate|p2a: Chansey|confusion")
        damage = next(
            index
            for index, line in enumerate(self_hit["events"])
            if line.startswith("|-damage|p2a: Chansey|") and line.endswith("|[from] confusion")
        )
        self.assertLess(switch, activate)
        self.assertLess(activate, damage)
        self.assertFalse(
            any(line.startswith("|move|p2a: Chansey|") for line in self_hit["events"]),
            self_hit,
        )

    def test_confusion_without_switch_still_uses_canonical_pair(self) -> None:
        state = _build_state(
            ("splash",),
            ("splash",),
            s2_volatiles=("confusion",),
            s2_volatile_durations={"confusion": 0},
        )
        report = json.loads(
            pokezero_search.branch_events(state, "splash", "splash", CTX, True, False)
        )
        self_hit = next(
            branch
            for branch in report["branches"]
            if any(line.endswith("|[from] confusion") for line in branch["events"])
        )
        self.assertEqual(self_hit["lossy"], [], self_hit)
        self.assertIn("|-activate|p2a: Chansey|confusion", self_hit["events"])

    def test_recoil_remains_recoil_not_confusion(self) -> None:
        state = _build_state(("doubleedge",), ("splash",), s2_hp=400, s2_maxhp=400)
        report = json.loads(
            pokezero_search.branch_events(state, "doubleedge", "splash", CTX, True, False)
        )
        recoil = next(
            branch
            for branch in report["branches"]
            if any("[from] Recoil" in line for line in branch["events"])
        )
        self.assertEqual(recoil["lossy"], [], recoil)
        self.assertFalse(any("confusion" in line for line in recoil["events"]), recoil)

    def test_ambiguous_sleep_talk_call_is_flagged_lossy(self) -> None:
        # An asleep Sleep Talker whose callable moves ALL produce an empty
        # delta (splash, and roar against a reserve-less side): the called
        # move cannot be identified, and the invariant is flag-lossy, never
        # silently drop (PR #727 review, LOW-2).
        state = _build_state(
            ("sleeptalk", "splash", "roar"), ("splash",), s1_status="sleep"
        )
        report = json.loads(
            pokezero_search.branch_events(state, "sleeptalk", "splash", CTX, True, False)
        )
        flagged = [
            b
            for b in report["branches"]
            if "sleeptalk_called_unidentified" in b["lossy"]
        ]
        self.assertTrue(flagged, report["branches"])


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
