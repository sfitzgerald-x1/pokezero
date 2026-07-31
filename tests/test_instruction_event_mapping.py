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
    s1_volatile_statuses=(),
    s1_types=("normal",),
    s1_ability=None,
    s1_boosts=None,
    s2_hp=100,
    s2_speed=100,
    s2_attack=100,
    s2_status="none",
    s2_sleep_turns=0,
    s1_status="none",
    s2_ability=None,
    s2_item=None,
    s2_maxhp=100,
    s2_volatile_statuses=(),
    s1_side_conditions=None,
    s2_side_conditions=None,
    s2_boosts=None,
):
    from pokezero.poke_engine_adapter import (
        BattleSpec,
        MoveSpec,
        PokemonSpec,
        SideSpec,
        build_poke_engine_state,
    )

    def mon(
        species,
        moves,
        *,
        hp=100,
        maxhp=100,
        speed=100,
        attack=100,
        status="none",
        ability=None,
        item=None,
        types=("normal",),
        sleep_turns=0,
    ):
        return PokemonSpec(
            id=species,
            level=100,
            types=types,
            hp=hp,
            maxhp=maxhp,
            attack=attack,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=speed,
            status=status,
            ability=ability,
            item=item,
            sleep_turns=sleep_turns,
            moves=tuple(MoveSpec(id=m, pp=32) for m in moves),
        )

    spec = BattleSpec(
        side_one=SideSpec(
            pokemon=(
                mon(
                    "rattata",
                    side_one_moves,
                    speed=s1_speed,
                    status=s1_status,
                    types=s1_types,
                    ability=s1_ability,
                ),
            ),
            volatile_statuses=s1_volatile_statuses,
            side_conditions=s1_side_conditions or {},
            boosts=s1_boosts or {},
        ),
        side_two=SideSpec(
            pokemon=(
                mon(
                    "chansey",
                    side_two_moves,
                    hp=s2_hp,
                    maxhp=s2_maxhp,
                    speed=s2_speed,
                    attack=s2_attack,
                    ability=s2_ability,
                    item=s2_item,
                    status=s2_status,
                    sleep_turns=s2_sleep_turns,
                ),
            ),
            volatile_statuses=s2_volatile_statuses,
            side_conditions=s2_side_conditions or {},
            boosts=s2_boosts or {},
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

    def test_ambiguous_sleep_talk_call_is_rejected_before_fold(self) -> None:
        # An asleep Sleep Talker whose callable moves ALL produce an empty
        # delta (splash, and roar against a reserve-less side): the called
        # move cannot be identified, and the invariant is flag-lossy, never
        # silently drop (PR #727 review, LOW-2).
        state = _build_state(
            ("sleeptalk", "splash", "roar"),
            ("splash",),
            s1_status="sleep",
            s1_volatile_statuses=("confusion",),
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
        for branch in flagged:
            self.assertTrue(branch["attribution_unsafe"], branch)
            self.assertIn(
                "sleeptalk_called_unidentified", branch["attribution_unsafe_reasons"], branch
            )
            self.assertIn("|cant|p1a: Rattata|slp", branch["events"])
            self.assertIn("|-activate|p1a: Rattata|confusion", branch["events"])
            self.assertIn("|move|p1a: Rattata|sleeptalk|p1a: Rattata", branch["events"])
            self.assertFalse(
                any("[from] residual" in line for line in branch["events"]),
                branch,
            )

    def test_nonempty_ambiguous_sleep_talk_tail_keeps_post_state_out_of_fold(self) -> None:
        # Tackle and Scratch share the same non-empty Gen 3 damage tail here,
        # so the delta cannot prove which called move owned the public effects.
        state = _build_state(
            ("sleeptalk", "tackle", "scratch"),
            ("splash",),
            s1_status="sleep",
            s1_speed=500,
        )
        report = json.loads(
            pokezero_search.branch_events(state, "sleeptalk", "splash", CTX, True, True)
        )
        unsafe = [
            branch
            for branch in report["branches"]
            if "sleeptalk_called_unidentified" in branch["attribution_unsafe_reasons"]
        ]
        self.assertTrue(unsafe, report["branches"])
        for branch in unsafe:
            self.assertTrue(branch["attribution_unsafe"], branch)
            self.assertLess(branch["post"]["p2"]["active_hp"], 100, branch)
            self.assertEqual(branch["post"]["p1"]["active_status"], "sleep", branch)
            self.assertIn("|cant|p1a: Rattata|slp", branch["events"])
            self.assertIn("|move|p1a: Rattata|sleeptalk|p1a: Rattata", branch["events"])
            self.assertFalse(
                any(line.startswith("|-damage|p2a: Chansey|") for line in branch["events"]),
                branch,
            )

    def test_confusion_self_hit_cancels_substitute_and_keeps_leftovers(self) -> None:
        # The native before-move confusion branch uses the
        # mon's own Attack, so these values yield its exact -38 self-hit. The
        # selected Substitute must never be rendered in that outcome.
        state = _build_state(
            ("splash",),
            ("substitute",),
            s1_speed=500,
            s2_hp=200,
            s2_maxhp=256,
            s2_speed=1,
            s2_attack=108,
            s2_item="leftovers",
            s2_volatile_statuses=("confusion",),
        )
        report = json.loads(
            pokezero_search.branch_events(state, "splash", "substitute", CTX, True, False)
        )
        self_hit = next(
            branch
            for branch in report["branches"]
            if "|-activate|p2a: Chansey|confusion" in branch["events"]
            and "|-damage|p2a: Chansey|162/256" in branch["events"]
        )
        self.assertIn(
            "|-heal|p2a: Chansey|178/256|[from] item: Leftovers",
            self_hit["events"],
        )
        self.assertNotIn(
            "|move|p2a: Chansey|substitute|p2a: Chansey",
            self_hit["events"],
        )
        # The activation plus untagged damage is the fold's exact public
        # confusion contract. It must mark the prior actor's move window so
        # semantic move damage stays separate and V3 can expose the marker.
        fold = pokezero_search.FoldState.initial("p1")
        fold.advance_in_place(LEAD_LINES)
        fold.advance_in_place(self_hit["events"])
        splash = next(
            token
            for token in fold.products_payload()["transition_tokens"]
            if token["kind"] == "move" and token["action"] == "splash"
        )
        self.assertTrue(splash["confusion_selfhit"])

    def test_confusion_survives_attract_block_without_a_move_action(self) -> None:
        state = _build_state(
            ("splash",),
            ("substitute",),
            s1_speed=500,
            s2_speed=1,
            s2_volatile_statuses=("confusion", "attract"),
        )
        branches = json.loads(
            pokezero_search.branch_events(state, "splash", "substitute", CTX, True, False)
        )["branches"]
        blocked = next(
            branch
            for branch in branches
            if "|cant|p2a: Chansey|Attract" in branch["events"]
        )
        self.assertIn("|-activate|p2a: Chansey|confusion", blocked["events"])
        self.assertIn("attract_immobilization_source_unknown", blocked["lossy"], blocked)
        self.assertFalse(blocked["attribution_unsafe"], blocked)
        self.assertFalse(
            any(line.startswith("|move|p2a: Chansey|substitute") for line in blocked["events"]),
            blocked,
        )

        fold = pokezero_search.FoldState.initial("p1")
        fold.advance_in_place(LEAD_LINES)
        fold.advance_in_place(blocked["events"])
        tokens = fold.products_payload()["transition_tokens"]
        self.assertTrue(
            any(token["kind"] == "cant" and token["action"] == "attract" for token in tokens),
            tokens,
        )
        self.assertFalse(
            any(
                token["kind"] == "move" and token["action"] == "substitute"
                for token in tokens
            ),
            tokens,
        )

    def test_attract_empty_tail_ambiguities_fail_closed(self) -> None:
        # An empty post-confusion tail is not unique evidence of Attract. Each
        # case can also be a real attempted move with no observable change.
        cases = {
            "protect": ("protect", "tackle", _build_state(
                ("protect",),
                ("tackle",),
                s1_speed=500,
                s2_speed=1,
                s2_volatile_statuses=("attract",),
            )),
            "immunity": ("splash", "tackle", _build_state(
                ("splash",),
                ("tackle",),
                s1_speed=500,
                s1_types=("ghost",),
                s2_speed=1,
                s2_volatile_statuses=("attract",),
            )),
            "miss": ("splash", "hydropump", _build_state(
                ("splash",),
                ("hydropump",),
                s1_speed=500,
                s2_speed=1,
                s2_volatile_statuses=("attract",),
            )),
            "status": ("splash", "toxic", _build_state(
                ("splash",),
                ("toxic",),
                s1_speed=500,
                s1_status="poison",
                s2_speed=1,
                s2_volatile_statuses=("attract",),
            )),
            "capped_boost": ("splash", "swordsdance", _build_state(
                ("splash",),
                ("swordsdance",),
                s1_speed=500,
                s2_speed=1,
                s2_boosts={"attack": 6},
                s2_volatile_statuses=("attract",),
            )),
            "opponent_capped_boost": ("splash", "charm", _build_state(
                ("splash",),
                ("charm",),
                s1_speed=500,
                s1_boosts={"attack": -6},
                s2_speed=1,
                s2_volatile_statuses=("attract",),
            )),
            "opponent_boost_immunity": ("splash", "charm", _build_state(
                ("splash",),
                ("charm",),
                s1_speed=500,
                s1_ability="clearbody",
                s2_speed=1,
                s2_volatile_statuses=("attract",),
            )),
            "side_condition": ("splash", "spikes", _build_state(
                ("splash",),
                ("spikes",),
                s1_speed=500,
                s2_speed=1,
                s1_side_conditions={"spikes": 3},
                s2_volatile_statuses=("attract",),
            )),
            "intrinsic_noop": ("splash", "splash", _build_state(
                ("splash",),
                ("splash",),
                s1_speed=500,
                s2_speed=1,
                s2_volatile_statuses=("attract",),
            )),
        }
        for name, (s1_move, s2_move, state) in cases.items():
            with self.subTest(name=name):
                report = json.loads(
                    pokezero_search.branch_events(
                        state, s1_move, s2_move, CTX, True, False
                    )
                )
                unsafe = [
                    branch
                    for branch in report["branches"]
                    if "attract_empty_tail_ambiguous" in branch["attribution_unsafe_reasons"]
                ]
                self.assertTrue(unsafe, report["branches"])
                for branch in unsafe:
                    self.assertTrue(branch["attribution_unsafe"], branch)
                    self.assertFalse(
                        any("|cant|p2a: Chansey|Attract" in line for line in branch["events"]),
                        branch,
                    )

    def test_confusion_crash_recoil_and_explosion_do_not_fake_self_hit(self) -> None:
        crash_state = _build_state(
            ("splash",),
            ("highjumpkick",),
            s1_speed=500,
            s2_speed=1,
            s2_volatile_statuses=("confusion",),
        )
        crash = next(
            branch
            for branch in json.loads(
                pokezero_search.branch_events(
                    crash_state, "splash", "highjumpkick", CTX, True, False
                )
            )["branches"]
            if any("[from] highjumpkick" in line for line in branch["events"])
        )
        self.assertIn("|move|p2a: Chansey|highjumpkick|p1a: Rattata|[miss]", crash["events"])
        self.assertIn("|-activate|p2a: Chansey|confusion", crash["events"])
        self.assertEqual(crash["lossy"], [], crash)

        recoil_state = _build_state(
            ("splash",),
            ("doubleedge",),
            s1_speed=500,
            s2_speed=1,
            s2_volatile_statuses=("confusion",),
        )
        recoil = next(
            branch
            for branch in json.loads(
                pokezero_search.branch_events(
                    recoil_state, "splash", "doubleedge", CTX, True, False
                )
            )["branches"]
            if any("[from] Recoil" in line for line in branch["events"])
        )
        self.assertIn("|move|p2a: Chansey|doubleedge|p1a: Rattata", recoil["events"])
        self.assertIn("|-activate|p2a: Chansey|confusion", recoil["events"])

        protected_explosion = _build_state(
            ("splash",),
            ("explosion",),
            s1_speed=1,
            s1_volatile_statuses=("protect",),
            s2_speed=500,
            s2_volatile_statuses=("confusion",),
        )
        explosion = next(
            branch
            for branch in json.loads(
                pokezero_search.branch_events(
                    protected_explosion, "splash", "explosion", CTX, True, False
                )
            )["branches"]
            if "|faint|p2a: Chansey" in branch["events"]
            and "|move|p2a: Chansey|explosion|p1a: Rattata" in branch["events"]
        )
        self.assertIn("|-activate|p1a: Rattata|Protect", explosion["events"])
        self.assertIn("|-activate|p2a: Chansey|confusion", explosion["events"])

        immune_explosion = _build_state(
            ("splash",),
            ("explosion",),
            s1_speed=1,
            s1_types=("ghost",),
            s2_speed=500,
            s2_volatile_statuses=("confusion",),
        )
        immune = next(
            branch
            for branch in json.loads(
                pokezero_search.branch_events(
                    immune_explosion, "splash", "explosion", CTX, True, False
                )
            )["branches"]
            if "|faint|p2a: Chansey" in branch["events"]
            and "|move|p2a: Chansey|explosion|p1a: Rattata" in branch["events"]
        )
        self.assertIn("|-immune|p1a: Rattata", immune["events"])
        self.assertIn("|-activate|p2a: Chansey|confusion", immune["events"])

    def test_pre_move_bookkeeping_does_not_hide_confusion_self_hit(self) -> None:
        cases = [
            (
                "substitute",
                _build_state(
                    ("splash",),
                    ("substitute",),
                    s1_speed=500,
                    s2_speed=1,
                    s2_volatile_statuses=("destinybond", "confusion"),
                ),
            ),
            (
                "bellydrum",
                _build_state(
                    ("splash",),
                    ("bellydrum",),
                    s1_speed=500,
                    s2_speed=1,
                    s2_volatile_statuses=("destinybond", "confusion"),
                ),
            ),
            (
                "curse",
                _build_state(
                    ("splash",),
                    ("curse",),
                    s1_speed=500,
                    s2_speed=1,
                    s2_volatile_statuses=("destinybond", "confusion"),
                ),
            ),
            (
                "tackle",
                _build_state(
                    ("splash",),
                    ("tackle", "splash"),
                    s1_speed=500,
                    s2_speed=1,
                    s2_item="choiceband",
                    s2_volatile_statuses=("confusion",),
                ),
            ),
            (
                "outrage",
                _build_state(
                    ("splash",),
                    ("outrage", "splash"),
                    s1_speed=500,
                    s2_speed=1,
                    s2_volatile_statuses=("confusion",),
                ),
            ),
            (
                "futuresight",
                _build_state(
                    ("splash",),
                    ("futuresight",),
                    s1_speed=500,
                    s2_speed=1,
                    s2_volatile_statuses=("confusion",),
                ),
            ),
        ]
        for selected, state in cases:
            with self.subTest(selected=selected):
                report = json.loads(
                    pokezero_search.branch_events(
                        state, "splash", selected, CTX, True, False
                    )
                )
                self_hit = next(
                    branch
                    for branch in report["branches"]
                    if "|-activate|p2a: Chansey|confusion" in branch["events"]
                    and any(
                        line.startswith("|-damage|p2a: Chansey|")
                        for line in branch["events"]
                    )
                )
                self.assertFalse(
                    any(
                        line.startswith(f"|move|p2a: Chansey|{selected}")
                        for line in self_hit["events"]
                    ),
                    self_hit,
                )
                fold = pokezero_search.FoldState.initial("p1")
                fold.advance_in_place(LEAD_LINES)
                fold.advance_in_place(self_hit["events"])
                splash = next(
                    token
                    for token in fold.products_payload()["transition_tokens"]
                    if token["kind"] == "move" and token["action"] == "splash"
                )
                self.assertEqual(splash["damage_fraction"], 0.0, splash)
                self.assertEqual(splash["confusion_selfhit_fraction"], 0.35, splash)
                self.assertEqual(splash["self_hp_cost"], 0.0, splash)
                self.assertFalse(splash["ko"], splash)
                self.assertTrue(splash["confusion_selfhit"], splash)

    def test_native_fold_keeps_historical_hit_after_confusion_selfhit(self) -> None:
        # The self-hit lands in the prior move window, but it must not erase
        # the historical fact that Drill Peck already damaged its defender.
        # Only the last-damage KO guard is cleared.
        lines = LEAD_LINES + [
            "|move|p2a: Chansey|Drill Peck|p1a: Rattata",
            "|-damage|p1a: Rattata|83/100",
            "|-activate|p1a: Rattata|confusion",
            "|-damage|p1a: Rattata|73/100",
        ]
        fold = pokezero_search.FoldState.initial("p1")
        fold.advance_in_place(lines)
        current = fold.to_payload()["current_window"]
        self.assertTrue(current["defender_hit_by_move"], current)
        self.assertFalse(current["defender_last_damage_by_move"], current)

    def test_lethal_self_hit_is_not_previous_move_damage_or_ko(self) -> None:
        state = _build_state(
            ("splash",),
            ("substitute",),
            s1_speed=500,
            s2_speed=1,
            s2_hp=20,
            s2_maxhp=100,
            s2_volatile_statuses=("confusion",),
        )
        report = json.loads(
            pokezero_search.branch_events(
                state, "splash", "substitute", CTX, True, False
            )
        )
        self_hit = next(
            branch
            for branch in report["branches"]
            if "|-activate|p2a: Chansey|confusion" in branch["events"]
            and "|faint|p2a: Chansey" in branch["events"]
        )
        fold = pokezero_search.FoldState.initial("p1")
        fold.advance_in_place(LEAD_LINES)
        fold.advance_in_place(self_hit["events"])
        splash = next(
            token
            for token in fold.products_payload()["transition_tokens"]
            if token["kind"] == "move" and token["action"] == "splash"
        )
        # This fold starts from the public lead's 100/100 condition, so the
        # separate self-hit fraction is the full observed fraction.
        self.assertEqual(splash["damage_fraction"], 0.0, splash)
        self.assertEqual(splash["confusion_selfhit_fraction"], 1.0, splash)
        self.assertEqual(splash["self_hp_cost"], 0.0, splash)
        self.assertFalse(splash["ko"], splash)
        self.assertTrue(splash["confusion_selfhit"], splash)

    def test_memento_behind_protect_has_one_activation(self) -> None:
        state = _build_state(
            ("protect",),
            ("memento",),
            s1_speed=1,
            s2_speed=500,
        )
        report = json.loads(
            pokezero_search.branch_events(
                state, "protect", "memento", CTX, True, False
            )
        )
        (branch,) = report["branches"]
        self.assertEqual(
            branch["events"],
            [
                "|",
                "|move|p1a: Rattata|protect|p1a: Rattata",
                "|move|p2a: Chansey|memento|p1a: Rattata",
                "|-activate|p1a: Rattata|Protect",
                "|",
                "|upkeep",
                "|turn|2",
            ],
        )

    def test_confusion_expiry_wake_and_own_tempo_lifecycle(self) -> None:
        expiry_state = _build_state(
            ("splash",),
            ("splash",),
            s1_speed=500,
            s2_speed=1,
            s2_volatile_statuses=("confusion",),
        )
        expiry = next(
            branch
            for branch in json.loads(
                pokezero_search.branch_events(expiry_state, "splash", "splash", CTX, True, False)
            )["branches"]
            if "confusion_expiry_timing_unobservable" in branch["attribution_unsafe_reasons"]
            and "|move|p2a: Chansey|splash||[still]" in branch["events"]
        )
        self.assertIn("|move|p2a: Chansey|splash||[still]", expiry["events"])
        self.assertNotIn("|-end|p2a: Chansey|confusion", expiry["events"])
        self.assertTrue(expiry["attribution_unsafe"], expiry)

        wake_state = _build_state(
            ("splash",),
            ("substitute",),
            s1_speed=500,
            s2_speed=1,
            s2_status="sleep",
            s2_sleep_turns=3,
            s2_volatile_statuses=("confusion",),
        )
        wake_hit = next(
            branch
            for branch in json.loads(
                pokezero_search.branch_events(wake_state, "splash", "substitute", CTX, True, False)
            )["branches"]
            if "|-activate|p2a: Chansey|confusion" in branch["events"]
        )
        self.assertNotIn("|cant|p2a: Chansey|slp", wake_hit["events"])

        own_tempo_state = _build_state(
            ("splash",),
            ("substitute",),
            s1_speed=500,
            s2_speed=1,
            s2_ability="owntempo",
            s2_volatile_statuses=("confusion",),
        )
        own_tempo = json.loads(
            pokezero_search.branch_events(
                own_tempo_state, "splash", "substitute", CTX, True, False
            )
        )["branches"]
        self.assertEqual(len(own_tempo), 1, own_tempo)
        self.assertIn("|move|p2a: Chansey|substitute|p2a: Chansey", own_tempo[0]["events"])
        self.assertNotIn("|-activate|p2a: Chansey|confusion", own_tempo[0]["events"])

    def test_collapsed_confusion_crash_branch_is_rejected_without_naked_damage(self) -> None:
        state = _build_state(
            ("splash",),
            ("highjumpkick",),
            s1_speed=500,
            s2_speed=1,
            s2_attack=143,
            s2_volatile_statuses=("confusion",),
        )
        collision = next(
            branch
            for branch in json.loads(
                pokezero_search.branch_events(
                    state, "splash", "highjumpkick", CTX, True, False
                )
            )["branches"]
            if "confusion_selfhit_ambiguous_executed_self_damage"
            in branch["attribution_unsafe_reasons"]
        )
        self.assertIn(
            "confusion_selfhit_ambiguous_executed_self_damage", collision["lossy"], collision
        )
        self.assertNotIn("|-activate|p2a: Chansey|confusion", collision["events"])
        self.assertTrue(collision["attribution_unsafe"], collision)
        self.assertFalse(
            any("|move|p2a: Chansey|highjumpkick" in line for line in collision["events"]),
            collision,
        )
        self.assertFalse(
            any(line.startswith("|-damage|") for line in collision["events"]), collision
        )


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
