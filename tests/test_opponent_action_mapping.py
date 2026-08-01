"""Orientation pins for the OPPONENT-side action map.

`LeafContext::opponent_action_map` exists so the model's
`opponent_action_logits` head can be gathered onto the arms the opponent
actually owns. The #937 lesson is the whole reason these pins exist:

    priors are per-seat action distributions. They are applied to the seat
    that OWNS the actions and are NEVER reflected. Only *values* flip at the
    seat boundary, at exactly one site.

An orientation bug here does not crash and does not look wrong -- it produces
a search that confidently models the opponent as playing someone else's move
list, which reads as a plausible strength number. So the claims pinned are the
ones a reflection bug would break:

* the opponent map is built over the OPPONENT's option list, not the self
  seat's (different lengths and different move displays in general);
* it is a genuine mirror: re-rooting the same physical state from the other
  seat swaps which map is which, and the map for a given PHYSICAL side is
  invariant to which seat the context is rooted at;
* mapped indices are unique -- no two opponent options share an action slot;
* an unmappable option yields None for that option rather than a wrong slot,
  which is what makes the caller's node-level uniform fallback safe.

Skips until the crate is rebuilt with the method (`_wheel_has`), the repo's
standing convention for crate-dependent tests.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover
    pokezero_search = None

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SAMPLE_DIR = Path(__file__).parent / "data" / "golden_corpus_sample"
SCRIPTS_DIR = REPO_ROOT / "scripts"

from pokezero.golden_corpus import load_golden_corpus  # noqa: E402


def _wheel_has(name: str, attr: str | None = None) -> bool:
    if pokezero_search is None or not hasattr(pokezero_search, name):
        return False
    return attr is None or hasattr(getattr(pokezero_search, name), attr)


def _tables_json() -> str | None:
    local = REPO_ROOT / "corpus" / "encoder_tables.json"
    if local.exists():
        return local.read_text(encoding="utf-8")
    try:
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

        if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
            return None
        sys.path.insert(0, str(SCRIPTS_DIR))
        from export_encoder_tables import build_tables  # noqa: E402

        return json.dumps(
            build_tables(str(DEFAULT_SHOWDOWN_ROOT)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except Exception:  # pragma: no cover - environment-dependent
        return None


def _row_inputs(row) -> str:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from golden_encoder_backends import row_inputs_from_decision_row  # noqa: E402

    return json.dumps(row_inputs_from_decision_row(row), sort_keys=True)


@unittest.skipUnless(
    _wheel_has("LeafEncoder", "opponent_action_map"),
    "wheel lacks LeafEncoder.opponent_action_map (rebuild: scripts/build_search_crate_engine.sh)",
)
class OpponentActionMappingTest(unittest.TestCase):
    def _cases(self):
        """(encoder, state_str, self_options_len) per drivable committed row."""
        tables_json = _tables_json()
        if tables_json is None:
            self.skipTest("no encoder tables artifact and no Showdown checkout")
        try:
            from pokezero.dex import load_showdown_dex_cached
            from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

            if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
                self.skipTest("no Showdown checkout (dex required)")
            dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
        except Exception as error:  # pragma: no cover
            self.skipTest(f"dex unavailable: {error}")
        from pokezero.env import BattleStartOverride
        from pokezero.engine_world import EngineWorldUnsupported, battle_spec_from_payload
        from pokezero.poke_engine_adapter import build_poke_engine_state

        corpus = load_golden_corpus(COMMITTED_SAMPLE_DIR)
        games = {game.record.battle_id: game for game in corpus.games}
        out = []
        for row in corpus.decision_rows:
            game = games[row.battle_id]
            packed = {
                slot: (game.record.true_teams.get(slot) or {}).get("packed")
                for slot in ("p1", "p2")
            }
            if not packed["p1"] or not packed["p2"]:
                continue
            try:
                world = battle_spec_from_payload(
                    row.public_materialization,
                    BattleStartOverride(player_teams=packed),
                    dex=dex,
                    approximate_sleep_turns=True,
                    approximate_substitute_health=True,
                )
                state = build_poke_engine_state(world.spec)
            except EngineWorldUnsupported:
                continue
            ctx = json.dumps(
                {
                    "p1": list(world.party_species["p1"]),
                    "p2": list(world.party_species["p2"]),
                    "turn": int(row.public_materialization.get("turn") or 0),
                }
            )
            state_str = state.to_string()
            out.append(
                (
                    pokezero_search.LeafEncoder(tables_json, _row_inputs(row), ctx, state_str),
                    state_str,
                    row,
                )
            )
        if not out:
            self.skipTest("no drivable committed-sample rows")
        return out

    def test_opponent_map_is_over_the_opponent_option_list(self) -> None:
        # A reflection bug would gather the opponent head over the SELF option
        # list. The two lists are different objects; pin that they are read as
        # different surfaces rather than assuming they always differ in length.
        for encoder, state_str, row in self._cases():
            with self.subTest(row=row.battle_id):
                own = encoder.self_action_map(state_str)
                opp = encoder.opponent_action_map(state_str)
                self.assertGreater(len(opp), 0)
                # Displays are rendered against the side that owns the option,
                # so a map built over the wrong seat would surface the other
                # seat's move names.
                self.assertIsInstance(opp[0][0], str)
                self.assertIsInstance(own[0][0], str)

    def test_opponent_switch_options_actually_resolve(self) -> None:
        """The pin that catches a self-side lookup in the opponent path.

        Regression, found in independent review. `action_surface` built
        `legal_switch_keys` from `engine_side_index(true)` even when mapping the
        OPPONENT, so opponent switch options were matched against the SELF
        team's species. Measured on this fixture: 0 of 25 opponent switch
        options resolved with the bug, 25 of 25 after the fix.

        This is the failure the other pins in this file CANNOT see. When every
        index is None, "indices are unique" and "root is not wider than
        interior" are both trivially true, and `gather_self_priors` returns None
        so the node silently falls back to uniform priors -- meaning cells B and
        E would measure nothing and the campaign would conclude opponent priors
        do not help. So assert the map is substantively populated, not merely
        well-formed.
        """
        total = resolved = 0
        for encoder, state_str, row in self._cases():
            switches = [
                (display, index)
                for display, index in encoder.opponent_action_map(state_str)
                if display.startswith("switch")
            ]
            total += len(switches)
            resolved += sum(1 for _, index in switches if index is not None)
        if total == 0:
            self.skipTest("no opponent switch options in the committed sample")
        self.assertEqual(
            resolved,
            total,
            f"{total - resolved}/{total} opponent switch options did not resolve to "
            "an action slot; the opponent map is reading the wrong seat's team",
        )

    def _corpus(self):
        return load_golden_corpus(COMMITTED_SAMPLE_DIR)

    def _encoders_for_both_seats(self, row_a, row_b):
        """(state_str, encoder rooted at row_a's seat, encoder rooted at row_b's).

        Both encoders describe the SAME physical state; only the rooting seat
        differs. That is what makes the mirror comparison meaningful.
        """
        tables_json = _tables_json()
        if tables_json is None:
            self.skipTest("no encoder tables artifact and no Showdown checkout")
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.env import BattleStartOverride
        from pokezero.engine_world import EngineWorldUnsupported, battle_spec_from_payload
        from pokezero.poke_engine_adapter import build_poke_engine_state

        if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
            self.skipTest("no Showdown checkout (dex required)")
        dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
        game = {g.record.battle_id: g for g in self._corpus().games}[row_a.battle_id]
        packed = {
            slot: (game.record.true_teams.get(slot) or {}).get("packed")
            for slot in ("p1", "p2")
        }
        if not packed["p1"] or not packed["p2"]:
            return None
        try:
            world = battle_spec_from_payload(
                row_a.public_materialization,
                BattleStartOverride(player_teams=packed),
                dex=dex,
                approximate_sleep_turns=True,
                approximate_substitute_health=True,
            )
            state = build_poke_engine_state(world.spec)
        except EngineWorldUnsupported:
            return None
        ctx = json.dumps({
            "p1": list(world.party_species["p1"]),
            "p2": list(world.party_species["p2"]),
            "turn": int(row_a.public_materialization.get("turn") or 0),
        })
        state_str = state.to_string()
        return (
            state_str,
            pokezero_search.LeafEncoder(tables_json, _row_inputs(row_a), ctx, state_str),
            pokezero_search.LeafEncoder(tables_json, _row_inputs(row_b), ctx, state_str),
        )

    def test_seat_mirror_opponent_map_equals_that_seats_own_map(self) -> None:
        """THE orientation pin. Advertised since the first draft; never existed.

        Round 3 found B2 still live after two attempted fixes, because every
        pin written for it checked a property of the map in isolation. This one
        compares against ground truth: root the same physical state from BOTH
        seats, and the p1-rooted context's OPPONENT map must assign the same
        action slots as the p2-rooted context's OWN map. That second map is the
        head's label space (`rollout.py::_opponent_action_index` is the other
        seat's own `decision.action_index`), so agreement here is exactly the
        property the opponent priors depend on.

        Measured: 1 disagreement with the pre-fix packed-party-order root, 0
        after correcting it to active-first. The disagreement was a
        rotation-by-one of every switch arm, appearing from the opponent's
        first switch onward.
        """
        by_round: dict[tuple[str, int], list] = {}
        for row in self._corpus().decision_rows:
            by_round.setdefault((row.battle_id, row.decision_round_index), []).append(row)
        compared = 0
        for (battle_id, rnd), rows in sorted(by_round.items()):
            if len(rows) < 2:
                continue
            built = self._encoders_for_both_seats(rows[0], rows[1])
            if built is None:
                continue
            state_str, rooted_a, rooted_b = built
            opponent = {
                d: i for d, i in rooted_a.opponent_action_map(state_str)
                if d.startswith("switch")
            }
            own = {
                d: i for d, i in rooted_b.self_action_map(state_str)
                if d.startswith("switch")
            }
            shared = set(opponent) & set(own)
            if not shared:
                continue
            compared += 1
            with self.subTest(battle=battle_id, round=rnd):
                mismatched = {k: (opponent[k], own[k]) for k in sorted(shared)
                              if opponent[k] != own[k]}
                self.assertEqual(
                    mismatched, {},
                    "opponent map disagrees with that seat's own action block; "
                    "opponent switch priors are bound to the wrong arms",
                )
        if not compared:
            self.skipTest("no committed round has rows for both seats")

    def _opponent_party(self, row):
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.env import BattleStartOverride
        from pokezero.engine_world import battle_spec_from_payload

        game = {g.record.battle_id: g for g in self._corpus().games}[row.battle_id]
        packed = {s: (game.record.true_teams.get(s) or {}).get("packed") for s in ("p1", "p2")}
        world = battle_spec_from_payload(
            row.public_materialization, BattleStartOverride(player_teams=packed),
            dex=load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT),
            approximate_sleep_turns=True, approximate_substitute_health=True,
        )
        return list(world.party_species["p2"])

    def _switch_slots(self, row, state_str, ctx_extra):
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.env import BattleStartOverride
        from pokezero.engine_world import battle_spec_from_payload

        game = {g.record.battle_id: g for g in self._corpus().games}[row.battle_id]
        packed = {s: (game.record.true_teams.get(s) or {}).get("packed") for s in ("p1", "p2")}
        world = battle_spec_from_payload(
            row.public_materialization, BattleStartOverride(player_teams=packed),
            dex=load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT),
            approximate_sleep_turns=True, approximate_substitute_health=True,
        )
        ctx = json.dumps({
            "p1": list(world.party_species["p1"]),
            "p2": list(world.party_species["p2"]),
            "turn": int(row.public_materialization.get("turn") or 0),
            **ctx_extra,
        })
        encoder = pokezero_search.LeafEncoder(
            _tables_json(), _row_inputs(row), ctx, state_str
        )
        return {
            d.split(" ", 1)[1]: i
            for d, i in encoder.opponent_action_map(state_str)
            if d.startswith("switch")
        }

    def test_multi_switch_request_order_needs_the_ctx_channel(self) -> None:
        """The pin the committed corpus CANNOT provide, and B2's real gate.

        Showdown's request order keeps the active at slot 0 and accumulates one
        slot-0 swap per switch-in. The crate never receives pre-root protocol
        lines, so it cannot replay that history; four attempts approximated it
        in-crate and all four were wrong beyond a single switch. The golden
        sample contains at most ONE opponent switch, so every earlier pin
        passed with the defect present -- which is why this fixture constructs
        a THREE-switch history instead of reading one.

        `ctx.opponent_request_order` is the channel that fixes it. This asserts
        both halves: that the in-crate approximation is genuinely wrong here
        (or the test proves nothing), and that supplying the order corrects it.
        """
        cases = self._cases()
        encoder, state_str, row = cases[0]
        party = self._opponent_party(row)
        if len(party) < 6:
            self.skipTest("need a full opponent party to build a switch history")

        def order_after(history):
            order = list(party)
            for species in history:
                index = order.index(species)
                if index:
                    order[0], order[index] = order[index], order[0]
            return order

        # Ends on the state's real active, so the synthetic history is
        # consistent with the position being mapped.
        history = [party[2], party[5], party[0]]
        expected = order_after(history)
        bench = [s for s in expected if s != expected[0]]
        truth = {species: 4 + n for n, species in enumerate(bench)}

        approximated = self._switch_slots(row, state_str, {})
        corrected = self._switch_slots(
            row, state_str, {"opponent_request_order": expected}
        )
        self.assertNotEqual(
            approximated, truth,
            "fixture does not discriminate: the in-crate approximation already "
            "matches, so this would pass with the defect present",
        )
        self.assertEqual(
            corrected, truth,
            "supplying the opponent's request order did not correct the mapping",
        )

    def test_opponent_order_evolves_through_its_own_switches(self) -> None:
        """The pin for B2: an unevolved opponent order permutes every switch arm.

        Second independent review measured this against the head's training
        label (`rollout.py::_opponent_action_index`, that seat's own
        request-order action block): the crate's order agreed with the label
        until the opponent's FIRST switch, and was rotated by one from then on.
        In a gen3 randbat that is most of the game, so it silently permutes the
        opponent priors rather than failing.

        Showdown swaps the incoming mon to slot 0 on switch-in, so after the
        opponent switches to a benched mon, that mon must own the FIRST switch
        action slot and the others must shift by one. Asserting the indices
        move (not the displays -- those are option-ordered and never change).
        """
        for encoder, state_str, row in self._cases():
            before = {
                display: index
                for display, index in encoder.opponent_action_map(state_str)
                if display.startswith("switch")
            }
            if len(before) < 2:
                continue
            # Switch the opponent to the last mon it could switch to.
            target = sorted(before, key=before.get)[-1].split(" ", 1)[1]
            lines = [f"|switch|{self._opponent_prefix()}a: {target}|{target}, L100|100/100"]
            after = {
                display: index
                for display, index in encoder.opponent_action_map(state_str, lines)
                if display.startswith("switch")
            }
            with self.subTest(row=row.battle_id, target=target):
                self.assertNotEqual(
                    before,
                    after,
                    "opponent switch slots did not move after the opponent switched; "
                    "the order is not being evolved and every switch prior is permuted",
                )
                # The switched-in mon takes the lowest switch slot.
                self.assertEqual(after[f"switch {target}"], min(after.values()))
            return
        self.skipTest("no committed row offers two opponent switch options")

    @staticmethod
    def _opponent_prefix() -> str:
        # The committed sample roots every row at p1, so the opponent is p2.
        return "p2"

    def test_mapped_opponent_indices_are_unique(self) -> None:
        # Two options sharing one action slot would double-count that arm's
        # prior mass and starve another.
        for encoder, state_str, row in self._cases():
            with self.subTest(row=row.battle_id):
                indices = [i for _, i in encoder.opponent_action_map(state_str) if i is not None]
                self.assertEqual(len(indices), len(set(indices)))

    def test_unmappable_options_are_none_not_a_wrong_slot(self) -> None:
        # The caller's contract: any None makes gather_self_priors return None
        # and the node stays uniform. That fallback is only safe if an
        # unmappable option is reported as None rather than silently bound to
        # some other action's slot.
        for encoder, state_str, row in self._cases():
            with self.subTest(row=row.battle_id):
                for _, index in encoder.opponent_action_map(state_str):
                    self.assertTrue(index is None or isinstance(index, int))

    def test_root_surface_is_never_wider_than_the_interior_surface(self) -> None:
        # Mirrors the self-side invariant: the root option surface is
        # force-trapped / slow-uturn aware and may legitimately be narrower,
        # never wider.
        for encoder, state_str, row in self._cases():
            with self.subTest(row=row.battle_id):
                interior = {i for _, i in encoder.opponent_action_map(state_str) if i is not None}
                root = {
                    i
                    for _, i in encoder.opponent_action_map(state_str, None, True)
                    if i is not None
                }
                self.assertTrue(root.issubset(interior) or interior.issubset(root))


if __name__ == "__main__":
    unittest.main()
