"""`leaf.rs` must DERIVE the move-trap flag from the branch, not pass the root's.

The exact defect shape `tests/test_leaf_self_recharge_derivation.py` pins for MUSTRECHARGE, and
made live by the same kind of fix.

`TRAPPED` is absent from `leaf.rs::VOLATILE_MAP` -- the parser records the move trap in its own
`meanlook_trap` tracker rather than the TRACKED_VOLATILES bag -- so `tracked_volatiles` cannot
carry it, and `encoder.rs` reads `{prefix}_meanlook_trap` straight out of `observation_metadata`
for NUMERIC_MEANLOOK_TRAP on every v3 AND v4 layout. `leaf.rs` mutates the ROOT's metadata in
place, so with no write of its own the ROOT's bit was stamped onto every leaf in the subtree.

That was harmless only because it was UNREACHABLE: a move-trapped root was refused as
`self_request_state_unsupported`, so no trapped position ever entered a search. Routing the
parser's move trap into the world payload makes those roots searchable, and the branch's TRAPPED
then moves independently of the root -- the engine drops it when the trapper leaves the field and
carries it through a Baton Pass (`gen3/state.rs:825` `TRAPPED => baton_passing`).

And the clinching argument is the one C141 made for MUSTRECHARGE: the leaf's ACTION SURFACE is
already live, because switch options come from the engine's `get_all_options`, which honours
`Side::trapped` (`gen3/state.rs:493`). A frozen bag beside a live action surface is a
self-contradiction inside one observation, not merely a stale value.

Like its neighbour, this drives the real `LeafEncoder` and reads the metadata `leaf.rs` actually
emits, because the freeze is invisible to every Python test: reverting the write leaves the suite
green, and the only other things that catch it (`leaf_root_parity` / `leaf_vs_reality`) need the
gitignored corpus and do not run in CI.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

try:
    import pokezero_search
except Exception:  # pragma: no cover
    pokezero_search = None

from test_leaf_encoder import (  # noqa: E402
    _row_inputs,
    _sample_rows_with_folds,
    _tables_json,
)


@unittest.skipUnless(pokezero_search is not None, "pokezero_search wheel not installed")
class LeafDerivesTheMoveTrapFlagTest(unittest.TestCase):
    """One property, both directions, driven through the real encoder."""

    def _pair(self):
        """A root world and the same world with OUR OWN active move-trapped.

        Both are built by `battle_spec_from_payload`, so neither is a hand-edited state string --
        the only difference is the `meanlookTrap` payload key this PR adds, which is exactly the
        input the leaf ignored.
        """
        tables_json = _tables_json()
        if tables_json is None:
            self.skipTest("no encoder tables artifact and no Showdown checkout")
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_world import (
            EngineWorldUnsupported,
            battle_spec_from_payload,
        )
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.poke_engine_adapter import build_poke_engine_state

        if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
            self.skipTest("no Showdown checkout (dex required)")
        dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)

        corpus, _folds = _sample_rows_with_folds()
        games = {game.record.battle_id: game for game in corpus.games}
        for row in corpus.decision_rows:
            if row.player_id != "p1":
                continue
            game = games[row.battle_id]
            packed = {
                slot: (game.record.true_teams.get(slot) or {}).get("packed")
                for slot in ("p1", "p2")
            }
            if not packed["p1"] or not packed["p2"]:
                continue
            kwargs = dict(
                dex=dex, approximate_sleep_turns=True, approximate_substitute_health=True
            )
            trapped_payload = json.loads(json.dumps(row.public_materialization))
            trapped_payload["sides"]["p1"]["meanlookTrap"] = True
            try:
                free = battle_spec_from_payload(
                    row.public_materialization,
                    BattleStartOverride(player_teams=packed),
                    **kwargs,
                )
                trapped = battle_spec_from_payload(
                    trapped_payload,
                    BattleStartOverride(player_teams=packed),
                    **kwargs,
                )
            except EngineWorldUnsupported:
                # A world this corpus row cannot express. Legitimately skippable -- but ONLY this
                # exception: a bare `except Exception` would swallow a genuine world-construction
                # regression and report it as "no suitable row".
                continue
            free_state = build_poke_engine_state(free.spec).to_string()
            trapped_state = build_poke_engine_state(trapped.spec).to_string()
            if free_state == trapped_state:
                # The trap did not reach the state, so this row proves nothing either way.
                continue
            context = json.dumps(
                {
                    "p1": list(free.party_species["p1"]),
                    "p2": list(free.party_species["p2"]),
                    "turn": int(row.public_materialization.get("turn") or 0),
                }
            )
            turn = int(row.observation_metadata.get("turn_number") or 0)
            return tables_json, _row_inputs(row), context, free_state, trapped_state, turn
        self.fail(
            "no committed-sample p1 row could build BOTH the free and move-trapped world. "
            "The sample is COMMITTED (tests/data/golden_corpus_sample/), so on any machine with "
            "a Showdown checkout and the patched wheel this is a regression -- most likely the "
            "`meanlookTrap` payload key no longer reaching the engine state -- not an "
            "environment difference. `skipTest` here would be this repo's own denominator rule "
            "unapplied to its own pin: a run that measured nothing is not a pass."
        )

    def _self_flag(self, encoder, leaf_state, turn):
        md = json.loads(encoder.leaf_inputs_json(leaf_state, turn))["observation_metadata"]
        return md.get("self_meanlook_trap")

    def test_a_branch_that_gains_the_trap_reports_it(self) -> None:
        """Root FREE, leaf TRAPPED. Under the freeze this returned the root's `false`."""
        tables, row_inputs, ctx, free_state, trapped_state, turn = self._pair()
        encoder = pokezero_search.LeafEncoder(tables, row_inputs, ctx, free_state)
        self.assertIs(
            self._self_flag(encoder, trapped_state, turn),
            True,
            "leaf did not derive our own TRAPPED from the branch -- the root's bit is being "
            "stamped onto the subtree",
        )

    def test_a_branch_that_loses_the_trap_reports_that_too(self) -> None:
        """The other direction: the trapper left the field, so the trap is over.

        A frozen write returns the root's `true` here and is wrong -- but so would a write
        hardcoded to `false`, so this direction alone proves nothing. It is the pair that binds.
        """
        tables, row_inputs, ctx, free_state, trapped_state, turn = self._pair()
        encoder = pokezero_search.LeafEncoder(tables, row_inputs, ctx, trapped_state)
        self.assertIs(
            self._self_flag(encoder, free_state, turn),
            False,
            "leaf kept a trap the branch no longer has -- the flag is not tracking the branch",
        )

    def test_the_flag_is_not_simply_constant(self) -> None:
        """Non-vacuity for both: the two states really do disagree.

        Without this, a derivation returning the same value for every input could satisfy one of
        the assertions above depending on which constant it chose.
        """
        tables, row_inputs, ctx, free_state, trapped_state, turn = self._pair()
        encoder = pokezero_search.LeafEncoder(tables, row_inputs, ctx, free_state)
        self.assertNotEqual(
            self._self_flag(encoder, free_state, turn),
            self._self_flag(encoder, trapped_state, turn),
            "the same encoder returns one value for both states -- nothing is being derived",
        )

    def test_the_flag_follows_the_trapped_SEAT(self) -> None:
        """`self_` and `opponent_` must not be wired to the same side.

        C141's injection proof caught exactly this shape on the recharge pair (a self-side write
        wired to the opponent's volatile), so the two-key loop is pinned on both keys here.
        """
        tables, row_inputs, ctx, free_state, trapped_state, turn = self._pair()
        encoder = pokezero_search.LeafEncoder(tables, row_inputs, ctx, free_state)
        md = json.loads(encoder.leaf_inputs_json(trapped_state, turn))["observation_metadata"]
        self.assertIn("opponent_meanlook_trap", md)
        self.assertIs(md["opponent_meanlook_trap"], False, "p1 is trapped, not p2")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
