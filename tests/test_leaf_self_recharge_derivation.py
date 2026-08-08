"""`leaf.rs` must DERIVE the self-side MUSTRECHARGE flag from the branch, not pass the root's.

The freeze on that write was lifted once `_recharging_slots` went symmetric, because leaving it
was worse than before: `engine_search` began BUILDING self-recharge worlds that previously failed
closed, so the stale root flag became reachable in production search at depth > 0.

Review then found the lift **pinned by nothing** — reverting it at the binary level left 146/146
Python tests green. The only things that caught it were `leaf_root_parity` / `leaf_vs_reality`,
which need the gitignored corpus and do not run in CI. That is the "guard nothing pins" pattern,
in the same change that added a CI step to close it elsewhere.

This drives the real `LeafEncoder` and reads the metadata `leaf.rs` actually emits. It does not
need the corpus; it needs a Showdown checkout for the dex, like its neighbours in
`test_leaf_encoder.py`.
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
class LeafDerivesTheSelfRechargeFlagTest(unittest.TestCase):
    """One property, both directions, driven through the real encoder."""

    def _pair(self):
        """A root world and the same world with OUR OWN slot locked to recharge.

        Both are built by `battle_spec_from_payload`, so neither is a hand-edited state string —
        the only difference is `recharging_slots`, which is exactly the input the freeze used to
        ignore at the leaf.
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
            try:
                free = battle_spec_from_payload(
                    row.public_materialization,
                    BattleStartOverride(player_teams=packed),
                    **kwargs,
                )
                locked = battle_spec_from_payload(
                    row.public_materialization,
                    BattleStartOverride(player_teams=packed),
                    recharging_slots=("p1",),
                    **kwargs,
                )
            except EngineWorldUnsupported:
                # A world this corpus row cannot express. Legitimately skippable -- but ONLY
                # this exception: a bare `except Exception` here would swallow a genuine
                # regression in world construction and report it as "no suitable row".
                continue
            free_state = build_poke_engine_state(free.spec).to_string()
            locked_state = build_poke_engine_state(locked.spec).to_string()
            if free_state == locked_state:
                # The lock did not reach the state, so this row proves nothing either way.
                continue
            context = json.dumps(
                {
                    "p1": list(free.party_species["p1"]),
                    "p2": list(free.party_species["p2"]),
                    "turn": int(row.public_materialization.get("turn") or 0),
                }
            )
            turn = int(row.observation_metadata.get("turn_number") or 0)
            return tables_json, _row_inputs(row), context, free_state, locked_state, turn
        self.fail(
            "no committed-sample p1 row could build BOTH the free and locked world. "
            "The sample is COMMITTED (tests/data/golden_corpus_sample/), so on any machine with "
            "a Showdown checkout this is a regression -- most likely `recharging_slots` no "
            "longer reaching the engine state -- not an environment difference. "
            "`skipTest` here would be this repo's own denominator rule unapplied to its own "
            "pin: a run that measured nothing is not a pass."
        )

    def _self_flag(self, encoder, leaf_state, turn):
        md = json.loads(encoder.leaf_inputs_json(leaf_state, turn))["observation_metadata"]
        return md.get("self_must_recharge")

    def test_a_branch_that_gains_the_lock_reports_it(self) -> None:
        """Root FREE, leaf LOCKED. Under the freeze this returned the root's `false`."""
        tables, row_inputs, ctx, free_state, locked_state, turn = self._pair()
        encoder = pokezero_search.LeafEncoder(tables, row_inputs, ctx, free_state)
        self.assertIs(
            self._self_flag(encoder, locked_state, turn),
            True,
            "leaf did not derive our own MUSTRECHARGE from the branch — the root freeze is back",
        )

    def test_a_branch_that_loses_the_lock_reports_that_too(self) -> None:
        """The other direction, and the one the freeze got RIGHT by accident.

        Root LOCKED, leaf FREE: a branch that consumed the recharge turn. A frozen write returns
        the root's `true` here and is wrong; but so would a write hardcoded to `false`, so this
        direction alone proves nothing — it is the pair that binds.
        """
        tables, row_inputs, ctx, free_state, locked_state, turn = self._pair()
        encoder = pokezero_search.LeafEncoder(tables, row_inputs, ctx, locked_state)
        self.assertIs(
            self._self_flag(encoder, free_state, turn),
            False,
            "leaf kept a consumed MUSTRECHARGE — the flag is not tracking the branch",
        )

    def test_the_flag_is_not_simply_constant(self) -> None:
        """Non-vacuity for both: the two states really do disagree.

        Without this, a derivation that returned the same value for every input could satisfy one
        of the assertions above depending on which constant it chose.
        """
        tables, row_inputs, ctx, free_state, locked_state, turn = self._pair()
        encoder = pokezero_search.LeafEncoder(tables, row_inputs, ctx, free_state)
        self.assertNotEqual(
            self._self_flag(encoder, free_state, turn),
            self._self_flag(encoder, locked_state, turn),
            "the same encoder returns one value for both states — nothing is being derived",
        )

    def test_the_opponent_side_still_works(self) -> None:
        """The lift changed one write into a two-side loop; the side that was already live must
        not have been broken by the refactor."""
        tables, row_inputs, ctx, free_state, locked_state, turn = self._pair()
        encoder = pokezero_search.LeafEncoder(tables, row_inputs, ctx, free_state)
        md = json.loads(encoder.leaf_inputs_json(locked_state, turn))["observation_metadata"]
        self.assertIn("opponent_must_recharge", md)
        self.assertIs(md["opponent_must_recharge"], False, "p1 is locked, not p2")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
