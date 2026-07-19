"""Mapping assertion for self-side model priors (engine-swap integration).

The prior wiring maps the model's action-block priors onto engine
``MoveChoice``s through ``LeafContext::self_action_map`` — a map DERIVED from
``action_surface``, the leaf encoder's own candidate/legal-mask builder, so
the correspondence under test is the one the observations themselves carry.
Machine-checkable claims, per drivable committed-sample decision row:

- interior-surface map (``get_all_options`` — the option order every interior
  decision node uses, and the surface the depth-0 root-parity gate proved
  reproduces the recorded request mask): every engine option maps, indices
  are unique, and the mapped set EQUALS the legal bits of the RECORDED
  golden ``legal_action_mask`` (post-#730 F2: switch indices on golden
  order);
- root-surface map (``root_get_all_options`` — force-trapped / slow-uturn
  aware, the ROOT decision node's option order): every option maps and the
  mapped set is CONTAINED in the recorded legal bits (the root surface may
  legitimately be narrower, never wider).

The full-corpus sweep of the same assertions is
``scripts/prior_mapping_assert.py`` (run over golden-v2 + scenarios in the
PR's verification evidence).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover
    numpy = None

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


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipUnless(
    _wheel_has("LeafEncoder", "self_action_map"), "wheel lacks LeafEncoder.self_action_map"
)
class PriorActionMappingCommittedSampleTest(unittest.TestCase):
    """Option→action-index map vs the recorded request masks (depth 0)."""

    def test_committed_sample_maps_match_recorded_masks(self) -> None:
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
        from pokezero.engine_world import (
            EngineWorldUnsupported,
            battle_spec_from_payload,
        )
        from pokezero.poke_engine_adapter import build_poke_engine_state

        corpus = load_golden_corpus(COMMITTED_SAMPLE_DIR)
        games = {game.record.battle_id: game for game in corpus.games}
        driven = 0
        for index, row in enumerate(corpus.decision_rows):
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
            encoder = pokezero_search.LeafEncoder(
                tables_json, _row_inputs(row), ctx, state_str
            )
            recorded_legal = {
                action_index
                for action_index, legal in enumerate(
                    numpy.asarray(row.arrays.legal_action_mask).flatten().tolist()
                )
                if legal
            }
            label = f"sample row {index} ({row.battle_id}#r{row.decision_round_index})"

            # Interior option surface: exact set equality with the recorded mask.
            interior = encoder.self_action_map(state_str)
            interior_indices = [action_index for _, action_index in interior]
            self.assertNotIn(
                None, interior_indices, f"{label}: unmapped interior option in {interior}"
            )
            self.assertEqual(
                len(set(interior_indices)),
                len(interior_indices),
                f"{label}: interior map not injective: {interior}",
            )
            self.assertEqual(
                set(interior_indices),
                recorded_legal,
                f"{label}: interior map {interior} != recorded mask {sorted(recorded_legal)}",
            )

            # Root option surface (force-trapped / slow-uturn aware): may be
            # narrower than the request mask, never wider.
            root = encoder.self_action_map(state_str, root=True)
            root_indices = [action_index for _, action_index in root]
            self.assertNotIn(None, root_indices, f"{label}: unmapped root option in {root}")
            self.assertEqual(
                len(set(root_indices)), len(root_indices), f"{label}: root map not injective"
            )
            self.assertTrue(
                set(root_indices) <= recorded_legal,
                f"{label}: root map {root} outside recorded mask {sorted(recorded_legal)}",
            )
            driven += 1
        self.assertGreater(driven, 0, "no committed-sample row could be driven")


if __name__ == "__main__":
    unittest.main()
