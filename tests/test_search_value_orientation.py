"""Seat orientation of the leaf value the engine MCTS backs up.

Two conventions meet inside ``search_batched_multi_encoded`` and they are NOT
the same convention:

* the leaf observation is encoded from the SEARCHING SEAT's perspective (its own
  SELF / OPPONENT pokemon token blocks), and the checkpoint's value target is
  ``+1`` iff that seat won the game
  (``pokezero.dataset._terminal_value_for_player``) — so the model value is
  SELF-relative;
* ``rust/pokezero-search/src/tree.rs`` is SIDE-ONE-absolute: terminal branches
  are priced from poke-engine ``battle_is_over`` (``+1`` = side one won), backup
  adds the chance-node expectation to both seats' stats unchanged, and the only
  seat flip lives in ``MoveStats::puct`` (``1 - mean`` for side two).

``engine_world.py`` pins ``slot_sides = {"p1": "side_one", "p2": "side_two"}``,
so the searching seat is side two exactly when PokeZero plays p2 — half of every
mirrored-pair evaluation. The crate must reflect the model value ONCE at that
seat boundary. These tests pin that it does, and that the reflection is
seat-constant rather than depth-dependent.

Method: the leaf model is a TorchScript stub whose value orientation is fixed BY
CONSTRUCTION — ``v = tanh(20 * (mean SELF-block HP fraction - mean
OPPONENT-block HP fraction))`` read out of the observation's own token blocks.
Ground truth for "which engine side leads" is read off the constructed world
spec, never off search output.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover
    numpy = None

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover
    pokezero_search = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SAMPLE_DIR = Path(__file__).parent / "data" / "golden_corpus_sample"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TRANSITION_TOKEN_BUDGET = 32
DEPTHS = (1, 2, 3, 4)
SIMS = 96
# Only roots with a real HP asymmetry carry a signed expectation.
MIN_HP_LEAD = 0.05

from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3  # noqa: E402
from pokezero.showdown import (  # noqa: E402
    NUMERIC_HP_FRACTION,
    NUMERIC_PRESENT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    SELF_POKEMON_TOKEN_COUNT,
    SELF_POKEMON_TOKEN_OFFSET,
)

_crate_ready = bool(
    pokezero_search is not None
    and getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False)
    and hasattr(getattr(pokezero_search, "LeafEncoder", None), "encode_leaf")
)


def _tables_json() -> str | None:
    local = REPO_ROOT / "corpus" / "encoder_tables.json"
    if local.exists():
        payload = json.loads(local.read_text(encoding="utf-8"))
    else:
        try:
            from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

            if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
                return None
            sys.path.insert(0, str(SCRIPTS_DIR))
            from export_encoder_tables import build_tables  # noqa: PLC0415

            payload = build_tables(
                str(DEFAULT_SHOWDOWN_ROOT),
                observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
            )
        except Exception:  # pragma: no cover - environment-dependent
            return None
    if payload.get("layout", {}).get("schema_version") != OBSERVATION_SCHEMA_VERSION_V3:
        return None
    payload["layout"]["default_feature_masks"][
        "transition_token_budget"
    ] = TRANSITION_TOKEN_BUDGET
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _build_oriented_stub(path: Path) -> Path:
    """A value head whose SELF-relative orientation is fixed by construction."""

    class OrientedHpValue(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hp_col = int(NUMERIC_HP_FRACTION)
            self.present_col = int(NUMERIC_PRESENT)
            self.self_lo = int(SELF_POKEMON_TOKEN_OFFSET)
            self.self_hi = int(SELF_POKEMON_TOKEN_OFFSET + SELF_POKEMON_TOKEN_COUNT)
            self.opp_lo = int(OPPONENT_POKEMON_TOKEN_OFFSET)
            self.opp_hi = int(
                OPPONENT_POKEMON_TOKEN_OFFSET + OPPONENT_POKEMON_TOKEN_COUNT
            )
            self.action_count = 9

        def forward(
            self,
            categorical_ids: torch.Tensor,
            numeric_features: torch.Tensor,
            token_type_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            history_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num = numeric_features[:, 0]
            hp = num[:, :, self.hp_col]
            present = num[:, :, self.present_col]
            own = hp[:, self.self_lo : self.self_hi] * present[:, self.self_lo : self.self_hi]
            opp = hp[:, self.opp_lo : self.opp_hi] * present[:, self.opp_lo : self.opp_hi]
            own_n = present[:, self.self_lo : self.self_hi].sum(-1).clamp(min=1.0)
            opp_n = present[:, self.opp_lo : self.opp_hi].sum(-1).clamp(min=1.0)
            value = torch.tanh(20.0 * (own.sum(-1) / own_n - opp.sum(-1) / opp_n))
            zeros = torch.zeros(
                numeric_features.shape[0],
                self.action_count,
                dtype=numeric_features.dtype,
                device=numeric_features.device,
            )
            return zeros, value, zeros

    torch.jit.script(OrientedHpValue().eval()).save(str(path))
    return path


def _side_hp(side) -> float:
    total = 0.0
    for mon in side.pokemon:
        if mon.maxhp > 0:
            total += max(mon.hp, 0) / mon.maxhp
    return total


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(torch is None, "requires torch")
@unittest.skipUnless(_crate_ready, "pokezero_search lacks the model feature")
class SearchValueOrientationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables_json = _tables_json()
        if cls.tables_json is None:
            raise unittest.SkipTest("no encoder tables artifact and no Showdown checkout")
        try:
            from pokezero.dex import load_showdown_dex_cached
            from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

            if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
                raise unittest.SkipTest("no Showdown checkout (dex required)")
            dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
        except unittest.SkipTest:
            raise
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"dex unavailable: {error}")

        from pokezero.engine_world import (
            EngineWorldUnsupported,
            battle_spec_from_payload,
        )
        from pokezero.env import BattleStartOverride
        from pokezero.golden_corpus import (
            GOLDEN_CORPUS_SCHEMA_VERSION,
            load_golden_corpus,
        )
        from pokezero.golden_corpus_fold import iter_fold_records
        from pokezero.poke_engine_adapter import build_poke_engine_state

        sys.path.insert(0, str(SCRIPTS_DIR))
        from golden_encoder_backends import row_inputs_from_decision_row  # noqa: PLC0415

        corpus = load_golden_corpus(COMMITTED_SAMPLE_DIR)
        folds = {
            int(record["array_row_index"]): record["fold_state"]
            for record in iter_fold_records(
                COMMITTED_SAMPLE_DIR, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
            )
        }
        games = {game.record.battle_id: game for game in corpus.games}
        positions: list[dict] = []
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
            row_inputs = row_inputs_from_decision_row(row)
            row_inputs["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION_V3
            positions.append(
                {
                    "row_index": index,
                    "player_id": row.player_id,
                    "self_is_side_one": row.player_id == "p1",
                    "s1_hp": _side_hp(world.spec.side_one),
                    "s2_hp": _side_hp(world.spec.side_two),
                    "state_str": state.to_string(),
                    "row_inputs": json.dumps(row_inputs, sort_keys=True),
                    "ctx": json.dumps(
                        {
                            "p1": list(world.party_species["p1"]),
                            "p2": list(world.party_species["p2"]),
                            "turn": int(row.public_materialization.get("turn") or 0),
                        }
                    ),
                    "fold_state": folds[index],
                }
            )
            if len(positions) >= 8:
                break
        if not positions:
            raise unittest.SkipTest("no committed-sample row could be driven")
        cls.positions = positions

        layout = json.loads(cls.tables_json)["layout"]
        cls.layout = layout
        cls.tmpdir = tempfile.TemporaryDirectory()
        artifact = _build_oriented_stub(Path(cls.tmpdir.name) / "oriented_hp_value.pt")
        cls.native = pokezero_search.NativeLeafModel(
            str(artifact),
            device="cpu",
            window=1,
            tokens=int(layout["token_count"]),
            categorical_features=int(layout["categorical_feature_count"]),
            numeric_features=int(layout["numeric_feature_count"]),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmpdir"):
            cls.tmpdir.cleanup()

    def _root_v01(self, position: dict) -> float:
        """The model's own (SELF-relative) value on the root observation."""
        encoder = pokezero_search.LeafEncoder(
            self.tables_json,
            position["row_inputs"],
            position["ctx"],
            position["state_str"],
        )
        fold = pokezero_search.FoldState.from_payload(position["fold_state"])
        row = encoder.encode_leaf(
            position["state_str"], fold, json.loads(position["ctx"])["turn"]
        )
        values, _logits, _priors, _count = self.native.eval_obs_flat(
            1,
            numpy.frombuffer(row["categorical_ids"], dtype="<i4").astype("int64").tolist(),
            numpy.frombuffer(row["numeric_features"], dtype="<f8").astype("float32").tolist(),
            numpy.frombuffer(row["token_type_ids"], dtype="<i2").astype("int64").tolist(),
            [bool(flag) for flag in row["attention_mask"]],
            [True] * int(self.layout.get("window", 1) or 1),
        )
        return 0.5 * (values[0] + 1.0)

    def _search(self, position: dict, depth: int) -> dict:
        fold = pokezero_search.FoldState.from_payload(position["fold_state"])
        return json.loads(
            self.native.search_batched_multi_encoded(
                position["state_str"],
                SIMS,
                1,
                self.tables_json,
                position["row_inputs"],
                position["ctx"],
                fold,
                depth,
                1.4,
                7,
                True,
                False,  # priors off: this gate is about VALUES only
            )
        )

    def _signed_positions(self) -> list[dict]:
        return [
            position
            for position in self.positions
            if abs(position["s1_hp"] - position["s2_hp"]) > MIN_HP_LEAD
        ]

    def test_model_value_is_self_relative(self) -> None:
        """The leaf value tracks the OBSERVING seat, not side one.

        This is the fact that forces the reflection: on a p2-seated root the raw
        model value points the opposite way from the tree's own terminal-branch
        convention.
        """
        signed = self._signed_positions()
        if not signed:
            self.skipTest("no committed-sample root has an HP asymmetry")
        for position in signed:
            with self.subTest(row=position["row_index"], seat=position["player_id"]):
                v01 = self._root_v01(position)
                self_leads = (
                    position["s1_hp"] > position["s2_hp"]
                    if position["self_is_side_one"]
                    else position["s2_hp"] > position["s1_hp"]
                )
                self.assertEqual(
                    v01 > 0.5,
                    self_leads,
                    f"stub value {v01} does not track the OBSERVING seat's HP lead",
                )

    def test_backed_up_root_value_is_side_one_oriented_on_both_seats(self) -> None:
        """The tree's root value is side-one-absolute regardless of the seat.

        With a value model that is monotone in the observing seat's HP lead, a
        correctly reflected tree must report ``root_value > 0.5`` exactly when
        SIDE ONE leads -- on p1-seated and p2-seated roots alike. Feeding the
        self-relative value through unreflected inverts every p2 root.
        """
        signed = self._signed_positions()
        if not signed:
            self.skipTest("no committed-sample root has an HP asymmetry")
        seats_seen = set()
        for position in signed:
            side_one_leads = position["s1_hp"] > position["s2_hp"]
            for depth in DEPTHS:
                with self.subTest(
                    row=position["row_index"], seat=position["player_id"], depth=depth
                ):
                    report = self._search(position, depth)
                    root_value = float(report["root_value"])
                    self.assertEqual(
                        root_value > 0.5,
                        side_one_leads,
                        f"root_value {root_value} is not side-one oriented "
                        f"(side one leads: {side_one_leads})",
                    )
            seats_seen.add(position["player_id"])
        self.assertIn(
            "p2",
            seats_seen,
            "the gate is vacuous without a p2-seated root (side two = the reflected seat)",
        )

    def test_orientation_is_depth_constant_not_depth_parity_dependent(self) -> None:
        """No even/odd split in the backed-up root value.

        A per-level sign or perspective error -- the one shape the depth-1
        parity result in docs/mcts_degradation_findings.md cannot exclude on its
        own -- separates even from odd depths. Any orientation defect in this
        path is seat-constant instead: it either holds at every depth or at
        none.
        """
        signed = self._signed_positions()
        if not signed:
            self.skipTest("no committed-sample root has an HP asymmetry")
        for position in signed:
            with self.subTest(row=position["row_index"], seat=position["player_id"]):
                values = {
                    depth: float(self._search(position, depth)["root_value"])
                    for depth in DEPTHS
                }
                sides = {depth: value > 0.5 for depth, value in values.items()}
                self.assertEqual(
                    len(set(sides.values())),
                    1,
                    f"root value crosses 0.5 with depth: {values}",
                )
                even = [values[d] for d in DEPTHS if d % 2 == 0]
                odd = [values[d] for d in DEPTHS if d % 2 == 1]
                self.assertLess(
                    abs(sum(even) / len(even) - sum(odd) / len(odd)),
                    0.25,
                    f"even- and odd-depth root values separate: {values}",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
