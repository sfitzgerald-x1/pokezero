"""Priors-in-the-loop gates for the encoded model search (engine swap).

`search_batched_multi_encoded` with `model_priors` (the default) prices the
root observation once for root priors and maps each priced branch's policy
output onto its child decision node's options. These tests pin the surface:

- batch=1 determinism holds with priors on (two identical runs, identical
  stats — priors are a deterministic function of the observations);
- the report carries the prior telemetry (`model_priors`, `root_priors`
  summing to 1 over the acting seat's arms, `prior_branches`,
  `prior_fallbacks`);
- `model_priors=False` restores uniform priors (kill switch for A/B);
- visit conservation holds in both modes.

Values-vs-exploration invariance is pinned Rust-side
(tree.rs `priors_reweight_exploration_not_values`); the mapping itself is
gated by tests/test_prior_action_mapping.py + scripts/prior_mapping_assert.py.
Random-weights artifact at the real v2.2 shape — never a strength claim.
"""

from __future__ import annotations

import importlib.util
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

from pokezero.golden_corpus import (  # noqa: E402
    GOLDEN_CORPUS_SCHEMA_VERSION,
    load_golden_corpus,
)
from pokezero.golden_corpus_fold import iter_fold_records  # noqa: E402

_crate_ready = bool(
    pokezero_search is not None
    and getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False)
    and hasattr(getattr(pokezero_search, "LeafEncoder", None), "self_action_map")
)


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


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_model", REPO_ROOT / "scripts" / "export_model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(torch is None, "requires torch")
@unittest.skipUnless(_crate_ready, "pokezero_search lacks the model feature or prior surfaces")
class ModelPriorsEncodedSearchTest(unittest.TestCase):
    """Encoded search with self-side model priors on a committed-sample world."""

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
        from pokezero.env import BattleStartOverride
        from pokezero.engine_world import (
            EngineWorldUnsupported,
            battle_spec_from_payload,
        )
        from pokezero.poke_engine_adapter import build_poke_engine_state

        sys.path.insert(0, str(SCRIPTS_DIR))
        from golden_encoder_backends import row_inputs_from_decision_row  # noqa: E402

        corpus = load_golden_corpus(COMMITTED_SAMPLE_DIR)
        fold_states = {}
        for record in iter_fold_records(
            COMMITTED_SAMPLE_DIR, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
        ):
            fold_states[int(record["array_row_index"])] = record["fold_state"]
        games = {game.record.battle_id: game for game in corpus.games}
        cls.position = None
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
            cls.position = {
                "state_str": state.to_string(),
                "row_inputs": json.dumps(row_inputs_from_decision_row(row), sort_keys=True),
                "ctx": json.dumps(
                    {
                        "p1": list(world.party_species["p1"]),
                        "p2": list(world.party_species["p2"]),
                        "turn": int(row.public_materialization.get("turn") or 0),
                    }
                ),
                "fold_state": fold_states[index],
                "self_side": "side_one" if row.player_id == "p1" else "side_two",
            }
            break
        if cls.position is None:
            raise unittest.SkipTest("no committed-sample row could be driven")

        # Random-weights artifact at the REAL v2.2 shape (throughput/mechanics
        # only — never a strength claim).
        export = _load_export_module()
        from pokezero.neural_policy import (
            EntityTokenTransformerPolicy,
            TransformerPolicyConfig,
        )

        tables = json.loads(cls.tables_json)
        config = TransformerPolicyConfig.compact_category(
            category_vocab=tuple(tables["vocab"]["tokens"]),
            category_oov_buckets=int(tables["vocab"]["oov_buckets"]),
            categorical_feature_count=int(tables["layout"]["categorical_feature_count"]),
            numeric_feature_count=int(tables["layout"]["numeric_feature_count"]),
            token_count=int(tables["layout"]["token_count"]),
            embedding_dim=32,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=64,
            dropout=0.0,
            observation_schema_version="pokezero.observation.v2.2",
        )
        torch.manual_seed(20260719)
        model = EntityTokenTransformerPolicy(config).eval()
        shim = export.build_exportable_module(model)
        cls.tmpdir = tempfile.TemporaryDirectory()
        artifact = Path(cls.tmpdir.name) / "priors_test_random.pt"
        export.export_torchscript(
            shim, export.make_random_inputs(config, export.TRACE_BATCH, seed=7), artifact
        )
        cls.native = pokezero_search.NativeLeafModel(
            str(artifact),
            device="cpu",
            window=1,
            tokens=int(tables["layout"]["token_count"]),
            categorical_features=int(tables["layout"]["categorical_feature_count"]),
            numeric_features=int(tables["layout"]["numeric_feature_count"]),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmpdir"):
            cls.tmpdir.cleanup()

    def _search(self, *, sims: int, batch: int, seed: int, model_priors: bool) -> dict:
        fold = pokezero_search.FoldState.from_payload(self.position["fold_state"])
        report = self.native.search_batched_multi_encoded(
            self.position["state_str"],
            sims,
            batch,
            self.tables_json,
            self.position["row_inputs"],
            self.position["ctx"],
            fold,
            2,  # max_depth
            1.4,
            seed,
            True,
            model_priors,
        )
        return json.loads(report)

    def test_priors_telemetry_and_batch1_determinism(self) -> None:
        first = self._search(sims=48, batch=1, seed=5, model_priors=True)
        second = self._search(sims=48, batch=1, seed=5, model_priors=True)
        self.assertEqual(first["side_one"], second["side_one"])
        self.assertEqual(first["side_two"], second["side_two"])
        self.assertEqual(first["chance_nodes"], second["chance_nodes"])
        self.assertTrue(first["model_priors"])
        root_priors = first["root_priors"]
        self.assertIsInstance(root_priors, list)
        self_arms = first[self.position["self_side"]]
        self.assertEqual(len(root_priors), len(self_arms))
        self.assertAlmostEqual(sum(root_priors), 1.0, places=3)
        self.assertEqual(first["prior_fallbacks"], 0)
        # Visit conservation in both modes.
        for side in ("side_one", "side_two"):
            self.assertEqual(sum(entry["visits"] for entry in first[side]), 48)

    def test_kill_switch_restores_uniform(self) -> None:
        report = self._search(sims=48, batch=1, seed=5, model_priors=False)
        self.assertFalse(report["model_priors"])
        self.assertIsNone(report["root_priors"])
        self.assertEqual(report["prior_branches"], 0)
        for side in ("side_one", "side_two"):
            self.assertEqual(sum(entry["visits"] for entry in report[side]), 48)

    def test_batched_priors_run(self) -> None:
        report = self._search(sims=64, batch=8, seed=11, model_priors=True)
        self.assertTrue(report["model_priors"])
        self.assertGreaterEqual(report["prior_branches"] + report["prior_fallbacks"], 0)
        for side in ("side_one", "side_two"):
            self.assertEqual(sum(entry["visits"] for entry in report[side]), 64)


if __name__ == "__main__":
    unittest.main()
