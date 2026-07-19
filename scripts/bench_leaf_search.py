"""End-to-end model-priced search bench: REAL leaf observations vs the
template stub (the engine-swap capstone measurement).

Runs the full loop — root state -> decision/chance tree -> branch ->
synthesized events -> per-outcome fold advance -> native leaf encode ->
batched TorchScript eval -> exact-expectation backup — on real golden-corpus
positions (world from the recorded payload + true teams, root fold from the
schema-v2 sidecar), and compares it against the #716-style template-stub
batched search (identical tree mechanics, leaf rows copied from a constant
template) to price the real-observation overhead honestly.

The model is a RANDOM-WEIGHTS artifact at the real v2.2 shape (151 tokens x
51 categorical x 155 numeric, window 1) with small transformer dims — this is
a throughput measurement, never a strength claim.

Usage:
    PYTHONPATH=src python scripts/bench_leaf_search.py \
        --corpus corpus/golden-v2 --tables corpus/encoder_tables.json \
        [--positions 3] [--sims 256,1024] [--depths 2,3] [--batch 16] \
        [--devices cpu] [--embedding-dim 64] [--json report.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy  # noqa: E402
import pokezero_search  # noqa: E402
import torch  # noqa: E402

from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    battle_spec_from_payload,
)
from pokezero.golden_corpus import (  # noqa: E402
    GOLDEN_CORPUS_SCHEMA_VERSION,
    load_golden_corpus,
)
from pokezero.golden_corpus_fold import iter_fold_records  # noqa: E402
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT  # noqa: E402
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402

from golden_encoder_backends import row_inputs_from_decision_row  # noqa: E402


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_model", REPO_ROOT / "scripts" / "export_model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_artifact(tables: Mapping[str, Any], out_dir: Path, *, embedding_dim: int, device: str) -> Path:
    """Random-weights TorchScript artifact at the real v2.2 observation shape."""

    from pokezero.neural_policy import EntityTokenTransformerPolicy, TransformerPolicyConfig

    export = _load_export_module()
    config = TransformerPolicyConfig.compact_category(
        category_vocab=tuple(tables["vocab"]["tokens"]),
        category_oov_buckets=int(tables["vocab"]["oov_buckets"]),
        categorical_feature_count=int(tables["layout"]["categorical_feature_count"]),
        numeric_feature_count=int(tables["layout"]["numeric_feature_count"]),
        token_count=int(tables["layout"]["token_count"]),
        embedding_dim=embedding_dim,
        transformer_layers=1,
        attention_heads=2,
        feedforward_dim=2 * embedding_dim,
        dropout=0.0,
        observation_schema_version="pokezero.observation.v2.2",
    )
    torch.manual_seed(20260719)
    model = EntityTokenTransformerPolicy(config).eval()
    shim = export.build_exportable_module(model)
    if device == "mps":
        shim = shim.to("mps")
    path = out_dir / f"bench_leaf_random_{device}.pt"
    inputs = export.make_random_inputs(config, export.TRACE_BATCH, seed=7)
    if device == "mps":
        inputs = tuple(t.to("mps") for t in inputs)
    export.export_torchscript(shim, inputs, path)
    return path


def drivable_positions(corpus_dir: Path, count: int) -> list[dict[str, Any]]:
    """First `count` move-request rows (one per battle) whose world constructs."""

    corpus = load_golden_corpus(corpus_dir)
    fold_states: dict[int, Mapping[str, Any]] = {}
    for record in iter_fold_records(
        corpus_dir, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
    ):
        fold_states[int(record["array_row_index"])] = record["fold_state"]
    games = {game.record.battle_id: game for game in corpus.games}
    dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
    positions: list[dict[str, Any]] = []
    used_battles: set[str] = set()
    for index, row in enumerate(corpus.decision_rows):
        if len(positions) >= count:
            break
        if row.battle_id in used_battles:
            continue
        if row.observation_metadata.get("request_kind") != "move":
            continue
        # Mid-game positions (some history in the fold) price the realistic
        # regime; skip the empty-history openers.
        if int(row.observation_metadata.get("turn_number") or 0) < 5:
            continue
        game = games[row.battle_id]
        packed = {s: (game.record.true_teams.get(s) or {}).get("packed") for s in ("p1", "p2")}
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
        fold_state = fold_states.get(index)
        if fold_state is None:
            continue
        used_battles.add(row.battle_id)
        positions.append(
            {
                "label": f"{row.battle_id}#r{row.decision_round_index}/{row.player_id}",
                "state_str": state.to_string(),
                "row_inputs": json.dumps(row_inputs_from_decision_row(row), sort_keys=True),
                "ctx": json.dumps(
                    {
                        "p1": list(world.party_species["p1"]),
                        "p2": list(world.party_species["p2"]),
                        "turn": int(row.public_materialization.get("turn") or 0),
                    }
                ),
                "fold_state": fold_state,
                "template": template_arrays(row),
            }
        )
    return positions


def template_arrays(row: Any) -> dict[str, list]:
    """The row's own golden observation as flat template buffers (stub arm)."""

    arrays = row.arrays
    return {
        "categorical_ids": numpy.asarray(arrays.categorical_ids, dtype=numpy.int64)
        .flatten()
        .tolist(),
        "numeric_features": numpy.asarray(arrays.numeric_features, dtype=numpy.float32)
        .flatten()
        .tolist(),
        "token_type_ids": numpy.asarray(arrays.token_type_ids, dtype=numpy.int64)
        .flatten()
        .tolist(),
        "attention_mask": [bool(v) for v in numpy.asarray(arrays.attention_mask).flatten()],
        "history_mask": [True],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--sims", default="256,1024")
    parser.add_argument("--depths", default="2,3")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    tables_json = args.tables.read_text(encoding="utf-8")
    tables = json.loads(tables_json)
    sims_list = [int(v) for v in args.sims.split(",")]
    depth_list = [int(v) for v in args.depths.split(",")]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    positions = drivable_positions(args.corpus, args.positions)
    if not positions:
        print("no drivable positions found", file=sys.stderr)
        return 2
    print(f"positions: {[p['label'] for p in positions]}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for device in devices:
            artifact = build_artifact(
                tables, Path(tmp), embedding_dim=args.embedding_dim, device=device
            )
            native = pokezero_search.NativeLeafModel(
                str(artifact),
                device=device,
                window=1,
                tokens=int(tables["layout"]["token_count"]),
                categorical_features=int(tables["layout"]["categorical_feature_count"]),
                numeric_features=int(tables["layout"]["numeric_feature_count"]),
            )
            for position in positions:
                fold = pokezero_search.FoldState.from_payload(position["fold_state"])
                template = position["template"]
                for depth in depth_list:
                    for sims in sims_list:
                        stub = json.loads(
                            native.search_batched_multi(
                                position["state_str"],
                                sims,
                                args.batch,
                                template["categorical_ids"],
                                template["numeric_features"],
                                template["token_type_ids"],
                                template["attention_mask"],
                                template["history_mask"],
                                depth,
                                1.4,
                                args.seed,
                                True,
                            )
                        )
                        encoded = json.loads(
                            native.search_batched_multi_encoded(
                                position["state_str"],
                                sims,
                                args.batch,
                                tables_json,
                                position["row_inputs"],
                                position["ctx"],
                                fold,
                                depth,
                                1.4,
                                args.seed,
                                True,
                            )
                        )
                        row = {
                            "device": device,
                            "position": position["label"],
                            "depth": depth,
                            "sims": sims,
                            "batch": args.batch,
                            "stub_elapsed_s": stub["elapsed_s"],
                            "stub_sims_per_s": stub["iterations_per_s"],
                            "stub_model_evals": stub["model_evals"],
                            "encoded_elapsed_s": encoded["elapsed_s"],
                            "encoded_sims_per_s": encoded["iterations_per_s"],
                            "encoded_model_evals": encoded["model_evals"],
                            "encoded_lossy_renders": encoded["lossy_renders"],
                            "encoded_branch_folds": encoded["branch_folds"],
                            "overhead_x": (
                                encoded["elapsed_s"] / stub["elapsed_s"]
                                if stub["elapsed_s"] > 0
                                else float("inf")
                            ),
                            "encode_overhead_us_per_eval": (
                                1e6
                                * (encoded["elapsed_s"] - stub["elapsed_s"])
                                / max(1, encoded["model_evals"])
                            ),
                            "stub_argmax": max(
                                stub["side_one"], key=lambda a: a["visits"]
                            )["move"],
                            "encoded_argmax": max(
                                encoded["side_one"], key=lambda a: a["visits"]
                            )["move"],
                        }
                        results.append(row)
                        print(
                            f"[{device}] {position['label']} depth={depth} sims={sims}: "
                            f"stub {stub['iterations_per_s']:8.1f} sims/s "
                            f"({1000*stub['elapsed_s']:7.1f} ms) | "
                            f"encoded {encoded['iterations_per_s']:8.1f} sims/s "
                            f"({1000*encoded['elapsed_s']:7.1f} ms) | "
                            f"overhead {row['overhead_x']:5.2f}x "
                            f"({row['encode_overhead_us_per_eval']:6.1f} us/eval) | "
                            f"evals {encoded['model_evals']} lossy {encoded['lossy_renders']}"
                        )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
