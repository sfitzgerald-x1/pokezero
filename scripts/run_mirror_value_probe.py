#!/usr/bin/env python3
"""Collect one immutable mirror cohort, then fit matched privileged/public heads.

All game records, feature arrays, predictions and head weights stay remote.
No replay/resume, no PPO/GAE, no test-driven head or epoch selection.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokezero.mirror_value_probe import (CHAMPION_SHA, cohort, collect_battle, digest,
    file_sha, load_runtime, paired_battle_interval, write_new, champion_linear_head)

_RUNTIME = None
_INITIALIZATION_ERROR = None


def initialize(checkpoint, showdown, node, threads):
    global _RUNTIME, _INITIALIZATION_ERROR
    # Pool replaces a process whose initializer raises. Bind failures must be
    # delivered as ordinary task failures, so the first failure stops the
    # cohort instead of creating an unbounded worker-restart loop.
    try:
        _RUNTIME = load_runtime(checkpoint, showdown, node, threads)
    except BaseException as error:
        _INITIALIZATION_ERROR = (type(error).__name__, str(error))


def worker(arguments):
    if _INITIALIZATION_ERROR is not None:
        kind, message = _INITIALIZATION_ERROR
        raise RuntimeError(f"mirror runtime initialization failed: {kind}: {message}")
    row, output, max_rounds = arguments
    return collect_battle(row, output, _RUNTIME, max_rounds=max_rounds)


def collect(config, output):
    if output.exists():
        raise ValueError("fresh output root required; no resume/refill")
    output.mkdir(parents=True, exist_ok=False)
    rows = cohort(config["root"], config["battles"])
    write_new(output / "cohort.json", dict(schema="pokezero.mirror_cohort.v1", config=config,
        config_sha256=digest(config), battles=rows, roster_sha256=digest(rows)))
    receipts = []
    started = time.monotonic()
    # One model + one Node simulator per process, at most workers*threads CPUs.
    # Pool's context exit terminates pending/active workers on the first error;
    # the outer PID1 owner reaps any remaining simulator descendants. Never
    # drain a queued scientific roster after a collector failure.
    with multiprocessing.get_context("spawn").Pool(processes=config["workers"],
        initializer=initialize, initargs=(config["checkpoint"], config["showdown"], config["node"], config["threads"])) as pool:
        arguments = [(row, str(output), config["max_rounds"]) for row in rows]
        for receipt in pool.imap_unordered(worker, arguments, chunksize=1):
            receipts.append(receipt)
            print(json.dumps(dict(event="battle_completed", completed=len(receipts),
                battle_id=receipt["battle_id"], paired_rows=receipt["paired_rows"],
                actual_terminal=receipt["actual_terminal"], elapsed_seconds=time.monotonic() - started)), flush=True)
    receipts.sort(key=lambda r: r["index"])
    if [r["battle_id"] for r in receipts] != [r["battle_id"] for r in rows]:
        raise ValueError("cohort coverage mismatch")
    manifest = dict(schema="pokezero.mirror_collection.v1", status="COMPLETE_FIXED_COHORT",
        config=config, config_sha256=digest(config), cohort_sha256=file_sha(output / "cohort.json"),
        battles=receipts, battle_receipt_hashes={f"battle-{r['index']:04d}/battle.json":
            file_sha(output / f"battle-{r['index']:04d}/battle.json") for r in rows},
        elapsed_seconds=time.monotonic() - started, new_games=len(receipts),
        paired_rows=sum(r["paired_rows"] for r in receipts),
        known_rows=sum(r["known_rows"] for r in receipts), unknown_rows=sum(r["unknown_rows"] for r in receipts),
        terminal_battles=sum(r["actual_terminal"] is not None and not r["actual_terminal"]["capped"] for r in receipts),
        no_refill=True, sibling_measurement=False)
    write_new(output / "collection.json", manifest)
    return manifest


def load_collection(root, expected_sha):
    root = Path(root)
    if file_sha(root / "collection.json") != expected_sha:
        raise ValueError("collection manifest binding")
    m = json.loads((root / "collection.json").read_text())
    if m["status"] != "COMPLETE_FIXED_COHORT" or m["config"]["champion_sha256"] != CHAMPION_SHA:
        raise ValueError("completed exact champion collection required")
    if m["config"].get("probe_contract") != "owning-wire-true-moves-champion-linear-v2":
        raise ValueError("corrected owning-wire and true-move collection required")
    expected_roster = cohort(m["config"]["root"], m["config"]["battles"])
    if len(m["battles"]) != len(expected_roster):
        raise ValueError("all fixed battles required, including unknowns")
    blocks = []
    for expected, row in zip(expected_roster, m["battles"]):
        if any(row[k] != expected[k] for k in expected):
            raise ValueError("frozen battle roster mismatch")
        dest = root / f"battle-{row['index']:04d}"
        rel = f"battle-{row['index']:04d}/battle.json"
        if file_sha(dest / "battle.json") != m["battle_receipt_hashes"][rel]:
            raise ValueError("battle receipt binding")
        receipt = json.loads((dest / "battle.json").read_text())
        for name in ("features.npz", "boundaries.jsonl.gz"):
            if file_sha(dest / name) != receipt["files"][name]:
                raise ValueError("battle artifact binding")
        with np.load(dest / "features.npz", allow_pickle=False) as data:
            block = {name: data[name] for name in data.files if not name.endswith("_tokens")}
            token_shapes = {arm: data[arm + "_tokens"].shape for arm in ("public", "privileged")}
        n = row["paired_rows"]
        if n % 2 or len(receipt["identities"]) != n:
            raise ValueError("paired row count")
        for a, b in zip(receipt["identities"][::2], receipt["identities"][1::2]):
            if a["seat"] != "p1" or b["seat"] != "p2" or any(a[k] != b[k] for k in ("root_id", "turn_number", "decision_round_index")):
                raise ValueError("same-root opposite-seat identity required")
        from pokezero.mirror_value_probe import terminal_labels
        y, status = terminal_labels(receipt["actual_terminal"], [r["seat"] for r in receipt["identities"]])
        if not np.array_equal(block["outcomes"], y, equal_nan=True) or not np.array_equal(block["label_status"], status):
            raise ValueError("actual terminal labels required")
        width = receipt["contract"]["width"]
        for arm in ("public", "privileged"):
            x, mask, pred = (block[arm + suffix] for suffix in ("_pooled", "_mask", "_values"))
            if (x.shape != (n, width) or token_shapes[arm] != (n, 23, width) or x.dtype != np.float32 or not np.isfinite(x).all()
                or mask.shape != (n, 23) or mask.dtype != bool or not mask.any(1).all()
                or pred.shape != (n,) or not np.isfinite(pred).all()):
                raise ValueError("frozen feature shape/dtype/finiteness")
        block.update(split=row["split"], battle_id=row["battle_id"])
        blocks.append(block)
    for split in ("train", "validation", "test"):
        if sum(int((b["label_status"] == 0).sum()) for b in blocks if b["split"] == split) == 0:
            raise ValueError("known terminal outcomes required in each preassigned split; no refill")
    return m, blocks


def predict(head, blocks, arm, split, batch_size=256):
    import torch
    predictions = []
    if head is not None:
        head.eval()
    with torch.no_grad():
        for b in blocks:
            if b["split"] != split:
                continue
            if head is None:
                p = b[arm + "_values"].copy()
            else:
                parts = []
                for start in range(0, len(b["outcomes"]), batch_size):
                    stop = start + batch_size
                    parts.append(head(torch.from_numpy(b[arm + "_pooled"][start:stop]),
                        torch.from_numpy(b[arm + "_mask"][start:stop]))[0].numpy())
                p = np.concatenate(parts)
            if not np.isfinite(p).all():
                raise ValueError("nonfinite head predictions")
            predictions.append(dict(battle_id=b["battle_id"], values=p, outcomes=b["outcomes"], status=b["label_status"]))
    return predictions


def evaluate(predictions):
    from pokezero.value_precision_probe import calibration_metrics
    p = np.concatenate([r["values"] for r in predictions])
    y = np.concatenate([r["outcomes"] for r in predictions])
    known = np.concatenate([r["status"] == 0 for r in predictions])
    result = calibration_metrics(p[known], y[known])
    loss = float(((p[known] - y[known]) ** 2).sum())
    missing = p[~known]
    result.update(total_rows=len(p), unknown_rows=int((~known).sum()),
        full_roster_mse_identification_bounds=[
            float((loss + np.square(np.maximum(np.abs(missing) - 1, 0)).sum()) / len(p)),
            float((loss + np.square(np.abs(missing) + 1).sum()) / len(p))],
        bounds_interpretation="identification bounds over unknown outcomes [-1,1], not confidence intervals")
    return result


def fit(collection_root, output, *, collection_sha, epochs=20, batch_size=256, seed=20261029, threads=8):
    import torch
    from pokezero.neural_policy import load_transformer_checkpoint
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if output.exists():
        raise ValueError("fresh refit directory required")
    manifest, blocks = load_collection(collection_root, collection_sha)
    output.mkdir(exist_ok=False)
    result = dict(schema="pokezero.mirror_value_refit.v1", collection_sha256=collection_sha,
        head_kind="champion_linear_tanh_warmstart", epochs=epochs, batch_size=batch_size, learning_rate=.001,
        seed=seed, selection="minimum validation MSE separately per arm; earliest epoch ties; matched champion linear+tanh architecture; no test selection",
        training="actual uncapped terminal outcomes only, MSE loss; no PPO/GAE/value clipping",
        split_unit="whole battle, both seats and all turns together", new_games=manifest["new_games"],
        paired_rows=manifest["paired_rows"], known_rows=manifest["known_rows"], unknown_rows=manifest["unknown_rows"],
        arms={}, baseline={})
    selected = {}
    checkpoint = Path(manifest["config"]["checkpoint"])
    if file_sha(checkpoint) != CHAMPION_SHA:
        raise ValueError("exact champion warm-start checkpoint required")
    champion, _ = load_transformer_checkpoint(checkpoint, map_location="cpu")
    champion.float().eval()
    width = blocks[0]["public_pooled"].shape[-1]
    result["initialization"] = dict(checkpoint_sha256=CHAMPION_SHA, parameters=width+1,
        copied_weight_and_bias=True, verified_before_any_optimizer_step=True, arms={})
    for arm in ("public", "privileged"):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        head = champion_linear_head(champion)
        initial_max_difference, bitwise = 0., True
        with torch.no_grad():
            for b in blocks:
                for start in range(0, len(b["outcomes"]), 128):
                    p = head(torch.from_numpy(b[arm + "_pooled"][start:start+128]), None)[0].numpy()
                    expected = b[arm + "_values"][start:start+128]
                    initial_max_difference = max(initial_max_difference, float(np.max(np.abs(p - expected))))
                    bitwise = bitwise and np.array_equal(p, expected)
        if initial_max_difference > 2e-7:
            raise ValueError("warm start predictions differ from champion: " + str(initial_max_difference))
        result["initialization"]["arms"][arm] = dict(max_abs_difference=initial_max_difference,
            bitwise_equal=bitwise, tolerance=2e-7, compared_rows=sum(len(b["outcomes"]) for b in blocks))
        optimizer = torch.optim.Adam(head.parameters(), lr=.001)
        history = []
        best, best_mse, best_epoch = None, math.inf, 0
        training = [b for b in blocks if b["split"] == "train"]
        for epoch in range(1, epochs + 1):
            head.train()
            for number in rng.permutation(len(training)):
                b = training[number]
                rows = np.flatnonzero(b["label_status"] == 0); rng.shuffle(rows)
                for start in range(0, len(rows), batch_size):
                    ix = rows[start:start + batch_size]
                    values, logits = head(torch.from_numpy(b[arm + "_pooled"][ix]), torch.from_numpy(b[arm + "_mask"][ix]))
                    target = torch.from_numpy(b["outcomes"][ix])
                    loss = (values - target).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            validation = evaluate(predict(head, blocks, arm, "validation"))["mse"]
            history.append(dict(epoch=epoch, validation_mse=validation))
            if validation < best_mse:
                best_mse, best_epoch = validation, epoch
                best = {k: v.detach().clone() for k, v in head.state_dict().items()}
            print(json.dumps(dict(event="head_epoch", arm=arm, epoch=epoch, validation_mse=validation)), flush=True)
        head.load_state_dict(best, strict=True)
        selected[arm] = head
        torch.save(dict(state_dict=best, kind="champion_linear_tanh_warmstart", width=width, selected_epoch=best_epoch,
            collection_sha256=collection_sha, arm=arm), output / (arm + "-head.pt"))
        result["arms"][arm] = dict(selected_epoch=best_epoch, validation_mse=best_mse,
            history=history, artifact_sha256=file_sha(output / (arm + "-head.pt")))
    # Both heads and epoch choices are frozen before computing test results.
    test = {}
    for arm in ("public", "privileged"):
        test[arm] = predict(selected[arm], blocks, arm, "test")
        result["arms"][arm]["test"] = evaluate(test[arm])
        result["baseline"][arm] = evaluate(predict(None, blocks, arm, "test"))
        with (output / (arm + "-test-predictions.npz")).open("xb") as stream:
            np.savez(stream, predictions=np.concatenate([r["values"] for r in test[arm]]),
                outcomes=np.concatenate([r["outcomes"] for r in test[arm]]),
                status=np.concatenate([r["status"] for r in test[arm]]))
    losses = {arm: [] for arm in test}
    for a, b in zip(test["public"], test["privileged"]):
        if a["battle_id"] != b["battle_id"]:
            raise ValueError("paired test battle identity mismatch")
        known = a["status"] == 0
        if known.any():
            for arm, row in (("public", a), ("privileged", b)):
                losses[arm].append(float(((row["values"][known] - row["outcomes"][known]) ** 2).mean()))
    result["paired_test"] = paired_battle_interval(losses["public"], losses["privileged"])
    public_mse = result["arms"]["public"]["test"]["mse"]
    privileged_mse = result["arms"]["privileged"]["test"]["mse"]
    result["observed_position_weighted_relative_mse_reduction"] = (public_mse - privileged_mse) / public_mse
    public_losses, privileged_losses = np.asarray(losses["public"]), np.asarray(losses["privileged"])
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(public_losses), (10000, len(public_losses)))
    shares = (public_losses[ix].mean(1) - privileged_losses[ix].mean(1)) / public_losses[ix].mean(1)
    result["paired_test"]["relative_mse_reduction"] = float((public_losses.mean() - privileged_losses.mean()) / public_losses.mean())
    result["paired_test"]["relative_reduction_interval90"] = np.quantile(shares, [.05, .95]).tolist()
    result["paired_test"]["preassigned_test_battles"] = sum(b["split"] == "test" for b in blocks)
    result["paired_test"]["unknown_test_battles"] = result["paired_test"]["preassigned_test_battles"] - len(public_losses)
    bounds = []
    for a, b in zip(test["public"], test["privileged"]):
        known = a["status"] == 0
        delta = b["values"] - a["values"]
        base = b["values"] ** 2 - a["values"] ** 2
        actual = (b["values"][known] - b["outcomes"][known]) ** 2 - (a["values"][known] - a["outcomes"][known]) ** 2
        lower, upper = base - 2 * np.abs(delta), base + 2 * np.abs(delta)
        lower[known] = actual; upper[known] = actual
        bounds.append([float(lower.mean()), float(upper.mean())])
    result["paired_test"]["full_roster_difference_identification_bounds"] = np.mean(bounds, axis=0).tolist()
    result["paired_test"]["identification_bounds_interpretation"] = "Equal-battle point-estimand bounds over unknown outcomes in [-1,1]; not confidence intervals. Bootstrap intervals describe terminal-known battles only."
    result.update(status="TWO_MATCHED_OUTCOME_REFITS_COMPLETE", model_fits=2,
        limitations=["Privileged rows include true current move identities and PP; no token-budget or architecture expansion.",
            "Only simultaneous requested-seat roots enter A2; forced replacement boundaries retained separately as provenance.",
            "Outcome residuals combine randomness and model error; this is no identified hidden-information fraction.",
            "Unrefit privileged champion baseline is out of distribution, not a learned full-information evaluator.",
            "No strength, search, PPO, or sibling-action measurement; no future revealed facts enter earlier observations."])
    result["artifact_hashes"] = {p.name: file_sha(p) for p in sorted(output.iterdir())}
    write_new(output / "refit-summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "fit"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--collection-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "collect":
        if file_sha(args.config) != args.config_sha256:
            raise ValueError("config binding")
        result = collect(json.loads(args.config.read_text()), args.output)
    else:
        result = fit(args.collection, args.output, collection_sha=args.collection_sha256)
    print(json.dumps({k: result[k] for k in ("status", "new_games", "paired_rows", "known_rows", "unknown_rows")}), flush=True)


if __name__ == "__main__":
    main()
