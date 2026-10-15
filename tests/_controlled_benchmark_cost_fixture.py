"""Authored bridge surfaces in a real owned subprocess; never model games."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pokezero.foulplay_bridge as actual_bridge

from pokezero.controlled_benchmark_cost import worker_main, _json


directory = Path(sys.argv[1])
mode = sys.argv[2]
recipe = _json(directory / "recipe.json")


def parser():
    if mode == "startup_failure":
        raise RuntimeError("authored startup failure")
    return actual_bridge.build_arg_parser()


def bridge(args):
    if mode in ("timeout", "escaped_child"):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   start_new_session=True)
        if mode == "timeout":
            time.sleep(60)
    if mode == "cpu_child":
        subprocess.run([sys.executable, "-c",
                        "import time; end=time.process_time()+.1\nwhile time.process_time()<end: pass"],
                       check=True)
    if mode == "missing_summary":
        return 0
    n = min(1, recipe["games"]) if mode == "partial_failure" else recipe["games"]
    config = actual_bridge._config_from_args(actual_bridge.build_arg_parser().parse_args(args))
    rows = tuple(actual_bridge.ControlledFoulPlayGameResult(
        battle_id=f"authored-{i}", seed=recipe["seed_start"] + i, winner="PokeZeroBot",
        pokezero_won=True, decision_rounds=3, pokezero_decisions=4,
        root_puct_searches=0, root_puct_fallbacks=0, capped=mode == "capped",
        policy_elapsed_seconds=(.025, .1)) for i in range(n))
    summary = actual_bridge.ControlledFoulPlayBenchmarkResult(
        config=config, policy_id="authored-not-a-model", games=rows,
        checkpoint_sha256=recipe["input_sha256"][str(Path(recipe["checkpoint"]).resolve())]).to_dict()
    if mode == "malformed_summary":
        summary["game_results"][0]["seed"] += 1
    (directory / "summary.json").write_text(json.dumps(summary))
    if mode == "input_drift":
        Path(recipe["checkpoint"]).write_bytes(b"authored changed checkpoint")
    if mode in ("partial_failure", "cleanup_failure"):
        raise RuntimeError("authored benchmark failure")
    return 0


sys.modules["pokezero.foulplay_bridge"] = SimpleNamespace(build_arg_parser=parser, main=bridge)
raise SystemExit(worker_main(directory))
