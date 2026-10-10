"""Create-only Phase A preparation; no search or battle launch.

Usage: PYTHONPATH=src python scripts/prepare_search_over_raw.py --output DIR
  --exposure-registration historical.json --exposure-registration qualification.json
  --plan approved-plan.md
Every exposure registration must have an explicit top-level ``seeds`` roster.
Preparing a cohort is not validation of the scientific execution environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import uuid

from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, phase_a_contract, require)


def prepare(exposure_registrations, plan, output, *, namespace=None):
    bindings = {}
    excluded = set(ENGINEERING_EXCLUDED_SEEDS)
    for path in exposure_registrations:
        path = Path(path).resolve()
        raw = path.read_bytes()
        registration = json.loads(raw)
        seeds = registration.get("seeds")
        require(isinstance(seeds, list) and seeds and all(type(s) is int and 0 <= s < 2**32 for s in seeds),
            "exposure registration needs a nonempty uint32 seeds roster")
        excluded.update(seeds)
        bindings[str(path)] = hashlib.sha256(raw).hexdigest()
    require(bindings, "explicit exposure inventory required")
    plan = Path(plan).resolve()
    bindings[str(plan)] = hashlib.sha256(plan.read_bytes()).hexdigest()
    configs = [SearchConfiguration("raw")]
    for arm in ("incumbent", "reference"):
        for belief in ("public", "oracle"):
            for leaf in ("model", "hp_fraction", "raw_rollout"):
                for seconds in (1., 3., 10.):
                    configs.append(SearchConfiguration(arm, belief, leaf, seconds,
                        20 if arm == "reference" else 1))
    contract = phase_a_contract(namespace or str(uuid.uuid4()),
        excluded_seeds=sorted(excluded), configurations=configs)
    contract.update(input_hashes=bindings, excluded_seeds=sorted(excluded),
        exposure_scope="supplied registrations only; full historical exposure inventory must be checked before collection",
        execution_ready=False, remaining_gates=["bind source/model/engine/simulator identities",
            "implement and test public-only search adapters and fresh source-root collector",
            "benchmark paired terminal continuations and audit resource ceilings",
            "seal validation evidence and freeze exploration selection before opening it"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "contract.json").open("x") as stream:
        json.dump(contract, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-registration", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.exposure_registration, args.plan, args.output)
    print(json.dumps(dict(status=result["status"], namespace=result["namespace"],
        configurations=len(result["configurations"]), root_slots=400, execution_ready=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
