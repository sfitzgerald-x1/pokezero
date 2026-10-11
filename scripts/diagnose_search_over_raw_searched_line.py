"""Reproduce one excluded native-to-Showdown ancestry check with random weights.

Development fixture dependencies are explicit: tests/test_model_priors_search
supplies the existing real-v3-shape TorchScript artifact and encoder tables.
This driver does not accept scientific checkpoints, roots, panels or retries.
It emits a sanitized engineering readout to stdout; private states stay local.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_model_priors_search import _EncodedSearchFixture  # noqa: E402
from pokezero.mcts_eval.search_over_raw_searched_line import run_deterministic_fixture  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showdown-root", required=True, type=Path)
    args = parser.parse_args()
    # This shared test fixture creates fresh RANDOM model weights, not a
    # candidate checkpoint. Its corpus loader is only an encoder smoke setup;
    # the actual searched root comes exclusively from the authored fixture.
    _EncodedSearchFixture.setUpClass()
    try:
        result = run_deterministic_fixture(native_model=_EncodedSearchFixture.native,
            tables_json=_EncodedSearchFixture.tables_json, showdown_root=args.showdown_root)
        print(json.dumps(result, sort_keys=True, indent=2))
        # A correspondence mismatch is a NON-ZERO diagnostic result even when
        # capture/replay plumbing completed and its focused tests pass.
        return 1 if any(row["result"]["status"] != "MATCHED_PROJECTION_NOT_FULL_STATE"
            for row in result["evidence"]["leaves"]) else 0
    finally:
        _EncodedSearchFixture.tearDownClass()


if __name__ == "__main__":
    raise SystemExit(main())
