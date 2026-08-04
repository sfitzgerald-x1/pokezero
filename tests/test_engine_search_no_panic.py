"""The owner's exact crash repro, run end to end against the REAL installed wheel.

``rust/pokezero-search/tests/gen3_terminal_options.rs`` pins the invariant at the
source — `get_all_options` never returns an empty option vector. This file is the
other half: it drives the actual bench the crash was found in, because the
malformed state only arises several plies deep through the belief-world
constructor and the search's own exploration, not from any state a unit test
would think to build.

The failure it guards is a HARD CRASH, not a degraded decision:
``Node::maximize_ucb_for_side`` (third_party/poke-engine-src/src/mcts.rs)
initialises ``let mut choice = 0`` and returns it unchanged for an empty option
slice, so ``Node::expand`` indexes ``s2_options[0]`` into a zero-length vector and
panics with ``index out of bounds: the len is 0 but the index is 0``, aborting the
whole run. Before
``third_party/poke-engine-gen3-terminal-options.patch`` this fired within ~5 games
from essentially any seed block.

REBUILDING. This drives the real search, so it needs a wheel that carries BOTH
halves of the binding: the compiled extension and poke-engine-py's Python
wrapper (`python/poke_engine/__init__.py`), which is where
`monte_carlo_tree_search` — the entrypoint `pokezero.engine_search` calls —
actually lives. An install with only the extension imports cleanly and exposes
the native `mcts`, then dies mid-search on a bare AttributeError. Build in a venv
of your own (never the shared one), mirroring
tests/test_engine_move_trap_wiring.py:

    uv venv /path/to/venv --python 3.13
    uv pip install --python /path/to/venv/bin/python -e .
    scripts/setup_poke_engine.sh /path/to/venv/bin/python
    /path/to/venv/bin/python -m unittest tests.test_engine_search_no_panic

`setup_poke_engine.sh` is the only supported build: it fetches the pinned sdist,
applies third_party/poke-engine-gen3-patches.txt with --fuzz=0, and installs from
the sdist ROOT, whose pyproject sets `python-source = "python"` so the wrapper is
included. Verified to produce the entrypoint from a clean venv.

Skipped — never failed — when the wheel, the search entrypoint, or a built
Showdown checkout is missing, so a half-installed binding reports the build
command instead of a red test. Slow (a real 15-game bench), so it is not part of
the default fast suite.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest
from _showdown_root import showdown_root_str

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_SHOWDOWN_ROOT = showdown_root_str()
GAMES = 15
SEED_START = 7000


def _showdown_root() -> str | None:
    root = os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT
    return root if pathlib.Path(root).is_dir() else None


class EngineSearchDoesNotPanicTest(unittest.TestCase):
    """15 games from seed 7000 must all complete, with zero engine panics."""

    def setUp(self) -> None:
        from pokezero.poke_engine_adapter import (
            PokeEngineMctsEntrypointMissingError,
            require_mcts_entrypoint,
        )
        from pokezero.poke_engine_backend import probe_poke_engine

        if not probe_poke_engine().ready:
            self.skipTest("poke-engine is not installed/ready")
        # Probed in-process, BEFORE spawning the bench: the subprocess would only
        # surface a half-installed binding as a non-zero exit and a stack trace in
        # captured output, i.e. a red test for an environment problem. `sys.executable`
        # runs the bench, so this interpreter's binding is the one that matters.
        try:
            require_mcts_entrypoint()
        except PokeEngineMctsEntrypointMissingError as exc:
            self.skipTest(str(exc))
        if _showdown_root() is None:
            self.skipTest("no built Showdown checkout (set POKEZERO_SHOWDOWN_ROOT)")

    def test_the_bench_completes_without_an_index_panic(self) -> None:
        completed = subprocess.run(
            [
                sys.executable, "-m", "pokezero.engine_search",
                "--showdown-root", _showdown_root(),
                "--games", str(GAMES),
                "--seed-start", str(SEED_START),
                "--opponent", "simple-legal",
                "--out", os.devnull,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        combined = completed.stdout + completed.stderr

        # Assert on the panic FIRST: it is the specific regression, and a bare
        # exit-code assertion would not say which failure happened.
        self.assertNotIn(
            "index out of bounds",
            combined,
            "MCTS indexed an empty option vector — see "
            "third_party/poke-engine-gen3-terminal-options.patch",
        )
        self.assertNotIn("PanicException", combined, "the engine panicked")
        self.assertEqual(
            completed.returncode, 0, f"bench exited {completed.returncode}:\n{combined[-4000:]}"
        )
        # Every game must have produced a result line, so a run that exits 0
        # after silently dropping games still fails.
        self.assertEqual(
            sum(line.startswith("seed ") for line in combined.splitlines()),
            GAMES,
            f"expected {GAMES} completed games:\n{combined[-4000:]}",
        )


if __name__ == "__main__":
    unittest.main()
