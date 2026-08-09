#!/usr/bin/env python
"""H8: measure the matched mass riding on the comparator's +/-9 % fallback window.

Ledger row H8 (`reports/c138_known_gaps_ledger.md`) says the window
``[0.92*eng - 1, 1.09*eng + 1]`` in :func:`roll_components_agree` "carries
unmeasured matched mass", and names the settling measurement: *count boundaries
whose accept came from the window rather than exact fan membership*. The
state-level counter ``strict:no_damage_rolls`` is 0 in both windows, which bounds
the STATE-level fallback at zero and says nothing about the per-branch one --
because the window is also reached when ``legal`` IS available and the observed
magnitude simply is not in it.

Two modes, and they answer two different questions:

``--mode count``
    Leaves the comparator's behaviour EXACTLY as it ships and only tallies how
    often the window is the accept path, split by whether ``legal`` was
    unavailable (``window_accept_legal_none``) or available-but-missed
    (``window_accept_legal_miss``). This is a usage census: an upper bound on
    the mass at risk, not the mass that depends on it, because a boundary can
    match on a branch that never touched the window.

``--mode disable``
    Removes the window accept entirely -- exact equality or membership in the
    enumerated fan, nothing else. The DELTA in ``transitions_matched`` against a
    ``--mode count`` run of the same window on the same build is the number of
    boundaries whose accept actually DEPENDED on the window. That is H8's
    number; the census above is not.

Nothing here reimplements the comparator. Both variants are produced by an AST
rewrite of the SHIPPED source of :func:`roll_components_agree`, obtained with
``inspect.getsource``, so the file under the certification pin
(``scripts/engine_transition_differential.py``) is never edited and the two
variants cannot drift from it: if the shipped function changes shape, the
rewrite fails loudly rather than silently measuring an old body.

The rewrite is verified before it is installed:

* exactly ONE window test is found (``if not (low <= magnitude <= high)``), and
  the search is by structure, not by line number or text;
* in ``count`` mode the transformed AST differs from the original by exactly the
  inserted counter statement, checked by re-dumping both with the insertion
  removed;
* in ``disable`` mode the transformed body ends the loop iteration with
  ``return False`` at that site.

Usage::

    PYTHONPATH=src python scripts/c152_h8_window_census.py \\
        --mode count --census out_census.json -- \\
        --games 200 --seed-start 19000000 --keep-repro 25 --json out_sweep.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

C152_COUNTS: Counter = Counter()

_MARKER = "_c152_window_tally"


def _is_window_test(node: ast.AST) -> bool:
    """Structural match for ``if not (low <= magnitude <= high): return False``.

    Matched on shape rather than on source text so a comment or reflow cannot
    make this silently find nothing -- and the caller asserts there is exactly
    one, so a second window appearing would also be a loud failure rather than a
    half-measurement.
    """
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
        return False
    inner = test.operand
    if not isinstance(inner, ast.Compare) or len(inner.ops) != 2:
        return False
    if not all(isinstance(op, ast.LtE) for op in inner.ops):
        return False
    names = {
        getattr(inner.left, "id", None),
        *(getattr(c, "id", None) for c in inner.comparators),
    }
    if names != {"low", "magnitude", "high"}:
        return False
    return len(node.body) == 1 and isinstance(node.body[0], ast.Return)


def _find_window_tests(tree: ast.AST) -> list[tuple[ast.AST, str, int]]:
    found: list[tuple[ast.AST, str, int]] = []
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            for index, item in enumerate(value):
                if _is_window_test(item):
                    found.append((parent, field, index))
    return found


def _tally_stmt() -> ast.stmt:
    """``_c152_window_tally(legal)`` -- the only thing ``count`` mode inserts."""
    return ast.parse(f"{_MARKER}(legal)").body[0]


def build_variant(mode: str) -> tuple[Any, str]:
    import engine_transition_differential as etd

    original_src = textwrap.dedent(inspect.getsource(etd.roll_components_agree))
    tree = ast.parse(original_src)
    sites = _find_window_tests(tree)
    if len(sites) != 1:
        raise SystemExit(
            f"expected exactly one +/-9% window test in roll_components_agree, "
            f"found {len(sites)}. The shipped comparator changed shape; fix this "
            f"script rather than measuring an old body."
        )
    parent, field, index = sites[0]
    block = getattr(parent, field)

    if mode == "count":
        block.insert(index + 1, _tally_stmt())
        # The transformed tree must differ from the original by EXACTLY the
        # inserted statement. Re-dump with it removed and compare.
        check = copy.deepcopy(tree)
        c_parent, c_field, c_index = _find_window_tests(check)[0]
        del getattr(c_parent, c_field)[c_index + 1]
        if ast.dump(check) != ast.dump(ast.parse(original_src)):
            raise SystemExit("count-mode rewrite changed more than the inserted tally")
    elif mode == "disable":
        block[index] = ast.parse("return False").body[0]
    else:
        raise SystemExit(f"unknown mode {mode}")

    ast.fix_missing_locations(tree)
    transformed_src = ast.unparse(tree)
    namespace = dict(etd.__dict__)
    namespace[_MARKER] = lambda legal: C152_COUNTS.update(
        ["window_accept_legal_none" if legal is None else "window_accept_legal_miss"]
    )
    exec(compile(tree, filename=f"<c152 {mode}>", mode="exec"), namespace)
    return namespace["roll_components_agree"], transformed_src


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("count", "disable"), required=True)
    parser.add_argument("--census", required=True, help="write the census JSON here")
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="arguments forwarded verbatim to engine_transition_differential.main",
    )
    args = parser.parse_args(argv)
    forwarded = [a for a in args.rest if a != "--"]

    import engine_transition_differential as etd

    variant, transformed_src = build_variant(args.mode)
    etd.roll_components_agree = variant

    status = etd.main(forwarded)

    census = {
        "schema": "c152-h8-window-census/1",
        "mode": args.mode,
        "what": (
            "count: shipped behaviour, tallying every component accepted by the "
            "+/-9% window. disable: the window accept removed, so the sweep's "
            "transitions_matched is the count that survives WITHOUT it."
        ),
        "forwarded_argv": forwarded,
        "counts": dict(C152_COUNTS),
        "transformed_source": transformed_src,
    }
    Path(args.census).write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
    print(f"-> {args.census}: {dict(C152_COUNTS)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
