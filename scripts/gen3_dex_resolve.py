#!/usr/bin/env python
"""Resolve gen3 move/condition data through Showdown's OWN Dex.

Read this instead of `data/moves.ts`, and instead of walking the mod chain by
hand. `Dex.mod('gen3')` applies the whole inheritance chain the simulator
actually uses; a hand-walked chain is a guess that happens to be right most of
the time, which is the worst kind.

The inheritance-chain trap has produced five wrong reads in this program:

  * Spikes      — the layer fractions live in the gen4 mod, not base
  * burn        — the residual fraction changed at gen6, so base is wrong for gen3
  * Flail       — gen3 has its OWN override; the gen4 ladder is not it
  * Thunder Wave — gen6 declares a value BELOW gen3 in the chain
  * Rest (PP)   — gen8 declares it; neither gen3 nor base carries the value
  * stall       — gen4's `inherit: true` resolves to GEN5's full definition, not
                  base, so the Protect decay ladder is x2 (1/2, 1/4, 1/8) and not
                  x3. Both base and gen5 define the same three callbacks, so the
                  callback NAMES below are identical either way and do not
                  discriminate — this probe cannot yet close that trap. Reporting
                  the surviving callback's source mod would.

Each cost a wrong claim in a hand-off before it was caught. None would have
happened against a resolved read.

Usage::

    python scripts/gen3_dex_resolve.py toxic spikes flail
    python scripts/gen3_dex_resolve.py --condition slp tox confusion
    python scripts/gen3_dex_resolve.py --json toxic        # full resolved object
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_SHOWDOWN_ROOT = Path.home() / "workspace" / "pokerena" / "vendor" / "pokemon-showdown"

# Fields worth showing by default. Everything else is in --json.
_MOVE_FIELDS = ("accuracy", "basePower", "type", "category", "pp", "priority", "target")
_SUMMARY_JS = r"""
const {Dex} = require('./dist/sim/dex.js');
const gen = Dex.mod(process.argv[1]);
const kind = process.argv[2];
const ids = process.argv.slice(3);
const out = {};
for (const id of ids) {
  const entry = kind === 'condition' ? gen.conditions.get(id) : gen.moves.get(id);
  if (!entry || !entry.exists) { out[id] = {missing: true}; continue; }
  out[id] = JSON.parse(JSON.stringify(entry));
  // Callback bodies are where HP-proportional / duration rules live; JSON drops
  // functions, so record WHICH ones exist — their presence is the finding.
  out[id]._callbacks = Object.keys(entry).filter(k => typeof entry[k] === 'function');
  if (entry.condition) {
    out[id]._conditionCallbacks =
      Object.keys(entry.condition).filter(k => typeof entry.condition[k] === 'function');
  }
}
console.log(JSON.stringify(out));
"""


def resolve(ids, *, showdown_root: Path, gen: str = "gen3", kind: str = "move") -> dict:
    result = subprocess.run(
        ["node", "-e", _SUMMARY_JS, "--", gen, kind, *ids],
        cwd=str(showdown_root), capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"dex resolution failed (is {showdown_root}/dist built?):\n{result.stderr[:600]}"
        )
    return json.loads(result.stdout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="+", help="move or condition ids")
    ap.add_argument("--gen", default="gen3")
    ap.add_argument("--condition", action="store_true", help="resolve conditions, not moves")
    ap.add_argument("--showdown-root", type=Path, default=DEFAULT_SHOWDOWN_ROOT)
    ap.add_argument("--json", action="store_true", help="dump the full resolved object")
    args = ap.parse_args(argv)

    kind = "condition" if args.condition else "move"
    resolved = resolve(args.ids, showdown_root=args.showdown_root, gen=args.gen, kind=kind)

    if args.json:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0

    print(f"resolved through Dex.mod({args.gen!r}) — {kind}s")
    for id_ in args.ids:
        entry = resolved.get(id_) or {}
        if entry.get("missing"):
            print(f"  {id_:<14} (does not exist in {args.gen})")
            continue
        fields = " ".join(
            f"{name}={entry.get(name)!r}" for name in _MOVE_FIELDS if name in entry
        )
        print(f"  {id_:<14} {fields}")
        for key in ("_callbacks", "_conditionCallbacks"):
            if entry.get(key):
                label = "callbacks" if key == "_callbacks" else "condition callbacks"
                print(f"  {'':<14}   {label}: {', '.join(sorted(entry[key]))}")
    print(
        "\nNote: a value here is the SIMULATOR's, chain fully applied. Callback names "
        "are listed because the rule often lives in the body (basePowerCallback, "
        "durationCallback, onResidual) — read those in source, but only after "
        "confirming HERE which file the callback survives from."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
