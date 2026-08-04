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
                  callback NAMES alone do not discriminate: two independent
                  readers resolved this one differently in the same review cycle.

Each cost a wrong claim in a hand-off before it was caught. None would have
happened against a resolved read.

`--sources` closes the last of those traps
==========================================

Callback *names* were never enough — `stall` proved it. `--sources` reports the
mod that contributes each field and each callback **body**, which is the thing
the reader actually needs. It reproduces the documented answer for every trap
above that turns on provenance:

    stall     -> callback bodies from **gen5**      (not base: the x2 ladder)
    flail     -> callback bodies from **gen3**      (gen3's own 48-scale override)
    brn       -> callback bodies from **gen6**      (the residual fraction change)
    rest      -> values from **gen8**: pp           (declared below gen3)

Read the `**mod**` in that line before opening any source file; it tells you
which file to open. Guessing it is the step that has gone wrong five times.

Usage::

    python scripts/gen3_dex_resolve.py toxic spikes flail
    python scripts/gen3_dex_resolve.py --condition slp tox confusion
    python scripts/gen3_dex_resolve.py --condition stall --sources   # whose body?
    python scripts/gen3_dex_resolve.py --json toxic        # full resolved object
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Deliberately NOT pokezero.local_showdown.default_showdown_root(): this script drives `node`
# directly and imports no pokezero module, so reaching for the shared helper would add a hard
# dependency to a standalone tool. It does honour POKEZERO_SHOWDOWN_ROOT, which is the axis
# that actually drifts -- a de-personalized fallback that silently ignores the override is the
# same bug in a quieter form.
DEFAULT_SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT")
    or Path.home() / "workspace" / "pokerena" / "vendor" / "pokemon-showdown"
)

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

# --- WHICH MOD CONTRIBUTED THE BODY -----------------------------------------
#
# `_SUMMARY_JS` answers "which callbacks survive"; this answers "whose body".
# That second question is the one the `stall` trap turned on: base and gen5 both
# define onStart/onRestart/onStallMove, so the NAMES are identical either way and
# two readers resolved the same `inherit: true` to different generations.
#
# It works by walking the mod chain and reporting, per field, the FIRST mod that
# owns it — an `inherit: true` entry contributes only the keys it names, so the
# first owner of a callback is the mod whose body survives.
#
# **This must never touch `Dex`.** `Dex.mod()` merges inherited data INTO the
# required data objects in place, and `require` caches those objects, so any
# raw-table read after a `Dex.mod()` call in the same process reports the MERGED
# result. Measured: with a `Dex.mod('gen3')` first, gen3 appears to own `stall`
# outright (`counterMax: 8` + all three callbacks) when its source file does not
# mention `stall` at all. Hence the separate process below, and hence this
# comment — the failure is silent and looks like a legitimate answer.
_SOURCE_JS = r"""
const fs = require('fs');
const kind = process.argv[1];
const chain = process.argv[2].split(',');
const ids = process.argv.slice(3);
const file = (mod) => mod === 'base'
  ? './dist/data/' + (kind === 'condition' ? 'conditions' : 'moves') + '.js'
  : './dist/data/mods/' + mod + '/' + (kind === 'condition' ? 'conditions' : 'moves') + '.js';
const out = {};
for (const id of ids) out[id] = {_chain: chain, owners: {}, inheritsAt: []};
for (const mod of chain) {
  const f = file(mod);
  if (!fs.existsSync(f)) continue;
  const mo = require(f);
  const table = mo.Conditions || mo.Moves || mo.conditions || mo.moves;
  if (!table) continue;
  for (const id of ids) {
    const e = table[id];
    if (!e) continue;
    if (e.inherit) out[id].inheritsAt.push(mod);
    for (const k of Object.keys(e)) {
      if (k === 'inherit') continue;
      // First mod in the chain to own a key is the one whose value/body wins.
      if (!(k in out[id].owners)) {
        out[id].owners[k] = {mod, isFunction: typeof e[k] === 'function'};
      }
    }
  }
}
console.log(JSON.stringify(out));
"""


def _mod_chain(showdown_root: Path, gen: str) -> list:
    """Mod inheritance chain, discovered in its OWN process (see _SOURCE_JS)."""
    js = (
        "const {Dex}=require('./dist/sim/dex.js');let m=process.argv[1],c=[];"
        "for(let i=0;i<16&&m;i++){c.push(m);m=Dex.mod(m).parentMod;}"
        "console.log(JSON.stringify(c));"
    )
    r = subprocess.run(
        ["node", "-e", js, "--", gen],
        cwd=str(showdown_root), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"mod-chain discovery failed:\n{r.stderr[:600]}")
    return json.loads(r.stdout)


def resolve_sources(ids, *, showdown_root: Path, gen: str = "gen3", kind: str = "move") -> dict:
    """Per-field owning mod. Runs in a process that never imports Dex."""
    chain = _mod_chain(showdown_root, gen)
    r = subprocess.run(
        ["node", "-e", _SOURCE_JS, "--", kind, ",".join(chain), *ids],
        cwd=str(showdown_root), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"source resolution failed:\n{r.stderr[:600]}")
    return json.loads(r.stdout)


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
    ap.add_argument(
        "--sources", action="store_true",
        help="also report WHICH mod contributes each field and callback body "
             "(the check that separates identical callback names across mods)",
    )
    args = ap.parse_args(argv)

    kind = "condition" if args.condition else "move"
    resolved = resolve(args.ids, showdown_root=args.showdown_root, gen=args.gen, kind=kind)
    sources = (
        resolve_sources(args.ids, showdown_root=args.showdown_root, gen=args.gen, kind=kind)
        if args.sources else {}
    )

    if args.json:
        if sources:
            for id_, src in sources.items():
                if isinstance(resolved.get(id_), dict):
                    resolved[id_]["_sources"] = src
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
        src = sources.get(id_)
        if src:
            owners = src.get("owners") or {}
            if src.get("inheritsAt"):
                print(f"  {'':<14}   inherit: true at {', '.join(src['inheritsAt'])}")
            bodies = {k: v for k, v in owners.items() if v.get("isFunction")}
            values = {k: v for k, v in owners.items() if not v.get("isFunction")}
            for label, group in (("callback bodies from", bodies), ("values from", values)):
                if not group:
                    continue
                by_mod = {}
                for field, info in sorted(group.items()):
                    by_mod.setdefault(info["mod"], []).append(field)
                rendered = "; ".join(
                    f"**{mod}**: {', '.join(fields)}" for mod, fields in by_mod.items()
                )
                print(f"  {'':<14}   {label} {rendered}")
    print(
        "\nNote: a value here is the SIMULATOR's, chain fully applied. Callback names "
        "are listed because the rule often lives in the body (basePowerCallback, "
        "durationCallback, onResidual) — read those in source, but only after "
        "confirming HERE which file the callback survives from."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
