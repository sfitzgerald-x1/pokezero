# C141 — the recharge gates, against an actually-injected bad write

**Era / provenance.** `corpus/golden-v4`, 1295 decision rows / 1271 same-seat boundaries.
Tree: `main` at `9a128bec` (task 4 merged). Engine crate rebuilt twice during this run, once
with the injection and once without; the injection is reverted and the tree is clean.

Producing commands, verbatim:

```sh
export POKEZERO_SHOWDOWN_ROOT=<showdown checkout>
# build the crate with (or without) the injection:
( cd rust/pokezero-search && CARGO_BUILD_JOBS=8 .venv/bin/python -m maturin build \
    --release --skip-auditwheel --out /tmp/wheel -i .venv/bin/python )
uv pip install --python .venv/bin/python --force-reinstall /tmp/wheel/*.whl
# measure:
.venv/bin/python scripts/leaf_root_parity.py \
    --corpus corpus/golden-v4 --tables corpus/encoder_tables_v4.json
```

## Why this exists

Task 4's middle clause is *"show they now catch an injected bad write."* PR #1156 shipped the
gate fix and the symmetry fix, but what it pinned was the **derivation** — unit tests over
fixtures showing that a write contradicting the parser tracker disagrees with the gate's world.
Review said so plainly and made me soften three "pinned two-way" claims:

> no test or measurement shows a gate catching an end-to-end bad write

That is the gap this closes. No new code — this is a measurement.

## The injected write

Added to `leaf.rs` beside the existing `opponent_must_recharge` write: the symmetric self-side
write the source comments describe, with a realistic copy-paste bug — it reads the **opponent's**
`volatile_statuses` instead of our own.

```rust
md.insert(
    "self_must_recharge".into(),
    json!(opp_side.volatile_statuses.contains(&PokemonVolatileStatus::MUSTRECHARGE)),
);
```

Chosen over "always true" or "always false" because those are caught by anything. This one is
correct on most rows and wrong only where the two sides' recharge states differ, which is the
shape a real bug takes.

Verified live rather than assumed: `self_must_recharge` present in the compiled `.so` → `True`
with the injection, `False` after the revert.

## Result

`leaf_root_parity`, same corpus, same denominator (1006 measured of 1295, 289 skipped) in all
three rows:

| run | exit | diverged | families |
|---|---|---|---|
| clean crate, current gates | **0** | **0** | — |
| **injected**, current gates | **1** | **5** | 5 × `self_team/CATEGORY_VOLATILE_OFFSET` `got=877 want=0` |
| **injected**, pre-task-4 gates | 1 | 5 | **4** × `self_team` + 1 × `opponent_team` `got=0 want=877` |
| revert + rebuild | **0** | **0** | byte-identical to the baseline row |

`877` is the MUSTRECHARGE volatile id — the same signature C112 records for P3.

**The acceptance clause is met.** The gates go from `diverged 0 / EXIT=0` to `diverged 5 /
EXIT=1` on an injected bad write, at an unchanged denominator, and return to exactly the
baseline when it is reverted. The catch is not a denominator artifact.

## What this does NOT show, measured rather than assumed

I expected the pre-task-4 gates to **ratify** this write. They mostly do not — they catch 4 of
the 5 self-side rows. The discrimination is real but narrow:

- the old gates **ratify 1 of the 5** self-side rows the new gates catch;
- and they report 1 **spurious** `opponent_team` divergence of their own — the known gate
  artifact at `(1009, 18, p1)`, where the partner decision row is absent from the corpus so the
  candidate rule misses a lock the tracker sees.

So on this corpus the honest claim is "the fixed gates catch strictly more, and stop inventing
one," not "the old gates were blind." The reason is structural and was already measured in
#1156: **the recorded action and the parser tracker disagree on exactly one row of
`corpus/golden-v4`**, and it is opponent-side. A corpus with no self-side disagreement cannot
discriminate the two derivations on the self side, however the write is chosen. The fixture
tests in `tests/test_recharge_gate_derivation.py` are what cover that, and they are fixtures for
this reason rather than for convenience.

Two further limits, stated so nobody reads more into the table:

- `leaf_vs_reality` is unchanged across all four rows (956+315, diverged 917). Its defect-class
  gate is dominated by the 917 pre-existing divergences, so a 5-row change does not move its
  verdict. `leaf_root_parity` is the gate that discriminates here because it was at zero.
- This exercises the SELF side only. The opponent side has been live since before this work.

## Disposition

No code change. C112's **P3 stays open**: `leaf.rs`'s self-side MUSTRECHARGE volatile is still
root-frozen, and this run is evidence that when someone does lift it, the gates will hold the
implementation honest — which is the property #1156 claimed and could not previously show.
