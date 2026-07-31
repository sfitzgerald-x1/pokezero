# C26 Result: Substitute Clears Active Partial Trapping

## Ground Truth

The Showdown-backed `partialtrapsubstitute` fixture ran eight fixed seeds. Fire
Spin landed on four; all four exercised the measured boundary and passed:

1. Ninetales successfully used Fire Spin on Blissey.
2. On the next boundary Blissey successfully used Substitute.
3. Showdown emitted `-end ... Fire Spin [partiallytrapped] [silent]` before
   upkeep and emitted no Fire Spin residual damage on that boundary.

Missed Fire Spin setup seeds were skipped rather than treated as evidence. The
exact reproducible command is:

```sh
PYTHONPATH=src .venv/bin/python scripts/gen3_switch_differential.py \
  --showdown-root "$POKEZERO_SHOWDOWN_ROOT" \
  --seeds 3 4 5 6 7 8 9 10 --only partialtrapsubstitute
```

On a landed setup, the measured Showdown protocol boundary was:

```text
|move|p2a: Blissey|Substitute|p2a: Blissey
|-start|p2a: Blissey|Substitute
|-end|p2a: Blissey|Fire Spin|[partiallytrapped]|[silent]
|-damage|p2a: Blissey|435/651
|upkeep
```

The fixture now requires the silent release marker, which distinguishes this
Substitute lifecycle from ordinary partial-trap expiry.

## Native Reproduction

Before this patch, the direct native fixture produced:

```text
Damage SideTwo: 25
ChangeSubstituteHealth SideTwo: 25
ApplyVolatileStatus SideTwo: SUBSTITUTE
Damage SideTwo: 6
```

The final instruction is the incorrect residual partial-trap chip. The narrow
patch removes `PARTIALLYTRAPPED` only after Substitute has successfully paid its
cost and been applied. It adds no generic volatile clearing.

## Regression Boundary

The native fixture verifies all of the following:

- successful Substitute removes the active target-side partial trap before
  residual processing;
- a Substitute attempt at exactly one-quarter HP fails and leaves the partial
  trap and its chip in place;
- an ordinary trapped turn keeps the partial trap and its chip; and
- the independent source-departure switch-out and Baton Pass controls in the
  companion trap/perish fidelity suite still cover their separate lifecycle.

The vendored patch stack applies from the pinned upstream source with strict
`git apply` / `patch --fuzz=0` handling. No upstream fixture refresh was needed
for this one-file lifecycle patch.
