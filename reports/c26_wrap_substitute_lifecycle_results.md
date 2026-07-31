# C26 Result: Substitute Clears Active Partial Trapping

## Ground Truth

The Showdown-backed `partialtrapsubstitute` fixture ran eight fixed seeds. Fire
Spin landed on four; all four exercised the measured boundary and passed:

1. Ninetales successfully used Fire Spin on Blissey.
2. On the next boundary Blissey successfully used Substitute.
3. Showdown emitted `-end ... Fire Spin [partiallytrapped] [silent]` before
   upkeep and emitted no Fire Spin residual damage on that boundary.

Missed Fire Spin setup seeds were skipped rather than treated as evidence.

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
- existing switch-out and Baton Pass controls continue to cover the independent
  source-departure lifecycle.

The vendored patch stack applies from the pinned upstream source with strict
`git apply` / `patch --fuzz=0` handling. No upstream fixture refresh was needed
for this one-file lifecycle patch.
