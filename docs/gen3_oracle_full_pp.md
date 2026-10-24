# Explicit Gen3 full-PP oracle profile

Legacy search intentionally stops recording ordinary PP deductions at 10 PP.
That is not a valid full-successor-state oracle. The compile-time
`oracle-full-pp` feature charges every ordinary caller, clamps Pressure at the
available PP, and charges Sleep Talk's caller before its recursive effect.
Gen3's `useMoveInner` override does not charge Pressure on a called move.

This profile is off by default. It does not certify other mechanics, change
damage enumeration, or alter historical installed runtimes or measurements.
Use `POKEZERO_ENUMERATE_ROLLS=1` before the first native call when enumerating
the damage law; that runtime setting is separate from the PP build feature.

The search crate selects the profile with `--features oracle-full-pp`.
For the Python wheel, use the ordinary verified source acquisition and ordered
patch applicator, then build the extracted sdist root with:

```sh
uv pip install --python "$oracle_python" --no-cache "$patched_source" \
  --config-settings="build-args=--features poke-engine/gen3,poke-engine/oracle-full-pp --no-default-features"
```

An oracle entry must bind the actual binary and behaviorally check high PP,
last-PP Pressure, and asleep Sleep Talk before accepting a panel. A feature
name or unchanged package version is not binary provenance.

Focused regression suites compare both feature-on and legacy behavior. The
authored `tests/gen3_oracle_full_pp_showdown.cjs` control runs against the
immutable pinned Showdown checkout; it is not a scientific state panel.
