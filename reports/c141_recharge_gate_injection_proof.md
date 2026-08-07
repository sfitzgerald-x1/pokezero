# C141 — the recharge gates, against an actually-injected bad write

**Era / provenance.** `corpus/golden-v4`, 1295 decision rows / 1271 same-seat boundaries,
**12 battles**. Tree: `main` at `9a128bec` (task 4 merged). The engine crate was rebuilt with and
without the injection; the injection is reverted and `self_must_recharge` is confirmed absent
from the final `.so`.

Producing commands. Note the `../../` — `.venv/bin/python` does not resolve after the `cd`, and
an earlier revision of this report printed a path that could not be run:

```sh
export POKEZERO_SHOWDOWN_ROOT=<showdown checkout>
( cd rust/pokezero-search && CARGO_BUILD_JOBS=8 ../../.venv/bin/python -m maturin build \
    --release --skip-auditwheel --out /tmp/wheel -i ../../.venv/bin/python )
uv pip install --python .venv/bin/python --force-reinstall /tmp/wheel/*.whl
.venv/bin/python scripts/leaf_root_parity.py \
    --corpus corpus/golden-v4 --tables corpus/encoder_tables_v4.json
.venv/bin/python scripts/leaf_vs_reality.py \
    --corpus corpus/golden-v4 --tables corpus/encoder_tables_v4.json
```

## Why this exists

Task 4's middle clause is *"show they now catch an injected bad write."* PR #1156 shipped the
gate fix and the symmetry fix, but what it pinned was the **derivation** — fixture tests showing
a write contradicting the parser tracker disagrees with the gate's world. Review said so and
made me soften three "pinned two-way" claims to:

> no test or measurement shows a gate catching an end-to-end bad write

That is the gap this closes. No code change — this is a measurement.

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

Chosen over "always true" / "always false" because those are caught by anything. This one is
correct on most rows and wrong only where the two sides' recharge states differ.

Verified live rather than assumed: `self_must_recharge` present in the compiled `.so` → `True`
with the injection, `False` after the revert; the reverted `.so` hashes equal to the
pre-injection baseline, bit for bit, and the injected build is byte-reproducible across
independent builds.

`877` is the MUSTRECHARGE volatile id — checked in the v4 vocab both directions
(`volatile:mustrecharge -> 877`, `877 -> volatile:mustrecharge`), not pattern-matched from C112.

## Result

`leaf_root_parity`. Denominator is **1006 measured of 1295, 289 skipped** on every row:

| run | exit | diverged | families |
|---|---|---|---|
| clean crate, current gates | **0** | **0** | — |
| **injected**, current gates | **1** | **5** | 5 × `self_team/CATEGORY_VOLATILE_OFFSET` |
| clean crate, pre-task-4 gates | 1 | **1** | 1 × `opponent_team` — **already red before any injection** |
| **injected**, pre-task-4 gates | 1 | 5 | 4 × `self_team` + the same 1 × `opponent_team` |
| revert + rebuild | **0** | **0** | byte-identical to the baseline row |

**The acceptance clause is met.** The current gates go from `diverged 0 / EXIT=0` to
`diverged 5 / EXIT=1` on an injected bad write, at an unchanged denominator, and return to
exactly the baseline when it is reverted. The catch is not a denominator artifact, and it is not
circular: the `want` side is the pre-recorded golden array from production Python
(`corpus/golden-v4/arrays.npz`), the injection lives only in the `got` path, and the world's
`recharging_slots` seed comes from the recorded parser tracker — independent of `leaf.rs`.

**Caught in both directions**, which the harness's single exemplar line hides. Read off the
golden array at `categorical_ids[row][1:7, 33]`:

```
row 952  want [0,0,0,0,0,0]      got=877 want=0   spurious volatile
row 953  want [877,0,0,0,0,0]    got=0 want=877   ERASED a real one
row 956  want [0,0,0,0,0,0]      got=877 want=0
row 957  want [0,0,0,0,0,0]      got=877 want=0
row 958  want [877,0,0,0,0,0]    got=0 want=877   ERASED a real one
```

3 spurious, 2 erased. An earlier revision of this report transcribed the harness's `e.g. row 952
got=877 want=0` as if it characterized all five.

`leaf_vs_reality` **also catches it**: defect class **123 → 125**, `EXIT=1` both before and
after. Its denominator line is unchanged (956+315, diverged 917), and an earlier revision of
this report read that line and wrongly concluded the whole gate was unmoved. The exit *bit* does
not flip because that gate was already red at 123 on a clean crate; the gating quantity is `defect_rows` —
`return 1 if (defect_rows != 0 or matchup_arm_failed or denominator) else 0` in
`scripts/leaf_vs_reality.py` — not the 917. (Cited by expression, not line: this reference has
now been invalidated TWICE by the very commit introducing it, once by the caveat edit and once
by a rewrap two lines away.)

## Limits — read these before quoting the table

**The entire positive signal is one battle of twelve.** All 5 rows carrying any tracker lock are
in `golden-gen3randombattle-1009` (rounds 16–19, two recharge episodes). Drop that game and
`diverged` is 0 and this injection passes silently. This is the most load-bearing limit here and
an earlier revision omitted it.

**The old gates mostly caught it too.** I expected the pre-task-4 gates to *ratify* this write.
They do not — they catch 4 of the 5. The discrimination is real but narrow, and both halves of
it are **one row, not two**: `row 956 = (1009, 18, p1)`. Round 18 has only a `p1` decision row,
so the candidate rule cannot see p2's lock; the old gate's world therefore carries no
MUSTRECHARGE on p2, the injected write (reading `opp_side`) returns `False`, and `False` happens
to equal the recorded `self_must_recharge=False`. The bad write is right by accident because the
world was wrong in a compensating way. That single cause produces simultaneously the ratified
self-side row and the spurious `opponent_team` divergence — an earlier revision listed them as
two independent bullets, which read as two defects.

Because the old gates were already at `diverged 1` on a clean crate, the injection moved them
`1 → 5`, and never moved their **verdict** at all.

**This exercises the SELF side only.** The opponent side has been live since before this work.
(Restored: an earlier revision carried this limit, and the rewrite that added the one-battle
limit dropped it.)

**Catchability depends on the SHAPE of the wrong write, not just on its being wrong.** This
corpus discriminates only because the two sides' recharge states differ on those 5 rows. A
self-side bug that read the self side with a different defect -- a wrong party index, say --
has no guarantee of being caught here at all.

**Why no self-side injection can do better on this corpus.** The recorded action and the parser
tracker disagree on exactly one row of `corpus/golden-v4`, and it is opponent-side. On the self
side they never disagree, so no self-side write can be ratified by the old derivation and caught
by the new one *on the self side*, however it is chosen — the one row of discrimination above
comes from the opponent-side disagreement the write happens to read.
`tests/test_recharge_gate_derivation.py` covers what the corpus cannot; that is why it uses
fixtures.

## Disposition

No code change. C112's **P3 stays open**: `leaf.rs`'s self-side MUSTRECHARGE volatile is still
root-frozen, and this run is evidence that when someone lifts it, the gates will hold the
implementation honest — the property #1156 claimed and could not previously show.
