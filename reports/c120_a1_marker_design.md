# C120 — what A1's "residuals already ran" marker needs, and why the residue artifacts cannot supply it

C116 Phase 3 item 9 is "the A1 harness marker, 2 rows, with a revert-failing pin". Before
writing it: the marker's design depends on a fact the residue artifacts **do not contain**.
Recording that, because it changes the shape of the work from a matcher edit to a harness
change with a data dependency.

Era: `main` `48468b67`. A1 rows: `19000020/50`, `19000059/27` (dev), `19100002/53`,
`19100154/75`, `19100181/45` (holdout).

## 1. The row shape, re-measured

`19000059/27`, replayed at this era:

```
class=component_extra_in_engine:psn   gating=exact   choices={'p1': 'crawdaunt', 'p2': 'omastar'}
Showdown:  |turn|27          -- and nothing else. Both sides: no components at all.
engine:    ToggleSideOneForceSwitch
           Damage SideOne: 16
           pct=100.00   p1 exact=psn=-16   p2 exact=(none)
```

So this is a **replacement** boundary. The engine runs an end-of-turn phase here; Showdown
does not, because it already ran one at the *faint* boundary. The engine's deferral is
faithful gen3 — `end_of_turn_is_deferred`, verified against the real sim by
`gen3_switch_differential.py::faintresiduals` — so neither side is wrong. The harness is
comparing a boundary against the wrong one.

## 2. The problem with the obvious marker

A marker of the form "this side's residuals already ran this turn, so do not expect them
here" needs to know that Showdown's tick landed on the **preceding** boundary. That is the
observable the marker keys on.

**That boundary is not in the artifact.** The sweep retains repros only for *divergent*
boundaries (`keep_repro`), and step 26 matched — so there is no record of it to check. The
same holds for all five A1 rows: in every case the boundary carrying Showdown's tick is,
by construction, one that compared clean.

This is not a gap in what is knowable — it is a gap in what is *retained*. Three ways to
close it, in increasing cost:

1. **Re-sweep with adjacent-boundary retention** for the A1 seeds only, so the pair is
   visible. Cheapest, and it produces the evidence the pin needs anyway.
2. **Carry the marker in the harness's own turn state** rather than deriving it from
   protocol: when the engine defers, set a flag consumed at the next boundary. This needs no
   new data but is the more invasive change, and it can only be validated against (1).
3. **Widen the comparison to a boundary pair** when a faint is present. Largest blast radius;
   would need its own adjudication against the program's rule that a row may never be closed
   by widening an attribution rule — and this is close enough to that line to require an
   explicit argument, not an assumption.

## 3. Why this is being recorded rather than implemented

Option (2) is the tempting one because it needs no new measurement. It is also the one that
would let me write a marker, watch two rows close, and never establish that the tick was
where I assumed. That is the shape of every error this program has corrected in the last
week: A5's mechanism, C117's three A2 filings, C118's inverted hook — each was a plausible
reading of structure that a cheap measurement contradicted.

So the next action is **(1), a re-sweep of the five A1 seeds with adjacent-boundary
retention**, and the pin should assert the *pair*: Showdown's tick on the faint boundary and
its absence on the replacement, with the engine mirrored. A pin that asserts only "no psn
expected at step 27" would pass for the wrong reason on any boundary that happens to have no
residual.

## 4. What is already established, and needs no re-derivation

- The deferral is faithful gen3 (`gen3_switch_differential.py::faintresiduals`), so **this is
  a harness fix, not an engine fix.** A1 must not become an engine change.
- All five rows carry a forced-switch marker in the engine's instruction stream
  (`ToggleSideOneForceSwitch` on the dev pair; C117 §4 files `19100002/53` and `19100154/75`
  as A1 on a faint-to-sandstorm-tick signature and `19100181/45` as "the engine ran a
  residual phase Showdown did not").
- `19100002/53` has `branch_count: 1`, which is what ruled out A4/A7 for it — there is no fan
  to partition. That also means A1's rows are unaffected by the Phase 2 decision either way,
  so item 9 is genuinely independent of Phase 2 and can proceed in parallel, as the plan says.
