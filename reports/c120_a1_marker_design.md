# C120 v2 — A1 is two shapes, not one, and v1's keying fact was false

> **v1 claimed Showdown runs its residual block on the faint boundary and the engine on the
> replacement, so a "residuals already ran" marker could key on that. The fixture v1 itself
> cites says the opposite. And v1's "all five rows have this property" is false for two of
> five. Both corrections below; v1 should not be cited.**

## 0. The two corrections, measured

**(a) Showdown defers too.** `scripts/gen3_switch_differential.py` is the repo's Showdown
ground-truth gate, and its `faintresiduals` scenario is measured *on the faint ply* with
`expect={"residual_block_ran": False, "survivor_healed": False}` — the comment explains that
`runAction` sees the pending switch flag, issues a switch request, and returns with the queued
residual action untouched, so the protocol block ends at `|faint|`. Its sibling
`faintresidualsdeferred` is measured on the *replacement* ply with
`residual_block_ran: True`, `replacement_took_sandstorm: True`, `switch_precedes_residuals: True`.

**So Showdown defers in THAT SCENARIO.** ~~So Showdown defers exactly like the engine.~~

> **Correction (review of #1087). The generalisation was wrong, and I made it by reading a
> fixture and not opening the artifacts.** `faintresidualsdeferred` establishes its own
> configuration and nothing wider. Placement is *conditional*, which the vendored
> `poke-engine-gen3-residual-defer-on-faint.patch:19-20` states in terms: "a faint caused BY the
> residual block, which has already run, never gets the flag and so is never mistaken for a
> pending one." The fixture exercises only the faint-from-a-move arm.
>
> Checked against the retained repros for the three rows this report KEEPS in A1 —
> `19000059/27`, `19000020/50`, `19100181/45` — all three record **`observed_only=[]`**:
> Showdown emitted no residual component at that boundary at all. If Showdown had run the whole
> block on the replacement ply the way the fixture expects, those boundaries would MATCH. They
> diverge. So the recorded evidence says Showdown did **not** defer on these rows, and v1's
> description matched the artifacts where this revision did not.
>
> I consulted the artifacts at §(b) below for the two rows where they helped, and not for the
> three where they refute the headline. That is precisely the failure this document was written
> to correct in v1, repeated one level up.
>
> The keying claim inverts with it: the tick is demonstrably *not* on the divergent boundary and
> `|turn|N` advanced, so it landed earlier — the observable v1 assumed is the one the data
> supports. v1's §4 disposition therefore stands unrefuted by this document, and is not
> re-opened here.

**(b) A1 is two shapes.** Deserialising each row's recorded `engine_state`:

| row | `s1.force_switch` | active changed | observed `fainted` | class sign |
|---|---|---|---|---|
| `19000020/50` | **True** | both | `[]` | **extra** in engine |
| `19000059/27` | **True** | both | `[]` | **extra** |
| `19100181/45` | **True** | both | `[]` | **extra** |
| `19100002/53` | False | neither | `['p1']` | **missing** |
| `19100154/75` | False | neither | `['p2']` | **missing** |

The last two carry **no forced-switch marker**, nobody has switched, and a side is fainted in
the observed post-state: they **are** the faint boundary, not the replacement. Their class sign
is **opposite** — the engine is *missing* a tick Showdown has. A marker of the form "do not
expect residuals here" cannot address a missing component.

So v1's "all five rows carry a forced-switch marker" is wrong (3 of 5), and its "in every case
the boundary carrying Showdown's tick is, by construction, one that compared clean" is refuted
for those two — Showdown's tick is at the divergent boundary itself and **is** in the artifact.
The design covers at most **3** rows.

> **Correction (review of #1087).** An earlier revision ended that sentence "…which is why plan
> item 9 says '2 rows'". Three errors in one clause: 3 does not explain 2; the two pairs are
> **disjoint** (item 9's pair is the dev pair recorded at `reports/c111_residue_row_causes.md:55`
> — `19000020/50`, `19000059/27` — while the pair this report ejects is the holdout pair
> `19100002/53`, `19100154/75`); and the citation is to a plan document that is **not in this
> repository**, so no reader can check it. Struck.
>
> **The ejected pair has since been settled, and the ejection was right.** Both rows were the
> *battle-end weather* cause — Showdown chips the winner at the final boundary and the engine
> did not — and #1092 fixed it, closing `19100002/53` and `19100154/75` with nothing opened.
> They were never A1.

**(c) v1's cost ordering was wrong.** v1 listed a re-sweep with adjacent-boundary retention as
the cheapest route. It is third-cheapest: `python scripts/gen3_switch_differential.py --only
faintresiduals faintresidualsdeferred faintresidualscontrol` drives the real Node sim and
measures which ply carries the block, and the fixture's recorded `expect` answers it at **zero**
cost. v1's framing — "not a gap in what is knowable, a gap in what is *retained*" — is wrong
about this fact: it was neither missing nor unretained, it was already in the tree with the
opposite answer.

## 1. What survives from v1, verified

- **The sweep-retention claim is correct.** `engine_transition_differential.py:2264` appends to
  `repros` only when `verdict == "diverged"`, so matched boundaries retain nothing but
  histogram counters, and `replay_residue.py` cannot reach a non-retained step. That is true —
  it is just not the binding constraint, because the fact in question lives in a fixture.
- `19100002/53` has `branch_count: 1`, which is what ruled out A4/A7 for it.
- v1's §3 self-critique — that writing the marker from an assumption would let two rows close
  without establishing the assumption — was right, and is exactly what caught this. It just did
  not go far enough: I applied it to the marker and not to my own §1.

## 2. Disposition

**A1 splits.** Three rows (`19000020/50`, `19000059/27`, `19100181/45`) are replacement
boundaries carrying an *extra* engine component, and are the marker's scope. Two
(`19100002/53`, `19100154/75`) are faint boundaries *missing* a component Showdown has, need a
different cause, and should not be filed A1.

**Next action:** run the three-scenario command against the live sim to confirm the fixture's
recorded ply placement still holds, then scope the marker to the three-row group and re-argue
whether it is a harness or engine fix on the measured placement rather than the assumed one.

## 3. Why v1 was wrong

v1 cited a fixture for the half that suited its argument and did not read the half that
refuted it. The fixture is 13 lines and names both plies explicitly. This is the same failure as
C118 v1 — verifying that a citation resolves rather than that it says what you need — and it is
now the second time in this program that a report has been corrected for exactly it.

---

*v1 follows, retained because the corrections above are only legible against it.*

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
