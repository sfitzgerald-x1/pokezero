# C156 — closing #1205: the workflow guard scan's four blind spots

**Subject.** `EveryWorkflowTestCountGuardMatchesItsModuleTests._guards()` in
`tests/test_unreachable_readjudication.py` re-derives every `Ran N tests` guard in
`.github/workflows/engine-fidelity-gates.yml` from its module's AST. At `dbb40c5c` it resolved
**22** of the file's **26** executable `python -m unittest` invocations and said nothing at all
about the other four. This closes that.

**Result, taken from the merge ref and not from the branch — see §5:**
**26 executable / 26 resolved / 0 unresolved.**

---

## 1. Per-site diagnosis

I did not assume the four shared a cause. They do, and the shared cause is the only one: for each,
the `Ran N tests` guard **exists, is correct, and sits outside the scan's fixed twelve-line
lookahead**, pushed down by an explanatory comment block between the invocation and the guard. The
scan's window was `range(index, min(index + 12, len(lines)))`, i.e. it reached a gap of at most 11
lines; a site with no guard in that window produced **no output row at all** — not an error, not a
skip, absent.

Measured at `dbb40c5c` by walking the file:

| step | invocation | guard | gap | stated | AST-derived | why the scan missed it |
|---|---|---|---|---|---|---|
| Denominator rule | `:620` | `:632` | **12** | 31 | 31 | the boundary case — exactly one line past the window |
| Spread-gate provenance pin | `:659` | `:675` | 16 | 6 | 6 | a 15-line "READ THIS BEFORE TRUSTING A GREEN" comment |
| Engine stat attestation | `:985` | `:1015` | 30 | 16 | 16 | a comment plus three intervening skip-shape assertions |
| Seed registry coverage | `:1596` | `:1619` | 23 | 41 | 41 | a 21-line comment recording 27 → 30 → 41 |

Three things the diagnosis rules out, each named because the brief asked whether the four differed:

* **Not a missing guard.** All four steps carry an exact, correct guard. `grep -nE 'Ran [0-9]+
  tests'` over the workflow returns 33 lines, of which 7 are comments and **26 are executable** —
  one per executable invocation, a 1:1 pairing with no orphan on either side.
* **Not `load_tests`, and not an unreadable module.** `_methods()` reads all four by AST and returns
  31 / 6 / 16 / 41, matching every guard. No module in scope defines `load_tests` in either form.
* **The two-module step is a real complication and is not the cause.** The Denominator rule step at
  `:620` is one site invoking `tests.test_differential_denominator` **and**
  `tests.test_denominator_adoption` across a backslash continuation, and its guard is the pair's
  sum. The old scan's separate 7-line *target* window already picked up both modules correctly; it
  was the *guard* window that failed. Fixing only the two-module shape would have fixed nothing.

**What the invariance should have suggested.** `reports/c155_terminal_disposition_register.md`
recorded that the count churned (24/20 → 25/21 → 26/22) while the unresolved *set* stayed exactly
these four across four different trees. An invariant set under a churning count is the signature of
a single structural cause, not four coincidences — which is what it turned out to be.

---

## 2. What changed

**`tests/test_unreachable_readjudication.py`**

1. **Step extent is derived, not guessed.** New `_run_bodies()` returns the line span of every
   `run:` shell body from YAML indentation (both the block-scalar and the inline form). New
   `_sites()` returns **one entry per executable invocation**, `(line, targets, guard_line,
   stated)`, with `guard_line=None` for an unresolved one. A guard belongs to the last executable
   invocation above it *within the same `run:` body*. `_guards()` is now a filter over `_sites()`,
   so its return shape and its two existing callers are unchanged.
   * Comment lines are excluded on **both** sides. The file carries the invocation string inside a
     invocation-carrying comment (exactly one, its line derived by the pin and deliberately not
  typed here) and `Ran N tests` inside seven more; counting either has
     already shipped as an arithmetic error once (c155 §6).
   * Targets come from the whole shell command, followed across `\` continuations, which is what
     makes the two-module denominator step correct by construction rather than by a window width
     that happens to reach.
2. **Coverage is asserted** — `test_no_executable_unittest_invocation_escapes_the_scan` requires the
   unresolved set to be empty. This is #1205's actual subject.
3. **Anti-vacuity for it** — `test_the_scan_sees_every_invocation_a_flat_scan_sees` cross-checks the
   structured walk against a flat, structure-free pass over the file. "Zero unresolved" is true and
   worthless for a scan that sees nothing, and that failure mode is the same class as the defect.
4. **The floor is derived** (#1205 fix 2). `assertGreaterEqual(len(guards), 20)` had zero margin at
   a moment when the scan returned exactly 20 — a floor equal to its own subject. It is replaced by
   an equality between the guard lines the scan **resolved** and the executable `Ran N tests` lines
   the file **writes**, computed by a scan sharing no code with `_sites()`.
5. **`load_tests = <callable>`** (#1205's review residual) now raises alongside `def load_tests`.
   "No module in scope uses it today" is a fact about today.
6. **`derived == printed` rests on THREE assumptions; two are pinned and the third is named.**
   `test_every_scanned_module_matches_the_ast_derivations_assumptions` asserts, across all 26
   scanned modules, that **(a)** no class inherits test methods from a base declared in the same
   module and **(b)** no class carrying `test*` methods fails to name a `TestCase` base. #1205's
   review verified both by hand; a hand verification does not survive the next module. **(c)** is
   `setUpClass` — see §4, where it is disclosed rather than claimed.
7. **The battery size is derived** — `test_the_stated_battery_size_is_the_enumerated_lists_length`
   reads the enumerated list's own length and requires the docstring header, the workflow comment
   **and this report** to state it. All three are now in the loop.
8. **The invocation is matched by regex, not by literal.** `python3 -m unittest` and
   `python -munittest` are both spellings `unittest` honours; `"python -m unittest" in line` sees
   neither. Measured at `dbb40c5c`: respelling one step's invocation to `python3 -munittest` drops
   the old scan from 22 resolved guards to **21**, with no error — #1205's shape from a second
   cause. `re.compile(r"python3?\s+-m\s*unittest")` closes it.

**`.github/workflows/engine-fidelity-gates.yml`** — the C154 step's guard `Ran 34` → `Ran 38`
(bounded replace on that step only; an unbounded one is what caused #1204's round-two defect), the
battery comment 23 → 35 (derived, see change 7 above), and a comment recording the blindness
this closes.

**`tests/test_terminal_disposition_register.py`** — the coverage triple now takes **both halves**
from `_sites()`. It previously imported `resolved` from C154 and then re-implemented the pairing
rule locally as `0 <= g - number <= 12` — the second copy its own docstring forbids, and a copy of
the exact window being replaced. Left alone it would have reported four unresolved sites against a
scan that resolves all 26. The expectation is `[]` rather than deleted: an empty set that must stay
empty still reddens on the event that mattered.

**`reports/c155_terminal_disposition_register.md`**, **`reports/c138_known_gaps_ledger.md`** — the
#1205 record updated to its closure. The register's self-referential numbers are pinned by
`tests/test_terminal_disposition_register.py`; moving `resolves **22**` → `resolves **26**` and
`leaves **four**` → `leaves **none**` was driven by those pins going red first.

---

## 3. The mutation battery

Every mutation was applied to the tree, run with `PYTHONDONTWRITEBYTECODE=1`, and reverted.
**M24–M27 were run against `dbb40c5c` first** — a mutation that is red on the fixed tree proves
nothing about the broken one, and the whole claim here is that coverage was *absent*.

⚠ **The numbering below is the module docstring's enumerated list, which
`test_the_stated_battery_size_is_the_enumerated_lists_length` derives.** A first revision of this
table used its own numbering, which could not be reconciled with the pinned one — review could not
map the two. Entries **1–23** are #1204's and its review's and are not restated here; **24–34** are
C156's. Rows with no number are **controls**, not battery entries: they exercise coverage that
already existed or that lives in another module's battery.

Every row is an observed run, not a prediction. `n/a` means the mutation has no meaning at
`dbb40c5c` because it edits something C156 introduced.

| # | mutation | at `dbb40c5c` | on this branch |
|---|---|---|---|
| 24 | seed-registry guard `Ran 41` → `Ran 99` | **GREEN — blind** | RED |
| 25 | spread-gate guard `Ran 6` → `Ran 99` | **GREEN — blind** | RED |
| 26 | stat-attestation guard `Ran 16` → `Ran 99` | **GREEN — blind** | RED |
| 27 | denominator-pair guard `Ran 31` → `Ran 99` | **GREEN — blind** | RED |
| 28 | a new executable step with **no** count guard, placed above the seed-registry step | **GREEN — blind** | RED |
| 29 | `_run_bodies()` stubbed to return no bodies at all | n/a | RED |
| 30 | `load_tests = <callable>` (assignment form) at a **resolved-at-base** site ‡ | GREEN — uncovered | RED |
| 31 | a subclass of a locally-declared base, same **resolved-at-base** site ‡ | GREEN — uncovered | RED |
| 32 | a guard's `if`-block deleted and restated as a comment outside the `run:` body | **GREEN — blind** | RED |
| 33 | the battery total in the workflow comment left at its old value | GREEN — unpinned | RED |
| 34 | a guard weakened to the **floor** shape `Ran [0-9]+ tests` | **GREEN — blind** | RED |
| 35 | a workflow line number re-typed into this module's prose at its correct value — **seven shapes**, §3.1 | GREEN — uncovered | RED |
| 23 | a test added to this module with its own workflow guard left stale | RED | RED |
| 22 | final-holdout guard `Ran 25` → `Ran 24` | RED | RED |
| — | entry 28's step placed directly above a **guarded** one | RED, wrong reason † | RED |
| — | entry 28, run against the register's coverage triple instead | RED | RED |
| — | `def load_tests(...)` — pre-existing coverage, re-verified | RED | RED |
| — | `class MutantNotATestCase(object)` with a `test*` method (review's) | GREEN | RED |
| — | register prose `resolves **26**` → `**22**` (c155's module) | n/a | RED |
| — | register prose `leaves **none**` → `**four**` (c155's module) | n/a | RED |
| NC1 | **control** — 40 comment lines between an invocation and its guard | GREEN | GREEN |
| NC1b | **control** — NC1's padding **plus** that same guard falsified to 99 | **GREEN — blind** | RED |

**Battery: 35 applied, 35 caught, plus 2 controls.** The module's enumerated list, its docstring
header, the workflow comment and this report all state that number, and
`test_the_stated_battery_size_is_the_enumerated_lists_length` derives it from the list — entry 33 is
that pin. The two register-prose rows belong to `tests/test_terminal_disposition_register.py` and
are **not** added to c155's own battery, whose A/B blocks are numbered consecutively and pinned at
three sites; renumbering them for two entries would churn more than it records.

‡ **The site matters and a first revision did not say so.** Entries 30 and 31 are applied to
`tests/test_drag_limit_is_a_last_resort.py`, whose step (`:1112` invocation, `:1113` guard, gap 1)
was **resolved** at `dbb40c5c`. Their green at base is therefore evidence that the *check* was
absent, not that the *site* was blind — the objection review raised, and the reason the site is now
named rather than left to be looked up.

† **Entry 28's placement variant is red at `dbb40c5c` for the wrong reason, and is recorded as a
false positive.** The old twelve-line window ran past the end of the unguarded step and picked up
the *next* step's `Ran 128 tests`, reporting a mismatch against the wrong module. Entry 28 proper is
the same mutation placed where nothing is in reach, and there `dbb40c5c` is green. The `run:`-body
rule cannot misattribute: a guard belongs to an invocation in its own body or to none.

**Entry 34 is a capability gain the first revision did not claim.** A guard weakened from an exact
count to `Ran [0-9]+ tests` matches nothing in the scan's `Ran (\d+) tests` pattern, so the site
becomes *unresolved* and reddens — where at base it simply left the scan. That is exactly the shape
`fleet-worker.yml:45` uses today (§4), so the follow-up is cheaper than it looks: the scan already
detects it.

**NC1 alone proves nothing, which is why NC1b exists.** NC1 is the *edit that created all four blind
spots* — a comment block growing between an invocation and its guard — and it is green on **both**
trees. At `dbb40c5c` it is green by going blind; here it is green by resolving the site. Identical
verdict, opposite reason. NC1b separates them by falsifying the padded step's guard as well: still
green at `dbb40c5c`, red here. A control unpaired with a red is indistinguishable from an inert pin,
which is `reports/c154` §6's own finding applied to this pass.

### 3.1 Entry 35 — the guard that shipped covering nothing

⚠ **Review's finding C went through three fixes and the middle two were worse than the defect.**
Recorded in full because the pattern is this PR's own subject and this is its third recurrence in
one lineage (C154's bullet → #1205 → here).

1. **The defect.** Four citations of the invocation-carrying comment, typed as `:1202`, went stale
   **inside the commit that moved it** — C156's own eleven-line workflow comment.
2. **Fix 1 pinned the typed citation** to the computed line. It reddened on any edit above it and
   turned the negative control NC1 red. Rejected on that measurement; review reproduced the
   reddening and withdrew its own suggestion.
3. ⚠ **Fix 2 forbade a PHRASE — and covered nothing, and shipped.** The phrase
   `invocation-carrying comment at :NNNN` is a spelling *this pass invented*.
   `grep -rn 'invocation-carrying comment at' tests/ reports/` returns **exactly one line: the regex
   literal itself.** None of the four real citations was written that way. Review restored two of
   them verbatim and the module reported **`Ran 38 tests … OK`**. A guard verified only against a
   string its author chose is not verified — it is the fail-open this PR exists to remove, shipped
   inside the docstring advertising it as the closure.
4. **Fix 3 is scoped by VALUE.** Every workflow line this module could cite — the comment, each
   executable invocation, each guard — is computed, and the module's own text may not contain any of
   them as `:NNNN`. A phrase is evaded by rewording; a value cannot be, because the value *is* what
   a citation is.

**Proved against the real strings, not a chosen one.** Each row restores a spelling that actually
existed at `b167c3d1`, typed at the value correct on this tree (written `:NNNN` here because the guard
forbids the report from carrying that value, which is itself the guard working):

| shape | text restored | verdict |
|---|---|---|
| H1 | `…carries the invocation string inside one at :NNNN and` (was module `:1272`) | **RED** |
| H2 | `#: NOT sites; :NNNN is one and counting it into the` (was module `:1083`) | **RED** |
| H3 | `a comment at :NNNN and \`Ran N tests\` inside seven more` (was module `:1076`) | **RED** |
| H4 | ``comment at `:NNNN` and`` in the report (was report `:63`) | **RED** |
| H5 | the denominator-step residual review named, re-typed at its correct value | **RED** |
| H6 | a `Ran N tests` **guard** line re-typed at its correct value | **RED** |
| H7 | the contact-ability **invocation** line re-typed at its correct value | **RED** |
| **NC1** | 40 comment lines inserted between an invocation and its guard | **GREEN** |

NC1 green is the criterion fix 1 failed, so fix 3 satisfies both constraints at once: it fires on
every real citation shape and it does not fire on unrelated line motion.

⚠ **What it does NOT catch, stated because a negative claim is only as wide as its check.** It
catches a citation typed at its **correct** value — which is exactly when this defect is born, since
C156 typed `:1202` while the comment *was* at 1202 — and it is blind to one that has **already**
drifted, because a stale number equals no computed value. That is not hypothetical: `:469` for the
contact-ability step had been wrong since #1204 (the invocation is at `:482`, at `dbb40c5c` and at this head) and nothing found it;
review found it by hand while reading this pass. Both live citations in the module are de-numbered
rather than left for the next reader. The report's §1 table keeps its line numbers on purpose,
because it scopes them to `dbb40c5c` and a citation scoped to a commit cannot go stale.

---

## 4. What is not closed

* **A guard is still only as good as the run it grades.** This pin asserts that every step's stated
  count equals its module's AST method count. It does not and cannot assert that the step *ran* —
  `^OK$` and skip-set assertions do that, per step, and four steps in this file are shape-only in CI
  by design and say so.
* ⚠ **A THIRD assumption behind `derived == printed`, named because review found the claim of two
  wrong, with a live counterexample in this tree.** A class that skips at `setUpClass` contributes
  **zero** to `testsRun`, so the printed total drops below the AST count. On the merge ref,
  `python -m unittest tests.test_spread_gate_provenance` prints **`Ran 1 test … OK (skipped=2)`**
  against an AST-derived **6** — in one of the four modules this PR just brought into coverage.
  It is not a defect and it is not a hole: CI has the Showdown dependency those classes gate on, so
  the step prints `Ran 6` there, and a class-level skip appearing in CI would print a *smaller*
  number and redden its exact-count guard. **It fails closed.** What was wrong was only the count of
  assumptions, and the pin's docstring now names three. (c) is not pinnable from a local scan,
  because the printed count under it is a property of the CI environment.
* **Scope of the negatives above.** "No module defines `load_tests`" is
  `ast.parse` over the 26 modules `_sites()` names, nothing wider. "26 executable invocations" is
  lines matching `re.compile(r"python3?\s+-m\s*unittest")` in
  `.github/workflows/engine-fidelity-gates.yml` whose stripped form does not start with `#` — this
  file only. **The other two workflows are not scanned and both have sites this scan would flag:**
  `neural-smoke.yml:29` runs `python -m unittest tests.test_neural_policy` with no count guard at
  all, and `fleet-worker.yml:35` runs six modules on one command with no count guard, while
  `:45` guards a different step with `Ran [0-9]+ tests` — a floor, not a count, so it cannot
  detect a shrinking suite. Measured by `grep -n 'unittest\|Ran ' .github/workflows/*.yml`.
  Widening the scan across workflows is **filed, not done here**: it is a change to two files this
  PR otherwise does not touch, and each site needs a count derived and reviewed on its own.
* **The pairing rule assumes one guard per invocation.** Where a `run:` body holds several
  invocations, the first guard below each is taken; guards *trailing* the last invocation beyond the
  first are not paired. No step in this file has more than one invocation, so that branch is
  exercised by construction only, not by the tree. Review probed the two-invocations-two-trailing-
  guards shape and it **reddens** — the second guard is unclaimed, so the resolved set stops
  matching the file's written guard lines. Fails closed, and scoped to that: it is detected, not
  correctly attributed.
* **Eight ways of writing a `run:` body were probed by review and all eight fail closed** — block
  scalars with chomping indicators, heredocs, inline `run:`, a last step running to EOF, and a
  `- run: |` step with no `name:` key. "Fail closed" here means the site is reported unresolved or
  the flat-scan cross-check disagrees, never that it is silently dropped.

---

## 5. Re-derived triple, from the merge ref

`executable / resolved / unresolved`, derived by the committed scan, not typed:

| tree | commit | scan | executable | resolved | unresolved |
|---|---|---|---|---|---|
| `origin/main` | `dbb40c5c` | pre-C156 | 26 | 22 | **4** |
| merge-base | `dbb40c5c` — identical to main, branch cut from it | pre-C156 | 26 | 22 | **4** |
| this branch head | `3b3b82c5` | C156 | 26 | **26** | **0** |
| **`refs/pull/1208/merge`** | **`f07f7a6b`** | C156 | **26** | **26** | **0** |
| **`refs/pull/1208/merge`**, re-derived after the next push | **`d4b309e4`** | C156 | **26** | **26** | **0** |
| **`refs/pull/1208/merge`**, re-derived after the review round | **`a9ecb72f`** | C156 | **26** | **26** | **0** |
| **`refs/pull/1208/merge`**, re-derived after finding C's third fix | **`8326e877`** | C156 | **26** | **26** | **0** |

⚠ **The merge ref moves with every push, so the SHA above is the one measured and not "the" merge
ref.** Each row is a run, not a figure carried forward from the row above it; `reports/c153`'s rule
is that a number can be correct when taken and false later because its tree was replaced. The PR
body carries the derivation at the final head.

**The merge row is the one CI grades and the only one that settles the question.** `reports/c155`
§6 records the round where a branch-derived count shipped wrong because `main` had moved underneath
it — 25 on each tree in isolation, 26 on the merge. Reproduce:

```
git fetch origin 'refs/pull/1208/merge:refs/mergerefs/1208' -f
git worktree add /tmp/pz-merge1208 --detach refs/mergerefs/1208
cd /tmp/pz-merge1208 && PYTHONDONTWRITEBYTECODE=1 python3 - <<'EOF'
import sys; sys.path.insert(0, 'tests')
from test_unreachable_readjudication import EveryWorkflowTestCountGuardMatchesItsModuleTests as S
lines = S._lines()
carrying = [n for n, l in enumerate(lines, 1) if 'python -m unittest' in l]
comments = [n for n in carrying if lines[n - 1].strip().startswith('#')]
sites = S._sites()
print(len(carrying), len(comments), len(carrying) - len(comments),
      sum(s[2] is not None for s in sites), sum(s[2] is None for s in sites))
EOF
```

Output at `f07f7a6b`: `27 1 26 26 0` — 27 lines carry the invocation, 1 is a comment, **26
executable, 26 resolved, 0 unresolved.** `tests.test_unreachable_readjudication` on that ref: **Ran
38 tests, OK**; `tests.test_terminal_disposition_register`: **Ran 42 tests, OK**.
