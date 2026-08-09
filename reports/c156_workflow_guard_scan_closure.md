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
     comment at `:1202` and `Ran N tests` inside comments at seven more places; counting either has
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
6. **The AST derivation's two unstated assumptions are pinned** —
   `test_every_scanned_module_matches_the_ast_derivations_assumptions` asserts, across all 26
   scanned modules, that no class inherits test methods from a base declared in the same module and
   no non-`TestCase` class carries `test*` methods. #1205's review verified both by hand; a hand
   verification does not survive the next module.
7. **The battery size is derived** — `test_the_stated_battery_size_is_the_enumerated_lists_length`
   reads the enumerated list's own length and requires the docstring header **and** the workflow
   comment to state it. Both said 23 and nothing checked either; adding ten entries would have left
   both stale.

**`.github/workflows/engine-fidelity-gates.yml`** — the C154 step's guard `Ran 34` → `Ran 38`
(bounded replace on that step only; an unbounded one is what caused #1204's round-two defect), the
battery comment 23 → 31, and a comment recording the blindness this closes.

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

Every row below is an observed run, not a prediction. `n/a` means the mutation has no meaning
at `dbb40c5c` because it edits something C156 introduced.

| # | mutation | at `dbb40c5c` | on this branch |
|---|---|---|---|
| M24 | seed-registry guard `Ran 41` → `Ran 99` | **GREEN — blind** | RED |
| M25 | spread-gate guard `Ran 6` → `Ran 99` | **GREEN — blind** | RED |
| M26 | stat-attestation guard `Ran 16` → `Ran 99` | **GREEN — blind** | RED |
| M27 | denominator-pair guard `Ran 31` → `Ran 99` | **GREEN — blind** | RED |
| M28 | a new executable step with **no** count guard, placed above the seed-registry step | **GREEN — blind** | RED |
| M28v | the same step placed directly above a **guarded** one | RED, wrong reason † | RED |
| M28b | M28, run against the register's coverage triple instead | RED | RED |
| M29 | `_run_bodies()` stubbed to return no bodies at all | n/a | RED |
| M30 | `load_tests = <callable>` added to a scanned module (assignment form) | GREEN — uncovered | RED |
| M31 | `def load_tests(...)` added to a scanned module | RED | RED |
| M32 | a subclass of a locally-declared base, inheriting its test methods | GREEN — uncovered | RED |
| M33 | a guard's `if`-block deleted and restated as a comment outside the `run:` body | **GREEN — blind** | RED |
| M34 | a test added to this module with its own workflow guard left stale (battery 23) | RED | RED |
| M35 | final-holdout guard `Ran 25` → `Ran 24` (battery 22) | RED | RED |
| M36 | the battery total in the workflow comment left at its old value | GREEN — unpinned | RED |
| M37 | register prose `resolves **26**` set back to `**22**` | n/a | RED |
| M38 | register prose `leaves **none**` set back to `**four**` | n/a | RED |
| NC1 | **control** — 40 comment lines between an invocation and its guard | GREEN | GREEN |
| NC1b | **control** — NC1's padding **plus** that same guard falsified to 99 | **GREEN — blind** | RED |

**Battery: 33 applied, 33 caught, plus 2 controls.** The module's enumerated list, its docstring
header and the workflow comment now all state that number and
`test_the_stated_battery_size_is_the_enumerated_lists_length` derives it from the list — M36 is that
pin. M37 and M38 are `tests/test_terminal_disposition_register.py`'s and are **not** added to c155's
own battery, whose A/B blocks are numbered consecutively and pinned at three sites; renumbering them
for two entries would churn more than it records. They are recorded here instead.

† **M28v's red at `dbb40c5c` is a false positive and is recorded as one.** The old twelve-line window
ran past the end of the unguarded step and picked up the *next* step's `Ran 128 tests`, reporting a
mismatch against the wrong module. That is misattribution, not coverage — M28 is the same mutation
with the new step placed where nothing is in reach, and there `dbb40c5c` is green. The `run:`-body
rule cannot misattribute: a guard belongs to an invocation in its own body or to none.

**NC1 alone proves nothing, which is why NC1b exists.** NC1 is the *edit that created all four blind
spots* — a comment block growing between an invocation and its guard — and it is green on **both**
trees. At `dbb40c5c` it is green by going blind; here it is green by resolving the site. Identical
verdict, opposite reason. NC1b separates them by falsifying the padded step's guard as well: still
green at `dbb40c5c`, red here. A control unpaired with a red is indistinguishable from an inert pin,
which is `reports/c154` §6's own finding applied to this pass.

---

## 4. What is not closed

* **A guard is still only as good as the run it grades.** This pin asserts that every step's stated
  count equals its module's AST method count. It does not and cannot assert that the step *ran* —
  `^OK$` and skip-set assertions do that, per step, and four steps in this file are shape-only in CI
  by design and say so.
* **Scope of the negatives above.** "No module defines `load_tests`" is
  `ast.parse` over the 26 modules `_sites()` names, nothing wider. "26 executable invocations" is
  lines containing `python -m unittest` in
  `.github/workflows/engine-fidelity-gates.yml` whose stripped form does not start with `#` — this
  file only. **The other two workflows are not scanned and both have sites this scan would flag:**
  `neural-smoke.yml:29` runs `python -m unittest tests.test_neural_policy` with no count guard at
  all, and `fleet-worker.yml:35` runs six modules on one command with no count guard, while
  `:45` guards a different step with `Ran [0-9]+ tests` — a floor, not a count, so it cannot
  detect a shrinking suite. Measured by `grep -n 'unittest\|Ran ' .github/workflows/*.yml`.
  Widening the scan across workflows is **filed, not done here**: it is a change to two files this
  PR otherwise does not touch, and each site needs a count derived and reviewed on its own.
* **The pairing rule assumes one guard per invocation.** Where a `run:` body holds several
  invocations, the first guard below each is taken. No step in this file has more than one
  invocation, so that branch is exercised by construction only, not by the tree.

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
