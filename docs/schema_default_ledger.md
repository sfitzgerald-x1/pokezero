# The schema-default conflation, and the ledger that bounds it

## The defect class

Code and tests that read the global `OBSERVATION_SCHEMA_VERSION` (or `DEFAULT_REPLAY_OBSERVATION_SPEC`)
when they mean *a specific version* or *a schema property*. While the default happens to satisfy
what the site wanted, the conflation is free and invisible. It becomes visible only when the
default moves — and then everything that conflated the two fails at once, with failure messages
that name the wrong subsystem.

It is not a style problem. `TransformerPolicyConfig.token_count` defaulted off the process-wide spec rather than the config's
own stamped schema, so a config stamped one schema silently carried another's `token_count`. That
one is FIXED (`__post_init__` derives it).

Its two sibling **feature widths are still live, on this branch and on main** —
`neural_policy.py:245-246` default off `DEFAULT_REPLAY_OBSERVATION_SPEC` and nothing recomputes
them, so today:

    v2    config=(51,155)  spec=(39,121)
    v2.1  config=(51,155)  spec=(39,140)
    v4    config=(51,155)  spec=(41,132)

An earlier version of this paragraph described both as found *and fixed*, in the past tense. The
widths fix is not in this PR. This is stated rather than smoothed because the paragraph is the
justification the rest of the document rests on.

## N, and the command that produces it

```sh
.venv/bin/python scripts/schema_default_ledger.py            # rows
.venv/bin/python scripts/schema_default_ledger.py --by-file  # per file
.venv/bin/python scripts/schema_default_ledger.py --json     # machine-readable
```

The number is not the artifact — **the command is**. Any figure quoted from memory is worthless;
this effort produced four wrong ones (6, ~4, 97, 94) before the ledger existed, each stated
without an established denominator and two of them from hand-picked file lists.

### Reads are matched in every spelling

A default can be reached as a bare name, an alias, or a module attribute at any depth:

    OBSERVATION_SCHEMA_VERSION                        # bare
    SV                                                # from ... import ... as SV
    O.OBSERVATION_SCHEMA_VERSION                      # import pokezero.observation as O
    pokezero.observation.OBSERVATION_SCHEMA_VERSION   # import pokezero.observation
    pokezero.OBSERVATION_SCHEMA_VERSION               # import pokezero  (the PUBLIC spelling --
                                                      #   this constant is in pokezero.__all__;
                                                      #   DEFAULT_REPLAY_OBSERVATION_SPEC is not)
    observation.OBSERVATION_SCHEMA_VERSION            # from . import observation  (RELATIVE)

Only the first was matched originally. **Each of the others was demonstrated to add default reads
with N unchanged and the gate green** — the same defect class as the any-of bug: *a denominator
blind to a spelling*.

It recurred **four** times, and the reason is the same every time: each fix was verified once by
hand and pinned by no test, so the next edit to the matcher reopened it.

| round | spelling |
|---|---|
| 2 | any-of surface matching |
| 5 | `from ... import GLOBAL as ALIAS` |
| 6 | a dotted base, and bare `import pokezero` (the public spelling) |
| 7 | `from . import <mod>` — **relative**, and live in the package |

Round 7's is the strongest occurrence record of the four: relative imports are already used inside
the package (`src/pokezero/linear_policy.py:24`, `src/pokezero/selfplay.py:17`), whereas dotted reads
of these globals appear **0** times anywhere. Its cause was also the most embarrassing —
`node.module.startswith("pokezero")` is `None` for a relative import, so the branch never fired,
three lines below a comment disclaiming exactly that kind of enumeration-from-memory.

The enumeration now lives in `tests/test_schema_default_ledger.py`:
`LedgerSeesEverySpellingTest` (14 positive spellings, 6 lookalike negatives) and
`SurfaceDerivationSeesEverySpellingTest` (10 declaration spellings, 3 negatives). Over-matching is
tested too, because an inflated denominator is the same defect as a deflated one — and that test
immediately caught a live one: every name imported from a pokezero module was registered as a module
base, so `ObservationSpec.OBSERVATION_SCHEMA_VERSION` scored a row. Module-vs-name is now resolved
against the filesystem rather than guessed from capitalisation.

#### The figures in this section, and the greps that disagree

Three previously-stated numbers were wrong. Corrected, with the command for each:

| claim | stated | derived | command |
|---|---|---|---|
| `import pokezero.<mod> as <alias>` in the tree | 31 | **24** | `ast.Import` walk over `git ls-files -z '*.py'` — 524 files, 0 unparsed |
| any dotted read of either global | implied nonzero | **0** | `git grep -hoE '\w+\.(OBSERVATION_SCHEMA_VERSION\|DEFAULT_REPLAY_OBSERVATION_SPEC)\b' -- '*.py' \| wc -l` |
| module-qualified *defaults* in `src/` — the motive given for one matcher arm | "31 times" | **0** | follows from the row above: no dotted read exists, so none is a default |

Grep disagrees with the AST, and the reconciliation is the point — an earlier revision quoted "27"
with no command, cited three line numbers that were all wrong, and named the wrong file set:

```
   24  real imports                          ast.Import walk over tracked .py
 +  2  prose inside .py files                scripts/schema_default_ledger.py:300   (a comment)
                                             tests/test_schema_default_ledger.py:413 (a probe string)
 = 26  git grep ... -- '*.py' | wc -l
 +  1  this document's own spelling table    docs/schema_default_ledger.md:45
 = 27  git grep ... | wc -l   (every tracked file)
```

Every grep-only hit is prose *about* the idiom rather than a use of it — one comment, one probe
string, and one line of this file counting itself. Conversely a `^`-anchored grep reports **0** bare
`import pokezero` against the AST's **3** (`scripts/hc_depth_grid.py:135`,
`src/pokezero/public_projection.py:2283`, `src/pokezero/truth_differential.py:912`), because all
three are indented inside functions. **The AST count is the one to quote**, and where the tools
disagree the difference is attributed line by line rather than settled by preferring a number.

The matcher covers the spellings with zero live occurrences as *guards against a spelling that has
not landed*, which is the honest reason to keep them — not, as earlier text implied, as a fix for
sites already there.

### Surfaces are DERIVED, not listed

`derive_surfaces()` scans `src/` for every class attribute or parameter whose *default* is one of
the globals. The first version hardcoded three call names and undercounted by **187 sites** —
`LocalShowdownConfig.observation_spec` alone has ~133 callers, none counted. That is this
program's own error class committed inside the instrument built to retire it, so the list is now
derived and a new surface is counted the day it is written.

Alternate constructors cannot be derived (they do not re-declare the field), so
`EXTRA_CONSTRUCTORS` names them and **hard-fails** if its owner stops defaulting to a global.

**Spelling coverage matters more here than for reads.** A missed read loses one row; a missed
*declaration* loses the surface and with it every call site — `LocalShowdownConfig` alone is 133 of
the 391. For six rounds this function recognised exactly one spelling (`name: T = GLOBAL`) while the
read matcher accumulated four, and the round-5 alias fix was applied to the read matcher and never
here. Quantified: with the same defaulted field and ten constructions, the plain spelling moved N by
+11 (1 declaration + 10 callers) while `field(default=GLOBAL)`, an aliased global, and an
un-annotated attribute each moved it by **+1** — the ten callers invisible. Combined with a relative
import, even the +1 vanished.

Six spellings now derive, all pinned: annotated, un-annotated (`ast.Assign`), aliased,
`field(default=…)` (**201** `field(default` uses in `src/`), `field(default_factory=lambda: …)`, and
module-qualified at any depth. The last of those needed a whole-chain walk rather than checking the
two ends: `a.b.GLOBAL.attr` puts the global in the middle, where neither the outermost `.attr` nor
the leftmost root finds it.

### It fails loudly, in every mode

An unparsable or missing file is reported as an `UNPARSED` row and the script **exits 2 in every
output mode, including `--json`**. That mattered: an earlier version returned 0 from the `--json`
path — the one mode the CI gate consumes — so regenerating the allowlist under an interpreter that
could not parse a file silently dropped N from 202 to 104, baked the `UNPARSED` marker into the
allowlist, and the gate went green over 98 permanently unmeasured sites.

## The vocabulary

- **A specific version** → name it: `OBSERVATION_SCHEMA_VERSION_V2_2`.
- **A property** → `schema_with(transition_region=True)`, `schema_with(turn_merged=True, grouped_layout=False)`.

`schema_with` returns the **newest** supported schema with the requested properties, raises on an
unsatisfiable set, and raises when given no properties — because that would just be a spelling of
"the current default".

Newest-first has a real hazard worth knowing: `schema_with(turn_merged=True)` returns **v3**, not
v2.2. Mechanically converting a v2.2 test to the property form *changes which schema it runs
under*. If the subject genuinely is one version, name that version.

## The gate

`tests/test_schema_default_ledger.py` re-derives the site set and compares it to
`tests/data/schema_default_allowlist.json`. Python cannot make a module constant unreadable, so
"fail-closed" is an authorship-time check: a new default reader fails in the PR that introduces
it, not during the next rotation.

Rows key as a **multiset** on `file::owner::kind::unclosed`:

- Not a set — that let a plain ADDITION at an existing key pass unseen. A `Counter` difference
  catches a key whose count grew (verified: 391 → 392 reddens; invisible to a set).
  **What it does NOT close:** migrating one site and adding another under the *same*
  `file::owner::kind::unclosed` leaves the count unchanged and passes. 26 keys carry more than one row, the
  largest 4. An earlier version of this document claimed the multiset closed that case; it does
  not, and saying so was worse than the gap.
- Not including the line number — that made the allowlist interpreter-dependent (`ast` reports a
  different `lineno` for the same multi-line call across versions) and broke CI with an
  off-by-one.
- `owner` is the **innermost** enclosing scope. The first version used `ast.walk` (breadth-first)
  with `setdefault`, which locked in the outermost — every method of a `TestCase` collapsed onto
  the class name. Measured at the time: 115 of 202 rows (57%) invisible to any key comparison. On
  the current tree the same collapse would hide 38%; both figures are stated because a single
  percentage with no tree attached is the kind of number this document is about.
- `unclosed` is **part of the key**. Without it a row could silently gain routes: deleting both
  width kwargs from an existing call took it from defaulting one route to three with N, the key
  count and every gate test unchanged — a regression of exactly the shape this ledger cites as its
  reason to exist. The trade-off, stated because the alternative is stating only the win: closing a
  route also changes the key, so a strict *improvement* (adding `numeric_feature_count=`) reddens
  the gate and requires regenerating the allowlist.

**The allowlist is grandfathered exposure, not a list of blessed readers.** It is expected to
shrink. A row is legitimate only if the site *answers* "nobody said" — the definition of the
default, the CLI's `or` fallback, a fingerprint that must hash whatever is current, a config
type's own field default. A site that *consumes* the answer is a conflation.

### Routes that still add a default read with N unchanged

Enumerated because the alternative — this document's own earlier habit — is to state the closed
holes and leave the open ones for a reviewer to find. Each was verified by probe against the
committed tree (N = 391), and none is claimed to be closed:

| route | probe result |
|---|---|
| A *consuming* read added inside `src/pokezero/observation.py` | 0 rows. `DEFINITION_SITES` excludes the whole **file**, not the definition's lines, so the one file most able to conflate is the one file unmeasured. |
| A surface declared **outside** `src/` | 1 row for the declaration, **0 for its callers** (5 callers → 0). `derive_surfaces()` only scans `src/`, so a test-local config class with a defaulted field is a private, uncounted surface. |
| Rebinding a constructor — `LSC = LocalShowdownConfig` | 0 rows for 3 call sites. Only the original name is matched. |
| `getattr` / `vars` / `importlib` | 0 rows. Any dynamic lookup is invisible to an AST matcher; this is a floor on the approach, not a bug in it. |
| Migrating one site and adding another at the same `file::owner::kind::unclosed` | count unchanged, passes. 26 keys carry >1 row, largest 4 (above). |

The first three are narrowable and the fourth is not. They are listed rather than fixed because
each trade cost against a real hole, and an unstated hole is the failure mode this whole document
exists to record: **the gate bounds exposure, it does not eliminate the class.**

## Exposure is not realised failure

Two different measures, both needed:

- **N** — how many sites *could* conflate. A site that takes the default and genuinely does not
  care is counted, correctly: the day it starts caring, nothing warns anyone.
- **A rotation** — how many *actually do*, today. Measuring that means rotating the default and
  counting what breaks; the harness for it is not part of this PR.

Claiming the class is dead on a rotation result alone repeats the original mistake: a true number
answering a narrower question than the one asked.
