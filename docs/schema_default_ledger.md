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

A default can be reached as a bare name, an alias, or a module attribute:

    OBSERVATION_SCHEMA_VERSION                    # bare
    SV                                            # from ... import ... as SV
    observation.OBSERVATION_SCHEMA_VERSION        # import pokezero.observation as observation

Only the first was matched. The other two were demonstrated to add default reads with N unchanged
and the gate green — the same defect class as the any-of bug: **a denominator blind to a spelling.**
The idiom is not exotic; `import pokezero.<mod> as <alias>` appears 31 times in the tree. Aliases
are now resolved per file, so the reported kind is the *global*, not the local name.

### Surfaces are DERIVED, not listed

`derive_surfaces()` scans `src/` for every class attribute or parameter whose *default* is one of
the globals. The first version hardcoded three call names and undercounted by **187 sites** —
`LocalShowdownConfig.observation_spec` alone has ~133 callers, none counted. That is this
program's own error class committed inside the instrument built to retire it, so the list is now
derived and a new surface is counted the day it is written.

Alternate constructors cannot be derived (they do not re-declare the field), so
`EXTRA_CONSTRUCTORS` names them and **hard-fails** if its owner stops defaulting to a global.

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

## Exposure is not realised failure

Two different measures, both needed:

- **N** — how many sites *could* conflate. A site that takes the default and genuinely does not
  care is counted, correctly: the day it starts caring, nothing warns anyone.
- **A rotation** — how many *actually do*, today. Measuring that means rotating the default and
  counting what breaks; the harness for it is not part of this PR.

Claiming the class is dead on a rotation result alone repeats the original mistake: a true number
answering a narrower question than the one asked.
