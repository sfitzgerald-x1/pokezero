# The schema-default conflation, and the ledger that bounds it

## The defect class

Code and tests that read the global `OBSERVATION_SCHEMA_VERSION` (or `DEFAULT_REPLAY_OBSERVATION_SPEC`)
when they mean *a specific version* or *a schema property*. While the default happens to satisfy
what the site wanted, the conflation is free and invisible. It becomes visible only when the
default moves — and then everything that conflated the two fails at once, with failure messages
that name the wrong subsystem.

It is not a style problem. Two production bugs of this exact shape were found and fixed while
building this ledger: `TransformerPolicyConfig.token_count` and its two feature widths defaulted
off the process-wide spec rather than the config's own stamped schema, so a config stamped one
schema silently carried another's shape.

## N, and the command that produces it

```sh
.venv/bin/python scripts/schema_default_ledger.py            # rows
.venv/bin/python scripts/schema_default_ledger.py --by-file  # per file
.venv/bin/python scripts/schema_default_ledger.py --json     # machine-readable
```

The number is not the artifact — **the command is**. Any figure quoted from memory is worthless;
this effort produced four wrong ones (6, ~4, 97, 94) before the ledger existed, each stated
without an established denominator and two of them from hand-picked file lists.

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

Rows key as a **multiset** on `file::owner::kind`:

- Not a set — that let a diff migrate one site and add another at the same key, which is what an
  ordinary PR looks like. A `Counter` difference catches a key whose count grew.
- Not including the line number — that made the allowlist interpreter-dependent (`ast` reports a
  different `lineno` for the same multi-line call across versions) and broke CI with an
  off-by-one.
- `owner` is the **innermost** enclosing scope. The first version used `ast.walk` (breadth-first)
  with `setdefault`, which locked in the outermost — every method of a `TestCase` collapsed onto
  the class name, and 57% of rows became invisible to any key comparison.

**The allowlist is grandfathered exposure, not a list of blessed readers.** It is expected to
shrink. A row is legitimate only if the site *answers* "nobody said" — the definition of the
default, the CLI's `or` fallback, a fingerprint that must hash whatever is current, a config
type's own field default. A site that *consumes* the answer is a conflation.

## Exposure is not realised failure

Two different measures, both needed:

- **N** — how many sites *could* conflate. A site that takes the default and genuinely does not
  care is counted, correctly: the day it starts caring, nothing warns anyone.
- **A rotation** — how many *actually do*, today. That is what `scripts/schema_rotation_drill.sh`
  measures.

Claiming the class is dead on a rotation result alone repeats the original mistake: a true number
answering a narrower question than the one asked.
