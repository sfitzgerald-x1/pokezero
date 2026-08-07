# C142 — red-run index for the belief-surface follow-on

**Why this exists.** `docs/engine_fidelity_program_20260801.md:194` requires a pin to ship with
its red run "**with the output quoted**". An audit scored the four PRs in this effort **2 of 4**:
#1123 and #1160 documented theirs, #1154 and #1156 published only kill *counts* — no output, no
commands — and **no exact mutation command existed anywhere in any of the four**. So the pins were
real (the auditor drove every one red independently) but not reproducible from the record.

This index is the missing half. Every entry is the exact mutation, run at
`a9e7f88c`, with output pasted from the run and not retyped. Restore the file after each.

Environment for all of them:

```sh
export POKEZERO_SHOWDOWN_ROOT=<a pokemon-showdown checkout>
# from the repo root, using .venv/bin/python
```

---

## Task 1 — the spread gate rides checkpoint provenance

**Acceptance:** both directions pinned, each shown red on a tree with the gate keyed off global
config instead.

**Mutation.** `src/pokezero/showdown.py:6500`, replace provenance with a constant:

```python
-                    exact_spreads=schema_v4,
+                    exact_spreads=True,     # or False — a global, not provenance
```

```
$ .venv/bin/python -m unittest tests.test_spread_gate_provenance

--- global = True (v4 spreads for everyone) ---
FAIL: test_a_v3_checkpoint_can_never_receive_v4_spreads (… species='Salamence',
      column='NUMERIC_EXPECTED_ATK_HIGH')
AssertionError: 0.33473389355742295 != 0.33613445378151263 : NUMERIC_EXPECTED_ATK_HIGH:
  a v3 checkpoint did NOT receive the legacy spread it trains against
  (got 0.33473389355742295, legacy 0.33613445378151263, v4-corrected 0.33473389355742295).

--- global = False (v3 spreads for everyone) ---
FAIL: test_a_v4_checkpoint_can_never_receive_v3_spreads (… species='Salamence',
      column='NUMERIC_EXPECTED_ATK_HIGH')
AssertionError: 0.33613445378151263 != 0.33473389355742295 : NUMERIC_EXPECTED_ATK_HIGH:
  a v4 checkpoint did NOT receive the corrected spread
  (got 0.33613445378151263, v4-corrected 0.33473389355742295, legacy 0.33613445378151263).

--- restored ---
Ran 6 tests … OK
```

Each setting reddens **only the direction it violates**, and the message names which. That is the
two-way property; a single-direction pin would go red in both.

**Botched first attempt, recorded because the failure mode is the point.** I first mutated to
`os.environ.get(...)`. `showdown.py` does not import `os`, so the run gave `FAILED (errors=7)` —
a crash, not a red run, and it would have read as a pass of the acceptance if I had only looked at
"is it red". Check that a red run is red *for the reason claimed*.

---

## Task 3 — the denominator rule

**Acceptance:** forcing any harness to skip 100% of boundaries makes it exit nonzero, pinned.

**Mutation.** `scripts/differential_denominator.py`, make the rule toothless at the top of `gate()`:

```python
 def gate(reports):
+    return 0  # MUTANT: denominator rule removed
     if not reports:
```

```
$ .venv/bin/python -m unittest tests.test_denominator_adoption
FAIL: test_a_fully_skipped_corpus_exits_nonzero (… harness='leaf_vs_reality')
FAIL: test_a_fully_skipped_corpus_exits_nonzero (… harness='leaf_root_parity')
FAIL: test_a_fully_skipped_corpus_exits_nonzero (… harness='prior_mapping_assert')
FAIL: test_a_fully_skipped_corpus_exits_nonzero (… harness='fidelity_gate_events')
  File "tests/test_denominator_adoption.py", line 140 …

--- restored ---
Ran 13 tests … OK
```

All four subtests red independently — the acceptance is per-harness, so one harness surviving
would mean the rule is adopted in name only there.

The three guard pins added later (vocabulary `raise`, the `NOT CHECKED` render phrase, the
`contained=` subscripts) have their mutation table in #1154's body; their exact mutations are
`if unknown:` → deleted, `parts.append("rule 4 NOT CHECKED…")` → `pass`, and
`contained=report["boundaries"]` → `report.get("renamed")`.

---

## Task 4 — recharge symmetry, gates first

### The gate derivation

**Mutation.** `fidelity_gate_events.production_recharging_slots`, back to the circular rule:

```python
-    if anchor_metadata.get("self_must_recharge") is True:
-        slots.append(seat)
+    # derive from the RECORDED CHOSEN CANDIDATE instead of the parser tracker
+    if any(c.get("action_index") == idx and normalize_id(c.get("move_id")) == "recharge" …):
+        slots.append(seat)
```

```
$ .venv/bin/python -m unittest tests.test_recharge_gate_derivation
FAIL: test_the_rules_differ_wherever_the_action_record_lies
FAIL: test_it_catches_a_write_that_CONTRADICTS_the_tracker_false_case
FAIL: test_the_fixed_rule_locks_our_own_slot_from_the_TRACKER
Ran 12 tests … FAILED (failures=3)

--- restored ---
Ran 12 tests … OK
```

### The production change

`_recharging_slots` symmetry, the fallback's self-lock prefix, and the `"No Move"` mapping each
have their own mutation and count in #1156 and the follow-up; the one worth repeating here is the
mutation that **survived 24 tests** until review found it — discarding the self lock only when the
reconstruction *succeeds*:

```python
-        return self_slot + self._opponent_recharging_fallback(context, opponent_slot)
+        _r = self._opponent_recharging_fallback(context, opponent_slot)
+        return (self_slot + _r) if not _r else _r
```

Now caught by `test_self_lock_survives_when_the_fallback_SUCCEEDS` (1 failure). Every other test in
that class drives the reconstruction to an early `return ()`, where `self_slot + ()` and
`()`-plus-prefix are indistinguishable.

### The injected bad write

Full producing chain, measured output, and limits: `reports/c141_recharge_gate_injection_proof.md`.
That one was documented at the time and is the model this index is trying to match.

---

## What this index does not fix

It is written **after** the PRs merged, so the merge record of #1154 and #1156 still lacks its red
runs; this is a retrofit, not a correction to those bodies. The house rule wants the output in the
PR that introduces the pin. The way to satisfy it is to paste the run when the pin is written —
which costs nothing at the time and cannot be reconstructed for free later, as this document
demonstrates.
