# C117 — the validation holdout, and what it says about the 208 → 7 trajectory

C116 Phase 1 item 5. The first out-of-window measurement in this program's history.

Era: `main` `89bbabe4`, engine fingerprint `8a00d812b41566a0`, 58 patches, mass gate
(`tests/test_branch_mass_reconstruction.py`) green at 4 tests. Artifacts
`/tmp/sweep_holdout.json` (holdout) and `/tmp/sweep_ca.json` (dev window, same engine).

---

## 1. The headline: the dev window is flattering by 3.5×

| window | seeds | boundaries | matched | diverged | rate | errors |
|---|---|---|---|---|---|---|
| dev | 19000000–19000199 | 15,224 | 15,217 | **7** | 0.0460% | 0 |
| **validation holdout** | 19100000–19100199 | 15,396 | 15,371 | **25** | **0.1624%** | 0 |

`matched + diverged == boundaries_measured` on both. The holdout divergence rate is
**3.53×** the dev window's.

**This is C116's M6, confirmed rather than feared.** The entire 208 → 7 era iterated
against seeds 19000000–19000199, and no holdout existed anywhere in the program. The
plan said the stopping condition "can be satisfied by a window overfit"; it can, and
partially was. Any statement of the form "the residue is 7" describes *one particular
200-game window* and must not be read as a fidelity rate.

That is the honest headline and it is recorded before any mitigation below.

## 2. But the excess is concentrated, not diffuse

Two structural facts change the interpretation.

**Class concentration.** Eleven of the twenty-five rows — 44% — carry a single class:

| count | class |
|---|---|
| **11** | `limit:world_sample_drag_target` |
| 4 | `roll_scaled_component` |
| 2 | `component_missing_in_engine:sandstorm` |
| 2 | `limit:roll_divergent_lethality` |
| 1 each | `component_extra_in_engine:itemleftovers,psn,sandstorm`; `component_extra_in_engine:spikes`; `component_mismatch:itemleftovers|leechseed`; `component_missing_in_engine:itemleftovers,movewish`; `component_missing_in_engine:leechseed`; `component_missing_in_engine:psn` |

**Game concentration.** Four games produce fourteen of the twenty-five rows:
`19100122` alone produces **seven** (steps 13, 117, 147, 151, 164, 169, 181), plus
`19100180` three, and `19100072` and `19100142` two each. Twenty-one of the 200 games
produce no divergence at all. So this is not a uniformly worse window; it is a window
containing a few games that exercise a mechanism the dev window barely touches.

## 3. The eleven drag rows are B1, not comparison limits — one verified at source

`19100122/13` replayed in full. Whirlwind drags p1's Sableye out and Shuckle in:

```
|-damage|p1a: Sableye|210/239|[from] Spikes
|drag|p1a: Shuckle|Shuckle, L98, F|172/198 slp
|-damage|p1a: Shuckle|148/198 slp|[from] Spikes
|-heal|p1a: Shuckle|160/198 slp|[from] item: Leftovers
observed p1: itemleftovers=+12, spikes=-29, spikes=-24
```

The engine emits **five** arms at 20% each — it *does* enumerate the candidate set —
and one of them carries the realised target with components
`spikes=-29, rolled move=-24`.

**So the divergence is not the drag target. It is that the second Spikes tick renders
as `move` (rolled) instead of `spikes` (exact).** The engine found the right
replacement; the renderer mis-tagged its Spikes damage. That is exactly cause **B1**,
which C111 v2 recorded from the dev window's `19000008/54`, where the same class name
was applied to a row in which "the engine dragged the *same* Pokémon and only a
component tag differed".

**Attribution status, stated precisely.** One of the eleven is verified at source. The
other ten share the class and the window's phazing-heavy character, so B1 is the
*hypothesis* for them, not an established fact. Ten replays are owed before this is a
finding rather than a lead. Under the C116 M1 rule none of the eleven may be called a
limit without a demonstration artifact, and the one row examined actively refutes the
limit reading.

If B1 accounts for all eleven, the holdout residue is **14** and the rate 0.0909% —
still 1.98× the dev window, so the overfit finding survives the mitigation entirely.

## 4. What this changes about the plan

**B1's priority inverts.** It was one row on the dev window and last in the Phase 3
queue. On the holdout it is the single largest class and 44% of all divergence. Ordered
by rows × search impact — the program's own ranking rule — it is now first, and it is
cheap: a renderer tag fix plus a classifier fix that stops applying the drag limit on
the mere presence of `|drag|`, measured separately per the program's rule.

**The remaining fourteen are mostly already-named causes.** `sandstorm` ×2 and
`roll_divergent_lethality` ×2 are the A4/A7 residual-lethality family; the
`itemleftovers` / `leechseed` / `psn` singletons are the A2 ordered-phase family; the
`movewish` row touches Wish, which #1066 ordered at step 7 and C116 §4's
pool-reachability check confirms is reachable (16 species). `roll_scaled_component` ×4
is the one group with no obvious owner and should be replayed first among them.

**This is evidence for the compression claim, not against it.** "36 rows was never 36
investigations" predicted that out-of-window rows would attribute to existing causes
rather than open new ones. On first inspection they do: no new mechanism has appeared,
one known classifier defect dominates, and the rest fall into two already-named
families. The count went up; the *number of causes* did not.

## 5. What is owed

1. Replay the remaining ten drag rows and either confirm B1 or open a cause.
2. Replay the four `roll_scaled_component` rows — the only unattributed group.
3. Fix B1 (two commits, measured separately), then re-sweep both windows.
4. Leave `19,200,000+` untouched. Per the §J.7 amendment it must appear in **exactly
   one** measurement in the whole record; this report deliberately does not touch it.

## 6. A note on what this measurement cost

Nothing. It is 200 games, nine minutes, and it should have been run months ago —
before the first fix, as a baseline. The reason it was not is that the dev window's
counter was falling, and a falling number is a poor prompt to ask whether you are
measuring the right thing. The plan was right to make this Phase 1 rather than Phase 4.
