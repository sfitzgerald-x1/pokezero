# The k0 depth grid: does a history-free checkpoint decay with depth?

**Date:** 2026-07-29
**Status:** k0 is flat across depth; k64 decays. The difference points the way the
off-distribution-history hypothesis predicts, but at n=100 per cell it is
**suggestive and not significant** (slope difference +0.110, z = 1.28). Reported
as a direction, not a cause.

Companion to `docs/mcts_degradation_findings.md`. That document's §9 established
the durable fact this one tests: search loses to its own prior, monotonically
worse with rollout depth, on a checkpoint that carries a 64-token history region.

---

## 1. The question

Deep-rollout leaves are encoded from FOLD-ADVANCED history windows — event
sequences synthesized from engine branches rather than replayed from real
Showdown protocol. A history-carrying (k64) checkpoint's value head never saw
that distribution in training, and the exposure compounds per ply.

A budget-0 (k0) checkpoint carries no history content at all: its transition
region is present in the layout but fully masked, so a leaf encode writes zero
transition rows and zeroes every transition attention bit. It therefore cannot
be fed synthesized history, and cannot suffer this failure mode.

**Prediction:** k0 flat across depth, k64 decaying.

---

## 2. Can a budget-0 checkpoint be searched at all?

Yes. This was the first thing checked, because a refusal would have been the
finding. `resolve_checkpoint_contract`, `materialize_search_artifacts`,
`export_model.py`, `export_encoder_tables.py` and `EngineMctsPolicy` all accept
the k0 checkpoint with no relaxation and no local patch. Schema
`pokezero.observation.v3`, 87 tokens, `transition_token_count` 64,
`transition_token_budget` 0.

The crate handles the zero budget correctly by construction, not by accident:
`budget = 0` makes `tokens[start..]` empty (`encoder.rs:2323`), so no transition
row is written, and `filled = 0` (`encoder.rs:761`) clears every transition
attention bit.

**The k0 deployment path is validated for engine search.**

---

## 3. Setup

Both arms: engine MCTS vs the SAME checkpoint's raw policy, `s1024 / batch 64 /
worlds 4`, seeds 600000–600099, seats mirrored by seed parity. n = 100 per cell.

| | checkpoint | budget |
|---|---|---|
| **k0** | `v3hist-k0-enthalf-5m-20260724` @ `iteration-2519` | 0 |
| **k64** | `v3hist-k64-enthalf-5m-20260723` @ `iteration-2657` | 64 |

Same architecture (d=512, 3 layers, window 1, 87 tokens) and comparable training
state (policy accuracy 0.752 vs 0.726, value loss 0.0578 vs 0.0579).

Image `scott-experiment:mcts-rb-bf72636` — the same build §9 used, chosen so the
k64 arm is comparable to §9 rather than to nothing. It predates #937 and #939.

---

## 4. Results

Each checkpoint is scored against its own raw policy, so 0.500 is the null for
both. The measured null is reported because raw-vs-raw is 0.500 only in
expectation: with two identical deterministic policies the winner of a seeded
game is a property of the game, so each checkpoint has its own baseline.

### 4.1 k0 — flat

| cell | n | score | Wilson 95% | p1 | p2 |
|---|---|---|---|---|---|
| control (raw v raw) | 100 | 0.570 | [0.472, 0.663] | 0.480 | 0.660 |
| d1 | 100 | 0.490 | [0.394, 0.587] | 0.520 | 0.460 |
| d2 | 100 | 0.460 | [0.366, 0.557] | 0.540 | 0.380 |
| d4 | 100 | 0.500 | [0.404, 0.596] | 0.600 | 0.400 |
| d6 | 100 | 0.470 | [0.375, 0.567] | 0.560 | 0.380 |

Slope d1 → d6: **−0.020** (z = −0.29). Flat. On p1 alone: 0.520 → 0.560, **+0.040**.

### 4.2 k64 — decays, on freshly exported tables

| cell | n | score | Wilson 95% | p1 | p2 |
|---|---|---|---|---|---|
| control (raw v raw) | 100 | 0.485 | [0.389, 0.582] | 0.420 | 0.540 |
| d1 | 100 | 0.480 | [0.385, 0.577] | 0.520 | 0.440 |
| d4 | 100 | 0.370 | [0.282, 0.468] | 0.440 | 0.300 |
| d6 | 100 | 0.360 | [0.273, 0.458] | 0.440 | 0.280 |

Slope d1 → d6: **−0.120** (z = −1.77). On p1 alone: 0.520 → 0.440, **−0.080**.

The §9 decay reproduces (0.360 at d6, identical to §9's 0.360) on tables rebuilt
from scratch, so none of the encoder-table defects in §6 is its cause.

### 4.3 The comparison

Both arms sit at 0.52 on p1 at d1 and separate only as depth grows.

| | k0 | k64 | difference | z |
|---|---|---|---|---|
| slope d1 → (d4+d6), pooled | −0.005 | −0.115 | **+0.110** | +1.28 |
| slope d1 → (d4+d6), p1 only | +0.060 | −0.080 | **+0.140** | +1.15 |
| slope d1 → (d4+d6), p2 only | −0.070 | −0.150 | **+0.080** | +0.67 |

The sign is the predicted one in every split. **None reaches significance.**

The deep-cell LEVEL contrast does reach it (pooled 0.485 vs 0.365, z = +2.45),
but that number is not baseline-free — the two checkpoints' measured nulls differ
by 0.085, which is most of the 0.120. The slope cancels the baseline and is the
honest test. It gives z ≈ 1.3.

**Roughly 400 games per cell would be needed to resolve a 0.11 slope difference.**

---

## 5. Seat split

Cells are reported split by seat because this build predates #937, which fixed a
seat-CONSTANT model-value inversion: on p2-seated roots the model's leaf values
entered the side-one-absolute tree unreflected.

`p1 − p2` is **positive at every search depth for both checkpoints** (+0.06 to
+0.20) and **negative in both controls** (−0.18, −0.11). The controls never touch
the search engine, so the sign flip between control and search cells is the
inversion's signature, and it is model-independent — exactly what a seat-constant
defect predicts.

The §9 reference makes it starkest: at d6, k64 as-shipped scores **0.540 on p1
and 0.180 on p2**. The headline 0.360 is the average of a healthy seat and a
broken one.

The k0-vs-k64 contrast survives the split: it is measured on p1 alone, where the
inversion does not apply, and is larger there (+0.140) than pooled.

---

## 6. Three encoder-table defects found on the way

Found while establishing provenance. **None of them causes the depth decay** —
§4.2 re-ran k64 on tables with all three corrected and the decay is unchanged —
but two are live for other consumers and all three are fixed here.

1. **Feature masks were never derived from the checkpoint.** `_layout_payload`
   built `default_feature_masks` from `ObservationFeatureMasks()` defaults;
   `--checkpoint` threaded the observation spec through but not the masks. Every
   tables file on disk claims `tier2_investment: False` (every trained checkpoint
   is True) and the full history region (a k0 checkpoint wants 0).

   **Not live on the engine-search path.** `_latch_encoder_tables_to_model_config`
   overwrites the mask block with the checkpoint's own before the crate parses
   the tables. Confirmed empirically, not by reading: d1 run against as-shipped
   tables and against corrected tables gave **bit-identical outcomes on all 100
   seeds** — same winner, decision count and fallback count on every one.

2. **`validate_encoder_tables` compared only four layout scalars.** Those describe
   the observation's SHAPE; tables of the right width can still describe filling
   a region the checkpoint masks. The mask block is now compared too.

3. **Stale artifacts are adopted forever, and this one IS live.** Artifacts are
   reused whenever the file exists, and the reuse key covers the checkpoint and
   the exporter revision but not the code that enumerates vocabulary tokens. The
   k64 tables cached on 07-28 carry a **1233**-token vocab; this build produces
   **1234** (`volatile:solarbeam` was added). The vocab is a positional list, so
   every token from index 1204 on is renumbered and the crate resolves a
   different embedding row than the root encode for the same value. The latch
   rewrites only the mask block; it does not touch the vocab.

   **§9's k64 grid ran on those stale tables.** §4.2 above re-ran it on fresh
   ones: 0.360 → 0.360. Real defect, not the cause.

`validate_encoder_tables` now rebuilds the vocabulary when given a showdown root,
and `EXPORTER_REVISION` is bumped to v3 to invalidate the bad artifacts.

---

## 7. Triangulation across the three lanes

| lane | question | result |
|---|---|---|
| ply-parity (#937) | value orientation applied per ply? | **refuted** — but found a seat-CONSTANT inversion, fixed |
| handcrafted-leaf | is the decay in the tree or the value? | **the tree is exonerated** — with a handcrafted leaf, depth *helps* (0.196 → 0.289 → 0.328 over d1/d2/d4) |
| **this lane** | is it the model's history region? | k0 flat, k64 decays; **suggestive, not significant** |

Read together:

- The handcrafted-leaf lane puts the decay **in the model leaf-value pathway**,
  not in PUCT, chance nodes, backup, or the depth cap. The identical tree
  improves with depth once the network is removed from the leaf.
- Within that pathway, ply-parity found one concrete defect — the seat inversion
  — which is model-independent and explains the p2 collapse, and is now fixed.
- This lane asks whether anything model-DEPENDENT remains after that. On the
  uncontaminated p1 seat, k64 still slopes down (0.520 → 0.440) where k0 slopes
  up (0.520 → 0.560). That residual is what the history hypothesis predicts, and
  it is where the remaining signal is — but it is a 1.2-sigma result.

Which combination implies what:

| handcrafted lane | this lane (k0 vs k64 slope) | implication |
|---|---|---|
| tree exonerated | k0 flat, k64 decays (**observed**) | decay is model-leaf; a model-dependent component survives the seat fix — history is the live candidate |
| tree exonerated | both decay equally | decay is model-leaf but history-independent; the seat inversion plus leaf-value quality account for it |
| tree also decays | either | the tree would have been implicated and this lane could not isolate anything |

The observed row is the first. It does **not** license calling synthesized
history the cause — §9's lesson is that each prior mechanism was plausible,
explained the table's shape, and was wrong. It licenses one more falsifiable
experiment, specified below.

---

## 8. Next step, chosen so it can fail cleanly

Re-run k0 and k64 at d1/d4/d6 on **current main** (post-#937, post-#939), at
**n ≈ 400 per cell**, both seats now valid.

- If the k64 slope stays negative while k0's stays flat at that power, the
  history region is implicated with an interval that means something.
- If both go flat, the seat inversion was the whole effect and this lane's
  direction was noise.

A rebuild is required (#937 is in the crate). It was not done here: on this build
the p1 half already measures fixed-build behaviour, and n=100 per cell was the
binding limit rather than the seat contamination.

Caveats to carry: the two checkpoints are different training runs (iteration 2519
vs 2657) and a residual "different model" explanation is not fully excluded, only
mitigated by each being scored against its own prior. `s/decision` figures here
predate #939 and are ~2x inflated; relative comparisons hold, absolute seconds
do not.

---

## 9. Artifacts

`docs/audit_artifacts/k0-depth-grid-20260729/` — every cell's per-game record,
its own provenance block (which tables it ran against, and any drift between
those tables and the checkpoint), plus the §9 k64 reference for the seat split.

    PYTHONPATH=src python scripts/k0_grid_report.py \
      --results docs/audit_artifacts/k0-depth-grid-20260729/results --contrast c:k64c
