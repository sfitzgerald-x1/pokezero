# Constructed tactics: depth WORKS in the model-leaf search

**Status:** COMPLETE. Verdict **(a)** — the tree converts depth into finding
provably forced wins, with the production model leaf and with the handcrafted
control leaf, at exactly the depths the positions require. The flat strength
ladders of `docs/mcts_dual_grid_findings.md` are a property of the game
distribution and the budget/cap interaction, not of the search mechanism.

**Date:** 2026-07-29.
**Build:** branch `scott/depth-tactics-suite` off main @ `4f20e83`; engine
fingerprint `5b29e611468d3baa930984d5b8557280835e72f1ce38d8dc3c6b183e15c344dc`
(34 vendored patches, 8 crate sources; `scripts/engine_behavioral_probes.py`
all-PASS on this venv), crate built with the model feature
(`tests.test_crate_model_leafeval` parity 0.000e+00).
**Checkpoint (the §8 campaign checkpoint):** `v3hist-k64-enthalf-5m-20260723`
@ `iteration-2657`, sha256
`7d363855b5282498afbe5b76763754b614501a42d7281be612ede6930bbdb1b1`, schema
`pokezero.observation.v3`, transition budget 64, vocab 1216 (artifact reuse
key `054071e376d133a0` — byte-identical key to the artifacts already cached on
the cluster next to the checkpoint, i.e. the same contract recent campaigns
ran). Root encode latched via `env_config_from_checkpoint_provenance`
(masks + spec + vocab); leaf tables exported from the checkpoint under
exporter v4 (vocabulary inside the reuse key).
**Artifacts:** `docs/audit_artifacts/depth-tactics-20260729/probe-priors-on.json`
and `probe-priors-off.json` (per-cell visit distributions, Q values, root
values, reached depths, terminal-branch counts, solver tables, engine state
strings). Tooling: `scripts/depth_tactics_probe.py`; regression suite:
`tests/test_search_depth_tactics.py`.

---

## 1. The question

The dual grid measured playing strength depth-invariant (d1≈d2≈d4≈d6 at
s1024, n=400/cell) — suspicious, but confounded: at s1024 the budget starves
the cap (mean reached depth ~2.5), and random-battle positions rarely *need*
depth. This probe removes both confounds. Each constructed position carries a
**forced multi-turn win whose first move looks locally inferior**, such that a
depth-d search with d < N *cannot* distinguish the winning move and a search
with d ≥ N *must* find it: the forced line ends in exact terminal branches
(`battle_is_over`, priced {0,1} in the tree) inside the horizon.

Three possible outcomes were pre-registered: (a) both leaves find the forced
lines at depth ⇒ depth works, ladders are distributional; (b) the hc leaf
finds them and the model leaf does not at the same depth ⇒ defect in how
model values back up; (c) neither finds them ⇒ tree/backup depth defect the
ladders masked. **The data are (a), unanimously.**

## 2. Design and proofs

Six positions (`scripts/depth_tactics_probe.py::POSITIONS`), each materialized
through the real env boundary — packed Custom Game teams
(`BattleStartOverride`) plus the scenario bridge patch for current HP/PP —
so the model's root observation is a genuine `LocalShowdownEnv` observation of
the position, and the searched engine state is constructed by the production
world path (`EngineMctsPolicy`'s own signals → `world_battle_spec` →
`build_poke_engine_state`). Species and moves stay inside the gen3 randbats
universe (the leaf encoder's vocab-drift guard confirms zero OOV tokens);
movesets are composed for forcing-ness — this is a unit test of the tree, not
an eval.

Design invariants:

- **Single-agent forcing.** The opponent side is one mon with one move (a
  fixed-damage Seismic Toss clock in five positions), so every reached
  decision has a singleton opponent option set and "forced" needs no
  simultaneous-move caveats. The solver asserts the singleton.
- **Proofs against the engine's own game.** `solve_win_bounds` /
  `solve_forced` walk EVERY branch of `pe.generate_instructions`
  (branch_on_damage=true — the branching the tree itself uses at plies 1-2,
  plus KO-splits) to the proof horizon. The engine applies
  trunc(0.925·max) per damaging branch, splits at KO-straddles
  (min = trunc(0.85·max)), branches accuracy and crit-KO. Because crit
  branches that flip a KO are real branches (1/16), an attacking trap arm can
  never be a *strict* forced loss; the honest claim is a probability bound:
  - `p_win_lower(win move)` = win probability the searcher can force
    (non-terminal horizon leaves count 0);
  - `p_win_upper(trap move)` = the most the trap can possibly win
    (non-terminal leaves count 1).
- **Needed depth is measured, not assumed:** the flip point of an exact
  fixed-horizon expectimax with the crate's own HP-fraction leaf — what an
  infinitely-sampled depth-h control search converges to.

| position | shape | win vs trap | p_win (win) | p_win ≤ (trap) | needed depth |
|---|---|---|---|---|---|
| `hb-recharge-trap` | Snorlax 210/461 {return, hyperbeam} vs Registeel 85/301 {seismictoss} | return vs hyperbeam | **1.0** | 0.0625 | **2** |
| `sd-race` | Snorlax 310/461 {return, swordsdance} vs Registeel 160/301 | swordsdance vs return | **1.0** | 0.176 | **2** |
| `perish-clock` | Lapras 310/401 {perishsong, surf} + Registeel 95/301 bench vs Blissey 620/651 {seismictoss} | perishsong vs surf | **1.0** | **0.0 (strict)** | **5** |
| `immediate-ko-control` | Snorlax 110/461 vs Registeel 55/301 | hyperbeam vs return | 0.9 (miss branch) | 0.0625 | **1** (all depths must agree) |
| `sd-race-deep` | Snorlax 410/461 vs Registeel 250/301 | swordsdance vs return | **1.0** | 0.176 | **3** |
| `hb-recharge-trap-b` | Tauros 210/291 {return, hyperbeam(pp=1)} vs Umbreon 310/331 | return vs hyperbeam | **1.0** | 0.121 | **2** |

Hand-derived lines (full arithmetic in each spec's `derivation`; engine rolls
cross-checked with `pe.calculate_damage` on the materialized states):

- **hb-recharge-trap (d2):** toss=100 exact; return avg 46 (max 50/min 42),
  hyperbeam avg 67 (max 73/min 62). At 85 hp the beam cannot KO (73 < 85, no
  straddle) but out-damages return at one ply; the recharge turn hands B the
  race (A dies to the third toss before acting). Return leaves 39 ≤ min 42 ⇒
  certain KO at ply 2. d1 sees 67 > 46; d2 sees the terminal.
- **sd-race (d2):** +2 return avg 92; 46+46 = 0+92 makes depth-2 cumulative
  damage tie *by construction* — the tie is broken toward the win by the +2
  crit-KO branch (200 ≥ 160, an exact 1/16 terminal), so the measured flip is
  h2. Spam deals 138 < 160 on every branch and A dies during T4.
- **perish-clock (d5, the keystone):** surf (avg 75) can never race 620 hp
  and never reaches crit range ⇒ the trap arm is a **strict** forced loss —
  the only position with zero crit lottery. Perish sung at T1 faints B when
  the count expires at the end of T4 while the benched fodder survives
  untouched; sung at T2 or later, our side is emptied mid-T5 before the count
  expires (A dies mid-T4, the one-toss fodder mid-T5). Tree plies ≠ game
  turns here: A's mid-T4 faint inserts a forced-replacement decision node, so
  the win terminal sits at ply 5 — d4 provably cannot see it, d6 must.
- **immediate-ko-control (d1):** hyperbeam min 62 ≥ 55 ⇒ certain KO now (gen3
  skips the recharge on a KO — engine-verified); return max 50 cannot KO and
  the second toss kills A. Positive control: every depth must agree, proving
  the probe is not simply anti-Hyper-Beam and deep search does not overthink
  a won position.
- **sd-race-deep (d3):** one dance + three +2 returns (276, final KO certain:
  66 ≤ min 85) beats four spam returns (184 < 250); h2 ties (92=92), h3
  separates on the gradient, ply-4 terminal proves it.
- **hb-recharge-trap-b (d2):** same mechanism as P1 with different tokens and
  the speed relation inverted (Tauros faster). Hyper Beam pinned to 1 PP:
  with a second use, hb/recharge/hb = 326 ≥ 310 would *win* — measured on the
  first design pass and closed. Return×2 leaves 88 ≤ min 102 ⇒ certain T3 KO.

The committed suite (`tests/test_search_depth_tactics.py`) pins all of
section 2 plus the control arm of section 3 as regressions.

## 3. Results

s4096 (cap-binding budget; these trees saturate far below it), c_puct 1.4,
deep_ko_split on, seeds {7, 1337, 900913}, wins counted over 3 seeds.
"Needed" row-cells are bold.

### 3.1 Control arm — `hp_fraction_crate` (`puct_search_multi`)

| position (needs) | d1 | d2 | d4 | d6 |
|---|---|---|---|---|
| hb-recharge-trap (2) | 0/3 | **3/3** | 3/3 | 3/3 |
| sd-race (2) | 0/3 | **3/3** | 3/3 | 3/3 |
| perish-clock (5) | 0/3 | 0/3 | 0/3 | **3/3** |
| immediate-ko-control (1) | **3/3** | 3/3 | 3/3 | 3/3 |
| sd-race-deep (3) | 0/3 | 0/3 | **3/3** | 3/3 |
| hb-recharge-trap-b (2) | 0/3 | **3/3** | 3/3 | 3/3 |

24/24 cells match the exact-solver prediction, unanimously across seeds. The
instrument behaves exactly as the theory of the tree says it must.

### 3.2 Model leaf, priors OFF (pure value-backup pathway)

| position (needs) | d1 | d2 | d4 | d6 |
|---|---|---|---|---|
| hb-recharge-trap (2) | 3/3* | **3/3** | 3/3 | 3/3 |
| sd-race (2) | 3/3* | **3/3** | 3/3 | 3/3 |
| perish-clock (5) | 0/3 | 0/3 | 0/3 | **3/3** |
| immediate-ko-control (1) | **3/3** | 3/3 | 3/3 | 3/3 |
| sd-race-deep (3) | 3/3* | 3/3* | **3/3** | 3/3 |
| hb-recharge-trap-b (2) | 0/3 | **3/3** | 3/3 | 3/3 |

\* = the model resolves the position EARLIER than the HP leaf can (see §4.2).

**The keystone row:** on `perish-clock` — the one position whose trap is a
strict forced loss and whose win is provably invisible below ply 5 — the
model-leaf search picks the trap at d1/d2/d4 (surf shares 0.49/0.45/0.43) and
flips unanimously at d6 (perishsong 0.95 visit share, Q 0.513 vs 0.021/0.002,
66 terminal branches in-tree, reached depth 5). That is depth, and only
depth, converting into the right answer through model-value backup.

### 3.3 Model leaf, priors ON (production shape)

Same resolution pattern as priors-off everywhere (perish-clock: d1 1/3, d2
2/3 from prior noise on near-tied Qs, d4 0/3, **d6 3/3**; every other
position at or before its needed depth, 3/3). Priors sharpen visit
concentration on arms the value already prefers; they did not flip any
argmax against the value pathway on these positions.

## 4. Reading the data

### 4.1 Verdict (a), and what exactly is exonerated

Traverse/expand/backup, exact-expectation chance resolution, terminal
pricing, KO-split branching, depth capping, and the model-leaf plumbing
(per-branch synthesized events → fold advance → native encode → TorchScript
eval → backup) all convert additional plies into finding forced wins at
exactly the depth the position demands — including through a forced-
replacement decision node (perish-clock's ply-5 terminal) and including NOT
degrading on a position deep search could overthink (control row). Combined
with the dual grid this closes the mechanism question: **the flat ladders are
not a search defect.** At s1024 the cap starves (mean reached ~2.5) and
random-battle decisions that both *need* ≥4 plies and are *reachable* by the
visit distribution are evidently too rare to move a 400-game strength cell.

### 4.2 The model leaf is a better shallow evaluator than the control leaf

On hb-recharge-trap, sd-race and sd-race-deep the model-leaf search finds the
win at d1 with priors OFF — the trained value function already ranks the
post-move states correctly one ply out (e.g. sd-race d1: Q 0.053 vs 0.036;
it values the boosted state, no terminal in sight), where the HP leaf
mathematically cannot (hp differentials point the wrong way until d2/d3).
This is the value head carrying real tactical knowledge; nothing here is a
shortcut artifact — on the two positions where one ply genuinely cannot know
(hb-recharge-trap-b's inverted speed shape, perish-clock), the model-leaf d1
takes the trap like everything else.

### 4.3 Caveats worth carrying forward

- **Absolute model values on these positions are wildly pessimistic** (root
  values 0.03–0.10 on provably WON positions at shallow depth; compare the
  HP leaf's 0.46–0.60). Constructed 1v1/2v1 sides with five empty party
  slots pattern-match "we lost five mons" for a model trained on 6v6
  randbats. Rankings between arms stay correct, and terminal backup restores
  the absolute scale once the horizon reaches the win (0.86 at d2 on P1;
  0.49–0.51 on the perish win — a running mean over early pessimistic leaf
  samples plus terminal 1.0s, not a calibrated probability). Anyone reusing
  these positions for value-calibration claims should not.
- **Tree plies ≠ game turns** when mid-turn faints force replacements. Depth
  budgeting or reached-depth analysis that equates the two will be off by
  one on exactly the positions where being right matters.
- **`max_depth_reached` counts decision-node depth**: a saturated cap-d tree
  reports d−1. The dual-grid "mean reached ~2.5 at cap 6" statements and
  these cells use the same convention.
- **Engine gap found in passing:** NIGHT SHADE is a no-op in the vendored
  gen3 engine (empty instruction list from `generate_instructions`;
  Seismic Toss is correct). Flagged for the divergence-ledger workflow as a
  separate task; this probe switched its clock to Seismic Toss and is
  unaffected.

## 5. Reproduction

```sh
# design pass only (no checkpoint): forcing proofs + flip tables
PYTHONPATH=src .venv/bin/python scripts/depth_tactics_probe.py --design-only

# full grid, both leaves (checkpoint = the §8 campaign checkpoint)
PYTHONPATH=src .venv/bin/python scripts/depth_tactics_probe.py \
  --checkpoint local-artifacts/v3hist-k64-enthalf-5m-20260723-iteration-2657.pt \
  --out docs/audit_artifacts/depth-tactics-20260729/probe-priors-on.json
PYTHONPATH=src .venv/bin/python scripts/depth_tactics_probe.py \
  --checkpoint local-artifacts/v3hist-k64-enthalf-5m-20260723-iteration-2657.pt \
  --no-model-priors \
  --out docs/audit_artifacts/depth-tactics-20260729/probe-priors-off.json

# committed regressions (control arm + proofs; no checkpoint needed)
PYTHONPATH=src .venv/bin/python -m unittest tests.test_search_depth_tactics
```

The checkpoint is not committed (40 MB; lives on the cluster at
`<private-store>/v3hist-k64-enthalf-5m-20260723/run/iteration-2657/transformer-policy.pt`);
`local-artifacts/` is untracked. Any engine-compatible checkpoint reproduces
the probe through the same latch path.
