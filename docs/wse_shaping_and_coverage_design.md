# WS-E dense shaping + hazard coverage curriculum — arm design

Status: design, 2026-07-03. Companion to [`mcts_design.md`](mcts_design.md)
(WS-E) and [`selfplay_mcts_roadmap.md`](selfplay_mcts_roadmap.md). Motivated by
the 2026-07-03 E1 and ΔV hazard-probe reads.

## Evidence this design responds to

**E1 (value-head search-readiness, 32+32 games × 4 checkpoints, held-out):**
all four candidate heads clear the Pearson ≥ 0.30 bar — belief-1.5m raw 0.515
(ECE 0.091, sign 0.700), fpdistill-1.5m isotonic 0.405, fpdistill-1m affine
0.369, no-belief-1.5m affine 0.309. Cross-pool on the shared belief-1.5m
held-out pool: belief 0.515 > fpdistill-1m 0.493 ≈ fpdistill-1.5m 0.489 ≫
no-belief 0.246. The roadmap's cited ~0.12 was from June-27 local smokes and
understated every candidate by 2–4×. Value heads rank outcomes usefully **in
aggregate**.

**ΔV hazard probe (300 games, 8k-state corpus, 1k ΔV states, Spikes 0→3
injection):**

| head | value_spread | ΔV self-Spikes | ΔV opp-Spikes | policy Spikes argmax | Rapid-Spin response |
|---|---|---|---|---|---|
| belief-1.5m | 0.508 | −0.019 (3.6% of spread) | +0.056 (10.9%) | 0.0 | ≈0 |
| fpdistill-1.5m | 0.360 | −0.009 (2.6%) | +0.087 (24.1%) | 0.0 | ≈0 |

Decomposition of the flat-Spikes failure (the roadmap's myopia concern):

1. **Self-side hazard credit assignment failed** — ΔV to hazards on one's own
   side is ≤4% of value spread in both heads. The value function is
   near-blind to standing on Spikes. Dense shaping is the lever; any 1-ply
   search using these heads inherits the blindness on exactly the lines where
   hazards decide games.
2. **An equilibrium/exploration component is also real** — fpdistill's value
   head moves 24% of spread when the *opponent's* side gets hazards, yet its
   policy still never sets Spikes (argmax 0.0). Where value signal exists and
   the policy is still flat, coverage — not credit assignment — is the
   binding constraint. Ecology caveat from the roadmap stands: in a
   low-switch self-play meta, low Spikes value is partially *correct*
   internally and wrong mainly vs foul-play/humans.

Both levers are therefore justified as separate arms with a shared yardstick.

## Arm 1 — WS-E dense shaping (metamon-style)

### What exists

`TrajectoryDatasetConfig` (`dataset.py`) already supports player-relative,
clipped, opt-in shaping from recorded observation metadata:
`hp_delta_return_weight`, `faint_delta_return_weight`,
`turn_penalty_after`/`turn_penalty`. Terminal win/loss remains the base
return. A June same-seed local ablation (0.5/0.5 weights, 3×256 local PPO
shape) was strength-neutral-to-negative — but that regime is not the current
256d-belief cluster recipe, and the metric that matters here is value-head
hazard/status sensitivity (ΔV), which the ablation never measured.

### Change 1: status-delta shaping term (small, dataset-only)

Rollout metadata already records per-Pokémon `status` on both visible teams
(`self_team[i].status`, `opponent_team[i].status`), so this is a pure
`dataset.py` change mirroring the faint-delta term:

- `status_delta_return_weight: float = 0.0` in `TrajectoryDatasetConfig`
  (+ CLI plumbing in `neural_cli` mirroring the hp/faint flags).
- `_VisiblePokemonSnapshot` gains `status: str | None`; the delta counts
  newly-inflicted minus newly-cured non-volatile statuses, opponent-side
  positive, self-side negative (identical player-relative convention to
  faints). Fainted mons drop out of the status count (metamon convention) so
  a kill is not double-punished as "status cleared".

Explicitly **not** adding a hazard/side-condition shaping term in the primary
arm: rewarding Spikes placement directly would hand-craft the answer the
probe is supposed to detect. (A `side_condition_return_weight` diagnostic
variant is cheap if the primary arm's ΔV stays flat, but it is a fallback,
not the plan.) Hazards must become visible to the value head through their
*consequences* — chip damage on switch-in is exactly what hp-delta shaping
propagates backward.

### Change 2: multi-γ value heads — deferred (phase 2)

Metamon-style multi-γ heads (a second value head at lower γ over the dense
shaped signal, keeping the sparse terminal head for search leaf evaluation)
require `neural_policy.py` head changes, checkpoint-compat guards, and a
loader migration. Gate this on the phase-1 read: if shaping fixes ΔV but
degrades terminal calibration (E1 Pearson/ECE on the shaped run's head), the
two-head split is the remedy. Do not build it speculatively.

### Arm 1 config (cluster)

Base: the standard 500k belief recipe used by the four active arms
(256d reference shape, same seed bands, same eval cadence), plus:

- `--hp-delta-return-weight 0.5`
- `--faint-delta-return-weight 0.5`
- `--status-delta-return-weight 0.25` (new; half the faint weight — a status
  is roughly "part of a faint" in gen3 value terms)
- no turn penalty (terminal-only comparability on game length)
- discount unchanged (0.9999); shaping magnitudes are clipped into [-1,1]
  with the terminal return exactly as today.

## Arm 2 — hazard coverage curriculum

### Lever

The equilibrium failure needs hazard *lines* in the self-play data, not
reward. Cheapest existing machinery: `ScriptedTeacherPolicy` already has
dedicated Spikes / Rapid-Spin / hazard-aware branches (`policy.py`:
`spikes_available`, `rapid_spin_clear_hazards`, side-hazard scoring), and
collection already supports mixed opponent pools
(`opponent_pool_policy_specs`). The June mixed-pool result (scripted-teacher
in pool was *worse* for strength) does not invalidate this: that experiment
measured immediate strength; this arm measures whether hazard consequences
enter the data distribution at all — strength is read at the end, ΔV and
behavior probes along the way.

### Arm 2 config (cluster)

Base recipe as arm 1 (no shaping changes — one lever per arm), plus:

- Opponent pool: 25% scripted-teacher (hazard-capable), 75% mirror/historical
  self-play — enough for hazard consequences to appear in ~ every 4th game
  without letting a scripted opponent dominate the learned equilibrium.
- Collection keeps sampled (non-greedy) action selection so the net's own
  hazard moves get occasional probability mass early; no ε-hack unless the
  mid-run behavior probe still shows Spikes argmax pinned at 0, in which case
  add a small root ε-mix over *move classes* as a follow-up knob (design
  TBD, not in this arm).

## Shared yardstick and reads

Both arms run the standard 500k evaluation ladder (max-damage / simple-legal
/ random-legal / foul-play-100 at 10k cadence, high-fidelity at 50k) so they
are directly comparable to the four live arms (512d, 256d-h8, metamon-s-h1,
256d-1m continuation). Additional per-arm reads at 250k and 500k:

1. **ΔV probe** (`scripts/hazard_probe.py --checkpoint <ckpt>`): success for
   arm 1 = `value_self_hazard_response` clearly < 0 and > 10% of
   `value_spread` (vs ≤4% today). Success for arm 2 = Spikes
   `argmax_rate` > 0 with self/opp ΔV at least preserved.
2. **E1 calibration** (`scripts/e1_value_readiness.sh` candidates swapped to
   the arm checkpoints): shaping must not tank held-out Pearson/ECE of the
   terminal head — if it does, that is the phase-2 multi-γ trigger, not a
   reason to abandon shaping.
3. **Behavior probe** on switch rate / hazard usage vs the unshaped twin.

Kill criteria: at 250k, if arm 1 shows no ΔV movement AND ≥3-point
foul-play regression vs the 256d baseline band, stop it; same for arm 2 on
argmax + regression. (The four live arms hold the 3–9% foul-play band at
100k–160k; use the matching-milestone read, not endpoint.)

## Sequencing

1. Land the `status_delta_return_weight` change (public repo, small PR, unit
   tests beside the existing faint-delta tests).
2. Launch both arms when GPU capacity frees from the current four
   (deploy-side launch configs live in the private deploy repo, per policy).
3. E0-oracle (search go/no-go with the belief-1.5m head) proceeds in
   parallel — search reads do not gate these training arms, and a search win
   with a hazard-blind head still inherits ceiling from this work.
