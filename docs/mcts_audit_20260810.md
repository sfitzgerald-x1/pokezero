# MCTS audit vs standard practice — what's sound, what deviates, and when search overrides the model

Written 2026-08-10 from a source read of `rust/pokezero-search/src/{tree.rs,lib.rs,model.rs}`
and `src/pokezero/engine_search.py`, prompted by the owner's observation that deeper
search does not meaningfully move strength. Companion telemetry (the override rate)
lands in the same PR as this document.

## 1. What is standard or better — audited, keep

- **Decoupled per-side PUCT at decision nodes** (each seat maximizes its own PUCT,
  side two on `1 − v`): the standard treatment for simultaneous-move games (DUCT).
  Known theoretical caveat — DUCT converges to coarse correlated rather than Nash
  equilibria in some games — but it is the textbook choice and not a defect.
- **Exact-expectation backup through chance nodes** (`tree.rs finalize`): every
  enumerated outcome priced at expansion; backups propagate `Σ p_k · mean_k`, so the
  chance layer contributes zero sampling variance. This is *better* than the sampled
  backup most implementations use.
- **Value orientation**: side-one win probability everywhere, seat flip only inside
  the selection formula, pinned by the mirror/parity test battery (the §10 findings
  fixed the one real orientation bug; the pins keep it dead).
- **No root Dirichlet noise**: correct — noise is a *training-exploration* device;
  these are eval-time decisions.
- **Final selection by aggregated visit share across belief worlds, argmax**:
  standard robust-child selection, marginalized over sampled worlds (ensemble/root
  determinization), with collapse multiplicity weighing correctly (#1009).
- Determinism per seed; early-stop only on a mathematically uncatchable argmax.

## 2. Deviations from standard practice, ranked by suspected strength impact

### 2.1 Opponent priors are OFF — and this is the leading suspect for flat depth

`MultiPlyConfig.use_opponent_priors` defaults false, so **every opponent node in the
tree explores uniformly**. The machinery to fix it exists (#1192 gather/apply,
#1207's gate) and has never been enabled in a strength campaign.

Why this flattens the depth curve, mechanically: with ~9 legal arms per seat, one
ply of joint actions is ~81 edges. Uniform opponent priors mean no knowledge prunes
the opponent half, so sims-per-subtree decays by ~an order of magnitude per ply:
at s1024/d4 the plies past the second see ~1 visit each. **Raising `max_depth` adds
legal depth the budget cannot populate — the tree becomes a wider bush, not a deeper
line.** AlphaZero never faces this because its policy prior prunes *every* node;
ours prunes only our own seat's half. The depth ladder being flat while the
forced-win suite (constructed positions, #966) shows depth *works* is exactly this
signature: depth helps when the line is narrow enough to populate.

**Recommendation:** a paired probe with `use_opponent_priors=true` at d4/s1024 —
one cell, 100 games, same seeds — is the single highest-information experiment
available. If the axis study is already running, it slots in immediately after on
the same node.

### 2.2 First-play urgency is a constant 0.5 — non-standard, interacts badly with 2.1

`MoveStats::mean()` returns 0.5 for unvisited arms. AlphaZero-family engines
initialize unvisited Q from the *parent's* value minus a reduction (LC0's
"FPU reduction"), because a constant optimistic floor misbehaves at both ends:
**behind** (parent Q < 0.5), every junk arm looks better than everything explored,
so the budget floods wide exactly when the search should be verifying the one
defensive line; **ahead** (parent Q > 0.5), unvisited arms look artificially bad, so
the search narrows prematurely and misses refutations. Combined with uniform
opponent priors, a losing-side search spends its budget proving junk opponent
replies are junk. Cheap, flag-gated change; probe it after 2.1.

### 2.3 Minor, tune only after 2.1/2.2

- `c_puct` fixed at 1.4 with no visit-scaled growth term (AZ's log schedule) —
  plausible tuning target, not a defect.
- Damage-roll branching collapses past ply 2 except KO-straddles (`deep_ko_split`) —
  engine-inherited, bounded bias, already documented and detector-gated.
- No transposition merging — standard for AZ-style trees; cost is compute, not bias.

## 3. When does search override the model? (the owner's question, answered from code)

The model enters twice: as **priors** (root and child self-priors — exploration
allocation only, values never touched) and as **leaf values** (the TorchScript
forward pricing non-terminal branches). The final choice is the argmax of
visit shares aggregated across worlds. So the chosen move differs from the model's
own argmax exactly when **backed-up values reallocate visits away from the top
prior**, which requires one of:

1. **In-tree tactical facts** — terminal branches price exactly 1.0/0.0 (a found KO
   or loss), the strongest override signal and the one thing the prior cannot see;
2. **Consistent cross-world value separation** — leaf values must contradict the
   prior ranking *in the same direction across belief worlds*; disagreement that
   averages out across worlds correctly does not move the aggregate (that is
   marginalization working, not a bug);
3. **Enough budget for Q to beat U** — at fixed sims with a sharp prior, PUCT visit
   allocation approximately follows the prior unless the Q gap is large; overrides
   are therefore rarer at low sims and with confident checkpoints, by construction.

Note the asymmetry this implies: search adds most over raw policy where terminals
are reachable within the populated depth (endgames, KO races) and least in quiet
midgame positions — consistent with the paired-eval deltas being positive but small.

## 4. The override-rate telemetry (this PR)

Per searched decision on the model path, the policy now compares the final choice
against `model_prior_argmax` — the acting seat's per-arm crate priors aggregated
over the *same records* as the visit aggregate (collapse multiplicity identical),
same deterministic tie-break — and reports:

- `stats.search_override_decisions` / `stats.search_override_unmeasured` (shard
  telemetry; rate = overrides / (searched − unmeasured));
- per-decision `model_argmax`, `search_argmax`, `model_override` in the
  `engine_mcts` metadata block (`None` = unmeasured, never coerced to agreement).

Reports from crate builds predating the per-arm `prior` field, and the
hp-fraction/legacy paths (uniform priors — "the model's argmax" does not exist
there), count as **unmeasured**. The crate change is one serialized field
(`prior`) in the per-arm stats JSON.

What to expect and how to read it: override rate is a *diagnostic*, not a KPI.
Near-zero at production config would confirm §3's prediction that search is mostly
reproducing the prior (and make 2.1/2.2 the lever); a healthy rate with flat
strength would instead say overrides are frequent but not *better* — pointing at
leaf-value quality, which is a different program. Read it alongside win rate on
override-vs-agree turns before concluding either way.

## 5. Suggested sequence

1. Land this telemetry; the axis study's cells produce the first override-rate
   numbers for free.
2. The 2.1 probe (`use_opponent_priors=true`, one cell, paired).
3. The 2.2 FPU-reduction probe, flag-gated, only after 2.1 reads out.
4. Revisit `c_puct` and selection-formula tuning only with 2.1/2.2 settled.
