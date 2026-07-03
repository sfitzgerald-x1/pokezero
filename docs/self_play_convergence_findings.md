# Self-play convergence & the LR-re-warm effect (fpdistill vs flagship)

Status: **findings note.** Empirical investigation of whether our self-play runs keep *learning* or
converge to a fixed point — and what the chained-continuation LR re-warm actually does. Complements
the LR-confound observations in [`foundation_belief_1_5m_results.md`](foundation_belief_1_5m_results.md)
and the myopia diagnosis behind [`mcts_design.md`](mcts_design.md).

## Motivating question
The fpdistill run's eval gains were small over 1.5M games (it started strong from the BC clone). Did
the model *materially* change, or did self-play converge to a (Nash-ish) fixed point and then orbit?

## Method
For checkpoints along a run's lineage we measured, on a fixed 701-state driver corpus:
- **Weight-space drift** — relative L2 `‖θ_end − θ_start‖ / ‖θ_start‖` per parameter group
  (encoder/trunk, policy_head, value_head, opponent-action head, embeddings).
- **Policy divergence** — mean `KL(π_start ‖ π_end)` over legal actions, argmax agreement, and policy
  entropy.

Caveats: (a) for **BC-seeded** runs (fpdistill), the value/opponent heads are *untrained* in the clone,
so their large drift is **RL-scaffolding initialization, not policy change** — read the policy via the
encoder+policy_head and the policy-KL. (b) The flagship's 1.5M→2.3M stage **crosses a chained-run
boundary with a ~2.4× LR re-warm**, so elevated drift there is expected and is the point of interest.

## Finding 1 — fpdistill froze into a fixed point by ~1M
BC clone → 1M → 1.5M (foul-play-distill line):

| group | clone→1M | 1M→1.5M |
|---|---:|---:|
| policy_head | 0.017 | 0.002 |
| encoder (trunk) | 0.311 | 0.031 |
| value_head | 0.764 | 0.047 | *(RL head built from scratch — not policy change)* |
| opp_head | 0.957 | 0.051 | *(same)* |

Policy divergence (701 states): argmax agreement clone-vs-1.5M **67.9%**; **KL(clone‖1M)=0.458,
KL(1M‖1.5M)=0.0075** (≈60× less); entropy clone 0.82 → 1.5M **1.15**.

Reading: self-play *did* move the policy off the imitation prior (32% of argmax decisions changed;
entropy rose — it softened the peaky BC policy and found its own equilibrium), but **essentially all
of that happened in the first ~1M games**, and the last 500k changed the policy almost nothing
(KL 0.0075). The `policy_head` barely moved *ever* (1.7%). This is the empirical signature of a
**self-play fixed point** — the policy stopped best-responding differently to itself. **The last ~500k
of the run was wasted**; fpdistill was effectively done by ~1M.

## Finding 2 — the flagship was converging too, until the LR re-warm re-activated it
Pure belief base (from-scratch → 1.5M) then the 1.5M→3M continuation:

| stage | encoder | policy_head | policy KL | argmax agree |
|---|---:|---:|---:|---:|
| 500k→1M | 0.089 | 0.064 | 0.091 | 82.9% |
| 1M→1.5M | 0.042 | 0.036 | **0.021** | **93.0%** |
| 1.5M→2.3M *(LR re-warm)* | 0.078 | 0.062 | **0.055** | 85.6% |

Reading: through 1.5M the flagship was **converging** — drift and policy-KL roughly *halved* each stage
(KL 0.091→0.021, agreement 83%→93%), heading into the same self-play equilibrium fpdistill hit. Then
the continuation's **~2.4× LR re-warm re-opened the step size**, and the policy started genuinely
moving again (KL 0.055, ≈2.6× the pre-continuation rate; ~14% of argmax decisions changed). This is
the LR sawtooth made concrete, and it matches the modest post-1.5M eval gains — so the 3M continuation
is doing **real (if modest) refinement**, not orbiting a frozen point.

## Synthesis
- **Self-play reliably converges to a fixed point by ~1–1.5M games.** Both runs show drift/policy-KL
  collapsing toward zero as they approach it (fpdistill hard-froze at 1M; the flagship was ~halving
  per stage toward it).
- **Chained-continuation LR re-warms buy more *in-basin* movement, not new capabilities.** The re-warm
  reactivates learning (flagship 1.5M→2.3M), which is why chained runs look like they "keep climbing"
  — but it's the same self-play basin. Expect the flagship to **re-freeze** as its LR re-anneals toward
  the 3M floor. (This is also the mechanism behind the earlier belief-vs-no-belief late max-damage
  "overtake": LR schedule, not capability.)
- **The fixed point is the *searchless* self-play equilibrium.** Neither run escaped into the
  hazard/setup/deep-foul-play lines (the measured myopia). More self-play games — or more LR sawtooth —
  cannot move a policy that's already at its self-play fixed point. **Escaping the basin needs a
  different signal**: test-time search (MCTS), dense reward shaping (cf. Metamon's hp/status/faint), or
  a coverage/opponent-demonstration curriculum — not more self-play or more capacity.

## Operational implications
- fpdistill's line was effectively done by ~1M; the 1.5M target over-spent for that run.
- For continuations, a single continuous LR schedule (no re-warm) converges and stays put; a re-warm
  buys a bounded window of additional in-basin refinement. Neither adds capability.
- Confirms the prioritization: the next lever is **search / shaping / curriculum**, and capacity or
  more self-play games will not break the myopia.
