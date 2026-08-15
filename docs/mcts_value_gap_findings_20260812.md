# Why doesn't search beat the model? — findings

Written 2026-08-12, answering `docs/mcts_value_gap_investigation_20260811.md`.

**The answer, in one line: it does — when the in-tree opponent is priced from the
model. With opponent priors off, search at maximum budget is indistinguishable from
the raw policy (−1.50 pp, p = 0.83). With them on, it beats raw by +10 to +14 pp.
The measurements that produced the premise were taken with opponent priors off.**

Cluster specifics — image digests, the pinned node, absolute storage paths, per-shard
artifacts — live in the private deployment repo under
`mcts/mcts-value-gap-20260812.json` and `reports/mcts-value-gap-20260812/`, not here.

---

## 0. The cells

One image (`8122bcc6`, engine fingerprint `82ba92a4`), one node, one checkpoint
(`v4prod-entfull-15m-20260807084357` iteration-1563), one seed band
(23000000–23000124, paired). Every load-bearing contrast is within this build.

| cell | games | override rate | ceiling share at D−1 | fallback | s/decision |
|---|---|---|---|---|---|
| oracle belief, priors ON | 230 | 25.98% | 59.73% | 0.411% | 13.36 |
| sampled belief, priors ON | 240 | 25.42% | 60.98% | 0.333% | 13.59 |
| sampled belief, priors OFF | 210 | 24.72% | 51.78% | 1.128% | 12.98 |
| priors ON at s2048 | 210 | 14.37% | 13.86% | 0.110% | 1.83 |
| raw policy (no search) | 240 | — | — | — | — |

Three shards were lost to opponent-process crashes (`foul-play exited with status 1`,
4 occurrences, ~1% of games); two cells stand at 21/25 because the wave was read
before its long tail finished. All comparisons are paired on the (seed, seat)
intersection, so unequal cells cost power, not validity. Exact n travels with every
figure.

---

## 1. The verdict

**The binding constraint is the in-tree opponent model — specifically whether the
opponent seat is priced from the model at all.**

| comparison | paired n | rates | delta | discordant | McNemar p |
|---|---|---|---|---|---|
| priors ON vs raw, s2048 | 210 | 55.24% vs 41.43% | **+13.81 pp** | 52 / 23 | **0.0011** |
| priors ON vs raw, s16384 | 240 | 54.17% vs 44.17% | **+10.00 pp** | 62 / 38 | **0.021** |
| **priors OFF vs raw, s16384** | 200 | 42.50% vs 44.00% | **−1.50 pp** | 40 / 43 | **0.826** |
| priors ON vs OFF, both s16384 | 200 | 51.00% vs 42.50% | +8.50 pp | 47 / 30 | 0.068 |
| oracle vs sampled belief | 230 | 56.96% vs 53.91% | +3.04 pp | 55 / 48 | 0.555 |

Read the third row against the first two. At **maximum** budget — 16,384 sims, depth 8,
60% of decisions reaching the depth ceiling — search without opponent priors does not
beat the raw policy at all. Turn priors on and the same search wins by 10 pp; turn them
on at **one-eighth the budget** (s2048, 1.83 s/decision against 13.59) and it wins by
13.81 pp, the most significant result in the campaign.

**So budget was never the constraint.** Cheap search with a priced opponent beats
expensive search without one, decisively. And the premise this investigation opened
from — search +0.03 to +0.06 over raw, not significant, at d4/s1024/w4 — was measured
with opponent priors **off**. That is why search looked worthless: it was.

### The single recommended next lever, and its cost

**Ship opponent priors on by default, and witness the applied counters in production.**
The flag already exists and its driver gate was cleared in a parallel goal; the change
is a default plus a witness, not new machinery. Cost: effectively zero compute — the
s2048 cell shows the win does not require the expensive configuration. **Expected gain:
+10 to +14 pp against FoulPlay**, larger than every effect the preceding
selection-tuning panel chased across four mechanisms.

Second lever, only after the first: improve *opponent-model quality*. This campaign
made the supervision available (`--opponent-journal full` records every FoulPlay move,
~8,300–10,500 labelled decisions per cell). Before scoping it, run the cheap
prerequisite in §5.

---

## 2. The hypothesis table

| # | hypothesis | verdict | measurement |
|---|---|---|---|
| H1 | Search mostly reproduces the prior | **REFUTED** | Override rate **25.42%** (2729/10,736 measured; 36 unmeasured) at max power; **14.37%** (1301/9055) at s2048. Search departs from the model's plain argmax on one decision in four. |
| H2 | The value head cannot separate the candidate moves | **SUPPORTED, and bounded** | Top-1 vs top-2 root Q gap **0.0181**, stable at 0.0170–0.0294 across every cell regardless of belief quality or calibration. The two best moves differ by ~1.8 pp of win probability, so perfect discrimination between them is worth at most ~1.8 pp. |
| H3 | Overrides happen but are not better than the model's move | **CANNOT RUN** | The §4b fork probe has no implementation and the blocker is structural — see §4. Confounded colour only: override rate 25.20% in wins against 26.40% in losses. |
| H4 | The in-tree opponent is the wrong opponent | **SUPPORTED — and it is the binding constraint** | Two independent readings. (a) Strength: priors ON beats raw by +10 to +14 pp; priors OFF does not beat raw at all (−1.50 pp, p = 0.826). (b) Prediction: the in-tree opponent matches FoulPlay's actual submitted move **28.9%** of the time (2847/9849) against the raw prior's **15.6%** (1522/9735) — search improves its opponent model and stays wrong ~7 times in 10. Caveat in §3. |
| H5 | Belief-world error dominates: search optimizes a fiction | **NO LARGE EFFECT DETECTED** | Oracle vs sampled **+3.04 pp**, n = 230, discordant 55/48, **p = 0.555**, 95% CI ≈ [−5.6, +11.8] pp. A perfect belief improves value **calibration** 3.3× (ECE 0.0547 → 0.0165) and moves the Q gap by 0.0001. Belief error is real, measurable, and does not convert to strength. |
| H6 | Engine-fidelity residue misleads deep lines | **DISPOSED, no compute** | By inspection of the residue ledger, as the plan directs: single-digit rows, all dispositioned, none touching the value head or belief model. Recorded so the report can say it was considered. |

### The mechanism, assembled

Opponent priors move four things at once, and the pattern is coherent:

| priors OFF → ON | change |
|---|---|
| ceiling share at D−1 | 51.78% → 60.98% |
| fallback rate | 1.128% → 0.333% (3.4× lower) |
| in-tree opponent's agreement with FoulPlay | 26.5% → 28.9% |
| **strength against raw** | **−1.50 pp → +10.00 pp** |

Without priors the opponent seat is unpriced, so sims spread across an unranked
opponent action set, the tree reaches its ceiling less often, refusals triple — and the
whole search buys nothing. The audit (`docs/mcts_audit_20260810.md`) named this
mechanism from telemetry alone; this campaign shows it converts to strength.

**Why H2 is real but secondary.** The Q gap sits at ~0.018 in every cell, including the
oracle arm where the value head's calibration error is 0.0165 — i.e. where the gap
finally *exceeds* the error. Fixing belief fixes calibration 3.3× and still buys no
wins. So the ceiling on move-discrimination is not noise: **the candidate moves are
genuinely close in value**, and any mechanism that sharpens which arm wins the argument
is bounded by ~1.8 pp. That is a decision-theoretic ceiling, and it explains the
preceding panel's four-mechanism null in one line. Search's real value lies elsewhere —
avoiding the raw policy's occasional much-worse choices, worth 10–14 pp.

---

## 3. Caveats that change how the numbers should be used

**H4's prediction number is not self-evidently a defect, and its ceiling is
unmeasured.** The in-tree opponent is a PUCT-optimal *best response* seeded with our own
priors; minimax deliberately models a strong opponent rather than the actual one, so low
agreement is partly by design. The real ceiling is FoulPlay's agreement with *itself* —
it is time-budgeted over an unseeded RNG and demonstrably does not reproduce its own
play. If that ceiling is ~60%, 28.9% is half of achievable; if ~90%, it is a genuine
deficiency. **Measure it before scoping opponent-model work** (§5).

**The priors ON-vs-OFF contrast is marginal on its own** (+8.50 pp, p = 0.068). The
strong evidence is the pair of independent comparisons against the shared raw arm:
priors ON separates (p = 0.021 and p = 0.0011), priors OFF does not (p = 0.826). Read
the conclusion off that pattern, not off the single ON-vs-OFF p-value.

**Two analyses were run, published, and retracted.** Recorded because the reasoning is
the reusable part.

*Retracted 1 — Brier across gap quartiles.* Rising Brier (0.1181 → 0.1704) was read as
"larger gaps are worse predicted". Invalid: for a calibrated forecaster Brier = p(1−p),
maximised at 0.5, and the large-gap bucket's predictions sit nearer 0.5, so it carries a
higher irreducible floor before accuracy enters. A base-rate comparison dressed as a
quality comparison.

*Retracted 2 — "confident overrides do worse".* The larger-gap half of overrides won
50.1% against the smaller-gap half's 56.4%, glossed as confidence predicting failure.
That is a causal claim about decisions drawn from a comparison of different positions.
Matching on the model's own predicted win probability narrows it (pooled −2.31 pp) but
cannot rescue it, because the matching covariate is the quantity under suspicion. The
better-supported reading is the opposite: search deviates precisely when it sees what
the raw head missed, so the override is a *marker* of value-head error rather than a
cause of loss.

*Retracted 3 — the oracle arm's fallback rate.* Reported at 4 shards as **lower** than
its sampled twin (0.238% against 0.350%); at 23 shards it is **higher** (0.411% against
0.333%). Both small, neither indicating a degraded arm, but a favourable early snapshot
should not have been stated as a result.

**§2's three denominators.** Override rate carries numerator, denominator and unmeasured
count throughout. The plan's second denominator asks for `opponent_priors_applied` vs
`refused` per seat: **no field of either name exists** in these artifacts. The nearest
real quantity is `opponent_prior_arm_decisions` — the count of decisions on which an
opponent prior arm was established — reported under that name and not as the plan's
measurement. Realized depth is reported as ceiling share at D−1, never as "cap
saturation", which is a definitional zero (`rust/pokezero-search/src/tree.rs:487/553`).

**The oracle arm was verified applied, not merely requested.**
`oracle_belief_decisions` = 10,452 = its decision count exactly: the true world was
searched on every decision. That gate matters because a truth world that cannot be built
takes production's ordinary fallback path rather than raising, and since the true
override is identical on every attempt, one refusal would refuse *all* worlds for that
decision.

---

## 4. What could not be measured

**§4b, the counterfactual fork probe — CANNOT RUN.** The bridge has no
`--replay`/`--fork`/`--resume` flag and no replay entry point, and the blocker is
structural rather than a missing feature: FoulPlay is time-budgeted over an unseeded
`rand::rng()`, so a replay cannot reproduce its moves and the trajectory diverges long
before the fork round. The plan's own fallback — self-play forks, defender not FoulPlay
— is equally unbuilt. H3 therefore has no clean verdict.

Worth recording: `--opponent-journal full`, added in this campaign for H4, is the
missing enabler. It records FoulPlay's submitted choice per round, which is exactly what
a pinned-trajectory replay would force to reconstruct the state at round R. The probe
moved from *impossible* to *one bridge mode away*.

**Three legs the plan expected to be free were not.** §1 asserts override telemetry
exists as #1235 and §5 schedules H1's production rate as a read from banked cells:
#1235 never merged and no banked shard carries a single override field — that leg was a
run, and the s2048 cell is it. §4a names `--belief-start-overrides` as the truth
machinery: it is a *sampled* PIMC planner whose own docstring says it never inspects the
opponent's private observation, and the engine-MCTS branch never reads it — built as
written, §4a would have compared two sampled arms. And H4 was mechanically dead: the
bridge decodes FoulPlay's move every round, but the default journal mode retains it only
for battles carrying a refusal address, so the canary read `recorded_decisions: 35`
beside `emitted_decisions: 0`. All three were found before the fleet launched.

**Not independent seeds.** 23000000–23000124 is a deliberate reuse of the
selection-tuning panel's primary band. Every number here is a paired within-seed
contrast; nothing may be read as measured on unseen seeds.

**Not comparable across builds, except one checked leg.** The banked arms carry
fingerprint `209a70af`. The only cross-build read attempted was raw-against-raw as a
consistency check: 106/240 = 44.17% here against 100/198 = 50.51% banked, difference
−6.34 pp, z = −1.32, **p = 0.186**, intervals overlapping. It passes, with low power —
"failed to reject", not "identical". The +10 pp figure is within-build and paired and
does not depend on it; what the 6 pp gap tempers is confidence in absolute levels, so
read search-over-raw as roughly +5 to +12 pp rather than exactly +10.0.

**`prior_fallbacks` is never converted to a rate.** It has no denominator in these
artifacts — a per-node count summed over rounds, which has exceeded its own shard's
decision count. Reported as a count only.

---

## 5. The cheap prerequisite before any opponent-model work

Measure **FoulPlay's self-agreement**: replay one position through FoulPlay N times and
count how often it picks the same move. It is time-budgeted over an unseeded RNG, so
this is its irreducible entropy and therefore the ceiling on any opponent predictor.

It costs minutes and it sizes the prize. The reference points that make it
interpretable:

| predictor | top-1 agreement with FoulPlay's actual move |
|---|---|
| random over legal actions (~9–10 typical) | ~10–11% |
| raw model prior | 15.6% (1522/9735) |
| in-tree opponent after search, priors ON | 28.9% (2847/9849) |
| in-tree opponent, oracle belief | 33.7% (3222/9569) |
| **FoulPlay against itself** | **unmeasured — run this first** |

Without that number, 28.9% cannot be called good or bad and an opponent-modelling
project cannot be sized. With it, the second lever in §1 becomes a costed decision.
