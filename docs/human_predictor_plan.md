# Human predictor: an opponent-decision model for MCTS search

Status: plan for review, 2026-07-15. Agent-executable; phases carry pre-registered
metrics, gates, and an explicit kill criterion. Includes an honest assessment of
whether this line is worth pursuing at all (owner requested criticism, not just a
plan).

## The thesis

Search currently imagines its opponent is itself: the opponent-action scenarios in
the MCTS tree (`root_opponent_action_scenarios` / candidate scenarios in the
capstone plan) are drawn from the same self-play policy that makes player-side
decisions. The proposal: keep the self-play model for OUR decisions, but predict
OPPONENT decisions with a model trained on human play. If human opponents
systematically differ from our policy — over-statusing, over-switching,
protect-scouting, mispricing endgames — a search that anticipates *actual* human
lines prices its branches against the opponent it really faces, instead of a
mirror.

## Is this a valid path? (assessment first, plan second)

**The case for:**

1. The opponent model is the largest unexamined free parameter in our search. We
   invested heavily in the value head (Step 0 calibration) and the prior (the
   policy itself), but the opponent distribution has always been the mirror
   assumption, untested against any real opponent population.
2. Gen 3 is a *stable* meta — a 20-year-old format. Human data doesn't rot; a
   predictor trained this year stays valid.
3. Precedent: FoulPlay (the strongest scripted baseline we face) is itself built
   on human-crafted opponent heuristics, and the fpbc lineage proved we can
   behavior-clone an external agent into our architecture.
4. The mechanism is cheap to test: prediction quality on held-out human decisions
   is measurable offline, before a single search game is played.

**The case against (each shapes the plan below):**

1. **Objective mismatch.** Our binding evaluations are paired games vs max-damage
   and FoulPlay — neither is human. A human-opponent model can be *correct* and
   still not move those numbers. If the deployment target is the eval gauntlet,
   the honest version of this idea is an **opponent-conditional predictor
   validated first on FoulPlay** (whose games we can generate in unlimited
   quantity); if the target is ladder play vs humans, we currently run no ladder
   evaluation and standing one up is its own program.
2. **Exploiting ghosts.** Best-responding in-tree to an average-human model that
   blunders makes search *worse* against opponents who don't. Value backups
   assume the opponent takes the modeled line; a strong opponent punishes the
   difference. Mitigation is built into the design: the human predictor enters as
   a **blend** λ with the self-play prior (λ=0 recovers today's search), never as
   a wholesale replacement, and λ is selected by paired games, not by faith.
3. **Data thinness and selection bias.** Measured 2026-07-15: gen3randombattle
   uploads run ~30/day currently (~5/day in 2020), all replays are opt-in
   uploads (skewed toward games players chose to save), and ~84% of a sampled
   page had null ratings. Order-of-magnitude corpus estimate: 10⁴–10⁵ replays ≈
   10⁶ decision labels — fine-tune territory for a 10M model, nowhere near
   from-scratch territory. By contrast gen9randombattle runs ~175 uploads/hour,
   and the Metamon project (UT-Austin) maintains 2.68M raw replays on Hugging
   Face — but for OU/NU/UU/Ubers tiers and Gen 9 OU, **not** random battles.
4. **An average-human model is not the opponent in front of you.** Predicting the
   population mean forfeits per-opponent adaptation. (Extension hooks exist —
   rating-conditioning tokens, and the encoder's `tendency_stats` per-mon
   tendency features — but they're out of scope for v1.)

**Verdict:** valid as a *conditioning experiment with a kill criterion*, not as a
committed bet. The plan's spine: measure whether humans actually play differently
from our policy (cheap, offline, decisive); build the opponent-model plumbing once
and validate it with a FoulPlay predictor where data is unlimited; only then spend
the scarce human data. If humans turn out to be predictable by our own prior, the
program stops at Phase H2 with a one-page negative result.

## Data sources (verified 2026-07-15)

| source | what | volume | fit |
|---|---|---|---|
| Showdown replay API (`replay.pokemonshowdown.com/search.json?format=gen3randombattle`) | official JSON API; 51/page, paginate with `before=<uploadtime>`; `.json`/`.log`/`.inputlog` per replay | ~30/day now, ≥2020 history; total TBD in H0 | primary corpus |
| Same API, `gen9randombattle` (and gen1/2/4 randbats) | adjacent-format human randbats decisions | gen9: ~175/hour | optional pretraining/transfer |
| `jakegrigsby/metamon-raw-replays` (Hugging Face, v6 2026-05) | 2.68M parsed-ready replays, anonymized; OU/NU/UU/Ubers gens 1–4 + gen9 OU | large | optional generic-human pretraining; no randbats; license unstated — verify before use |
| Own collector (new) | cron polling the search API, storing raw replay JSON on shared storage | grows ~30/day + backfill | keeps corpus current |

Collection etiquette: official API only (the client docs explicitly say don't
scrape HTML), ~1 req/s rate limit, resumable `before` cursor, public replays only
(`private: 0`), usernames anonymized at parse time with battle-consistent
pseudonyms (Metamon precedent). Raw data lives on cluster storage; nothing scraped
enters the public repo.

## Phases

### H0 — corpus inventory + collector (no GPU)

Full pagination sweep of gen3randombattle: total replay count, uploads/year,
rating histogram, games/replay length distribution. Same counts (one page each)
for gen1/2/4/9 randbats to size the transfer option. Stand up the collector cron.
Deliverable: `human-predictor/inventory.json` + collector running.
Gate G-H0: corpus ≥ 20k replays (≈1.5M decisions) OR an explicit owner decision
to proceed with transfer-heavy training below that.

### H1 — replays → decision dataset

Parse each replay from BOTH players' perspectives into (public observable state,
human action) pairs using the existing replay-import path and the public-replay
action bridge (PR #629) — the same machinery that materializes replay states for
the hazard audit, so observation encoding is byte-identical to what our models
train on. Splits: held-out test by **anonymized player id** (no player appears in
both train and test) *and* by time (newest 3 months held out) — either leak
inflates accuracy. Dedup exact replay ids across sources.
Gate G-H1: parse yield ≥90% of replays (gen3 protocol edge cases documented, not
silently dropped); split leakage audit passes.

### H2 — the decisive cheap measurement (kill criterion lives here)

Score on held-out human decisions, no training yet:

1. Zero-shot self-play prior (v2.2 2M checkpoint) — NLL, top-1, top-3, ECE,
   split by game phase and by move-vs-switch.
2. Trivial baselines: uniform-legal, max-damage-heuristic.

Then fine-tune the predictor (H3 arms) and compare. **Kill criterion,
pre-registered:** if the best fine-tuned predictor beats the zero-shot self-play
prior by <0.10 nats NLL and <3 points top-1 on held-out human decisions, humans
are already well-modeled by our own prior; the human-predictor line terminates
with a negative-result writeup. (The zero-shot gap itself is publishable insight
either way: it quantifies "how differently do humans play from our agent" for the
first time.)

### H3 — predictor training (arms, all initialized from the v2.2 checkpoint)

- **A: head-only fine-tune** (freeze trunk) — the sample-efficiency floor.
- **B: full fine-tune,** low LR — the capacity ceiling on thin data; watch
  overfit via player-split test.
- **C: residual logit correction** — predictor = self-play logits + small learned
  correction; the blend-friendly parameterization, degrades gracefully.
- Optional **D: gen9-randbats pretrain → gen3 fine-tune** if G-H0 came in thin.

Report the H2 metric battery per arm. Calibration matters as much as accuracy —
search consumes the *distribution*, not the argmax.
Gate G-H3: winning arm clears the kill criterion and ECE ≤ 0.05.

### H4 — into the search (plumbing validated on FoulPlay first)

1. **Plumbing + FoulPlay validation:** make the opponent-scenario distribution a
   pluggable model with blend weight λ (λ=0 ≡ today). Validate the mechanism
   end-to-end with a **FoulPlay predictor** trained on our own eval games
   (unlimited data, and it predicts an opponent we actually measure against).
   If opponent-conditioning can't improve search vs FoulPlay *with a FoulPlay
   predictor*, it won't improve anything — this is the mechanism's own kill test.
2. **Offline tree studies:** fixed-position suites; measure value shift and an
   exploiting-ghosts diagnostic (tree value vs deep-search ground truth — value
   inflation under the human model flags over-exploitation).
3. **Probe-scale paired games** via the directional-probe machinery (200
   mirrored games per opponent, fresh seed reservation): search+λ-blend arms vs
   max-damage and FoulPlay. Expectation set honestly: the human predictor may
   show nothing here (criticism #1); the FoulPlay-predictor arm is the one that
   should move these numbers. A human-ladder evaluation is explicitly out of
   scope and needs its own decision.

## Compute

Trivial next to training runs: H0–H2 are CPU + one GPU-day; H3 arms are
fine-tunes of a 10M model on ~10⁶ examples; H4 probe reuses directional-probe
sizing. The only standing cost is the collector cron.

## Hygiene

Public repo: code and the plan only. Cluster storage: raw replays, parsed
datasets, checkpoints. Anonymize player names at parse; keep replay ids for
dedup/provenance. Metamon datasets: verify license before any use. No private
cluster details in the public repo, as always.

## Decision points for the owner

1. Approve the collector cron (external API traffic, ~1 req/s).
2. G-H0 threshold call if the corpus lands under 20k replays.
3. H4 step 3 arm list (which λ values buy games).
4. Whether a human-ladder evaluation program should ever exist (separate doc if
   yes).
