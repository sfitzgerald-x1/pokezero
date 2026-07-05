# Observation compression: opponent-signal stats replace the 4-turn window

> **CORRECTIONS LAYER (2026-07-04, dual-review — AUTHORITATIVE, supersedes
> conflicting body text below; full editorial restructure lands with the
> implementation PRs):**
>
> **Mechanics (engine-verified):**
> 1. The randbats spread is NOT fully deterministic: the generator zeroes
>    Atk EV/IV on no-physical-attack sets and trims HP EVs on
>    Sub+Flail/Reversal, Sub+pinch-berry, and Belly Drum sets. Def/SpA/
>    SpD/Spe stay exact 85/31/neutral; **HP and Atk are
>    variant-conditioned**. Computed-stats features carry exact values
>    for the four fixed stats and per-variant values (or bounds) for
>    HP/Atk; residual margins absorb the ±few-% HP-denominator ambiguity.
> 2. Pinch berries activate **end-of-turn only** in gen3 → non-proc rule
>    3 prunes only after an end of turn at ≤25%, never mid-turn.
> 3. Drain-vs-sub reconstruction is **±1 HP** (heal = ceil(damage/2)),
>    not exact.
> 4. **Pursuit accuracy SETTLED**: the engine bypasses accuracy against a
>    switching target — removed from the Tier-2 open list.
> 5. Solar Beam is halved in rain, **sand, and hail** (sand reachable).
> 6. Wish heals 50% of the **recipient's** max HP.
>
> **Spec decisions closing review P0/P1s:**
> 7. The Tier-2 CB specification (whitelist, max-explanation margin,
>    two-strike, precision gate) is canonical; the exact-state section's
>    single-exceedance version is STRUCK.
> 8. The Rest flag is **candidate-conditioned on Early Bird** (5 reachable
>    carriers): wake known-2 iff Early Bird absent from candidates;
>    ambiguous {1,2} otherwise; a 1-turn Rest-wake confirms Early Bird
>    and restores determinism thereafter. The sleep-clause bit has LIVE
>    semantics (clears on wake/faint; tracks the currently-slept mon).
> 9. Canonical Tier-1 token field list (closes the schema): actor, action,
>    **`called` bit** (Sleep Talk executions — new, same mislearning class
>    as `transformed`), `transformed` bit, damage fraction,
>    `damage_outcome` enum {normal, blocked, immune, absorbed, hit-sub,
>    broke-sub, endured}, flags {crit, miss, KO, `pursuit-intercept`},
>    `n_hits`, effectiveness class, side-effect category, context trio
>    {own layers, opp layers, weather}, positional pair {absolute turn,
>    turns-ago}. Tier-2 reserved zero-masked slots: residual scalar +
>    validity bit, CB bit, investment bit — same spec version, no second
>    break. *(Implementation note, 2026-07-04: all four slots are now
>    materialized — residual 117 / validity 118 / CB bit 119 populated
>    behind the tier2 gate + mask; investment 120 populated by the
>    defender-side inference (`pokezero.investment`) behind its own
>    precision gate (`runs/investment-gate-2026-07-04`, PASS) and the
>    separate `tier2_investment` mask, which defaults OFF until v2.1
>    training adopts the column — pre-v2.1 encodes stay byte-identical.)*
>    *(Implementation note, spec v2.1 — `pokezero.observation.v2.1`,
>    checkpoint-driven dual schema, NOT a one-way break: the schema +
>    numeric width an env encodes resolve from the loaded checkpoint's
>    model_config, so live v2 runs keep scoring on main while fresh
>    trains stamp v2.1. Batch 1 adds (a) defender identity on move
>    transition tokens in the CATEGORY_MOVE_PRIORITY slot, unused on
>    transition rows — the defender is inferable from interleaved switch
>    tokens EXCEPT when K-truncation drops the anchoring switch, and
>    damage_fraction is defender-relative; (b) per-bucket revealed-move
>    PP-validity bits (columns 121–136, mirroring the PP-fraction
>    buckets), closing the revealed-at-0-PP collision; (c) the active
>    mon's substitute HP fraction (column 137) as presence + the
>    engine-verified initial floor(maxhp/4) — sub chip is not
>    protocol-derivable (the surviving-hit `-activate` carries no
>    magnitude; the drain heal leak is Tier-2 residual territory);
>    (d) per-mon PINNED Tier-2 conclusions on the opp-mon token surface
>    (CB pinned 138, investment pinned 139 — the current-state belief
>    channel, authoritative and switch-persistent; the tt-row cb_bit
>    stays the as-of-strike history record, self-describing under
>    K-truncation). Tier-2 conclusions never mutate the Tier-1
>    candidate sets — layer separation holds. Batch 2 populates BOTH
>    investment surfaces behind the `tier2_investment` mask under the
>    v2.1 schema ONLY: the per-mon pinned form (139, derived from the
>    annotated stream like CB pinned — the authoritative current-state
>    conclusion) and the tt-row as-of-strike code (120). Column 120
>    sits below the v2 census end, but the write is schema-gated on
>    top of the double mask, so the legacy v2 encode path never
>    touches it and v2-mode encodes stay byte-identical
>    unconditionally.)*
> 10. Residual encoding: signed fraction of defender max HP (observed
>    minus expected-median under the candidate-conservative baseline),
>    with a separate validity bit (masked ⇒ invalid, value 0); populated
>    on opponent attacks only in base Tier 2; margins ≥ protocol
>    quantization (1% HP) — "tight" bounds are tight only for our own
>    mons' exact HP.
> 11. Token emission rules: one token per *declared action* (move or
>    switch), so faint-replacements, Baton Pass completions, Pursuit
>    interceptions, and turn-1 lead send-outs each emit their own switch
>    token; **K is specified in tokens (128), not turns**; `turns-ago`
>    ties break by within-turn resolution order.
> 12. The ablation matrix re-anchors control arms at **512d** (the width
>    result makes 256d controls uninformative); all arms carry the
>    exact-state layer (isolating history encoding); the midground stat
>    ships **Tier-2-gated**, not in Tier-1 arms; E/C train at K=64 with
>    K=16 as the masked ablation.
> 13. The protect-pattern / prediction-channel observations route into
>    the midground stat's existing counters (no new dedicated stats —
>    subsumption principle holds): `blocked`-on-our-attack and
>    typing-explained-immune increment "no-predict"; doubled Pursuit
>    increments "predict".
> 14. Status-move clicks and no-action turns (`|cant|`: sleep, para,
>    flinch, recharge) emit tokens with action id + outcome; the
>    immune typing-split's type-chart computation is **Tier-2-gated**
>    with a fixture test (per the hard-rule asymmetry).
> 15. Trick × pruning: post-swap the recipient's item is **known**
>    (identity of the given item), variant pruning on the *original* item
>    freezes at its pre-swap state; non-proc rules apply to the current
>    known item only.

> **TURN-MERGED ADDENDUM (2026-07-05, v2.1 batch 3 — supersedes item 11's
> one-token-per-action emission for the turn-merged mode; the per-action
> extraction remains available and unchanged. Landed as observation schema
> **v2.2**, the third entry in the #512 checkpoint-driven dual-schema table:
> v2/v2.1 artifacts stay first-class, the stamped checkpoint schema selects
> the encode, and v2.1 remains the checkpoint-free default until the
> turn-merged ablation earns the slot):**
>
> **Schema.** One token per TURN carries the two DECLARED actions as
> ordered sub-blocks: first mover / second mover — resolution (speed)
> order becomes EXPLICIT structure instead of positional convention.
> Each sub-block carries the item-9 action fields (kind, move-or-species,
> called, transformed, damage fraction, outcome enum, crit/miss/KO/
> pursuit-intercept, n_hits, effectiveness, side-effect, Tier-2
> residual+validity+CB reserves). The context trio and the positional
> pair are stored ONCE per token (captured at the first sub-block's
> declaration). A sub-block is NEGATED when the side's declared action
> was consumed with zero protocol trace, and ABSENT when no declaration
> was expected (the empty half of a single replacement token).
> RestTalk's three protocol lines collapse to one sub-block
> (called-execution + `cant_reason`); Baton Pass completions collapse to
> `baton_pass_species`; both resynthesize exactly on flatten.
>
> **Engine-verified dispositions (vendored gen3 sim, 2026-07-05; live
> tests in `test_turn_merged_engine.py`):**
> - *Hazard sack*: a switch-in that faints to Spikes pauses the turn
>   (forceSwitch/wait); the opponent's declared action FIZZLES ENTIRELY —
>   no `|move|` line, no redirect to the replacement, even for
>   non-targeted moves (a declared Spikes layer also vanishes). Hazard
>   sacking is a true free pivot; the NEGATED sub-block is what makes it
>   learnable.
> - *Explosion/Selfdestruct double-faint*: both replacements are ONE
>   engine forceSwitch cycle, emitted back-to-back before `|upkeep` —
>   represented as one cold replacement-PAIR token (two switch
>   sub-blocks, engine emission order). Sequential same-turn faints
>   (move KO now, residual faint later) are separate request cycles and
>   stay single replacement tokens; the pairing signal is "other side
>   also pending a replacement at emission", never log adjacency.
> - *Pursuit KO of a switching target*: previously chosen switches
>   continue in Gen 2-4 (engine hint line) with no forceSwitch cycle —
>   the completion IS the target's declared switch sub-block. This
>   settles the completion-semantics question item 9's implementation
>   left open.
> - *Leads*: merged as a blind pair token (same simultaneity class as the
>   cold double replacement; turn 0 carries no real speed order).
>
> **Equivalence.** `flatten_turn_merged_tokens` reconstructs the
> per-action stream field-for-field; the ONE merge is the second
> sub-block's context trio (inherits the first mover's), which can
> differ only under a trio-changing first mover (side-effect ∈
> {hazard-set, hazard-clear, weather-set}). Corpus gate: 5 games ×
> both seats × every boundary — 313 per-action → 172 merged tokens
> (**45.0% reduction**), 1 trio allowance total.
>
> **Why the single trio is a re-parameterization, NOT information
> loss (#516 review, argument on record):** whenever the second
> mover's true trio differs from the stored one, the delta is
> deterministically recoverable from the merged token itself — the
> first sub-block's `side_effect` says the trio changed, its `action`
> says how (the weather move names the weather; a hazard set/clear is
> ±1 layer on the side implied by the actor role). Two hard
> mitigators bound even the residual modeling burden: the field token
> always carries the CURRENT field state, and every Tier-2
> residual/CB/investment computation runs on the per-action substrate
> with exact per-action trios (`annotate_turn_merged_tokens` maps
> conclusions back), so no inference arithmetic ever sees the stale
> trio. A per-sub-block context copy or delta flag is therefore
> deliberately NOT carried; this paragraph exists so "documented
> merge" is never mistaken for an accepted information hole.
>
> **SELF_HP_COST (v2.2 follow-up, census 153 → 155).** Each sub-block
> carries the fraction of the ACTOR'S max HP lost to its OWN declared
> action within that action's chunk (`NUMERIC_TT_SELF_HP_COST` /
> `NUMERIC_TM2_SELF_HP_COST`, v2.2-only columns — v2/v2.1 stay
> byte-frozen). Source classification (engine emission shapes verified
> against the vendored gen3 sim): INCLUDED — recoil family
> (`[from] Recoil` / Struggle recoil), crash on miss (verified: a bare
> untagged `|-damage|` on the actor after `|-miss|`), Substitute and
> Belly Drum costs (untagged self-target damage; also still the
> window's damage_fraction), Ghost-type Curse (verified: bare untagged
> actor damage), Pain Split's down-side (the actor's
> `-sethp … [from] move: Pain Split` below its previous ledger value),
> and self-faint moves (Explosion / Selfdestruct / Memento) where the
> cost is the actor's ENTIRE remaining fraction at strike — no
> self-damage line exists, the own-chunk `|faint|` is the protocol
> fact, and the move-id whitelist (not "any own-chunk actor faint")
> exists because Destiny Bond also faints the attacker inside its own
> chunk. EXCLUDED — confusion self-hits (the replaced move emits no
> `|move|` line, so no window exists; and it is not a cost of the
> CHOSEN action), entry-hazard damage on switch-ins (environmental,
> `[from] Spikes`-tagged, already derivable from the context trio),
> opponent-sourced tags (`ability: Rough Skin` / `Liquid Ooze`),
> Destiny Bond (opponent-set trap), and all residual-phase chip
> (unchanged chunk boundaries + KO-attribution vetoes).
>
> **NEGATED requires proof (review MED-1).** A missing second half is
> encoded `negated` only when consumption is CERTAIN: the turn closed
> (`|upkeep`/`|turn|N+1`/`|win`) or a mid-turn faint occurred (the
> engine-verified full cancel). A replay prefix cut at a mid-turn
> forceSwitch boundary — the Baton Pass completion choice is a real
> live decision point — encodes the opponent's unresolved action as
> `pending`, never as the free-pivot negation it would otherwise
> falsely assert.
>
> **K BUDGET UNIT CHANGE (loud).** The transition budget flag
> (`transition_token_budget`) counts TOKENS in BOTH modes — but in
> turn-merged mode a token is a whole TURN, so an unchanged K roughly
> DOUBLES the temporal horizon. The old 64-action default horizon is
> ≈ **32 turn tokens**; K=64 turn-merged ≈ the old K=128. Existing K=64
> configs that intend "the last ~32 turns" must move to
> `transition_token_budget=32` when turn-merged encoding lands; ablation
> arms comparing modes at fixed FLOPs should match TOKEN counts, arms
> comparing at fixed horizon should halve K. The flag's semantics
> (fill most-recent-N, oldest-first truncation, zero+mask the rest) are
> unchanged.

Status: design, 2026-07-03. Motivated by the width result (512d arm leading
its cohort on trajectory: 74.4% max-damage at 208k games) and the cost
structure that blocks scaling width further. Companion to
[`wse_shaping_and_coverage_design.md`](wse_shaping_and_coverage_design.md)
(separate arm; this doc changes the *observation*, not the reward) and
[`research_synthesis.md`](research_synthesis.md) (T2/T6).

## Motivation

The encoder consumes `window_size=4` complete observation copies
(`neural_policy.py`): each turn is ~46 tokens (1 field + 6 self-mon +
6 opponent-mon + 9 action candidates + **24 recent-event tokens**), so
~3/4 of the sequence is stale board state — and the 24 event tokens per
turn already duplicate most of what the history window provides, paying
twice for temporal context. Cutting to `window_size=1` plus a compact
opponent-signal block frees roughly 4× the sequence compute. That saving
is the feasibility budget for width: the point is not a smaller model, it
is **the same FLOPs spent on a wider trunk** (512d-class and beyond),
which current evidence says is the productive axis.

The bet: within-game temporal context in this game is compressible into
sufficient-statistic-style summaries (the poker-HUD precedent — aggregate
frequency stats as the standard compression of action history), and the
capacity freed is worth more than raw history. This is a hypothesis; the
ablation section makes it falsifiable.

**Evidence update (2026-07-03 evening, active-run curves):** the 256d
history-8 arm holds a consistent slight edge over 256d-4x at matched
milestones vs max-damage — temporal structure carries signal that pure
aggregates would destroy. This kills the strong "history is deadweight"
reading but not the encoding critique: snapshot replay makes the net
recover events by diffing 46-token state copies. The revised design
(below) keeps ordered history in a form ~23× cheaper per turn.

## Transition tokens: ordered history without snapshot replay

Replace history snapshots with **two transition tokens per past turn**
(one per side), each recording what that side did and what it caused:

- actor (species/slot), action (move id, or switch → incoming species)
- damage fraction dealt to the defender
- **damage residual vs the public calc's expected median roll** — one
  scalar carrying the set-inference content of the damage calc (positive
  ⇒ Choice Band / offensive investment evidence, negative ⇒ defensive
  investment; doubles as belief-engine evidence, same accumulator as the
  exact-state CB bit)
- flags: effectiveness class, crit, miss, KO

**The flags condition the residual; they are not merely beside it.** On
a `|-crit|` turn the residual is computed against the *crit-expected*
median — gen-3 crits are 2× and ignore attack drops, defense boosts, and
screens, so the crit median must drop those modifiers or a crit through
Reflect reads as a 3–4× outlier. Unconditioned, ~6% of damage events
(1/16 base, 1/8 high-crit moves) would inject 2× outliers into the
set-inference channel as manufactured Choice Band evidence. On a miss
the residual is masked (no damage ≠ weak-set evidence); multi-hit moves
carry an explicit `n_hits` field (from `|-hitcount|`) and normalize per
hit. The crit flag exists to explain the observation, not to learn from
past luck — crit-risk respect comes from outcomes shaping the value head
(per the luck-ledger decision).

### Implementation tiers (decided 2026-07-04 — complexity containment)

The modeling risk of this section lives entirely in the expected-damage
computation; everything else is protocol-announced fact. Tiered
accordingly:

- **Tier 1 (ships with the compression work; zero modeling risk):**
  transition tokens with actor, action, raw damage fraction, crit / miss
  / KO flags, `n_hits`, side-effect category — every field read straight
  from protocol events. No residual. The E/C ablation arms run on Tier 1
  alone; raw fractions + species embeddings are already strictly more
  legible than snapshot diffs.
- **Tier 2 (separate follow-up; all the modifier complexity, bounded three
  ways):** the damage residual and the conservative CB/investment bits.
  (a) Expected damage comes from **poke-engine queries** under the
  known-stats randbats world (fixed 85/31/neutral spread + public level
  make both sides' stats deterministic) — no hand-written calculator, so
  stat stages, screens, burn, weather, Explosion's defense-halving etc.
  are the engine's problem, not ours. (b) CB inference uses a
  **fixed-power physical move whitelist** (Hidden Power, Flail/Reversal,
  Rollout-class, Magnitude, and multi-hit moves never flip the bit).
  (c) The bit is **conservative by construction**: observed must exceed
  max-explainable = max over the species' candidate abilities × 1.1
  type-boost item × max roll × all public modifiers, by a margin, on
  **two independent events** (one calc edge case cannot flip it).
  Acceptance gate: precision ≈ 1.0 against ground-truth items extracted
  from the omniscient logs of controlled foul-play games (we own the
  BattleStream; true sets are recoverable) before any training run
  consumes the bit.
- **Called moves (Sleep Talk — verified the only reachable caller in the
  gen3 randbats movepool, on 40 species; Metronome/Assist/Mirror
  Move/Nature Power never appear):** the PP ledger charges the *calling*
  move (called moves spend no PP of their own in gen 3); the residual
  computes damage against the *executed* move's power; and no set
  evidence attaches to the called move (matching the belief engine's
  existing caller suppression). The `[from] Sleep Talk` protocol tag
  distinguishes the cases. Implementation may hard-code the Sleep Talk
  path; generic caller plumbing is unreachable in this format.
- **Transform (Ditto and Mew only in the pool; Ditto is all-transform):**
  transition tokens carry a **`transformed` bit** so copied-move usage
  is self-describing in the history stream — without it the net sees
  "Ditto used Flamethrower" as ordinary history and can mislearn set
  associations the belief engine already suppresses. The PP ledger
  scopes copied moves to the **transform instance**: gen-3 Transform
  grants 5 PP per copied move and a re-entry + re-transform grants a
  fresh 5, so copied-move PP is ephemeral (per instance, max 5,
  discarded on switch-out) and never charges the real set's ledger.
  Residual attribution keys on `transform_species` (stats are the
  copied target's, except HP), consistent with existing belief
  semantics.
  **Identity rule:** all history/stats/PP attribution keys on **slot +
  base species** (the protocol ident, e.g. `p2a: Ditto`), never on the
  acting species, and Transform never creates or updates an
  opponent-roster entry. This matters beyond bookkeeping hygiene: Ditto
  copies *our* active mon, and the opponent's team may legitimately
  contain the same species (no cross-team species clause) — acting-
  species attribution would charge copied-move usage against the real
  teammate's PP ledger and pollute its tendency stats. The
  `transformed` bit + `transform_species` are what let every consumer
  explain a foreign species acting for a known slot.
- **Damage-outcome enum, in two evidence classes (negate vs truncate):**
  attack tokens carry `damage_outcome ∈ {normal, blocked, immune,
  hit-sub, broke-sub, endured}`, self-describing without cross-token
  joins (the defender's own Protect/Sub click is a separate token the
  same turn). The classes have opposite residual semantics:
  - **Negation (`blocked`, `immune`) — no damage event occurred.** The
    attack connected with nothing; magnitude evidence is *undefined*,
    not zero. Residual: no evidence. Move-reveal/set evidence is
    unaffected (the click is always informative). `blocked` feeds the
    protect-pattern tendency stat. `immune` splits by explanation, with
    opposite inference content:
    - *typing-explained* (Earthquake into a Flying-type): tautological
      given public state — contributes **no** ability evidence (the
      attribution order is typing first, ability only as the residual
      explanation; without this check every EQ into a Skarmory pollutes
      candidate pruning). Its information lives in the prediction
      channel instead: attacking into a known immunity almost always
      means the attack targeted the mon that was leaving — an
      unambiguous "did not predict the switch" observation for the
      midground/prediction stat.
    - *not typing-explained* (EQ immune on a non-Flying): the immunity
      must be ability-sourced — confirmation-grade ability
      identification, variant pruning fires.
- **Healing: a side-effect category value, never a magnitude field.**
  Unlike damage, every gen-3 heal magnitude is deterministic given the
  action plus public state (Recover-class 50%, Rest 100%,
  Synthesis-class weather-scaled with weather public, drains =
  f(observed damage), Leech Seed = 1/8 seeded max HP, Pain Split
  averages public HP) — healing encodes nothing hidden, so it gets no
  inference channel. Leftovers heals route to the item channel as
  already specified.
- **Absorb-class abilities (Volt Absorb, Water Absorb, Flash Fire).**
  The outcome enum gains **`absorbed`** — negation class (residual: no
  evidence) but distinct from `immune`: the defender *gained* from the
  attack (25% heal, or Flash Fire's boost state), which is the one
  history lesson the net must never re-learn mid-game. Ability
  identification rides the existing `[from] ability:` machinery.
  Flash Fire's boost is **state, not just an event**: tracked as a
  volatile (protocol `-start`), and the Tier-2 residual must include it
  among public modifiers when the boosted mon attacks Fire moves
  (1.5×), else every post-absorb Flamethrower reads as phantom CB
  evidence. **Elimination direction:** landing normal damage on a mon
  whose candidate set includes the absorbing ability deterministically
  rules it out — variant pruning via the same non-trigger pattern the
  engine already implements for Intimidate ({Volt/Water Absorb, Flash
  Fire, Levitate} × "move connected normally").
- **Turn-order inference REMOVED (decided 2026-07-04).** Speed brackets
  were cut entirely, not conditioned. The payoff is near zero — base
  stats are computable from public info (see computed expected stats),
  so turn order's only inference content is weather-ability
  identification (Swift Swim/Chlorophyll) — while the cost is a full
  priority/action-order mechanics model (priority table, switch
  ordering, Pursuit interception, charge turns, speed ties, skip
  states), every row of which can corrupt a bracket. The asymmetry
  decides it: hard state features are trusted, so one engine error is
  persistent adversarial input; soft inference degrades gracefully.
  **Principle: hard rules only where the protocol makes them
  tautological** (PP, procs, duration timers — event bookkeeping, no
  mechanics model); anything needing a broad mechanics model stays raw.
  The net keeps the soft path for free: transition tokens are emitted
  in action order (within-turn sequence is implicit — no field, no
  model), and weather + candidate abilities are already features. If
  probes later show rain-sweeper identification failing, a narrow
  trigger rule can be added behind Tier-2-style validation.
- **Pursuit (variable power + out-of-order execution).** If the target
  is switching, Pursuit executes before the switch at doubled power
  (40→80) against the outgoing mon. Detection is protocol-tautological
  — event order alone (Pursuit's damage against a mon whose switch or
  faint-then-declared-replacement follows in the same turn) yields a
  `pursuit-intercept` flag, no mechanics model. Rules: the residual
  uses scenario-correct power (80 when intercepting) but **Pursuit is
  excluded from the CB whitelist entirely** (scenario-dependent power +
  an unverified gen-3 accuracy-bypass quirk on switching targets — on
  the Tier-2 engine-verification list — make it a poor CB probe at 40
  BP regardless). A doubled Pursuit is an *affirmative* switch-predict
  observation for the midground/prediction stat (sharper than the
  immunity-based no-predict signal). Faint-during-intercept still
  completes the declared switch — slot-first attribution covers the
  unusual transition; entry counters must not double-count.
- **Drain moves and Leech Seed.** The side-effect category carries a
  distinct `drain` value (damage + self-heal in one action); heal
  magnitude stays derivable (50% of observed damage). Interactions:
  *drain vs Substitute* — the attacker's public heal reconstructs the
  hidden sub damage exactly (damage = 2×heal), so for drain moves
  `hit-sub` upgrades from inequality evidence to an exact residual (do
  not blindly mask); *Liquid Ooze* reverses the drain and is a
  confirmation-grade ability reveal, already consumed by the generic
  `[from] ability` machinery. Leech Seed: seeded state is a tracked
  public volatile and the 1/8-max transfer is derivable (no storage —
  unlike Wish there is no hidden latency); chip arrives
  `[from] Leech Seed`-tagged so residual attribution already excludes
  it; the **recipient is the slot occupant, not the original seeder**
  (credit follows the slot, consistent with the Transform identity
  rule); Grass-type immunity routes as typing-explained negation.
- **Pending-effect rule: store latent state, never derivable
  expectations.** Next-turn Leftovers/Leech Seed/Ingrain expectations
  are rule applications over tracked state — storing them repeats the
  snapshot-replay mistake in miniature. But **pending Wish** (50% heal
  landing end of next turn on whatever occupies the slot; 16 species in
  the pool) is latent state no rule can reconstruct, and it is
  currently untracked — `showdown.py` tracks Future Sight's pending
  counter and has no Wish equivalent. Wish joins the exact-state
  pending/duration-counter family (same pattern as
  `future_sight_turns`), both sides.
  - **Truncation (`hit-sub`, `broke-sub`, `endured`) — full computed
    damage WAS dealt**, to a proxy or clipped at survival, so magnitude
    evidence exists in inequality form: `broke-sub` ⇒ damage ≥ sub HP;
    `hit-sub` on a **fresh** sub ⇒ damage < 25% of max HP (mild
    anti-CB evidence — masking this discards real information);
    `endured` ⇒ damage ≥ (pre-hit HP − 1), a *tight* lower bound since
    HP is exactly known. Conservative rule: sub bounds apply only when
    sub freshness is known (first hit on a just-set sub); cumulative
    multi-hit sub bookkeeping (sum ≥ 25% at break) is Tier-2 detail.
- **Tier-2 extension (note only): defender-side ability inference.** The
  same residual machinery on *our* attacks reveals the defender's
  ability (e.g. Thick Fat halving our Fire damage is observable against
  their deterministic stats). Symmetric, cheap once Tier 2 exists;
  recorded here so it isn't rediscovered.
- **Tier 3 (not now):** evidence-weighted belief posteriors —
  ReBeL-direction work, out of scope for this doc.

### The item universe is 13 items, mostly species-locked (verified 2026-07-04)

From the generator (`teams.ts`): seven species-locked assignments
(Farfetch'd→Stick, Latias/Latios→Soul Dew, Marowak→Thick Club,
Pikachu→Light Ball, Deoxys→White Herb, Linoone-sweeper→Silk Scarf,
Unown→CB/Twisted Spoon by set), plus a general pool of just
{Leftovers (default), Choice Band, Lum Berry, Salac/Petaya/Liechi
Berry}. No RNG trick items exist in the pool. Consequences:

- The randbat source's candidate variants **already carry items** (the
  spread-reproduction work replicated the generator's item logic), and
  variant pruning is joint — revealing Substitute prunes to variants
  whose items come along (Sub→pinch-berry, RestTalk→Leftovers
  correlations are structural, not inferred). Species locks collapse
  `possible_items` on upsert with no special-casing.
- The CB max-explanation tightens: the only 1.1× items in the format
  are themselves species-locked, so for the general population the
  non-CB item universe contains **zero damage modifiers** — exceedance
  needs to clear only candidate-ability variance and roll variance.
- **New exact-state rule family — non-proc pruning.** Three items in
  the general pool announce themselves *automatically* when their
  trigger occurs, so a trigger without the proc is deterministic
  negative evidence (PP-ledger epistemic class, not the descoped
  behavioral kind):
  1. *No-Leftovers:* ended a turn active and damaged with no itemized
     Leftovers heal ⇒ prune Leftovers variants. Full-HP turns yield no
     evidence.
  2. *No-Lum:* a successful status application not **instantly** cured
     ⇒ prune Lum variants (gen-3 Lum procs immediately, incl. on Rest's
     self-sleep; Shed Skin's end-of-turn ability-tagged cure is
     timing-distinguishable; Safeguard/Substitute blocks apply no
     status and yield no evidence).
  3. *No-pinch-berry:* HP at or below 25% after an action with no berry
     activation ⇒ prune Salac/Petaya/Liechi variants.
  Because Leftovers is the default item, the family compounds fast: one
  damaged turn removes the modal candidate, and a stuck status or a big
  hit reduces the general pool to **Choice Band by elimination** —
  making the Tier-2 damage residual confirmation rather than primary
  evidence for most mons. Knock Off/Trick and all positive activations
  emit explicit item events the engine already consumes.

**Status chip damage (toxic/burn/sand/Spikes) is attribution hygiene,
not a feature.** Chip is fully determined by public state the
observation already tracks (toxic stage, statuses, hazard layers,
weather) and carries no hidden information; past chip is summarized in
the HP bar. The pipeline rule: residuals are computed from the move's
own itemized damage event (protocol `[from]` tags separate chip from
move damage), never from turn-level HP deltas. No chip history tokens.
- side-effect category (status inflicted / hazard set / weather set /
  boost used)

Cost: 2 tokens/turn vs 46 — **16–24 turns of ordered history for less
than one snapshot**, so coverage extends to effectively the whole game
where h4/h8 snapshot windows were the ceiling. The 24 raw
`recent_public_events` tokens are subsumed and dropped (they are the
crude 3–4-turn version of this idea, currently paid *in addition to*
snapshots). Precedent: Metamon's transformer consumes explicit
(previous-action, reward) sequences rather than repeated state — the
strongest published agent in this domain uses decision history, not
snapshot history.

The compression spectrum is then: full snapshots (structure + restated
state) → transition tokens (structure only) → aggregate stats (no
structure). The stats block (below) is retained alongside transition
tokens: cross-turn aggregates spare the trunk from recomputing counts
over the token sequence, and they remain the z-descriptor vocabulary.

### Whole-game memory: two channels, two retention policies (decided 2026-07-04)

The observation stores the whole game **twice, in channels with
different retention properties**, and truncation is safe precisely
because durable information is routed out of the ordered channel:

1. **Unbounded aggregates (whole game, order-free, never truncated):**
   the exact-state layer and stats block — PP ledger, belief facts,
   non-proc prunings, counters, (count, opportunity) pairs — accumulate
   at constant size. Everything durable about a 40-turn-old turn
   (reveals, spend, tendencies) already lives here.
2. **Ordered transition window (recency/momentum):** fixed slot budget,
   zero-padded, masked, filled as turns arrive. Two positional signals
   per token: **absolute turn number** (game phase) and **turns-ago**
   (recency). Ordering carries short-range structure (the h8 > h4
   evidence); nothing suggests order at range 100 adds anything the
   aggregates miss.

**Budget: 64 turns (128 transition tokens), truncate oldest-first** —
the prefix is exactly what the aggregates have absorbed. Rationale:
healthy games run ~25–30 turns (50–60 tokens — whole-game coverage is
nearly free), while the long tail is RestTalk stall loops whose ordered
detail is worthless and whose real content (PP attrition) lives in the
ledger; attention is quadratic and every training example pays the full
slot budget, so size to ~p95 of healthy length, not the 250-turn cap;
and a 2-layer trunk attends poorly over 500 tokens — freed compute is
worth more as width. **K is a masked config, not a spec change**: the
ablation gets K ∈ {16, 64} for free, testing whether ordered range
beyond 16 pays at all once aggregates exist. If K=16 matches K=64, the
compression program has fully delivered: whole-game memory at a
twentieth of the original sequence cost.

## Design principle: evidence mass, not rates

Every tendency feature is a **(count, opportunity) pair**, never a bare
rate. A 100% switch-out rate over one observation and over eight are
different evidence; the net can compute the ratio but cannot recover the
mass. All features are player-relative and computed from public
information only (events, revealed moves, belief candidate sets, damage
estimates) — no oracle leakage.

## v1 feature set

### Exact-state features (belief-engine/state layer — computations, not tendencies)

These are deterministic bookkeeping the engine can do perfectly; they live
beside the PP ledger, not in the stats block. All approved 2026-07-03.

- **Computed expected stats** (REPLACES the earlier speed-bracket
  feature — removed 2026-07-04, rationale in the turn-order section
  below): opponent actual stats are *computable*, not inferable — the
  generator's fixed 85 EV / 31 IV / neutral spread plus public
  species+level determines them to within the HP-IV point. Expose the
  computed stat block as deterministic numeric features on opponent
  tokens. Pure arithmetic, zero inference, zero error surface.
- **Sleep clause consumed** (one bit per side): once a side has put an
  opposing mon to sleep, its remaining sleep moves are dead weight.
- **Sleep turn counters** per sleeping mon, both sides, with a
  **rest-sleep flag**: Rest is exactly 2 turns (wake turn *known*), natural
  sleep is 1–4 (wake is a hazard rate). The flag distinguishes "they know
  when they wake" from "they're gambling," which changes both sides' play
  around the sleeping mon.
- **Choice Band detection — damage-calc only.** One bit per opponent mon:
  observed damage exceeded the maximum non-CB roll for that move/matchup.
  This is deterministic proof (set-level variation cannot produce +50%
  damage). Repeat-move streaks are deliberately *not* used: repetition is
  consistent with CB but proves nothing.
- **Timed field-condition counters**: weather turns remaining (normalized
  /5) + permanent bit (see Weather section below), and the same for
  Reflect / Light Screen / Safeguard / Mist (deterministic 5-turn-class
  counters in gen 3).
- **Turns-in-battle counter** for the opponent's active mon. Cheap proxy
  with two readings: a fresh counter after a faint implies the KO context;
  a long counter signals commitment to the current position.

### PP-ledger subsumption principle

The exact PP ledger (feature 1 below) doubles as a **per-move usage
counter**, and revealed-move tokens already persist set knowledge. Together
they subsume, at zero extra cost, the whole family of reveal/usage-count
features considered and rejected as separate items: phazing usage, Pursuit
counts, Counter/Mirror Coat usage, recovery-move counts, Encore/Taunt
reveals, boost-history, Endure/reversal-kit alerts, Substitute counts, and
Protect patterns. Do not add dedicated features for these; if a later probe
shows the net failing to use one, the fix is representation/capacity, not
another counter. (Boom threat likewise: candidate-set collapse already
carries "Explosion is still possible," which is the decision-relevant
part.)

### Tendency features (stats block)

1. **Exact opponent PP ledger (belief engine — exact-state class).** For
   every revealed opponent move: remaining-PP fraction, maintained
   exactly — max PP from the randbats catalog, decrement per observed
   use, ×2 when our active mon has Pressure (our own ability is perfectly
   known, so the ledger is exact, not approximate). Exposed as a numeric
   feature on the opponent move slots, mirroring our own
   `NUMERIC_MOVE_PP_FRACTION`. Motivation: pp-stall is a real strategy in
   both directions, and per the subsumption principle above this single
   ledger carries the entire usage-count feature family.
2. **Global switch tendency:** (opponent switch count, opponent decision
   opportunities).
3. **Per-opponent-mon tendency triple** on the existing 6 opponent-mon
   tokens: (switched-out-before-attacking count, stayed-and-attacked
   count, turns-active). Deliberately *not* the full 6×6
   matchup-conditional matrix — sparse deadweight at battle length.
4. **Opponent weather reveals, with source.** Per weather type:
   {opponent set it this game, source was an ability}. The ability case
   (Drizzle/Drought/Sand Stream — all present in the gen3 randbats pool)
   is a double reveal: the mon's ability is confirmed (narrowing its set
   catalog) and the weather is **permanent** in gen 3, versus exactly 5
   turns for move weather (no extension items exist in gen 3). Source is
   parsed directly from the `|-weather|...|[from] ability:` protocol tag.
   Current gap being fixed: `_update_weather` keeps only a bare
   identifier — no duration, no source — so Tyranitar sand (plan the
   whole game around it) is indistinguishable from turn-4 Sandstorm
   (about to expire). Own-side weather usage is *not* tracked: the field
   token carries active state + the turns-remaining counter above, and
   our own past usage changes no decision.
5. **Midground/prediction signal**, over turns where *we* switched:
   (my-switch-turn count, opponent-move-better-vs-incoming count
   [margin-thresholded damage-calc comparison, to exclude
   good-vs-both moves], pivot-move count). High ratio = opponent predicts
   switches / midgrounds; low = they tunnel the active mon. Noisy by
   nature; the (count, rate) form lets the net weigh confidence.
6. ~~Opponent setup usage bits~~ — removed 2026-07-03: subsumed by
   revealed-move tokens + the PP ledger (see subsumption principle).

**Descoped — per-candidate-move declined-opportunity counters.** The
"they didn't click the KO move they might have" evidence is real, but
representing it requires per-candidate-move accumulators across the whole
candidate set — expanding an already expensive feature family that is
often deadweight on the net. Revisit only if the v1 stats show value and
a later probe shows the net misreading stay-in bluffs; the principled
version (behavior-likelihood belief updates) is ReBeL-direction work, not
an observation feature.

## Encoding

One new **stats token** (or two) per observation carrying the global
pairs (items 2, 4, 5), plus per-mon fields added to the existing opponent
tokens (items 1, 3, 6). New observation spec version with checkpoint
compatibility guards per `model_versioning.md`; `window_size=1` becomes a
config choice, not a new code path (the window machinery already
supports it).

## Interplay notes (both load-bearing)

- **These features are near-constant in mirror self-play** — against a
  copy of yourself, tendency stats carry almost no signal. They become
  informative exactly when opponents vary. Ship alongside the cross-arm
  opponent pools (#487 arms as each other's population); the pools make
  the features informative and the features let the net exploit what the
  pools expose.
- **The same statistics are the self-derived z-descriptor vocabulary**
  (switch rate, weather/setup usage, midground rate) for the
  conditioning mechanism in `research_synthesis.md` stack item 3. One
  instrumentation effort, two consumers.

## Ablation plan (500k, same seed bands and yardstick as the width arms)

| arm | history | stats block | width |
|---|---|---|---|
| A (control) | 4 snapshots | no | 256d |
| A8 (running) | 8 snapshots | no | 256d |
| B | none (window=1) | yes | 256d |
| **E (primary)** | **16-turn transition tokens** | yes | 256d |
| C (payoff) | 16-turn transition tokens | yes | 512d |
| D (optional) | GRU aggregator | no | 256d |

B vs E isolates the value of *ordered* history beyond its aggregates —
the exact question the h8 > h4 edge raises. E vs A/A8 tests whether
transition tokens preserve what snapshot history provides at a fraction
of the sequence cost. C spends the savings on width, which the active
runs say is the productive axis.

Reads: foul-play at matched milestones (primary), max-damage trajectory,
ΔV/behavior probes, and wall-clock/games-per-hour (the feasibility claim
is itself a measurable). Success for the compression hypothesis: B ≥ A on
foul-play at matched milestones with the expected sequence-compute
saving. Success for the program: C > both on trajectory at equal
wall-clock. D distinguishes "engineered stats" from "any temporal
compression."

## Explicit non-goals

- No reward changes (that is #487's arm).
- No cross-battle opponent memory — all counters reset per battle
  (randbats opponents are exchangeable across battles).
- No behavior-likelihood belief updates (descoped above).
