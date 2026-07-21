# Observation schema v3 — spec

Status: 2026-07-20, owner-approved direction for the next generation run.
Successor to `pokezero.observation.v2.2`. Additions motivated by the
Toxic redundant-clicking investigation and the sleep-clause / stall-loop
interpretability goals. **Design decision (owner question resolved): the fail
event and the clause state are SEPARATE signals** — `-fail` is a history marker
on the action's transition token and fires for many unrelated reasons (status
move on an already-statused target, Safeguard, clause blocks, …); the clause
bits are predictive current-state on the field token. Conflating them would make
the fail marker wrong for most fails and would break counterfactual
flag-flip probes. The model learns the correlation itself.

**v3 is still PRE-FREEZE** (the freeze gate is the input-audit program; the Rust
fold mirror + golden-corpus regeneration have not happened yet), so appending a
new numeric column here is legal: every v2/v2.1/v2.2 column keeps its position,
the corpus stays v2.2, and no shipped checkpoint has been trained at v3. After
Change 3 lands, **the v3 numeric feature count communicated to the Rust-mirror
work is 160** (v2.2's 155 + the four Change 1/2 columns + this one).

## Change 1 — `-fail` transition event (corrective signal)

- `transitions.py`: on a `|-fail|…` protocol line while an action transition
  is in flight (`current is not None`), set `current.fail = True`. Scope to
  the action window and do NOT condition on which slot the argument names —
  the engine sometimes names the actor, sometimes the target, depending on
  the effect. (The existing `-miss` handler's actor-side condition is correct
  for misses; fails need the window-scoped rule.)
- Emission: mirror `miss`'s emission convention exactly (same feature class,
  adjacent position) on the action transition token, gated to schema >= v3.
- Under v2.2 emission the bit must not exist: **v2.2 output stays
  byte-identical.**
- Note: `-miss` is already encoded (since the v2.1/v2.2 batches). After v3,
  a silent no-op is disambiguated: miss bit = accuracy miss, fail bit =
  move failed, neither = genuinely event-less resolution.

## Change 2 — sleep-clause state bits (predictive signal)

Gen3 randbats runs the gen3 `Standard` ruleset: **Sleep Clause Mod is active;
Freeze Clause Mod is NOT** (it exists only in `standarddraft`) — no freeze
flag, it would be a dead column.

- Two numeric 0/1 features on the FIELD token, schema >= v3 only:
  - `sleep_clause_blocks_self`: an opposing pokemon is currently asleep from
    a sleep OUR side induced → our sleep-inducing moves will fail.
  - `sleep_clause_blocks_opp`: symmetric (feeds the opponent-action head).
- **Public attribution rule (no move-window bookkeeping needed):** in gen3
  singles, sleep is only ever (a) induced by the opposing side's move or
  (b) self-inflicted Rest, and Rest tags its status line
  (`|-status|SLOT|slp|[from] move: Rest`). Therefore: a `-status slp` line
  WITHOUT the Rest tag ⇒ induced by the opposing side. Track, per side, the
  set of enemy slots it has publicly put to sleep.
- Clear a tracked victim when it wakes (`-curestatus … slp`) or faints.
  Switching out does NOT clear (sleep persists and is public on revealed
  mons). Natural Cure ambiguity resolves via the same `-curestatus` line the
  belief engine already consumes.
- Anti-leakage: derived ONLY from public protocol lines — no engine-side
  hidden state. Both bits are computable by either player from the log.

## Change 3 — consecutive-stall counter (Protect/Detect/Endure)

Motivated by the stall-loop interpretability goal: Protect/Detect/Endure lose
success probability with each consecutive use, so a policy that cannot see its
own stall streak double-clicks Protect into a coin-flip. One numeric feature
exposes the streak so the model can price the falling odds.

**Engine ground truth (verified before coding, vendored showdown
`data/conditions.ts:439-462`, the `stall` condition — "Protect, Detect, Endure
counter"):** a stall move adds the `stall` volatile; `onStart` sets
`effectState.counter = 3`; every subsequent consecutive stall runs `onStallMove`
= `success = this.randomChance(1, counter)`, and **`if (!success) delete
pokemon.volatiles['stall']`** — a failed stall deletes the volatile, so the
counter resets to its `onStart` value on the next stall. `onRestart` does
`counter *= 3` (bounded by `counterMax: 729`) on each success. The volatile also
evaporates (duration 2, reset to 2 only by `onRestart`) after a non-stall turn,
and all volatiles clear on switch/faint. So the engine's counter is exactly a
**consecutive-successful-stall streak**, reset by a failed stall, a non-stall
action, a switch-out, or a faint. Gen3 shares this ONE `stall` volatile across
Protect, Detect and Endure (all three set `stallingMove: true` and call
`addVolatile('stall')`; `data/moves.ts` protect 13960 / detect 3523 / endure
4802). Pool reachability in `data/random-battles/gen3/sets.json`: Protect (43
species) and Endure (4 species) are reachable; Detect is NOT in the gen3
randbats pool (0 species) but shares the `protect` volatile and is handled for
correctness.

**Public reconstruction (no hidden state).** One per-side counter tracks the
consecutive successful stall-move uses by that side's currently-active mon:

- **Increment** on the success-only `-singleturn` tag. Two tag shapes, both
  verified in the vendored data: Protect/Detect share `volatileStatus:
  'protect'` and emit `|-singleturn|SLOT|Protect` (`data/moves.ts:13980`);
  Endure emits `|-singleturn|SLOT|move: Endure` (`data/moves.ts:4822`). These
  `-singleturn` lines fire ONLY on success — a failed stall emits `-fail` and no
  `-singleturn`. Other `-singleturn` users (Focus Punch, Magic Coat, Snatch)
  normalize to other names and are excluded.
- **Reset to 0** on any of the five causes, mirroring the engine's volatile
  deletion: (1) the mon's action window containing a `-fail` for a stall move (a
  failed Protect/Detect/Endure — the `randomChance` miss that deletes the
  volatile); (2) any non-stall `|move|` by that mon; (3) `|cant|`; (4)
  switch-out / `|drag|`; (5) `|faint|`.
- Tracked in the same home as the sleep-clause tracker (`_ReplayParser` in
  `showdown.py`, snapshot-carried), the counter/snapshot shape mirroring
  `toxic_stage` exactly: a per-slot parser dict → `snapshot()` →
  `normalize_for_player` per-side scalar → written on the ACTIVE mon token.
  A tiny per-side "stall move in flight" flag (set on a stall `|move|`, consumed
  by its `-singleturn`/`-fail`) distinguishes reset cause (1) from an unrelated
  `-fail`; it is snapshot-carried too so a mid-window resume converges.

- **Encoding:** one new numeric feature on each side's ACTIVE pokemon token
  (like `NUMERIC_TOXIC_STAGE`), schema >= v3 only, value `min(1.0, count / 8.0)`.
  Derived only from public protocol lines, so both players compute both
  counters. Column `V3_NUMERIC_BASE + 4`; `_V3_NUMERIC_FEATURE_COUNT` and the v3
  numeric census floor go 159 → 160; v2.2 counts untouched. Under v2.2 the
  column does not exist — **v2.2 output stays byte-identical.**

## Schema plumbing

- New id `pokezero.observation.v3`, CLI choice `v3`
  (`observation_schema_version_from_choice`), feature-count constants
  `_V3_*` = v2.2 counts + the additions, entries in the per-schema count
  maps, checkpoint latching identical in structure to the v2.1→v2.2
  introduction (v2.2 checkpoints keep loading and encoding exactly as
  today — dual-schema support is the existing pattern).
- Vocab: no new categorical vocabulary rows required (all three changes are
  numeric bits) unless the miss-emission convention turns out to be
  categorical — in that case mirror it and extend the vocab by the one
  value, documented here.
- Change 3 adds one appended numeric column at `V3_NUMERIC_BASE + 4`
  (`NUMERIC_STALL_COUNTER`), bumping `V3_NUMERIC_EXTRA` 4 → 5 so
  `_V3_NUMERIC_FEATURE_COUNT` and the v3 numeric census floor become 160. The
  categorical census is unchanged.

## Acceptance (tests required)

1. Scripted protocol with a failed status move → fail bit set on that action
   transition under v3; absent under v2.2; v2.2 encoding of the same log is
   byte-identical to before the change.
2. Clause lifecycle, both directions: induced sleep → bit on; Rest → bit
   stays off; `-curestatus slp` → bit off; faint of the sleeper → bit off;
   switch-out of the sleeper → bit stays on.
3. Stall-counter lifecycle: the column rises 1/8, 2/8, … on consecutive
   Protects and resets to 0 on EACH of the five causes (failed stall `-fail`,
   non-stall move, `cant`, switch-out/drag, faint); Endure shares the counter;
   opponent side symmetric, both seats; snapshot round-trip preserves both
   counters; the v3 column position is pinned; a Protect-heavy log's v2.2
   encoding is byte-identical to before the change.
4. Existing v2.2 test suites pass untouched.

## Coordination (v3-stream / Rust fold)

The golden-corpus bit-exactness gate means this schema lands in BOTH
encoders: after this (production) implementation merges, the Rust fold
encoder (`rust/pokezero-search`) mirrors it and the golden corpus is
regenerated at v3. Until then the corpus stays on v2.2 and the gate is
unaffected (v2.2 output unchanged). The new generation run launches only
after both sides agree.

## Review dispositions (2026-07-20, post-implementation Opus review: SHIP)

- **Fail on switch rows is INTENDED.** A blocked switch-in Intimidate (Clear
  Body / Hyper Cutter / White Smoke) emits `-fail` inside the switch window,
  so a switch sub-block can carry `fail=True`. Kept deliberately: it is
  deterministic, public, disambiguated by the sub-block kind, and publicly
  reveals the opponent's ability class — informative signal, not noise.
- **Known accepted loss (v3-only, rare):** a Baton Pass completion switch is
  collapsed into `baton_pass_species` during turn-merging, dropping a
  `fail=True` from a BP-into-Clear-Body Intimidate block. To revisit at the
  Rust-mirror/corpus-regeneration milestone with a scenario test; either
  preserve fail onto the collapse or re-accept the loss explicitly.
- Golden-corpus tooling migrated to the schema-family membership tuple so a
  future default-schema bump cannot silently disable turn-merged capture.
