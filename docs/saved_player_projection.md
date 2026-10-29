# Saved trusted source to player observation inputs

`saved_player_projection.py` is a separate source-derived observation-input
projection. It does not change the old trusted cursor, shared GPU fold, original
collection or original observation tensors. It never claims captured player wire,
reconstructed stream bytes, full encoder parity or model-forward qualification.

The supported emitter is Showdown revision
`f76228a1354b5d0f307ca2d16101294ad3a2308b`, fixed Gen3 singles random battles,
six Pokémon per side, with the exact nine recorded rules including HP Percentage
Mod. Missing/changed setup, unproved message families, unknown hints, malformed
split-sensitive payloads or ambiguous private ownership refuse the whole batch.
No row filtering, skipping or guessed last-switch ownership is permitted.

## Source-derived message contract

Showdown `sim/battle.ts:33–60` selects each split's secret for its owning player
and omniscient channel, and the shared alternative for other players.
`sim/battle-stream.ts:432–447` applies this extraction and separately routes
sideupdates. The collector saved extracted omniscient lines and owning requests,
so missing `split` markers are expected; their absence is not proof of public
visibility.

`Pokemon.getHealth` (`sim/pokemon.ts:2060–2101`) gives the owning seat exact
`h/H`; the opposite seat receives `ceil(100*h/H)/100`, capped at 99 when h<H
would round to 100. Status suffixes are preserved. Zero is literal `0 fnt`.
The module translates only the opposite view's condition in switch/drag,
damage/heal and each separate Pain Split sethp message. It preserves details,
tags and order. It does not infer max HP from species, replace HP during
Transform, add extra messages, or alter faint messages.

`getFullDetails` (`sim/pokemon.ts:545–554`) changes only the health component
between split alternatives. Pain Split emits separate target messages
(`data/moves.ts:13141–13143`). Transform emits a shared message and retains the
holder's HP (`sim/pokemon.ts:1271–1353,1390–1410`). No Illusion/replace family is
admitted by this fixed observed source profile.

Pressure's exact silent signature is private to its named holder
(`data/mods/gen3/abilities.ts:145–150`). It is retained only in that owner's
input. The five observed nonsilent signatures have the public setAbility shape
`-ability|holder|Pressure|oldAbility|[from] sourceEffect|[of] target`.
`sim/pokemon.ts:1930–1949` emits this with ordinary `add`, not `addSplit`;
The old ability and sourceEffect fullname are common arguments. Gen3 Trace is
one producer (`data/mods/gen3/abilities.ts:186–195`); the compact census did not
retain each full sourceEffect string, so this implementation does not invent
those values or require all five to be Trace. The Start callback
inside setAbility is gated at gen>3. Thus these public Trace lines are kept in
both views, and are not confused with private Pressure startup messages.

Three recorded hint texts are shared: Knock Off
(`data/mods/gen4/moves.ts:703`), Pursuit continuation
(`sim/battle.ts:2791`), and Sleep Clause (`data/rulesets.ts:1395`). These calls
have no side argument, so `Battle.hint` uses common `add`
(`sim/battle.ts:3092–3100`). The Sleep Clause `-message` is also common
(`data/rulesets.ts:1394`). They are all preserved.

The Intimidate hint is private to its holder
(`data/mods/gen3/abilities.ts:62–74`). Its emission predicate proves that every
adjacent foe has Substitute. In fixed singles, **exactly one substituted side
therefore proves the holder is the opposite side**. The module uses an internal
complete production parser's public Substitute ledger for this proof. That
trusted parser is never normalized, encoded or returned as actor state. Both
substituted or neither substituted is an explicit refusal. Immediate switch
adjacency is not sufficient: switch and runSwitch are separate queued actions.
The full fixed roster must pass this ownership proof before any comparison;
the static implementation does not assert that it has done so.

The observed family roster is closed, plus one explicitly source-proved shared
family used by consumer controls: `-item` emitted by inherited base Trick
(`data/moves.ts:19906,19912`). Unknown families are not silent passthroughs.

## Missing Struggle warnings: guarded observation equivalence

`sim/side.ts:681–694` sends `-activate|owner|move: Struggle` privately from
chooseMove before executing the turn. The saved collector discarded all player
nonrequest sideupdates. The module neither fabricates the missing warning nor
guesses its ordering. A Struggle-only request can still select switch; the
147 requested versus118 committed source signatures are not interchangeable.

This omission is an observation-consumer contract, **not byte-wire parity**.
The reference helper guards an actionable owning Struggle-only boundary: no
pending Trick pairing, no pending belief switch and no own pending faint
replacement. It refuses rather than assume these clocks are irrelevant.

The exact warning is not a recognized action, confusion marker, volatile,
ability reveal or item reveal. Parser damage/volatile handlers do not match
its `move: Struggle` payload. The generic faint latch sees no pending owning
replacement. Belief residual phase changes on move/switch/drag/replace/cant or
bare framing, not this warning. The potentially destructive generic consumers
are `PublicBattleBeliefEngine._track_trick_pairing` and pending-switch resolution;
the helper requires both to be quiescent. The belief event counter changes,
but its only reads are snapshot/clone/debug serialization. It is not a tensor
input. Raw event history and recent diagnostic metadata also differ.

Transition `-activate` handling recognizes confusion/Protect/Detect/Substitute/
Endure; Pursuit scanning recognizes its own payload. Strike trackers align
contexts by action windows and event indices; an extra warning shifts raw
indices but does not create an action or strike. Tests compare action-aligned
contexts and all assessed producer outputs across subsequent updates, rather
than asserting raw index equality.

The focused controls use authored in-memory dex/set/vocabulary fixtures with
production parser, belief, normalization, Tier2, investment and five-tensor
encoding. They execute no Showdown source, simulator, model, saved roots or
network operation. Both owners' warning omissions survive following boundaries;
a seven-boundary sequence includes recoil/residual clocks, switching, an ordinary
own strike, a complete Trick item pair and later Struggle. Full parser semantic
state, every belief-engine field except diagnostic event_count, normalized state
except recent diagnostics, action-aligned annotation contexts/results, and all
five arrays agree. A negative control demonstrates that an illegally interleaved
warning really clears a pending Trick pairing and is refused by the guard.
This is source/fixture evidence, not a champion-vocabulary or full-corpus result.

## Integration API and chronology

`SavedSourcePlayerProjector(...).project_nonrequest_delta(lines)` emits one
frozen `PlayerProjectionBatch` containing both immutable seat line tuples,
separate nonrequest source offsets/hash, routing counts, revision, scope and
explicit missing-warning/byte-wire flags. The upstream attested trusted cursor
continues to validate the original mixed source and original references. Its
hashes are never renamed projected hashes.

`new_player_reference_environments(config, battle_id=...)` creates two fresh
offline production environments with separate parsers, belief engines, request
histories and annotation trackers. Explicit HP provenance is own exact/opponent
percentage, including a genuine exact-100 HP owner. A v4 spec and frozen supplied
vocabulary are required. There is no simulator process or restore.

`feed_player_reference_boundary(envs, batch, own_request_deltas=...,
requested_seats=..., request_tail_verified=True)` consumes **raw owning request
line strings**, including every new wait/asymmetric request, through separate
private channels. The caller must first prove in the original ordered mixed
delta that requests form its literal tail and no later nonrequest exists.
Separated arguments cannot establish this; default is refusal. If actual data
interleaves, abort the whole comparison and propose a separately reviewed
source-index routing interface. Do not reorder, skip or refill rows.

Both seat line tuples and all owning request identities are validated before
either environment advances. The helper refuses foreign requests, exact peer HP,
private peer Pressure, aliased histories and source/provenance overclaims. Both
histories advance at every boundary; only requested owners are observed. Actual
returned v4 schema and player/showdown/opponent identities must match each owner.
An ingestion/observation failure poisons both histories, preventing partial
reference use. The helper returns complete new five-tensor observations; original
stored observations remain trusted-collector reproduction references only.

Root must freeze the same validated immutable batch for production references
and GPU consumption. Never re-query an arbitrary stateful `for_seat` provider
after validating one result. Scope/Struggle/provenance flags stay attached to
every comparison; a trusted-only parity tally must not be relabeled public.

## Remaining gates

Before actual projected comparisons, bind the complete original8419-file runtime
census (including inherited moves/items/mods), or independently bind every
relevant additional emitter. The earlier41 targeted files alone are not that
full-runtime proof. Run the entire fixed roster once through chronology,
ownership and omission guards with no filtering. Regenerate complete production
Python references, then measure GPU per-column and model gates separately.
None of those measurements occurred in this source lane. If a private owner or
missing-stream treatment remains unproved, use fresh captured per-player streams
with separately owned requests under a new reviewed collection plan.
