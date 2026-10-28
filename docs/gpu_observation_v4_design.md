# GPU observation v4 design

The GPU policy must receive the champion's player-relative observation: 23 tokens, 132 numeric columns and 41 categorical columns, with the champion's own vocabulary and feature masks. The existing `gpu_engine/observation.py` is an omniscient compute placeholder. The new encoder will fold public events and each player's private request into persistent device state, then produce the same v4 columns as Python. The [complete column port plan](gpu_observation_v4_columns.md) names every Python writer and its device implementation.

## Source contract

[`showdown.py`](../src/pokezero/showdown.py) defines parsing, player-relative normalization, the v4 numeric projection, token order, request legality and column writers. [`belief.py`](../src/pokezero/belief.py) defines public reveal and exclusion ledgers, current versus original item state, PP charging and candidate narrowing. [`transitions.py`](../src/pokezero/transitions.py) supplies tendency aggregates even though v4 has no history tokens. [`investment.py`](../src/pokezero/investment.py) supplies damage-evidence narrowing. [`randbat.py`](../src/pokezero/randbat.py) supplies candidate variants and spread semantics. [`category_vocab.py`](../src/pokezero/category_vocab.py) supplies checkpoint label IDs. The encoder must use the checkpoint's resolved masks and vocabulary rather than current defaults.

The protocol adapter and engine adapter share one typed event contract. The Python parser remains the parity reference; existing Rust observation code is an implementation guide and a second comparison, not an authority that overrides Python's shipped champion inputs. Mechanical disagreements with Showdown are recorded separately from encoder disagreements on an identical event stream.

## Event schema

Use a versioned ordered stream, `pokezero.public_events.v1`. Each event carries battle identity, monotonically increasing sequence number, turn, phase, event kind, actor side and stable party slot, target side and stable party slot, and kind-specific arguments. Identity is an encounter/party identity, not a species name: Transform, Illusion-like replacement messages and duplicate species must not merge records. The host protocol adapter resolves protocol idents into these identities. Unknown opponent party slots have encounter identities until a public mapping exists; they do not borrow private team order.

The packed device representation uses sixteen int32 words per event: kind, sequence, turn, phase, actor side, actor slot, target side, target slot, and eight payload words. Float payloads use their float32 bit representation; tagged payload layouts are checked before dispatch. Static IDs have separate species, move, ability, item, status, volatile, weather, effect-source and vocabulary namespaces. An ID in one namespace cannot be substituted for another. The host retains raw protocol lines in remote parity records so no information is silently lost while validating this packing.

| Event family | Required payload and semantics | Python source |
| --- | --- | --- |
| Battle and decision boundaries | Start, turn, blank `\|` action-chunk end (`ACTION_CHUNK_END`), exact `\|upkeep`/residual boundary, decision phase, completed round, terminal outcome; preserve order within the turn. Outcome is a label, never an input feature. | `parse_showdown_replay`, `normalize_for_player`; `transitions` extraction |
| Arrival and identity | switch, drag, replace; public species/form, level, gender, public HP condition, arrival reason including Baton Pass | `_pokemon_from_public_line`, `_same_public_pokemon`, public state fold |
| Action and prevention | move, target, called-move source, miss, fail, immunity, cant reason, recharge and preparation; distinguish declarations from executed/called moves | `_public_event_from_line`, public state fold, belief move/PP ledger |
| HP and status | damage, heal, sethp, faint, status, curestatus, cureteam; preserve numerator, denominator and effect source, including counterpart for drain/recoil | `_updated_public_condition`, belief HP snapshots and residual phase |
| Stats and volatiles | boost, unboost, set/clear/invert/copy/swap boosts; start/end/singleturn/singlemove/activate, including substitute and trap clocks | public state fold and exact-state belief ledgers |
| Items and abilities | reveal, consume, remove, transfer, Trace/current ability; carry original versus currently held item and causal source separately | `PublicBattleBeliefEngine`, `_confirmed_ability_from_event`, `_trace_copy_source` |
| Field and side state | weather/source/duration, side start/end, Wish, Spikes, sleep clause; hazard and item-removal credit | field fold, `_encode_field_exact_state`, `_encode_field_credit_features` |
| Public evidence | move/ability/item exclusions, pressure PP charge, non-proc evidence, damage-derived investment pins and opponent action tendencies | `PublicBattleBeliefEngine`, investment and transitions |

A blank `|` protocol line is a semantic action-chunk boundary: `transitions.py` closes the acting-move window there as well as on exact `|upkeep`. Emit `ACTION_CHUNK_END` from both adapters; do not discard it as decorative whitespace. This ordering controls damage attribution and tendency features.

Wire tags such as `[from]` and `[of]` are semantic inputs. The adapter must map every tag read by the Python fold into a typed payload before that event can pass admission. New unrepresented tags or event kinds return an explicit unsupported code; they are not treated as no-ops. Decorative messages may be ignored only after source inspection shows that no observation, belief or tendency consumer reads them.

Private requests use a separate `pokezero.private_request.v1` input, scoped to one player: own team identity, actual stats, current HP/status/item/ability, active move order and PP, disabled moves, trapping/forced replacement, legal choices and canonical switch targets. The opponent's request is inaccessible to the public encoder. Request boundaries may arrive between public chunks; neither a missing request nor a waiting seat is converted into a fabricated legal action set.

Process fixed chunks of 64 events with an event-valid mask and a continuation cursor. Overflow drains another chunk before encoding a decision; it never truncates the stream. Candidate/volatile output buckets retain Python's documented truncation and diagnostics. The event transport itself has no lossy truncation. Device event counters and committed state let a repeated chunk be rejected rather than applied twice.

## Persistent state and beliefs

Keep public battle state once per battle, own private state per side, and beliefs per observer and opponent slot. Store clocks, public HP fractions and denominators, identity/form, boosts, status, active volatiles, move/PP evidence, original/current item facts, Trace copy, last executed move, recharge/Truant/choice/arrival state, last-round damage, hazard/item credit, and marginal and matchup tendency counters. Port the state transitions in event order; recomputing them from the final engine state would lose residual ordering and public information boundaries.

Represent a species' candidate set by bits over its stable variant table. Under the inspected Gen3 source hash `f5a5265143d423af` there are 220 species and 1,682 variants; the maximum is 44 for Swalot. Two uint32 words per slot therefore suffice for this pinned source. A table with more variants changes the static shape and must be admitted explicitly. Species unknown, no source, inconsistent reveal evidence, and an empty hard-evidence mask are distinct states. Preserve the raw reveal mask and conflict status separately from the effective candidate set used by the champion.

Static variant rows include move set, original item, original ability, generator spread, and stable variant identity. Precompute predicate masks and stat tables from the same Python source. Reveal/exclusion predicates compute a hard-evidence mask. Match `Gen3RandbatSource.summarize` exactly: if reveal filters reject every known variant, expose the full species pool and uncertainty 1.0, with an explicit inconsistent-evidence flag. Keep the empty hard-evidence mask for diagnosis. Store damage-evidence pins separately by variant identity. `narrow_candidate_variants` rejects an empty incoming pin without incrementing the conflict counter. It rejects a nonempty pin disjoint from the standing pin while retaining the standing pin and incrementing the conflict counter. `_apply_variant_pin` skips pins on an inconsistent source summary; when a pin is disjoint from a consistent summary, keep that summary and count the conflict. Otherwise intersect and preserve Python's uncertainty scaling and filtered possible-value ordering. Thus standing pins are monotone, but the effective public candidate set follows the source's fallback rules. Consumed or swapped items do not erase original-set evidence. Transformation changes battle state and copied stats without rewriting the original variant identity.

Expose the variant mask, source/species identity, unresolved reason and prior weights as encoder outputs. With an RNG, the existing sampler chooses uniformly among surviving candidates; without an RNG it enumerates source-order mixed-radix combinations. Preserve both modes and source ordering. Use equal static prior weights to represent the RNG mode and normalize only over surviving bits. These weights are a representation of the current sampler, not a claim about the generator's true probability distribution. Do not introduce a new weighted posterior during the champion parity gate. Belief tokens and opponent-set labels may consume this output only in the appended v5 surface.

## Memory at batch 8192

These are proposed storage bounds for both player views, before transformer activations and compiler temporaries. Static tables are shared across the batch. The 96 integer and 32 float public-mon words and 80 side words are allocation budgets; implementation must publish its actual census and cannot claim measured memory from these estimates.

| Buffer | Shape and dtype | MiB |
| --- | --- | ---: |
| Hard evidence, standing pin and effective masks | 3 × 8192 × 2 observers × 6 slots × 2 uint32 | 2.25 |
| Materialized posterior weights if needed | 8192 × 2 × 6 × 44 float32 | 16.50 |
| Public and mon ledgers budget | 8192 × 12 mons × 128 four-byte words | 48.00 |
| Side and private request ledgers budget | 8192 × 2 × 80 four-byte words | 5.00 |
| One event chunk | 8192 × 64 × 16 int32 | 32.00 |
| Numeric observation for both seats | 8192 × 2 × 23 × 132 float32 | 189.75 |
| Categorical observation for both seats | 8192 × 2 × 23 × 41 int32 | 58.9375 |
| Token types | 8192 × 2 × 23 int32 | 1.4375 |
| Attention mask | 8192 × 2 × 23 bool | 0.359375 |

The shared float32 prior vector is 6,728 bytes. Avoid storing dense posterior weights unless a consumer needs them; the mask and shared priors define them exactly. The planned buffers total about 354.23 MiB with dense posterior weights, or 337.73 MiB without. These totals exclude action metadata, predicate/static spread tables, long-lived evidence records, temporary scans, compiled executables, and model weights/activations. Benchmark actual peak memory at both required batches; do not infer fit from this table alone.

## Encoding and information boundaries

Token order is field 0, own party 1 through 6, opponent encounter slots 7 through 12, nine canonical actions 13 through 21, and tendency 22. Physical numeric indices come from `V4_NUMERIC_LAYOUT_GROUPS` and `v4_numeric_index`; writer constants cannot index the final tensor directly. Categorical labels use the loaded checkpoint vocabulary, including its exact unknown/padding behavior. Resolve move type, variable power and effects from the same dex rules as the Python writer. No damage matchup scalar is added.

A sampled search world may affect the transition but cannot expose hidden state to our observation. A leaf uses public state after that transition, our private state, and a copy of the root belief narrowed only by that branch's public reveals. The root belief is immutable across worlds. Cache keys include observer, public state, own private state and belief identity; they exclude unrevealed sampled opponent fields. Two worlds producing the same public transition must produce identical own observations. The opponent reply policy may use that opponent's own request, with its own view of our unrevealed set.

For the privileged probe, record both seats' public observations and own-token blocks with battle/turn/decision identity and outcome. An explicit privileged view swaps the other seat's own tokens into the opponent block; it never silently changes the public policy input. Pair only matching decision boundaries, preserving waiting/forced-replacement semantics. Split refits by complete battle before feature extraction to prevent two seats or adjacent decisions leaking between training and held-out data.

## Implementation sequence and acceptance

1. Build static label/variant/spread tables and a lossless protocol-to-typed-event adapter. Implement the persistent public/private/belief fold and all columns in the companion table. Freeze schema, source hashes, feature masks and vocabulary with each corpus.
2. Compare at least 10,000 recorded decisions from identical protocol prefixes and private requests. Every categorical column, token type, attention bit, legal action and canonical action identity must match; numeric columns must agree within 1e-6. Report differences by physical column and token role, including rare boundaries. No accepted divergence is specified by this design; any proposed exception needs a named column, reason and effect before it is admitted.
3. Run the champion's FP32 forward on the JAX inputs against the existing Python path: policy argmax agreement at least 99.9%, values within 1e-4, with ties, masks and any non-finite outputs reported. Play 1,000 games against Foul Play at the Python champion's rate within noise, using the registered paired/full-game readout.
4. Emit the same typed stream from the JAX engine. Compare streams where engine and Showdown mechanics agree. Measure completed self-play throughput and peak memory using the real encoder and bf16 forward. The bf16 admission is policy argmax agreement at least 99.5% versus FP32, with KL reported; timing synthetic placeholder observations does not pass this gate.

The champion module must support configurable width, layers, normalization, token-type feedforward experts and value-head variants while preserving the existing checkpoint path. A3 times the three specified shapes at batches 2,048 and 8,192. The new RMSNorm/MoE/value-head paths need their own numerical reference checks; they are not implied by champion forward parity.

Only after the engine-fed gate passes does the fidelity canary fine-tune on a few hundred thousand engine games. Training arms remain gated on that observation work and the Phase A reading. If encoding exceeds 25 percent of a step at batch 2,048 with bf16, report the measured breakdown and propose a simplified observation with an estimated speedup before changing its contract.
