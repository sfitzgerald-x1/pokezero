# Paper-policy-opponent implementation ledger

Updated October 4, 2026. Implements the staged plan in
the workspace report `paper-faithful-mcts-plan-20261003.md`.

This branch is implementation work, **not an enabled experiment or a claim of
paper reproduction**. No historical checkpoint, Kubernetes object, or shared
experiment artifact has been changed.

## Implemented native foundation

`policy_opponent.rs` samples a fixed full legal distribution independently of
opponent Q values, visits, virtual loss, FPU, and PUCT priors. The subject still
uses the existing PUCT selector. The sampler owns a separate explicit RNG seed
and binds each inferred distribution to one tree node's ordered legal surface.
It never reuses a cache across trees. Sparse action maps are strict: missing,
duplicated, out-of-range, negative, nonfinite, and zero-mass distributions fail;
there is no uniform or adversarial fallback. A sole native legal choice is
deterministic and requires no policy inference.

`traverse_with_policy_opponent` uses the existing engine chance traversal,
expansion, and backup implementation. Its provider receives the reached engine
state and parent branch key so a model integration can retrieve that node's
evolved information state. A provider failure reverses every applied branch and
removes this traversal's virtual-loss reservations before returning an error.
The incumbent traversal calls the shared core with no policy provider and keeps
its existing per-seat PUCT selections.

Focused tests cover:

- A known fixed-opponent game: risky wins against 90/10 stay/switch; robust wins
  against 50/50, from either subject seat.
- Sampling all legal actions, including a 1% arm, through a sparse slot map.
- Invalid distribution refusal, tiny positive mass, and zero-probability arms.
- Seeded replay, seat mirroring, and independence from opponent search stats.
- One inference per cached node and deterministic no-choice handling.
- Real engine traversal/state restoration and interior-provider failure cleanup.

## Implemented public-view and own-head reference

`policy_opponent_view.py` rebuilds the opponent's perspective through the
incumbent canonical parser, public belief engine, and observation encoder. Its
boundary takes only a complete public transcript and that seat's sampled own
request, not the subject request or a complete two-seat simulator state. It
removes live private split branches and projects exact HP into the registered
percentage representation. Child suffixes are projected independently so an
exact engine event cannot reinterpret an already-public root percentage. Raw
protocol spelling is retained for chronology-sensitive observer evidence.

Tests compare actual encoded categorical/numeric arrays, attention masks, and
legal masks across hidden-HP changes and seat mirrors. They also cover evolving
public move/species reveals, sparse slots, forced replacement, recharge state,
parent/sibling isolation, malformed requests, terminal refusal, and the
checkpoint's protocol-item-narrowing flag. Native callback integration now
exercises this reference at both root and reached child nodes.

`policy_opponent.py` uses the **own policy head** of the model on this view,
validates checkpoint schema/masks/vocabulary/belief-source binding, and gathers
the full distribution in exact native-option order. Missing/duplicated actions,
legal-surface disagreement, invalid probabilities, and illegal mass refuse
without fallback. A real Python PyTorch forward with intentionally contradictory
own/auxiliary heads proves which head is read. The opt-in native integration
also runs the registered champion's own head, with the same checkpoint exported
to TorchScript for the unchanged subject critic and prior path.

The reference requires one-snapshot checkpoints. Belief-narrowing investment
observers and legacy-history residual/investment observers remain unsupported
and explicitly refuse. V4 retired the history region and both pinned Tier-2
columns. The champion's narrowing switch is off, so its retained Tier-2 flags
have no encoded consumer. Tests inject nonzero residual, Choice Band, and
investment annotations and prove unchanged V4 input tensors; no champion flag
was disabled. Legacy schemas and narrowing-enabled configurations still refuse.

`policy_request.rs` constructs a complete request from only the sampled acting
`Side`: private stats, item, ability, gender, full party, sparse move slots,
exact PP, forced replacements, recharge, and explicit request order. Native
options must match that own-information legal surface exactly. The constructor
does not query the actual other seat to disclose an unrevealed trapping ability.
Complete bench PP is carried into canonical materialization.

`policy_bridge.rs` keeps immutable public prefixes per native branch, projects
exact public HP before calling Python, uses registered species display names,
evolves switch order, and corrects own PP from the root base plus the branch
move-charge ledger. The engine stops decrementing high PP; reading leaf PP
alone would be wrong. Low-PP moves are not double-charged. Every uncertain legal
surface, unsupported Transform observer, invalid probability row, or provider
exception stops the search instead of choosing a fallback.

The optional native call requires callback, independent policy seed, and
complete sampled request order together. It requires subject model priors,
disables the auxiliary opponent-prior mode, and retains the model-value leaf.
It reports opponent forwards, samples, and elapsed opponent time separately.
That time is included in the native decision deadline. Mode-off keeps the
incumbent report schema and performs no additional context copy or Dex lookup.

## Implemented high-level experimental call contract

`EngineMctsConfig(policy_opponent=True, policy_opponent_seed=<registered seed>)`
is an explicit opt-in. It requires model leaves, subject priors, strict
fallbacks, a fixed allocation, and no auxiliary opponent priors, rollout
substitution, or early-stop replay. The ordinary configuration remains off.
Initialization loads the own head from the supplied source checkpoint,
escalates fresh value-head initialization to an error, retains the calibration
fence, and requires a non-null matching belief-source binding.

Serial, parallel fixed-work, and parallel deadline dispatch all pass an
invocation-owned public-prefix callback and exact sampled request order. Each
callback has a separate vocabulary/OOV tracker, with aliases taken from the
same native encoder tables. Policy RNG seeds are derived by a versioned,
domain-separated SHA-256 split of the registered seed, battle, seat, decision,
and native chance seed. They consume no extra decision RNG draws. Duplicate
world accounting keeps actual invocations separate from belief multiplicity.

The canonical evaluator calls `eval()` and `to()` before forwarding. The shared
own-head runtime therefore has an inference lock; concurrent trees do not mutate
the module simultaneously. Lock waiting is charged to opponent and decision
wall time. This correctness-first serialization is a measured cost to profile,
not a throughput claim. Public-view construction remains invocation-owned.

Every native report must witness the requested opponent mode/seed, valid
inference/sample/time counters, and zero prior fallbacks. Fixed-work calls must
complete their exact registered iteration allocation; deadline calls retain the
existing completed/remaining-work checks. Provider failures or unsupported
world construction stop the decision even when another world is healthy.
Missing request order is a refusal, not an inferred permutation. Context-free
calls cannot use the ordinary uniform-legal shortcut. Failed experimental calls
still record whole-call wall time. Successful calls expose per-invocation
receipts and cumulative opponent counters, without adding an opponent block to
mode-off decision or statistics payloads.

## Remaining before a frozen root profile

Do not enable this seam by feeding the auxiliary opponent head or a seat-swapped
subject observation. It requires the champion's **own policy head** on an
explicitly constructed opponent information state.

1. Qualify the complete live context, public request-order walk, and sampled
   world construction on the frozen roster. A new real-champion live-opening
   gate runs both seats with fixed and belief-sampled worlds, using actual env
   observations, incremental folds, constructors, root encoding, public order,
   and native dispatch. The incumbent and opt-in construct identical worlds.
   The test injects only the registered immutable belief cache into the env's
   loader; it does not certify its publication recipe or wider observer seams.
   Separate root/child fixture gates still supply registered root inputs/order.
2. Validate observed trapping and Transform boundaries or preserve explicit
   refusals. Do not erase those roots or quietly alter their native options.
3. Finish runtime Dex/Showdown publication binding. The current loader generates
   source hash `f9e35e1fddae5064`, different from the champion's registered cache.
   Canonical cache contents excluding metadata are byte-identical, including
   all variants and resolved move/species metadata, with SHA-256
   `c718948e86320e5cafd9b22723d4f8e4c54f7ddd7a6cf8e45f020448f2196364`.
   This proves materialized-universe equality, not the original source recipe
   or an immutable image receipt; keep that distinction in the run manifest.
4. Freeze the outcome-independent profile roster, actual source/model/export
   hashes, engine fingerprint, immutable image, resource cap, and analysis.

Pending model rows must never cause uniform sampling at a child. A synchronous
provider can satisfy that invariant initially; profile its measured cost before
introducing any deferred batching optimization. Report policy forwards, draws,
full decision timing, and refusal reasons in the eventual experiment ledger.

## Fidelity and advancement ledger

| Plan requirement | Current state | Evidence needed next |
| --- | --- | --- |
| Fixed-policy opponent sampling | Root/child native callback and high-level opt-in tested | Registered root profile and complete live-context qualification |
| Subject PUCT and critic unchanged | Shared tree/critic path; subject root priors differential-tested; full live-opening construction matches incumbent | Expanded frozen-root comparison and profile |
| Opponent information-state privacy | Side-only constructor and pre-callback public projection tested | Expanded frozen-root observer coverage, trapping and Transform |
| Champion own-head provider | Real champion gate passes without changing feature flags | Runtime/source publication binding and high-level integration |
| Exact sparse action identity | Constructor, request order, branch switches and strict gather tested | Registered native root panel including refusal surfaces |
| Separate policy/chance randomness | Explicit sampler RNG, duplicate native trace, and versioned per-world seed split tested | Frozen run manifest and profile replay |
| Checkpoint/source/export/image binding | Actual checkpoint hash verified; local cache content pinned | Reproducible engine/export receipt and immutable image |
| Fixed-work and matched-time root profiles | Not launched | Frozen outcome-independent panel and bounded readout |
| Continuation screen | Not launched | Profile prerequisite and preregistered 16-root roster |
| Trajectory-level determinization | Controlled deviation remains | Stage-four audit/implementation, only if screen advances |
| Tree persistence and pruning | Not implemented | Stage-four information-state identity and memory tests |
| Published exploration/prior exponent | Unresolved | Exact thesis/author-source extraction and registered choice |
| Game pilot and independent confirmation | Not launched | Prior stage passes, throughput check, frozen manifests |
| Foul Play claim | None | Independent confirmation followed by pinned opponent benchmark |

No expensive screen or game cohort is justified by the sampler tests alone.
Keep the original plan's ceilings and gates. A stopped or failed stage is a
documented decision, not permission to redraw seeds or restart a full cohort.

## Verification evidence

An isolated worktree environment uses Python 3.13.13 and PyTorch 2.13.0,
matching tch 0.26.0. No ABI guard was bypassed or existing environment replaced.
The full engine source builder verified its sdist and applied the repository's
78 engine patches; both consumers rebuilt and all builder behavior probes
passed. The model-feature wheel build subsequently completed with exit 0.
The rebuilt native source fingerprint is
`a9f3f8197e8c65ce30f06c9870a6c350e5f725ec7a19f4832707b283066ccfdd`.

Latest native model-feature unit gate: 291 passed, zero failed, one existing
ignored test. Python integration/regression gate: 817 tests, exit 0, one
existing skip, including the opt-in real champion class in both seats, engine
search, legal mapping, parser/belief/world, and V4 annotation
retirement gate. Native-versus-Python TorchScript output parity was exact
(maximum absolute difference zero). These gates do not establish strength.

The portable CI-selected own-policy suite contains 52 tests with no skips; the
private champion gate is an additional explicit local check. Six static-parser
tests verify inherited class selection, imported fixtures, override deduplication,
and refusal of unknown/dynamic test members. The workflow's 38 exact-count guards
now verify against source without importing native modules. A seed-stream gate
checks the actual native chance-seed positional and proves no additional RNG
draws are consumed in serial, parallel, fixed-work, or deadline dispatch.
The additional champion live-opening gate uses the first committed historical
battle, chosen without candidate outcomes. Its initial run failed with exit 1
because the test used `seed` instead of the corpus's `battle_seed`; the fixture
field was corrected before the passing full-path gate. No production fallback
or provenance check was loosened to make the gate pass.

The downloaded iteration-9375 champion is 40,083,798 bytes and verifies as
`0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be`.
Its own-head provider uses V4, one snapshot, zero history budget, original
Tier-2 flags, and `investment_belief_narrowing=False`. The source cache has
registered metadata hash `f5a5265143d423af` and actual file SHA-256
`82950121aed1d3cc0b30c91debf7293b08ba101dea71df16ce95f1a83024460b`.
Checkpoint loading escalates fresh/random value-head initialization to an
error; the original calibration fence also runs. The explicit local gate uses
these bytes rather than substituting fixture weights or setting the source
hash to null.

The native integration initially failed because engine-normalized
`deoxysdefense` did not match the checkpoint's canonical species token. The
repair derives display names from registered Dex tables and tests form names,
punctuation, and preservation of unrelated event fields; no OOV assertion was
removed. Earlier constructor/PP test and compile failures were repaired before
the passing gates above. All nonzero verification exits were treated as
failures, not accepted as skips or partial evidence.

No cluster experiment, training job, root screen, game pilot, finalizer, or
deployment was launched by these changes. No historical object or shared
artifact was modified. The full staged goal remains incomplete and active.
