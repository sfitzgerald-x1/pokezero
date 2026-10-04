# Paper-policy-opponent implementation ledger

Updated October 4, 2026. Implements the staged plan in
`/Users/scott/Documents/New project/reports/paper-faithful-mcts-plan-20261003.md`.

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
checkpoint's protocol-item-narrowing flag. This is a correctness reference,
not proof that every native branch has been wired to it.

`policy_opponent.py` uses the **own policy head** of the model on this view,
validates checkpoint schema/masks/vocabulary/belief-source binding, and gathers
the full distribution in exact native-option order. Missing/duplicated actions,
legal-surface disagreement, invalid probabilities, and illegal mass refuse
without fallback. A real Python PyTorch forward with intentionally contradictory
own/auxiliary heads proves which head is read. This uses fixture weights, not a
verified champion checkpoint, and does not verify the native TorchScript path.

The reference currently requires one-snapshot checkpoints. Investment observers
and legacy-history residual observers are not implemented; configurations that
need them explicitly refuse rather than quietly receive zero annotations.
These are remaining integration requirements, not permission to change the
champion's feature flags. A supplied sampled own request must also be produced
by a certified private-side constructor; that constructor and complete bench-PP
materialization are still pending.

## Remaining stage-one integration

Do not enable this seam by feeding the auxiliary opponent head or a seat-swapped
subject observation. It requires the champion's **own policy head** on an
explicitly constructed opponent information state.

The next implementation step is native dual-perspective model integration:

1. Retain the subject's current value/prior observation unchanged.
2. Give the opponent its sampled team's legitimately known private facts.
3. Use the public-view reference to reconstruct its knowledge of the subject, not the
   subject request or full sampled engine state. Round private exact HP into the
   opponent-visible public representation where necessary.
4. Evolve that opponent observation along each branch, including newly revealed
   species, moves, abilities, items, request order, and forced-action boundaries.
5. Gather its own policy head through its exact native legal-option map.
6. Verify hidden-truth noninterference, full head/seat binding, replay, and
   engine action identity before adding an opt-in Python/native call contract.

Pending model rows must never cause uniform sampling at a child. A synchronous
provider can satisfy that invariant initially; profile its measured cost before
introducing any deferred batching optimization. Report policy forwards, draws,
full decision timing, and refusal reasons in the eventual experiment ledger.

## Fidelity and advancement ledger

| Plan requirement | Current state | Evidence needed next |
| --- | --- | --- |
| Fixed-policy opponent sampling | Native foundation implemented | Actual champion own-policy provider at root and child nodes |
| Subject PUCT and critic unchanged | Shared tree path retains selector/backup | Model-feature differential verification |
| Opponent information-state privacy | Canonical reference and tensor mutation tests implemented | Native branch observer differential tests and certified sampled-private request construction |
| Champion own-head provider | Strict Python reference implemented, fixture forward tested | Verified champion, matched native model runtime, root/child model integration |
| Exact sparse action identity | Strict gather tested | Root/branch engine mapping tests with the provider |
| Separate policy/chance randomness | Explicit sampler RNG implemented | End-to-end registered trace with both streams |
| Checkpoint/source/export/image binding | Not yet registered | Verify actual checkpoint hash and immutable receipt |
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

## Verification environment

The engine source was fetched using the repository's verified-sdist and patch
builder. Plain native unit tests do not require TorchScript. The initial
compile failed with exit 101 because a new test match arm returned the backup
value instead of unit; that test-code error was repaired before verification.
The subsequent full `cargo test --manifest-path rust/pokezero-search/Cargo.toml
--lib` completed with exit 0: 275 passed, zero failed, one existing ignored test.

The discovered laptop model runtime is PyTorch 2.12.1. This source requires
PyTorch 2.13.0 / tch 0.26.0. Do not bypass the ABI guard or call model-feature
verification complete on the older runtime. Use a matching isolated/runtime
image for that gate; preserve existing environments.

The public-view/provider tests completed with exit 0: 19 tests, including the
real Python own-head forward, in the existing Python PyTorch environment. The
isolated no-Torch environment skips that one forward test explicitly. An initial
new test fixture failed with exit 1 because it constructed a V4 checkpoint with
an impossible nonzero history budget; it was corrected to test a valid
exact-state-mask mismatch instead. Broader parser/belief/world regression tests
completed with exit 0: 267 tests, four existing environment-dependent skips.
Plain native tests were rerun: 275 passed, zero failed, one existing ignored
test. None of these results certifies model-feature native integration or
playing strength.
