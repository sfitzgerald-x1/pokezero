# Paper-policy-opponent implementation ledger

Updated October 4, 2026. Implements the staged plan in
the workspace report `paper-faithful-mcts-plan-20261003.md`.

This branch is default-off implementation work and a completed, explicitly
opted-in diagnostic, **not a deployment or a claim of paper reproduction**.
No historical checkpoint, Kubernetes object, or shared experiment artifact has
been changed. The current study disposition is **PARK_UNRESOLVED**; the
[October 4 readout](paper_policy_opponent_profile_readout_20261004.md) supersedes
the historical pre-launch status entries below.

## Implemented native foundation

`policy_opponent.rs` samples a fixed full legal distribution independently of
opponent Q values, visits, virtual loss, FPU, and PUCT priors. The subject still
uses the existing PUCT selector. The sampler owns a separate explicit RNG seed
and binds each inferred distribution to one tree node's ordered legal surface.
It never reuses a cache across trees. Sparse action maps are strict: missing,
duplicated, out-of-range, negative, nonfinite, and zero-mass distributions fail;
there is no uniform or adversarial fallback. A sole native legal choice is
deterministic, but still requires the private-safe request and legal-surface
certification before the provider can omit policy inference.

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
- One certification per cached node, inference only for multiple choices, and
  deterministic certified no-choice handling.
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
It reports provider certifications, multi-choice evaluations, samples, and
elapsed opponent time separately. For the canonical own-head provider each
multi-choice evaluation is one model forward; arbitrary diagnostic callbacks
are not a physical model-forward witness.
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

## Three-arm measurement foundation

`policy_opponent_profile.py` compares raw policy, incumbent MCTS, and the
own-policy-opponent candidate at a shared live replay boundary. Fixed-work and
matched-deadline rows are distinct; invocation receipts must conserve completed
and remaining iterations. The outer timer includes fresh observation/context
construction, inference/search, selection, and Showdown choice serialization.
Checkpoint loading, export, and historical prefix replay are preparation, not
decision latency. Execution order rotates by root and timing mode.

The raw arm uses canonical local, one-snapshot deterministic masked argmax.
Its opt-in probability receipt reuses the already computed own-head row rather
than adding a forward; ordinary decision metadata remains unchanged. A timing
sink measures its real encode/forward work and it reports zero search work.
Checkpoint, vocabulary, and non-null belief-source bindings remain required.

Source-prefix, action, sampling, and cleanup refusals retain root identity and
all available comparison rows. No failed root is redrawn. A row's `COMPLETE`
means execution/witness validation only: the foundation explicitly marks
roster/runtime qualification as pending. It is not a
launcher, a valid experiment manifest, or a strength result.

Thirteen focused foundation tests pass, including real PyTorch raw inference,
default-off metadata, seed separation, full legal probability/action identity,
timing-boundary exclusion of preparation, exact-work versus deadline prefixes,
and retained failed-arm telemetry. The portable suite now has 66 tests, with
no skips. A focused private live-champion test also passes for both seats:
actual public replay selects the canonical raw action and full probability row,
with a model hook witnessing exactly one forward. Independent foundation
review found no actionable defect. Wider frozen-root replay qualification and
measured search costs remain unrun.

## Remaining before a frozen root profile

Do not enable this seam by feeding the auxiliary opponent head or a seat-swapped
subject observation. It requires the champion's **own policy head** on an
explicitly constructed opponent information state.

1. Qualify the complete live context, public request-order walk, and sampled
   world construction on the frozen roster. A new real-champion live-opening
   gate runs both seats with fixed and belief-sampled worlds, using actual env
   observations, incremental folds, constructors, root encoding, public order,
   and native dispatch. The incumbent and opt-in construct identical worlds.
   The gate now rebuilds its belief source from the configured clean Showdown
   runtime and uses the real env cache loader without injection. It does not
   certify an immutable image publication or wider observer seams.
   Separate root/child fixture gates still supply registered root inputs/order.
2. Validate observed trapping and Transform boundaries or preserve explicit
   refusals. Do not erase those roots or quietly alter their native options.
3. Publish the verified runtime binding in an immutable image. A fresh clean
   Showdown checkout at `f76228a1354b5d0f307ca2d16101294ad3a2308b`, installed
   with `npm ci --ignore-scripts --no-audit --no-fund` and built with `node build`,
   independently re-enumerates (disk cache disabled) source hash
   `f5a5265143d423af`, matching the champion. Its executable Showdown dependency
   hash is `93f81c8eae3d9f769f579a3be5e87bc0d62df39c0c0166bd47a228238188e4b3`,
   exactly the completed practical pilot's runtime. The normalized
   `Gen3RandbatSource.to_payload()` content excluding metadata is identical to
   the registered cache, SHA-256
   `42f7cda15e712499a6cc4b2132b28e06bcddea4f37dc0c40fe522a9472a8a75a`.
   The earlier `f9e35e1fddae5064` came from the separate user-edited checkout;
   its edits remain untouched. Local reproducibility is now proven, but an
   immutable image receipt still remains outstanding.
4. Freeze the outcome-independent profile roster, actual source/model/export
   hashes, engine fingerprint, immutable image, resource cap, and analysis.

The selection-only roster is now pinned in
`docs/paper_policy_opponent_roster_20261004.json`, file SHA-256
`fa76bc9966bba48f0dd474a2450b4c362e49ac1452db18c39b084a256d30ff7a`.
It contains 32 roots: the first eight source paired seeds, both seats, and
decision indices 0 and 9. No candidate actions or outcomes were used to select
them. The proposed continuation subset is all sixteen index-9 roots across the
eight clusters, not the action-changing subset. All 138 available preceding
decision records are separately byte-bound. Six source histories have uncaptured
prior indices; the initial attempt to require every index failed with exit 1
at seed 2026100100/p1/index 6. These are retained as explicit missing-record
inventory, not repaired by inventing actions or replacing roots. Exact public
replay/coverage qualification remains required: this roster is neither a valid
run manifest nor proof that all roots can be searched. The loader checks the
roster byte hash, exact ordered denominator, continuation subset, prefix
inventory, refusal/no-redraw rules, source file identity/bytes, and original
source-shard terminal. Six focused tests cover the positive/negative boundaries.

Pending model rows must never cause uniform sampling at a child. A synchronous
provider can satisfy that invariant initially; profile its measured cost before
introducing any deferred batching optimization. Report policy forwards, draws,
full decision timing, and refusal reasons in the eventual experiment ledger.

## Fidelity and advancement ledger

Reference check: the [official thesis PDF](https://dspace.mit.edu/server/api/core/bitstreams/b13e7ad7-b176-4a0e-aa8b-24b7aa0634c3/content)
matches MIT's published MD5 `b9b73e45eb45bddad400ae207ce8720e`.
Page 21 specifies `Q + alpha * P**beta * sqrt(M)/(N+1)`, with
`alpha,beta` in `[0,1]`, and terminal values `+1/-1/0`. Sections 3.2.2-3
(pages 26-28) describe per-trajectory hidden completion, ten rejection attempts
before an incompatible forced completion, twenty workers exchanging statistics
every ten rollouts, and persistent statistics pruned by total faint count.
Text extraction plus rendered-page inspection verified those rules. The full
PDF search found no selected numerical MCTS alpha/beta; its appendix table
contains PPO parameters instead. Exact settings remain unresolved.

Implementation inference, not a reported thesis setting: for our `[0,1]`
win-value scale, the affine conversion `Q_paper = 2*Q_native - 1` requires
`c_native = alpha/2` when the prior term is identical. Copying an exploration
constant without this conversion is not equivalent. Stage one's incumbent
constant remains unchanged to isolate opponent modeling; a later faithful
variant must register its scale and prior exponent separately.

| Plan requirement | Current state | Evidence needed next |
| --- | --- | --- |
| Fixed-policy opponent sampling | Root/child native callback and high-level opt-in tested | Registered root profile and complete live-context qualification |
| Subject PUCT and critic unchanged | Shared tree/critic path; subject root priors differential-tested; full live-opening construction matches incumbent | Expanded frozen-root comparison and profile |
| Opponent information-state privacy | Side-only constructor and pre-callback public projection tested | Expanded frozen-root observer coverage, trapping and Transform |
| Champion own-head provider | Real champion gate passes without changing feature flags or injecting a belief cache | Immutable runtime/source image receipt and wider roots |
| Exact sparse action identity | Constructor, request order, branch switches and strict gather tested | Registered native root panel including refusal surfaces |
| Separate policy/chance randomness | Explicit sampler RNG, duplicate native trace, and versioned per-world seed split tested | Frozen run manifest and profile replay |
| Checkpoint/source/export/image binding | Actual checkpoint verified; clean rebuilt Showdown runtime matches the historical pilot and champion | Reproducible engine/export receipt and immutable image |
| Fixed-work and matched-time root profiles | Not launched | Frozen outcome-independent panel and bounded readout |
| Continuation screen | Not launched | Profile prerequisite and preregistered 16-root roster |
| Trajectory-level determinization | Controlled deviation remains | Stage-four audit/implementation, only if screen advances |
| Tree persistence and pruning | Not implemented | Stage-four information-state identity and memory tests |
| Published exploration/prior exponent | Formula and value-scale difference verified; selected numerical alpha/beta unresolved | Author-source settings or one explicitly justified registered choice; no sweep |
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
The latest rebuilt native source fingerprint is
`4bf56184eeddb642131d960609ed482a717991373f4d435c50a5b01f44263de0`.

Latest native model-feature unit gate: 291 passed, zero failed, one existing
ignored test. Python integration/regression gate: 817 tests, exit 0, one
existing skip, including the opt-in real champion class in both seats, engine
search, legal mapping, parser/belief/world, and V4 annotation
retirement gate. Native-versus-Python TorchScript output parity was exact
(maximum absolute difference zero). These gates do not establish strength.

The earlier portable CI-selected own-policy suite contained 53 tests with no skips; the
private champion gate is an additional explicit local check. Eight static-parser
tests verify inherited class selection, imported fixtures, override deduplication,
and refusal of unknown/dynamic test members. The workflow's 39 exact-count guards
now verify against source without importing native modules. A seed-stream gate
checks the actual native chance-seed positional and proves no additional RNG
draws are consumed in serial, parallel, fixed-work, or deadline dispatch.
The additional champion live-opening gate uses the first committed historical
battle, chosen without candidate outcomes. Its initial run failed with exit 1
because the test used `seed` instead of the corpus's `battle_seed`; the fixture
field was corrected before the passing full-path gate. No production fallback
or provenance check was loosened to make the gate pass.

The earlier additional gate ran all ten champion-class tests with the clean
pinned runtime and no live-source injection: exit 0. The 52 portable tests
also pass. The first remote CI run failed (exit 1) on two metadata/guard
defects: the expected harness closure omitted the two new imported provider
modules, and the older count scanner dropped class targets when a module was
also selected. The repair includes every target, strictly validates class and
method selection, retains the independent module/coverage pins, and adds two
negative/positive selector controls. All 74 provenance, count-parser, and
re-adjudication tests pass locally; all 38 workflow count guards re-derive.
Required remote CI must still pass before review/merge or experimental advance.
The corrected remote run passed those two checks, then failed with exit 1 on
the maintained terminal register's old engine-input count/fingerprint and
workflow-site counts. The register is re-derived at the unchanged current main:
15 crate sources, 99 hashed inputs, source fingerprint `a9f3f8197e8c65ce…`, and
39 guarded executable unittest sites including the new roster gate. Historical
sweep fingerprints, reserved seeds, and ratification statuses are unchanged.
The expanded local verification ran 133 roster, selector, re-adjudication,
terminal-register, and harness-provenance tests: exit 0. The installed-engine
check independently confirms the rebuilt 78-patch fingerprint remains current.
The new six-test roster suite also passes with Python site packages disabled,
qualifying the bare-checkout fast-CI path. One local run failed because a
maintained-register sentence changed the case expected by its prose pin; the
sentence was corrected and all 133 tests rerun, rather than weakening the pin.

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

### October 4 review repairs and current verification

An independent Codex review reproduced a real hidden-trapping refusal bypass:
when the native opponent surface had only one move, sampling skipped the
side-only request constructor. A hidden Shadow Tag ability could therefore
suppress switches without certification. Every new node now certifies its
request, including singleton and WAIT positions. The own-head provider omits a
network forward only after validating the complete view, checkpoint/source
contract, ordered action map, and exact observation legal mask. Tests retain
the refusal rather than accepting a full-state trapping disclosure.

The same review identified misleading work telemetry after this correction.
Provider calls and multi-choice model evaluations are now separate native
counters, high-level per-invocation receipts, and aggregate statistics. Missing,
negative, or inconsistent counters refuse. Independent verification against
the final rebuilt wheel passed all 53 portable tests, no skips, and a canonical
singleton search completed 64 traversals with one provider certification, zero
evaluations, and zero neural forwards. The hidden-trapping regression also
passed. Both review findings are closed; no new actionable finding was reported.

Final local verification at the fingerprint above:

- Both installed engine consumers rebuilt successfully, their source stamp is
  current, and all builder behavioral probes pass. No ABI guard was bypassed.
- Native release model-feature gate: 291 passed, zero failed, one existing
  ignored test; exit 0.
- Own-policy portable and real-champion combined gate: 63 tests, no skips;
  exit 0. This uses the clean pinned Showdown runtime and actual champion bytes.
- Replay/lattice, roster, seed registry, corpus census, terminal register,
  selector, current-source deadline wrapper, and provenance gate: 244 tests;
  exit 0. All 39 workflow count guards independently re-derive.

Remote run 37195088366 at the previous head failed the seed-registry and mass
gates (exit 1). Its other listed checks, including harness provenance, succeeded.
The failure causes were the frozen gameplay roster being misclassified as a
fidelity sweep and the corpus census retaining 406 after the roster became the
407th document. The classification repair is exact-path and SHA-256 bound,
parses the document first, and does not exempt copied or changed rosters. The
four fidelity bands, their reserved seeds, and all historical sweep evidence
remain unchanged. The four new registry tests bring that gate to 45. The one
new singleton provider test brings the portable gate to 53; its legacy scanner
test's stale 52 expectation initially failed, then was corrected to 53 and the
244-test gate rerun.

Remote run 37196476682 at `6e383b93` completed with exit 1 in its mass gate.
The other seven prerequisite jobs succeeded. Its 381-test arbiter suite had
two failures: a model-depth source guard's 900-character window stopped before
the real accumulation after opponent telemetry was added, and the maintained
rollout mutation battery still bound the previous `engine_search.py` bytes.
The former now uses a structural checker with padding, removed-write,
valueless-annotation, and uncalled-closure negative controls. Independent
review's declaration false positive was repaired and independently closed.
The checker explicitly proves syntactic instrumentation, not that every runtime
branch executed. The latter was repaired by rerunning the actual mutation
harness, never hand-editing evidence hashes: all 66 mutants were applied and
killed, with zero survivors, unapplied mutants, or did-not-run verdicts, and
all seven classifier controls produced their required verdicts. Source files
were restored byte-for-byte. After hardening the checker, its four affected
depth mutants were rerun and killed. The first attempted sweep refused before
edits (exit 1) because pytest was absent; pytest was installed only in this
isolated environment before the successful sweep.

Current final local verification: 77 portable/private-champion tests, 381 tests
in the exact previously failed remote arbiter/calibration suite, and the
six-test depth-instrumentation class all pass with exit 0 and no skips. The
native content fingerprint remains current; no native source changed in this
measurement-foundation step. Required CI on the next published head remains
pending, not superseded by these local passes.
The 244-test focused replay/registry/provenance gate and all 39 workflow count
guards have been rerun successfully against the measurement foundation.
The broader neural-policy regression suite also passes: 238 tests, ten existing
dependency/artifact skips, exit 0. Those skips are not counted as live-champion
coverage; the explicit 77-test gate above has none.

An exploratory legacy deadline-qualification runner test failed (exit 1) on
its intentional historical engine-fingerprint fence. That older qualification
is not evidence for this new engine, and its pins were not rewritten. The
current-source deadline-profile wrapper tests pass; a newly registered actual
profile is still required. Two earlier test invocations also exited 1 because
they named nonexistent modules; the discovered correct modules are included
in the passing 244-test gate. These are not claims of an all-repository pass.

The timing/replay adapter now accepts the explicit policy-opponent mode and
unsigned 64-bit seed, rejects conflicting arms before materialization, and
leaves the incumbent's argument shape unchanged. This forwards the opt-in;
it does not complete the raw/current/candidate three-arm harness, run the
frozen roots, or qualify their replay histories.

No cluster experiment, training job, root screen, game pilot, finalizer, or
deployment was launched by these changes. No historical object or shared
artifact was modified. The full staged goal remains incomplete and active.

### October 4 completed joint-action measurement

The default-off native measurement records node-local action pairs only after
their backups complete. Reservations and selection collisions are not completed
work. Root pair counts conserve actual traversals and both seats' visit marginals;
tree-wide counts are node/pair incidences, not a union across hidden worlds.
Collapsed belief multiplicity does not multiply compute receipts. Both search
arms opt in for the bounded profile, retain each native invocation separately,
and refuse malformed receipts or a failed world rather than hiding it behind a
healthy sibling. The raw arm remains unchanged.

The first rebuilt-wheel verification failed with exit 1: six of the 77 tests
errored because the new validator compared native-option-ordered visits with
visit-sorted report rows. Independent review reproduced the same P1 in four
enabled-wheel tests. The repair records explicit option labels and stable
report-sort indices, binding displayed moves and visits to native option identity
even for tied visits or duplicate display labels. It does not reorder the existing
report, sort away identity disagreements, or weaken marginal conservation.

Both installed engine consumers rebuilt successfully, followed by the model-feature
wheel and a fresh two-consumer artifact attestation. Current source fingerprint:
`2b019d6667a5bd36a741f55370e557c3f82414ab580bc0a4a212f21bc5928547`.
All builder behavioral probes pass. The final native gate has 293 passes and one
existing ignored test (exit 0); 37 profile/engine/lattice tests, 99 maintained
register/count/selector tests, and the 77-test portable/private-champion gate
pass (exit 0, no skips). Native on/off controls retain identical choices,
backups, root priors, depth, and evaluator counts; deadline tests charge the
receipt to actual native work. These checks do not qualify the frozen panel.

Remote run 37199006417 on published foundation `c1fa0eb63ea19dccd4c22922c3bbe901259e0c9f`
completed successfully, including the mass gate and `gate-status`; the other
PR checks also succeeded. That result did not certify the subsequent native
measurement change. Independent re-review closed the ordering P1 with no
new actionable findings; its four enabled-wheel canonical/native fixture tests
pass with exit 0 and no skips. Private-champion paths were verified by the main
77-test gate, not independently rerun by this reviewer. The fresh full mutation
sweep completed with exit 0: all 66 mutants applied and killed, no survivors,
unapplied mutants, or did-not-run defects, and all seven controls produced their
expected verdicts. Source files were restored byte-for-byte. The 381-test affected
arbiter/calibration suite, 247 focused replay/registry/provenance tests (including
the three new prefix-preflight tests), and all 39 count guards pass afterward.
Frozen-roster replay qualification, source-bound
runtime publication, the registered profile manifest, and the actual root screen
remain outstanding. No experiment has been launched by this measurement work.

### Frozen source-prefix preflight

Read-only access was verified on OLFUSA through the explicitly pinned `kx`
context (its minified context and cluster are both `olfusa`; the global default
context remains unrelated). The registered source shard's durable terminal
matches SHA-256 `dde2d620c973cd330014b853bfb29ea0a1960b07fbfa6e7773059f8903c96f78`.
Only the 155 distinct required files were copied to local scratch
`/tmp/pokezero-paper-root-source.PUme82`, not the full source cohort.

The frozen-roster loader and source verifier check all file hashes and source
identities against roster SHA-256
`fa76bc9966bba48f0dd474a2450b4c362e49ac1452db18c39b084a256d30ff7a`.
All 154 distinct public records parse and retain their canonical decision IDs.
The source-prefix preflight keeps all 32 registered roots: 31 prefixes validate,
with two explicitly source-owned repairs, and one refuses:

- Seed `2026100104`, p2, decision index 9, decision ID
  `d81daca99c6b1c0ff42357f84c9b1676706bd001afd8fe37dbddf87b20650a67`:
  `unresolved_public_event_for_non_source_player`.

That root is not replaced or silently dropped. This is source-byte and prefix
evidence only: no live observation equality, native root eligibility, timing,
candidate action selection, continuation outcome, or strength is qualified by
this preflight. Every source/shared artifact and existing Kubernetes object
remains unchanged.

The reusable read-only checker is
`scripts/qualify_policy_opponent_source_prefixes.py`, backed by
`policy_opponent_source_prefixes.py`. It rechecks the bytes actually decoded and
the roster/source identities after evaluation. Canonical/integrity errors abort
the preflight instead of becoming eligibility refusals; source-prefix errors keep
their root identities and repair ledgers. The resulting status explicitly is
not a strength or live-replay receipt. Nine roster/preflight tests pass with site
packages disabled; the existing CI gate now requires those nine rather than six.
The actual-source CLI independently reproduces 31 valid prefixes, one refusal,
two source-owned repairs, and zero replacement roots, with both live-replay and
profile qualification false. Independent review found no actionable defect:
its site-disabled nine-test suite and actual-source CLI both pass with exit 0,
reproducing the same denominator, repairs, refusal, and pending qualifications.
The reviewer corrected one summary invocation that had failed with exit 1 for
an omitted `PYTHONPATH`; no source or integrity checks were changed for it.

### Frozen live-public replay preflight

The completed joint-action and prefix changes were published at
`cd2cc06e6c0c0be698ae217b8d6b9d6bfc667920`. Remote engine-fidelity run
37201601877 completed successfully, including its mass gate and `gate-status`;
the secret-scanning, fleet-worker and neural-smoke checks also passed. These
results apply to that head, not subsequent live-preflight changes.

The new read-only `scripts/qualify_policy_opponent_live_replay.py` replays the
same byte-bound source records in the checkpoint-configured live Showdown
environment, without invoking a policy or native search. It requires an exact
clean runtime-content hash and checkpoint hash, checks them again afterward,
and repeats the source preflight. Each accepted root matches the entire public
observation, belief input, legal mask and every persisted actor-history
observation. Every legal action is serialized to a live Showdown choice. Missing
or different inputs refuse the original root; source integrity drift aborts the
whole check.

Actual-source invocation completed with exit 0: all 32 registered roots remain,
31 are live-public-valid, 130 earlier actor observations match, and the original
seed `2026100104` p2 decision-9 source-prefix refusal remains. Both source-owned
repairs are retained. Seed `2026100106` p1 decision 9, ID
`a7f3251c1922343b6d60a1286c3a43e46f3c58471fcde85ae068cd325372a34f`,
is a one-sided forced-replacement request. It is valid for the root profile but
not eligible for a simultaneous fixed-opponent continuation. Keep that explicit
disposition in the proposed 16-root continuation denominator; do not redraw it.

The actual runtime is clean Showdown commit
`f76228a1354b5d0f307ca2d16101294ad3a2308b`, content SHA-256
`93f81c8eae3d9f769f579a3be5e87bc0d62df39c0c0166bd47a228238188e4b3`.
Its belief-source hash `f5a5265143d423af` matches the champion checkpoint SHA-256
`0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be`.
The CLI reports image/native-world/profile qualification false and search not
invoked. Its aggregate live-replay qualification is also false because one
registered root is refused; the 31 individual root receipts are not discarded.

The first ad-hoc diagnostic failed with exit 1 on an incorrect configuration
helper import, before any replay; correcting the import reproduced 31/32. The
subsequent reusable CLI additionally verified complete actor history and runtime
binding. All 13 roster/source/live tests pass, including with site packages
disabled, and the existing CI count guard now requires 13. The 251-test focused
replay/registry/provenance suite and all 39 count guards pass with exit 0.
Independent review found no actionable findings; its 13-test suite and actual
CLI independently passed with exit 0 and reproduced the same identities,
repairs, one-sided request, refusal and pending qualifications.

This closes local public-input replay qualification, not native world formation,
opponent request-order certification, immutable image publication, the registered
profile/resource manifest, actual candidate comparisons or playing strength.
No cluster object/shared source artifact was modified and no candidate search,
continuation, training run or deployment was launched. The full goal remains
incomplete and active.

### Bounded registered root-profile launcher

All twelve remote PR checks on published head
`3dab091e1245370f7604581d5ccea35b7b2c8fb1` now pass, including engine-fidelity
run 37202592211. This does not certify subsequent launcher changes.

The new registration binds the original 32 roots, all three arms, fixed-work
and matched-deadline modes, champion/oracle/source/native/export/image hashes,
per-root RNG domains, and the existing d2/s256/b16/w4 practical configuration.
It retains the 16 proposed continuation identities, the one source refusal and
the separate one-sided eligibility disposition; no replacement roots or
strength/reproduction claims are authorized. The two-hour/96 CPU-hour budget
is shared with continuations. The first profile is serial on two CPUs, with
one Torch thread and one interop thread, no GPU, a 600-second root-worker cap,
and five seconds reserved for owned process cleanup.

`scripts/run_policy_opponent_root_profile.py` requires an actual clean attested
Linux/arm64 v8 source image and v4 copied model runtime. It checks both native
consumers, Torch, interpreter, exact two-CPU cgroup quota, champion, Showdown
content and source bytes. It copies the checkpoint to a fresh study input
directory so model exports do not change the historical checkpoint cache.
Registration, complete live-preflight evidence and study start are immutable
and externally byte-pinned before any root action is selected.

Each adjudicated arm writes its own create-only receipt after cleanup and
outside the decision timer. Late terminal/arm records survive a later timeout;
partial cap records are explicitly inventoried as diagnostic-only and excluded
from completed-root statistics. All 32 root dispositions remain visible.
Nonzero worker exits, missing/malformed terminals or final binding drift stop
readout publication. An exit-zero diagnostic readout is not a strength PASS
and never automatically authorizes advancement.

Independent review found and closed a setup-cap escape, inherited termination
signal masking, stale/reaped process ownership, an outer SIGTERM cleanup hole,
and a SIGINT ownership-publication race. The outer owned study supervisor now
bounds initialization, exports, preflight and root execution. Handled signals
are masked only while owned child identity is published, then unmasked by the
CLI before imports; external termination cascades into the separately owned
root group. Existing processes, Kubernetes objects and shared evidence are
never targeted. Real harmless nested-process tests prove timeout and outer
SIGTERM cleanup/reaping, not candidate search performance.

Verification: 37 site-disabled roster/source/live/registration/launcher tests
pass with exit 0, no skips. These include the real 32-root publication/control
loop with synthetic refusals/caps, immutable late evidence, exit-17 refusal and
final source drift. The full rebuilt-wheel/private-champion suite passes all
78 tests with exit 0, no skips; the earlier invocation without private inputs
skipped 11 and is not the private-champion evidence. Native source remains
current at fingerprint
`2b019d6667a5bd36a741f55370e557c3f82414ab580bc0a4a212f21bc5928547`.
An initial count check failed (exit 1) because the model-suite guard still
expected 66 rather than 67 after adding the durable-sink failure test; the
corrected workflow and all 39 static count guards pass. Independent review
reproduced 37 portable and 14 profile tests and all 39 count guards, found no
remaining actionable issues, and did not run an image or candidate search.
The Python-floor check passes all 656 tracked Python files; staged secret
scanning and diff checks pass with exit 0. These are local publication checks,
not CI evidence for this new head.

At that pre-launch foundation stage, the complete experiment was still pending
immutable source-image publication,
actual runtime/resource verification, native frozen-root profiling, bounded
continuations and the plan's conditional later stages. Synthetic launcher
fixtures, public replay qualification and local champion tests cannot stand in
for that execution. No new Kubernetes workload/shared artifact was created by
this launcher verification and historical artifacts remain unchanged.

## Executed profile and narrow correctness repair

Source `1c16419b19c513deda7c918c103619be91fd7c4f` passed all twelve PR checks,
published the immutable `a24a7d6a` image with complete v8 receipt and Showdown
sidecar, and executed the original 32-root profile on OLFUSA. Job
`paper-policy-profile-1c16419b-20261004-r1` completed at 14:21:30 UTC with
succeeded=1, exit 0 and zero restarts. The diagnostic took 1064.381 seconds
including initialization and reserved 0.591323 CPU-hours. It is not a strength
PASS. The readout retains 23 COMPLETE and nine REFUSED roots: five public-HP
projection refusals, two private/native legal-surface disagreements, one lost
public active permutation, and the original source-prefix refusal.

All 23 fully completed roots select the same canonical action as raw policy
and incumbent MCTS in both timing modes. This is conditional on completion,
not an all-root neutral effect or a strength conclusion. No continuations were
launched because the correctness prerequisite did not pass. The shared clock
started at 14:03:42.203 UTC; its 16:03:42.203 UTC deadline has expired. The old
readout's remaining-seconds field is a publication-time value, not permission
to reset the window or submit a new study.

The observed HP failure exposed a concrete protocol parser defect: canonical
Pain Split emits single-owner `-sethp` lines with optional `[from]`/`[silent]`
qualifiers. Both projections incorrectly treated a six-field qualified line
as two HP owners. The narrow repair distinguishes an actual second owner
from qualifiers, preserves public percentage projection and qualifier spelling,
and retains strict malformed/unknown-owner/private-surface refusals. The
existing tests now include single-owner, dual-owner, malformed-tail and
missing-HP cases. No fallback, roster, RNG, comparator or historical record
was changed. Local repair does not certify recovery of the five frozen roots,
and does not resolve the two legality or one permutation refusal.

Both native consumers rebuilt successfully after the repair; the current
fingerprint is `a547a1c37b2d4650dfe941441ed4d702b83131a2d5d255ee8d362998f4721c7c`.
Verification exits 0: 293 native model-feature tests pass with one existing
ignored test; all 78 rebuilt-wheel/private-champion tests and 37 portable
registration/launcher tests pass without skips; all 656 tracked Python files
parse on the Python 3.11 floor; Rust formatting and diff checks pass.
Independent review reports no actionable findings and independently reproduces
the old Pain Split refusal and repaired projection with all 16 view tests
passing. Remote CI for this repair remains separate from the passing frozen
source checks. No repaired source image or replacement experiment is certified
by these local tests.

### Source-only closeout after the bounded stop

Remote run `37239590432` on repair head `cbc11b2e` exited 1 in the
terminal-disposition register: three independently derived fingerprint checks
found the old `2b019d66` row instead of current `a547a1c3`. The repair updates
the maintained factual row and adds the source transition to its history;
no test, historical sweep or ratification status is changed. All 53 register
tests pass locally after that correction. Required CI on the next head remains
outstanding.

The terminal readout now records the available source-only diagnosis for all
remaining refusal classes and a receipt-level allocation audit. A hash-verified
public chronology and structural controls locate ordinal 21's order loss at
Whirlwind's newly revealed Mightyena: its original party index is not public.
The two legal-surface refusals cannot be attributed more narrowly from their
stored receipts; the failing legal sets and node depth were not retained. That
uncertainty is explicitly preserved, along with the evidence a future separately
authorized diagnosis would need. No simulator/search was invoked for this
closeout, and no extant cluster/shared artifact was changed.

The 218-file original root/arm inventory, exact-work productive backup
conservation, selected action identity, prior/visit summaries and strict
zero-fallback completed cells were read-only audited. These checks verify the
measurement boundary, not correctness coverage or playing strength. The
original category coverage and later conditional paper-reproduction stages
remain unqualified; the plan's permitted unresolved stop does not turn them
into achieved scientific results.
