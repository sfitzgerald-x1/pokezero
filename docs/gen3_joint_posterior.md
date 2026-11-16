# Gen 3 joint team posterior

This module is standalone until its serving and feature integration gates pass.
It models the ordered team from the pinned Gen 3 `randomTeam`, including the
revealed sets and hidden backline in one distribution. Its probability model is
independent uniform generator API choices, as in [the set compiler](gen3_generator_weights.md).
It does not enumerate the correlated finite PRNG seed space.

## Catalog and recurrence

`scripts/export_gen3_team_model.cjs` exports the pinned generator's 217 base
species, 218 selection tokens and 220 forms, with type, weakness, level and
pairwise legality constraints. Deoxys has two tokens and four forms. There are
16 reachable set contexts: rain, sun, status cure and Rapid Spin. Thirty species
have context-dependent set distributions.

The generator removes rejected candidates. All legality rejections are
monotone as the team grows, so a currently eligible base could not have lost
tokens earlier. Its next-accepted probability is therefore proportional to
its eligible token count. Evidence-excluded candidates stay in that prior
denominator. Forms and set variants then use the exact enumerated weights.

`joint_generator.JointGenerator` is the Python integer oracle. Exchangeable
bases share a DP group only if legality, token multiplicity and every context
transition agree. Required species and Wobbuffet remain distinct. The state
records group occupancy, set context and the outgoing battle flag. Original
generation slots may be fixed; reveal order and current party order are not
generation order. Nonnegative integer variant likelihoods encode public facts
and admitted damage support. Required species must occur somewhere on the team.

Every transition has an integer numerator over one common quantum. The root
mass over `quantum ** slots` is the evidence likelihood, including any terminal
factor. Once all required species have been selected, a legality-only suffix
may be reused. A stronger completion shortcut is accepted only for the pinned
catalog: `scripts/prove_gen3_team_completion.py` proves, with exact finite-domain
constraints, that every legal subset of at most five bases has an eligible
extension in either incoming Wobbuffet mode. Both Python and Rust bind that
shortcut to the actual catalog contents.

The Rust/PyO3 implementation lives in the existing `pokezero-search` crate. It
uses arbitrary-precision integers and a caller-owned `JointLegalityCache`.
The cache can be reused across decisions as reveals arrive; evidence and set
likelihoods do not enter its legality keys. Per-root mass states remain bound
to that root's evidence. Generic catalogs are supported for mathematical tests;
the serving wrapper rejects catalogs other than the measured pin.

## Both battle arms

`battle_joint_parameters` requires the owner, complete original own team and
team producer. Shared-battle generation creates p1 then p2 with one generator.
For a p1 owner, the incoming flag for the opponent follows the own team's first
five slots. Slot six does not update the flag in the pinned generator. For a
p2 owner, the opponent's outgoing flag is weighted by the exact likelihood of
the observed original own team under each possible incoming flag. Duplicate
source variants describing one physical set count as one observation.

Independently generated teams have different flag semantics and must declare
`producer='independent-teams'`. The serving boundary requires this producer/arm
binding and does not permit callers to substitute unbound flag weights.

## Bounded policy and feature payload

`NativeJointPosterior.solve` computes integer lower and upper masses. At a
budget frontier, the lower contribution is zero and the upper bounds all
remaining likelihood and terminal factors. The frozen lower mass defines the
sampled subdistribution; diagnostics and draws cannot expand its frontier.
The normalized lower distribution has certified total variation distance at
most `(upper - lower) / upper` from the exact posterior. Both Python and native
draw boundaries reject a certificate above 1/100.

`draw_with_stage1_fallback` requires an explicit battle context and a caller's
`stage1_sampler(count, rng)` callback. Its default 0.25-second admission budget
includes catalog checks, reverse preparation, construction, normalization,
draws and native-state release. The normalizer reserves 0.05 seconds for draws.
A failed or late certificate restores the caller's RNG before handing off to
stage 1, and reports `stage1_fallback`. That fallback has no joint TV guarantee.
The admission deadline rejects late results; it is not a hard interruption of
Python construction or the stage-1 callback. Invalid provenance fails closed.

`feature_payload()` exports source-order integer revealed-set marginals,
variant IDs, support, source/model/table bindings and the TV certificate.
Exact support is positive exact marginal mass. Under truncation, approximate
zeros never become hard pins: support retains every evidence-compatible variant
reachable in a generator context. This conservative support may over-include.
Feature-mask, fold identity and world-consumer wiring are separate integration
work; importing this module does not alter them.

## Validation

The native tests compare full ordered-team probabilities against 100 independent
tiny enumerations, including all context flags, legality caps, token and form
multiplicity, both battle flags and reverse terminal factors. Other fixtures
exercise accepted truncation, actual TV, RNG fallback, native input validation,
provenance and physical-set aliases. Native tests require the compiled extension
and do not skip when it is missing.

`gen3_joint_reference.cjs` is a validation-only acceptance sampler around the
unmodified generator; production inference never uses raw-team rejection.
The diagnostic accepts Boolean evidence and unit terminal factors. Its
canonical conditioning digest covers evidence, fixed generation slots and the
incoming battle flag. `gen3_joint_chi_square.py` requires matching digests,
catalog/source identity and distinct reference seeds, then compares at least
one million reference teams with at least ten thousand posterior draws. It
pools sparse histogram cells by margins and controls the eight-test family at
alpha 0.001. Non-unit likelihoods and reverse terminal factors are rejected by
this diagnostic instead of being interpreted as event probabilities.
