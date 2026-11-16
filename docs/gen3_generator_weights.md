# Gen 3 generator priors

The offline compiler executes `RandomGen3Teams.randomSet` from Showdown
`f76228a1354b5d0f307ca2d16101294ad3a2308b` with a branching PRNG. It traverses
`sample`, `sampleNoReplace`, `sampleIfArray`, `randomChance`, and Fisher–Yates
shuffle choices, multiplying and summing rational probabilities with BigInt.
Move order, cosmetic formes, shiny state, and duplicate generator routes are
summed into physical set identities; typed Hidden Power remains distinct.

The probability contract is **independent uniform API choices** (and `n/d`
for `randomChance`). It does not enumerate correlated PRNG seeds. In particular,
Showdown's finite 32-bit implementation of `random(3)` has a one-unit bucket
imbalance; the API-level model assigns exactly one third to each choice.
The statistical gate is not a proof about that tiny finite-state distinction.

The table covers all 220 species and all 16 combinations of rain, sun,
status-cure, and Rapid Spin at the time the generator creates a set. The pinned
`randomTeam()` assigns `teamDetails.spikes = 1`, never increments it, so the
`spikes >= 2` cull in `randomSet` is unreachable. Such custom contexts are
rejected. Context is generator state, not current battle weather.

The existing source list stays unchanged, including its zero-weight variants.
Equivalent physical sets occupy their first source-order entry; probability
from every generating role is summed there, and duplicate entries get zero.
Every context uses a common integer denominator per species. A generated set
missing from the source list, a species with more than 64 variants, or an
incomplete publication fails the compiler gate.

Provenance binds the source commit, the executed JS/JSON dependency digest,
compiler digest, source hash, full variant manifest, enumeration hash, and table
content hash. The executed digest was reproduced with an independent clean
build of the pinned source and lockfile. Dirty source or a different executable
build is refused. The Python loader validates the source order and all 16
normalized integer rows and returns immutable tuples.

## Rebuilding and checking

Use a built, clean pinned Showdown checkout in `$SHOWDOWN_ROOT`, Node 22,
and the project's Python environment. Export the source order, compile raw
outcomes, and then compile the published table:

```sh
PYTHONPATH=src:. python scripts/export_gen3_variant_manifest.py \
  --root "$SHOWDOWN_ROOT" --output variants.json
node scripts/gen3_generator_weights.cjs --root "$SHOWDOWN_ROOT" \
  --output enumeration.json
node scripts/gen3_generator_weights.cjs --root "$SHOWDOWN_ROOT" \
  --variants variants.json --output src/pokezero/data/gen3_generator_weights.json
```

`--species` and `--contexts` produce explicitly incomplete diagnostic
enumerations; they cannot be published as full tables.

The sample collector calls the unmodified `randomTeam()` and records the
actual context supplied to each `randomSet()`. Use distinct seeds per shard:

```sh
node scripts/gen3_generator_sample.cjs "$SHOWDOWN_ROOT" 1000000 \
  100,2026,11,15 sample.json
python scripts/gen3_generator_chi_square.py --enumeration enumeration.json \
  --variants variants.json --samples sample.json --output chi-square.json
python -m unittest tests.test_depthn_generator_weights -v
```

The statistical runner first reuses the compiler's strict publication and
source-coverage checks. It then checks sample identities, distinct seeds,
declared counts, at least one million sets, and sample support. Expected counts
are conditional on the observed species/context counts. Rare cells are pooled
using expected counts alone; degrees of freedom are pooled cells minus one per
species/context. The global chi-square gate uses alpha .001; the separate
context gates use a Bonferroni familywise alpha .001 (combined false-rejection
bound .002).

The exact Togetic fixture is four Staller sets at 1/8 each and the Bulky Support
set at 1/2. The initial million-set check covered 1,000,032 sets, with
chi-square 5762.3886 on 5767 degrees of freedom (p = .51465), no failing context
gates, and no uncovered sets.

## Consumer boundary

`load_generator_weights(source)` loads the versioned table.
`table.weights(species, team_context=...)` requires an explicit generator
context and returns integer weights in the source's variant order. The loader
does not select a policy for hidden team-generation context. This standalone
compiler/table change does not yet wire the prior into `set_posterior`, the
world sampler, or the V2 feature hook.
