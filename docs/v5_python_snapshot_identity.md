# Owning Python collector: serialized Pokémon identity

The bounded correction starts at source
`49eacf5b813c502ff9bb3b7e89e341e0f975458d`. It changes only the collector's
controller-side projection of original simulator Pokémon identity. No bridge,
actor input, team, world, RNG, model, GAE/cache/trainer or terminal-winner rule is
changed. It does not replay failed/lost cells or constitute runtime admission.

The original route is `collect_game` → `env.snapshot` → bridge `snapshotBattle`
→ `State.serializeBattle`. At pinned Showdown
`f76228a1354b5d0f307ca2d16101294ad3a2308b`, `sim/state.ts:46` explicitly omits
Pokémon top-level `name`, `fullname`, `id`, `level`, `happiness` and other
redundant fields. `serializePokemon` at line212 reattaches `state.set`; it does
not reattach a top-level name. HP and maximum HP are ordinary serialized fields;
`hpType` is an optional serialized value. Required set fields species, moves,
level, ability and item remain in that original set object. The constructor
populates these set fields from the exact supplied packed teams.

`sim/pokemon.ts:329` has already canonicalized empty/default `set.name` to the
base species identity. Its actual display name at line334 is
`set.name.substr(0,20)`. The collector now reads that retained original set name
and follows the exact 20 UTF-16-code-unit rule. It requires a nonempty string
and six unique resulting names on each side. Missing/invalid or colliding
identities refuse; species and a synthetic extra top-level name are not
fallbacks. For example, a form's original species can be `Deoxys-Attack` while
its actual default nickname is `Deoxys`.

Names only join original birth and terminal physical members, including party
reordering. The complete original set is still copied and checked unchanged at
terminal; exact HP validation and all terminal/unknown rules remain in place.
Birth snapshots stay controller-only. Actors receive only their existing
isolated owning streams, and this projection still occurs before per-game
actor creation or decisions. Thus a bad birth identity cannot produce a policy
row, hidden-set label, value/outcome target or gradient.

Focused controls use serializer-shaped records with `set.name` and no top-level
name, including form/nickname difference, name truncation/collision, missing
required fields, terminal reordering and unchanged actor-input/choice
invariance. No native simulator, constructor, checkpoint/model, forward or real
game is executed by these controls. The prior scientific failure and all its
attempted/unknown/not-attempted dispositions remain immutable evidence under
the old source condition. Any continuation requires a separately bound corrected
source and the one changed-producer review.
