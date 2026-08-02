# Terminal Residual Roll Branching Limit

The 76-line `poke-engine-gen3-terminal-toxic-roll-split.patch` experiment has
been withdrawn. It treated a pre-move Toxic arithmetic threshold as the
decision boundary, but that cannot model Gen 3's actual end-of-turn queue.

No production claim is made for residual-lethal non-direct damage rolls. The
engine continues to use its compact damage representative outside the existing
direct-KO splitter. This is deliberately a documented comparison limit, not a
partial fidelity fix.

The native and Python consumer ablation pins assert that this limit remains
explicit: branch generation restores the input state, preserves 100% probability
mass, and emits the compact representative rather than the withdrawn splitter.

## Why the local splitter was unsound

The decision is not `damage + Toxic >= hp`. It is the observable outcome of a
complete move pair followed by the speed-ordered residual handler queue. Before
a Toxic tick, that queue may heal or cure the target through Wish, Sitrus,
Rain Dish, Dry Skin, Lum/Chesto/Shed Skin, or another move's effects. Weather,
Leech Seed, Future Sight, Perish Song, and both active Pokemon's own residual
handlers can instead cause the first final faint. A non-final faint schedules a
replacement and defers the residual block; a final faint truncates it. Exact
speed ties are another branch.

The withdrawn code also enumerated only normal 85..100 rolls, represented a
critical hit at 92.5%, and assumed the defender was the only possible final
target. It therefore changed probability and tail instructions incorrectly for
mixed survival, residual-KO, and direct-KO ranges, including guaranteed and
increased critical hits and Battle/Shell Armor. Protect, misses, Substitute,
fixed damage, multi-hit moves, recoil, drain, and Explosion/Self-Destruct all
invalidate one or more assumptions made by that local check.

## Required design before retrying

Implement this at the move-pair boundary, after move resolution but before the
compact damage approximation is committed:

1. For each eligible one-hit, non-fixed normal and critical roll, run the exact
   move pair and the real reversible end-of-turn brancher in a probe state.
2. Compare normalized decision outcomes, retaining state that affects the next
   request plus the residual instruction tail. Do not compare raw damage alone.
3. Emit the exact 16-roll normal and 16-roll critical lattices only when at
   least two probe outcomes differ. Keep direct KO, replacement, battle-ended,
   residual truncation, item/cure consumption, and both tie orders distinct.
4. Reuse the same brancher for the Python wheel and Rust search crate. The
   probe must restore the state and probability mass exactly, and must cap the
   expansion at the two 16-roll lattices per eligible action.

The implementation is materially larger than the withdrawn patch because the
current splitter lives inside `generate_instructions_from_move`, before the
second action and before `add_end_of_turn_branches`. Moving the criterion to the
pair boundary is necessary to make the result sound; no local replacement was
merged.
