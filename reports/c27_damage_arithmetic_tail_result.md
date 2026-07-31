# C27 result: retained damage-arithmetic tail

## Method

Replayed the seven C26 target identities on the current two-consumer,
fingerprint-checked native build after PR #986. For every standard direct move,
`attest_damage_arithmetic_tail.py` compared:

1. the observed Showdown HP delta and secondary status;
2. the pure Python Gen 3 damage oracle transcribed from Showdown; and
3. the Rust `calculate_damage` maximum and all rendered instruction branches.

The oracle only admits a verdict where its complete visible modifier context is
implemented. The two confusion-only targets are therefore explicit comparison
limits, not guessed direct-move results.

## Exact public context

All five standard-move rows had zero active stat boosts and no active
damage-modifying volatile. The stat pair is Attack/Defense for physical moves
and Special Attack/Special Defense for special moves.

| Target | Move | Attacker context | Defender context | Weather | Stat pair |
| --- | --- | --- | --- | --- | --- |
| `2800700/20` | Sludge Bomb | Qwilfish, Poison Point, Salac Berry, healthy, Water/Poison | Raichu, Static, Leftovers, healthy, Electric | none | 203 / 138 |
| `3300207/69` | Sludge Bomb | Tentacruel, Liquid Ooze, Leftovers, healthy, Water/Poison | Breloom, Effect Spore, Leftovers, healthy, Grass/Fighting | none | 156 / 182 |
| `3301036/26` | Sludge Bomb | Venomoth, Shield Dust, Leftovers, healthy, Bug/Poison | Sharpedo, Rough Skin, Choice Band, healthy, Water/Dark | none | 162 / 116 |
| `3401017/55` | Fire Blast | Weezing, Levitate, Leftovers, healthy, Poison | Medicham, Pure Power, Leftovers, healthy, Fighting/Psychic | none | 184 / 168 |
| `3500021/19` | Sludge Bomb | Nidoking, Poison Point, Choice Band, healthy, Poison/Ground | Fearow, Keen Eye, Choice Band, healthy, Normal/Flying | sand | 198 / 152 |

Choice Band is included in the final Nidoking physical oracle. The other
listed abilities/items are preserved in the evidence context but do not modify
the particular direct hit.

## Result

| Target | Showdown damage | Oracle legal rolls | Rust maximum | Rendered nonterminal damage | Secondary | Verdict |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `2800700/20` | 122 | 117-138 | 138 | 127 | poison | fixed single-roll composition |
| `3300207/69` | 137 | 132-156 | 156 | 143 | poison | fixed single-roll composition |
| `3301036/26` | 121 | 117-138 | 138 | 127 | poison | fixed single-roll composition |
| `3401017/55` | 79 | 77-91 | 91 | 84 | burn | fixed single-roll composition |
| `3500021/19` | 159 | 153-181 | 181 | 167 | poison | fixed single-roll composition |
| `3001000/57` | n/a | n/a | n/a | n/a | confusion self-hit | comparison limit |
| `3300122/21` | n/a | n/a | n/a | n/a | confusion self-hit | comparison limit |

For all five direct rows, the Showdown damage is a legal oracle roll and the
Rust maximum equals the oracle maximum. This rules out a shared Gen 3
damage-arithmetic defect for the retained tail. The instruction generator
instead emits one nonterminal representative damage value. A branch containing
the observed poison/burn status exists, but none pairs it with the observed
lower damage roll; it pairs the status with the representative hit, which can
change whether the later residual is lethal.

## Conclusion

No production arithmetic or event-component-mapping patch is justified by this
evidence. The shared owner is a known simulation-composition limit: the native
chance tree does not cross product the 16 direct-damage rolls with secondary
effect outcomes. The component mapper accurately reports that native branch.

This lane deliberately does not alter the terminal/KO or strict-matcher lanes.
The reusable script records this distinction for later retained rows and fails
closed to a comparison limit for modifier contexts it cannot transcribe exactly.
