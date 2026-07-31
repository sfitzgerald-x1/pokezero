# C27 result: retained damage-arithmetic tail

## Historical Method (Unverified)

The prior narrative states that it replayed the seven C26 target identities on
a two-consumer, fingerprint-checked native build. For every standard direct move,
`attest_damage_arithmetic_tail.py` compared:

1. the observed Showdown HP delta and secondary status;
2. the pure Python Gen 3 damage oracle transcribed from Showdown; and
3. the Rust `calculate_damage` maximum and all rendered instruction branches.

The current executable oracle only admits a verdict where its complete visible
modifier context is implemented. The two confusion-only targets remain explicit
comparison limits, not guessed direct-move results.

## Historical Public Context (Unverified)

The prior narrative attributes zero active stat boosts and no active
damage-modifying volatile to all five standard-move rows. The stat pair is
Attack/Defense for physical moves and Special Attack/Special Defense for
special moves.

| Target | Move | Attacker context | Defender context | Weather | Stat pair |
| --- | --- | --- | --- | --- | --- |
| `2800700/20` | Sludge Bomb | Qwilfish, Poison Point, Salac Berry, healthy, Water/Poison | Raichu, Static, Leftovers, healthy, Electric | none | 203 / 138 |
| `3300207/69` | Sludge Bomb | Tentacruel, Liquid Ooze, Leftovers, healthy, Water/Poison | Breloom, Effect Spore, Leftovers, healthy, Grass/Fighting | none | 156 / 182 |
| `3301036/26` | Sludge Bomb | Venomoth, Shield Dust, Leftovers, healthy, Bug/Poison | Sharpedo, Rough Skin, Choice Band, healthy, Water/Dark | none | 162 / 116 |
| `3401017/55` | Fire Blast | Weezing, Levitate, Leftovers, healthy, Poison | Medicham, Pure Power, Leftovers, healthy, Fighting/Psychic | none | 184 / 168 |
| `3500021/19` | Sludge Bomb | Nidoking, Poison Point, Choice Band, healthy, Poison/Ground | Fearow, Keen Eye, Choice Band, healthy, Normal/Flying | sand | 198 / 152 |

The historical transcription says Choice Band is included in the final
Nidoking physical oracle. The other listed abilities/items require fresh,
hashed evidence before they can be relied upon as non-modifying context.

## Result Status

**Not independently attested at this commit.** The original seven replay input
rows are not committed in this checkout, and the prior JSON lacked their
hashes, command, source commit, and native-build fingerprint. The table below
is retained only as an unverified historical transcription; it is not evidence
for a production change or for a conclusion about the current engine.

The executable attestation now refuses stale native consumers and dirty or
unprovenanced source, hashes every supplied input report, records its command,
source commit, producer hash, and engine fingerprint, and emits a
machine-readable v2 result. A reviewer may rely on the five-direct-row scope
only after that v2 artifact is regenerated from the missing retained reports
and committed alongside this document. The two confusion rows remain
comparison limits in all cases.

## Historical Transcription (Unverified)

| Target | Showdown damage | Oracle legal rolls | Rust maximum | Rendered nonterminal damage | Secondary | Verdict |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `2800700/20` | 122 | 117-138 | 138 | 127 | poison | fixed single-roll composition |
| `3300207/69` | 137 | 132-156 | 156 | 143 | poison | fixed single-roll composition |
| `3301036/26` | 121 | 117-138 | 138 | 127 | poison | fixed single-roll composition |
| `3401017/55` | 79 | 77-91 | 91 | 84 | burn | fixed single-roll composition |
| `3500021/19` | 159 | 153-181 | 181 | 167 | poison | fixed single-roll composition |
| `3001000/57` | n/a | n/a | n/a | n/a | confusion self-hit | comparison limit |
| `3300122/21` | n/a | n/a | n/a | n/a | confusion self-hit | comparison limit |

If regenerated v2 evidence shows all five direct rows have a Showdown legal
roll and matching current native maximum, that evidence can support the narrow
statement that these five rows do not establish a shared native
max-arithmetic defect. It cannot derive worlds, establish branch composition
as fixed, or generalize beyond those rows. The two confusion rows are not
direct arithmetic comparisons.

## Conclusion

No production arithmetic or event-component-mapping claim is supported by the
currently committed material. In particular, this document does not establish
that the native chance tree's branch composition is fixed, that a world was
derived correctly, or that any historical composition mechanism remains true.
Those questions require the missing hashed reports and a fresh v2 replay.

This lane deliberately does not alter the terminal/KO or strict-matcher lanes.
The reusable script records this distinction for later retained rows and fails
closed to a comparison limit for modifier contexts it cannot transcribe exactly.
