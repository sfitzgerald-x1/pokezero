# C27 result: retained damage-arithmetic tail

## Historical Method (Unverified)

The prior narrative states that it replayed the seven C26 target identities on
a two-consumer, fingerprint-checked native build. For every standard direct move,
`attest_damage_arithmetic_tail.py` compared:

1. the observed Showdown HP delta and secondary status;
2. the pure Python Gen 3 damage oracle transcribed from Showdown; and
3. the Rust `calculate_damage` maximum and all rendered instruction branches.

The repaired executable admits a comparison only when every damage-bearing
native branch is represented, its exact pre-hit state and criticality are
known, and all branches in the observed criticality partition independently
produce identical oracle/native evidence. Opposite-criticality branches remain
visible but are not mixed into that comparison. The two confusion-only targets
remain explicit comparison limits, not guessed direct-move results.

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

The executable has a positive exact-support allowlist for only Sludge Bomb and
Fire Blast. For these five historical contexts it classifies Poison Point,
Liquid Ooze, Shield Dust, Static, Effect Spore, Rough Skin on a non-contact
Sludge Bomb, Keen Eye, Levitate, and defender Pure Power as irrelevant to the
direct formula. It likewise classifies healthy Salac Berry, Leftovers, defender
Choice Band, and sand on Poison damage as irrelevant. Attacker Choice Band is
modeled only for physical damage.

Gen 3 damage category is type-based: Poison-type Sludge Bomb is **Physical**, so
the Nidoking attack uses Attack/Defense and Choice Band applies. Fire-type Fire
Blast is **Special**, so Choice Band would not modify it. This is also asserted
against the loaded Gen 3 dex in the executable tests.

## Result Status

**Not independently attested at this commit.** The original seven replay input
rows are not committed in this checkout, and the prior JSON lacked their
hashes, command, source commit, native-build fingerprint, and Showdown
source-content identity. The table below
is retained only as an unverified historical transcription; it is not evidence
for a production change or for a conclusion about the current engine. No
clearance is claimed.

The configured Showdown checkout was dirty during this repair, and the v4
provenance gate refused measurement as designed. It was not modified here. No
fresh attestation JSON is committed.

The executable attestation now refuses stale native consumers and dirty or
unprovenanced source, hashes every supplied input report, records its command,
source commit, producer hash, engine fingerprint, and every built Showdown
JavaScript and JSON input plus the Gen 3 randbat set source. It emits a
machine-readable v4 result with the full branch population, explicit
reported/dropped/unsupported counts, branch state source, and criticality
partition. A reviewer may rely on the five-direct-row scope only after that v4
artifact is regenerated from the missing retained reports and committed
alongside this document. The two confusion rows remain comparison limits in all
cases.

The executable emits `showdown_outside_transcribed_oracle` when the native
maximum agrees with the transcribed formula but the observed non-KO Showdown HP
delta is absent from that formula's legal roll support. That verdict is neither
a native-arithmetic disagreement nor a clearance: it identifies an incomplete
transcription or mismatched captured context and requires separate
investigation.

## Historical Transcription (Unverified)

| Target | Showdown damage | Oracle legal rolls | Rust maximum | Secondary |
| --- | ---: | --- | ---: | --- |
| `2800700/20` | 122 | 117-138 | 138 | poison |
| `3300207/69` | 137 | 132-156 | 156 | poison |
| `3301036/26` | 121 | 117-138 | 138 | poison |
| `3401017/55` | 79 | 77-91 | 91 | burn |
| `3500021/19` | 159 | 153-181 | 181 | poison |
| `3001000/57` | n/a | n/a | n/a | confusion self-hit |
| `3300122/21` | n/a | n/a | n/a | confusion self-hit |

The prior `3300207/69` representative entry is internally inconsistent even as
a historical transcription: it paired a maximum of `156` with `143`, although
the previously claimed unconditional `0.925` rule yields `144`. The
representative values are therefore omitted from evidence. The executable
neither applies that rule nor emits a representative-damage field; only actual
rendered branch damage is recorded.

If regenerated v4 evidence shows all five direct rows have a Showdown legal
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
Those questions require the missing hashed reports, a clean Showdown checkout,
and a fresh v4 replay.

This lane deliberately does not alter the terminal/KO or strict-matcher lanes.
The reusable script records this distinction for later retained rows and fails
closed to a comparison limit for modifier contexts it cannot transcribe exactly.
