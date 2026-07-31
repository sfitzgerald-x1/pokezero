# C26 result: materialized damage-stat ownership

## Method

Replayed each target with the same deterministic legal-action stream and
production world constructor used by `engine_transition_differential.py`.
Before branch generation, compared every Python `PokemonSpec` against the
constructed Rust `State`:

- stored Attack, Defense, Special Attack, Special Defense, and Speed for all
  party members on both sides;
- active-side boost stages; and
- damage-relevant ability, item, status, and type inputs.

The replay used a current, two-consumer fingerprint-checked native build.

| Target | Public turn | Gating | Candidate worlds | Mismatches |
| --- | ---: | --- | ---: | ---: |
| `2800700/20` | 18 | exact | 1 | 0 |
| `3301036/26` | 26 | exact | 1 | 0 |
| `3401017/55` | 49 | exact | 1 | 0 |
| `3500021/19` | 17 | exact | 1 | 0 |
| `3300207/69` | 62 | support | 7 | 0 |
| `3001000/57` | 53 | support | 5 | 0 |
| `3300122/21` | 20 | support | 5 | 0 |

## Finding

The pre-registered construction-seam prediction held: all 21 materialized
candidate worlds had exact base-stat and active-stage agreement with the Rust
states passed to branch generation. This rules out a shared Python
materialization ownership bug for these retained rows.

No production patch is justified by this evidence. The remaining owners are
downstream: Gen 3 engine damage arithmetic and/or the component mapper's
attribution of the realized protocol event. The reusable diagnostic remains
available for future retained identities and fails visibly on either stored-stat
or active-stage corruption.
