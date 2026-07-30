# C16 bench Rest provenance prediction

## Scope and prior evidence

This prediction concerns the archived transition-differential reproduction
`seed=2000281`, decision boundary 99. It deliberately does **not** repeat the
three already-settled links: parser state surviving a bench, row annotation of
a synthetic benched Rest sleeper, and `restSleepAttempts -> 3-k` arithmetic.

On `d245811`, replaying that seed shows that Entei used `Sleep Talk` after its
own Rest, then switched out while still asleep. The `p1` public request retains
`Entei ... slp` on the bench. From `p2`'s view that is an opponent
`public_revealed` row, so the public-row construction is present. The parser,
however, removes `p1:entei` from `rest_sleep_counts` as soon as it sees the
sleep-usable move. Consequently `_apply_rest_sleep_provenance` has no public
counter to write and `_hp_and_status` can only use generic sleep when an
approximation is requested (the archived engine state records `rest_turns=0`).

## Prediction

The smallest correct repair is to retain a **public Rest clock** rather than
retiring it on `Sleep Talk`/`Snore`:

1. `|cant|...|slp` decrements Rest's public remaining-turn count.
2. A following direct Sleep Talk or Snore marks that decrement as refundable.
3. An ordinary sleeping turn clears the pending refundable run.
4. A later `|switch|` of the same sleeping mon applies the Gen 3
   `skippedTime` refund and clears the refundable run.
5. The materialization row carries the resulting exact remaining count. Any
   missing, malformed, or out-of-range public state remains unannotated so the
   world builder fails closed rather than guessing.

The parser key must continue to use the side plus normalized ident name. The
reproduction has an ordinary species name, so a species-key patch alone should
not change it.

## Predicted observable outcomes

| Check | Unpatched prediction | Patched prediction |
| --- | --- | --- |
| Historical Entei after Sleep Talk then bench | no Rest annotation; approximate world has `rest_turns=0` | explicit exact Rest counter after refund (`rest_turns=3`) |
| Benched Rest sleeper during unrelated turns | counter changes with wall-clock turns | counter is unchanged |
| Switch back after Sleep Talk/Snore | no recoverable exact counter | pending skipped turns are refunded exactly once |
| Plain asleep turn after Sleep Talk | refundable count incorrectly persists | pending refund is reset, matching Gen 3 |

## Scoring rule

Score this prediction as **confirmed** only if a genuine protocol-derived
opponent `public_revealed` row from the historical sequence lacks the exact
counter before the patch, then receives the expected counter after the patch;
the four controls above must also pass. A synthetic row alone is insufficient.
If a required public ordering is absent, the score is **inconclusive** and the
constructor must stay fail-closed.
