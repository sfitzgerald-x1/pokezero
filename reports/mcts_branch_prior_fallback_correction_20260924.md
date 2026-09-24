# Branch-prior fallback correction: causal record

## Question

Why did own-prior guided MCTS fail to improve reliably on the same checkpoint's
deterministic raw masked-policy selector, despite much greater search work?

## Pre-fix evidence

The sealed override audit at
`/shared/scott-experiment/mcts-sealed-override-audit-db4bc213-20260920-r7`
contains a complete, matched audit of every override in its four-seed panel:

* 65 overrides were observed.  Three are correctly classified as
  non-simultaneous and therefore cannot support a matched continuation claim.
* Of the 62 matched overrides, the MCTS choice wins 30 independent
  continuations and the raw-policy choice wins 32; 37 continuation pairs end
  in the same terminal result.  None is terminal immediately after the fixed
  joint step.
* The audit's branch-prior ledger attributes all 1,414 fallbacks in seed
  `2026092006` to its historical `unmapped_action` bucket.  This is a
  material distortion of an interior search branch, not a cosmetic telemetry
  counter.

The later source-bound forensic replay separates that historical bucket into
causes.  It identifies the dominant case as an engine-offered move at a fresh
switch which the reconstructed PP ledger incorrectly vetoes as illegal.  The
same replay separately preserves the sparse action-slot issue: fixed engine
slots such as `M2` must not be compacted to `M0` before policy priors are
attached.

## Correction

The corrected source revision is
`8353511bbfc9dcf2c79c7f15f39080df9c926871`.

At a fresh switch, an action returned by the engine is now admitted as legal
even when the reconstructed PP count is stale.  The existing PP-positive path
is retained for hidden choice-lock cases; normal, non-fresh legality remains
unchanged.  This makes the engine's present action surface authoritative while
preserving the reconstruction only as a widening mechanism rather than a veto.

The fix has focused regression coverage for the disagreement case, plus the
existing sparse-slot regression.  An independent review found the change
correctly scoped: it applies only to fresh-switch reconciliation and does not
relax non-fresh legality.

## What remains deliberately unclaimed

The focused corrected replay and its source mutation battery are running at
the time of this record.  A single completed control seed has no branch-prior
fallbacks, and the corrected `2026092006` p1 game has completed with zero
branch-prior fallbacks.  Neither observation establishes a repair by itself:
the full four-seed, both-seat replay must complete cleanly and reproduce the
sealed source/configuration before the historical concentration can be called
removed.

Even a clean focused replay would prove a search implementation correction,
not a strength gain.  The next required step is a durable, paired GPU
guided-MCTS-versus-raw-policy study using the corrected source, exact raw
selector, mirrored seats, and per-seed durable receipts.  Its result decides
whether the corrected search now improves strength or whether remaining work
should prioritize leaf-value accuracy, opponent modeling, or exploration and
backup behavior.
