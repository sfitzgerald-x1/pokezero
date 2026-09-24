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

The effect on the recorded root choice is narrower than the raw fallback total
suggests.  Across all eight fallback-bearing decisions in the forensic replay
(1,450 fallbacks: the 1,414 stale-PP cases plus the 36 residual cases), search
selected the model argmax every time: zero root overrides.  The fallback
therefore does not explain a directly observed changed root action in this
panel.  It remains an interior-prior correctness defect — and can change
interior values or become decisive elsewhere — but this result rules out
claiming those 1,450 events themselves caused the panel's root-action
overrides.  The seven-count difference from the earlier aggregate headline is
not attributed here; only sealed, per-decision ledgers are used for this
action-effect statement.

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

The corrected-source mutation battery has completed cleanly: all 66 scored
mutations were applied and killed, with zero survivors, skips, or
not-applied mutations.  Its seven controls reached their required verdicts,
the source tree was restored, and the recorded artifact is bound to this
source and harness.  This establishes that the relevant witness gate can
detect the scoped failures; it is not strength evidence.

The focused corrected replay is still required.  A single completed control
seed has no branch-prior fallbacks, and the corrected `2026092006` p1 game
has completed with zero branch-prior fallbacks.  Neither observation
establishes a repair by itself: the full four-seed, both-seat replay must
complete cleanly and reproduce the sealed source/configuration before the
historical concentration can be called removed.

One completed corrected replay arm does isolate a separate, smaller residual:
`2026092003` p2 has 36 `unmapped_action` fallbacks (27 at turn 14, one at
turn 20, and eight at turn 22), all classified as acting-side
`move_arms_unexplained`.  These are not the historical stale-PP class:
none is engine-missing, order-unavailable, or present-but-illegal.  In all
three affected public decision records the root search choice remains the raw
model argmax (zero overrides), so this evidence does not show a changed root
action in that replay.  It does show that the residual needs more
private-free mapping-surface instrumentation before proposing another fix;
it must not be folded into the fresh-switch repair claim.

Even a clean focused replay would prove a search implementation correction,
not a strength gain.  The next required step is a durable, paired GPU
guided-MCTS-versus-raw-policy study using the corrected source, exact raw
selector, mirrored seats, and per-seed durable receipts.  Its result decides
whether the corrected search now improves strength or whether remaining work
should prioritize leaf-value accuracy, opponent modeling, or exploration and
backup behavior.
