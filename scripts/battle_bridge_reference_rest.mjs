// Exact Gen 3 Rest reconstruction from the public protocol ledger, not a
// fresh/random induced-sleep timer. Keep skippedTime separate: Showdown refunds
// it only on the NEXT switch-in, including for an already-benched sleeper.
export function referenceRestState(row, ability) {
  if (row.restSleepProvenanceUnrepresentable || row.restSleepAttemptUnsettled) {
    throw new Error("Reference Rest requires settled, representable public provenance.");
  }
  const attempts = row.restSleepAttempts;
  const refunded = row.restSleepRefundedTime ?? 0;
  const skipped = row.restSleepSkippedTime ?? 0;
  if (![attempts, refunded, skipped].every(n => Number.isSafeInteger(n) && n >= 0) ||
      refunded + skipped > attempts) {
    throw new Error("Reference Rest requires valid public attempt/refund counts.");
  }
  // Historical prefixes used the ambiguous legacy marker for multiple causes.
  // Accept it only when the new, explicit active-skipped provenance explains it.
  if (row.restSleepRefundPending &&
      !(row.restSleepActiveRefundPending === true && row.active === true && skipped > 0)) {
    throw new Error("Reference Rest refuses ambiguous legacy refund provenance.");
  }
  if (row.restSleepActiveRefundPending && !(row.active === true && skipped > 0)) {
    throw new Error("Reference Rest active refund marker contradicts its counts.");
  }
  const earlyBird = String(ability).toLowerCase().replace(/[^a-z0-9]/g, "") === "earlybird";
  const time = 3 - attempts * (earlyBird ? 2 : 1) + refunded;
  if (time < 1 || time > 3) {
    throw new Error("Reference Rest public timer contradicts the sleeping condition.");
  }
  return {id: "slp", effectOrder: 0, time, startTime: 3, skippedTime: skipped};
}

export function bindReferenceRestSources(serializedSide, sideId) {
  // Bind AFTER active-first / actor-known team reordering. A Rest source is the
  // sleeper itself (Sleep Clause exempt), never the opponent or a stale slot.
  for (const [index, pokemon] of serializedSide.pokemon.entries()) {
    if (pokemon.status === "slp") {
      const self = `[Pokemon:${sideId}${"abcdef"[index]}]`;
      pokemon.statusState.source = self;
      pokemon.statusState.target = self;
    }
  }
}
