# Saved mirror roots for cost quarantine

`freeze_recorded_mirror_cost_roots.py` selects 20 distinct preassigned training
battles from a completed champion mirror collection. It first ranks training
battles by a fixed salted hash and alternates the chosen seat, giving ten roots
per seat. Within each selected battle it hash-ranks all boundaries requesting
that seat. Selection never uses outcomes, predictions, observed costs or failure
status, and never replaces a selected battle or boundary. Every recorded
nonterminal boundary is validated before seat eligibility: unique requested
roles must match actionable requests and exactly cover observations/actions;
committed actions must be legal, roots unique and chronological, and prefix and
root digests exact. The complete roster/config and selected champion contracts
are checked as well. Malformed records refuse instead of silently disappearing
from the candidate list.

The frozen address file binds the original roster/splits, collection, each
battle receipt, compressed source transcript, exact boundary, original root
identity, requests and actor observation. Freezing reads no feature arrays,
starts no simulator and makes no model or search calls. Terminal records and
outcome labels never enter selection.

The separate `reconstruct_cost_root` API can rebuild a selected boundary through
production protocol/parser/request-history ingestion without simulator reset,
step, branching or state restore. It checks every preceding requested actor's
recorded Python observation, maintaining the original tracker call order. Any
mismatch or missing root stops reconstruction; no skip or replacement exists.

The recorded source transcript is trusted mixed input, including omniscient
lines and both private request streams. It is routed exactly as the original
collector routed those streams. It must never be represented as public wire.
Search must receive the production `public_materialization_state` result, which
strips request payloads from the replay and carries only the actor's own request
and known move history separately. The other seat's private request is never a
policy input. These files do not establish JAX parity or action siblings.

Source tests use synthetic protocol/observation fixtures only. Actual saved-root
reconstruction and native timing require their own prospective review, source
and runtime hashes, cost-only quarantine namespace, fixed resource/concurrency
budget, absolute outer deadline and cleanup receipts. Cost observations must not
be pooled with strength or top-two sibling results. This helper does not launch
those phases or grant admission to scientific games.
