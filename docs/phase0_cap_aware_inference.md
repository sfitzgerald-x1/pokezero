# Prospective cap aware full game inference

This opt-in analysis follows the October 19 observation of a capped game in
the opponent-priors-OFF panel. It corrects a prospective design mismatch:
the human Phase 0 plan calls for worst-case handling of unknown outcomes,
whereas the first entry made any unknown cell an automatic disqualifier.

The active OFF entry keeps its registered advancement prohibition. Neither
this reader nor a supplementary calculation amends that entry, recovers its
capped outcome, or authorizes a replay. An unlaunched entry must register the
new method before collection and disclose its timing relative to earlier
observations, including observations on shared seeds.

## Test every compatible completion

Keep all 128 seed clusters and both seats in each arm. Validated wins,
losses and genuine draws score 1, 0 and 0.5. A round-capped game stays unknown
in [0,1]. Derive each seed's candidate-minus-control contrast interval.

`paired_t_completion_box` minimizes the nominal one-sided .025 paired-t lower
endpoint over every compatible completion. Simply filling candidate unknowns
with zero and control unknowns with one is insufficient: the studentized
test also depends on variance.

The endpoint is concave, so its minimum occurs at an interval-box vertex.
All contrast endpoints are quarter points. Scaling by four gives integers
from -4 to 4. A dynamic program retains the largest squared sum for each
attainable sum, at most 1,025 sums for 128 seeds. This finds a minimizing
completion without an unknown-count cutoff. Off-grid inputs are rejected,
not rounded. A possible constant completion leaves the test unavailable;
near-zero numerical margins do not pass.

Exact optimization does not make the t procedure finite-sample certified.
Its statistical interpretation remains approximate under the original
paired-seed assumptions, nominal .025 per candidate, not familywise .025.

## Admission remains separate

`summarize_capped_pairs` requires independent establishment of normal
completion, actual exit zero, clean cleanup and every registered cell
attempted. Every uncapped outcome must validate. Live, stopped, missing,
refused, failed or unvalidated-complete cells cannot enter this prospective
path. Such panels retain descriptive full-roster bounds.

The legacy `summarize_pairs` behavior is unchanged. Neither reader authorizes
deployment or retraining. The five-point deployment effect requirement uses
the adverse bound on possible panel point effects, not a five-point
lower-confidence-bound gate; timing and reliability still require separate
qualification.
