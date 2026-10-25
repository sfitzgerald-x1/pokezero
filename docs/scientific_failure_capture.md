# Scientific failure capture and diagnostic replay

Scientific runs must preserve refusals and their inputs. A failed roster stays
closed: do not resume it, redraw a failed world, substitute a root, extend its
slots, or use a diagnostic as a scored observation.

## Diagnostic replay

A transition-only diagnostic is allowed after its scope is declared and before
it runs. It may inspect a saved failed root using the identified build, but may
not select an action, score a game, resample a world, or write to any roster.
All outputs belong in an isolated quarantine directory. Its receipt must name
the root, build fingerprint, recovered reason, declaration, and zero scientific
writes, selected actions and scored games. Derive the regression fixture from
the captured bundle rather than reconstructing a convenient example.

This standing permission distinguishes diagnosis from scientific replay. It
does not authorize a new roster, a change to scientific settings or a mechanics
repair unrelated to an authorized deliverable.

## Capture before raising or dropping a world

`pokezero.failure_capture` supplies the shared durable writer and decision
scope. World constructors capture their input payload, flags and sampled
override before an outer handler can drop the world. Native branch and encoder
boundaries attach structured origin evidence before unwinding. The policy
writes that evidence to quarantine before propagating the original failure or
using an already specified fallback.

Each bundle includes the actual request, public log prefix, observation, root
identity and RNG state; the world specification and native serialized state;
the generating instruction list when one exists; stable reason codes; and the
offending instruction when uniquely identified. A failure before instruction
generation, or an ambiguity involving a whole branch, records that limitation
explicitly instead of inventing an instruction. Batched failures preserve the
per-row inputs and identify the failing row when native code provides it.

Files are private, exclusive, hash-linked and fsynced. Capture failure raises
`CaptureFailed`, outside ordinary catch-and-continue handlers: a run must not
silently proceed after losing its evidence. Input and terminal receipts reveal
an interrupted decision; an unmatched input is not evidence of successful exit.
Reason counts distinguish failure events from multiple tags on the same event.

## Scientific admission

P2 requires capture and fixture admission before creating its decision trace.
The Foul Play and head-to-head scientific owners require the same contract for
the incumbent engine policy. Set `POKEZERO_QUARANTINE_DIR` to the isolated run
directory and `POKEZERO_CAPTURE_ADMISSION` to its reviewed admission manifest.
Library fixtures may use the same capture scope without claiming scientific
admission. Other callers must explicitly bind scientific capture before they
can be used for a scientific roster.

Admission binds the actual runtime fingerprint, a file-backed inventory of
closed source runs and recovered refusal classes, and a bundle-derived fixture
for every observed class on that path. Each fixture links its source bundle and
diagnostic receipt; verification names the regression test, fixture digest,
actual build and successful exit. Missing, changed, wrong-root or unverified
evidence refuses admission. This checks evidence consistency; prospective
independent review establishes that the inventory covers the actual run history.

Passing a fixture does not reopen its original roster. A later scientific run
needs a fresh reviewed entry, unchanged decision rules, and its own accounting.
