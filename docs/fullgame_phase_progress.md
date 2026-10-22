# Controlled full game phase visibility

The single-benchmark CLI accepts an optional `--phase-progress-out NEW_PATH`.
The default is off. A new path is required for each invocation; existing files,
including old progress files and symlinks, are refused rather than adopted.
The path must differ from `--summary-out`. The multi-arm comparison CLI does
not offer this flag.

The bounded atomic JSON snapshot records a run identity, process ID, sequence,
public seed and seat, zero-based decision-round index, phase, started/returned
status, and monotonic and wall timestamps. It also retains the last returned
phase. It never receives actions, requests, observations, simulator states,
exception messages or outcomes. It is separate from scored summaries.

Phases cover game entry/return, bridge-event waits, decision boundaries,
candidate and neural-opponent controller calls, external-opponent choice
waits and choice submission. A started marker is written immediately before
the relevant operation; a returned marker follows successful return. A
pending call or exception therefore leaves its started marker visible.
Candidate controller time also includes any configured continuation oracle;
this is not native-search internal telemetry. Game return precedes result
publication and is not proof of a durably saved or validated game. Run return
means the benchmark returned, not that the outer process exited zero.

An observer can distinguish increasing decision indices within a long game
from an unchanged candidate-policy start or bridge wait. A marker alone does
not prove a deadlock, identify a native branch or explain a historical stop.
The timestamp is a phase observation, not a periodic heartbeat. Disk writes
add overhead outside the existing policy-call timers; enabled wall-clock
runs must account for that instrumentation. I/O/ownership failures stop the
invocation rather than silently claiming liveness.

The instrument does not change policy inputs, RNG streams, search work,
choices, caps or supervisor code. It must not reset a completed-game watchdog,
count as accepted scientific work, resume a stopped roster, or authorize a
new collection entry. Completed-game and actual terminal receipts remain the
authoritative sources for outcomes and execution status.
