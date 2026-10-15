# Reference worker cleanup

The actual persistent reference pool now keeps its original worker identity
until cleanup is resolved. This repairs saved process-group signalling after
reaping and surfaces runtime teardown failure. It changes lifecycle handling,
not the search algorithm, configuration, model or accepted scientific evidence.
It does not authorize games or qualify complete benchmark cost.

## Ownership and release

Each spawned worker isolates its POSIX session and reports that identity before
constructing a runtime, then waits for acknowledgment. The parent binds the
receipt to the actual spawned process PID and live OS session/group before
releasing construction, not merely a PID supplied by the pipe. Worker-local observation
retains actual `subprocess.Popen` handles, including constructors that spawn a
child and then raise. The worker's terminal cleanup handshake reports a runtime
close error and its still-live observed child handles; it remains the live group
leader until parent acknowledgment or forced termination.

The parent uses non-reaping `waitid` observation for failure diagnostics and
cleanup. `Process.is_alive`, `exitcode` and `join` can poll/reap; they are not
identity checks before signalling. CPython also polls older children during a
new `Process.start`. Startup checks therefore revoke group authority if a prior
leader has already terminated or been reaped. Unexpected death means uncertain
cleanup and refusal, not permission to signal a stale saved group.

Clean runtime closure with no live observed child permits acknowledgment and
normal worker exit. Forced cleanup first confirms the original leader is live,
unreaped and stopped, and its session/group identity still matches. Reported
direct children are signalled only after their parent PID is checked while
that worker's reaping is frozen. The last group signal occurs before reaping;
there is no TERM/sleep/KILL sequence after releasing the leader. A signal or
inspection failure cannot abort attempts on later workers. Exact unreaped-child
fallback and bounded joins retain errors without guessing another group.

The six-second runtime-close grace is shared across workers. POSIX non-reaping
ownership support is required before spawning. The pool owns one decision and
its teardown; concurrent external reaping of its child processes is not a
supported ownership protocol.

Initial pipe readability is not a completed message. Both normal failure
diagnostics and cleanup read the CPython POSIX framing with nonblocking reads,
shared deadlines and a 64 MiB frame limit. A partial frame from a stopped writer
refuses transport and still reaches owned cleanup; no background reader or
replacement worker is launched. This is the CPython POSIX transport contract,
not arbitrary IPC support.

## Refusal and evidence

`cleanup_evidence` distinguishes `CLEAN_OBSERVED_SCOPE` from `REFUSED`. Worker
records retain runtime-close error, observed live children, whether a group was
signalled before reaping, actual exit code, reaping status and cleanup exceptions.
A final group SIGKILL kills the waiting leader too: its actual exit is -9, not
an invented successful exit. Forced cleanup, malformed/missing handshake,
unexpected leader death or any cleanup exception refuses the pool.

Malformed pipe envelopes and signal/join errors cannot strand the rest of the
pool or bypass final evidence and connection closure. The parent preserves an
existing search/startup failure as the primary exception and adds cleanup
failure to its evidence. An otherwise successful context propagates cleanup
failure. Repeated close does not restart cleanup or erase the original record.

The clean status describes the observed runtime/child scope only. Both
`complete_descendant_accounting` and `cpu_accounting_complete` remain false.
Unobserved descendants, escaped grandchild sessions, independent source/runtime
qualification and full resource cost are not certified. The controlled benchmark
cost consumer still needs an explicit outer ownership integration before it
can admit reference recipes; this pool change alone does not enable them.

## Verification

Tests use actual spawned workers, subprocesses and the actual pool teardown,
with authored toy runtimes and no Pokémon/model games. They cover persistent
twenty-worker exchange, clean child reaping, stopped simulator child refusal,
constructor-spawns-then-raises, pre-boot hang, shared close grace, close error,
stopped versus terminal observation, CPython startup reaping, no post-reap group
signal, malformed cleanup envelopes, injected signal/join errors and retention
of the primary search error. The normal algorithm and atomic-work regressions
remain included.

```sh
PYTHONPATH=src:tests python -m unittest \
  test_paper_reference_cleanup test_paper_reference_parallel \
  test_paper_reference test_paper_reference_exchange \
  test_search_over_raw_atomic_work -v
```

Parent final verification passed 73 tests in 42.235 seconds, exit 0, no skips,
including the isolation acknowledgment and actual stopped partial-frame paths.
Independent Astra verification of the final source passed 25 parallel/cleanup
tests in 41.474 seconds, exit 0 without skips, with matching source/test hashes
and no remaining scoped blocker. This clears publication of lifecycle handling,
not cost-consumer integration or scientific qualification.
The existing dedicated reference workflow includes the new suite. Initial
verification failures remain failures: the old stopped-child fixture assumed
silent successful cleanup; a short stopped-transport timeout accidentally also
bounded startup. Independent review subsequently found malformed-message and
per-worker-exception gaps despite a passing 67-test intermediate run. Those
gaps were repaired and negatively tested. A later review found a partial-frame
receive could bypass the cleanup deadline, repaired in both receive paths and
tested against an actual stopped writer. These are engineering results, not
fresh-root scientific progress or complete cost proof.
