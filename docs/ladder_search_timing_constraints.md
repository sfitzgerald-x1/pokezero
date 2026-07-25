# Ladder Search Timing Constraints

**Status:** researched constraint baseline
**Scope:** Pokemon Showdown Gen 3 Random Battle ladder play
**Evidence snapshot:** 2026-07-25

## 1. Purpose

Test-time search must share a finite clock with observation construction, model inference,
network delivery, server processing, and recovery from a rejected choice. This document records
the external timing contract before we design an adaptive search scheduler.

The central result is:

> A fixed-duration policy that must remain safe for an arbitrarily long ladder game must submit
> every decision in strictly less than five server-side seconds. With the provisional delivery
> reserve below, PokeZero should cap ordinary search at three seconds from receipt of a request.

This is a safety bound, not a claim that every decision should receive three seconds. A future
scheduler may spend accumulated bank on selected high-value decisions.

## 2. Sources And Assumptions

The primary source is Pokemon Showdown `master` at
[`a97d40a3e71746c92ac238d9bd15ce550bf30617`](https://github.com/smogon/pokemon-showdown/tree/a97d40a3e71746c92ac238d9bd15ce550bf30617):

- [`server/room-battle.ts`](https://github.com/smogon/pokemon-showdown/blob/a97d40a3e71746c92ac238d9bd15ce550bf30617/server/room-battle.ts)
  defines the ladder timer, refill, acceleration, disconnect behavior, and timeout result.
- [`config/formats.ts`](https://github.com/smogon/pokemon-showdown/blob/a97d40a3e71746c92ac238d9bd15ce550bf30617/config/formats.ts)
  shows that Gen 3 Random Battle uses `Standard` and does not override the default timer.
- [`sim/SIM-PROTOCOL.md`](https://github.com/smogon/pokemon-showdown/blob/a97d40a3e71746c92ac238d9bd15ce550bf30617/sim/SIM-PROTOCOL.md)
  defines `|request|`, `rqid`, `/choose`, and invalid-choice responses.
- The official client at
  [`panel-battle.tsx`](https://github.com/smogon/pokemon-showdown-client/blob/2a5133088021c1fe2711a096802896b2055744a3/play.pokemonshowdown.com/src/panel-battle.tsx#L225-L282)
  updates the displayed countdown locally once per second. This display cadence is distinct from
  the authoritative server debit interval.
- The Showdown administrator's
  [timer explanation](https://www.smogon.com/forums/threads/ps-timer-updates.3646406/)
  provides historical intent; current source code is authoritative where the two differ.

The project's pinned local Showdown checkout at
`f76228a1354b5d0f307ca2d16101294ad3a2308b` has the same timer implementation as the upstream
snapshot above.

Ladder users can enable the timer. The bot must therefore behave as if the timer is on even if a
particular battle begins with it off.

Pokemon Showdown's linked
[Bot FAQ](https://gist.github.com/Kaiepi/becc5d0ecd576f5e7733b57b4e3fa97e) also warns that
accounts connecting from some VPS and proxy hosts may be locked automatically. Before sustained
ladder use, confirm the account and intended hosting environment with Pokemon Showdown staff;
timer safety does not bypass server policy or account trust requirements.

## 3. Exact Ladder Timer Contract

For default singles ladder battles:

| Constraint | Value |
|---|---:|
| Visible client countdown | 1-second local updates |
| Authoritative server debit interval | 5 seconds |
| Initial total bank | 210 seconds |
| Initial bank before grace | 150 seconds |
| Initial grace | 60 seconds |
| Maximum time for one request | 150 seconds |
| Refill through turn 100 | 10 seconds per new turn |
| Refill on turns 101-200 | 5 seconds per new turn |
| Refill after turn 200 | 5 or 0 seconds, alternating in an ordinary one-request-per-turn game |
| First mid-turn request | 5 seconds |
| Disconnect deadline | 60 seconds |
| Timeout behavior | forfeit; no automatic legal choice |

Important qualifications:

- The browser visibly counts down once per second. Internally, the current server subtracts five
  seconds from the authoritative bank every five seconds. The client interpolates the seconds
  between server timer messages; it is not evidence that the server mutates the bank each second.
- A new-turn refill is capped at 150 total seconds. The extra 60 seconds is initial grace, not a
  permanent 210-second bank.
- The 150-second request cap applies even when the total bank is higher.
- Forced replacements and other mid-turn requests are separate decisions. Search budgeting must
  operate per `|request|`, not merely per displayed battle turn.
- Mid-turn requests affect Showdown's request counter, so code should observe reported clock
  state rather than infer post-turn-200 refill parity from the displayed turn number.
- `Endless Battle Clause` prevents deliberately endless loops but does not define a small fixed
  maximum turn count. The safe constant-duration result must therefore handle an unbounded
  horizon.

## 4. Constant-Duration Derivation

Let `d` be elapsed server-side time from issuing a request until Showdown processes the choice.
Away from the exact tick boundary, charged time is:

```text
charged(d) = 5 * floor(d / 5)
```

Submitting exactly at 5, 10, or 15 seconds is a race with the server timer and is not safe.

| Battle phase | Refill | Largest steady request interval without spending bank |
|---|---:|---:|
| Through turn 100 | 10 seconds | strictly less than 15 seconds |
| Turns 101-200 | 5 seconds | strictly less than 10 seconds |
| Post-200 request with 5-second refill | 5 seconds | strictly less than 10 seconds |
| Post-200 request with zero refill | 0 seconds | strictly less than 5 seconds |

Therefore:

- A constant duration used for every decision in an arbitrarily long game must be **strictly
  below five server-side seconds**.
- There is no inclusive "five seconds" safe setting because delivery at the tick boundary can be
  processed after the first decrement.
- A phase-aware scheduler could alternate larger and smaller budgets after turn 200, but should
  use observed bank and request state rather than assume parity.
- Before turn 100, a sub-15-second response can be bank-neutral in the timer's quantized model.
  That is available capacity, not a recommended default.

### Finite-horizon example: ten-second decisions

Assume the timer is enabled from the first decision, the player receives exactly one request per
displayed turn, and each choice reaches the server after the second five-second debit. Then a
uniform ten-second response:

1. spends ten seconds on turn 1;
2. is bank-neutral from turns 2 through 100 because each new turn restores ten seconds;
3. spends five net seconds per turn from turn 101 onward because the refill drops to five seconds;
4. completes turn 127 with five seconds left and forfeits on turn 128 when the next ten-second
   charge reaches zero.

Thus the conservative answer is **127 completed turns, with timeout on turn 128**. Forced
replacement requests or other mid-turn decisions shorten that horizon. "Exactly ten seconds" is
also a server-event-loop boundary race: a choice processed just before the second debit can be
charged only five seconds, while one processed just after it is charged ten. Production budgeting
must not rely on winning that race.

## 5. Networking And Submission Reserve

The timer starts on the server before the `|request|` reaches the client. The client-side budget
must reserve time for:

1. Request delivery and event-loop dispatch.
2. Observation and legal-action construction outside the search timer.
3. Search cancellation and selection of a prepared fallback.
4. Outbound WebSocket delivery and Showdown command processing.
5. A short transient stall or one explicit invalid-choice recovery.

A 20-packet sample from the development laptop to `sim3.psim.us` on 2026-07-25 measured
86-95 ms round-trip time with no packet loss. This is a useful smoke measurement only; it is not
a production service-level objective and does not cover WebSocket queues or cluster placement.

Use this provisional reserve until ladder telemetry replaces it:

```text
delivery_reserve = max(2.0 seconds, 4 * rolling_p99_websocket_rtt + 0.5 seconds)
```

At the measured RTT, the two-second floor binds. The resulting bank-preserving ordinary search
deadline is:

```text
search_deadline_from_request_receipt = 3.0 seconds
```

Two seconds is intentionally conservative because an awaited WebSocket `send()` can report a
local write without proving that Showdown accepted the choice. The
[`websockets` latency API](https://websockets.readthedocs.io/en/15.0/topics/keepalive.html)
can measure Ping/Pong round-trip time, but application-level acceptance must be inferred from
Showdown protocol messages.

The reserve must be recalibrated from the actual ladder host:

- Record Ping/Pong RTT continuously.
- Record request-receipt to send-start and send-completion latency.
- Record request-receipt to next accepted battle event.
- Track p50, p95, p99, maximum, reconnects, invalid choices, and timer losses.
- Increase the reserve automatically when recent p99 latency rises.

## 6. Can A Turn Fail To Submit?

Yes. A turn can be lost even when model search completes before its nominal deadline.

| Failure | Consequence | Required behavior |
|---|---|---|
| WebSocket closes before or during send | choice may never reach Showdown | detect promptly, reconnect, and recover the current request |
| Choice arrives after the timer tick or timeout | bank is charged or the game is forfeited | stop search before the delivery reserve |
| Stale request | choice can target the wrong decision | append the request's `rqid` to every `/choose` |
| Illegal or unavailable choice | server returns an error and still expects a choice | consume the error/update and submit a legal fallback before the deadline |
| Search blocks the receive loop | timer, error, and disconnect messages queue unread | run search outside the WebSocket reader |
| Search crashes or overruns | no action is available at the deadline | prepare a base-policy action before starting search |
| Local send completes without server acceptance | client may falsely mark the turn complete | maintain request state until protocol evidence advances it |

Blindly retransmitting a choice is not an acknowledgment strategy. Retries must be keyed by
`(room_id, rqid)` and reconciled with the latest server request.

## 7. Current PokeZero Client Gaps

[`src/pokezero/online_client.py`](../src/pokezero/online_client.py) is sufficient for controlled
online probes, but it is not yet a fault-tolerant ladder search client:

- `OnlineBattleAgent.choose()` runs synchronously in the WebSocket receive loop.
- `/choose` does not include `REQUEST.rqid`.
- Invalid and unavailable choice messages do not trigger fallback handling.
- The outer connection has no reconnect-and-resume loop.
- No per-request watchdog cancels search and submits a prepared fallback.
- Timer-bank and WebSocket-latency telemetry are not retained.
- Successful local `send()` is not distinguished from server acceptance.

These are launch blockers for expensive ladder MCTS, not reasons to weaken search quality in the
offline harness.

## 8. Constraints For The Future Scheduler

The scheduler should not use "late game" as its only signal. A decisive decision can occur before
the state is fully revealed. It must instead combine:

- remaining total bank and current request time;
- hard delivery reserve;
- model uncertainty or policy entropy;
- disagreement between shallow search and the policy prior;
- tactical features such as imminent KOs, forced switches, or irreversible commitments;
- information value and number of plausible hidden worlds;
- measured search throughput for the current state.

Non-negotiable scheduler invariants:

1. Produce and retain a legal base-policy fallback before deeper search.
2. Never let search consume the delivery reserve.
3. Treat every `|request|`, including forced replacements, as a separately budgeted decision.
4. Use the server-reported clock when available and a conservative estimate when it is not.
5. Preserve a configurable total-bank floor rather than spending the opening bank uniformly.
6. Log the planned budget, actual search time, send time, acceptance evidence, and remaining bank
   for every decision.

## 9. Open Measurements

Before choosing final adaptive depth/breadth settings:

1. Run a local timer-enabled integration test that injects delayed requests, invalid choices,
   dropped connections, and tick-boundary sends.
2. Measure WebSocket RTT and end-to-end choice acceptance from the intended ladder execution
   environment.
3. Measure MCTS quality gain per additional 100 ms across representative early, middle, and late
   states.
4. Replay complete games through candidate budget controllers and compare strength, timeout risk,
   and bank trajectory.
5. Validate an emergency base-policy fallback under forced-switch and updated-request paths.

This document should be revised if Pokemon Showdown changes its timer source or if Gen 3 Random
Battle gains a format-specific timer rule.
