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

## 9. Adaptive MCTS Time Strategy

There should not be a fixed "start deep search on turn N" rule. Battle turn is a weak proxy for
the quantities that actually determine whether depth is useful:

- how many opponent sets and backline assignments remain plausible;
- how many root actions are competitive;
- how much the sampled worlds disagree about the best action;
- whether shallow and deeper searches select different actions;
- whether the current line contains an irreversible or forcing decision;
- how much clock remains above the emergency floor.

The search budget has two distinct breadth axes:

1. **Root-action breadth:** cover every legal move and switch before concentrating visits.
2. **Belief-world breadth:** cover materially different hidden teams, items, abilities, and
   movesets before deeply optimizing against one sampled world.

Early positions will usually favor belief-world breadth because hidden-state uncertainty is high.
Later positions will often favor depth because reveals reduce the effective number of worlds and
fewer surviving Pokemon reduce the action tree. These are tendencies, not phases: an early forced
tactical sequence can justify depth, while a late unrevealed Choice item can still justify world
breadth.

The current `spike-sac` scenario is already a warning against making "shallow but wide" absolute.
At depth 2, increasing simulations made both tested checkpoints increasingly confident in a wrong
action. Depth 4 selected the preservation switch throughout the sweep. Breadth can reduce sampling
error, but it cannot repair a search horizon that excludes the decisive consequence.

### Research basis

- Baier and Winands,
  [*Time Management for Monte Carlo Tree Search*](https://dke.maastrichtuniversity.nl/m.winands/documents/time_management_for_monte_carlo_tree_search.pdf),
  divide remaining time by estimated remaining moves, extend searches when the top two root moves
  are close, and stop early when the runner-up cannot catch the leader with the expected remaining
  simulations.
- Lan et al.,
  [*Learning to Stop: Dynamic Simulation Monte-Carlo Tree Search*](https://ojs.aaai.org/index.php/AAAI/article/download/16100/15907),
  label state/search uncertainty using a much larger reference search and learn when a smaller
  search can stop.
- Kaufmann and Koolen,
  [*Monte-Carlo Tree Search by Best Arm Identification*](https://proceedings.neurips.cc/paper/2017/hash/a6d259bfbfa2062843ef543e21d7ec8e-Abstract.html),
  formulate root selection as best-arm identification and propagate confidence intervals from
  deeper levels.
- Silver and Veness,
  [*Monte-Carlo Planning in Large POMDPs*](https://proceedings.neurips.cc/paper_files/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html),
  sample root states from the current belief during online planning.
- Cowling, Powley, and Whitehouse,
  [*Information Set Monte Carlo Tree Search*](https://eprints.whiterose.ac.uk/75048/),
  identify duplicated computation and strategy fusion as core weaknesses of independently searching
  a small number of determinizations.

### Proposed anytime controller

Run search in short blocks and decide after each block whether the next block buys more value as
world coverage or tree depth:

1. Prepare a legal base-policy fallback.
2. Complete a minimum tactical-depth sweep across every legal root action and an initial set of
   belief worlds. The minimum depth must be validated on the endgame scenario suite; `spike-sac`
   currently rules out depth 2 as a safe universal baseline.
3. Add belief worlds while the top-action ranking or value estimate changes materially when world
   coverage doubles.
4. Add within-world visits/depth when world-level rankings agree but the top root actions remain
   close, the principal variation is unstable, or the policy prior and search disagree.
5. Stop when confidence intervals separate the best action, the choice remains stable across a
   budget doubling, or the hard delivery deadline is reached.

The useful threshold is therefore a **value-of-computation crossover**, not a battle turn:

```text
marginal action-quality gain per millisecond from more depth
    >
marginal action-quality gain per millisecond from more belief-world coverage
```

### How to learn the threshold

Build a held-out corpus of real decision states, preferably from games against an independent
opponent, stratified by turn, Pokemon remaining, revealed-team fraction, belief entropy, legal-action
count, policy entropy, and tactical tags. For each state:

1. run nested world-count, simulation-count, and depth budgets with matched random seeds;
2. retain action rankings, values, confidence intervals, world disagreement, maximum depth reached,
   and wall time after every block;
3. use a much larger search as a reference, not as proof of optimality;
4. use exact scenario proofs where the endgame studio can establish a true forced line;
5. label the cheapest point that matches a near-optimal reference action within a chosen regret
   tolerance;
6. fit a small auditable scheduler or lookup table and validate it on entirely held-out battles.

The first scheduler should use direct search statistics rather than a learned auxiliary network:
top-two visit/value gap, action changes across blocks, per-world action disagreement, effective
belief sample size, and remaining-bank pressure. A learned uncertainty head is justified only if
those transparent signals leave substantial strength or clock efficiency on the table.

## 10. Open Measurements

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
6. Run the nested depth, simulation, and belief-world sweep above to locate the empirical
   breadth-to-depth crossover and test whether one global threshold is adequate.

This document should be revised if Pokemon Showdown changes its timer source or if Gen 3 Random
Battle gains a format-specific timer rule.
