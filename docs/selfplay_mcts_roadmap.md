# Roadmap: self-play + test-time MCTS toward ladder-competitive play

Status: active execution. This is the multi-workstream plan for getting pokezero from "coherent
but plateaued" to genuinely strong Gen 3 random-battle play, learned from first principles via
self-play (see [`goals.md`](goals.md)). It is written so independent agents can each own a
workstream in parallel and keep the critical path visible as evidence changes.

## The proven recipe we are following

The MIT thesis *"Winning at Pokémon Random Battles Using Reinforcement Learning"* (Jett Wang,
2024) reached **rank 8 / 1693 Elo** on the official gen4 random-battle ladder — the best known
non-human result — with a recipe that is directly applicable and was achieved on **modest
compute** (one GPU + ~80 CPU workers, ~3M self-play battles, ~4 days):

- **Train a policy/value net via PPO self-play — *without* MCTS in the loop.** The thesis explicitly
  diverges from AlphaZero here: the simulator is too slow to generate MCTS-improved targets at the
  scale a net needs to converge. PPO self-play is the training engine.
- **Add MCTS only at inference**, as a *policy-improvement operator* on top of the trained net.
- **Determinize hidden information**: sample the opponent's hidden set → the rollout becomes a
  (near) perfect-information game.
- **Restore the Markov property** by encoding multi-turn effect durations into the observation.

This de-risks the direction: it is an engineering+scale problem with a known-good shape, not a
research gamble. Our job is to reproduce it for Gen 3 on our stack and push past it.

The exact reference numbers — training budget (~3M battles / ~4 days), the full PPO hyperparameter
table, architecture, the inference-time MCTS details, and a line-by-line gap against our current
config — live in [`mit_thesis_reference_config.md`](mit_thesis_reference_config.md). Two facts from
it gate everything below: (1) the thesis's **net-alone** was already strong (~80% vs a heuristic
baseline) — the PPO **training** half is the load-bearing phase, and MCTS is a topper on top of it;
(2) that net needed **~3,000,000 self-play battles** with specific tuned hyperparameters (annealed
LR, `entropy_coef` 0.0588, 7 epochs, `gamma` 0.9999). Recipe **fidelity and scale** are therefore
the first-order levers, and no strength conclusion is meaningful until we are running near them.

## Current assets (what already exists)

- **Warm-pooled sim** (`local_showdown.py`, `scripts/battle_bridge.mjs`): battle-id-keyed bridge,
  ~0.4 ms/turn warm, byte-identical battles, ~2× collection throughput. Collectors reuse one env.
- **Replay-from-root prefix harness** (`replay_branching.py`): rebuilds a battle branch point from
  the original seed + recorded action prefix, with real-sim equivalence tests modulo Showdown
  timestamp lines, submits explicit divergent branch actions, then rolls out from the resulting
  state with normal policy semantics.
  `rollout_cli replay-benchmark` measures prefix-replay latency before considering snapshot/restore.
- **Flat branch-search harness** (`search.py`): enumerates legal root actions, evaluates each via
  replayed branch rollout, and selects by terminal outcome. This is a deliberately small search
  stepping stone before value-guided MCTS.
- **Value-guided branch scoring** (`search.py` + transformer value helper): enumerates legal root
  actions through the replay branch harness and scores non-terminal post-branch states with a value
  function over the player's observation history. This is the bridge from flat terminal rollout
  search toward PUCT/MCTS leaf evaluation.
- **Root PUCT branch scorer** (`search.py` + transformer prior helper): combines policy-head priors
  with value-branch scores at the search root, establishing the first PUCT-style selection layer
  before deeper tree expansion. `neural_cli root-puct-benchmark` evaluates that scorer on sampled
  rollout prefixes to measure action deltas and search latency, while
  `neural_cli root-puct-counterfactual` replays the recorded and selected branches forward for a
  first retrospective outcome signal before changing live rollout control.
- **Context-aware root-PUCT policy adapter** (`PolicyContext`, `RootPUCTSearchPolicy`): the rollout
  driver can now call policies with player-local decision context, and the root-PUCT adapter can
  select actions through a separate branch env without mutating the live rollout. Simultaneous-turn
  opponent actions are supplied by an explicit planner hook; `greedy_opponent_action_planner` can
  drive that hook from player-local opponent-action priors such as the transformer's auxiliary
  opponent-action head. In self-play/search benchmarks, the harness can additionally mask planned
  opponent actions against requested-player legal masks to avoid invalid branch submissions; this is
  recorded as search metadata. Because the mask includes opponent legal-action availability, this is
  a privileged benchmark safety guard and is not a substitute for full hidden-information
  determinization. The play benchmark can also opt into `--root-opponent-action-policy benchmark`,
  which asks the fixed benchmark opponent policy to choose the simultaneous non-search root action
  from its own private observation. That mode is intentionally privileged evaluation plumbing: it
  is useful for isolating fixed-opponent search modeling, but its win rates should not be read as
  hidden-information-realistic strength.
  The play benchmark also has an opt-in conservative value gate (`--min-value-improvement`) that
  keeps the raw policy-prior action unless the configured root-selected action beats it by a configurable
  value margin; aggregate diagnostics report gate uses/checks so M0 runs can tell whether the knob
  is actually changing decisions.
  The same benchmark can now switch root selection between current PUCT-score selection and pure
  branch-value selection (`--selection-mode puct|value`), which isolates whether the near-term M0
  issue is the branch value signal or the PUCT prior/score mixer.
  It can also run opt-in bounded leaf continuations (`--leaf-rollout-rounds`) before evaluating a
  root candidate, using real simulator steps to test whether short rollout leaves produce a better
  root value signal than immediate one-ply value-head evaluation. Leaf continuations default to raw
  checkpoint-vs-checkpoint policies, but `--leaf-rollout-opponent-policy benchmark` can use the
  fixed benchmark opponent for the non-search side during leaf rollouts. `--leaf-rollout-rounds-sweep`
  can run multiple leaf depths in one same-seed benchmark artifact so depth comparisons are less
  error-prone than separate one-off commands.
  `neural_cli root-puct-play-benchmark` compares raw checkpoint play against root-PUCT checkpoint
  play over full games on the same fixed-opponent matchups.
  A first local smoke run against `max-damage` proved the full-game path executes end to end with a
  current-schema checkpoint: 4 total games at `--games 1`, active-search decisions around
  3-4 decisions/s, and no runtime crash. This is harness evidence only, not strength evidence: the
  tiny 4-game-trained smoke checkpoint lost to `max-damage`, and one root-PUCT orientation capped at
  the 50-decision smoke limit.
  A first small current-schema probe then trained a 64-game scripted-teacher BC checkpoint and ran
  `--games 4` against `max-damage`: raw scored 3/8, root-PUCT scored 2/8, both with zero capped
  games. This is too small to decide strength, but it is negative search-lift evidence for the
  current shallow operator and confirms active-search throughput around 3 decisions/s.
  Benchmark JSON now includes root-PUCT decision diagnostics so follow-up M0 runs can see search
  counts, fallbacks, candidate counts, selected values/scores, and per-decision search latency.
  After adding opponent-action legality masking, a repeat `--games 4` probe on the same 64-game
  teacher-BC checkpoint showed the known invalid-action fallback was resolved (`211` root-PUCT
  searches, `0` fallbacks), but still did not show strength lift: raw and root-PUCT both scored
  1/8 vs `max-damage`, with zero capped games and active-search throughput around 2.7-2.8
  decisions/s. This points back to operator/value quality rather than submit-validity as the next
  M0 bottleneck.
  A follow-up probe with the conservative value gate on the same 64-game teacher-BC checkpoint and
  a fresh fixed seed range also failed to show lift: at `--min-value-improvement 0.0`, the gate made
  no changes (`0/244` uses/checks) and raw/root-PUCT both scored 3/8 vs `max-damage`; at
  `--min-value-improvement 0.25`, the gate changed a few decisions (`8/245`) but raw/root-PUCT still
  both scored 3/8. Both runs had zero fallbacks and zero capped games, so the next M0 work should
  focus on improving the search/value signal itself rather than only adding conservative vetoes.
  A value-selection probe (`--selection-mode value`) on the same checkpoint and seed range also
  scored 3/8 vs `max-damage`, matching raw and same-seed PUCT despite selecting from a different
  value/score profile. That makes "PUCT prior mixing alone" a weaker explanation for the current
  miss; the more likely M0 bottlenecks are value-head quality, one-ply depth, and opponent/chance
  modeling.
  A value-calibration pass on the same checkpoint's own 64-game training rollouts reinforces the
  value-quality concern even before testing generalization: over 3,809 in-sample examples, the value
  head measured MSE 0.752, MAE 0.810, sign accuracy 0.722, and expected calibration error 0.187.
  That is too noisy to trust as the only one-ply search leaf signal. The follow-up WS-E plumbing now
  makes this auditable rather than implicit: new transformer configs bound value outputs with `tanh`,
  calibration reports include return/turn/terminal slices, and `neural_cli train` / `neural_cli iterate`
  can write opt-in calibration artifacts. A one-epoch current-schema tanh teacher-BC split run then
  showed the remaining bottleneck clearly: held-out value calibration over 598 examples measured MSE
  1.036, MAE 0.960, sign accuracy 0.527, ECE 0.299, with positive-return sign accuracy only 0.048.
  This is not a search-ready value head; it is evidence that value/base-net quality must improve before
  more low-sample search tuning can produce a meaningful M0 verdict.
  A first bounded-rollout-leaf probe (`--leaf-rollout-rounds 1`) on the same checkpoint and
  max-damage seed range gave a small positive but still far-from-gate result: raw scored 3/8 and
  root-PUCT with one rollout leaf round scored 4/8, with zero capped games and zero search
  fallbacks. The run recorded 252 root-PUCT searches, actual candidate leaf rounds split across
  0-round immediate terminals and 1-round continuations, and active-search throughput around
  2.2-2.4 decisions/s. This is exploratory evidence that replacing immediate one-ply leaves may
  matter, but it is not enough to clear M0 or justify scale; the next read should either enlarge
  this leaf-depth probe or improve value/opponent modeling further.
  A same-seed depth-2 follow-up moved in the same direction: raw again scored 3/8, while
  root-PUCT with `--leaf-rollout-rounds 2` scored 5/8, with zero capped games and zero search
  fallbacks across 255 root-PUCT searches. Candidate leaf diagnostics showed 9 immediate terminal
  branches, 20 one-round continuations, and 1,419 two-round continuations; 80 candidate branches
  reached rollout terminals and 1,359 still required truncated leaf value evaluation. Active-search
  throughput stayed around 2.1-2.3 decisions/s. The sample is still too small for a strength claim,
  but the one-step and two-step probes now both point toward leaf depth/value-target quality as a
  better next M0 lever than more PUCT-score mixing.
  A larger same-seed sweep using the new `--leaf-rollout-rounds-sweep 0/1/2` harness then ran
  `--games 8` per mirrored matchup against `max-damage` on a fresh seed range. It preserved the
  same direction but still did not clear M0: raw scored 3/16, leaf0 scored 4/16, leaf1 scored 4/16,
  and leaf2 scored 5/16, with zero capped games and zero search fallbacks in every searched row.
  Leaf2 ran 436 root-PUCT searches at roughly 2.3-2.4 active-search decisions/s; candidate leaf
  diagnostics recorded 23 immediate terminal branches, 51 one-round continuations, 2,473 two-round
  continuations, 101 rollout-terminal candidate leaves, and 2,423 truncated value-head leaves. This
  strengthens the case that deeper leaf evaluation is the right M0 lever, but the absolute score
  remains much too low; next work should either run a larger depth sweep for variance reduction or
  improve the leaf evaluator/opponent model before spending larger training compute.
  A follow-up benchmark-opponent leaf rollout sweep did **not** improve that same M0 read. On the
  same `14062001` seed band and checkpoint, raw again scored 3/16 and leaf0/leaf1 matched 4/16, but
  leaf2 dropped from the default-opponent sweep's 5/16 to 4/16. A fresh `16062001` seed band showed
  the same benchmark-opponent shape: raw 3/16, leaf0 4/16, leaf1 4/16, leaf2 4/16. Both runs had
  zero capped games and zero root-PUCT fallbacks, and the JSON diagnostics confirmed
  `root_puct_leaf_rollout_opponent_policies: {"benchmark": ...}` for leaf1/leaf2 rows. This means
  simply matching the fixed benchmark opponent inside short leaf continuations is not enough to
  clear M0; the next useful search work should focus on value-target quality, root opponent-action
  modeling, determinization, or deeper search rather than only swapping the leaf opponent policy.
  A same-seed sweep that aligned both the simultaneous root opponent action and the leaf rollout
  opponent against the fixed `max-damage` policy produced the strongest search-lift signal so far,
  but only in the privileged evaluation setting. On seed band `14062001`, raw again scored 3/16;
  benchmark-root leaf0 scored 5/16, benchmark-root+leaf1 scored 7/16, and benchmark-root+leaf2
  scored 4/16, with zero capped games and zero root-PUCT fallbacks. Diagnostics confirmed
  `root_puct_opponent_action_policies: {"benchmark": ...}` for every searched row and
  `root_puct_leaf_rollout_opponent_policies: {"benchmark": ...}` for leaf1/leaf2. This suggests
  root opponent-action mismatch was a real part of the earlier search miss, while the leaf-depth
  result is not monotonic. Next work should convert this privileged fixed-opponent signal into a
  hidden-info-safe opponent model or determinization path before treating it as M0 progress.
  The first less-privileged conversion step now exists as `--root-opponent-action-scenarios`:
  instead of collapsing the simultaneous opponent root action to one greedy checkpoint-prior action,
  the search policy can enumerate a bounded top-k set from the checkpoint opponent-action head using
  the acting player's history, run each root candidate against those scenarios, and average branch
  values by scenario weight. This avoids asking the fixed benchmark opponent for a
  private-observation root action, but it still keeps the benchmark legal-mask safety guard; that
  guard is privileged and shapes both scenario support and scenario weights. This remains a harness
  step, not M0 evidence until measured and not a replacement for belief determinization.
- **Value-head calibration report** (`value_calibration.py`, `neural_cli value-calibration`):
  measures MSE/MAE/bias/sign accuracy, linear value-return Pearson correlation, and predicted-value
  calibration bins against rollout return targets; reports stratified return/turn/terminal slices; and
  can be emitted from standalone calibration, `neural_cli train`, or `neural_cli iterate`. This is the
  first WS-E metric before using the value head for MCTS leaf evaluation. The standalone command can
  also fit a calibration transform and save a calibrated checkpoint copy; pass a separate
  `--eval-data` set for a held-out calibration read. The default `--fit-method affine` preserves the
  legacy linear correction, while `--fit-method isotonic` stores a monotone empirical mapping for
  non-linear post-hoc calibration experiments. Isotonic can overfit small fit sets, so treat held-out
  evaluation and gates as required for any useful read. `value-calibration-compare` now fits affine
  and isotonic transforms on fit data, evaluates raw/affine/isotonic on held-out data, and records the
  selected method under an explicit metric so WS-E experiments do not rely on ad-hoc manual
  comparisons. Its default selection metric is Pearson correlation because search needs value
  ranking; calibration-error metrics remain available but are reported with warnings because they can
  prefer collapsed transforms. It also supports opt-in quality gates such as
  `--min-sign-accuracy`, `--max-expected-calibration-error`, and `--min-pearson-correlation` so
  experiment scripts can fail a weak value-head read without hardcoding project-wide readiness
  thresholds. Value-head consumers such as search leaf scoring read the stored transform when
  evaluating checkpoint values.
- **Value-head fine-tuning path** (`neural_cli train --initial-checkpoint --objective value-only
  --freeze-non-value-parameters`): warm-starts from an existing checkpoint, trains only the value head,
  and leaves the policy prior intact while calibration is improved. `--value-selection-data` can score
  each standalone-train epoch on held-out calibration and restore the best epoch by
  MAE/MSE/ECE/sign/bias/correlation metric. The correlation option is affine-invariant linear
  association, not calibration by itself, and is intended for workflows that separately check or fit
  calibration transforms. The training path now also has an opt-in pairwise value-ranking loss
  (`--value-ranking-loss-weight`, `--value-ranking-margin`) so WS-E experiments can optimize a global
  return-ordering proxy, not only value magnitude/calibration.
  `neural_cli iterate --value-selection` applies value-based epoch selection
  inside self-play iterations, writes a per-iteration sidecar, and stores the selected epoch in the
  run manifest. By default it selects on iteration/history training rollouts; for a
  cleaner value-calibration read, `--value-selection-heldout-games` collects separate current-policy
  self-play games that are used only for epoch selection and are not added to the training history.
  `neural_cli foundation-value-tune-plan/run/report` now wraps the standalone value-only path for a
  selected foundation candidate (`latest-accepted` by default, or `best-max-damage`) so WS-E
  experiments can fine-tune the value head of the retained base-policy checkpoint without manually
  reconstructing rollout/checkpoint paths. The wrapper can also take explicit `--calibration-data`
  paths so final calibration reporting can use a third split instead of the epoch-selection set.
  A first local value-only fine-tune on the anti-aggression foundation candidate then ran at
  `runs/foundation-anti-aggression-local-20260627/value-tune-pearson-001`: selected candidate
  iteration 3, `--epochs 3`, `--require-heldout-selection`, Pearson epoch selection, and the
  held-out value-selection rollout paths from the foundation run. It completed in 225.5 seconds and
  selected epoch 3. On that selection/calibration set, Pearson improved from the candidate's
  foundation read `0.2542` to `0.3617`, ECE improved from `0.3233` to `0.1160`, and sign accuracy
  improved from `0.6296` to `0.6532`; MAE worsened from `0.8133` to `0.8578`. Treat this as positive
  WS-E signal for value-only fine-tuning, not a final unbiased value-readiness verdict, because the
  reported calibration is on the same held-out paths used for epoch selection.
  A separate 32-game independent calibration sample then checked that value-tuned checkpoint against
  fresh self-play rollouts at
  `runs/foundation-anti-aggression-local-20260627/value-tune-pearson-001/independent-calibration-rollouts.jsonl`.
  On the same 4,470 independent examples, the original anti-aggression checkpoint measured
  Pearson `0.1161`, sign `0.5060`, ECE `0.5491`, and MAE `0.9821`; the value-tuned checkpoint
  measured Pearson `0.1235`, sign `0.5329`, ECE `0.1715`, and MAE `0.9788`. This is positive
  out-of-sample evidence for calibration magnitude and a small sign/ranking improvement, but the
  independent Pearson remains far too weak to treat the value head as search-ready.
  A follow-up value-only tune then enabled the new global pairwise ranking loss with
  `--value-ranking-loss-weight 0.25` and used that same independent 32-game split as final
  calibration data. It trained successfully in 237.8 seconds and slightly improved selection-set
  Pearson (`0.3617` -> `0.3631`), but independent ranking was effectively unchanged: Pearson
  `0.1238`, sign `0.5336`, ECE `0.1684`, and MAE `0.9786` over the same 4,470 examples. Treat this
  as plumbing evidence and a tiny positive read for the ranking-loss lever, not a value-head
  breakthrough; the value/base-net bottleneck remains before root-PUCT micro-tuning should resume.
  A smoke-scale run at `runs/value-head-wse-local-20260627/heldout-selection-ppo-smoke` verified the
  held-out iterate path end-to-end with 2 PPO iterations, 8 training games + 4 held-out selection
  games per iteration, current-vs-current mirror collection, and eval-only `max-damage`. It is
  plumbing evidence, not strength evidence: both candidates failed the incumbent head-to-head
  advancement check, both scored 2/8 versus `max-damage`, and the latest accepted checkpoint remained
  the starting value-selected teacher-BC checkpoint. Held-out ECE selected epoch 2 in both iterations
  (0.099 then 0.628), while iteration training-rollout calibration ECE was 0.091 then 0.180.
  `neural_cli iterate --collector-advancement-mode always` now provides an explicit exploratory
  mode for PPO proof-of-life runs where every saved candidate becomes the next rollout collector
  even if it fails the incumbent benchmark. This is useful for testing whether the loop can produce
  an actual arms-race progression, but it is not promotion evidence and should stay separate from
  accepted-checkpoint gating.
  A follow-up smoke at `runs/value-head-wse-local-20260627/always-advance-ppo-smoke` verified that
  progression path: iteration 2 collected from iteration 1's checkpoint instead of the original
  value-selected teacher-BC checkpoint, while `latest_accepted_checkpoint` correctly remained the
  original starting checkpoint. This is still smoke-scale evidence only. Iteration 1 failed the
  incumbent benchmark at 3/8 and scored 1/8 vs `max-damage`; iteration 2 tied iteration 1 at 4/8
  and scored 3/8 vs `max-damage`, with zero capped games across both 32-game benchmark bundles.
  Held-out value-selection ECE remained noisy (0.922 then 0.700), and iteration calibration stayed
  weak enough that the value/base-net bottleneck remains the near-term focus before larger MCTS
  reads.
  A post-hoc affine calibration smoke then fit `transformer-teacher-bc-split-value-selected.pt` on
  its split training rollouts and evaluated on the held-out split. The calibrated copy used
  `scale=1.814966`, `bias=0.004515`, and moved held-out MAE from 0.9584 to 0.9246 and MSE from
  0.9446 to 0.9403, but sign accuracy barely moved (0.5920 to 0.5936) and ECE worsened from
  0.0484 to 0.0879. This validates the applied-transform plumbing and may help value magnitude, but
  it is not enough to make the current value head search-ready; the WS-E/WS-A bottleneck remains.
  A targeted current-schema strength read confirmed the base-net gap: the same
  `transformer-teacher-bc-split-value-selected.pt` checkpoint scored 150/600 (0.250) against
  `max-damage` over a 300-game-per-orientation benchmark, with 3 ties and zero capped games
  (`runs/max-damage-goal-local-20260627/current-schema-value-selected-vs-max-damage-300.json`).
  This is a valid current-observation-schema checkpoint but not a useful M0 search substrate; the
  next WS-A task is to produce a stronger current-schema PPO/base-net checkpoint.
  A medium current-schema PPO run then tested that directly:
  `runs/current-schema-ppo-local-20260627/aggressive-always-advance-256x3` ran 3 iterations from the
  value-selected checkpoint with 256 games/iteration, mirror self-play, `aggressive-damage` in the
  training pool, `max-damage` eval-only, collection temperature 1.4, entropy 0.01, and always-advance
  collector progression. Each candidate beat its incumbent head-to-head (0.535, 0.530, 0.530), and
  PPO diagnostics were mechanically healthy (`ppo_valid_fraction=1.0`, clip fraction
  0.145-0.187, entropy rising 1.01 -> 1.27), but the external yardstick stayed flat:
  54/200, 55/200, then 50/200 versus `max-damage`, with zero capped games. This is useful negative
  evidence: current-schema PPO is able to move internally, but this short aggressive-curriculum run
  still did not produce a stronger base net for M0.
  A first controlled temporal-aggregator probe then trained the new GRU pooling path on the same
  current-schema teacher split and value-selected on the same held-out rollouts. It did **not**
  improve the base-net/value bottleneck: held-out value-selected ECE was 0.0741 versus the mean
  model's 0.0484, MAE was 0.9875 versus 0.9584, sign accuracy was 0.570 versus 0.592, and the
  selected GRU checkpoint scored 142/600 (0.237) against `max-damage` versus the mean checkpoint's
  150/600 (0.250), again with zero capped games
  (`runs/max-damage-goal-local-20260627/current-schema-gru-value-selected-vs-max-damage-300.json`).
  This validates the GRU path as runnable but is negative evidence for "GRU alone fixes the current
  teacher-data base net"; temporal memory should next be tested inside a genuine self-play setup,
  not by further polishing this same small BC split.
  That follow-up self-play test then ran the same 3x256 aggressive always-advance PPO shape as the
  mean-pooled run, warm-starting from the GRU value-selected checkpoint and using the same seed bands.
  The result was a small/noisy uptick but still no useful base-net breakthrough: GRU scored
  52/200, 57/200, and 57/200 versus `max-damage` across iterations (166/600 total, 0.277), compared
  with the prior mean run's 54/200, 55/200, and 50/200 (159/600 total, 0.265), with zero capped games
  in both runs. PPO health looked similar (`ppo_valid_fraction=1.0`, clip fraction 0.147-0.183,
  entropy 0.99 -> 1.26), so temporal memory remains a plausible lever but is not enough by itself;
  the next WS-A work should focus on stronger training signal/opponent-pool pressure rather than
  treating GRU as the bottleneck fix.
  A generic mixed-pressure pool test then added `scripted-teacher` alongside `aggressive-damage`
  while keeping mirror self-play, always-advance PPO, and the same 3x256 shape. That was worse, not
  better: the mixed pool scored 47/200, 49/200, and 43/200 versus `max-damage` (139/600 total,
  0.232) versus the aggressive-only mean run's 159/600 (0.265), with zero capped games. This narrows
  the WS-A opponent-pool diagnosis: generic fixed-opponent diversity is not enough. The next pool
  lever should be a targeted anti-aggression/counterplay curriculum or richer self-play signal that
  directly rewards exploiting predictable nukes, not simply adding the scripted teacher to the pool.
  The return-target path now has opt-in, clipped shaping knobs for visible HP-differential
  deltas, faint-differential deltas, and late-turn penalties. Defaults remain terminal-only for
  comparability; shaped targets are a WS-A/WS-E experiment lever, not yet evidence of strength.
  A controlled same-seed shaping ablation then tested conservative weights
  (`--hp-delta-return-weight 0.5`, `--faint-delta-return-weight 0.5`,
  `--turn-penalty-after 180`, `--turn-penalty 0.005`) on the same 3x256 aggressive
  always-advance PPO shape and seed bands as the mean/GRU runs. It scored 50/200, 49/200,
  and 55/200 versus `max-damage` (154/600 total, 0.257), with zero capped games. The
  unshaped mean run on the same seed bands scored 159/600, and the GRU run scored 166/600.
  A second shaped run on fresh seed bands scored 156/600. This is negative evidence for
  these shaping weights as a standalone base-net fix; shaping remains a tunable lever, not
  the current bottleneck answer.
- **Entity-token transformer policy+value net** (`neural_policy.py`) — richer than the thesis's
  3-layer MLP; already has policy, value, and opponent-action heads. New configs bound value outputs
  with `tanh`; legacy checkpoints remain loadable with linear value outputs. The trunk now also has
  an opt-in recurrent temporal aggregator (`--temporal-aggregator gru`) for base-net/value-head
  experiments that need explicit turn-sequence state instead of masked-mean history pooling.
- **Public belief engine** (`belief.py`) — narrows the opponent's hidden set from observable facts;
  a better basis for determinization than ad-hoc set sampling. It now exposes bounded
  player-relative opponent determinizations for search, preserving unknowns instead of inventing
  hidden facts.
- **Self-play iterate loop** (`neural_cli iterate` → `neural_selfplay.py`, `selfplay.py`,
  `collection.py`) — collect → train → benchmark, with promotion gates (`evaluation.py`).
  PPO training now records objective-health diagnostics in epoch metrics and TensorBoard:
  valid behavior-probability coverage, raw value-baselined advantage mean/std, probability-ratio
  mean, clip fraction, and policy entropy in nats. These are tuning/audit signals for WS-A, not policy
  strength metrics.
  The neural path now supports compact training-cache chunks as an alternative to raw
  `training-rollouts.jsonl` materialization, with a default 50 GiB active-cache ceiling, optional
  raw rollout omission, post-train chunk deletion, and the same controls exposed through
  `neural foundation-plan/run`. This makes mid-scale foundation runs practical without changing the
  PPO contract or writing deployment-specific assumptions into the public repo.
  Multi-worker self-play collection keeps deterministic record ordering while bounding in-flight
  game futures, so large recipe chunks do not queue the entire game range in memory before writing.
  The foundation wrapper also exposes the nested neural architecture knobs, including `--layers 0`
  for the CPU-fast pooled encoder path, so training-throughput probes can be launched from the same
  audited foundation recipe interface instead of bespoke `neural iterate` commands.
  Standalone `neural train` can write a compact train summary artifact with end-to-end elapsed time,
  train-call elapsed time, input data size, checkpoint size, final metrics, and cache lifecycle bytes;
  use that artifact rather than external logs when judging the 20k-games-under-10-minutes training gate.
  In-process neural self-play iteration manifests also record training elapsed time, training input
  bytes, checkpoint bytes, and cache deletion bytes so foundation runs can be judged without scraping
  process logs.
  The foundation wrapper also has a `midscale` profile (`5 × 10,000` games) for the 50k
  recipe-faithful rising-curve gate before any full multimillion-battle spend.
- **Benchmark harness** (`collection.benchmark_rollouts`, `neural_cli benchmark`) — vs
  random/simple/max-damage baselines.
- **Neural self-play report yardstick curves** (`neural_cli report`) — text reports include
  fixed-opponent benchmark curves such as `max-damage`, `random-legal`, and `simple-legal`, with
  per-iteration win rate, games, and capped-game counts. This makes WS-F strength reads visible
  without relying on blended benchmark win rate or TensorBoard-only inspection. Rolling incumbent
  head-to-heads stay out of this yardstick block because they are advancement evidence, not
  fixed-strength references. The same report now prints a conservative `foundation_readiness` block:
  value calibration present/missing, latest-iteration `max-damage` yardstick sample size, and a
  `foundation_evidence_status` that says whether these prerequisites are present/sample-sized. This
  is not a value-quality or strength verdict; it is a guardrail against over-reading search deltas when
  the value/base-net/eval evidence is incomplete.
  `neural_cli foundation-compare` compares multiple foundation-run wrapper summaries side by side,
  including latest fixed-yardstick rates, the best observed `max-damage` yardstick checkpoint, and
  value calibration signals, so WS-A/WS-E arms can be selected from comparable artifacts before
  spending more time on search tuning. It also supports caller-supplied quality thresholds for
  experiment-local gating. By default comparison gates use latest rows for continuity, while
  alternative candidate sources are explicit model-selection modes rather than implicit relabeling.
  Best-row selection prefers sample-sized `max-damage` rows when any exist and labels below-milestone
  rows explicitly when only smoke-sized evidence is available.
  For yardstick-gated runs where the latest saved checkpoint may be rejected, `foundation-compare`
  can switch the comparison/gate candidate with `--candidate-source latest-accepted` or
  `--candidate-source best-max-damage` so model-selection reads do not accidentally judge a
  non-retained latest checkpoint.
- **Online ladder client** (`online_client.py`) — can play a checkpoint against the live server.
- **Raw-facts observation** with Markov-restoring encodings (turn count, future-sight, toxic stage,
  screens) already present (`showdown.py`).

## Where we stand / why we plateau

~20 prior training-method variants converged to ~0.46–0.52 vs max-damage (the imitation ceiling).
Pure searchless self-play risks settling into a mediocre local equilibrium. The fixes — a history
pool, exploration pressure, and (above all) a search improvement operator — are exactly the recipe
above, and are all knowledge-free / on-mission.

The latest teacher-cut evidence is a wiring milestone, **not** a strength signal. A clean loop that
allows a learned checkpoint as one-shot initialization and then removes teacher relabeling, fixed
teacher opponents, and imitation terms ran successfully and reached `0.2825` versus `max-damage` over
a 400-game yardstick row. It is tempting to read that as "self-play cannot rise beyond the teacher
ceiling," but that conclusion is unsupported: the run was **3 iterations × 256 games = 768 battles
(~0.026% of the recipe's ~3M-battle budget) with materially off-recipe hyperparameters** — zero
exploration entropy, a single epoch, and pre-annealing constant LR (see
[`mit_thesis_reference_config.md`](mit_thesis_reference_config.md)). The thesis net needed ~1 day /
40M steps just to reach 80% vs a heuristic baseline. So this run validates the teacher-cut *plumbing*
and gives an early-training datapoint; it says nothing about whether the loop can clear the ceiling.
The real wall is upstream of search and upstream of "does self-play work": **we have not yet run the
training half at recipe fidelity or anywhere near recipe scale.**

Near-term priority order:

1. **Recipe-fidelity pass on the PPO config.** Align the teacher-cut/foundation training knobs to the
   thesis reference (`entropy_coef`≈0.0588, `n_epochs`≈7, annealed LR, `gamma`≈0.9999, `gae_lambda`,
   clip ranges, value coef, batch size) — see the gap table in the reference doc. Several of these
   (exploration entropy, epochs, LR annealing) were first-order gaps in the earlier teacher-cut pilot.
   **Status:** the config-fidelity half now exists as `neural iterate --experiment-preset
   recipe-fidelity` (and `neural foundation-plan/run --recipe-fidelity`, usable with the teacher-cut
   variant). It bundles the expressible Table A.3 knobs (entropy 0.0588, 7 epochs, gamma 0.9999, GAE
   lambda 0.754, clip 0.0829, value coef 0.4375, new `--max-grad-norm` 0.5430, batch 1024, base LR
   5.9e-5, MIT thesis LR annealing over a configurable game-progress denominator, standard collection
   temperature), records them in the manifest/run summary, and a
   `recipe_fidelity` audit (`neural report`, foundation summaries) verifies a run is actually
   on-recipe rather than just named so. **Missing recipe components (from the paper — not yet
   implemented):**
   - **Value-function clipping** (`clip_range_vf`) — second-order; close opportunistically.

   The remaining unsupported knob is surfaced in the audit's `unsupported_knobs` so a run is never
   silently mis-labelled on-recipe. For full-budget runs, the default denominator should remain the
   recipe-scale budget; for cheap 50k-100k midscale reads, set `--learning-rate-schedule-total-games`
   to the read's own total game count so the annealing curve exercises an `x: 0 -> 1` sweep instead
   of barely moving over the first few percent of a 3M-game schedule. Config fidelity is otherwise
   independent of scale (item 2).
2. **Scale the training half toward the recipe budget.** Drive battles from ~10³ toward ~10⁶, tracking
   a net-alone strength curve against a stable smooth baseline (a SimpleHeuristics-style bot), the way
   the thesis validated every 20k steps. This is precisely where distributed collection (WS-B) becomes
   on-recipe and justified — the thesis used 80 CPU workers; our fleet is the equivalent, not premature
   optimization.
   **Status:** the collection engine is proven at **~200 games/s** — ~4× the ~46 g/s needed for 4M
   battles/day — so the recipe's ~3–4M-battle budget is now a matter of **hours, not days**, and a
   mid-scale read is **minutes**. Remaining WS-B work is closing the full collect→train→promote loop
   (the central trainer must keep pace, or it becomes the new bottleneck) plus the single-box
   equivalence test.
   A first 20k-game fast PPO/cache checkpoint is not a candidate to scale: job logs reported that
   the 20k compact-cache training job consumed about 21 GiB of active cache, deleted the cache after
   training, saved a sub-megabyte checkpoint, and completed the one-epoch CPU-fast train in about
   nine minutes. Its
   300-game-per-orientation read versus `max-damage` scored **11/600** (`1.83%`), with one capped
   game and zero ties. Treat that as negative evidence for the specific fast 20k setting and a
   reminder that throughput/storage plumbing is solved before learning quality is solved; it is not
   evidence against a recipe-faithful mid-scale run because a single fast 20k-game setting is neither
   recipe-faithful nor a rising net-alone curve experiment.
   **Guardrail — do NOT spend the full multimillion budget yet.** Gate the recipe-scale run on a
   **cheap mid-scale recipe-faithful run (~50–100k battles — minutes at current throughput) whose
   net-alone curve actually *rises*** vs the smooth baseline. A flat recipe-faithful mid-scale curve
   means stop and fix the recipe, **not** buy more battles. Throughput is no longer the constraint;
   recipe fidelity and a confirmed rising curve are. Use `neural foundation-run --profile midscale
   --recipe-fidelity` as the standard 50k public wrapper shape for that gate.
3. **Only judge "can self-play clear the ceiling" after (1)+(2).** Treat sub-300-game rows and any
   pre-fidelity/pre-scale run as wiring checks, not strength evidence; 300+ games is the default floor.
4. **Phase 2 — value head + inference-time MCTS.** Improve value-head ranking/calibration to a concrete
   bar, then add test-time MCTS (determinization via Showdown's randbats generator) as a topper on an
   already-strong net. **These are the gate for the ladder *endgame* (net+MCTS), not a precondition for
   the training run:** in the recipe, training is pure PPO self-play with no search in the loop, and the
   value head co-trains as the PPO critic, so neither needs to be *solved* before the multimillion run.
   They should, however, be **probed cheaply in parallel** on the mid-scale checkpoint from item 2
   (value-head calibration read + a small test-time-MCTS lift probe) to build confidence that search
   will lift *before* the full training spend. `poke-engine` assessment feeds both phases: it could cut
   CPU battle/branch cost enough to make recipe-scale training and MCTS rollouts affordable — gated on
   proving Gen 3 randbats equivalence first.

## Strategy hypothesis & go/no-go gates

**Core hypothesis (unverified):** prior self-play stalled at ~0.52 because every variant was
teacher-anchored *and* none ran at recipe fidelity or scale. The recipe asserts the training-half
levers — (a) enough **scale** (~3M battles), (b) **exploration pressure** (non-zero entropy), (c)
**recipe-faithful PPO** (annealed LR, 7 epochs, tuned clip/discount), and optionally (d) a real
opponent *league* — produce a strong **net on their own**, with test-time MCTS as a topper. Gen4→gen3
transfer is also assumed. Treat this as a hypothesis to test, not a given.

**The training half is the load-bearing bet; search is a topper** (this corrects an earlier framing
that treated test-time search as the primary plateau-breaker). The thesis's *net-alone* already
reached ~80% vs a heuristic baseline and beat that heuristic head-to-head; MCTS then added a
meaningful-but-modest lift on top. So the dominant risk is **not** "search won't lift a weak net" —
it is "we have not run recipe-faithful PPO self-play at anywhere near recipe scale." Do not gate the
scale/fidelity work behind a local search-lift demo, and do not let test-time search become a variant
treadmill: recent micro-probes confirmed that tuning search on a weak teacher-BC prior with a poorly
calibrated value head produces unreadable 8-16 game deltas.

Keep the root-PUCT/determinization machinery for phase 2, but the near-term substrate work is the
recipe-fidelity + scale pass on training, tracked by a net-alone strength curve. Resume search reads
only once there is a meaningfully strong base net and a sample size that can support a decision.

**Go/no-go gates:**
- **M0 — training-half gate (primary).** Recipe-faithful PPO self-play (config aligned to the
  reference table) must show a *net-alone* strength curve that **rises** as battles scale, tracked
  against a stable smooth baseline (SimpleHeuristics-style), the way the thesis validated every 20k
  steps. This is the load-bearing gate and what justifies fleet compute — **not** a local search
  demo. Falsification requires a recipe-faithful run at meaningful scale (≥ tens of thousands of
  battles) that stays flat, not a 768-battle off-recipe pilot. Use a current-observation-schema
  checkpoint; older checkpoints can fail on numeric-feature shape mismatches and should be retrained
  or skipped.
- **M0b — search-lift gate (phase 2, after M0).** On a meaningfully strong net, net+MCTS must add
  lift (target ~0.60+ vs max-damage, well past the 0.52 plateau) at a 300+ game sample. This does
  **not** gate the scale/fidelity work above. The first 64-game-BC probe showed no lift and the
  value-calibration split showed the current value head is not yet a reliable leaf evaluator — so
  treat 8-16 game root-PUCT runs as plumbing checks only, and defer the real read until the value
  head clears its bar and the base net is strong.
- **M1 gate:** the per-iteration strength curve must *rise* over ≥10 league iterations; a multi-
  iteration flatline = stuck → revisit scale, exploration, league diversity, and (phase 2) search.

---

## Workstreams (parallelizable)

Each workstream lists scope, concrete steps, deliverables, acceptance criteria, and dependencies.
WS-A, WS-C, WS-E, WS-F can start in parallel today; the first milestone (M0) proves search on a
*modest* net before WS-B commits fleet compute to scaling. WS-D depends on WS-C + WS-E + a decent
(not great) net from WS-A.

### WS-A — Self-play PPO training loop (the RL engine)
**Owner goal:** a robust PPO self-play loop that reliably *climbs* on a fixed strength yardstick,
with anti-stagnation machinery.

Steps:
1. Audit/solidify the PPO path in `neural_cli iterate` / `neural_selfplay.py`: advantage estimation,
   value-head loss weighting, entropy bonus, capped-game return, gradient/clip settings.
   PPO updates now train on the current iteration's rollout shard only, while retaining cumulative
   rollout paths in the manifest for provenance. This preserves the roadmap's on-policy consistency
   requirement; supervised/RWR objectives may still use cumulative history.
   A same-seed 3x256 rerun of the aggressive always-advance PPO shape with this corrected
   iteration-local PPO input path scored 54/200, 49/200, and 57/200 versus `max-damage`
   (160/600 total, 0.267), with zero benchmark capped games and one capped collection game.
   The prior cumulative-stale PPO run on the same seed bands scored 159/600, so the fix improves
   training validity but does not by itself lift the short-run base-net ceiling.
   Rollout collection now records raw behavior-policy value estimates, and PPO training has an
   opt-in `--ppo-target-mode gae` / `--gae-lambda` path that derives advantages and value targets
   from those recorded baselines when present while falling back to legacy discounted returns for
   fixed-opponent or old-rollout rows. This is target-quality plumbing, not yet strength evidence.
   The current CPU proof-of-concept recipe is captured as `neural iterate --experiment-preset
   foundation-arms-race`: PPO + GAE, mirror collection, always-advance collector progression,
   spread historical opponents, higher collection temperature, held-out Pearson value selection,
   value calibration, and a fixed `max-damage` benchmark reference. The preset is a reproducibility
   convenience for WS-A/WS-E foundation runs, not promotion evidence.
   `neural foundation-plan/run/report` wraps this preset with smoke/pilot profile defaults and a
   compact summary artifact so CPU foundation attempts can be launched and audited consistently.
   A second preset, `neural iterate --experiment-preset recipe-fidelity` (and the foundation
   wrapper's `--recipe-fidelity` flag), reuses that same arms-race scaffolding but overrides the PPO
   hyperparameters to the MIT thesis Table A.3 values (entropy 0.0588, 7 epochs, gamma 0.9999, GAE
   lambda 0.754, clip 0.0829, value coef 0.4375, new `--max-grad-norm` 0.5430, batch 1024, base LR
   5.9e-5, MIT thesis LR annealing over a configurable game-progress denominator, standard
   collection temperature 1.0). It is the config-fidelity half of near-term
   priority #1. A `recipe_fidelity` audit (printed by `neural report`, embedded in foundation run
   summaries, and computed by `recipe_fidelity_audit()`) compares the *actual* resolved config
   against the reference table so a run is verifiable as recipe-fidelity, and flags the remaining
   unexpressed value-function clipping knob under `unsupported_knobs`.
   The highest-priority WS-A question is now explicit: can self-play break the scripted-teacher
   ceiling at all? The `teacher-cut` foundation variant is the clean experiment contract for that
   question. It permits a one-shot learned checkpoint as the initial collector, then removes fixed
   teacher/heuristic training opponents, relies on mirror/history self-play, keeps the reward signal
   to game outcome only, and uses `max-damage` as an eval-only yardstick. Do not read sub-300-game
   rows as strength evidence; use them only as wiring checks. If no learned initial checkpoint is
   supplied, the variant is a random-legal cold start rather than a teacher-bootstrap run.
   A real local teacher-cut pilot then ran from the compatible current-schema teacher-BC/value
   seed at `runs/foundation-teacher-cut-local-20260627/pilot-001`. The run completed three
   iterations of 256 self-play games each, with no fixed training opponents, zero capped collection
   games, and sample-sized 400-game `max-damage` rows. It did not break the teacher ceiling:
   `max-damage` win rates were 0.2275, 0.2775, then 0.2825. The latest checkpoint remained strong
   against weak baselines (`random-legal=0.9375`, `simple-legal=0.7925`), and value calibration was
   present over 10,102 examples (sign accuracy 0.6446, ECE 0.1245, Pearson 0.2985), but this is
   still far below the roughly 0.57 scripted-teacher bar. Treat this as useful negative WS-A
   evidence: teacher-cut PPO is wired and can climb slightly, but the current base-net/update
   recipe is not yet escaping the imitation ceiling.
   A value-only tune of that teacher-cut iteration-3 checkpoint then used the three iteration
   training shards, selected epochs by Pearson on the held-out value-selection shards, and reported
   final calibration on a fresh 32-game independent mirror split at
   `runs/foundation-teacher-cut-local-20260627/value-tune-best-max-pearson-001/independent-calibration-rollouts.jsonl`.
   It did not materially improve the value head: on 2,749 independent examples, the original
   checkpoint measured Pearson 0.1679, sign 0.5300, ECE 0.1201, and MAE 0.9744; the value-tuned
   checkpoint measured Pearson 0.1694, sign 0.5278, ECE 0.1437, and MAE 0.9739. Treat this as
   another negative WS-E read for simple value-only fine-tuning on the current teacher-cut data.
   The wrapper also supports `--variant opponent-signal` for the H3 ablation path: it keeps the
   same foundation recipe but raises opponent-action auxiliary supervision, so the result can be
   compared against the baseline wrapper before spending more effort on search tuning.
   A real local `foundation-run --profile pilot` then validated the wrapper and cold-start recipe
   end-to-end at `runs/foundation-pilot-local-20260627/pilot-001`: 3 iterations, 256 games per
   iteration, mirror collection, always-advance PPO, held-out Pearson value selection, and 400-game
   fixed-yardstick rows. The run produced sample-sized foundation evidence but no base-net lift:
   `max-damage` win rates were 0.045, 0.052, then 0.025; `simple-legal` fell from 0.330 to 0.205;
   and `random-legal` ended at 0.527. Latest value calibration was present over 19,269 examples
   with sign accuracy 0.6168, ECE 0.1252, and Pearson 0.3170. This proves the current CPU foundation
   loop is runnable and auditable, but a cold random-legal start is not producing a useful base net;
   the next WS-A/WS-E experiment should improve the learning signal or initialization before any
   further search-tuning reads.
   A follow-up `value-calibration-compare` pass on the same pilot's held-out
   `value-selection-training-rollouts.jsonl` confirmed that post-hoc calibration is not a hidden fix
   for this checkpoint. Iteration 1 selected isotonic, but its held-out Pearson remained negative
   (`raw=-0.1534`, `affine=-0.1534`, `isotonic=-0.1277`) and the command emitted the weak-ranking
   warning. Iteration 2 selected raw by Pearson (`raw=0.2668`, `affine=0.2668`,
   `isotonic=0.2370`), even though affine improved MAE/ECE. Iteration 3 also selected raw
   (`raw=0.2619`, `affine=0.2615`, `isotonic=0.2509`). The read is useful because it separates
   calibration-error improvements from search-relevant ranking: for the latest pilot checkpoint,
   affine/isotonic transforms should not be treated as making the value head more useful for MCTS.
   The H3 `opponent-signal` variant then ran the same local 3x256 pilot shape at
   `runs/foundation-opponent-signal-local-20260627/pilot-001`, with
   `--opponent-action-loss-weight 1.0`. This produced a better but still weak fixed-yardstick read:
   `max-damage` win rates were 0.022, 0.110, then 0.062; `simple-legal` rose to 0.403 by iteration
   3, and `random-legal` ended at 0.603. Latest foundation-readiness value calibration was present
   over 19,670 examples with sign accuracy 0.5264, ECE 0.1975, and Pearson 0.2318. A held-out
   `value-calibration-compare` on iteration 3 selected isotonic and improved Pearson from raw
   0.3270 to 0.3581, with MAE improving from 0.9484 to 0.9186. This is real positive evidence for
   the H3/base-net direction versus the cold baseline, but it is not yet a search-ready value head:
   max-damage remains far below the M0 target and value ranking is still modest.
   A resumed H3 continuation then added three more iterations to the same run after fixing neural
   resume benchmark seed advancement. The continuation used fresh benchmark seed bands
   `1000600`, `1000800`, and `1001000`. It did not produce sustained base-net improvement:
   iteration 4 briefly looked healthier (`max-damage=0.100`, `simple-legal=0.412`,
   `random-legal=0.685`, value sign 0.5985, ECE 0.0942, Pearson 0.2784), but iterations 5 and 6
   regressed (`max-damage=0.022` then `0.018`, `simple-legal=0.212` then `0.152`,
   `random-legal=0.432` then `0.370`). The latest iteration 6 value read was present and
   sample-sized over 21,728 examples (sign 0.5986, ECE 0.1073, Pearson 0.2130), but the latest
   fixed-yardstick strength is worse than iteration 3 and fails the local foundation quality gate
   used for this follow-up because `max-damage` is below 0.05. Treat this as negative evidence for
   simply continuing the same H3 run; the next WS-A lever should change the training signal,
   initialization, or selection/retention policy rather than expecting more identical iterations to
   climb.
   The foundation wrapper now exposes `temporal-gru` and `opponent-signal-gru` variants so the
   optional GRU temporal aggregator can be evaluated as the next CPU WS-E/WS-A foundation lever
   before returning to larger root-PUCT reads.
   A local `temporal-gru` smoke passed end-to-end, then a `--profile pilot` run at
   `runs/foundation-gru-local-20260627/pilot-001` produced sample-sized evidence for the isolated
   GRU temporal-aggregation lever. It showed a stronger early fixed-yardstick spike than the prior
   H3 run (`max-damage=0.142` at iteration 1 versus H3's best `0.110`) but did not sustain it:
   iterations 2 and 3 fell to `0.105` and `0.030`, while `simple-legal` fell from `0.540` to
   `0.170` and `random-legal` from `0.767` to `0.417`. Latest value calibration was present over
   19,344 examples with sign accuracy 0.6271, ECE 0.0691, and Pearson 0.1075. Treat this as useful
   evidence that the GRU arm can find a stronger early checkpoint, but the always-advance foundation
   loop still forgets/regresses; the next WS-A lever should focus on retention/advancement or
   training-signal changes rather than simply continuing the same GRU run.
   The neural foundation wrapper now supports `--collector-advancement-mode yardstick-gate` for this
   retention follow-up. This mode keeps the fixed `max-damage` benchmark as a cheap collector
   retention yardstick: the first candidate initializes the baseline, and later candidates become the
   next rollout collector only when they beat the best accepted `max-damage` win rate so far. This is
   not a final promotion gate, but it directly targets the GRU/H3 always-advance regression pattern.
   The first local yardstick-gated GRU pilot at
   `runs/foundation-gru-yardstick-local-20260627/pilot-001` passed in 1,697 seconds and confirmed
   the retention mechanism: iteration 1 initialized the collector baseline at `max-damage=0.085`,
   iteration 2 improved to `0.113` and advanced, and iteration 3 regressed to `0.092` and did **not**
   become the next collector. The run's `current_policy` and `latest_accepted_checkpoint` correctly
   remained at iteration 2 even though the latest saved checkpoint is iteration 3. This is positive
   evidence for collector retention, not a base-net breakthrough: best `max-damage` remained only
   `0.113/400`, and the latest value read was weak (`sign=0.4249`, `ECE=0.2819`, `Pearson=0.2302`).
   The next WS-A/WS-E lever should improve value/base-net quality rather than returning to search
   micro-tuning.
   The foundation wrapper now also exposes `anti-aggression` and `anti-aggression-gru` variants to
   make the next targeted opponent-pool experiment reproducible: they add `aggressive-damage` to the
   fixed self-play opponent pool while preserving `max-damage` as eval-only, with the `-gru` arm
   combining that pressure with temporal aggregation. This is curriculum plumbing for the targeted
   anti-aggression/counterplay thread below.
   A local `anti-aggression` pilot with yardstick-gated collector advancement then produced the
   strongest fixed-yardstick foundation read so far:
   `runs/foundation-anti-aggression-local-20260627/pilot-001` ran 3 iterations, 256 games per
   iteration, `aggressive-damage` in the training opponent pool, `max-damage` eval-only, and
   400-game fixed-yardstick rows. Iteration 1 initialized the collector baseline at
   `max-damage=0.128`; iteration 2 regressed to `0.043` and did not advance; iteration 3 recovered
   and advanced at `0.180`, with `simple-legal=0.570` and `random-legal=0.900`. This is positive
   evidence for targeted anti-aggression pressure plus yardstick retention as the best current
   base-policy direction. It is not a search-ready foundation: the latest value read still has weak
   ranking/calibration (`sign=0.6296`, `ECE=0.3233`, `Pearson=0.2542`), so the next critical work
   remains value/base-net quality rather than more low-sample search-operator tuning.
2. **History/league opponent pool — diversity, not just recency:** sample opponents from a bounded
   set of *past* checkpoints (not just the latest) to kill non-transitive cycling and forgetting.
   Crucially, guard pool *diversity*: a pool of near-identical aggression-exploiters (the failure
   mode we already hit) induces no learning pressure. Add a behavioral-diversity check and/or a
   dedicated exploiter agent folded back into the pool. Wire through the existing promotion registry
   / historical-opponent plumbing.
   Neural self-play now has an opt-in `--historical-opponent-selection spread` mode that chooses
   promoted/history opponents across the available checkpoint range instead of only taking the
   most recent entries. The current spread policy keeps both the oldest and newest selectable
   checkpoints when capacity allows. This is a first pool-pressure lever; behavioral diversity
   scoring and learned/exploiter pool management remain open.
3. **Exploration pressure:** expose and tune entropy coefficient + collection temperature; ensure
   collection samples (not greedy) so the policy keeps exploring.
4. **Fixed-yardstick eval every iteration** (see WS-F) and persist the strength curve.
5. Remove imitation as a *crutch*: support cold self-play from a weak/random init as the on-mission
   path; keep the scripted-teacher bootstrap only as an optional warm-start/control, clearly flagged.

Deliverable: `neural_cli iterate` that trains a net via league self-play with exploration and a
per-iteration strength curve.
Acceptance: strength vs the fixed yardstick **rises** across ≥10 iterations (not a flatline);
no degenerate-collapse (capped-game rate bounded).
Touches: `neural_selfplay.py`, `selfplay.py`, `collection.py`, `neural_cli.py`.

### WS-B — Distributed scaling (parallel collection → central train)
**Owner goal:** turn one-box self-play into a CPU fleet hitting the thesis's ~3M-battles budget in
days. Collection is the CPU bottleneck; fan it out.

**Status:** collection throughput is **proven at ~200 games/s** (well above the ~46 g/s needed for
4M battles/day), so the ~3M-battle budget is now ~hours. The remaining WS-B work is closing the full
**collect→train→promote loop with the central trainer keeping pace** (else it becomes the new
bottleneck), plus the single-box **equivalence test** (acceptance below). Note: throughput being
solved does **not** unlock the full multimillion run — that is gated by the WS-A guardrail
(confirm a *rising* mid-scale recipe-faithful net-alone curve) before spending the budget.

Steps:
1. **Collection/train split:** make collectors emit rollout JSONL to shared storage keyed by
   iteration + shard; a central step trains on the aggregated shards and publishes the next
   checkpoint; collectors pick up the new checkpoint. (This is the distributed form of the existing
   collect→train loop; the *code* for sharded collection + aggregation lives in the tracked repo.)
2. **Iteration controller:** a loop that, per iteration, launches N collector shards against the
   current checkpoint, waits, runs the central train, and advances.
3. **On-policy consistency (critical — PPO is on-policy):** use **synchronous iterations with a
   barrier** — every collector shard uses checkpoint N; train N→N+1 only after all shards finish.
   Do **not** mix rollouts collected under different checkpoints into one PPO update; stale rollouts
   degrade PPO. (If we later want asynchronous collection, switch to a staleness-tolerant objective
   — out of scope for v1.)
4. **Data pipeline at scale:** rollout JSONL is ~TB-scale at 3M battles (≈215 MB / 200 games).
   Design shard layout, cross-shard shuffle for training, and a retention policy (keep recent
   iterations, prune old) so storage and train-time I/O stay bounded.
5. **Hardware split:** collection is CPU (the fleet); the **central train step benefits from a GPU**
   (the thesis trained the net on one GPU). Provision the train step accordingly — collection stays
   CPU-only, training is not.
6. **Fleet deployment** (a CPU pod fleet): container image, parallel-collection manifests, shared
   storage, and the iteration controller. All environment/location specifics are deliberately kept
   **out of this (public) repo**.
7. Throughput target: enough parallel CPU to reach ~3M battles in single-digit days.

Deliverable: sharded-collection + central-train code (tracked); the fleet deployment itself is kept
out of this repo.
Acceptance: end-to-end iteration runs across many workers; aggregate games/hour scales ~linearly
with workers; identical rollout records to the single-box path (equivalence test).
Touches: `collection.py`, `neural_selfplay.py`, `rollout_cli.py` (tracked). Deployment manifests are
kept out of this repo.

### WS-C — Battle forking / snapshot-restore (the MCTS enabler)
**Owner goal:** explore alternative lines from a battle position — the prerequisite for MCTS. Pick
the *simplest* mechanism that meets the per-move search budget; do not assume snapshot/restore.

Steps:
0. **Verify how the thesis did rollouts first.** It ran MCTS over the Showdown sim, so branch
   exploration is solved prior art — match its approach (snapshot/restore vs replay-from-root)
   before inventing. This is the single highest-leverage de-risking step in the whole plan.
1. **Prefer replay-from-root if the warm sim makes it cheap enough.** With determinization, each
   rollout re-simulates from the battle's recorded line + a sampled opponent set; warm sim
   (~0.4 ms/turn) may make this fast enough for shallow search and avoids state serialization
   entirely. Validate the per-move cost against a realistic search budget.
2. **Only if replay-from-root is too slow:** build snapshot/restore — investigate `BattleStream`
   serialization (`Battle.toJSON()` / restart-from-state) at our pinned commit, then extend
   `battle_bridge.mjs` (already battle-id-keyed) with `snapshot {battleId}` and
   `fork {fromBattleId,newBattleId,state}`; expose `LocalShowdownEnv.snapshot()/fork()`.
3. Validate either path: explore divergent lines from turn N and confirm each is byte-identical to a
   from-scratch battle that took the same actions (modulo timestamps).
4. Assess `poke-engine` as an optional fast reversible backend. The initial assessment is captured
   in [`poke_engine_assessment.md`](poke_engine_assessment.md): licensing is favorable for
   `poke-engine` itself, the Python binding exposes apply/reverse instructions, and Gen 3 feature
   support exists, but Showdown-equivalence for Gen 3 random battles is unproven and must be tested
   before using it for training/search.

Deliverable: a forking/rollout mechanism (replay-from-root preferred) + equivalence tests.
Acceptance: divergent lines are deterministic and identical to ground-truth replays; per-move rollout
cost fits the search budget.
Touches: `local_showdown.py`, and `scripts/battle_bridge.mjs` only if snapshot/restore is needed.
Risk: this gates all of WS-D — validate the mechanism in days, not weeks. Last-resort fallback is a
learned/in-process model (much larger effort).

### WS-D — Test-time MCTS (the policy-improvement operator)
**Owner goal:** a search-augmented policy that measurably beats the raw net, mirroring the thesis.
Depends on WS-C (forking) + a *decent* net from WS-A (not a great one) + a **well-calibrated value
head** (WS-E). MCTS leaf evaluation is bounded by value quality: a noisy value head makes search
*worse* than the raw policy, so value calibration is a hard prerequisite, not a nicety.

Steps:
1. MCTS skeleton over the forkable sim, guided by the net (PUCT-style: prior from policy head, value
   from value head; back up values).
2. **Determinization:** at the search root, sample the opponent's hidden set from the belief engine
   (`belief.py`) → search a (near) perfect-information instance; average over a few sampled sets.
3. **Chance handling — damage-roll grouping:** collapse damage outcomes to KO / no-KO branches (per
   Foul Play) instead of all 16 rolls; optionally best/worst/avg-case chance aggregation
   (*-minimax-style) as a knob.
4. **Opponent move during search:** start with the thesis's approach (opponent plays the net
   policy); leave a hook for DUCT (decoupled UCB, true simultaneous-move handling) as an upgrade.
5. Search budget / time control; integrate as an alternate `select_action` (net+MCTS).

Deliverable: a net+MCTS policy usable in benchmark and ladder play.
Acceptance: net+MCTS **beats net-alone** by a clear margin on the fixed yardstick and head-to-head.
Touches: new search module (e.g. `search/mcts.py`), `belief.py`, `local_showdown.py` (fork API),
`neural_policy.py` (priors/value), `policy.py` (a search-policy adapter).

### WS-E — Value-head calibration + observation/belief support (on the MCTS critical path)
**Owner goal:** a value head good enough to guide MCTS, a Markov-complete observation, and a clean
belief-sampling API for the searcher. Not "lighter" — WS-D's search quality is bounded by the value
head, so this gates M0/M3.
Steps: audit and improve value-target construction (terminal return, discount, capped-game value)
and measure value-head **calibration** (predicted vs realized outcome); confirm multi-turn-effect
duration encodings are complete; expose a clean belief-determinization (opponent-set sampling) API.
The calibration metric/artifact path now exists; the open work is improving the value targets/model
until held-out calibration is good enough to guide search. The standalone value-calibration commands
can now compare raw/affine/isotonic on held-out data and fit affine or isotonic stored transforms;
isotonic is a WS-E calibration lever, not a substitute for improving the underlying value ranking.
The opt-in pairwise value-ranking loss is the next model-side lever to test because MCTS leaf
selection depends on value ordering more than absolute calibration alone. It is a global
batch-ranking proxy, not direct supervision on sibling leaves from a single search tree.
One current model-side lever is the optional recurrent temporal aggregator, which should be evaluated
as a base-net/value-head upgrade before resuming larger root-PUCT reads. The dataset path also
exposes optional clipped shaped return targets for visible HP/faint deltas and late-turn pressure;
these should be ablated before replacing terminal-only targets as the default.
Deliverable: a calibration metric + improved value targets + a belief-sampling API.
Acceptance: value-head calibration is good enough that net+MCTS > net-alone (verified jointly in M0);
WS-D can request sampled opponent sets.
Touches: `showdown.py`, `belief.py`, `dataset.py`, `randbat_vocab.py`.

### WS-F — Evaluation, strength tracking, and ladder
**Owner goal:** a *fixed* yardstick to detect climbing vs stagnation, and a path to human-relative Elo.
Steps:
1. Per-iteration eval vs a frozen set: max-damage + a few frozen past checkpoints; persist the curve.
2. A larger eval (≥300–400 games) for low-variance strength reads at milestones.
3. Ladder path: use `online_client.py` to play checkpoints on the live server for human-relative Elo
   (the ultimate goal).
Deliverable: strength-curve tracking + a ladder eval runbook.
Acceptance: a flat multi-iteration curve reliably signals stagnation (→ add search); a rising curve
confirms progress.
Touches: `collection.py`, `evaluation.py`, `neural_cli.py`, `online_client.py`.

---

## Sequencing & milestones

**Ordering principle: prove the load-bearing bet (search) cheaply *before* spending fleet compute on
scale, but do not keep tuning search on an unreadable foundation.** Search improves a decent net; it
does not rescue a value head that cannot rank leaves.

- **Now (parallel, ordered by current bottleneck):** WS-E (improve value calibration beyond the new
  report/artifact plumbing), WS-A (produce a stronger current-schema base net, including H3
  opponent-signal ablations), WS-F (fixed yardstick with milestone-scale samples), and WS-C/WS-D
  harness hardening only where it removes known blockers.
  WS-B (full fleet scaling) can be scaffolded but is **not** on the critical path to M0.
  Treat additional 8-16 game root-PUCT probes as plumbing checks only. The next strength-relevant
  milestone should either improve the value/base net or run search at milestone scale on a foundation
  checkpoint whose value head is demonstrably more reliable.
- **M0 — Prove search lifts a modest net (the de-risking gate):** WS-C + minimal WS-D + WS-E on a
  cheap/early WS-A net → **net+MCTS clears ~0.60 vs max-damage** (past the 0.52 plateau). Pass →
  scale. Fail → fix the operator (search depth / value head / DUCT) before any fleet compute.
- **M1 — Scaled self-play net:** WS-A + WS-B + WS-F → a league-trained net on the fleet at ~thesis
  scale, with a *rising* strength curve (M1 gate).
- **M2 — Full MCTS:** harden WS-D (determinization over multiple sampled sets, roll-grouping, search
  budget; DUCT if opponent-as-policy limits strength) on the scaled net.
- **M3 — Search beats net at scale:** net+MCTS clearly beats net-alone and baselines at the larger
  eval (≥300–400 games).
- **M4 — Ladder:** WS-F ladder path — measure human-relative Elo; iterate.

## Anti-stagnation guardrails (apply throughout)
- League/history-pool opponents (WS-A) — not just the latest self.
- Exploration pressure (entropy/temperature) so the policy doesn't collapse early.
- Fixed-yardstick strength curve (WS-F) — the early-warning signal for local minima.
- Search (WS-D) as the ultimate improvement operator — the thing that pulls the policy out of a
  searchless local optimum.

## Open questions / risks
- **Forking/rollout mechanism (WS-C)** is the biggest unknown; validate in days. Prefer
  replay-from-root over snapshot/restore; last-resort fallback is a learned/in-process model.
- **Value-head quality (WS-E)** is a hard MCTS dependency — a noisy value makes search worse than the
  raw policy. Measure calibration, don't assume it.
- **On-policy staleness (WS-B):** distributed PPO must use synchronous, single-checkpoint iterations
  or it degrades — do not mix checkpoints in one update.
- **Sim speed for search:** warm sim is ~0.4 ms/turn, but MCTS multiplies sim calls; per-move budget
  matters (the thesis worked within a 10 s/move ladder timer).
- **Simultaneous moves:** start with opponent-plays-policy (simple); upgrade to DUCT if it limits
  strength vs stronger opponents.
- **In-loop MCTS (true AlphaZero) is a research *stretch*, not near-term.** The thesis avoided it for
  sim-speed reasons, and in-loop MCTS for *simultaneous-move* games is genuinely hard. The validated
  path is PPO-self-play-then-test-time-MCTS; treat in-loop as a later experiment, not a milestone.
- **Plateau-break + transfer are hypotheses:** prior self-play stalled at 0.52, and the thesis was
  gen4, not gen3 (different mechanics: type-based phys/spec split, gen3 sleep/ability set). Hold the
  go/no-go gates.
- **Ladder eval is noisy/slow and the online client (`online_client.py`) is young** — it needs
  reconnect/timeout hardening before ladder Elo is a trustworthy signal.
- **Compute:** thesis hit rank 8 on ~3M battles / one GPU / ~80 CPU / 4 days — our budget target.

## References
- MIT thesis (PPO self-play + test-time MCTS; rank 8 gen4) — the blueprint.
- Foul Play (DUCT + damage-roll grouping). Technical Machine (expectiminimax). *-Minimax / MCMS
  (best/worst/avg chance). metamon (offline-RL human-level). See `docs/max_damage_exploration_learnings.md`
  for the plateau analysis that motivates the search direction.
