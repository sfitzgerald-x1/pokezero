#!/usr/bin/env python3
"""Can the value head rank the two moves search is choosing between?

THE QUESTION, and why the numbers we already have cannot answer it.

The value-gap investigation reported a top-1/top-2 root Q gap of 0.0192 against a
calibration error of 0.0516 and concluded the search cannot resolve its own arms. Both of
those are properties of SEARCH OUTPUT: `top_arms[].q` is the crate's `MoveStats::mean()`
-- `total_value / visits`, the backed-up subtree mean over every simulation through an arm
(`rust/pokezero-search/src/tree.rs:120`) -- and the ECE was computed from that same
backed-up Q against the realized outcome
(`deployment/mcts/analyze_value_gap.py:618-624`). The raw value head appears in no banked
shard at all.

So the head's own ability to rank siblings has never been measured. This measures it.

WHY NOT ECE. Global calibration cannot answer the question even in principle. Two heads
can share an ECE of 0.05: one off by a constant +0.05 on every position, which cancels
exactly in a comparison and ranks siblings perfectly; one unbiased on average but
scattering +/-0.05 independently per position, which destroys ranking whenever the true gap
is ~0.02. Only the second breaks search, and recalibration -- the obvious remedy -- fixes
only the first. Getting this distinction wrong costs a training programme.

WHAT IS MEASURED. For each sampled decision, with the opponent's reply held fixed:

    v_head(A), v_head(B)   the head's value at each arm's successor state
    w_true(A), w_true(B)   empirical win rate from N rollouts to terminal from each

and then the only quantity that matters:

    does sign(v_head(A) - v_head(B)) agree with sign(w_true(A) - w_true(B)) ?

reported BY TRUE-GAP BUCKET, because a head that ranks correctly when the truth is wide and
coin-flips when it is narrow has a different problem from one that is wrong everywhere --
and search lives in the narrow bucket.

THE MEASUREMENT TRAP, one level down. Ground truth is itself a noisy estimate. At N
rollouts the standard error of a win rate near 0.5 is 0.5/sqrt(N): 0.05 at N=100, still
2.6x the 0.019 gap it must resolve. Paired rollouts (common random numbers across the two
arms) cancel much of the shared variance, which is why `--paired-seeds` is on by default.
Even so, the honest output is accuracy per bucket with the resolvable floor stated, not a
single number pretending to resolve 0.019.

SEARCH STACK. This uses the Python replay/rollout machinery (`replay_branching.py`,
`rollout.py`), not the Rust crate MCTS the banked numbers came from. That is deliberate and
sound: the quantity is a property of the HEAD and of successor states produced by the env,
not of any search. The successors here come from `replay_trajectory_branch`, so no search
is involved on either side of the comparison.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rollout_seed(base: int, battle_id: str, round_index: int, arm: int, trial: int,
                 *, paired: bool) -> int:
    """Seed for one rollout.

    With `paired=True` the arm index is EXCLUDED from the hash, so arm A's trial i and
    arm B's trial i share a seed and therefore share their chance draws wherever the two
    lines coincide. That is common random numbers: it cancels the shared component of the
    variance, which is most of it, and is the difference between resolving a 0.02 gap at a
    few hundred rollouts and needing several thousand.
    """
    parts = [str(base), battle_id, str(round_index), str(trial)]
    if not paired:
        parts.append(str(arm))
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:12], 16)


def score_pairs(pairs: Sequence[Mapping[str, Any]], buckets: Sequence[float]) -> dict:
    """Sign-agreement between the head's ordering and ground truth, per true-gap bucket."""
    edges = list(buckets) + [float("inf")]
    out: dict[str, Any] = {"buckets": [], "n_pairs": len(pairs)}
    for lo, hi in zip(edges, edges[1:]):
        sel = [p for p in pairs if lo <= abs(p["true_gap"]) < hi]
        # A pair whose ground-truth gap is zero has no correct ordering; excluded rather
        # than counted as a failure of the head.
        sel = [p for p in sel if p["true_gap"] != 0.0]
        if not sel:
            out["buckets"].append({"lo": lo, "hi": hi, "n": 0, "accuracy": None})
            continue
        agree = sum(1 for p in sel
                    if (p["head_gap"] > 0) == (p["true_gap"] > 0) and p["head_gap"] != 0)
        n = len(sel)
        lo_ci, hi_ci = wilson(agree, n)
        out["buckets"].append({
            "lo": lo, "hi": hi, "n": n, "agree": agree, "accuracy": agree / n,
            "ci95": [lo_ci, hi_ci],
            "beats_chance": lo_ci > 0.5,
            "mean_true_gap": statistics.mean(abs(p["true_gap"]) for p in sel),
            "mean_head_gap": statistics.mean(abs(p["head_gap"]) for p in sel),
        })
    allsel = [p for p in pairs if p["true_gap"] != 0.0]
    if allsel:
        agree = sum(1 for p in allsel
                    if (p["head_gap"] > 0) == (p["true_gap"] > 0) and p["head_gap"] != 0)
        out["overall"] = {"n": len(allsel), "agree": agree,
                          "accuracy": agree / len(allsel),
                          "ci95": list(wilson(agree, len(allsel)))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--showdown-root", required=True, type=Path)
    ap.add_argument("--games", type=int, default=4,
                    help="source games; decisions are sampled from their trajectories")
    ap.add_argument("--seed-start", type=int, default=24000000)
    ap.add_argument("--decisions-per-game", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=64,
                    help="rollouts per arm. SE near 0.5 is 0.5/sqrt(N) -- 0.0625 at 64 -- "
                         "so read the per-bucket floor, not a single headline number")
    ap.add_argument("--paired-seeds", action="store_true", default=True,
                    help="common random numbers across the two arms (default on)")
    ap.add_argument("--no-paired-seeds", dest="paired_seeds", action="store_false")
    ap.add_argument("--max-decision-rounds", type=int, default=250,
                    help="rollout cap; 250 is effectively 'play to terminal'")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--buckets", default="0.0,0.02,0.05,0.10,0.20")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.neural_policy import (
        evaluate_transformer_action_priors, evaluate_transformer_observation_value,
        load_transformer_checkpoint,
    )
    from pokezero.replay_branching import replay_trajectory_branch, replay_trajectory_branch_rollout
    from pokezero.rollout import RolloutConfig, continue_rollout_from_current_state
    from pokezero.search import player_observation_history, terminal_value_for_player

    buckets = [float(x) for x in args.buckets.split(",")]
    print(f"loading checkpoint {args.checkpoint}", flush=True)
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)

    def head_value(observations) -> float:
        """The raw value head on an observation history. NOT a backed-up Q.

        This is the quantity that appears in no banked shard, and the whole point of the
        probe. Calibration transform included, so it is the number search would see.
        """
        return evaluate_transformer_observation_value(
            model=model, result=result, observations=observations, device=args.device)

    def make_policy():
        from pokezero.neural_policy import TransformerSoftmaxPolicy
        return TransformerSoftmaxPolicy(
            model=model, result=result, device=args.device, temperature=1.0, sample=True)

    cfg = RolloutConfig(max_decision_rounds=args.max_decision_rounds)
    pairs: list[dict] = []
    terminal_pairs: list[dict] = []
    skipped = collections.Counter()

    for gi in range(args.games):
        seed = args.seed_start + gi
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=args.showdown_root))
        env.reset(seed=seed)
        policies = {"p1": make_policy(), "p2": make_policy()}
        source = continue_rollout_from_current_state(
            env=env, policies=policies, config=cfg, seed=seed,
            battle_id=f"probe-{seed}", starting_decision_round_index=0)
        traj = source.trajectory
        rounds = source.decision_round_count
        print(f"game {gi} seed={seed}: {rounds} decision rounds, "
              f"terminal winner={source.terminal.winner}", flush=True)

        # Sample decisions spread across the game rather than clustered at the opening:
        # the banked row cap already biases toward early-game and repeating that bias here
        # would make the probe unrepresentative of the decisions search actually faces.
        if rounds < 4:
            skipped["game_too_short"] += 1
            continue
        step = max(1, rounds // (args.decisions_per_game + 1))
        prefixes = list(range(step, rounds, step))[: args.decisions_per_game]

        for prefix in prefixes:
            seat = "p1"
            # The two arms to compare. Taken from the POLICY's top-2 priors rather than from
            # a search, so the probe measures the head on the pair a searcher would be
            # deciding between without needing a search in the loop.
            try:
                # The SAME builder search uses (search.py:3846, called with
                # through_decision_round=prefix at search.py:2021). An earlier revision
                # filtered turn_index < prefix, deleting the observation AT the decision.
                # The head consumes a sliding window, so that is a non-contiguous input it
                # was never trained on -- and because both arms shared the hole, nothing
                # would have looked wrong. It would have produced a confident
                # "the head cannot rank siblings" from an input search never produces.
                obs_hist = player_observation_history(
                    traj, player_id=seat, through_decision_round=prefix)
                arm_a, arm_b, opp_action = _top_two_and_opponent(
                    traj, seat, prefix, model, result, args.device,
                    evaluate_transformer_action_priors, obs_hist)
            except Exception as exc:                      # noqa: BLE001
                skipped[f"arm_selection:{type(exc).__name__}"] += 1
                continue
            if arm_a is None or arm_b is None:
                skipped["fewer_than_two_legal_arms"] += 1
                continue

            rec: dict[str, Any] = {"seed": seed, "prefix": prefix, "seat": seat,
                                   "arm_a": arm_a, "arm_b": arm_b}
            ok = True
            for label, arm in (("a", arm_a), ("b", arm_b)):
                branch_actions = {seat: arm, ("p2" if seat == "p1" else "p1"): opp_action}
                # HEAD value at this arm's successor -- one branch, no rollout.
                try:
                    benv = LocalShowdownEnv(LocalShowdownConfig(showdown_root=args.showdown_root))
                    benv.reset(seed=seed)
                    br = replay_trajectory_branch(
                        benv, traj, prefix_decision_round_count=prefix,
                        branch_actions=branch_actions, check_prefix_observations=False)
                    hist, branch_terminal = _post_branch_history(br, seat, obs_hist)
                    if hist is None:
                        if branch_terminal is not None:
                            # The branch ENDED the battle, so there is no successor state
                            # and the head cannot be asked about one. An earlier revision
                            # scored it on the PRE-BRANCH position: for a pair where both
                            # arms end the game that makes head_a == head_b exactly, so
                            # head_gap == 0, which score_pairs counts as a miss -- turning
                            # every decisive pair into a deterministic zero in the widest
                            # bucket, for a reason that has nothing to do with the head.
                            #
                            # Recorded with its exact ground truth and EXCLUDED from the
                            # ranking metric. The exclusion is a real limitation of the
                            # measurement, not a defect: the wide-gap bucket is
                            # under-sampled by construction, and the count is printed so a
                            # reader can see that rather than infer a head failure.
                            rec[f"true_{label}"] = (
                                1.0 if branch_terminal.winner == seat
                                else (0.5 if branch_terminal.winner is None else 0.0))
                            rec[f"terminal_{label}"] = True
                            continue
                        skipped["seat_not_requested_after_branch"] += 1
                        ok = False
                        break
                    rec[f"head_{label}"] = head_value(hist)
                except Exception as exc:                  # noqa: BLE001
                    skipped[f"branch:{type(exc).__name__}"] += 1
                    ok = False
                    break
                # GROUND TRUTH: N rollouts to terminal from that same successor.
                wins = 0.0
                done = 0
                failed_trials: set[int] = set()
                for trial in range(args.rollouts):
                    rseed = rollout_seed(seed, f"probe-{seed}", prefix,
                                         0 if label == "a" else 1, trial,
                                         paired=args.paired_seeds)
                    try:
                        renv = LocalShowdownEnv(LocalShowdownConfig(showdown_root=args.showdown_root))
                        renv.reset(seed=seed)
                        rr = replay_trajectory_branch_rollout(
                            renv, traj, prefix_decision_round_count=prefix,
                            branch_actions=branch_actions,
                            policies={"p1": make_policy(), "p2": make_policy()},
                            rollout_config=cfg, check_prefix_observations=False)
                    except Exception:                     # noqa: BLE001
                        # Counted, never silent. A dropped trial also BREAKS the paired-seed
                        # design: if arm A's trial 7 fails and arm B's does not, the two
                        # arms no longer share their common random numbers and the variance
                        # cancellation the whole design rests on is gone for that pair.
                        failed_trials.add(trial)
                        skipped["rollout_failed"] += 1
                        continue
                    term = rr.continuation.terminal
                    # A capped game has no winner. Counted as a half, and counted
                    # SEPARATELY, because silently treating it as a loss would bias
                    # exactly the long grindy lines that stall.
                    if term.winner is None and term.capped:
                        rec[f"capped_{label}"] = rec.get(f"capped_{label}", 0) + 1
                        wins += 0.5
                    else:
                        wins += 1.0 if term.winner == seat else 0.0
                    done += 1
                if done == 0:
                    skipped["no_rollouts_completed"] += 1
                    ok = False
                    break
                rec[f"true_{label}"] = wins / done
                rec[f"rollouts_{label}"] = done
                rec[f"failed_{label}"] = sorted(failed_trials)
            if not ok:
                continue
            # PAIRING RECONCILIATION. Common random numbers only cancel variance if the two
            # arms ran the SAME trials. If either arm lost trials, the shared component is
            # gone for the trials the other kept, so the pair's true_gap is no longer a
            # paired estimate. Recorded rather than silently accepted; a pair whose arms
            # disagree on which trials survived is marked and excluded from the paired
            # claim, because using it would quietly reintroduce the variance the design
            # exists to remove.
            if rec.get("terminal_a") or rec.get("terminal_b"):
                # Exact ground truth, but no head estimate for the terminal arm, so the
                # pair cannot test the head's ORDERING. Kept in the record and counted.
                skipped["terminal_branch_no_head_estimate"] += 1
                terminal_pairs.append(rec)
                continue
            fa, fb = set(rec.get("failed_a", ())), set(rec.get("failed_b", ()))
            rec["pairing_intact"] = (fa == fb)
            if not rec["pairing_intact"]:
                skipped["pairing_broken_by_failed_trials"] += 1
                continue
            rec["head_gap"] = rec["head_a"] - rec["head_b"]
            rec["true_gap"] = rec["true_a"] - rec["true_b"]
            pairs.append(rec)
            print(f"  prefix {prefix}: head {rec['head_a']:+.4f}/{rec['head_b']:+.4f} "
                  f"(gap {rec['head_gap']:+.4f})  true {rec['true_a']:.3f}/{rec['true_b']:.3f} "
                  f"(gap {rec['true_gap']:+.3f})", flush=True)

    if not pairs:
        print(f"CANNOT RUN: no scorable pairs. skipped={dict(skipped)}")
        return 2

    scored = score_pairs(pairs, buckets)
    # Resolution from what actually RAN. Printing 0.5/sqrt(--rollouts) would overstate the
    # precision of every pair that lost a trial, and the whole point of the probe is that
    # the ground truth's own noise is the thing most likely to fool it.
    realised = [min(p["rollouts_a"], p["rollouts_b"]) for p in pairs
                if p.get("rollouts_a") and p.get("rollouts_b")]
    n_eff = min(realised) if realised else 0
    se = (0.5 / math.sqrt(n_eff)) if n_eff else None
    exact = len(terminal_pairs)
    print(f"\n=== sibling discrimination, {len(pairs)} pairs ===")
    if se is None:
        print("ground-truth resolution: CANNOT RUN -- no pair completed rollouts on both "
              "arms, so no rollout-based gap is resolvable")
    else:
        print(f"ground-truth resolution: worst-case {n_eff} rollouts/arm actually completed "
              f"-> SE ~{se:.4f} near 0.5"
              f"{' (paired seeds reduce this)' if args.paired_seeds else ''}")
    if exact:
        print(f"  {exact} pairs had an arm END the battle: exact ground truth, but NO head "
              f"estimate exists for a state that does not exist, so they are excluded from "
              f"the ranking metric. The widest bucket is under-sampled by that much.")
    print(f"{'true-gap bucket':>18s} {'n':>5s} {'accuracy':>9s} {'95% CI':>16s} {'>chance':>8s}")
    for b in scored["buckets"]:
        if not b["n"]:
            continue
        hi = "inf" if b["hi"] == float("inf") else f"{b['hi']:.2f}"
        print(f"{b['lo']:.2f}-{hi:>6s}{'':>5s} {b['n']:5d} {b['accuracy']:9.3f} "
              f"[{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}] {str(b['beats_chance']):>8s}")
    o = scored.get("overall")
    if o:
        print(f"{'OVERALL':>18s} {o['n']:5d} {o['accuracy']:9.3f} "
              f"[{o['ci95'][0]:.3f},{o['ci95'][1]:.3f}]")
    print("\nA bucket whose CI includes 0.500 has NOT shown the head can rank at that gap.")
    print("The bucket that matters is the SMALLEST one -- search lives at gaps near 0.02.")
    if skipped:
        print(f"skipped: {dict(skipped)}")
    if args.json:
        args.json.write_text(json.dumps(
            {"config": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
             "ground_truth_se": se, "scored": scored, "pairs": pairs,
             "terminal_pairs_excluded": terminal_pairs,
             "skipped": dict(skipped)}, indent=1, default=str))
        print(f"wrote {args.json}")
    return 0


def _post_branch_history(branch_result, seat, prefix_history):
    """Prefix history plus the observation at the branched successor.

    ReplayBranchResult carries (prefix, branch_round, step_result); the post-branch
    observations live on step_result.observations (env.StepResult). An earlier revision read
    a non-existent `.observations` off the branch result itself, which would have appended
    nothing and silently scored the head on the PREFIX instead of the successor -- the two
    arms would then have produced identical values and every pair would have tied.
    """
    step_result = getattr(branch_result, "step_result", None)
    terminal = getattr(step_result, "terminal", None)
    obs = dict(getattr(step_result, "observations", {}) or {})
    nxt = obs.get(seat)
    if nxt is None:
        # Two legitimate reasons the seat has no observation: the branch ENDED the battle
        # (LocalShowdownEnv.step returns observations only for next_requested, and nothing
        # on terminal), or the seat is simply not requested next. A terminal branch is the
        # most informative pair in the sample -- arm A a winning KO against arm B a whiff
        # is |true_gap| ~ 1.0 -- so discarding it would systematically restrict the sample
        # to non-decisive positions and empty the wide-gap bucket for a reason that has
        # nothing to do with the head. Signalled, not raised.
        return None, terminal
    return tuple(prefix_history) + (nxt,), terminal


def _top_two_and_opponent(traj, seat, prefix, model, result, device,
                          evaluate_transformer_action_priors, history):
    """The two arms to compare, plus the opponent's reply to hold fixed.

    The opponent's action is FIXED across the two arms on purpose. This is a
    simultaneous-move game, so a successor is only defined given both actions; varying the
    opponent between arms would compare two different subgames and attribute the difference
    to the head.
    """
    step = next((s for s in traj.steps
                 if s.player_id == seat and s.turn_index == prefix), None)
    if step is None:
        raise LookupError(f"no step for {seat} at round {prefix}")
    mask = list(step.observation.legal_action_mask)
    legal = [i for i, ok in enumerate(mask) if ok]
    if len(legal) < 2:
        return None, None, 0
    # The FULL seat history, not a single observation. TransformerSoftmaxPolicy tensorises
    # history[-window_size:] (neural_policy.py:1725-1741), so a length-1 call returns a
    # different prior vector and the "top two" would not be the pair the model would
    # actually be deciding between -- worst exactly at the mid-game positions this probe
    # deliberately samples, where the history is longest.
    priors = evaluate_transformer_action_priors(
        model=model, result=result, observations=tuple(history), device=device)
    ranked = sorted(legal, key=lambda i: -priors[i])
    opp = "p2" if seat == "p1" else "p1"
    ostep = next((s for s in traj.steps
                  if s.player_id == opp and s.turn_index == prefix), None)
    opp_action = int(ostep.action_index) if ostep is not None else 0
    return ranked[0], ranked[1], opp_action


if __name__ == "__main__":
    raise SystemExit(main())
