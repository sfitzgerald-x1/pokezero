#!/usr/bin/env python3
"""Re-measure a DIFFERENT value head on the ALREADY-BANKED sibling pairs.

WHY THIS EXISTS. Phase 3 advances an arm on beta and ECE. ECE comes out of training. beta
comes from `mcts/sibling_beta.py`, which regresses the banked `true_gap` on the banked
`head_gap` -- and `head_gap` is a property of the head that produced the bank. A tuned head
therefore has no beta until something recomputes `head_gap` for it. Nothing did, so Phase 3
had a gate that no tuned checkpoint could be scored against.

WHY THE BANK CANNOT SIMPLY BE RE-READ. `campaign-store/vhprobe-pairs-*.json` stores SCALARS
(`true_a`, `true_b`, `head_gap`, `noise_var`, `seed`, `prefix`, `seat`, `arm_a`, `arm_b`) and
deliberately drops the per-trial outcome dumps and every observation tensor. There is nothing
in it to run a head on. The decision states have to be REBUILT.

WHAT IS AND IS NOT RECOMPUTED. The expensive half of the bank is the ground truth: R=64
paired rollouts to terminal per arm, 465 pairs, ~59.5k battles. That half is a property of the
ENVIRONMENT and the state, not of the head, so it is COPIED FORWARD unchanged -- `true_a`,
`true_b`, `noise_var`, `rollouts_*`, `failed_*`, `pairing_intact`, `capped_*`, `drawn_*`. Only
the head values are recomputed. That is the whole economy of this tool: ~1.5% of the producer's
replay work (2 branch replays per pair instead of 2 + 2*64).

HOW THE STATE IS REBUILT, and why a single replay serves every candidate head at once. The
producer's states are a deterministic function of (source seed, the policy that played the
source game, the policy that ranks the arms). A value-only fine-tune leaves the trunk and the
policy head bit-identical, so all of that is identical across cells. This tool therefore
replays each source game ONCE with the STATE checkpoint, and evaluates every `--head`
checkpoint at the same successor observation histories.

THAT PRESUMPTION IS CHECKED, NOT ASSUMED, TWICE:

  1. TRUNK IDENTITY. Every parameter outside `value_head.*` is compared bit-for-bit against
     the state checkpoint and a difference REFUSES the run. If a head's trunk differed, it
     would have played a different source game and the pairs would not be the same pairs --
     silently, because the output schema cannot show it.
  2. REPRODUCTION. The state checkpoint is itself scored as a head named `__reproduce__` and
     its values are compared against the BANKED `head_a`/`head_b`. If the replay drifted, this
     is what says so. It is the only evidence that the states rebuilt here are the states the
     ground truth was measured at, and it is on by default because a rescore that silently
     scores different states produces a confident wrong beta.

     Reproduction is bit-sensitive: the source game is played by SAMPLING from the policy, so
     a last-bit difference in the priors can flip one action and diverge the whole trajectory.
     Run this on the same device the bank was produced on (the 20260815 bank: `device=cuda`).
     A CPU rescore of a CUDA bank is expected to fail this check, and failing loudly is the
     point.

ARM AGREEMENT is the third check and the cheapest. The bank records `arm_a`/`arm_b`; the
replay re-derives them from the policy's top-2 priors. A pair whose re-derived arms disagree
is DROPPED and counted, never silently rescored -- its `true_a`/`true_b` describe two other
moves.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
# The producer is the specification. Its arm-selection, post-branch-history and gap-unit
# helpers are IMPORTED rather than restated: a rescore that computed `head_gap` even slightly
# differently from the bank would produce a beta that is not comparable to the baseline's,
# which is the single thing this tool exists to make comparable. The units conversion in
# `finalize_pair_gaps` (return scale -> win probability) is exactly the defect that once made
# a perfect head read 0.0225, so it is reused, not reimplemented.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import value_head_sibling_probe as probe  # noqa: E402

REPRODUCE = "__reproduce__"
VALUE_HEAD_PREFIX = "value_head"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trunk_difference(state_sd, head_sd) -> list[str]:
    """Keys outside `value_head.*` that are absent or unequal. Empty means bit-identical.

    Compared on the LOADED modules' state dicts, not on the raw payloads, so what is checked
    is what will actually run. A payload can carry a key the module ignores -- `v1`'s seed
    file ships a 1x512 `value_head.weight` beside a `value_head_hidden=256` config -- and
    comparing payloads would compare something no forward pass reads.
    """
    import torch

    out = []
    for key, ref in state_sd.items():
        if key.split(".")[0] == VALUE_HEAD_PREFIX:
            continue
        got = head_sd.get(key)
        if got is None or got.shape != ref.shape or not torch.equal(got.cpu(), ref.cpu()):
            out.append(key)
    return out


def parse_heads(specs: list[str]) -> dict[str, Path]:
    heads: dict[str, Path] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).parent.name or Path(spec).stem, spec
        if name == REPRODUCE:
            raise SystemExit(f"{REPRODUCE!r} is reserved for the state checkpoint.")
        if name in heads:
            raise SystemExit(f"duplicate --head name {name!r}")
        heads[name] = Path(path)
    if not heads:
        raise SystemExit("at least one --head NAME=PATH is required")
    return heads


def merge(shards: list[Path], out: Path) -> int:
    """Pool rescore shards for ONE head into the single file `sibling_beta.py` consumes.

    A merge step is needed because the replay is sharded by source seed while beta is a
    regression over the POOLED pairs -- eight per-shard slopes averaged is a different
    estimand, and the baseline was computed on all 465 at once.

    Every gate here exists because pooling is where two different experiments become one
    confident number:

    - shards must name the SAME head checkpoint and the SAME state checkpoint sha, or the
      pooled `head_gap` column mixes two heads;
    - shards must come from the same banked file by SHA, or the reused ground truth is not
      one ground truth;
    - a shard that FAILED its reproduction check is refused, not down-weighted: its states
      are not the banked states. That refusal is also why `reproduction_pooled` carries no
      failure count: any shard with one would have been rejected, so a pooled counter could
      only ever print zero. A field that reads like evidence and can only hold one value is
      worse than no field, so what is pooled instead is the loosest tolerance any shard was
      actually checked at (`tol_max`), which a reader cannot otherwise recover;
    - (seed, prefix) must be unique across shards, which catches overlapping --shard/
      --num-shards arithmetic. Silent duplication would tighten the bootstrap CI on
      resampled copies of the same pair.
    """
    pairs: list[dict] = []
    heads, banks, states, names = set(), set(), set(), set()
    seen: dict[tuple, str] = {}
    repro: dict[str, Any] = {
        "n": 0, "max_abs_delta": 0.0, "tol_max": None,
        "note": ("Pairs rebuilt and reproduced across the pooled shards. There is deliberately"
                 " no pooled failure count: a shard reporting any pair outside its own"
                 " tolerance is REFUSED below rather than pooled, so such a counter could only"
                 " ever read zero. `tol_max` is the loosest tolerance any pooled shard was"
                 " checked at -- shards may be run with different --reproduce-tol, and the"
                 " pooled column is only as strong as the weakest one."),
    }
    dropped: collections.Counter = collections.Counter()
    for path in shards:
        doc = json.loads(Path(path).read_text())
        rs = doc.get("rescore")
        if not rs:
            raise SystemExit(f"CANNOT RUN: {path} has no `rescore` block, so it was not "
                             f"written by this tool and its head_gap provenance is unknown.")
        heads.add(doc.get("head_checkpoint"))
        names.add(doc.get("head_name"))
        banks.add(rs.get("banked_pairs_sha256"))
        states.add(rs.get("state_checkpoint_sha256"))
        r = rs.get("reproduction") or {}
        if r.get("n_outside_tol"):
            raise SystemExit(
                f"CANNOT RUN: {path} reports {r['n_outside_tol']} pairs outside its "
                f"reproduction tolerance. Its rebuilt states are not the banked states and "
                f"pooling it would launder that into a comparable-looking beta.")
        repro["n"] += r.get("n") or 0
        repro["max_abs_delta"] = max(repro["max_abs_delta"], r.get("max_abs_delta") or 0.0)
        tol = r.get("tol")
        if tol is not None:
            repro["tol_max"] = (float(tol) if repro["tol_max"] is None
                                else max(repro["tol_max"], float(tol)))
        for k, v in (rs.get("dropped") or {}).items():
            dropped[k] += v
        for pr in doc.get("pairs") or []:
            key = (pr["seed"], pr["prefix"], pr["seat"])
            if key in seen:
                raise SystemExit(f"CANNOT RUN: pair {key} appears in both {seen[key]} and "
                                 f"{path}. Overlapping shards would double-count it.")
            seen[key] = str(path)
            pairs.append(pr)
    for label, vals in (("head_checkpoint", heads), ("head_name", names),
                        ("banked_pairs_sha256", banks),
                        ("state_checkpoint_sha256", states)):
        if len(vals) != 1:
            raise SystemExit(f"CANNOT RUN: shards disagree on {label}: {sorted(vals)}")
    payload = {
        "schema": "pokezero.phase0.vhprobe-pairs.v1",
        "source": f"merge of {len(shards)} rescore shards",
        "head_checkpoint": heads.pop(), "head_name": names.pop(),
        "merged_from": [str(p) for p in shards],
        "reproduction_pooled": repro,
        "dropped_pooled": dict(dropped),
        "n_pairs": len(pairs), "pairs": pairs,
    }
    Path(out).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    tol_max = repro["tol_max"]
    print(f"merged {len(pairs)} pairs from {len(shards)} shards -> {out}  "
          f"(reproduction: {repro['n']} checked, every shard within its own tol"
          f"{'' if tol_max is None else f', loosest {tol_max:g}'}, "
          f"max delta {repro['max_abs_delta']:.3e})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="append", default=[], type=Path, metavar="SHARD",
                   help="pool rescore shards for one head and exit; needs --merge-out")
    ap.add_argument("--merge-out", type=Path, default=None)
    ap.add_argument("--pairs", type=Path,
                    help="the banked pairs file whose ground truth is being reused")
    ap.add_argument("--head", action="append", default=[], metavar="NAME=PATH",
                    help="a checkpoint to score. Repeatable; all are scored on ONE replay, "
                         "which is sound only because the trunk-identity check passes.")
    ap.add_argument("--state-checkpoint", type=Path, default=None,
                    help="the checkpoint that PRODUCED the bank, i.e. the one that plays the "
                         "source games and ranks the arms. DEFAULT: read off the bank's own "
                         "provenance, so it cannot be mismatched by hand.")
    ap.add_argument("--showdown-root", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--device", default="cuda",
                    help="MUST match the device the bank was produced on or the reproduction "
                         "check will fail; the 20260815 bank is cuda.")
    ap.add_argument("--max-decision-rounds", type=int, default=250,
                    help="must equal the producer's; the bank records it in config")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--reproduce-tol", type=float, default=1e-4,
                    help="max |replayed - banked| head value tolerated for __reproduce__")
    ap.add_argument("--max-reproduce-failures", type=int, default=0,
                    help="pairs allowed to exceed --reproduce-tol before the run REFUSES. "
                         "Default 0: a rescore of states that are not the banked states is "
                         "worse than no rescore, because its beta looks comparable.")
    ap.add_argument("--allow-unstamped-belief", action="store_true")
    args = ap.parse_args()

    if args.merge:
        if not args.merge_out:
            raise SystemExit("--merge needs --merge-out")
        return merge(args.merge, args.merge_out)
    missing = [f for f, v in (("--pairs", args.pairs), ("--showdown-root", args.showdown_root),
                              ("--out-dir", args.out_dir)) if v is None]
    if missing:
        raise SystemExit(f"required for a rescore run: {', '.join(missing)}")

    heads = parse_heads(args.head)
    doc = json.loads(args.pairs.read_text())
    banked = doc.get("pairs") or doc.get("rows") or []
    if not banked:
        raise SystemExit(f"CANNOT RUN: no pairs in {args.pairs}")
    prov = doc.get("provenance") or []
    prov_ckpts = sorted({p.get("checkpoint") for p in prov if p.get("checkpoint")})
    if args.state_checkpoint is None:
        if len(prov_ckpts) != 1:
            raise SystemExit(
                f"CANNOT RUN: the bank's provenance names {len(prov_ckpts)} producing "
                f"checkpoints {prov_ckpts}, so --state-checkpoint cannot be inferred. The "
                f"states depend on which one played the source games.")
        args.state_checkpoint = Path(prov_ckpts[0])
    elif prov_ckpts and str(args.state_checkpoint) not in prov_ckpts:
        # NOT fatal: a bit-identical copy at another path is legitimate and common (the bank
        # points into a training run; the rescore stages its own copy). Say so, loudly, and
        # let the reproduction check be the arbiter -- it tests the thing that matters.
        print(f"NOTE: --state-checkpoint {args.state_checkpoint} is not the path in the "
              f"bank's provenance ({prov_ckpts}). The reproduction check below is what "
              f"decides whether it is the same weights; a mismatch will fail it.", flush=True)
    prov_belief = sorted({p.get("belief_set_source_hash") for p in prov
                          if p.get("belief_set_source_hash")})

    from pokezero.local_showdown import (
        LocalShowdownConfig, LocalShowdownEnv, env_config_from_checkpoint_provenance,
    )
    from pokezero.neural_policy import (
        TransformerSoftmaxPolicy, category_vocab_from_model_config,
        evaluate_transformer_action_priors, evaluate_transformer_observation_value,
        feature_masks_from_model_config, load_transformer_checkpoint,
        observation_spec_from_model_config,
    )
    from pokezero.replay_branching import replay_trajectory_branch
    from pokezero.rollout import RolloutConfig, continue_rollout_from_current_state
    from pokezero.search import player_observation_history

    print(f"state checkpoint (plays the source games, ranks the arms): "
          f"{args.state_checkpoint}", flush=True)
    state_model, state_result = load_transformer_checkpoint(
        args.state_checkpoint, map_location=args.device)
    if prov_belief and getattr(state_result, "belief_set_source_hash", None) not in prov_belief:
        raise SystemExit(
            f"CANNOT RUN: state checkpoint belief hash "
            f"{getattr(state_result, 'belief_set_source_hash', None)!r} is not the bank's "
            f"{prov_belief!r}. The env is pinned to the checkpoint's provenance, so the "
            f"replayed observations would not be the banked ones.")

    # The state checkpoint is scored FIRST and under a reserved name, so the reproduction
    # evidence is produced by the same code path as every real head rather than by a
    # bespoke branch that could pass while the real path is broken.
    loaded: dict[str, Any] = {REPRODUCE: (state_model, state_result)}
    state_sd = state_model.state_dict()
    for name, path in heads.items():
        model, result = load_transformer_checkpoint(path, map_location=args.device)
        diff = trunk_difference(state_sd, model.state_dict())
        if diff:
            raise SystemExit(
                f"REFUSING: head {name!r} ({path}) differs from the state checkpoint on "
                f"{len(diff)} parameters OUTSIDE value_head.*: {diff[:6]}. Then it would have "
                f"played a different source game and ranked different arms, so the banked "
                f"true_a/true_b do not describe its pairs. Re-run the producer for it "
                f"instead of rescoring.")
        vct = getattr(result, "value_calibration_transform", None)
        n_vh = sum(v.numel() for k, v in model.state_dict().items()
                   if k.split(".")[0] == VALUE_HEAD_PREFIX)
        n_trunk = sum(1 for k in state_sd if k.split(".")[0] != VALUE_HEAD_PREFIX)
        print(f"head {name!r}: trunk VERIFIED bit-identical ({n_trunk} keys "
              f"compared outside value_head.*), value-head params {n_vh}, "
              f"value_calibration_transform {'NONE (identity)' if vct is None else vct}",
              flush=True)
        loaded[name] = (model, result)

    model_config = state_result.model_config
    env_spec = observation_spec_from_model_config(model_config)
    env_masks = feature_masks_from_model_config(model_config)
    env_vocab = category_vocab_from_model_config(model_config, args.showdown_root)
    _belief_reported: set[int] = set()

    def make_env():
        """The producer's env construction, through the mandated single entry point.

        Reproduced from `value_head_sibling_probe.main` rather than shared, because that one
        is a closure inside a 650-line main(). The four axes (spec, vocab, masks, belief
        source) and the fail-closed belief assertion are the load-bearing part: an env built
        under the repo default instead of the checkpoint's provenance yields observations the
        head never saw, shape-compatibly.
        """
        base = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
        cfg_env = env_config_from_checkpoint_provenance(
            base, env_masks, context="rescore_value_head",
            required_specs=env_spec, required_vocabs=env_vocab)
        env = LocalShowdownEnv(cfg_env)
        want = getattr(state_result, "belief_set_source_hash", None)
        got = getattr(env, "belief_set_source_hash", None)
        if want is None and not args.allow_unstamped_belief:
            raise SystemExit(
                "CANNOT RUN: the state checkpoint carries no belief_set_source_hash while the "
                "env is pinned belief-ON. Pass --allow-unstamped-belief and say so.")
        if want is not None and got != want:
            raise SystemExit(f"CANNOT RUN: belief-set-source mismatch. checkpoint {want!r} "
                             f"vs env {got!r}.")
        if not _belief_reported:
            _belief_reported.add(1)
            print(f"belief set source: PINNED ON, hash "
                  f"{'VERIFIED' if want else 'UNVERIFIED (no hash)'} {str(want)[:16]}",
                  flush=True)
        return env

    def make_policy():
        """The source game's policy. STATE checkpoint only, and sampling, exactly as the
        producer: `deterministic=False, sampling_temperature=1.0`. Making it deterministic
        here would replay a different game from the banked one."""
        return TransformerSoftmaxPolicy(
            model=state_model, result=state_result, device=args.device,
            deterministic=False, sampling_temperature=1.0)

    by_seed: dict[int, list[dict]] = collections.defaultdict(list)
    for rec in banked:
        by_seed[int(rec["seed"])].append(rec)
    seeds = sorted(by_seed)
    mine = seeds[args.shard::args.num_shards] if args.num_shards > 1 else seeds
    print(f"{len(banked)} banked pairs over {len(seeds)} source seeds; this shard takes "
          f"{len(mine)} seeds ({args.shard}/{args.num_shards})", flush=True)

    cfg = RolloutConfig(max_decision_rounds=args.max_decision_rounds)
    out: dict[str, list[dict]] = {name: [] for name in loaded}
    dropped: collections.Counter = collections.Counter()
    repro_deltas: list[float] = []
    t0 = time.time()

    for si, seed in enumerate(mine):
        want_pairs = sorted(by_seed[seed], key=lambda r: r["prefix"])
        try:
            env = make_env()
            env.reset(seed=seed)
            source = continue_rollout_from_current_state(
                env=env, policies={"p1": make_policy(), "p2": make_policy()},
                config=cfg, seed=seed, battle_id=f"probe-{seed}",
                starting_decision_round_index=0)
        except Exception as exc:                                        # noqa: BLE001
            dropped[f"source_replay:{type(exc).__name__}"] += len(want_pairs)
            print(f"seed {seed}: SOURCE REPLAY FAILED ({exc}); {len(want_pairs)} pairs "
                  f"dropped", flush=True)
            continue
        traj = source.trajectory
        print(f"[{si + 1}/{len(mine)}] seed={seed}: {source.decision_round_count} rounds, "
              f"winner={source.terminal.winner}, {len(want_pairs)} banked pairs, "
              f"{time.time() - t0:.0f}s elapsed", flush=True)

        for want in want_pairs:
            seat, prefix = want["seat"], int(want["prefix"])
            opp = "p2" if seat == "p1" else "p1"
            try:
                obs_hist = player_observation_history(
                    traj, player_id=seat, through_decision_round=prefix)
                arm_a, arm_b, opp_action = probe._top_two_and_opponent(
                    traj, seat, prefix, state_model, state_result, args.device,
                    evaluate_transformer_action_priors, obs_hist)
            except Exception as exc:                                    # noqa: BLE001
                dropped[f"arm_selection:{type(exc).__name__}"] += 1
                continue
            if arm_a is None or arm_b is None:
                # `_top_two_and_opponent` signals "fewer than two legal actions here" by
                # returning (None, None, 0). The producer cannot have banked a pair from such
                # a decision, so this IS a divergence -- but a different one from "the replay
                # ranked two other moves", and the two have different remedies. Counting both
                # as `arm_mismatch` made that distinction unrecoverable from the output.
                dropped["no_sibling_pair_fewer_than_two_legal_actions"] += 1
                print(f"    prefix {prefix}: replay has fewer than two legal actions, so there "
                      "is no sibling pair to score -- dropped", flush=True)
                continue
            if (arm_a, arm_b) != (want["arm_a"], want["arm_b"]):
                # The replay diverged at or before this decision. Its ground truth describes
                # two OTHER moves, so rescoring it would attach the head's opinion of one
                # pair of siblings to the rollouts of another.
                dropped["arm_mismatch"] += 1
                print(f"    prefix {prefix}: ARM MISMATCH replay ({arm_a},{arm_b}) vs banked "
                      f"({want['arm_a']},{want['arm_b']}) -- dropped", flush=True)
                continue
            vals: dict[str, dict[str, float]] = {n: {} for n in loaded}
            ok = True
            for label, arm in (("a", arm_a), ("b", arm_b)):
                try:
                    benv = make_env()
                    benv.reset(seed=seed)
                    br = replay_trajectory_branch(
                        benv, traj, prefix_decision_round_count=prefix,
                        branch_actions={seat: arm, opp: opp_action},
                        check_prefix_observations=True)
                    hist, branch_terminal = probe._post_branch_history(br, seat, obs_hist)
                except Exception as exc:                                # noqa: BLE001
                    dropped[f"branch:{type(exc).__name__}"] += 1
                    ok = False
                    break
                if hist is None:
                    if branch_terminal is not None:
                        # The producer banks these separately (`terminal_pairs_excluded`) with
                        # no head value, so a pair reached here means the replay diverged into
                        # a terminal branch the bank does not have.
                        dropped["terminal_branch_not_in_bank"] += 1
                        ok = False
                        break
                    fb = benv.observe(seat)
                    if fb is None:
                        dropped["observe_fallback_returned_none"] += 1
                        ok = False
                        break
                    if not want.get(f"observe_fallback_{label}"):
                        # The producer stamps this flag. Its presence here but not in the bank
                        # means the two runs took different paths at this successor.
                        dropped["observe_fallback_divergence"] += 1
                        ok = False
                        break
                    hist = (*obs_hist, fb)
                elif want.get(f"observe_fallback_{label}"):
                    dropped["observe_fallback_divergence"] += 1
                    ok = False
                    break
                for name, (m, r) in loaded.items():
                    vals[name][label] = evaluate_transformer_observation_value(
                        model=m, result=r, observations=hist, device=args.device)
            if not ok:
                continue
            for name in loaded:
                rec = dict(want)
                rec["head_a"] = vals[name]["a"]
                rec["head_b"] = vals[name]["b"]
                probe.finalize_pair_gaps(rec)
                # true_gap is recomputed by finalize_pair_gaps from the COPIED true_a/true_b,
                # so it must land on the banked value. If it does not, the bank's schema is
                # not what this tool believes it is and every ground-truth reuse is suspect.
                if abs(rec["true_gap"] - float(want["true_gap"])) > 1e-12:
                    raise SystemExit(
                        f"REFUSING: recomputed true_gap {rec['true_gap']!r} != banked "
                        f"{want['true_gap']!r} at seed={seed} prefix={prefix}. The ground "
                        f"truth being reused is not the ground truth in the file.")
                rec["rescored_by"] = name
                out[name].append(rec)
            d = max(abs(vals[REPRODUCE]["a"] - float(want["head_a"])),
                    abs(vals[REPRODUCE]["b"] - float(want["head_b"])))
            repro_deltas.append(d)
            if d > args.reproduce_tol:
                print(f"    prefix {prefix}: REPRODUCTION MISS {d:.3e} "
                      f"(replay {vals[REPRODUCE]['a']:+.6f}/{vals[REPRODUCE]['b']:+.6f} vs "
                      f"banked {want['head_a']:+.6f}/{want['head_b']:+.6f})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_repro = len(repro_deltas)
    n_bad = sum(1 for d in repro_deltas if d > args.reproduce_tol)
    print(f"\n=== REPRODUCTION: {n_repro} pairs rebuilt, "
          f"{n_bad} outside tol {args.reproduce_tol:g} ===", flush=True)
    if n_repro:
        print(f"  |replayed - banked| head value: max {max(repro_deltas):.3e}  median "
              f"{statistics.median(repro_deltas):.3e}  mean "
              f"{statistics.fmean(repro_deltas):.3e}")
    print(f"  dropped: {dict(dropped)}")
    meta_common = {
        "schema": doc.get("schema", "pokezero.phase0.vhprobe-pairs.v1"),
        "source": f"rescore of {args.pairs.name} (ground truth REUSED, head_gap RECOMPUTED)",
        "rescore": {
            "tool": Path(__file__).name,
            "banked_pairs": str(args.pairs),
            "banked_pairs_sha256": sha256_file(args.pairs),
            "state_checkpoint": str(args.state_checkpoint),
            "state_checkpoint_sha256": sha256_file(args.state_checkpoint),
            "device": args.device,
            "shard": args.shard, "num_shards": args.num_shards,
            "seeds": mine,
            "n_banked_pairs_for_these_seeds": sum(len(by_seed[s]) for s in mine),
            "reproduction": {
                "n": n_repro, "n_outside_tol": n_bad, "tol": args.reproduce_tol,
                "max_abs_delta": (max(repro_deltas) if repro_deltas else None),
                "median_abs_delta": (statistics.median(repro_deltas)
                                     if repro_deltas else None),
                "note": ("|replayed head value - banked head value| for the STATE checkpoint "
                         "re-scored through this tool's own path. This is the evidence that "
                         "the rebuilt states are the states the banked ground truth was "
                         "measured at."),
            },
            "dropped": dict(dropped),
            "elapsed_seconds": time.time() - t0,
        },
        # Carried through so a reader of a rescored file can still see how the ground truth
        # was made. Dropping it would leave a pairs file whose true_a/true_b have no
        # provenance at all.
        "provenance_of_reused_ground_truth": prov,
    }
    for name, rows in out.items():
        path = args.out_dir / f"pairs-{name.strip('_')}.json"
        payload = dict(meta_common)
        payload["head_checkpoint"] = str(
            args.state_checkpoint if name == REPRODUCE else heads[name])
        payload["head_name"] = name
        payload["n_pairs"] = len(rows)
        payload["pairs"] = rows
        path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
        sd_head = (statistics.pstdev([r["head_gap"] for r in rows])
                   if len(rows) > 1 else float("nan"))
        print(f"wrote {path}  ({len(rows)} pairs, sd(head_gap) {sd_head:.5f})")

    if n_bad > args.max_reproduce_failures:
        raise SystemExit(
            f"REFUSING to certify: {n_bad} pairs reproduced outside tol "
            f"{args.reproduce_tol:g} (allowed {args.max_reproduce_failures}). The rebuilt "
            f"states are not the banked states, so any beta computed from these files is "
            f"NOT comparable to the baseline's. Files were written anyway, for diagnosis "
            f"only -- do not score them. Most likely cause: --device {args.device} differs "
            f"from the device the bank was produced on.")
    if n_repro == 0:
        raise SystemExit("REFUSING to certify: no pair was rebuilt, so nothing is verified.")
    print(f"\nCERTIFIED: {n_repro} states rebuilt and reproduced within "
          f"{args.reproduce_tol:g}. The rescored head_gap values are comparable to the "
          f"bank's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
