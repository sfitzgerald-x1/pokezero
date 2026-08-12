#!/usr/bin/env python3
"""Seed-paired evaluation against FoulPlay: one search config vs its own raw arm.

The FoulPlay power-config campaign needs a number that neither committed
harness produces on its own:

* ``scripts/mcts_acceptance_h2h.py`` has within-seed pairing, the provenance
  block and shard discipline -- but both of its arms are self-play.
* ``foundation/high-fidelity-eval.sh --foulplay-only`` has the FoulPlay
  opponent -- but no engine-MCTS arm and no pairing.

This driver is the third path: it drives ``python -m pokezero.foulplay_bridge``
once per (arm, seat) over a shared seed band, so the search arm and the raw arm
face FoulPlay on the SAME seeds from BOTH seats. The deliverable is the paired
delta (search - raw) per seed, not either arm's absolute rate.

Pairing contract
----------------
A "pair" here is one battle seed. Each seed is played four times in a full
cell: {search, raw} x {p1, p2}. Seat is a within-pair factor, so the two seats
face the same two teams -- the property ``power_h2h.py`` lost by deriving seat
from seed parity. Seeds are consumed as a contiguous band and the band is
asserted exactly-once by the launcher, not here.

Arms are run as SEPARATE invocations on purpose. The raw arm is
search-config-independent, so one raw shard per checkpoint pairs with every
search cell -- running it inside each cell would multiply its cost by the
number of cells for no additional evidence.

Usage (one arm of one cell)::

    python scripts/foulplay_paired_eval.py \\
        --checkpoint <trimmed.pt> --showdown-root <path> \\
        --arm search --seed-start 7800000 --pairs 200 \\
        --depth 4 --sims 1024 --batch 64 --worlds 4 \\
        --out shard.json

Emitting the two arms is enough to score a cell; ``--pair-with`` merges them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SEATS = ("p1", "p2")
SCHEMA_VERSION = "pokezero.foulplay-paired-shard.v1"

# FoulPlay's own budget is part of the OPPONENT DEFINITION, not a tuning knob:
# every arm and every cell must face the same opponent strength or the paired
# delta is measuring the opponent instead of the search config.
FOULPLAY_SEARCH_TIME_MS = 1000

# The full FoulPlay-family thread pin. Unpinned BLAS in a CPU-capped pod is a
# ~10x thrash, and it lands on the FoulPlay side, which silently weakens the
# opponent. Mirrors foundation/foulplay-k8s-probe.sh.
THREAD_PIN_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "POKEZERO_TORCH_NUM_THREADS": "1",
    "POKEZERO_TORCH_NUM_INTEROP_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
}


def checkpoint_tag(checkpoint: str, explicit: str | None = None) -> str:
    """Short stable label for a checkpoint, used to keep cells distinct.

    Prefer an EXPLICIT tag from the caller. Inferring it from the filename is a
    trap here: both campaign checkpoints are copies of files named
    `transformer-policy.pt`, so per-checkpoint subdirectories would give k0 and
    k1 the same stem -- collapsing cell A into G and R0 into R1. That fails
    loud at merge time (conflicting scores) but only after the GPU-hours are
    spent, so the launcher passes --checkpoint-tag from the campaign JSON key
    and the inferred stem is a fallback for ad-hoc runs.
    """
    if explicit:
        return explicit
    path = Path(checkpoint)
    stem = path.stem or "ckpt"
    # Disambiguate the known-colliding case without needing the flag.
    if stem in {"transformer-policy", "model", "checkpoint"} and path.parent.name:
        return f"{path.parent.name}-{stem}"
    return stem


def config_id_for(args: argparse.Namespace) -> str:
    """The cell identity that provenance and the merger key on."""
    # EVERY arm is checkpoint-qualified. Two collisions motivate this, and both
    # would corrupt a delta rather than error:
    #   - R0 (k0) and R1 (k1) are both "raw", and the raw arm is the DENOMINATOR
    #     of every paired delta;
    #   - cells A (k0) and G (k1) run the SAME search config, so an unqualified
    #     id pools them -- and cell G's entire job is the checkpoint contrast.
    tag = checkpoint_tag(args.checkpoint, getattr(args, "checkpoint_tag", None))
    if args.arm == "raw":
        return f"raw@{tag}"
    base = f"d{args.depth}-s{args.sims}-b{args.batch}-w{args.worlds}"
    # The flag changes search semantics, so it is part of the cell identity --
    # two cells that differ only by opponent priors must not merge.
    if args.opponent_priors:
        base = f"{base}+opp-priors"
    return f"{base}@{tag}"


def bridge_argv(args: argparse.Namespace, *, seat: str) -> list[str]:
    argv = [
        sys.executable, "-m", "pokezero.foulplay_bridge",
        "--checkpoint", str(args.checkpoint),
        "--showdown-root", str(args.showdown_root),
        "--games", str(args.pairs),
        "--seed-start", str(args.seed_start),
        "--pokezero-player", seat,
        "--search-time-ms", str(FOULPLAY_SEARCH_TIME_MS),
        "--policy-mode", "raw" if args.arm == "raw" else "engine-mcts",
        "--summary-out", str(seat_summary_path(args, seat)),
    ]
    # The bridge defaults these to a REPO-RELATIVE checkout
    # (DEFAULT_FOULPLAY_ROOT = <repo>/third_party/foul-play) and there is no
    # environment fallback, so a deployment that ships foul-play anywhere else
    # cannot reach it. Every other harness passes them explicitly; this driver
    # could not, which cost a full campaign probe:
    #   FileNotFoundError: foul-play Python not found at
    #   /opt/pokezero/third_party/foul-play/.venv/bin/python
    if args.foulplay_root:
        argv += ["--foulplay-root", str(args.foulplay_root)]
    if args.foulplay_python:
        argv += ["--foulplay-python", str(args.foulplay_python)]
    if args.arm != "raw":
        argv += [
            "--engine-depth", str(args.depth),
            "--engine-sims", str(args.sims),
            "--engine-batch", str(args.batch),
            "--engine-worlds", str(args.worlds),
        ]
        if args.engine_model_path:
            argv += ["--engine-model-path", str(args.engine_model_path)]
        if args.engine_tables_path:
            argv += ["--engine-tables-path", str(args.engine_tables_path)]
        if args.opponent_priors:
            argv.append("--engine-opponent-priors")
    if args.device:
        argv += ["--device", args.device]
    return argv


def seat_summary_path(args: argparse.Namespace, seat: str) -> Path:
    out = Path(args.out)
    return out.parent / f"{out.stem}-{seat}.json"


def run_seat(args: argparse.Namespace, seat: str) -> dict:
    """One bridge invocation. Non-zero exit is terminal -- never a partial shard."""
    argv = bridge_argv(args, seat=seat)
    env = dict(os.environ)
    env.update(THREAD_PIN_ENV)
    print(f"[{seat}] {' '.join(argv)}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(argv, env=env, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        raise SystemExit(
            f"bridge exited {completed.returncode} for seat {seat}; refusing to "
            "write a partial shard (a short arm silently biases the paired delta)"
        )
    summary_path = seat_summary_path(args, seat)
    if not summary_path.exists():
        raise SystemExit(f"bridge wrote no summary at {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["_seat_wall_s"] = round(time.perf_counter() - started, 1)
    return summary


def per_seed_outcomes(summary: dict, seat: str) -> dict[int, dict]:
    """Map battle seed -> the row the pairing joins on.

    Keyed by seed rather than by position: a bridge that skipped or reordered a
    game must produce a MISSING pair, not a silently mis-joined one.
    """
    rows: dict[int, dict] = {}
    for game in summary.get("game_results", []) or []:
        seed = game.get("seed")
        if seed is None:
            continue
        # `pokezero_score` is the bridge's name for the capstone score (win 1.0,
        # tie/cap 0.5). Read strictly: a `.get(..., 0.0)` here would turn a
        # renamed key into a silent all-losses arm, which reads as a real and
        # very large paired delta rather than as a broken shard.
        if "pokezero_score" not in game:
            raise SystemExit(
                f"game_results row for seed {seed} has no 'pokezero_score'; the "
                "bridge summary schema changed -- refusing to score this shard"
            )
        rows[int(seed)] = {
            "seed": int(seed),
            "seat": seat,
            "won": bool(game.get("pokezero_won")),
            "tied": bool(game.get("tied")),
            "capped": bool(game.get("capped")),
            "score": float(game["pokezero_score"]),
        }
    return rows


def seat_block(summary: dict, seat: str) -> dict:
    """Per-seat reporting is mandatory: seat asymmetry is the #937 bug class.

    THE OPPONENT-MOVE JOURNAL IS DELIBERATELY NOT LIFTED HERE, and that makes the
    `-p1`/`-p2` bridge summaries load-bearing artifacts rather than scratch files.

    The journal (`game_results[i].opponent_moves`, see the OPPONENT-MOVE JOURNAL
    block in `pokezero/foulplay_bridge.py`) is what makes a recorded fallback address
    replayable, since foul-play's move cannot be re-derived. It is per-battle and the
    merged shard this function feeds has no per-battle rows at all, so lifting it
    would roughly double the merged shard for a copy of data sitting in the sibling
    file. Measured on eras 61-64: the merged shards' addresses are a strict subset of
    the bridge summaries' -- 0 of 656 distinct `(battle, round, seat)` are missing
    there -- so nothing is lost while both files are kept together.

    Delete or stop shipping the `-p1`/`-p2` summaries and every address in the merged
    shard silently becomes unreplayable again. Nothing enforces that today. If a
    harness ever needs the merged shard to stand alone, lift the journals here rather
    than dropping the siblings.
    """
    engine = summary.get("engine_mcts") or {}
    timing = summary.get("policy_timing") or {}
    return {
        "seat": seat,
        "games": summary.get("completed_games"),
        "complete": summary.get("complete"),
        "wins": summary.get("wins"),
        "win_rate": summary.get("win_rate"),
        "score": summary.get("score"),
        "score_rate": summary.get("score_rate"),
        "ties": summary.get("ties"),
        "capped_games": summary.get("capped_games"),
        # Both walls, deliberately. The first is the 20 s/turn gate's field; the
        # second includes non-searched decisions and reads LOW when fallback is
        # high, so reporting only it would hide a contaminated cell.
        "search_wall_per_searched_decision": engine.get(
            "search_wall_per_searched_decision"
        ),
        "wall_per_decision_mean": timing.get("average_elapsed_seconds"),
        "wall_per_decision_p95": timing.get("p95_elapsed_seconds"),
        "fallback_rate": engine.get("fallback_rate"),
        "fallback_reasons": engine.get("fallback_reasons"),
        "world_failure_reasons": (engine.get("policy_stats") or {}).get(
            "world_failure_reasons"
        ),
        "depth_reached_mean": (engine.get("policy_stats") or {}).get(
            "depth_reached_mean"
        ),
        "depth_reached_max": (engine.get("policy_stats") or {}).get("depth_reached_max"),
        "policy_stats": engine.get("policy_stats"),
        "seat_wall_s": summary.get("_seat_wall_s"),
    }


def build_parser() -> argparse.ArgumentParser:
    """The driver's CLI, exposed so tests can pin it.

    Kept separate from main() because every test here builds a Namespace by
    hand: without this, deleting or renaming an add_argument left the whole
    suite green while a real run died with AttributeError inside bridge_argv.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", choices=("search", "raw"), required=True)
    ap.add_argument("--seed-start", type=int, required=True,
                    help="first BATTLE seed of this shard's band")
    ap.add_argument("--pairs", type=int, required=True,
                    help="battle seeds in this shard (games = 2 x pairs, one per seat)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--sims", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--opponent-priors", action="store_true",
                    help="engine-mcts opponent-side model priors (cells B/E)")
    ap.add_argument("--checkpoint-tag", default=None,
                    help="explicit short label for this checkpoint (e.g. k0). Keeps cells "
                         "distinct when two checkpoints share a filename, which the "
                         "campaign copies do.")
    ap.add_argument("--foulplay-root", default=None,
                    help="foul-play checkout. Defaults to the bridge's repo-relative "
                         "third_party/foul-play, which is wrong for any image that "
                         "ships it elsewhere.")
    ap.add_argument("--foulplay-python", default=None,
                    help="foul-play venv interpreter. Defaults to "
                         "<foulplay-root>/.venv/bin/python.")
    ap.add_argument("--engine-model-path", default=None)
    ap.add_argument("--engine-tables-path", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-build-check", action="store_true",
                    help="offline/dry use only; never for a scored shard")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # REFUSED, deliberately: the MAPPING is fixed, the CRATE-SIDE GATHER is not.
    #
    # The opponent request order is computed by
    # `engine_search.opponent_request_order`, which reuses
    # `determinization._public_opponent_team_index_walk` -- the code that
    # already maintained that permutation while decoding recorded opponent
    # switch actions. Two independent review rounds measured it against the
    # live Showdown request order: 13,614 and 7,267 decisions, zero wrong,
    # ~0.7% fail-closed, with three known-wrong reference implementations
    # scoring 81-96% wrong on the identical rows (those controls are in
    # scripts/measure_opponent_request_order.py).
    #
    # What is still unverified is everything DOWNSTREAM of the map:
    #   - the crate's gather/apply path has zero Rust tests;
    #   - the flag-on native-slot pin cannot run without libtorch;
    #   - no in-image gate has confirmed APPLIED priors end to end against a
    #     real checkpoint.
    #
    # CLOSED since: when no order could be supplied the crate substituted a
    # one-swap approximation -- `wrong_one_swap`, one of the three controls
    # cited above at 81-96%, recorded at ~91% in its own docstring -- so the
    # ~0.7% Python fails closed on was fail-OPEN in the crate. The crate now refuses
    # instead -- all-`None` action map, node stays uniform, refusal counted in
    # `prior_fallbacks` -- so a withheld order costs a counted fallback rather
    # than a confident wrong prior.
    #
    # A wrong gather does not fail. It returns a confident paired delta and the
    # campaign reads "opponent priors do not help" off a permutation. Cells B
    # and E stay refused until the §8 in-image gate trio clears them.
    # --opponent-priors was REFUSED here from the switch-ordering fix until
    # 2026-08-12. The refusal is kept below as history rather than deleted,
    # because its reasons are the checklist that lifted it, and a future reader
    # deciding whether to trust a flag-on number needs to see what was required.
    #
    #   "The fix has not cleared independent review"
    #     -> Two adversarial review rounds on
    #        scott/opponent-prior-observability. Round one returned NOT SAFE and
    #        found six defects, two of them blockers introduced by that branch: a
    #        red native-call-contract guard, and a stale head fingerprint in the
    #        C155 register that failed CI. Round two verified every fix by
    #        measurement -- re-running both mutations itself -- and signed off.
    #
    #   "Nothing has run against a real checkpoint"
    #     -> Still true of THIS gate, and deliberately so: the evidence below is
    #        white-box and fixture-based. The first flag-on cell against a real
    #        checkpoint is the run this lift enables, and its applied-rate must
    #        be published with it.
    #
    #   "Four prior attempts each looked correct under their own tests"
    #     -> This is the one that mattered, and it is why the bar was a MUTATION
    #        rather than a passing suite. M9 (branch: opponent_prefix() ->
    #        self_prefix()) is killed by
    #        test_a_branch_that_switches_the_opponent_evolves_its_request_order:
    #        applying the mutation fails it deterministically while all eleven
    #        pre-existing tests in that file still pass. Review then found M9'
    #        -- the same defect one ply down, in the chaining arm, uncovered and
    #        biting at the --depth 4 this driver defaults to -- now killed by
    #        test_a_deeper_seam_chains_the_parent_branch_order_not_the_root.
    #        Both recorded KILLED in priors.rs's census, and pinned against
    #        deletion by tests/test_opponent_prior_fixture_pins.py, because no
    #        workflow builds the model feature and the kills were otherwise
    #        enforced by nothing.
    #
    #   "the section 8 in-image gate confirms applied priors end to end"
    #     -> scripts/prior_mapping_assert.py exits 0 on a REGENERATED corpus at
    #        fingerprint 3d8215d631d95edb: 782/782 random battery, 390/390
    #        scenarios, 0 diverged. Getting there fixed two real defects in the
    #        gate itself (it omitted the transformed_slots production passes, and
    #        two of four approximation flags) and refused 12 rows gen3 provably
    #        cannot express -- it has no Struggle arm in get_all_options -- under
    #        skip:engine_unsupported:struggle_only_surface. END TO END, measured
    #        in-image: flag off gives opponent_priors_applied 0 and a zero
    #        digest; flag on gives 335 applied, 0 refused, a 100% applied rate
    #        and digest 6ebae0c1fa3cb45e.
    #
    # WHAT IS STILL NOT GUARANTEED, so the lift does not read as more than it is:
    # the applied rate is BRANCH-level (RootPriorResolution is not
    # seat-attributed, so root refusals are invisible to it), and gen3 still
    # cannot express a Struggle-only surface. Any flag-on result must be
    # published WITH its applied rate -- a delta whose priors may not have
    # applied is the exact false conclusion this refusal existed to prevent.

    # HARD STOP before any game. A stale build does not error, it produces a
    # plausible number -- same standing as in mcts_acceptance_h2h.
    from engine_build_fingerprint import assert_fresh, compute_fingerprint  # noqa: PLC0415

    assert_fresh(skip=args.skip_build_check)
    engine_fingerprint = compute_fingerprint()["fingerprint"]
    print(f"engine fingerprint: {engine_fingerprint}", flush=True)

    from mcts_acceptance_h2h import (  # noqa: PLC0415
        _source_commit,
        assert_vocab_alignment,
        build_provenance,
        checkpoint_category_vocabulary,
    )
    from pokezero.local_showdown import (  # noqa: PLC0415
        LocalShowdownConfig,
        env_config_from_checkpoint_provenance,
    )
    from pokezero.neural_policy import (  # noqa: PLC0415
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        load_transformer_model_config,
        observation_spec_from_model_config,
    )

    config_id = config_id_for(args)
    model_config = load_transformer_model_config(args.checkpoint)
    env_config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(
            showdown_root=args.showdown_root,
            set_belief_source=True,
            category_vocab=checkpoint_category_vocabulary(model_config, args.showdown_root),
        ),
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=category_vocab_from_model_config(model_config, args.showdown_root),
        context="foulplay paired eval",
    )

    # Gate (c): root == checkpoint == leaf. The 'volatile:solarbeam' hashed-row
    # drift seen in the July-30 shard logs is exactly what this refuses.
    leaf_tables = args.engine_tables_path
    if args.arm != "raw" and leaf_tables is None:
        from pokezero.mcts_eval.lattice import materialize_search_artifacts  # noqa: PLC0415
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract  # noqa: PLC0415

        leaf_tables = materialize_search_artifacts(
            resolve_checkpoint_contract(
                args.checkpoint, model_device=args.device, showdown_root=args.showdown_root
            ),
            showdown_root=args.showdown_root,
        )["tables_path"]
    vocab_sha256 = assert_vocab_alignment(model_config, env_config, leaf_tables)
    provenance = build_provenance(
        args.checkpoint, config_id, args.arm,
        vocab_sha256=vocab_sha256, commit=_source_commit(),
    )

    started = time.perf_counter()
    summaries = {seat: run_seat(args, seat) for seat in SEATS}

    seats = {seat: seat_block(summaries[seat], seat) for seat in SEATS}
    rows = {seat: per_seed_outcomes(summaries[seat], seat) for seat in SEATS}
    expected = set(range(args.seed_start, args.seed_start + args.pairs))
    missing = {
        seat: sorted(expected - set(rows[seat])) for seat in SEATS
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "arm": args.arm,
        "config_id": config_id,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": summaries["p1"].get("checkpoint_sha256"),
        "engine_fingerprint": engine_fingerprint,
        "provenance_sha256": provenance,
        "commit": _source_commit(),
        "seed_start": args.seed_start,
        "pairs": args.pairs,
        "foulplay_search_time_ms": FOULPLAY_SEARCH_TIME_MS,
        "opponent_priors": bool(args.opponent_priors),
        # Named, never silently dropped: an incomplete seat makes the paired
        # delta unscoreable for those seeds and the merger must see it.
        "missing_seeds": missing,
        "per_seat": seats,
        "rows": [row for seat in SEATS for row in rows[seat].values()],
        "wall_s": round(time.perf_counter() - started, 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== SHARD COMPLETE ===")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
