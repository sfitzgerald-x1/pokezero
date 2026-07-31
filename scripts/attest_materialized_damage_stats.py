#!/usr/bin/env python
"""Replay current boundaries and audit only ``BattleSpec`` -> native transport.

This intentionally shares the differential's deterministic action selection,
world-construction, and strict matcher paths.  It is not a second simulator:
the result can clear only the adapter transport seam for a boundary that still
diverges on the current build.  Belief-world derivation and native damage
arithmetic remain deliberately outside its evidence boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from engine_build_fingerprint import assert_fresh  # noqa: E402
from engine_transition_differential import (  # noqa: E402
    _checkpoint_provenance,
    _fold,
    _prepare_boundary,
    _true_teams_from_bridge_snapshot,
    evaluate_boundary_strict,
    observed_boost_deltas,
    unpack_team,
)
from pokezero.dex import load_showdown_dex  # noqa: E402
from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy  # noqa: E402
from pokezero.engine_stat_attestation import attest_battle_spec_transport_variants  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.randbat import Gen3RandbatSource  # noqa: E402


def _target(value: str) -> tuple[int, int]:
    seed, separator, step = value.partition("/")
    if not separator or not seed.isdigit() or not step.isdigit():
        raise argparse.ArgumentTypeError("targets must be SEED/STEP")
    return int(seed), int(step)


def _hash_source_files(repo_root: Path, paths: Sequence[Path]) -> str:
    """Hash source paths and bytes with stable repository-relative names."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(repo_root))):
        relative = path.relative_to(repo_root).as_posix()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _public_source_provenance(
    repo_root: Path,
    *,
    source_commit: str,
) -> dict[str, object]:
    """Bind the recorded commit to a clean checkout or an explicit source hash."""

    git_metadata = repo_root / ".git"
    if not git_metadata.exists():
        roots = (
            repo_root / "src" / "pokezero",
            repo_root / "scripts",
        )
        paths = [
            path
            for root in roots
            if root.exists()
            for path in root.rglob("*.py")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        pyproject = repo_root / "pyproject.toml"
        if pyproject.is_file():
            paths.append(pyproject)
        if not paths:
            raise RuntimeError(
                "could not resolve executable public source files for provenance"
            )
        return {
            "public_source_tree_status": "explicit_hash_without_git",
            "public_source_tree_sha256": _hash_source_files(repo_root, paths),
            "public_source_tree_hash_scope": "src/pokezero/**/*.py;scripts/**/*.py;pyproject.toml",
        }
    try:
        head = subprocess.run(
            ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        if source_commit != head:
            raise RuntimeError(
                "recorded public source commit does not match the executed checkout "
                f"(recorded {source_commit}, checkout {head})"
            )
        result = subprocess.run(
            (
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "could not verify the tracked public source tree before attestation"
        ) from error
    dirty = result.stdout.rstrip("\n")
    if dirty.strip():
        paths = ", ".join(
            line[3:].strip() for line in dirty.splitlines()[:8] if len(line) > 3
        )
        raise RuntimeError(
            "refusing to write a transport attestation from a dirty tracked public "
            f"source tree ({paths or 'tracked changes detected'}); commit or revert "
            "the tracked changes so source_commit is reproducible"
        )
    tracked = subprocess.run(
        ("git", "-C", str(repo_root), "ls-files", "-z"),
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    paths = [
        repo_root / value.decode("utf-8")
        for value in tracked
        if value and (repo_root / value.decode("utf-8")).is_file()
    ]
    return {
        "public_source_tree_status": "clean_tracked_checkout",
        "public_source_tree_sha256": _hash_source_files(repo_root, paths),
        "public_source_tree_hash_scope": "git_ls_files",
        "public_source_checkout_head": head,
    }


def _resolved_showdown_provenance(source: Gen3RandbatSource) -> dict[str, object]:
    """Record the exact content-resolved randbat/Showdown source identity."""

    metadata = source.metadata.to_payload()
    source_hash = str(metadata.get("source_hash") or "")
    if not source_hash:
        raise RuntimeError(
            "refusing to write a transport attestation without a resolved "
            "Showdown/randbat source hash"
        )
    return {
        "showdown_randbat_source_hash": source_hash,
        "showdown_randbat_source": metadata,
    }


def attest_target(
    *,
    env: LocalShowdownEnv,
    policy: EngineMctsPolicy,
    dex: Any,
    seed: int,
    target_step: int,
    max_steps: int,
) -> dict[str, object]:
    """Recreate one target and attest it only when it still strictly diverges."""

    env.reset(seed=seed, format_id="gen3randombattle")
    true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
    packed = {slot: true_teams[slot]["packed"] for slot in ("p1", "p2")}
    override = BattleStartOverride(player_teams=packed)
    teams = {slot: unpack_team(packed[slot]) for slot in ("p1", "p2")}
    rng = random.Random(seed ^ 0x5EED)
    cumulative = list(env.protocol_lines)

    for step in range(1, max_steps + 1):
        if env.terminal() is not None:
            return {
                "seed": seed,
                "step": target_step,
                "status": "terminal_before_target",
                "terminal_at_or_before_step": step - 1,
            }
        requested = tuple(env.requested_players())
        actions: dict[str, int] = {}
        for player in requested:
            legal = [index for index, allowed in enumerate(env.legal_actions(player)) if allowed]
            if not legal:
                return {"seed": seed, "step": target_step, "status": "no_legal_action"}
            actions[player] = rng.choice(legal)

        prepared = None
        if set(requested) == {"p1", "p2"}:
            prepared = _prepare_boundary(
                env=env,
                flags_policy=policy,
                override=override,
                teams=teams,
                dex=dex,
                actions=actions,
                cumulative=cumulative,
                counts=Counter(),
                approximate_sleep=False,
                hidden_counter_support=True,
            )
        before = len(cumulative)
        env.step(actions)
        step_lines = tuple(str(line) for line in env.protocol_lines[before:])
        cumulative.extend(step_lines)
        if step != target_step:
            continue
        if prepared is None:
            return {
                "seed": seed,
                "step": target_step,
                "status": "unmaterializable_target",
                "requested_players": list(requested),
            }

        transport = attest_battle_spec_transport_variants(
            prepared["specs"],
            prepared["states"],
            variant_construction=prepared.get("variant_construction") or (),
        )
        if transport["status"] == "dropped_variant_construction":
            return {
                "seed": seed,
                "step": target_step,
                "status": "dropped_variant_construction",
                "turn": prepared["turn"],
                "gating": prepared["gating"],
                "hidden_counter_candidate_worlds": (
                    len(transport["variant_construction"])
                    if prepared["gating"] == "support"
                    else 0
                ),
                **{key: value for key, value in transport.items() if key != "status"},
            }

        observed_fold = _fold(cumulative)
        observed = type(prepared["pre_features"])(
            p1_hp=observed_fold.p1_hp,
            p2_hp=observed_fold.p2_hp,
            p1_status=observed_fold.p1_status,
            p2_status=observed_fold.p2_status,
            fainted=_fold(step_lines).fainted,
            weather=observed_fold.weather,
            side_conditions=observed_fold.side_conditions,
        )
        active_changed = {
            slot: any(
                line.startswith((f"|switch|{slot}a", f"|drag|{slot}a", f"|replace|{slot}a"))
                for line in step_lines
            )
            for slot in ("p1", "p2")
        }
        verdict, misses, branch_count = evaluate_boundary_strict(
            states=prepared["states"],
            slot_sides=prepared["slot_sides"],
            choices=prepared["choices"],
            party_display=prepared["party_display"],
            turn=prepared["turn"],
            pre_features=prepared["pre_features"],
            observed=observed,
            step_lines=step_lines,
            observed_boosts=observed_boost_deltas(step_lines),
            active_changed=active_changed,
            counts=Counter(),
        )
        transport_matches = transport["status"] == "transport_attested"
        return {
            "seed": seed,
            "step": target_step,
            "status": (
                "target_diverged_transport_attested"
                if verdict == "diverged" and transport_matches
                else "target_not_eligible_for_transport_clearance"
            ),
            "turn": prepared["turn"],
            "gating": prepared["gating"],
            "boundary_verdict": verdict,
            "branch_count": branch_count,
            "branch_misses": misses[:12],
            "hidden_counter_candidate_worlds": (
                len(transport["variant_construction"])
                if prepared["gating"] == "support"
                else 0
            ),
            **{key: value for key, value in transport.items() if key != "status"},
        }
    return {"seed": seed, "step": target_step, "status": "max_steps_exceeded"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", default=DEFAULT_SHOWDOWN_ROOT)
    parser.add_argument("--target", type=_target, action="append", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--json", type=Path)
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(effective_argv)
    assert_fresh()
    provenance = _checkpoint_provenance()
    if not provenance.get("source_commit") or not provenance.get("engine_fingerprint"):
        raise RuntimeError(
            "refusing to write a transport attestation without source commit and "
            "current engine-fingerprint provenance"
        )

    source_tree = _public_source_provenance(
        REPO_ROOT,
        source_commit=str(provenance["source_commit"]),
    )
    dex = load_showdown_dex(args.showdown_root)
    set_source = Gen3RandbatSource.from_showdown_root(args.showdown_root)
    provenance = {
        **provenance,
        **source_tree,
        **_resolved_showdown_provenance(set_source),
    }
    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True))
    policy = EngineMctsPolicy(
        dex=dex,
        set_source=set_source,
        config=EngineMctsConfig(worlds=1, search_time_ms=1),
    )
    try:
        rows = [
            attest_target(
                env=env,
                policy=policy,
                dex=dex,
                seed=seed,
                target_step=step,
                max_steps=args.max_steps,
            )
            for seed, step in args.target
        ]
    finally:
        env.close()
    payload = {
        "schema_version": "pokezero.battle_spec_transport_attestation.v3",
        "command": shlex.join(
            [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *effective_argv]
        ),
        "provenance": provenance,
        "scope": "BattleSpec_to_native_State_transport_only",
        "does_not_attest": [
            "belief_world_to_BattleSpec_derivation",
            "native_branch_generation",
            "native_Gen3_damage_arithmetic",
        ],
        "targets": rows,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all(row["status"] == "target_diverged_transport_attested" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
