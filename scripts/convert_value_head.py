#!/usr/bin/env python3
"""Rewrite a checkpoint's value-head width, so the Phase 3 V1 arm can be CREATED.

WHY THIS AND NOT A `train` FLAG. PR #1263 tried to add `--value-head-hidden` to `train` and was
closed: on a warm start `neural_cli.py` derives `model_config` wholly from the checkpoint, so the
flag was a silent no-op exactly where the arm needs it, and the validator tolerance it added was
reachable ONLY from lineage tools that cannot pass the flag -- where it replaced a correct
refusal with silent loss of trained head parameters. The premise was also wrong:
`load_transformer_checkpoint` runs BEFORE `_validate_initial_model_config` and keys off the
checkpoint's own config, so nothing in the validator blocks a widened warm start.

What actually works, and what this does: rewrite the config stamp.  Its legacy mode lets
`load_state_dict_allowing_fresh_value_head` (#1262) reinitialise the head on the next load. The
Phase-3 mode additionally accepts `--head-init-seed` and materialises those new tensors into the
checkpoint itself.  This matters because a warm-start loader runs before training's DDP seed is
installed; a seed recorded only in the trainer would otherwise not identify the V1 initial head.
In both cases the trunk carries over byte-identically; only the head starts fresh.

WHAT THIS REFUSES TO DO, because each was a real failure mode in this series:

  * It will not narrow a head without `--force-narrow`. Narrowing discards the widened head's
    trained parameters -- `514h + 1`, i.e. 4,113 / 8,225 / 16,449 / 32,897 at h = 8/16/32/64 on a
    512-dim trunk -- and the load path would accept it with only a warning. (An earlier version of
    this docstring said "577 of them", which is not the size of any head in this repo: the
    incumbent is 513 at emb=512 and 129 at emb=128.)
  * It will not report success on a checkpoint that does not reload with the head it claims. It
    VERIFIES the result rather than assuming it, because this series has already shipped an
    export whose own parity check passed on a wrong architecture. (Note it does NOT strip the
    stale head tensors -- an earlier version of this docstring said it did, and the code that
    matched broke against `load_state_dict_allowing_fresh_value_head`'s truncated-write guard.
    See the comment at the strip site for why leaving them is both necessary and correct.)
  * It will not overwrite its input, and it writes atomically (temp + fsync + os.replace), so an
    interrupted run cannot leave a half-written checkpoint that loads.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pokezero.neural_policy import (  # noqa: E402
    EntityTokenTransformerPolicy,
    TransformerPolicyConfig,
    load_transformer_checkpoint,
    load_state_dict_allowing_fresh_value_head,
    require_torch,
)

VALUE_HEAD_PREFIX = "value_head."


def _materialize_seeded_value_head(torch, config, state_dict, *, seed: int) -> tuple[dict, set[str]]:
    """Build the new head under one explicit CPU seed and copy the old trunk exactly.

    The normal V1 transition is a linear source head to a two-layer MLP, so the shared loader
    leaves all four MLP tensors fresh while loading every non-head tensor from the source.  We
    persist that initialized model rather than relying on a later warm-start process to make a
    fresh, ambient-RNG draw.  Keeping the RNG fork local also makes conversion repeatable without
    perturbing an embedding caller's randomness.
    """
    with torch.random.fork_rng(devices=[]):
        # `torch.manual_seed` also mutates accelerator generators, while this CPU-only
        # construction needs only the default CPU generator.  The surrounding fork restores
        # that generator, so embedding callers retain their CPU and accelerator RNG streams.
        torch.random.default_generator.manual_seed(seed)
        model = EntityTokenTransformerPolicy(config)
    fresh = set(load_state_dict_allowing_fresh_value_head(model, state_dict))
    head_params = {f"{VALUE_HEAD_PREFIX}{name}" for name in model.value_head.state_dict()}
    if fresh != head_params:
        raise SystemExit(
            "REFUSING: seeded materialization did not replace the entire value head. Expected "
            f"{sorted(head_params)}, got {sorted(fresh)}. A V1 initializer may not mix old and "
            "new head tensors.")
    return dict(model.state_dict()), head_params


def _publish_verified_candidate(candidate: Path, destination: Path, *, force: bool) -> None:
    """Publish a verified candidate without weakening the no-clobber contract.

    ``os.replace`` is correct only for the explicit ``--force`` case.  A predicate such as
    ``destination.exists()`` cannot reserve the no-force name: another writer may create it
    while the private candidate is being verified.  A hard-link creation in the same directory
    is atomic and fails with ``FileExistsError`` instead of replacing that late writer.  The
    temporary and destination share a directory specifically so this is the same filesystem.
    """
    if force:
        os.replace(candidate, destination)
        return
    try:
        os.link(candidate, destination)
    except FileExistsError as exc:
        raise SystemExit(
            f"REFUSING: {destination} appeared while the candidate was being verified; "
            "a no-force conversion never overwrites another writer's output."
        ) from exc
    candidate.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="source checkpoint (never modified)")
    ap.add_argument("--output", required=True, help="destination; refuses to overwrite the input")
    ap.add_argument("--value-head-hidden", type=int, required=True,
                    help="new hidden width. Must be >= TransformerPolicyConfig."
                         "VALUE_HEAD_HIDDEN_MIN; a width of 1 is a scalar reparameterisation of "
                         "one linear functional and is refused by the config itself.")
    ap.add_argument("--head-init-seed", type=int, default=None,
                    help="materialize a wholly fresh widened V1 head under this non-negative "
                         "seed. Required by provenance-bound V1 training; without it, legacy "
                         "conversion defers fresh initialization to the next checkpoint load.")
    ap.add_argument("--force", action="store_true",
                    help="replace --output if it already exists")
    ap.add_argument("--force-narrow", action="store_true",
                    help="permit a conversion that DISCARDS trained head parameters. Off by "
                         "default: narrowing throws away work the load path would accept with "
                         "only a warning.")
    args = ap.parse_args()

    torch = require_torch()
    src, dst = Path(args.checkpoint), Path(args.output)
    if not src.exists():
        raise SystemExit(f"CANNOT RUN: {src} does not exist")
    if dst.resolve() == src.resolve():
        raise SystemExit(
            "REFUSING: --output is the same file as --checkpoint. A conversion that overwrites "
            "its own input leaves no way back if the result is wrong."
        )

    # `weights_only=True`: an earlier version of this script was the ONLY checkpoint reader in the
    # repo opting out of the safe unpickler, for a job that needs nothing from it -- arbitrary code
    # execution on a checkpoint from a shared volume. `load_transformer_checkpoint` reads the same
    # payload safely, and `convert_region_trim.py:100` already set the precedent.
    if dst.exists() and not args.force:
        raise SystemExit(
            f"REFUSING: {dst} already exists. Overwriting a different trained checkpoint is the "
            "same hazard the input-overwrite refusal names. Pass --force to replace it."
        )
    payload = torch.load(src, map_location="cpu", weights_only=True)
    if "model_config" not in payload or "state_dict" not in payload:
        raise SystemExit(f"CANNOT RUN: {src} carries no model_config/state_dict")
    # Gate the schema BEFORE writing. Previously the output was written first and the
    # verification then died on a raw traceback, leaving an unverified artifact on disk with a
    # widened stamp. `convert_region_trim.py:101` refuses first; so does this now.
    from pokezero.neural_policy import NEURAL_POLICY_SCHEMA_VERSION  # noqa: PLC0415

    if payload.get("schema_version") != NEURAL_POLICY_SCHEMA_VERSION:
        raise SystemExit(
            f"REFUSING: {src} declares schema {payload.get('schema_version')!r}, not "
            f"{NEURAL_POLICY_SCHEMA_VERSION!r}. Converting an unsupported schema would write an "
            "artifact this tool cannot verify."
        )
    old_config = TransformerPolicyConfig.from_dict(payload["model_config"])
    old_width = old_config.value_head_hidden
    new_width = args.value_head_hidden

    if args.head_init_seed is not None and args.head_init_seed < 0:
        raise SystemExit("REFUSING: --head-init-seed must be a non-negative integer")

    if old_width == new_width or (not old_width and not new_width):
        raise SystemExit(
            f"NOTHING TO DO: the checkpoint's value_head_hidden is already {old_width!r}."
        )
    widening = (old_width or 0) < (new_width or 0)
    if not widening and not args.force_narrow:
        raise SystemExit(
            f"REFUSING: {old_width!r} -> {new_width!r} NARROWS the value head, which discards "
            "its trained parameters. The load path would accept this with only a warning, which "
            "is why the refusal lives here. Pass --force-narrow if that is genuinely intended."
        )
    if args.head_init_seed is not None and old_width:
        raise SystemExit(
            "REFUSING: --head-init-seed is only defined for the incumbent linear-to-MLP V1 "
            "transition. A width-to-width conversion can retain compatible tensors; make that "
            "a separately reviewed initializer rather than claiming a wholly fresh head.")

    # IN LEGACY MODE, THE STALE HEAD TENSORS ARE LEFT IN PLACE, and that is deliberate -- my
    # first version of this tool stripped them and broke itself against my own safety net.  The
    # seeded V1 mode below deliberately replaces all of them with the materialized MLP head.
    #
    # Stripping looked right: "the config claim and the weights must not disagree". But a
    # checkpoint with NO `value_head.` tensor is indistinguishable from a truncated write, which
    # is exactly what `load_state_dict_allowing_fresh_value_head` refuses -- so the converted
    # file would not load at all:
    #
    #   REFUSING to load: missing=['value_head.0.weight', ...] unexpected=[]
    #   ... the checkpoint here carries value-head tensors: False
    #
    # Leaving them is also correct on the merits, and the distinction from #1263 matters. There,
    # a config claiming a widened head while TRAINING an incumbent one produced an artifact whose
    # weights and config disagreed after the fact. Here the file is an INPUT: on load the head is
    # reinitialised to the shape the config claims, via the rename tolerance, and a warning says
    # so. The stale tensors are never used; the claim is honoured at load time. The verification
    # at the end of this function is what proves that rather than assuming it.
    head_tensors = sorted(k for k in payload["state_dict"] if k.startswith(VALUE_HEAD_PREFIX))
    if not head_tensors:
        raise SystemExit(
            f"REFUSING: {src} carries no {VALUE_HEAD_PREFIX}* tensor, so it is indistinguishable "
            "from a truncated write and the converted file would not load. Convert a checkpoint "
            "that has a value head."
        )
    payload["model_config"] = dataclasses.replace(
        old_config, value_head_hidden=(new_width or None)).to_dict()
    materialized_head_params: set[str] | None = None
    if args.head_init_seed is not None:
        materialized, materialized_head_params = _materialize_seeded_value_head(
            torch, TransformerPolicyConfig.from_dict(payload["model_config"]),
            payload["state_dict"], seed=args.head_init_seed)
        payload["state_dict"] = materialized
        payload["phase3_value_head_initialization"] = {
            "schema": "pokezero.phase3.value-head-initialization.v1",
            "kind": "linear-to-mlp-fresh",
            "seed": args.head_init_seed,
            "head_parameter_names": sorted(materialized_head_params),
        }
    # NOTE: no `training_result`/`result` sync here, deliberately. An earlier version had one
    # and both it and the PR body claimed it mattered; `save_transformer_checkpoint` writes
    # `model_config` at the top level only, and no checkpoint in this repo carries such a key, so
    # the loop was unreachable and the claim was false.

    dst.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=str(dst.parent), delete=False)
    candidate = Path(handle.name)
    try:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        # 0644, matching `save_transformer_checkpoint` and `convert_region_trim`.
        # `NamedTemporaryFile` creates 0600 and `os.replace` preserves it, so a converted
        # checkpoint read by a different UID in the cluster failed to open.
        os.chmod(handle.name, 0o644)
    except BaseException:
        handle.close()
        candidate.unlink(missing_ok=True)
        raise

    # VERIFY THE RESULT RELOADS WITH THE HEAD IT CLAIMS. A conversion that writes a checkpoint
    # nothing can read back is worse than no conversion, and this series has shipped exactly
    # that once already (an export whose own parity check passed on a wrong architecture).
    # Load the private candidate through the SAME loader that production consumers use, then
    # independently take the reinitialised set from the shared load primitive.  This must happen
    # BEFORE publication:
    # `--force` is permission to replace an existing verified output only with another verified
    # output, never to delete it when a malformed source fails the candidate reload.
    #
    # An earlier version parsed tensor names out of the warning message text -- and that message
    # contains the output PATH, so `--output ".../value_head.2.bias/out.pt"` injected a name into
    # the set and the tool reported a width sweep as "all 4 head tensors are FRESH" while
    # `value_head.2.bias` was carried from the stale head. That is the mixed-provenance defect
    # this block exists to catch, reachable by choosing a directory name. Parsing prose for a
    # value the callee already returns was the whole mistake.
    try:
        # The checkpoint's model_config/state_dict are not its complete consumer contract:
        # `load_transformer_checkpoint` also requires training_config and is the production
        # boundary V1 must actually cross.  Do not certify a hand-assembled partial payload just
        # because its tensors instantiate in this script.
        loaded_candidate, _ = load_transformer_checkpoint(candidate, map_location="cpu")
        payload_back = torch.load(candidate, map_location="cpu", weights_only=True)
        config_back = TransformerPolicyConfig.from_dict(payload_back["model_config"])
        model = EntityTokenTransformerPolicy(config_back)
        reinit = set(load_state_dict_allowing_fresh_value_head(
            model, payload_back["state_dict"]))
    except BaseException:
        # The reload is exactly what this block exists to test.  It is still private here, so a
        # malformed source cannot delete or corrupt a pre-existing `--force` destination.
        candidate.unlink(missing_ok=True)
        raise
    head_type = type(model.value_head).__name__
    expected = "Sequential" if new_width else "Linear"
    if config_back.value_head_hidden != (new_width or None) or head_type != expected:
        candidate.unlink(missing_ok=True)
        raise SystemExit(
            f"CONVERSION VERIFICATION FAILED: reloaded head is {head_type} with width "
            f"{config_back.value_head_hidden!r}, expected {expected} / "
            f"{(new_width or None)!r}. The unpublished candidate has been removed."
        )

    head_params = {f"value_head.{n}" for n in dict(model.value_head.state_dict())}
    carried = head_params - reinit
    if args.head_init_seed is not None:
        if reinit:
            candidate.unlink(missing_ok=True)
            raise SystemExit(
                "CONVERSION VERIFICATION FAILED: a seeded materialized head was reinitialised "
                f"on reload ({sorted(reinit)}). The output does not contain the claimed V1 "
                "initialization bytes.")
        if materialized_head_params != head_params:
            candidate.unlink(missing_ok=True)
            raise SystemExit(
                "CONVERSION VERIFICATION FAILED: the materialized head parameter inventory "
                "changed before reload. The unpublished candidate has been removed.")
        loaded_head = {f"{VALUE_HEAD_PREFIX}{name}": tensor
                       for name, tensor in loaded_candidate.value_head.state_dict().items()}
        expected_head = {f"{VALUE_HEAD_PREFIX}{name}": tensor
                         for name, tensor in model.value_head.state_dict().items()}
        if set(loaded_head) != set(expected_head) or any(
                not torch.equal(loaded_head[name], expected_head[name]) for name in expected_head):
            candidate.unlink(missing_ok=True)
            raise SystemExit(
                "CONVERSION VERIFICATION FAILED: seeded value-head tensors changed across the "
                "production reload. The unpublished candidate has been removed.")
        print(f"  verified: reloads as {head_type}, width {config_back.value_head_hidden!r}; "
              f"all {len(head_params)} head tensors are FRESH and materialized under seed "
              f"{args.head_init_seed}")
    elif carried:
        # Carried-over tensors are only legitimate if they came from a head of the SAME shape.
        source_head = {k: v for k, v in payload["state_dict"].items()
                       if k.startswith(VALUE_HEAD_PREFIX)}
        live = {f"value_head.{n}": t for n, t in model.value_head.state_dict().items()}
        mismatched = sorted(
            n for n in carried
            if n not in source_head or tuple(source_head[n].shape) != tuple(live[n].shape))
        if mismatched:
            candidate.unlink(missing_ok=True)
            raise SystemExit(
                f"CONVERSION VERIFICATION FAILED: head tensors {mismatched} were neither "
                "reinitialised nor carried from a same-shape source, so this checkpoint's head "
                f"has MIXED provenance while claiming width {new_width}. The unpublished "
                "candidate has been removed."
            )
        if len(carried) == len(head_params):
            print(f"  verified: reloads as {head_type}, width "
                  f"{config_back.value_head_hidden!r}; the ENTIRE head carried over from "
                  "a same-shape source, so nothing is untrained")
        else:
            print(f"  verified: reloads as {head_type}, width "
                  f"{config_back.value_head_hidden!r}; {len(reinit)} of "
                  f"{len(head_params)} head tensors are FRESH, and the rest "
                  f"({sorted(carried)}) carried from a same-shape source")
    else:
        print(f"  verified: reloads as {head_type}, width "
              f"{config_back.value_head_hidden!r}; all {len(head_params)} head tensors "
              "are FRESH")
    if carried:
        print("  the trunk is carried over. NOTE the head is NOT wholly fresh -- see the line "
              "above; a width sweep carries any same-shape tensor, so this is not a clean V1 "
              "arm unless you intended the partial carry-over.")
    else:
        print("  the trunk is carried over; only the value head starts fresh. That is the V1 arm.")
    try:
        _publish_verified_candidate(candidate, dst, force=args.force)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    print(f"value_head_hidden {old_width!r} -> {new_width!r}")
    if args.head_init_seed is None:
        print(f"  {len(head_tensors)} stale head tensor(s) left in place (ignored on load, and "
              f"required to distinguish this from a truncated write): {head_tensors}")
    else:
        print(f"  replaced {len(head_tensors)} incumbent head tensor(s) with the verified "
              f"seeded MLP initialization: {head_tensors}")
    print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
