#!/usr/bin/env python3
"""Rewrite a checkpoint's value-head width, so the Phase 3 V1 arm can be CREATED.

WHY THIS AND NOT A `train` FLAG. PR #1263 tried to add `--value-head-hidden` to `train` and was
closed: on a warm start `neural_cli.py` derives `model_config` wholly from the checkpoint, so the
flag was a silent no-op exactly where the arm needs it, and the validator tolerance it added was
reachable ONLY from lineage tools that cannot pass the flag -- where it replaced a correct
refusal with silent loss of trained head parameters. The premise was also wrong:
`load_transformer_checkpoint` runs BEFORE `_validate_initial_model_config` and keys off the
checkpoint's own config, so nothing in the validator blocks a widened warm start.

What actually works, and what this does: rewrite the config stamp, and let
`load_state_dict_allowing_fresh_value_head` (#1262) reinitialise the head on the next load. The
trunk carries over byte-identically; only the head starts fresh, which is the arm.

WHAT THIS REFUSES TO DO, because each was a real failure mode in this series:

  * It will not narrow a head. Going from a widened head back to the incumbent DISCARDS trained
    parameters (577 of them, measured), and the load path would accept it with a warning. A
    conversion tool is not the place to make that quiet.
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
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pokezero.neural_policy import (  # noqa: E402
    FreshValueHeadWarning,
    TransformerPolicyConfig,
    load_transformer_checkpoint,
    require_torch,
)

VALUE_HEAD_PREFIX = "value_head."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="source checkpoint (never modified)")
    ap.add_argument("--output", required=True, help="destination; refuses to overwrite the input")
    ap.add_argument("--value-head-hidden", type=int, required=True,
                    help="new hidden width. Must be >= TransformerPolicyConfig."
                         "VALUE_HEAD_HIDDEN_MIN; a width of 1 is a scalar reparameterisation of "
                         "one linear functional and is refused by the config itself.")
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

    payload = torch.load(src, map_location="cpu", weights_only=False)
    if "model_config" not in payload or "state_dict" not in payload:
        raise SystemExit(f"CANNOT RUN: {src} carries no model_config/state_dict")
    old_config = TransformerPolicyConfig.from_dict(payload["model_config"])
    old_width = old_config.value_head_hidden
    new_width = args.value_head_hidden

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

    # THE STALE HEAD TENSORS ARE LEFT IN PLACE, and that is deliberate -- my first version of
    # this tool stripped them and broke itself against my own safety net.
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
    # `training_result` carries its own copy of the config in these payloads; keep them in step.
    for holder in ("training_result", "result"):
        block = payload.get(holder)
        if isinstance(block, dict) and "model_config" in block:
            block["model_config"] = payload["model_config"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=str(dst.parent), delete=False)
    try:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, dst)
    except BaseException:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise

    print(f"value_head_hidden {old_width!r} -> {new_width!r}")
    print(f"  {len(head_tensors)} stale head tensor(s) left in place (ignored on load, and "
          f"required to distinguish this from a truncated write): {head_tensors}")
    print(f"  wrote {dst}")

    # VERIFY THE RESULT RELOADS WITH THE HEAD IT CLAIMS. A conversion that writes a checkpoint
    # nothing can read back is worse than no conversion, and this series has shipped exactly
    # that once already (an export whose own parity check passed on a wrong architecture).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, result = load_transformer_checkpoint(dst)
    fresh = [w for w in caught if issubclass(w.category, FreshValueHeadWarning)]
    head_type = type(model.value_head).__name__
    expected = "Sequential" if new_width else "Linear"
    if result.model_config.value_head_hidden != (new_width or None) or head_type != expected:
        raise SystemExit(
            f"CONVERSION VERIFICATION FAILED: reloaded head is {head_type} with width "
            f"{result.model_config.value_head_hidden!r}, expected {expected} / "
            f"{(new_width or None)!r}. {dst} is not usable; delete it."
        )
    if not fresh:
        raise SystemExit(
            "CONVERSION VERIFICATION FAILED: the reload did not report a fresh value head, so "
            "the head was NOT reinitialised and this checkpoint's weights do not match its "
            "config claim. That is the defect this tool exists to prevent."
        )
    print(f"  verified: reloads as {head_type}, width "
          f"{result.model_config.value_head_hidden!r}, head reported UNTRAINED "
          f"({len(fresh)} warning)")
    print("  the trunk is carried over; only the value head starts fresh. That is the V1 arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
