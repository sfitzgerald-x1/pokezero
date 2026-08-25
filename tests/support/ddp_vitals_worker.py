"""Two-rank gloo worker: rank 1 is INACTIVE (the placeholder shard shape).

This is the shape that deadlocked: a rank-local condition around a collective. Rank 1 gets
local_examples=0 -> active=False; if the loss function skips any reduction on that path, rank 0
blocks forever and the harness times out instead of asserting.
"""
import json, os, sys

def main() -> int:
    import torch
    sys.path.insert(0, os.environ["REPO_SRC"])
    from pokezero.neural_policy import (
        DistributedTrainingContext, TransformerPolicyOutput, TransformerTrainingConfig,
        _distributed_transformer_loss,
    )
    rank = int(os.environ["RANK"])
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2)
    n = 4
    logits = torch.zeros(n, 9); logits[:, 0] = 1.5
    output = TransformerPolicyOutput(
        policy_logits=logits,
        value=torch.tensor([0.6, 0.1, 0.9, 0.3]),
        opponent_action_logits=torch.zeros(n, 9),
    )
    legal = torch.ones(n, 9, dtype=torch.bool)
    if rank == 0:
        # One recorded action that is ILLEGAL under the current mask: its logit is filled with
        # -1e9, so the raw k3 term is astronomically large and the per-sample clamp must bite.
        # This makes ppo_kl_outliers differ between the ranks, which is what turns the parity
        # comparison into a real test of that collective rather than 0 == 0.
        legal[0, 0] = False
        legal[0, 1] = True
    tensors = {
        "legal_action_mask": legal,
        "action_indices": torch.zeros(n, dtype=torch.long),
        "returns": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "action_probabilities": torch.full((n,), 1.0 / 9.0),
        "action_probability_mask": torch.ones(n, dtype=torch.bool),
        "opponent_action_mask": torch.zeros(n, dtype=torch.bool),
        "opponent_action_indices": torch.zeros(n, dtype=torch.long),
    }
    config = TransformerTrainingConfig(objective="ppo", normalize_advantage=False, opponent_action_loss_weight=0.0)
    # Rank 1 is the inactive placeholder: zero local examples.
    _, pieces = _distributed_transformer_loss(
        output, tensors, config,
        context=DistributedTrainingContext(rank=rank, world_size=2, local_rank=rank, backend="gloo"),
        local_examples=n if rank == 0 else 0,
        global_examples=n,
    )
    out = {k: pieces[k] for k in ("ppo_kl_sum", "ppo_kl_outliers", "value_ev_examples",
                                  "value_ev_target_sum", "value_ev_residual_square_sum")}
    print("RESULT " + json.dumps({"rank": rank, **out}), flush=True)
    torch.distributed.destroy_process_group()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
