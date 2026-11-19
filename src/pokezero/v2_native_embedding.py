"""Opt-in first-order embedding gradient kept opaque to compiler decomposition.

Imported only for explicit native-backward execution. The forward preserves
the existing embedding parameter/module; the backward invokes ATen's native
dense implementation. No global decomposition table is changed.
"""
import torch


@torch.library.custom_op('v2_runtime::embedding_dense_gradient', mutates_args=())
def embedding_dense_gradient(grad: torch.Tensor, indices: torch.Tensor,
                             num_weights: int, padding_idx: int) -> torch.Tensor:
    return torch.ops.aten.embedding_dense_backward.default(
        grad, indices, num_weights, padding_idx, False)


@embedding_dense_gradient.register_fake
def _fake_gradient(grad, indices, num_weights, padding_idx):
    return grad.new_empty((num_weights, grad.shape[-1]))


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices, padding_idx):
        ctx.save_for_backward(indices)
        ctx.num_weights = weight.shape[0]
        ctx.padding_idx = padding_idx
        return torch.nn.functional.embedding(indices, weight, padding_idx)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return embedding_dense_gradient(grad, indices, ctx.num_weights, ctx.padding_idx), None, None


def validate(module):
    if (type(module) is not torch.nn.Embedding or module.weight.dtype != torch.float32 or module.padding_idx != 0
            or module.max_norm is not None or module.scale_grad_by_freq or module.sparse):
        raise ValueError('native backward requires FP32 dense category embedding with padding0,no max_norm/frequency scaling')


def lookup(module, indices):
    # Attributes are checked once by Trainer, outside the compiled update.
    return _Embedding.apply(module.weight, indices, 0)
