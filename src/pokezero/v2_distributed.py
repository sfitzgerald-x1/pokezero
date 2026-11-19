"""Canonical global batches over disjoint V2 collector shards.

Rows are exchanged once per round, not once per optimizer step. Every original
construction block remains a segment. Each global batch is split contiguously
across ranks, including zero-weight tails. Masked-loss denominators and PPO
advantage moments are global; gradients are summed before clipping/AdamW.
This is different from averaging independently normalized rank-local losses.
"""
from dataclasses import dataclass, replace
import hashlib

import numpy as np

from . import v2_trainer as v2


def recipe_train(reference, recipe):
    """Explicit S0 recipe package; preserve every other direct-reference control."""
    recipes = dict(R5=(512, 5), R2=(2048, 2), R1=(4096, 1))
    if recipe not in recipes or reference.batch_size != 512 or reference.epochs != 5:
        raise ValueError('R5/512x5 reference and explicit R5/R2/R1 recipe required')
    batch, epochs = recipes[recipe]
    return replace(reference, batch_size=batch, epochs=epochs,
        learning_rate=reference.learning_rate * batch / 512)


def shard_bounds(games, rank, world_size):
    if (type(games) is not int or games <= 0 or type(world_size) is not int or world_size <= 0
            or games % world_size or type(rank) is not int or not 0 <= rank < world_size):
        raise ValueError('positive game count divisible into nonempty rank shards required')
    return games * rank // world_size, games * (rank + 1) // world_size


@dataclass(frozen=True)
class Layout:
    destinations: np.ndarray  # rank for each input row (input ranks concatenated)
    slots: np.ndarray         # padded local output slot for each input row
    counts: tuple            # true global row count in each optimizer batch
    segments: tuple          # padded local segment lengths
    digest: str              # canonical keys, independent of collection partition
    local_batch: int


def canonical_layout(keys, batches, global_batch, world_size):
    """Address rows by (block, frame, row-in-block, owner), never collection order."""
    if (type(global_batch) is not int or type(world_size) is not int or world_size < 1
            or global_batch <= 0 or global_batch % world_size):
        raise ValueError('global optimizer batch must divide exactly across ranks')
    keys = np.asarray(keys, dtype=np.int64)
    if keys.ndim != 2 or keys.shape[1] != 3 or not len(keys):
        raise ValueError('nonempty (stream, frame, owner) rows required')
    position = {}
    for block, streams in enumerate(batches):
        for row, stream in enumerate(streams):
            if stream in position:
                raise ValueError('duplicate declared game stream')
            position[int(stream)] = (block, row)
    if any(int(s) not in position for s in keys[:, 0]):
        raise ValueError('row outside the declared global roster')
    if (keys[:, 1] < 0).any() or not np.isin(keys[:, 2], [0, 1]).all():
        raise ValueError('nonnegative frame and original owner required')
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('duplicate global training row')
    block = np.asarray([position[int(s)][0] for s in keys[:, 0]], np.int64)
    row = np.asarray([position[int(s)][1] for s in keys[:, 0]], np.int64)
    order = np.lexsort((keys[:, 2], row, keys[:, 1], block))
    local = global_batch // world_size
    destinations = np.empty(len(keys), np.int64); slots = np.empty(len(keys), np.int64)
    counts = []; segments = []; offset = 0; step = 0
    for n in np.bincount(block, minlength=len(batches)):
        n = int(n)
        if not n:
            continue
        within = np.arange(n)
        selected = order[offset:offset + n]
        destinations[selected] = (within % global_batch) // local
        slots[selected] = (step + within // global_batch) * local + within % local
        batches_in_segment = (n + global_batch - 1) // global_batch
        counts.extend(min(global_batch, n - i * global_batch) for i in range(batches_in_segment))
        segments.append(batches_in_segment * local)
        offset += n; step += batches_in_segment
    digest = hashlib.sha256(keys[order].astype('<i8').tobytes()).hexdigest()
    return Layout(destinations, slots, tuple(counts), tuple(segments), digest, local)


def gather(value, runtime):
    if runtime.world_size == 1:
        return [value]
    import torch.distributed as dist
    values = [None] * runtime.world_size
    dist.all_gather_object(values, value, group=runtime.process_group)
    return values


def distribute_store(store, keys, batches, runtime):
    """All-to-all real rows, then pad each rank's slice of each global batch.

    Only compact row addresses/statuses are replicated on the host. Feature
    tensors stay on the devices and are exchanged one column at a time. The
    input store is consumed to avoid holding two full copies after exchange.
    """
    torch = v2._torch()
    if runtime.v2_config.order != 'ordered' or not runtime.v2_config.distributed_rows:
        raise ValueError('explicit canonical distributed file order required')
    keys = np.asarray(keys, dtype=np.int64)
    if len(keys) != store.rows:
        raise ValueError('one original address per local training row required')
    status = store.terminal_status or ('UNKNOWN',) * store.rows
    metadata = dict(keys=keys, complete=np.asarray([s == 'COMPLETE' for s in status], bool), store_id=store.store_id,
        fields={k: (tuple(t.shape[1:]), str(t.dtype)) for k, t in store.tensors.items()}, bindings=store.bindings)
    parts = gather(metadata, runtime)
    if any(p['fields'] != metadata['fields'] or p['bindings'] != metadata['bindings']
            or p['store_id'] != metadata['store_id'] for p in parts):
        raise ValueError('all ranks must share row columns and namespace')
    lengths = [len(p['keys']) for p in parts]; offsets = np.cumsum([0, *lengths])
    all_keys = np.concatenate([p['keys'] for p in parts])
    layout = canonical_layout(all_keys, batches, runtime.train_config.batch_size, runtime.world_size)
    start, end = offsets[runtime.rank:runtime.rank + 2]
    local_dest = layout.destinations[start:end]
    send_indices = np.argsort(local_dest, kind='stable')
    send_counts = np.bincount(local_dest, minlength=runtime.world_size).tolist()
    incoming = [np.flatnonzero(layout.destinations[a:b] == runtime.rank) + a for a, b in zip(offsets, offsets[1:])]
    receive_counts = [len(i) for i in incoming]
    receive_indices = np.concatenate(incoming)
    receive_slots = layout.slots[receive_indices]
    target_rows = len(layout.counts) * layout.local_batch
    # Empty tails on a rank still need a valid dummy input. Exchange one valid
    # row per field as a padding template; its training weight will be zero.
    template_rank = next(i for i, n in enumerate(lengths) if n)
    sends = torch.as_tensor(send_indices, device=store.device)
    slots = torch.as_tensor(receive_slots, device=store.device)
    output = {}
    for name in sorted(list(store.tensors)):
        original = store.tensors.pop(name)
        template = original[:1].clone() if runtime.rank == template_rank else torch.empty(
            (1, *original.shape[1:]), dtype=original.dtype, device=store.device)
        if runtime.world_size > 1:
            import torch.distributed as dist
            dist.broadcast(template, template_rank, group=runtime.process_group)
        sent = original.index_select(0, sends).contiguous()
        received = torch.empty((sum(receive_counts), *original.shape[1:]), dtype=original.dtype, device=store.device)
        if runtime.world_size > 1:
            dist.all_to_all_single(received, sent, output_split_sizes=receive_counts,
                input_split_sizes=send_counts, group=runtime.process_group)
        else:
            received.copy_(sent)
        value = template.expand(target_rows, *template.shape[1:]).clone()
        value.index_copy_(0, slots, received)
        output[name] = value
        del original, sent, received, template
    valid = torch.zeros(target_rows, dtype=torch.bool, device=store.device); valid[slots] = True
    output['training_weights'] = torch.where(valid, output['training_weights'], 0.)
    complete = np.concatenate([p['complete'] for p in parts])
    status = np.full(target_rows, 'UNKNOWN', dtype=object)
    status[receive_slots] = np.where(complete[receive_indices], 'COMPLETE', 'UNKNOWN')
    output.update(store.bindings, terminal_status=tuple(status))
    result = v2.row_store(output, runtime.model_config, runtime.loss_config, segments=layout.segments,
        store_id=v2._sha(dict(original=store.store_id, keys=layout.digest, counts=layout.counts,
            global_batch=runtime.train_config.batch_size, world_size=runtime.world_size)))
    result.distributed_layout = layout
    return result


class GlobalOrder(v2.BatchOrder):
    def __init__(self, store, runtime):
        layout = store.distributed_layout
        super().__init__(store.rows, store.segments, layout.local_batch,
            order='ordered', seed=runtime.train_config.seed, world_size=1)
        self.layout = layout; self.rank_count = runtime.world_size
        if self.steps_per_epoch != len(layout.counts):
            raise ValueError('padded row layout differs from global batch accounting')

    def describe(self):
        return dict(super().describe(), rank_count=self.rank_count, global_counts=list(self.layout.counts),
            global_keys_sha256=self.layout.digest, loss_reduction='global_weighted_terms_v1')

    def rank_batch(self, step, rank):
        return step

    def step_rows(self, step):
        return self.layout.counts[step]


class DistributedTrainer(v2.Trainer):
    """Synchronous data parallel update with globally normalized loss terms."""
    def __init__(self, runtime):
        if (not runtime.v2_config.distributed_rows or runtime.v2_config.order != 'ordered'
                or runtime.v2_config.mode == 'reference'
                or runtime.train_config.batch_size % runtime.world_size):
            raise ValueError('ordered distributed static loss and divisible global batch required')
        super().__init__(runtime)

    def _batch_order(self, store):
        if not hasattr(store, 'distributed_layout'):
            raise ValueError('canonical redistributed row store required')
        return GlobalOrder(store, self.runtime)

    def _batch(self, store, order, epoch, index, pad):
        batch, _, host = super()._batch(store, order, epoch, index, pad)
        n = max(0, min(order.batch_size, order.layout.counts[index] - self.runtime.rank * order.batch_size))
        return batch, n, host

    def _sum(self, value):
        if self.runtime.world_size > 1:
            import torch.distributed as dist
            dist.all_reduce(value, group=self.runtime.process_group)
        return value

    def _step(self, batch, store, host):
        torch = v2._torch(); w = batch['training_weights'].detach(); active = w > 0
        ppo = batch['ppo_advantage_mask'] & active
        zeros = torch.zeros_like(active)
        masks = dict(entropy=active, uniform_kl=active, policy_loss=ppo,
            search_policy_loss=batch.get('search_policy_target_mask', zeros) & active,
            value_loss=(batch['ppo_value_target_mask'] | batch.get('search_value_target_mask', zeros)) & active)
        if self.runtime.model_config.treatment:
            masks.update(opponent_action_loss=batch['opponent_semantic_action_target_mask'] & active,
                hidden_species_loss=batch['hidden_set_target_mask'] & active[:, None],
                hidden_variant_loss=batch['hidden_set_target_mask'] & active[:, None],
                survival_loss=batch['pokemon_end_target_mask'] & active[:, None],
                final_hp_loss=batch['pokemon_end_target_mask'] & active[:, None])
        names = v2.metric_names(self.runtime.model_config, self.runtime.loss_config)[:-1:2]
        denominators = torch.stack([torch.where(masks[n], w if masks[n].ndim == 1 else w[:, None], 0).sum() for n in names])
        advantage = torch.where(ppo, batch['ppo_advantages'].detach(), 0.)
        pw = torch.where(ppo, w, 0.)
        combined = self._sum(torch.cat([denominators, torch.stack([ppo.sum(), (advantage * pw).sum()])]))
        denominators = combined[:len(names)]; count, numerator = combined[-2:]
        total = denominators[names.index('policy_loss')].clamp(min=torch.finfo(w.dtype).tiny)
        mean = numerator / total
        variance = self._sum((torch.where(ppo, (advantage - mean).square(), 0.) * pw).sum()) / total
        batch = dict(batch, _v2_global_denominators=torch.where(denominators > 0, denominators, 1.),
            _v2_global_advantage=torch.stack([count, mean, variance]))
        return super()._step(batch, store, host)

    def _all_reduce_gradients(self, participating):
        # Each local gradient already uses the GLOBAL term denominator.
        torch = v2._torch()
        gradients = [torch.zeros_like(p) if p.grad is None else p.grad for p in self._parameters]
        flat = self._sum(torch.cat([g.reshape(-1) for g in gradients])); offset = 0
        for p, g in zip(self._parameters, gradients):
            p.grad = flat[offset:offset + g.numel()].view_as(g); offset += g.numel()

    def _flush(self, pending):
        # Metrics are local contributions too. Reduce only metrics, not the
        # already-global gradient norm or rank participation/finite flags.
        values = self._buffer[:len(pending), :len(self._names)].contiguous()
        self._sum(values); self._buffer[:len(pending), :len(self._names)].copy_(values)
        return super()._flush(pending)
