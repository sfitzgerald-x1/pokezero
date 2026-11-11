"""Opt-in public retained-v4 -> v5 observation adapter; no runner dispatch.

The original 23-token prefix is copied verbatim. Six appended tokens use the
existing source-ordered public posterior, original current-sampler prior, and
original observation-boundary summary. No physical engine/private opponent
state, sampled world, terminal truth or training label is an input.
"""
from dataclasses import dataclass
import weakref
from typing import NamedTuple

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_contract as contract
from pokezero.randbat import canonical_gen3_randbat_species_id
from pokezero.v5_model import V5ModelConfig
from . import public_belief as belief, public_belief_snapshot as snapshot
from . import public_context, v4_encoder
from .private_team import vocabulary_binding


SCHEMA = 'pokezero.gpu_public_observation.v5'
V4_REFUSED, NAMESPACE, POSTERIOR, PUBLIC_INPUT = (1 << i for i in range(4))


class Tables(NamedTuple):
    source_binding: object
    encoder_binding: object
    prior: object                 # [source species including0,64], original uniform sampler
    variant_valid: object         # [source species including0,64], full original order
    counts: object
    binding: object


@dataclass(frozen=True, eq=False)
class Adapter:
    model_config: V5ModelConfig
    tables: Tables


_ADAPTERS = weakref.WeakKeyDictionary()


def compile_adapter(catalog, *, model_config):
    """Bind the actual retained catalog and complete original source row axis.

    SourceTables has no variant-frequency field. Its original variant_weights
    helper implements the existing unweighted public sampler (not Showdown
    generator frequencies). Reuse that helper on the full original source pool
    for the prior; never infer priors from the narrowed posterior.
    """
    v4_encoder.check_catalog(catalog)
    if type(model_config) is not V5ModelConfig:
        raise ValueError('explicit v5 model/source namespace required')
    category_sha = np.asarray(vocabulary_binding(catalog.vocabulary), dtype='>u4').tobytes().hex()
    if (category_sha != model_config.category_vocabulary_sha256
            or catalog.vocabulary.size != model_config.category_vocab_size):
        raise ValueError('original retained category namespace differs from v5 model')
    source = catalog.source
    names = tuple(canonical_gen3_randbat_species_id(name) for name in source.species_names[1:])
    rows = tuple((name, tuple(v.to_summary() for v in variants))
        for name, variants in zip(names, source.variants[1:]))
    source_sha = contract._sha(dict(source_hash=source.source_hash, format_id='gen3randombattle',
        generation=3, source_ordered_variants=rows))
    counts = tuple(len(variants) for variants in source.variants)
    if (source.species_names[0] != '' or counts[0] != 0 or not rows
            or any(not 0 < count <= 64 for count in counts[1:])
            or model_config.label_vocabulary.source_binding_sha256 != source_sha
            or model_config.label_vocabulary.species_variants != tuple(zip(names, counts[1:]))):
        raise ValueError('complete original source-order v5 variant namespace required, capacity64')
    width = source.device.variant_valid.shape[-1]
    if width != max(counts):
        raise ValueError('source variant storage and original row counts differ')
    count_array = j.asarray(counts, j.int32)
    summary = belief.BeliefSummary(source.device.pool, count_array,
        j.ones_like(count_array, j.float32), j.where(count_array > 0, belief.RESOLVED, belief.UNKNOWN_SPECIES),
        j.zeros_like(count_array, bool))
    prior = belief.variant_weights(summary, source.device)
    prior = j.pad(prior, ((0, 0), (0, 64 - width)))
    valid = j.pad(source.device.variant_valid, ((0, 0), (0, 64 - width)))
    digest = contract._sha(dict(schema=SCHEMA, category=category_sha, label=model_config.label_vocabulary.sha256,
        source_rows=source_sha, source=np.asarray(source.device.binding).tolist(),
        encoder=np.asarray(catalog.binding).tolist(), prior=np.asarray(prior).tolist()))
    tables = Tables(source.device.binding, catalog.binding, prior, valid, count_array,
        j.asarray(np.frombuffer(bytes.fromhex(digest), dtype='>u4').astype(np.uint32)))
    adapter = Adapter(model_config, tables)
    _ADAPTERS[adapter] = (weakref.ref(catalog), tables)
    return adapter


def check_adapter(adapter):
    """Host lifetime/catalog check, separate from cached device inference."""
    if isinstance(j.zeros((), j.int32), jax.core.Tracer):
        raise ValueError('v5 observation host check cannot be traced')
    record = _ADAPTERS.get(adapter) if type(adapter) is Adapter else None
    if record is None or record[0]() is None or adapter.tables is not record[1]:
        raise ValueError('original live v5 observation adapter required')
    return v4_encoder.check_catalog(record[0]())


class Observation(NamedTuple):
    numeric_features: object
    categorical_ids: object
    token_type_ids: object
    attention_mask: object
    legal_action_mask: object
    switch_targets: object
    admitted: object
    error: object
    posterior_weights: object
    prior_weights: object
    posterior_reason: object
    pin_conflict: object


def _require(value, shape, dtype, name):
    if getattr(value, 'shape', None) != shape or getattr(value, 'dtype', None) != j.dtype(dtype):
        raise ValueError('typed v5 public adapter field required: ' + name)


def append_device(encoded, public_state, adapter, *, observer_sides):
    """Append to an actual Encoded for the SAME public state/owning observer.

    The callsite supplies that actual pairing; this adapter does not attest the
    origin of copied device carries. Host callers check_adapter before cached
    calls. The only repeated operation is the original pure public boundary
    value-copy, needed because Encoded omits pin_conflict. Its returned context
    and counters are discarded, so persistent pin conflict counts never advance.
    Exact mask/reason/weight joins bind this copy to the supplied encoded output.
    Refused rows remain present and explicit; unknown species/source mass does
    not itself refuse an otherwise admitted original observation.
    Admission here covers this adapter; original owning-driver refusals remain
    separate inputs to future runner integration.
    """
    if type(encoded) is not v4_encoder.Encoded or type(public_state) is not v4_encoder.State or type(adapter) is not Adapter:
        raise ValueError('actual retained Encoded and public State required')
    record = _ADAPTERS.get(adapter)
    catalog = record[0]() if record is not None else None
    if catalog is None:
        raise ValueError('live original retained encoder catalog required')
    tables = adapter.tables; config = adapter.model_config; a = encoded.observation
    b = a.numeric_features.shape[0]; rows = j.arange(b)
    if b <= 0:
        raise ValueError('nonempty retained observation required')
    for value, shape, dtype, name in (
            (a.numeric_features, (b, 23, 132), j.float32, 'numeric'),
            (a.categorical_ids, (b, 23, 41), j.int32, 'categories'),
            (a.token_type_ids, (b, 23), j.int32, 'roles'),
            (a.attention_mask, (b, 23), j.bool_, 'attention'),
            (a.legal_action_mask, (b, 9), j.bool_, 'legal'),
            (a.switch_targets, (b, 5), j.int32, 'switch_targets'),
            (a.admitted, (b,), j.bool_, 'v4 admission'),
            (observer_sides, (b,), j.int32, 'owning observers'),
            (encoded.posterior_mask, (b, 6, 2), j.uint32, 'posterior mask'),
            (encoded.posterior_weights, (b, 6, catalog.source.device.variant_valid.shape[-1]), j.float32, 'posterior weights'),
            (encoded.posterior_reason, (b, 6), j.int32, 'posterior reason'),
            (public_state.binding, (b, 8), j.uint32, 'encoder binding'),
            (public_state.cursor.source_binding, (b, 8), j.uint32, 'source binding')):
        _require(value, shape, dtype, name)
    observer = j.clip(observer_sides, 0, 1)
    namespace = (j.all(public_state.binding == tables.encoder_binding, axis=-1)
        & j.all(public_state.cursor.source_binding == tables.source_binding, axis=-1)
        & (observer_sides >= 0) & (observer_sides < 2))
    context, copied, copy_ok = public_context.observation_boundary_copy(public_state.context,
        public_state.beliefs, public_state.mons, catalog.source.device, catalog.context, valid=a.admitted)
    summary, cache_ok = snapshot.read(context.refresh.controller.visible, copied, catalog.source.device)
    species = copied.species[rows, observer]
    species_ok = (species >= 0) & (species < tables.prior.shape[0])
    safe_species = j.clip(species, 0, tables.prior.shape[0] - 1)
    present = a.attention_mask[:, 7:13]
    expected_weights = belief.variant_weights(summary, catalog.source.device)[rows, observer]
    expected_weights = j.where(present[..., None], expected_weights, 0)
    reason = summary.reason[rows, observer]
    effective = summary.effective[rows, observer]
    count = summary.count[rows, observer]
    conflict = summary.pin_conflict[rows, observer]
    index = j.arange(64, dtype=j.uint32)
    selected = ((j.take(effective, index // 32, axis=-1) >> (index % 32)) & 1).astype(bool)
    posterior_ok = (copy_ok & j.all(cache_ok, axis=(1, 2)) & j.all(species_ok, axis=-1)
        & j.all(encoded.posterior_mask == effective, axis=(1, 2))
        & j.all(encoded.posterior_reason == reason, axis=-1)
        & j.all(j.isfinite(encoded.posterior_weights) & (encoded.posterior_weights == expected_weights), axis=(1, 2))
        & j.all(j.sum(selected, axis=-1) == count, axis=-1)
        & ~j.any(selected & ~tables.variant_valid[safe_species], axis=(1, 2)))
    unknown = (reason == belief.UNKNOWN_SPECIES) | (reason == belief.NO_SOURCE)
    public_ok = (j.all(j.isfinite(a.numeric_features), axis=(1, 2))
        & j.all((a.categorical_ids >= 0) & (a.categorical_ids < config.category_vocab_size), axis=(1, 2))
        & j.all(a.token_type_ids == j.asarray((0,) + (1,) * 6 + (2,) * 6 + (3,) * 9 + (5,), j.int32), axis=-1)
        & j.all(j.where(~present, (species == 0) & (count == 0) & unknown & ~conflict, True), axis=-1)
        & j.all(j.where(unknown, (count == 0) & ~conflict, (count > 0) & (species > 0)), axis=-1))
    error = (j.where(a.admitted, 0, V4_REFUSED) | j.where(namespace, 0, NAMESPACE)
        | j.where(posterior_ok, 0, POSTERIOR) | j.where(public_ok, 0, PUBLIC_INPUT)).astype(j.int32)
    admitted = error == 0
    use_prior = present & ~unknown
    prior = j.where(use_prior[..., None], tables.prior[safe_species], 0)
    posterior = j.pad(encoded.posterior_weights, ((0, 0), (0, 0), (0, 64 - encoded.posterior_weights.shape[-1])))
    prior_count = j.where(use_prior, tables.counts[safe_species], 0)
    summaries = j.stack((prior_count.astype(j.float32) / j.float32(64),
        count.astype(j.float32) / j.float32(64), reason.astype(j.float32) / j.float32(3),
        conflict.astype(j.float32)), axis=-1)
    tail = j.concatenate((posterior, prior, summaries), axis=-1)
    tail = j.where(admitted[:, None, None], tail, 0)
    return Observation(j.concatenate((a.numeric_features, tail), axis=1),
        j.concatenate((a.categorical_ids, a.categorical_ids[:, 7:13]), axis=1),
        j.concatenate((a.token_type_ids, j.full((b, 6), 7, j.int32)), axis=1),
        j.concatenate((a.attention_mask, j.ones((b, 6), bool)), axis=1),
        a.legal_action_mask, a.switch_targets, admitted, error,
        j.where(admitted[:, None, None], posterior, 0), j.where(admitted[:, None, None], prior, 0), reason, conflict)


def actor_inputs(observation, adapter):
    """Named public29 fields plus explicit namespaces; no labels/private view.

    Refused rows carry history_mask=False. They are retained in the batch and
    cannot be admitted by the v5 actor's device validity check.
    """
    if type(observation) is not Observation or type(adapter) is not Adapter:
        raise ValueError('explicit appended public v5 observation required')
    config = adapter.model_config
    return dict(schema=contract.TRAINING_CONTRACT_SCHEMA, observation_schema=contract.OBSERVATION_SCHEMA_VERSION_V5,
        category_vocabulary_sha256=config.category_vocabulary_sha256,
        label_vocabulary_sha256=config.label_vocabulary.sha256, view='PUBLIC',
        categorical_ids=observation.categorical_ids[:, None], numeric_features=observation.numeric_features[:, None],
        token_type_ids=observation.token_type_ids[:, None], attention_mask=observation.attention_mask[:, None],
        history_mask=observation.admitted[:, None])
