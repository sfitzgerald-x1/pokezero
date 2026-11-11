"""Training-only host captures/cache bridge for the explicit v5 collector.

No JAX/Torch import occurs at module load. Source-defined set fields come from
the original constructor pool; neither PPO targets nor later observations can
create hidden-set or terminal truth. Torch is needed only to finalize justified
PPO traces with the original v5 GAE helper.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from numbers import Real

import numpy as np

from pokezero import dataset, v5_contract as c, v5_data
from pokezero.dex import normalize_id
from pokezero.observation import PokeZeroObservationV0, ObservationPerspective
from pokezero.trajectory import TrajectoryStep


SCHEMA = 'pokezero.engine_collection_capture.v5'
BOOTSTRAP_SCHEMA = 'pokezero.public_value_bootstrap.v5'


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _tuple(value):
    return tuple(_tuple(x) for x in value) if isinstance(value, (list, tuple)) else value


@dataclass(frozen=True)
class CapturedDecision:
    row: v5_data.DecisionRow
    # Input token order: six own request rows, six public encounter rows.
    physical: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, 'physical', tuple(self.physical))
        if (type(self.row) is not v5_data.DecisionRow or len(self.physical) != 12
                or any(type(i) is not int or not -1 <= i < 6 for i in self.physical)):
            raise ValueError('actual source-frame token physical identities required')
        for group in (self.physical[:6], self.physical[6:]):
            valid = tuple(i for i in group if i >= 0)
            if len(set(valid)) != len(valid):raise ValueError('ambiguous physical token correspondence')
        metadata = self.row.step.metadata.get('v5_capture')
        if metadata is not None:
            body = {key:value for key,value in metadata.items() if key not in ('schema','source_frame_sha256')}
            body['public_inputs'] = self.row.observation.model_inputs()
            if (metadata.get('schema') != SCHEMA or sha(body) != self.row.source_frame_sha256
                    or metadata.get('source_frame_sha256') != self.row.source_frame_sha256
                    or _tuple(body.get('physical')) != self.physical
                    or (body.get('stream'),body.get('frame'),body.get('owner')) !=
                        (self.row.observation.address.stream,self.row.observation.address.frame,self.row.observation.address.owner)
                    or body.get('choice') != self.row.step.action_index
                    or body.get('behavior_probability') != self.row.step.action_probability
                    or body.get('value_estimate') != self.row.step.value_estimate):
                raise ValueError('literal source-frame capture differs from behavior/input row')


@dataclass(frozen=True)
class TerminalCapture:
    stream: int
    frame: int
    status: str
    source_sha256: str
    winner: int | None = None
    hp: tuple = ()
    max_hp: tuple = ()

    def __post_init__(self):
        c.DecisionAddress(self.stream, self.frame, 0); c._binding(self.source_sha256)
        object.__setattr__(self, 'hp', _tuple(self.hp)); object.__setattr__(self, 'max_hp', _tuple(self.max_hp))
        if self.status not in ('COMPLETE', 'CAPPED', 'REFUSED', 'FAILED'):
            raise ValueError('literal collection disposition required')
        if self.status != 'COMPLETE':
            if self.winner is not None or self.hp or self.max_hp:
                raise ValueError('unknown terminal cannot carry outcome/end truth')
            return
        if type(self.winner) is not int or self.winner not in (0, 1, 2):
            raise ValueError('actual winner0/1 or explicit tie2 required')
        if len(self.hp) != 2 or len(self.max_hp) != 2 or any(len(x) != 6 for x in (*self.hp, *self.max_hp)):
            raise ValueError('all twelve literal terminal HP/maxHP values required')
        if any(type(h) is not int or type(m) is not int or not 0 <= h <= m or m <= 0
                for hs, ms in zip(self.hp, self.max_hp) for h, m in zip(hs, ms)):
            raise ValueError('valid literal terminal HP required')


def matching_hidden_set(species, *, level, moves, ability, item, hidden_power_type, source):
    """All source-defined full set fields, in original row order, no best match.

    Nature/EV/IV/role/source-set-id are not coordinates on Gen3RandbatVariant's
    public variant axis. Their absence never selects a preferred duplicate.
    Hidden Power uses the constructor's literal derived type, not a stat guess.
    """
    sid = source.species_id(species)
    if not sid:return None
    literal_type = normalize_id(hidden_power_type)
    normalized_moves = tuple(normalize_id(value) for value in moves)
    if any(value.startswith('hiddenpower') and value != 'hiddenpower'
            and (not literal_type or value != 'hiddenpower' + literal_type) for value in normalized_moves):
        return None
    def move_id(value):
        value = normalize_id(value)
        return 'hiddenpower' + normalize_id(hidden_power_type) if value == 'hiddenpower' and hidden_power_type else value
    actual_moves = tuple(sorted(move_id(value) for value in moves))
    if (type(level) is not int or level <= 0 or not actual_moves or len(set(actual_moves)) != len(actual_moves)
            or any(value == 'hiddenpower' for value in actual_moves)):
        return None
    matched = []
    for index, variant in enumerate(source.variants[sid]):
        if (variant.level == level and normalize_id(variant.ability) == normalize_id(ability)
                and normalize_id(variant.item) == normalize_id(item)
                and tuple(sorted(normalize_id(value) for value in variant.moves)) == actual_moves):
            matched.append(index)
    if not matched:return None
    return c.HiddenSet(canonical_species(source.species_names[sid]), tuple(matched))


def canonical_species(value):
    from pokezero.randbat import canonical_gen3_randbat_species_id
    return canonical_gen3_randbat_species_id(value)


def public_observation(encoded, appended, *, row, owner, address, adapter, public_species):
    """Serialize literal v4 prefix; canonical contract tail must round-trip f32.

    PublicPosterior's Python normalization is the contract serialization form.
    Require its float32 bytes to equal the actual device tail before retaining it.
    No feature is derived from hidden-set targets or physical opponent state.
    """
    if not bool(np.asarray(appended.admitted)[row]):raise ValueError('refused v5 input row')
    original = encoded.observation
    old = PokeZeroObservationV0(*(_tuple(np.asarray(getattr(original, name))[row].tolist()) for name in
        ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'legal_action_mask')),
        perspective=ObservationPerspective.from_showdown_slot(f'p{owner+1}', f'p{owner+1}'),
        schema_version=c.OBSERVATION_SCHEMA_VERSION_V4)
    vocabulary = adapter.model_config.label_vocabulary
    posteriors = []
    for slot, sid in enumerate(public_species):
        reason = int(np.asarray(appended.posterior_reason)[row, slot])
        words = np.asarray(encoded.posterior_mask)[row, slot]
        mask = int(words[0]) | int(words[1]) << 32
        name = vocabulary.species[int(sid) - 1] if sid else None
        count = 0 if reason in (1, 3) else vocabulary.variant_count(name)
        prior = tuple(map(float, np.asarray(appended.prior_weights)[row, slot, :count]))
        posteriors.append(c.PublicPosterior(vocabulary, name, mask, prior, c.BELIEF_REASONS[reason],
            bool(np.asarray(appended.pin_conflict)[row, slot])))
    observation = c.append_belief_tokens(old, posteriors, address=address,
        category_vocabulary_sha256=adapter.model_config.category_vocabulary_sha256)
    for name in ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'legal_action_mask'):
        dtype = np.float32 if name == 'numeric_features' else None
        if not np.array_equal(np.asarray(getattr(observation, name), dtype=dtype),
                np.asarray(getattr(appended, name))[row]):
            raise ValueError('canonical v5 contract differs from actual captured device input')
    return old, observation


def capture_decision(old, observation, *, battle_id, seed, turn, choice, probability, value,
        physical, hidden_sets, source_frame_sha256, capture_metadata, opponent_action=None, privileged=None):
    if (not isinstance(probability, Real) or isinstance(probability, bool) or not np.isfinite(probability)
            or not 0 < probability <= 1 or not isinstance(value, Real) or isinstance(value, bool)
            or not np.isfinite(value) or not -1 <= value <= 1):
        raise ValueError('literal finite raw actor behavior probability/value required')
    labels = c.V5Labels(observation.address, opponent_next_action=opponent_action, hidden_sets=tuple(hidden_sets))
    step = TrajectoryStep(f'p{observation.address.owner+1}', turn, old, tuple(old.legal_action_mask), choice,
        action_probability=float(probability), value_estimate=float(value),
        metadata={'v5_capture': dict(capture_metadata, schema=SCHEMA, source_frame_sha256=source_frame_sha256)})
    return CapturedDecision(v5_data.DecisionRow(battle_id, seed, 'gen3randombattle', step, observation,
        labels, source_frame_sha256, privileged), tuple(physical))


def finish(decisions, terminals):
    """Attach only literal terminal truth through saved source-frame identities."""
    terminal_by_stream = {x.stream: x for x in terminals}
    if len(terminal_by_stream) != len(terminals):raise ValueError('one final disposition per stream required')
    result = []
    for captured in decisions:
        row = captured.row; address = row.observation.address
        terminal = terminal_by_stream[address.stream]
        if terminal.frame < address.frame:raise ValueError('terminal predates actual decision')
        status = 'REFUSED' if terminal.status == 'FAILED' else terminal.status
        outcome = None; end = (None,) * 12
        if status == 'COMPLETE':
            outcome = 0 if terminal.winner == 2 else 1 if terminal.winner == address.owner else -1
            labels = []
            for index, physical in enumerate(captured.physical):
                side = address.owner if index < 6 else 1 - address.owner
                labels.append(None if physical < 0 else c.PokemonEnd(
                    terminal.hp[side][physical] > 0, terminal.hp[side][physical] / terminal.max_hp[side][physical]))
            end = tuple(labels)
        row = v5_data.with_terminal(row, address=c.DecisionAddress(address.stream, terminal.frame, address.owner),
            status=status, outcome=outcome, pokemon_end=end, source_sha256=terminal.source_sha256)
        result.append(replace(captured, row=row))
    return tuple(result)


def cache_batches(decisions, *, loss_config, batch_size, discount=1.):
    """Original GAE on literal complete trajectories or actual public bootstrap.

    For an unknown ending, the last captured public value can bootstrap EARLIER
    decisions in that same owner/stream. It cannot bootstrap itself; that final
    row retains false policy/value masks. Every active unknown critic row gets
    an explicit later same-owner v5_bootstrap capture, never a terminal zero.
    """
    from pokezero.v5_loss import V5LossConfig, gae_targets
    if type(loss_config) is not V5LossConfig or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('explicit v5 loss recipe and positive batch size required')
    if isinstance(discount,bool) or not isinstance(discount,Real) or not np.isfinite(discount) or not 0<=discount<=1:
        raise ValueError('finite explicit discount in0..1 required')
    decisions = tuple(decisions); groups = {}
    for index, captured in enumerate(decisions):
        if type(captured) is not CapturedDecision:raise ValueError('actual captured decision required')
        captured.__post_init__()
        address = captured.row.observation.address
        groups.setdefault((address.stream, address.owner), []).append(index)
    examples = [None] * len(decisions)
    for indices in groups.values():
        rows = [decisions[i].row for i in indices]
        identity = lambda row:(row.battle_id,row.seed,row.format_id,
            row.observation.category_vocabulary_sha256,row.observation.label_vocabulary.sha256,
            row.terminal_source_sha256)
        if any(identity(row)!=identity(rows[0]) for row in rows):
            raise ValueError('same original battle/seed/format/source namespace and terminal capture required for GAE')
        frames = [r.observation.address.frame for r in rows]
        if frames != sorted(set(frames)):raise ValueError('one ordered actual decision per source frame/owner required')
        if len({r.labels.terminal_status for r in rows}) != 1:raise ValueError('one original ending per trajectory required')
        complete = rows[0].labels.terminal_status == 'COMPLETE'
        if complete and any(r.labels.outcome != rows[0].labels.outcome for r in rows):raise ValueError('literal outcome drift')
        valid_count = len(rows) if complete else max(0, len(rows) - 1)
        advantages = targets = []
        if valid_count:
            import torch
            values = torch.tensor([r.step.value_estimate for r in rows[:valid_count]], dtype=torch.float32)
            rewards = torch.zeros_like(values)
            next_values = torch.tensor([r.step.value_estimate for r in rows[1:valid_count+1]]
                + ([0.] if complete else []), dtype=torch.float32)
            terminated = torch.zeros(valid_count, dtype=torch.bool); continues = torch.ones_like(terminated)
            continues[-1] = False
            if complete:terminated[-1] = True; rewards[-1] = rows[-1].labels.outcome
            advantages, targets = gae_targets(rewards, values, next_values, terminated, continues,
                discount=discount, policy_lambda=loss_config.policy_lambda, value_lambda=loss_config.value_lambda)
        for position, (index, row) in enumerate(zip(indices, rows)):
            active = position < valid_count; o = row.step.observation
            metadata = dict(row.step.metadata)
            metadata['v5_return_known'] = complete
            if active and not complete:
                anchor = rows[-1]
                metadata['v5_bootstrap'] = dict(schema=BOOTSTRAP_SCHEMA, address=dict(
                    stream=anchor.observation.address.stream, frame=anchor.observation.address.frame,
                    owner=anchor.observation.address.owner), source_frame_sha256=anchor.source_frame_sha256,
                    value=anchor.step.value_estimate)
            known_return = row.labels.outcome if complete else 0.
            examples[index] = dataset.TrajectoryExample(row.battle_id, row.seed, row.format_id, row.step.player_id,
                row.step.turn_index, (o.categorical_ids,), (o.numeric_features,), (o.token_type_ids,),
                (o.attention_mask,), (True,), tuple(o.legal_action_mask), row.step.action_index,
                row.labels.outcome if complete and position == len(rows)-1 else 0., known_return,
                value_estimate=row.step.value_estimate, ppo_advantage=float(advantages[position]) if active else None,
                ppo_value_target=float(targets[position]) if active else None, action_probability=row.step.action_probability,
                step_metadata=metadata, terminal_capped=row.labels.terminal_status == 'CAPPED')
    for start in range(0, len(decisions), batch_size):
        batch = dataset.training_batch_from_examples(examples[start:start+batch_size])
        yield v5_data.attach_batch(batch, [x.row for x in decisions[start:start+batch_size]])


def write_cache(path, decisions, *, loss_config, batch_size, discount=1.):
    batches = tuple(cache_batches(decisions, loss_config=loss_config, batch_size=batch_size, discount=discount))
    if not batches:raise ValueError('no actual decisions; preserve ledger without an invented cache row')
    # Existing v5 writer accepts one batch. Concatenate using its namespace-safe adapter.
    return v5_data.write_cache(path, dataset._combine_training_batches(batches))
