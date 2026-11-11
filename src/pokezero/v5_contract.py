"""Append-only v5 observation and training-label contract.

This host adapter defines data, not a new default encoder or a trainer. It never
loads weights, runs a network, samples a world or chooses a collection seed.
Legacy v4 readers remain v4 readers; a v5 consumer must opt into this contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from numbers import Real
import re
from types import MappingProxyType
from typing import Mapping

from .actions import ACTION_COUNT
from .observation import OBSERVATION_SCHEMA_VERSION_V4, ObservationSpec, PokeZeroObservationV0
from .search_labels import validate_search_targets


OBSERVATION_SCHEMA_VERSION_V5 = 'pokezero.observation.v5'
TRAINING_CONTRACT_SCHEMA = 'pokezero.training_contract.v5'
V4_TOKENS, V5_TOKENS, NUMERIC_FEATURES, CATEGORICAL_FEATURES = 23, 29, 132, 41
OPPONENT_OFFSET, BELIEF_OFFSET, BELIEF_TYPE, MAX_VARIANTS = 7, 23, 7, 64
V4_SPEC = ObservationSpec(categorical_feature_count=41, numeric_feature_count=132,
    transition_token_count=0, schema_version=OBSERVATION_SCHEMA_VERSION_V4)
BELIEF_REASONS = ('RESOLVED', 'UNKNOWN_SPECIES', 'INCONSISTENT_REVEALS', 'NO_SOURCE')
PINNED_SOURCE_CENSUS = MappingProxyType(dict(source_hash='f5a5265143d423af', species=220,
    variant_rows=1682, maximum=44, maximum_species=('swalot',), species_over64=0))
DECISIONS = (
    ('belief_type_7', 'Preserve retired recent-event type4 and transition type6; append a new role.'),
    ('fixed_feature_widths', 'Pinned public source f5a5265143d423af has220species/1682variant rows, maximum44(Swalot); capacity64 retains every original row and fits posterior64+prior64+4summary fields.'),
    ('source_axis_admission', 'Bind every source-ordered variant identity before encoding; a future source above64 refuses without truncation and needs an explicit representation change.'),
    ('semantic_opponent_labels', 'MOVE identity and SWITCH target species never reuse the nine actor slot indices.'),
    ('unknown_terminal_mask', 'A capped/refused game has no survival-to-end or final-HP supervision.'),
    ('duplicate_set_rows', 'Uniform mass over all source rows matching the true set preserves duplicate identities.'),
    ('public_leaf_D', 'Depth1 labels use the public value leaf and opponent own-seat reply policy; actor stays raw.'),
    ('expert_width_512', 'Three role-specific512-wide feedforward experts keep the common512x3 trunk near the planned10M scale.'),
    ('value_distribution', 'Use the existing value-probe101-bin[-1,1] HL-Gauss support and0.75-bin smoothing convention.'),
    ('D_root_value', 'The root value is the improved-policy expectation of all legal completed Q values; mix equally with the observed outcome.'),
    ('scratch_production', 'Warm-start migration remains possible; initial Phase C production arms start from scratch.'),
)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _binding(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('exact source/vocabulary SHA256 binding required')


def _identifier(value):
    if type(value) is not str or re.fullmatch('[a-z0-9]+', value) is None:
        raise ValueError('canonical source identity required')


def _real(value, *, low=0.0, high=1.0):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('real numerical value required')
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError('finite in-range numerical value required')
    return result


def _list(value):
    return value.tolist() if hasattr(value, 'tolist') else value


@dataclass(frozen=True)
class DecisionAddress:
    stream: int
    frame: int
    owner: int

    def __post_init__(self):
        if (any(type(x) is not int or x < 0 for x in (self.stream, self.frame))
                or type(self.owner) is not int or self.owner not in (0, 1)):
            raise ValueError('nonnegative stream/frame and original owner0/1 required')


def source_variant_census(source, *, capacity=MAX_VARIANTS):
    """Read all static public source rows, without teams, worlds or models.

    Counts are diagnostics, not a projection/pruning policy. The binding also
    includes every original ordered variant identity and its source facts.
    """
    from .randbat import Gen3RandbatSource
    if (type(source) is not Gen3RandbatSource or source.metadata.generation != 3
            or source.metadata.format_id != 'gen3randombattle'
            or type(capacity) is not int or capacity <= 0):
        raise ValueError('original static Gen3 randbat source and positive capacity required')
    rows = tuple((name, tuple(variant.to_summary() for variant in universe.variants))
        for name, universe in source.universes.items())
    if not rows or any(not variants for _, variants in rows):
        raise ValueError('source requires nonempty original species variant axes')
    counts = tuple((name, len(variants)) for name, variants in rows)
    largest = max(count for _, count in counts)
    return dict(source_hash=source.metadata.source_hash, species=len(counts),
        variant_rows=sum(count for _, count in counts), maximum=largest,
        maximum_species=tuple(name for name, count in counts if count == largest),
        species_variants=counts, species_over_capacity=tuple((name, count) for name, count in counts if count > capacity),
        source_rows_sha256=_sha(dict(source_hash=source.metadata.source_hash,
            format_id=source.metadata.format_id, generation=3, source_ordered_variants=rows)))


@dataclass(frozen=True)
class LabelVocabulary:
    """Source-order label namespace, separate from checkpoint category rows."""
    source_binding_sha256: str
    moves: tuple[str, ...]
    species_variants: tuple[tuple[str, int], ...]

    def __post_init__(self):
        _binding(self.source_binding_sha256)
        object.__setattr__(self, 'moves', tuple(self.moves))
        object.__setattr__(self, 'species_variants', tuple(tuple(row) for row in self.species_variants))
        if any(len(row) != 2 for row in self.species_variants):
            raise ValueError('species/count source rows required')
        names = self.species
        if not self.moves or not names or len(set(self.moves)) != len(self.moves) or len(set(names)) != len(names):
            raise ValueError('nonempty unique source move/species namespaces required')
        for name in (*self.moves, *names):
            _identifier(name)
        for _, count in self.species_variants:
            if type(count) is not int or not 1 <= count <= MAX_VARIANTS:
                raise ValueError('source variant axis must contain1..64 original rows per species')

    @classmethod
    def from_public_source(cls, source, *, move_ids, expected_source_hash):
        census = source_variant_census(source)
        if (type(expected_source_hash) is not str or re.fullmatch('[0-9a-f]{16}', expected_source_hash) is None
                or census['source_hash'] != expected_source_hash):
            raise ValueError('original public set source hash differs')
        if census['species_over_capacity']:
            raise ValueError('full source axis exceeds64; no rows removed: ' + repr(census['species_over_capacity']))
        return cls(census['source_rows_sha256'], tuple(move_ids), census['species_variants'])

    @property
    def species(self):
        return tuple(row[0] for row in self.species_variants)

    @property
    def sha256(self):
        return _sha(asdict(self))

    @property
    def opponent_classes(self):
        return tuple('MOVE:' + name for name in self.moves) + tuple('SWITCH:' + name for name in self.species)

    def variant_count(self, species):
        try:
            return dict(self.species_variants)[species]
        except KeyError as error:
            raise ValueError('species outside bound variant source') from error


@dataclass(frozen=True)
class PublicPosterior:
    vocabulary: LabelVocabulary
    species: str | None
    effective_mask: int
    prior_weights: tuple[float, ...]
    reason: str = 'RESOLVED'
    pin_conflict: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'prior_weights', tuple(self.prior_weights))
        if (type(self.vocabulary) is not LabelVocabulary or self.reason not in BELIEF_REASONS
                or type(self.pin_conflict) is not bool or type(self.effective_mask) is not int
                or not 0 <= self.effective_mask < 1 << MAX_VARIANTS):
            raise ValueError('original public posterior fields required')
        if self.species is None or self.reason == 'NO_SOURCE':
            if self.reason not in ('UNKNOWN_SPECIES', 'NO_SOURCE') or self.effective_mask or self.prior_weights:
                raise ValueError('unknown/unavailable public species carries zero source mass')
            if self.species is not None:
                self.vocabulary.variant_count(self.species)
            return
        count = self.vocabulary.variant_count(self.species)
        if (self.reason == 'UNKNOWN_SPECIES' or len(self.prior_weights) != count
                or not 0 < self.effective_mask < 1 << count):
            raise ValueError('posterior mask and prior axis must match the bound public species')
        weights = tuple(_real(value) for value in self.prior_weights)
        if sum(weights) <= 0 or sum(value for i, value in enumerate(weights) if self.effective_mask >> i & 1) <= 0:
            raise ValueError('public prior and effective posterior need positive mass')
        object.__setattr__(self, 'prior_weights', weights)

    def numeric_row(self):
        if not self.prior_weights:
            prior = posterior = (0.0,) * MAX_VARIANTS
        else:
            total = sum(self.prior_weights)
            prior = tuple(value / total for value in self.prior_weights)
            selected = tuple(value if self.effective_mask >> i & 1 else 0.0 for i, value in enumerate(prior))
            mass = sum(selected)
            posterior = tuple(value / mass for value in selected)
            prior += (0.0,) * (MAX_VARIANTS - len(prior))
            posterior += (0.0,) * (MAX_VARIANTS - len(posterior))
        return posterior + prior + (len(self.prior_weights) / MAX_VARIANTS,
            self.effective_mask.bit_count() / MAX_VARIANTS,
            BELIEF_REASONS.index(self.reason) / 3, float(self.pin_conflict))


@dataclass(frozen=True)
class V5Observation:
    address: DecisionAddress
    categorical_ids: tuple
    numeric_features: tuple
    token_type_ids: tuple
    attention_mask: tuple
    legal_action_mask: tuple
    posteriors: tuple[PublicPosterior, ...]
    category_vocabulary_sha256: str
    view: str = 'PUBLIC'
    schema_version: str = OBSERVATION_SCHEMA_VERSION_V5

    def __post_init__(self):
        _binding(self.category_vocabulary_sha256)
        if (type(self.address) is not DecisionAddress or self.view not in ('PUBLIC', 'PRIVILEGED')
                or self.schema_version != OBSERVATION_SCHEMA_VERSION_V5):
            raise ValueError('explicit v5 decision/view contract required')
        for name, width in (('categorical_ids', CATEGORICAL_FEATURES), ('numeric_features', NUMERIC_FEATURES)):
            rows = tuple(tuple(row) for row in _list(getattr(self, name)))
            if len(rows) != V5_TOKENS or any(len(row) != width for row in rows):
                raise ValueError('v5 requires29tokens with unchanged v4 feature widths')
            immutable_rows = []
            for row in rows:
                immutable_row = []
                for value in row:
                    if name == 'categorical_ids':
                        if type(value) is not int or value < 0:
                            raise ValueError('nonnegative original category row required')
                        immutable_row.append(value)
                    else:
                        # Convert a permitted real exactly once, then retain
                        # that immutable finite value, never caller objects.
                        immutable_row.append(_real(value, low=-math.inf, high=math.inf))
                immutable_rows.append(tuple(immutable_row))
            object.__setattr__(self, name, tuple(immutable_rows))
        for name, count, kind in (('token_type_ids', V5_TOKENS, int), ('attention_mask', V5_TOKENS, bool),
                ('legal_action_mask', ACTION_COUNT, bool)):
            row = tuple(_list(getattr(self, name)))
            if len(row) != count or any(type(value) is not kind for value in row):
                raise ValueError('typed v5 masks/token roles required')
            object.__setattr__(self, name, row)
        object.__setattr__(self, 'posteriors', tuple(self.posteriors))
        if (len(self.posteriors) != 6 or any(type(p) is not PublicPosterior for p in self.posteriors)
                or len({p.vocabulary.sha256 for p in self.posteriors}) != 1
                or self.token_type_ids[BELIEF_OFFSET:] != (BELIEF_TYPE,) * 6
                or any(not 0 <= value < 16 for value in self.token_type_ids)):
            raise ValueError('six same-source public posteriors and appended token role7 required')
        if (self.token_type_ids[:V4_TOKENS] != (0,) + (1,) * 6 + (2,) * 6 + (3,) * 9 + (5,)
                or self.attention_mask[BELIEF_OFFSET:] != (True,) * 6):
            raise ValueError('preserve v4 token roles and attend all six public belief tokens')
        if self.view == 'PUBLIC' and self.categorical_ids[BELIEF_OFFSET:] != self.categorical_ids[OPPONENT_OFFSET:OPPONENT_OFFSET + 6]:
            raise ValueError('belief categorical identity must copy its original public opponent slot')
        if self.numeric_features[BELIEF_OFFSET:] != tuple(p.numeric_row() for p in self.posteriors):
            raise ValueError('belief token bytes must derive only from the declared public posterior')

    @property
    def label_vocabulary(self):
        return self.posteriors[0].vocabulary

    def model_inputs(self):
        """Actor inputs exclude labels, hidden truth and the optional peer view."""
        if self.view != 'PUBLIC':
            raise ValueError('privileged view cannot enter the public actor input surface')
        return {name: getattr(self, name) for name in
            ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'legal_action_mask')}


def append_belief_tokens(v4, posteriors, *, address, category_vocabulary_sha256):
    if type(v4) is not PokeZeroObservationV0 or v4.schema_version != OBSERVATION_SCHEMA_VERSION_V4:
        raise ValueError('an explicit original v4 observation is required')
    v4.validate(V4_SPEC)
    if type(address) is not DecisionAddress:
        raise ValueError('original owning decision address required')
    if v4.perspective is not None and v4.perspective.showdown_slot != ('p1', 'p2')[address.owner]:
        raise ValueError('public perspective and owning decision address differ')
    posterior = tuple(posteriors)
    if len(posterior) != 6:
        raise ValueError('one public posterior per opponent slot required')
    if any(not _list(v4.attention_mask)[OPPONENT_OFFSET + i] and p.species is not None
            for i, p in enumerate(posterior)):
        raise ValueError('an unseen public slot cannot expose its private species posterior')
    categories = tuple(tuple(row) for row in _list(v4.categorical_ids))
    numeric = tuple(tuple(row) for row in _list(v4.numeric_features))
    types = tuple(_list(v4.token_type_ids))
    if types != (0,) + (1,) * 6 + (2,) * 6 + (3,) * 9 + (5,):
        raise ValueError('original v4 token positions and roles must remain unchanged')
    return V5Observation(address, categories + categories[OPPONENT_OFFSET:OPPONENT_OFFSET + 6],
        numeric + tuple(p.numeric_row() for p in posterior), types + (BELIEF_TYPE,) * 6,
        tuple(_list(v4.attention_mask)) + (True,) * 6, tuple(_list(v4.legal_action_mask)),
        posterior, category_vocabulary_sha256)


def privileged_view(public, peer_own_v4, *, peer_address, opponent_slot_map, peer_category_vocabulary_sha256):
    """Optional training-only swap, with explicit physical-slot correspondence."""
    if (type(public) is not V5Observation or public.view != 'PUBLIC'
            or type(peer_address) is not DecisionAddress
            or peer_address != replace(public.address, owner=1 - public.address.owner)
            or type(peer_own_v4) is not PokeZeroObservationV0):
        raise ValueError('same original stream/frame and opposite owning seat required')
    _binding(peer_category_vocabulary_sha256)
    if peer_category_vocabulary_sha256 != public.category_vocabulary_sha256:
        raise ValueError('peer own tokens must use the same original category namespace')
    peer_own_v4.validate(V4_SPEC)
    if peer_own_v4.perspective is not None and peer_own_v4.perspective.showdown_slot != ('p1', 'p2')[peer_address.owner]:
        raise ValueError('peer observation perspective differs')
    slots = tuple(opponent_slot_map)
    if len(slots) != 6 or any(type(slot) is not int for slot in slots) or set(slots) != set(range(6)):
        raise ValueError('explicit complete physical opponent-slot permutation required')
    peer_cat, peer_num, peer_mask = map(_list,
        (peer_own_v4.categorical_ids, peer_own_v4.numeric_features, peer_own_v4.attention_mask))
    def swap(original, peer):
        return original[:OPPONENT_OFFSET] + tuple(peer[1 + slot] for slot in slots) + original[OPPONENT_OFFSET + 6:]
    return replace(public, categorical_ids=swap(public.categorical_ids, peer_cat),
        numeric_features=swap(public.numeric_features, peer_num),
        attention_mask=swap(public.attention_mask, peer_mask), view='PRIVILEGED')


@dataclass(frozen=True)
class OpponentAction:
    kind: str
    identity: str

    def __post_init__(self):
        if self.kind not in ('MOVE', 'SWITCH'):
            raise ValueError('semantic MOVE or SWITCH label kind required')
        _identifier(self.identity)

    def index(self, vocabulary):
        if self.kind not in ('MOVE', 'SWITCH'):
            raise ValueError('semantic MOVE identity or SWITCH target species required')
        try:
            return vocabulary.opponent_classes.index(self.kind + ':' + self.identity)
        except ValueError as error:
            raise ValueError('opponent action outside bound semantic source namespace') from error


@dataclass(frozen=True)
class HiddenSet:
    species: str
    variant_indices: tuple[int, ...]

    def __post_init__(self):
        _identifier(self.species)
        object.__setattr__(self, 'variant_indices', tuple(self.variant_indices))

    def probabilities(self, vocabulary):
        count = vocabulary.variant_count(self.species)
        indices = tuple(self.variant_indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < count for i in indices)):
            raise ValueError('original matching source variant rows required')
        return tuple(1.0 / len(indices) if i in indices else 0.0 for i in range(MAX_VARIANTS))


@dataclass(frozen=True)
class PokemonEnd:
    survived: bool
    final_hp_fraction: float

    def __post_init__(self):
        hp = _real(self.final_hp_fraction)
        if type(self.survived) is not bool or self.survived != (hp > 0):
            raise ValueError('survival and actual final HP must agree')
        object.__setattr__(self, 'final_hp_fraction', hp)


@dataclass(frozen=True)
class RootImprovement:
    policy_target: tuple[float, ...]
    root_value: float
    completed_q: tuple[float | None, ...]
    worlds: int
    depth: int = 1
    leaf: str = 'PUBLIC_VALUE'
    replies: str = 'OPPONENT_OWN_POLICY'
    label_only: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'policy_target', tuple(self.policy_target))
        object.__setattr__(self, 'completed_q', tuple(self.completed_q))

    def validate(self, legal):
        _real(self.root_value, low=-1, high=1)
        for p in self.policy_target:
            _real(p)
        validate_search_targets(self.policy_target, self.root_value, legal)
        if (type(self.worlds) is not int or self.worlds not in (1, 2) or type(self.depth) is not int
                or self.depth != 1 or self.leaf != 'PUBLIC_VALUE'
                or self.replies != 'OPPONENT_OWN_POLICY' or self.label_only is not True
                or len(self.completed_q) != ACTION_COUNT):
            raise ValueError('D requires label-only depth1,1..2worlds, public leaves and own-seat reply policy')
        for q, active in zip(self.completed_q, legal):
            if active:
                _real(q, low=-1, high=1)
            elif q is not None:
                raise ValueError('illegal actions have no completed-Q target')
        expected = sum(p * q for p, q, active in zip(self.policy_target, self.completed_q, legal) if active)
        if not math.isclose(self.root_value, expected, rel_tol=0, abs_tol=1e-6):
            raise ValueError('root value must equal the improved-policy completed-Q expectation')


def search_target_columns(root_improvement, outcome, legal):
    """Original D policy/value target algebra, also used by direct device rows.

    A missing game outcome never becomes a zero-valued known value target.
    This helper accepts no observation substitute and changes no PPO columns.
    """
    if type(root_improvement) is not RootImprovement:
        raise ValueError('original typed root improvement required')
    if outcome is not None:
        _real(outcome, low=-1, high=1)
        if outcome not in (-1, 0, 1):
            raise ValueError('literal final game outcome must be loss/tie/win')
    root_improvement.validate(legal)
    return dict(search_policy_target=root_improvement.policy_target,
        search_policy_target_mask=True,
        search_value_target=0.0 if outcome is None else (outcome + root_improvement.root_value) / 2,
        search_value_target_mask=outcome is not None)


@dataclass(frozen=True)
class V5Labels:
    address: DecisionAddress
    opponent_next_action: OpponentAction | None = None
    hidden_sets: tuple[HiddenSet | None, ...] = (None,) * 6
    pokemon_end: tuple[PokemonEnd | None, ...] = (None,) * 12
    terminal_status: str = 'IN_PROGRESS'
    outcome: float | None = None
    root_improvement: RootImprovement | None = None
    search_status: str = 'UNSELECTED'
    search_refusal: str | None = None

    def __post_init__(self):
        if type(self.address) is not DecisionAddress:
            raise ValueError('original label decision address required')
        object.__setattr__(self, 'hidden_sets', tuple(self.hidden_sets))
        object.__setattr__(self, 'pokemon_end', tuple(self.pokemon_end))

    def columns(self, observation):
        if self.address != observation.address or observation.view != 'PUBLIC':
            raise ValueError('training labels must bind the same public decision')
        if (len(self.hidden_sets) != 6 or len(self.pokemon_end) != 12
                or self.terminal_status not in ('IN_PROGRESS', 'COMPLETE', 'CAPPED', 'REFUSED')):
            raise ValueError('six opponent and twelve physical Pokemon slots with explicit terminal status required')
        if self.terminal_status != 'COMPLETE' and (self.outcome is not None or any(x is not None for x in self.pokemon_end)):
            raise ValueError('unknown/capped/refused terminal has no outcome/survival/final-HP labels')
        if self.outcome is not None:
            _real(self.outcome, low=-1, high=1)
            if self.outcome not in (-1, 0, 1):
                raise ValueError('literal final game outcome must be loss/tie/win')
        if (self.search_status not in ('UNSELECTED', 'LABELED', 'REFUSED')
                or (self.search_status == 'LABELED') != (type(self.root_improvement) is RootImprovement)
                or self.root_improvement is not None and type(self.root_improvement) is not RootImprovement
                or (self.search_status == 'REFUSED' and not (type(self.search_refusal) is str and bool(self.search_refusal)))
                or (self.search_status != 'REFUSED' and self.search_refusal is not None)
                or self.search_status == 'REFUSED' and self.root_improvement is not None):
            raise ValueError('explicit labeled/refused/unselected search disposition required')
        if self.opponent_next_action is not None and type(self.opponent_next_action) is not OpponentAction:
            raise ValueError('typed semantic opponent action required')
        vocabulary = observation.label_vocabulary
        for label, posterior in zip(self.hidden_sets, observation.posteriors):
            if label is not None and (type(label) is not HiddenSet
                    or posterior.species is not None and label.species != posterior.species):
                raise ValueError('hidden-set label must use the same physical public species slot')
        if any(label is not None and type(label) is not PokemonEnd for label in self.pokemon_end):
            raise ValueError('typed actual terminal Pokemon labels required')
        hidden = tuple(label.probabilities(vocabulary) if label is not None else (0.0,) * MAX_VARIANTS
            for label in self.hidden_sets)
        result = dict(label_vocabulary_sha256=vocabulary.sha256, search_status=self.search_status,
            search_refusal=self.search_refusal,
            opponent_action_target=-1 if self.opponent_next_action is None else self.opponent_next_action.index(vocabulary),
            opponent_action_target_mask=self.opponent_next_action is not None,
            hidden_set_targets=hidden, hidden_set_target_mask=tuple(x is not None for x in self.hidden_sets),
            hidden_set_species=tuple(-1 if x is None else vocabulary.species.index(x.species) for x in self.hidden_sets),
            survival_targets=tuple(0.0 if x is None else float(x.survived) for x in self.pokemon_end),
            final_hp_targets=tuple(0.0 if x is None else x.final_hp_fraction for x in self.pokemon_end),
            pokemon_end_target_mask=tuple(x is not None for x in self.pokemon_end))
        if self.root_improvement is not None:
            result.update(search_target_columns(self.root_improvement, self.outcome,
                observation.legal_action_mask))
        return result


def batch_columns(observations, labels):
    """Pack actual v5 consumer columns, with a one-frame transformer window.

    PPO behavior/action/advantage columns belong to the caller and are never
    rewritten by these training-only targets. This does not relabel a v4 cache.
    """
    observations, labels = tuple(observations), tuple(labels)
    if not observations or len(observations) != len(labels):
        raise ValueError('nonempty aligned v5 observations and labels required')
    if (any(type(o) is not V5Observation or type(t) is not V5Labels for o, t in zip(observations, labels))
            or len({o.label_vocabulary.sha256 for o in observations}) != 1
            or len({o.category_vocabulary_sha256 for o in observations}) != 1):
        raise ValueError('one exact observation/label namespace per v5 batch required')
    rows = [target.columns(obs) for obs, target in zip(observations, labels)]
    result = dict(schema=TRAINING_CONTRACT_SCHEMA, observation_schema=OBSERVATION_SCHEMA_VERSION_V5,
        batch_size=len(rows),
        category_vocabulary_sha256=observations[0].category_vocabulary_sha256,
        label_vocabulary_sha256=observations[0].label_vocabulary.sha256,
        addresses=tuple(asdict(o.address) for o in observations), history_mask=((True,),) * len(rows),
        legal_action_mask=tuple(o.legal_action_mask for o in observations))
    for name in ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask'):
        result[name] = tuple((o.model_inputs()[name],) for o in observations)
    for name in ('hidden_set_targets',
            'hidden_set_target_mask', 'hidden_set_species', 'survival_targets', 'final_hp_targets',
            'pokemon_end_target_mask', 'search_status', 'search_refusal'):
        result[name] = tuple(row[name] for row in rows)
    result['opponent_semantic_action_targets'] = tuple(row['opponent_action_target'] for row in rows)
    result['opponent_semantic_action_target_mask'] = tuple(row['opponent_action_target_mask'] for row in rows)
    for target, source, absent in (('search_policy_targets', 'search_policy_target', (0.0,) * ACTION_COUNT),
            ('search_policy_target_mask', 'search_policy_target_mask', False),
            ('search_value_targets', 'search_value_target', 0.0), ('search_value_target_mask', 'search_value_target_mask', False)):
        result[target] = tuple(row.get(source, absent) for row in rows)
    return result


def warm_start_shape_plan(source_shapes: Mapping, target_shapes: Mapping, *,
                          source_category_vocabulary_sha256, target_category_vocabulary_sha256,
                          source_schema=OBSERVATION_SCHEMA_VERSION_V4, target_schema=OBSERVATION_SCHEMA_VERSION_V5):
    """Plan normal named/shape-compatible copies without reading model tensors.

    Belief tokens append rows, so numeric/category projections need no resizing.
    New normalization/expert/head names remain fresh. Production uses scratch.
    """
    _binding(source_category_vocabulary_sha256); _binding(target_category_vocabulary_sha256)
    if (source_schema != OBSERVATION_SCHEMA_VERSION_V4 or target_schema != OBSERVATION_SCHEMA_VERSION_V5
            or source_category_vocabulary_sha256 != target_category_vocabulary_sha256):
        raise ValueError('v4 category rows/OOV layout must retain their original namespace')
    for name in ('category_embedding.weight', 'token_type_embedding.weight', 'numeric_projection.weight'):
        if name not in source_shapes or name not in target_shapes or tuple(source_shapes[name]) != tuple(target_shapes[name]):
            raise ValueError('append-only input/trunk tensor shape mismatch: ' + name)
    projection = tuple(target_shapes['numeric_projection.weight'])
    categories = tuple(target_shapes['category_embedding.weight'])
    roles = tuple(target_shapes['token_type_embedding.weight'])
    if (len(projection) != 2 or projection[1] != NUMERIC_FEATURES or len(categories) != 2
            or len(roles) != 2 or roles[0] < 8 or projection[0] != categories[1] or projection[0] != roles[1]):
        raise ValueError('v4 feature projection/embedding widths and appended role7 must fit')
    copied = tuple(sorted(name for name, shape in target_shapes.items()
        if name in source_shapes and tuple(shape) == tuple(source_shapes[name])))
    return dict(copy=copied, initialize=tuple(sorted(set(target_shapes) - set(copied))),
        unused_source=tuple(sorted(set(source_shapes) - set(copied))), production_initialization='scratch')


def phase_c_arms():
    """Initial four allocations are experiment identities, never RNG seed values."""
    common = dict(observation_schema=OBSERVATION_SCHEMA_VERSION_V5, initialization='scratch',
        width=512, layers=3, attention_heads=8, norm='pre_rmsnorm', feedforward='token_type_moe',
        expert_width=512, expert_roles=('pokemon', 'action', 'field_belief'),
        privileged_critic=False, actor='RAW_POLICY', search_depth=0, label_fraction=0.0)
    control = dict(common, arm='A', replicate=0, value_head='mean_linear_tanh', auxiliary_heads=False,
        loss_profile='CURRENT_PPO')
    treatment = dict(common, arm='B', replicate=0, value_head='attention_two_layer_mlp_hl_gauss',
        auxiliary_heads=True, loss_profile='VALUE_AUX_PPO', value_clip=None, value_lambda=0.95,
        value_hidden=(512, 512), value_bins=101, value_support=(-1.0, 1.0), value_sigma_ratio=0.75,
        entropy_bonus=True, uniform_kl='UNIFORM_TO_POLICY_ON_LEGAL_ACTIONS', historical_opponent_pool=True)
    return (control, treatment, dict(treatment, arm='D', search_depth=1, label_fraction=0.25,
        sampled_worlds=(1, 2), leaf='PUBLIC_VALUE', replies='OPPONENT_OWN_POLICY', label_only=True),
        dict(treatment, replicate=1))


def phase_c_next_step(*, endpoint_complete, b_minus_a_lower90, d_minus_b_lower90):
    if endpoint_complete is not True or b_minus_a_lower90 is None or d_minus_b_lower90 is None:
        raise ValueError('both endpoint contrasts required before selecting the next round')
    b = _real(b_minus_a_lower90, low=-1, high=1)
    d = _real(d_minus_b_lower90, low=-1, high=1)
    if d > 0:
        return 'ACT_WITH_SEARCH_NEXT_FREE_GPU'
    return 'ADVANCE_B' if b > 0 else 'RUN_C_ON_B_FREED_GPUS'


def contract_descriptor():
    return dict(schema=TRAINING_CONTRACT_SCHEMA, observation_schema=OBSERVATION_SCHEMA_VERSION_V5,
        base_schema=OBSERVATION_SCHEMA_VERSION_V4, base_tokens=V4_TOKENS, tokens=V5_TOKENS,
        numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES,
        belief_type=BELIEF_TYPE, belief_variant_capacity=MAX_VARIANTS,
        pinned_source_census=dict(PINNED_SOURCE_CENSUS),
        auxiliary_targets=('opponent_semantic_action', 'hidden_set_species', 'hidden_set_variant_rows',
            'pokemon_survival', 'pokemon_final_hp'),
        initial_arms=phase_c_arms(), decisions=dict(DECISIONS),
        consumer_integration='EXPLICIT_V5_CONSUMER_REQUIRED_LEGACY_DEFAULT_UNCHANGED')
