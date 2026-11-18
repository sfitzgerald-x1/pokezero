"""Native joint posterior transport and explicit certified truncation policy."""
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from time import perf_counter
from .joint_generator import JointGenerator, JointMember, JointBudgetExceeded, ImpossibleJointEvidence, _pick
from .generator_weights import SHOWDOWN_PIN, EXECUTED_PIN, digest


def validate_pinned_catalog(model, manifest, table):
    """Serving accepts only the measured catalog; generic oracles stay separate."""
    model_hash = '74d83850ba3cb47ced59caa136e2775d8dc6d7ee664e06dd8e3e80dce35d7aa9'
    if (model.get('schema') != 'gen3-joint-team-model-v1'
            or model.get('showdown_commit') != SHOWDOWN_PIN or model.get('executed_sha256') != EXECUTED_PIN
            or model.get('sha256') != model_hash
            or digest({k: v for k, v in model.items() if k != 'sha256'}) != model_hash):
        raise ValueError('unsupported pinned team catalog')
    if (table.provenance != '22c41e107150650a76d274946acad097a14d017806f9c07d472e3565b1df2528'
            or table.source_hash != 'f5a5265143d423af'
            or table.variant_manifest_sha256 != 'e582efbaeaac4feeeb8f438ef7ebbe153c47a9a4206340d829221192fe2019c8'
            or digest(manifest) != table.variant_manifest_sha256):
        raise ValueError('unsupported pinned weight catalog')
    rows = {s: dict(denominator=str(w.denominator), variant_ids=w.variant_ids,
                   contexts={str(c): tuple(map(str, row)) for c, row in enumerate(w.contexts)})
            for s, w in table.species.items()}
    row_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if row_hash != 'cb598f07d970aaa79a84756e4751367ed8054c46364d82b59f2d8518d677810b':
        raise ValueError('unsupported pinned weight contents')


def own_team_terminal_weights(model, manifest, table, own_team):
    """Likelihood of the exact ORIGINAL second team for each incoming flag.

    Own entries are (source species ID, variant index) in generation order.
    The two integer likelihoods share an omitted common denominator. This is
    the terminal factor on the first team's outgoing Wobbuffet flag.
    """
    if len(own_team) != model["slots"] or len({s for s, _ in own_team}) != len(own_team):
        raise ValueError("complete original own team required")
    evidence = {}
    for species, vi in own_team:
        variants = manifest["species"][species]
        if type(vi) is not int or not 0 <= vi < len(variants):
            raise ValueError("invalid own variant")
        def identity(v):
            return (v['species'], v['level'], tuple(sorted(v['moves'])), v['ability'], v['item'])
        physical = identity(variants[vi])
        evidence[species] = tuple(int(identity(v) == physical) for v in variants)
    fixed = {slot: species for slot, (species, _) in enumerate(own_team)}
    probabilities = []
    for flag in (False, True):
        try:
            generator = JointGenerator(model, manifest, table, evidence=evidence, fixed_slots=fixed, initial_wobbuffet=flag)
            probabilities.append(Fraction(generator.normalizer(), generator.quantum ** generator.slots))
        except ImpossibleJointEvidence:
            probabilities.append(Fraction(0))
    den = lcm(*(p.denominator for p in probabilities))
    values = tuple(p.numerator * (den // p.denominator) for p in probabilities)
    common = gcd(*values)
    if not common:
        raise ImpossibleJointEvidence("own original team is outside the generator")
    return tuple(w // common for w in values)


def battle_joint_parameters(model, manifest, table, *, owner, own_initial_team, producer):
    """Select forward/reverse conditioning from the explicitly bound producer.

    Battle generates p1 then p2 using one shared generator. Independently
    generated team pools must declare their different flag semantics.
    """
    if owner not in ('p1', 'p2') or producer not in ('shared-battle', 'independent-teams'):
        raise ValueError('explicit supported owner and team producer required')
    if len(own_initial_team) != model['slots']:
        raise ValueError('complete original own team required')
    if producer == 'independent-teams':
        return dict(initial_wobbuffet=False, terminal_weights=(1, 1))
    if owner == 'p1':
        flag = any(species == 'wobbuffet' for species, _ in own_initial_team[:-1])
        return dict(initial_wobbuffet=flag, terminal_weights=(1, 1))
    return dict(initial_wobbuffet=False, terminal_weights=own_team_terminal_weights(model, manifest, table, own_initial_team))


def native_payload(joint):
    # Catalog classes exclude evidence and incoming battle flags, making the
    # legality cache reusable as new members/sets are revealed in the game.
    profiles, classes, base_class = {}, [], {}
    for bi, base in enumerate(joint.bases):
        profile = (tuple(base['types']), tuple(base['weak']), tuple(base['double_weak']),
                   base['level100'], tuple(base['incompatible']))
        if profile not in profiles:
            profiles[profile] = len(classes); classes.append(base)
        base_class[bi] = profiles[profile]
    caps = []
    for field, limit in (('types', 2), ('weak', 3), ('double_weak', 1)):
        for kind in joint.model['types']:
            members = tuple(i for i, base in enumerate(classes) if kind in base[field])
            if members and (members, limit) not in caps:
                caps.append((members, limit))
    caps.append((tuple(i for i, base in enumerate(classes) if base['level100']), 1))
    if len(caps) > 64 or len(classes) > 128:
        raise ValueError('native legality capacity exceeded')
    masks = [sum(1 << j for j, (members, _) in enumerate(caps) if i in members) for i in range(len(classes))]
    incompatible = [[j for j, other in enumerate(classes) if other['name'] in base['incompatible']] for base in classes]
    limits = [sum(1 << j for j, (_, limit) in enumerate(caps) if limit == n) for n in (1, 2, 3)]
    groups = []
    required = {joint.base_group[bi]: 1 << i for i, bi in enumerate(joint.required_bases)}
    for gi, members in enumerate(joint.groups):
        bi = members[0]
        upper = max(joint.evidence[joint.required_bases[bi]]) if bi in joint.required_bases else 1
        groups.append(dict(size=len(members), tokens=joint.group_tokens[gi], **{'class': base_class[bi]},
                           required=required.get(gi, 0), fixed=joint.fixed_group.get(gi), wobb=gi in joint.wobb_groups,
                           upper=str(upper), rows=[[str(den), [[after, str(n)] for after, n in row]] for den, row in joint.group_rows[gi]]))
    return json.dumps(dict(slots=joint.slots, quantum=str(joint.quantum), terminal=list(map(str, joint.terminal_weights)), initial_flag=joint.initial_wobbuffet,
                           completable=joint.certified_completion, groups=groups,
                           legality=dict(masks=masks, limits=limits, incompatible=incompatible)), separators=(',', ':'))


class NativeJointPosterior:
    """One root, retaining a caller-owned legality cache across root changes.

    solve() may return a rigorously truncated subdistribution. It is drawable
    only after the caller explicitly admits its exact rational TV certificate.
    A failed certificate raises; the serving caller owns stage-1 fallback.
    """
    def __init__(self, model, manifest, table, *, legality_cache=None, **kwargs):
        import pokezero_search as native
        self.reference = JointGenerator(model, manifest, table, **kwargs)
        self.legality_cache = legality_cache if legality_cache is not None else native.JointLegalityCache()
        self.native = native.NativeJointGenerator(native_payload(self.reference), self.legality_cache)
        self.accepted = False
        self.result = None

    def solve(self, *, seconds, max_states=2_000_000, max_tv=Fraction(0)):
        if not isinstance(max_tv, Fraction) or not 0 <= max_tv <= Fraction(1, 100):
            raise ValueError('explicit rational TV limit in [0, .01] required')
        self.result = json.loads(self.native.solve(seconds, max_states))
        lower, upper = (int(self.result[key]) for key in ('lower', 'upper'))
        if not upper:
            raise ImpossibleJointEvidence('no generator team satisfies the evidence')
        self.tv_bound = Fraction(upper - lower, upper)
        self.accepted = lower > 0 and self.tv_bound <= max_tv
        if not self.accepted:
            raise JointBudgetExceeded(f'joint certificate exceeds TV limit: {self.tv_bound} > {max_tv}')
        return self.result

    def draw_many(self, count, rng):
        if not self.accepted:
            raise JointBudgetExceeded('joint posterior is not certified for drawing')
        paths = self.native.draw_paths(count, rng.getrandbits(64))
        joint = self.reference
        result = []
        for path in paths:
            selected, team, context = set(), [], 0
            for slot, (gi, after) in enumerate(path):
                candidates = [bi for bi in joint.groups[gi] if bi not in selected]
                bi = candidates[rng.randrange(len(candidates))]
                options = [(entry, weight) for entry, weight in joint._variant_options(bi, context) if entry[2] == after]
                (species, vi, _) = _pick(options, rng)
                team.append(JointMember(species, vi, slot, context)); selected.add(bi); context = after
            result.append(tuple(team))
        return tuple(result)

    def revealed_marginal_weights(self):
        """Integer marginals of the admitted draw distribution, in source order.

        Truncated zero weights are not proofs of impossibility. Consumers
        must retain a conservative support pin when tv_bound is nonzero.
        """
        if not self.accepted:
            raise JointBudgetExceeded('joint posterior is not certified for features')
        joint = self.reference
        masses = {s: [Fraction(0)] * len(joint.manifest['species'][s]) for s in joint.evidence}
        for gi, before, after, value in json.loads(self.native.revealed_context_masses()):
            members = joint.groups[gi]
            if len(members) != 1:
                raise AssertionError('required group is not a singleton')
            options = [(entry, weight) for entry, weight in joint._variant_options(members[0], before) if entry[2] == after]
            total = sum(w for _, w in options)
            for (species, vi, _), weight in options:
                masses[species][vi] += Fraction(int(value) * weight, total)
        result = {}
        for species, row in masses.items():
            if sum(row) != int(self.result['lower']):
                raise AssertionError('revealed marginal mass differs from root normalizer')
            denominator = lcm(*(v.denominator for v in row))
            counts = tuple(v.numerator * (denominator // v.denominator) for v in row)
            common = gcd(*counts)
            result[species] = tuple(v // common for v in counts)
        return result

    def feature_payload(self):
        """Source-bound support/weight export for the opt-in feature consumer.

        An approximate zero is never a hard pin. For an admitted truncated
        distribution, upper support retains every hard/strike-compatible
        variant reachable in any generator context. This can over-include;
        it cannot exclude mass merely because the DP did not expand it.
        """
        weights = self.revealed_marginal_weights()
        joint = self.reference
        rows = {}
        for species, values in weights.items():
            support = tuple(w > 0 for w in values)
            if self.tv_bound:
                contexts = (0,) if joint.fixed_species.get(0) == species else range(16)
                support = tuple(likelihood > 0 and any(joint.table.species[species].contexts[c][vi] > 0 for c in contexts)
                                for vi, likelihood in enumerate(joint.evidence[species]))
            rows[species] = dict(variant_ids=joint.table.species[species].variant_ids, support=support, weights=values)
        binding = dict(schema='gen3-joint-features-v1', source_hash=joint.table.source_hash,
                       table_sha256=joint.table.provenance, model_sha256=joint.model.get('sha256'),
                       evidence=joint.evidence, fixed_slots=joint.fixed_species,
                       initial_wobbuffet=joint.initial_wobbuffet, terminal_weights=joint.terminal_weights,
                       tv_bound=(self.tv_bound.numerator, self.tv_bound.denominator), rows=rows)
        provenance = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return dict(schema=binding['schema'], provenance=provenance, source_hash=binding['source_hash'],
                    table_sha256=binding['table_sha256'], model_sha256=binding['model_sha256'],
                    tv_bound=binding['tv_bound'], rows=rows)


@dataclass(frozen=True)
class JointDrawOutcome:
    teams: object
    mode: str
    tv_bound: Fraction | None
    elapsed_seconds: float
    reason: str | None
    features: object = None


def draw_with_stage1_fallback(model, manifest, table, *, count, rng, stage1_sampler,
                              budget_seconds=0.25, draw_reserve_seconds=0.05,
                              legality_cache=None, battle_context, export_features=False, **kwargs):
    """Try the authorized TV<=.01 joint draw; preserve the stage-1 RNG on fallback.

    stage1_sampler(count, rng) is the existing per-slot consumer, supplied by
    the integration owner. A fallback does not claim a joint TV guarantee.
    The budget includes preparation and W-world draws. Late certificates or
    draws are discarded and explicitly diagnosed instead of being served.
    """
    if not 0 < draw_reserve_seconds < budget_seconds or count <= 0:
        raise ValueError('invalid joint draw budget/count')
    started = perf_counter()
    validate_pinned_catalog(model, manifest, table)
    if not isinstance(battle_context, dict) or set(battle_context) != {'owner', 'own_initial_team', 'producer'}:
        raise ValueError('explicit complete battle context required')
    if 'initial_wobbuffet' in kwargs or 'terminal_weights' in kwargs:
        raise ValueError('battle context and explicit flag weights are mutually exclusive')
    rng_state = rng.getstate()
    tv, reason, joint = None, None, None
    try:
        kwargs.update(battle_joint_parameters(model, manifest, table, **battle_context))
        joint = NativeJointPosterior(model, manifest, table, legality_cache=legality_cache, **kwargs)
        remaining = budget_seconds - draw_reserve_seconds - (perf_counter() - started)
        if remaining <= 0:
            raise JointBudgetExceeded('joint construction consumed normalizer budget')
        joint.solve(seconds=remaining, max_tv=Fraction(1, 100))
        tv = joint.tv_bound
        if perf_counter() - started > budget_seconds - draw_reserve_seconds:
            raise JointBudgetExceeded('certificate completed after normalizer deadline')
        teams = joint.draw_many(count, rng)
        features = joint.feature_payload() if export_features else None
        if export_features:
            prior_options=dict(kwargs)
            prior_options['evidence']={s:(1,)*len(w) for s,w in joint.reference.evidence.items()}
            prior=NativeJointPosterior(model,manifest,table,legality_cache=legality_cache,**prior_options)
            remaining=budget_seconds-(perf_counter()-started)
            if remaining<=0:raise JointBudgetExceeded('feature prior construction exceeded deadline')
            prior.solve(seconds=remaining,max_tv=Fraction(1,100))
            prior_features=prior.feature_payload()
            features['prior_rows']=prior_features['rows']
            features['prior_tv_bound']=prior_features['tv_bound']
            features['provenance']=hashlib.sha256(json.dumps((features['provenance'],prior_features['provenance']),
                separators=(',',':')).encode()).hexdigest()
            del prior
        mode = 'exact' if joint.result['exact'] else 'certified'
        del joint
        joint = None
        elapsed = perf_counter() - started
        if elapsed > budget_seconds:
            raise JointBudgetExceeded('joint draw completed after deadline')
        return JointDrawOutcome(teams, mode, tv, elapsed, None, features)
    except (JointBudgetExceeded, ImpossibleJointEvidence) as exc:
        reason = type(exc).__name__ + ':' + str(exc)
    del joint
    rng.setstate(rng_state)
    teams = stage1_sampler(count, rng)
    return JointDrawOutcome(teams, 'stage1_fallback', None, perf_counter() - started, reason)
