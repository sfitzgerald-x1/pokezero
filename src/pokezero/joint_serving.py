"""Public protocol adapter for one joint draw and explicit stage-1 handoff.

Only owning protocol lines, the exact original own request and public beliefs
enter this boundary. Current party coordinates never become generation slots.
"""
from collections import OrderedDict
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path
import random

from . import determinization as worlds, set_posterior
from .dex import normalize_id
from .generator_weights import load_generator_weights, source_manifest
from .joint_native import draw_with_stage1_fallback
from .randbat import canonical_gen3_randbat_species_id
from .tier2 import canonical_move_id


@dataclass(frozen=True)
class JointPlan:
    teams: tuple | None
    posts: object
    feature_payload: object
    diagnostic: object
    source_hash: str
    game_sha256: str
    fallback_plan: object = None


class JointSession:
    """Source snapshot and bounded, game-local legality cache ownership."""
    def __init__(self, source, *, producer, held_games=64, deadline=None):
        if producer not in ('shared-battle', 'independent-teams'):
            raise ValueError('explicit team producer required')
        self.source, self.producer, self.held_games = source, producer, held_games
        self.manifest = source_manifest(source)
        self.table = load_generator_weights(source)
        self.model = json.loads((Path(__file__).with_name('data') / 'gen3_team_model.json').read_text())
        self.games = OrderedDict()

    def cache(self, game, *, deadline=None):
        from .joint_budget import check_deadline
        check_deadline(deadline,.175 if deadline is not None else None)
        import pokezero_search
        if game not in self.games:
            self.games[game] = pokezero_search.JointLegalityCache()
        self.games.move_to_end(game)
        while len(self.games) > self.held_games:
            self.games.popitem(last=False)
        check_deadline(deadline,.175 if deadline is not None else None)
        return self.games[game]

    def _fallback_payload(self, posts, game):
        rows = {canonical_gen3_randbat_species_id(key): dict(
            variant_ids=self.table.species[canonical_gen3_randbat_species_id(key)].variant_ids,
            support=post.support, weights=post.weights) for key, post in posts.items()}
        prior = {s: dict(row, support=(True,)*len(row['weights']), weights=(1,)*len(row['weights']))
                 for s, row in rows.items()}
        return dict(schema='gen3-joint-features-v1', source_hash=self.table.source_hash, table_sha256='stage1',
            model_sha256=self.model['sha256'], rows=rows, prior_rows=prior, tv_bound=None, prior_tv_bound=None,
            provenance=hashlib.sha256(json.dumps((game,self.table.source_hash,'stage1',rows),
                sort_keys=True,separators=(',',':')).encode()).hexdigest())

    def _plan(self, teams, posts, payload, mode, reason, tv, game, original_posts, fallback=None):
        if payload is not None:
            payload['game_sha256'] = game
            payload['payload_sha256'] = hashlib.sha256(json.dumps(payload,sort_keys=True,
                separators=(',',':')).encode()).hexdigest()
        diagnostic = dict(kind='PUBLIC_JOINT_POSTERIOR', status='CLEAR', fatal=False, mode=mode,
            tv_bound=None if tv is None else [tv.numerator,tv.denominator], seconds=0.0,
            fallback=mode=='stage1_fallback', reason=reason, source_hash=self.table.source_hash,
            table_sha256=self.table.provenance, model_sha256=self.model['sha256'], game_sha256=game,
            producer=self.producer, owner=None,
            over_budget=False, allocation=dict(expansion_stop=.175,normalizer_return=.200,export_stop=.225,
                handoff=.250,max_states=65_536))
        diagnostic['narrowed_vs_hard'] = any(sum(p.support)<sum(p.hard) for p in posts.values())
        diagnostic['narrowed_vs_stage1'] = any(sum(p.support)<sum(original_posts[k].support) for k,p in posts.items())
        draws = None if teams is None else [[(m.species,m.variant_index,m.slot,m.incoming_context) for m in team] for team in teams]
        diagnostic['law_sha256'] = hashlib.sha256(json.dumps((game,self.table.provenance,self.model['sha256'],
            mode,diagnostic['tv_bound'],draws,payload),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return JointPlan(teams,posts,payload,diagnostic,self.table.source_hash,game,fallback)

    @staticmethod
    def finish_handoff(plan, deadline, *, elapsed_seconds=None, remaining_seconds=None):
        # Called after the GC guard restores its prior state. Both plans and
        # their hashes already exist, so choosing fallback allocates no plan.
        elapsed = (elapsed_seconds or deadline.elapsed_seconds)()
        late = elapsed > .25 or (remaining_seconds or deadline.remaining_seconds)() <= 0
        if late and plan.teams is not None:
            plan = plan.fallback_plan
            plan.diagnostic['reason'] = 'JointBudgetExceeded:completed handoff exceeded deadline'
        plan.diagnostic['seconds'] = elapsed
        plan.diagnostic['over_budget'] = late
        return plan

    def prepare(self, context, lines, *, dex, seed, count, binding=None, features=False, deadline=None):
        from .joint_budget import check_deadline, defer_joint_gc, new_joint_deadline
        if deadline is None:
            with defer_joint_gc():
                deadline = new_joint_deadline()
                plan = self.prepare(context,lines,dex=dex,seed=seed,count=count,binding=binding,
                    features=features,deadline=deadline)
                elapsed,remaining = deadline.elapsed_seconds,deadline.remaining_seconds
            return self.finish_handoff(plan,deadline,elapsed_seconds=elapsed,remaining_seconds=remaining)
        view = worlds.player_belief_view_from_payload(context.observation.metadata.get('belief_view'))
        if view is None:
            raise ValueError('PUBLIC_SET_POSTERIOR_BELIEF_UNAVAILABLE')
        posts = set_posterior.posterior(lines,context.public_materialization_state.self_request,
            source=self.source,owner=view.self_slot,beliefs=view.opponent_pokemon,
            format_id=context.format_id,dex=dex,binding=binding)
        stage1_posts = {}
        evidence = self._evidence(context,view,posts,stage1_posts=stage1_posts)
        fixed = self._generation_slots(lines,view.opponent_slot)
        game = set_posterior.game_lines_sha256(lines)
        fallback = self._plan(None,stage1_posts,self._fallback_payload(stage1_posts,game) if features else None,
            'stage1_fallback',None,None,game,stage1_posts)
        fallback.diagnostic['owner'] = view.self_slot
        try:
            check_deadline(deadline,.175)
            try:
                own = self._own_initial(lines)
            except (KeyError,ValueError) as error:
                fallback.diagnostic['reason'] = type(error).__name__+':'+str(error)
                return fallback
            check_deadline(deadline,.175)
            outcome = draw_with_stage1_fallback(self.model,self.manifest,self.table,
                count=count,rng=random.Random(seed),stage1_sampler=lambda count,rng:None,
                legality_cache=self.cache(context.battle_id,deadline=deadline),deadline=deadline,
                battle_context=dict(owner=view.self_slot,own_initial_team=own,producer=self.producer),
                evidence=evidence,fixed_slots=fixed,export_features=features,export_posterior=True)
            if outcome.teams is None:
                fallback.diagnostic['reason'] = outcome.reason
                return fallback
            payload = outcome.features
            posts = {key: replace(post,weights=tuple(payload['rows'][canonical_gen3_randbat_species_id(key)]['weights']),
                support=tuple(payload['rows'][canonical_gen3_randbat_species_id(key)]['support']),
                provenance=payload['provenance']) for key,post in posts.items()}
            plan = self._plan(outcome.teams,posts,payload,outcome.mode,None,outcome.tv_bound,game,stage1_posts,fallback)
            plan.diagnostic['owner'] = view.self_slot
            check_deadline(deadline)
            return plan
        except TimeoutError as error:
            # Invalid original own recipes already used stage1; budget expiry
            # has the same explicit handoff, never a partial joint payload.
            fallback.diagnostic['reason'] = type(error).__name__+':'+str(error)
            return fallback

    def _own_initial(self, lines):
        request = set_posterior._first_request(lines)
        if request is None:
            raise ValueError('original own request unavailable for joint conditioning')
        result = []
        for mon in request['side']['pokemon']:
            details = mon['details']
            species = canonical_gen3_randbat_species_id(details.split(',')[0])
            level = worlds._level_from_details(details) or 100
            moves = {canonical_move_id(m) for m in mon['moves']}
            if 'hiddenpower' in moves:
                if not mon.get('hpType'):
                    raise ValueError('original Hidden Power type unavailable')
                moves.remove('hiddenpower'); moves.add('hiddenpower' + normalize_id(mon['hpType']))
            ability = normalize_id(mon.get('baseAbility', mon.get('ability', '')))
            item = normalize_id(mon.get('item', ''))
            variants = self.manifest['species'][species]
            matches = [i for i, v in enumerate(variants) if v['level'] == level
                and set(map(canonical_move_id, v['moves'])) == moves
                and normalize_id(v['ability']) == ability and normalize_id(v['item']) == item]
            if not matches:
                raise ValueError('original own set outside pinned source: ' + species)
            variant=variants[matches[0]]
            fixture=worlds._fixture_from_variant_payload(variant,fallback_species=species,set_source=self.source)
            base=worlds._base_stats_for_species(self.source,species)
            if fixture is None or base is None:raise ValueError('original own spread unavailable')
            expected=worlds._computed_stats(base,evs=fixture.evs,ivs=fixture.ivs,level=level)
            if any(mon.get('stats',{}).get(key)!=expected[key]for key in('atk','def','spa','spd','spe')):
                raise ValueError('original own stats outside pinned recipe: '+species)
            condition=str(mon.get('condition','')).split()[0]
            if '/'not in condition or int(condition.split('/')[1])!=expected['hp']:
                raise ValueError('original own maximum HP outside pinned recipe: '+species)
            result.append((species, matches[0]))
        if len(result) != 6:
            raise ValueError('six original own members required')
        return tuple(result)

    @staticmethod
    def _generation_slots(lines, foe):
        # The opening switch is the original lead. Later switch/request slots
        # are current party coordinates and cannot constrain generation order.
        for line in lines:
            p = line.split('|')
            if len(p) > 3 and p[1] == 'switch' and p[2].startswith(foe):
                return {0: canonical_gen3_randbat_species_id(p[3].split(',')[0])}
        return {}

    def _evidence(self, context, view, posts,*,stage1_posts=None):
        constraints = worlds._public_opponent_move_slot_constraints(context, view.opponent_slot)
        evidence = {}
        for belief in view.opponent_pokemon:
            species = canonical_gen3_randbat_species_id(belief.species)
            post = posts[set_posterior.species_key(belief.species)]
            slot = constraints.get(worlds._normalize_species_id(belief.species), {})
            hp = worlds._condition_max_hp(belief.condition)
            likelihood = []
            sampler_indices=[]
            for index,(variant, weight) in enumerate(zip(self.source.universe_for(belief.species).variants, post.weights)):
                payload = variant.to_summary()
                valid = (not hp or worlds._variant_public_max_hp(payload, fallback_species=belief.species,
                         set_source=self.source) == hp)
                if valid and slot:
                    valid = worlds._fixture_from_variant_payload(payload, fallback_species=belief.species,
                        set_source=self.source, move_slot_constraints=slot, gender=belief.gender) is not None
                if valid and post.hard[index]:sampler_indices.append(index)
                original_item=getattr(belief,'original_public_item',None)
                if original_item:
                    valid=valid and normalize_id(variant.item or '')==normalize_id(original_item)
                likelihood.append(weight if valid else 0)
            if stage1_posts is not None:
                stage1_posts[set_posterior.species_key(belief.species)]=set_posterior.restrict(post,sampler_indices)if sampler_indices else post
            evidence[species] = tuple(likelihood)
        return evidence


def opponent_team(plan, index, *, context, source, rng):
    """Materialize the jointly sampled sets, preserving public party fitting."""
    if plan.teams is None or plan.source_hash != source.metadata.source_hash:
        raise ValueError('admitted source-bound joint team required')
    view = worlds.player_belief_view_from_payload(context.observation.metadata.get('belief_view'))
    shown = {canonical_gen3_randbat_species_id(b.species): b for b in view.opponent_pokemon}
    constraints = worlds._public_opponent_move_slot_constraints(context, view.opponent_slot)
    fixtures = []
    for member in plan.teams[index]:
        universe = source.universes[member.species]
        variant = universe.variants[member.variant_index]
        belief = shown.get(member.species)
        public_species=belief.species if belief is not None else variant.species
        gender = belief.gender if belief is not None else None
        gender = gender or worlds._sample_gender(public_species, set_source=source, rng=rng)
        fixture = worlds._fixture_from_variant_payload(variant.to_summary(), fallback_species=public_species,
            set_source=source, gender=gender,
            move_slot_constraints=constraints.get(worlds._normalize_species_id(public_species), {}))
        if fixture is None:
            raise ValueError('joint source fixture unavailable')
        fixtures.append(fixture)
    slots = worlds._public_opponent_team_index_constraints(context, opponent_slot=view.opponent_slot, team_size=6)
    if slots is None:
        raise ValueError('public opponent switch constraints are inconsistent')
    fitted = worlds._fixture_team_with_index_constraints(tuple(fixtures), slots, team_size=6,
        public_context=context, opponent_slot=view.opponent_slot)
    if fitted is None:
        raise ValueError('joint team could not satisfy public party constraints')
    return fitted
