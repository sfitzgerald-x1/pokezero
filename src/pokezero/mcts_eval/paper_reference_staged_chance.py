"""Opt-in, guarded constant-chance conditioning for a public replay program.

This accelerates a conditioning kernel, not search or strength selection. It
never reads the real opponent's action or real hidden state. Unsupported public
programs retain the original joint kernel. A supported program whose sampled
active context violates the certificate REFUSES; it must not reject that latent
team as though it had zero likelihood.

Law: for hidden anchor theta and retained private prefix xi, each guarded public
stage event E_i has the same chance-seed predicate for every eligible theta/xi.
Thus its likelihood c_i (under the existing 64-bit seed-to-PRNG mapping) is
constant. Sampling the FULL champion policy once, rejecting incompatible actions,
and conditioning chance alone divides the joint path density by product(c_i),
a common constant. Finite K retry factors product(1-(1-c_i)**K) are common too.
The actual accepted chance/private state is retained. No independent-uniform
damage/crit assumption, forced champion action, or uniform Substitute HP is used.

The first certificate is a move/state program, NOT a study-seed whitelist:
Lugia Psychic against the source-fixed Shuckle set; fresh Substitute, surviving
Encore, then Wrap / Rest / one or two Rest-blocked sleep turns. Other programs
remain joint. The fourth turn has a distinct certificate; the historical
three-turn certificate is not relabeled.
Scientific use still requires independent observation and fresh qualification.
"""
from dataclasses import dataclass
import hashlib
import math

from .paper_reference import ReferenceRefusal
from .paper_reference_pending import public_history

SCHEMA = 'pokezero.constant-chance-substitute.v1'
REST_TAIL_SCHEMA = 'pokezero.constant-chance-substitute.rest-tail.v2'
JOINT_TAIL_SCHEMA = 'pokezero.constant-chance-prefix-joint-tail.v3'
SOURCE_HASH = 'f5a5265143d423af'
MAX_CHANCE_ATTEMPTS = 2048
COMPILED_TREE_FILES = 500
COMPILED_TREE_SHA256 = '1ee403ea5274fe32bc30329ead814c4c117ff548f5f576242d7f0c3d108642b1'
ENGINE_HASHES = {
    "dist/sim/battle.js": "3b2d22e90d30286c5b31c227cbb14d2812b92cdcd122a0e03040542eb5f0ded5",
    "dist/sim/battle-actions.js": "a88494530b8c454baa39501f0ec6526010cd721de6377750db876e598c6c9109",
    "dist/sim/pokemon.js": "2852489c58aee5dbf15317d980cd078ae136a83e0d2ee63f2977eec10ba26aa7",
    "dist/sim/state.js": "1a61018be79a8d2a591a129e00379d2d491d8283daed2b8e95cc26bf93f3d540",
    "dist/sim/prng.js": "327745d5c0bb090c8d1c0d52a89528533ca3822333dd5346e91c36bbbb4c15b8",
    "dist/data/mods/gen3/scripts.js": "94aca2c90fee0a9717403476adbdcf1792e627a51a33813f25fd027bec6770da",
    "dist/data/mods/gen3/conditions.js": "b465bee5b5c2997bcfdddca9a9c76dcaf51db1a6fc015929649ab3724d21405c",
    "dist/data/mods/gen3/moves.js": "994c1d38339508cbd9ff084cdabe31e44464d9bdab0bbdbb545fd15fbe74d089",
    "dist/data/mods/gen4/moves.js": "c8ce90fd3fe3dfd652162aa67adb9a293bfc56940b1c1a1dd9218bad43c476ec",
    "dist/data/mods/gen4/abilities.js": "abea72b57790028f4ba2c7afe615760af66efd2bc69e690e0fbd0b2b81dab915",
    "dist/data/moves.js": "fe51b8838705f803c402515154ad8685c648de3bb38fc99dc8a669e23ac9a504",
    "dist/data/items.js": "6418cffb3daa00094f07ff187a37aff7b6ae06f68330c9f9b3202cfd3da21ab3",
    "dist/data/abilities.js": "4364cf3f3ebea58318154114e0b784c0f35b754385f844bbcf58568f07bd630d",
    "dist/data/rulesets.js": "ae7075a4e576a481713962b24768f970831efb3a7947872ea17ec8501471c968",
    "data/random-battles/gen3/teams.ts": "c41f6a1e60688d29cfdc1171028b17c3b7bc75e205c956d84d012e78ec2e0a67",
    "data/random-battles/gen3/sets.json": "367bf8b3a45700aa6c0bc44089b8a4db623dd909f88d7fc2935a066418d3d4cb"
}
ENGINE_HASHES.update({
    'dist/data/random-battles/gen3/teams.js': 'b4db5c8a2bc7e5fc0d523649433da2085613061723dbddbd2c75d715c11494d3',
    'dist/config/formats.js': '6ae56a3ff1d824b0dec728db69b3ade472694fdc0798bd7bf1f836e4b2790601',
    'dist/sim/dex.js': '903574084d94cf03ea1e10149fa5cf1a1da00132256bb0201c2aea6a80215c88',
    'dist/sim/dex-conditions.js': '0455338330f1581b123bb7fb86151dcf6ca7199e07727344dc725a6ae786162a',
    'dist/data/mods/gen4/scripts.js': 'ccb990587bd46b5e4843d748c4b76c43ad30134fc517506517294ace1d47680d',
    'dist/data/mods/gen4/conditions.js': '8ba336b9823d9b806ca6259640fd8cc58666c22b7389b35b34c72937ae46357e',
})


@dataclass(frozen=True)
class ConstantChancePlan:
    subject: str
    opponent: str
    own_actions: tuple[int, ...]
    expected_histories: tuple[tuple[str, ...], ...]

    @property
    def receipt(self):
        stages = len(self.own_actions)
        return dict(schema=SCHEMA if stages == 3 else REST_TAIL_SCHEMA, public_stages=stages,
                    chance_predicate='same seed predicate across certified active support',
                    policy='full legal champion distribution, one draw per stage',
                    zero_likelihood_filter=('Encore expires before the three observed no-end residuals'
                        if stages == 3 else 'Encore expires before the four observed no-end residuals'),
                    sub_hp='actual accepted private chance state; never uniform/replaced',
                    max_wrap_damage=12, initial_sub_hp=66,
                    engine_hashes=dict(ENGINE_HASHES), set_source_hash=SOURCE_HASH,
                    compiled_tree=dict(files=COMPILED_TREE_FILES, sha256=COMPILED_TREE_SHA256,
                        algorithm='sorted relative path + NUL + file SHA256 + newline; all dist JS/JSON'))

    def compatible_move(self, stage, candidate):
        if candidate.get('kind') != 'move' or not candidate.get('legal'):
            return False
        move = candidate.get('move_id')
        return move == ('wrap', 'rest')[stage] if stage < 2 else move in {'wrap', 'rest', 'toxic', 'encore'}


@dataclass(frozen=True)
class StagedPrefixJointPlan:
    """A certified constant-chance prefix followed by ordinary joint rejection.

    The suffix has no constant-likelihood claim. Its original full-policy and
    one chance draw per request are retained. Any suffix mismatch rejects the
    entire fresh anchor/prefix proposal, not just chance at the suffix stage.
    """
    prefix: ConstantChancePlan
    own_actions: tuple[int | None, ...]
    expected_histories: tuple[tuple[str, ...], ...]

    @property
    def subject(self):
        return self.prefix.subject

    @property
    def opponent(self):
        return self.prefix.opponent

    @property
    def receipt(self):
        return dict(schema=JOINT_TAIL_SCHEMA, public_stages=len(self.own_actions),
            certified_prefix_stages=len(self.prefix.own_actions), prefix_law_certificate=self.prefix.receipt,
            suffix='original joint full-policy/chance draw; mismatch rejects entire fresh proposal',
            suffix_chance_draws_per_request=1, suffix_chance_conditioning=False,
            live_opponent_action_used=False, live_hidden_hp_used=False)


def _program_histories(before, current, stages=3):
    if current[:len(before)] != before:
        return None
    ends = [i+1 for i in range(len(before), len(current)) if current[i].startswith('|turn|')]
    if len(ends) != stages or ends[-1] != len(current):
        return None
    return tuple(current[:end] for end in ends)


def verify_compiled_tree(root):
    """Close over inherited handlers, Dex metadata, custom formats and generators.

    The earlier selected-file manifest remains explanatory evidence, not the
    dependency closure. Pin every compiled JS/JSON file and the complete path
    roster so unlisted additions/removals cannot change the effective engine.
    Checked once per prepared factory, not per chance retry or sampled world.
    """
    entries = []
    for path in (root/'dist').rglob('*'):
        if path.is_symlink():
            raise ReferenceRefusal('constant-chance compiled tree contains an unbound symlink')
        if path.is_file() and path.suffix in ('.js', '.json'):
            entries.append(path)
    entries.sort(key=lambda p:p.relative_to(root).as_posix())
    fingerprint = hashlib.sha256()
    for path in entries:
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fingerprint.update((relative+'\0'+digest+'\n').encode())
    if len(entries) != COMPILED_TREE_FILES or fingerprint.hexdigest() != COMPILED_TREE_SHA256:
        raise ReferenceRefusal('constant-chance complete compiled engine/generator closure drift')


def build_constant_chance_plan(factory):
    """Public-only eligibility; unsupported programs do NOT lose positions."""
    transition = factory.pending_transition
    actions = (transition.own_action, *transition.continuation_actions)
    if len(actions) not in (3, 4) or any(type(a) is not int for a in actions) or len(set(actions)) != 1:
        return None
    subject = factory.state.player_id
    opponent = 'p2' if subject == 'p1' else 'p1'
    before, current = public_history(transition.before_state), public_history(factory.state)
    histories = _program_histories(before, current, len(actions))
    if histories is None:
        return None
    return _certify_public_plan(factory, actions, histories, before)


def _certify_public_plan(factory, actions, histories, before):
    subject = factory.state.player_id
    opponent = 'p2' if subject == 'p1' else 'p1'
    previous, stages = before, []
    for history in histories:
        stages.append(history[len(previous):])
        previous = history
    own_id, opp_id = subject+'a: Lugia', opponent+'a: Shuckle'
    for stage, lines in enumerate(stages):
        moves = [line for line in lines if line.startswith('|move|')]
        required = ['|move|'+own_id+'|Psychic|'+opp_id]
        if stage < 2:
            target = own_id if stage == 0 else opp_id
            required.append('|move|'+opp_id+'|'+('Wrap', 'Rest')[stage]+'|'+target)
        if moves != required or any('|-end|'+own_id+'|Encore' == line for line in lines):
            return None
        if stage == 0 and '|-activate|'+own_id+'|Substitute|[damage]' not in lines:
            return None
        if stage == 1 and '|-status|'+opp_id+'|slp|[from] move: Rest' not in lines:
            return None
        if stage >= 2 and '|cant|'+opp_id+'|slp' not in lines:
            return None
        if any(line.startswith(('|switch|', '|drag|', '|faint|', '|-boost|', '|-unboost|',
                                '|-weather|', '|-start|', '|-end|', '|-sidestart|', '|-sideend|'))
               for line in lines):
            return None
    if factory.set_source.metadata.source_hash != SOURCE_HASH:
        raise ReferenceRefusal('constant-chance source certificate drift')
    variants = factory.set_source.universes['shuckle'].variants
    if not variants or any(v.level != 98 or v.ability.lower() != 'sturdy'
            or v.item.lower() != 'leftovers' or set(v.moves) != {'wrap', 'rest', 'toxic', 'encore'}
            for v in variants):
        raise ReferenceRefusal('constant-chance generator support drift')
    root = factory.env.config.resolved_showdown_root()
    verify_compiled_tree(root)
    for path, digest in ENGINE_HASHES.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ReferenceRefusal('constant-chance effective engine certificate drift: '+path)
    return ConstantChancePlan(subject, opponent, actions, histories)


def build_staged_prefix_joint_plan(factory):
    """Public-only dispatch; the suffix is never forced or locally conditioned.

    Truncating the *proof's public program*, not a world or observation, permits
    reuse of an already certified three/four-stage likelihood factor. Prefer
    the longest eligible prefix. No seed, opponent action, or hidden state is
    consulted. Non-turn-aligned histories retain the original joint kernel.
    """
    transition = factory.pending_transition
    actions = (transition.own_action, *transition.continuation_actions)
    before, current = public_history(transition.before_state), public_history(factory.state)
    histories = _program_histories(before, current, len(actions))
    if histories is None:
        return None
    for stages in (4, 3):
        if len(actions) <= stages:
            continue
        prefix_actions = actions[:stages]
        if any(type(a) is not int for a in prefix_actions) or len(set(prefix_actions)) != 1:
            continue
        prefix = _certify_public_plan(factory, prefix_actions, histories[:stages], before)
        if prefix is not None:
            return StagedPrefixJointPlan(prefix, actions, histories)
    return None


def _joint_suffix(factory, plan, rng):
    """Original joint kernel on the suffix, without any chance retry or pruning."""
    steps = []
    for stage in range(len(plan.prefix.own_actions), len(plan.own_actions)):
        factory.check_sampling_deadline()
        own_action = plan.own_actions[stage]
        requested = set(factory.env.requested_players())
        if (plan.subject in requested) != (own_action is not None):
            return None
        actions = {}
        if own_action is not None:
            mask = factory.env.observe(plan.subject).legal_action_mask
            if not 0 <= own_action < len(mask) or not mask[own_action]:
                return None
            actions[plan.subject] = own_action
        legal, probabilities, opponent_action = (), (), None
        if plan.opponent in requested:
            legal, evaluation = factory.evaluator(factory.env.observe(plan.opponent))
            probabilities = tuple(evaluation.priors)
            if (not legal or len(legal) != len(probabilities) or sum(probabilities) <= 0
                    or any(not math.isfinite(p) or p < 0 for p in probabilities)):
                raise ReferenceRefusal('joint suffix full champion policy invalid')
            opponent_action = rng.choices(legal, weights=probabilities, k=1)[0]
            actions[plan.opponent] = opponent_action
        if not requested or set(actions) != requested:
            return None
        chance_seed = rng.getrandbits(64)
        factory.env.reseed_simulator_rng(chance_seed)
        factory.env.step(actions)
        steps.append(dict(own_action=own_action, opponent_action=opponent_action,
            opponent_legal=list(legal), opponent_priors=list(probabilities), chance_seed=chance_seed,
            chance_draws=1, conditioning='original joint suffix'))
        if (factory.env.terminal() is not None
                or public_history(factory.env.public_materialization_state(plan.subject))
                    != plan.expected_histories[stage]):
            return None
    return steps


def validate_active_support(plan, snapshot, stage):
    """Check the proof context on a FRESH HYPOTHETICAL world only.

    Encore values that expire are the only deterministic zero-likelihood
    exclusion. Every other unsupported latent field is a REFUSAL, never a team
    rejection. Bench identity/order, move-slot permutations and positive legal
    opponent PP are intentionally unrestricted and remain in champion priors.
    """
    stages = len(plan.own_actions)
    if stages not in (3, 4) or len(plan.expected_histories) != stages or not 0 <= stage < stages:
        raise ReferenceRefusal('constant-chance stage outside certified Rest program')
    battle = snapshot.bridge_snapshot['battle']
    if battle.get('formatid') != 'gen3randombattle' or battle.get('gameType') != 'singles':
        raise ReferenceRefusal('constant-chance format support drift')
    field = battle['field']
    if (field.get('weather') or field.get('terrain')
            or set(field.get('pseudoWeather', {})) != {'sleepclausemod'}):
        raise ReferenceRefusal('constant-chance field support drift')
    sides = {side['id']:side for side in battle['sides']}
    if set(sides) != {'p1', 'p2'}:
        raise ReferenceRefusal('constant-chance side support drift')
    for side in sides.values():
        if set(side.get('sideConditions', {})) - {'spikes'} or any(side.get('slotConditions', [])):
            raise ReferenceRefusal('constant-chance slot/side-condition support drift')
    own, opp = [sides[side]['pokemon'][0] for side in (plan.subject, plan.opponent)]
    expected = [
        (own, '[Species:lugia]', 70, 'pressure', 264,
         {'atk':131,'def':223,'spa':167,'spd':257,'spe':195}, ['Psychic','Flying']),
        (opp, '[Species:shuckle]', 98, 'sturdy', 198,
         {'atk':75,'def':506,'spa':75,'spd':506,'spe':65}, ['Bug','Rock']),
    ]
    for mon, species, level, ability, maxhp, stats, types in expected:
        if (str(mon['species']).lower() != species.lower() or mon['set']['level'] != level
                or mon['ability'] != ability or mon['item'] != 'leftovers'
                or mon['maxhp'] != maxhp or mon['baseMaxhp'] != maxhp
                or mon['storedStats'] != stats or mon['types'] != types
                or mon.get('addedType') or mon.get('transformed') or mon.get('fainted')
                or not mon.get('isActive') or any(mon['boosts'].values())
                or mon['speed'] != stats['spe']):
            raise ReferenceRefusal('constant-chance active damage/event support drift')
    if ({slot['id'] for slot in own['moveSlots']} != {'substitute','recover','toxic','psychic'}
            or {slot['id'] for slot in opp['moveSlots']} != {'wrap','rest','toxic','encore'}
            or own['status'] or set(own['volatiles']) != {'encore','substitute'}
            or opp['volatiles']):
        raise ReferenceRefusal('constant-chance move/status/volatile support drift')
    encore, sub = own['volatiles']['encore'], own['volatiles']['substitute']
    if encore.get('move') != 'psychic' or type(encore.get('duration')) is not int:
        raise ReferenceRefusal('constant-chance Encore support drift')
    psychic = next(slot for slot in own['moveSlots'] if slot['id']=='psychic')
    if psychic['pp'] <= stages-stage:
        raise ReferenceRefusal('constant-chance public Psychic PP expiry outside certificate')
    if type(sub.get('hp')) is not int or not 1 <= sub['hp'] <= 66 or (stage==0 and sub['hp']!=66):
        raise ReferenceRefusal('constant-chance fresh/retained Substitute support drift')
    if stage < 2:
        if opp['status']:
            raise ReferenceRefusal('constant-chance awake prefix support drift')
    elif (opp['status'] != 'slp' or opp['statusState'].get('time') != 5-stage
          or opp['statusState'].get('startTime') != 3 or opp['statusState'].get('skippedTime') != 0
          or opp['statusState'].get('source') != '[Pokemon:'+plan.opponent+'a]'):
        raise ReferenceRefusal('constant-chance Rest-source timer support drift')
    return encore['duration'] > stages-stage


def sample_staged_path(factory, prior, plan, rng, evidence, *, max_attempts=2048):
    """Same bounded ownership/clock contract as joint conditioning."""
    from .paper_reference_showdown import decision_state
    if type(max_attempts) is not int or not 1 <= max_attempts <= 2048:
        raise ReferenceRefusal('invalid constant-chance rejection limit')
    prefix = plan.prefix if isinstance(plan, StagedPrefixJointPlan) else plan
    rejected = []
    for attempt in range(max_attempts):
        factory.check_sampling_deadline()
        anchor_rng_state = rng.getstate()
        world, accepted = prior(rng), False
        try:
            initial = factory.env.snapshot()
            if not validate_active_support(prefix, initial, 0):
                rejected.append(dict(attempt=attempt, reason='deterministic zero-likelihood Encore expiry'))
                continue
            steps, valid = [], True
            for stage, own_action in enumerate(prefix.own_actions):
                factory.check_sampling_deadline()
                if set(factory.env.requested_players()) != {plan.subject,plan.opponent}:
                    raise ReferenceRefusal('constant-chance simultaneous request support drift')
                own_mask = factory.env.observe(plan.subject).legal_action_mask
                if not 0 <= own_action < len(own_mask) or not own_mask[own_action]:
                    raise ReferenceRefusal('constant-chance known actor action support drift')
                observation = factory.env.observe(plan.opponent)
                legal, evaluation = factory.evaluator(observation)
                probabilities = tuple(evaluation.priors)
                if (not legal or len(legal) != len(probabilities) or sum(probabilities)<=0
                        or any(not math.isfinite(p) or p<0 for p in probabilities)):
                    raise ReferenceRefusal('constant-chance full champion policy invalid')
                opponent_action = rng.choices(legal,weights=probabilities,k=1)[0]
                candidates = {row['action_index']:row for row in observation.metadata['action_candidates']}
                if not prefix.compatible_move(stage,candidates[opponent_action]):
                    rejected.append(dict(attempt=attempt,reason='full policy choice incompatible',stage=stage))
                    valid = False
                    break
                snapshot = factory.env.snapshot()
                if not validate_active_support(prefix,snapshot,stage):
                    raise ReferenceRefusal('accepted prefix lost eligible Encore support')
                matched = False
                # Keep the exact retry law and RNG sequence. Retain one immutable
                # hypothetical branch point locally in the bridge rather than
                # transporting its full simulator state for every rejected trial.
                search_snapshot = factory.env.snapshot_for_search()
                try:
                    for chance_attempt in range(MAX_CHANCE_ATTEMPTS):
                        factory.check_sampling_deadline()
                        seed = rng.getrandbits(64)
                        factory.env.step_from_search_snapshot_for_conditioning(
                            search_snapshot, {plan.subject:own_action,plan.opponent:opponent_action},
                            chance_seed=seed)
                        if (factory.env.terminal() is None
                                and public_history(factory.env.public_materialization_state(plan.subject))
                                == prefix.expected_histories[stage]):
                            steps.append(dict(own_action=own_action,opponent_action=opponent_action,
                                opponent_move=candidates[opponent_action]['move_id'],
                                opponent_legal=list(legal),opponent_priors=list(probabilities),
                                chance_seed=seed,chance_attempts=chance_attempt+1))
                            matched = True
                            break
                finally:
                    factory.env.release_search_snapshot(search_snapshot)
                if not matched:
                    rejected.append(dict(attempt=attempt,reason='bounded constant-chance exhaustion',stage=stage))
                    valid = False
                    break
            if not valid:
                continue
            if isinstance(plan, StagedPrefixJointPlan):
                suffix = _joint_suffix(factory, plan, rng)
                if suffix is None:
                    rejected.append(dict(attempt=attempt, reason='different joint suffix public transition/request'))
                    continue
                steps.extend(suffix)
            factory.check_sampling_deadline()
            current = factory.env.public_materialization_state(plan.subject)
            if (decision_state(factory.env.observe(plan.subject),player=plan.subject) != factory.root
                    or public_history(current) != public_history(factory.state)
                    or current.deferred_opponent_action_player != factory.state.deferred_opponent_action_player):
                raise ReferenceRefusal('constant-chance exact target root validation failed')
            hp = {side['id']:side['pokemon'][0]['volatiles']['substitute']['hp']
                  for side in factory.env.snapshot().bridge_snapshot['battle']['sides']
                  if 'substitute' in side['pokemon'][0]['volatiles']}
            evidence.update({k:v for k,v in prior.receipts[-1].items() if k not in ('ordinal','status','released')})
            evidence.update(status='ROOT_VALIDATED',substitute_policy_conditioning=dict(
                algorithm=plan.receipt['schema'],attempts=attempt+1,max_attempts=max_attempts,
                max_chance_attempts=MAX_CHANCE_ATTEMPTS,rejected=rejected,steps=steps,
                chance_transport='bridge-resident restore/reseed/step; lazy observations',
                anchor_rng_state=anchor_rng_state,
                sampled_substitute_hp=hp,law_certificate=plan.receipt,
                prior_actor_root_key=prior.root.key.hex(),current_actor_root_key=factory.root.key.hex(),
                live_opponent_action_used=False,live_hidden_hp_used=False))
            release_prior = world.release
            def release():
                release_prior()
                factory._release(evidence)
            world.release = release
            accepted = True
            return world
        finally:
            if not accepted:
                world.close()
    error = ReferenceRefusal('constant-chance conditioning exhausted its explicit rejection cap')
    error.sampling_diagnostic=dict(schema=SCHEMA,attempts=max_attempts,rejected=rejected,accepted_worlds=0)
    raise error
