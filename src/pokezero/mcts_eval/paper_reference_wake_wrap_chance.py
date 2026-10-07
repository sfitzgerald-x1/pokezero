"""Guarded ten-turn Wake/Wrap conditioning, not a forced opponent policy.

The first five stages retain the separately qualified Wake/Rest certificate.
The additional stages are failed Substitute / two Rest-blocked moves / waking
Wrap hitting the surviving Substitute / Encore / Wrap missing. Each compatible
opponent action is still drawn ONCE from its full legal champion distribution.
An incompatible draw rejects the entire anchor, including all prior chance.

For the pinned engine and guarded active support, the later chance predicates
are constant across latent teams and retained private prefixes: sleep and
failed Substitute are deterministic; Wrap cannot break the Substitute even on
a critical hit; Encore bypasses Substitute and all sampled durations survive
the remaining turn; the final Wrap miss is the same accuracy predicate. Actual
damage, Encore duration, PRNG and PP state are retained, never replaced. Hidden
bench identity, ordering, move-slot ordering and policy probabilities remain
unrestricted. Thus dividing by these chance-event likelihoods (and bounded
retry factors) changes the path density only by a common constant.

This certificate is opt-in through the existing staged-conditioning gate. It
requires fresh qualification and independent legacy replay before study use.
It does not expand a deadline, force moves, or certify an arbitrary suffix.
"""
from dataclasses import dataclass
import re

from .paper_reference import ReferenceRefusal
from .paper_reference_pending import public_history
from .paper_reference_staged_chance import (
    ConstantChancePlan, StagedPrefixJointPlan, _certify_public_plan,
    _program_histories, validate_active_support,
)

WAKE_WRAP_SCHEMA = 'pokezero.constant-chance-wake-wrap-encore-miss.v6'
WAKE_WRAP_JOINT_SCHEMA = 'pokezero.constant-chance-wake-wrap-prefix-joint-tail.v7'


@dataclass(frozen=True)
class WakeWrapChancePlan(ConstantChancePlan):
    prefix: ConstantChancePlan

    @property
    def receipt(self):
        return dict(schema=WAKE_WRAP_SCHEMA, public_stages=10,
            prefix_law_certificate=self.prefix.receipt,
            chance_predicate='same seed predicate across certified active support',
            policy='full legal champion distribution, one draw per stage',
            zero_likelihood_filter='original Encore expires at the fifth residual',
            retained_private_state='actual Substitute HP, new Encore duration, PP and PRNG',
            initial_sub_hp=66, max_wrap_damage=12, wrap_hits_before_final_miss=2,
            minimum_sub_hp_before_second_wrap=54, minimum_sub_hp_after_second_wrap=42,
            new_encore_duration_before_final_turn=[3, 4, 5, 6],
            suffix_chance_conditioning=False,
            live_opponent_action_used=False, live_hidden_hp_used=False)

    def compatible_move(self, stage, candidate):
        if stage < 5:
            return self.prefix.compatible_move(stage, candidate)
        if candidate.get('kind') != 'move' or not candidate.get('legal'):
            return False
        move = candidate.get('move_id')
        return (move in {'wrap', 'rest', 'toxic', 'encore'} if stage in (5, 6)
            else move == {7: 'wrap', 8: 'encore', 9: 'wrap'}.get(stage))


def build_wake_wrap_plan(factory):
    """Recognize public move structure, never a seed, root ordinal or outcome."""
    transition = factory.pending_transition
    actions = (transition.own_action, *transition.continuation_actions)
    if (len(actions) < 10 or any(type(a) is not int for a in actions[:10])
            or len(set(actions[:5])) != 1 or len(set(actions[5:10])) != 1):
        return None
    before, current = public_history(transition.before_state), public_history(factory.state)
    histories = _program_histories(before, current, len(actions))
    if histories is None:
        return None
    subject = factory.state.player_id
    opponent = 'p2' if subject == 'p1' else 'p1'
    own, opp = subject+'a: Lugia', opponent+'a: Shuckle'
    for stage in range(5, 10):
        lines = list(histories[stage][len(histories[stage-1]):])
        expected = ['|move|'+own+'|Substitute|'+own, '|-fail|'+own+'|move: Substitute']
        if stage in (5, 6):
            expected.append('|cant|'+opp+'|slp')
            # Exact public HP is bound below; Leftovers is deterministic here.
            heals = [line for line in lines if line.startswith('|-heal|')]
            if len(heals) > 1 or any(not re.fullmatch(
                    re.escape('|-heal|'+own+'|')+r'[1-9][0-9]*/264\|\[from\] item: Leftovers', line)
                    for line in heals):
                return None
            expected.extend(heals)
        elif stage == 7:
            expected += ['|-curestatus|'+opp+'|slp|[msg]', '|move|'+opp+'|Wrap|'+own,
                '|-activate|'+own+'|Substitute|[damage]']
        elif stage == 8:
            expected += ['|move|'+opp+'|Encore|'+own, '|-start|'+own+'|Encore']
        else:
            expected += ['|move|'+opp+'|Wrap|'+own+'|[miss]', '|-miss|'+opp+'|'+own]
        if (len(lines) < 2 or not re.fullmatch(r'\|turn\|[1-9][0-9]*', lines[-1])
                or lines != expected+['|upkeep', lines[-1]]):
            return None
    prefix = _certify_public_plan(factory, actions[:5], histories[:5], before)
    if prefix is None:
        return None
    plan = WakeWrapChancePlan(subject, opponent, actions[:10], histories[:10], prefix)
    return plan if len(actions) == 10 else StagedPrefixJointPlan(plan, actions, histories)


def validate_wake_wrap_support(plan, snapshot, stage):
    """Unsupported support REFUSES; only old zero-likelihood Encore rejects."""
    if len(plan.own_actions) != 10 or len(plan.expected_histories) != 10 or not 0 <= stage < 10:
        raise ReferenceRefusal('Wake/Wrap stage outside ten-turn certificate')
    sides = {side['id']: side for side in snapshot.bridge_snapshot['battle']['sides']}
    own = sides[plan.subject]['pokemon'][0]
    action = plan.own_actions[stage]
    required_move = 'psychic' if stage < 5 else 'substitute'
    if (type(action) is not int or not 0 <= action < len(own['moveSlots'])
            or own['moveSlots'][action]['id'] != required_move):
        raise ReferenceRefusal('Wake/Wrap own public action/slot support drift')
    if stage < 5:
        return validate_active_support(plan.prefix, snapshot, stage)
    battle = snapshot.bridge_snapshot['battle']
    field = battle['field']
    if (battle.get('formatid') != 'gen3randombattle' or battle.get('gameType') != 'singles'
            or field.get('weather') or field.get('terrain')
            or set(field.get('pseudoWeather', {})) != {'sleepclausemod'}):
        raise ReferenceRefusal('Wake/Wrap format/field support drift')
    sides = {side['id']: side for side in battle['sides']}
    if set(sides) != {'p1', 'p2'} or any(set(side.get('sideConditions', {}))-{'spikes'}
            or any(side.get('slotConditions', [])) for side in sides.values()):
        raise ReferenceRefusal('Wake/Wrap side/slot support drift')
    own, opp = [sides[player]['pokemon'][0] for player in (plan.subject, plan.opponent)]
    contexts = (
        (own, 'lugia', 70, 'pressure', 264, {'atk':131,'def':223,'spa':167,'spd':257,'spe':195},
            ['Psychic','Flying'], {'substitute','recover','toxic','psychic'}),
        (opp, 'shuckle', 98, 'sturdy', 198, {'atk':75,'def':506,'spa':75,'spd':506,'spe':65},
            ['Bug','Rock'], {'wrap','rest','toxic','encore'}),
    )
    for mon, species, level, ability, maxhp, stats, types, moves in contexts:
        if (str(mon['species']).lower() != '[species:'+species+']' or mon['set']['level'] != level
                or mon['ability'] != ability or mon['item'] != 'leftovers'
                or mon['maxhp'] != maxhp or mon['baseMaxhp'] != maxhp or mon['storedStats'] != stats
                or mon['speed'] != stats['spe'] or mon['types'] != types or any(mon['boosts'].values())
                or mon.get('addedType') or mon.get('transformed') or mon.get('fainted')
                or not mon.get('isActive') or {slot['id'] for slot in mon['moveSlots']} != moves):
            raise ReferenceRefusal('Wake/Wrap active event support drift')
    required_volatiles = {'substitute', 'encore'} if stage == 9 else {'substitute'}
    if own['status'] or set(own['volatiles']) != required_volatiles or opp['volatiles'] or opp['hp'] != 198:
        raise ReferenceRefusal('Wake/Wrap status/volatile/Rest HP support drift')
    lower, upper = (54, 65) if stage < 8 else (42, 64)
    sub_hp = own['volatiles']['substitute'].get('hp')
    if type(sub_hp) is not int or not lower <= sub_hp <= upper:
        raise ReferenceRefusal('Wake/Wrap retained Substitute survival bound drift')
    before = plan.expected_histories[stage-1]
    hp_lines = [line for line in before if line.startswith((
        '|-heal|'+plan.subject+'a: Lugia|', '|-damage|'+plan.subject+'a: Lugia|'))]
    match = re.match(r'^\|-(?:heal|damage)\|[^|]+\|([0-9]+)/264(?:\||$)', hp_lines[-1]) if hp_lines else None
    if match is None or own['hp'] != int(match[1]) or (stage >= 7 and own['hp'] != 264):
        raise ReferenceRefusal('Wake/Wrap exact public own HP support drift')
    substitute = next(slot for slot in own['moveSlots'] if slot['id'] == 'substitute')
    if substitute['pp'] <= 10-stage:
        raise ReferenceRefusal('Wake/Wrap Substitute PP/Encore expiry outside certificate')
    if stage >= 6 and own.get('lastMove') != '[DataMove:substitute]':
        raise ReferenceRefusal('Wake/Wrap failed Substitute last-move support drift')
    if stage < 8:
        timer = opp.get('statusState', {})
        if (opp['status'] != 'slp' or timer.get('time') != 8-stage or timer.get('startTime') != 3
                or timer.get('skippedTime') != 0 or timer.get('source') != '[Pokemon:'+plan.opponent+'a]'):
            raise ReferenceRefusal('Wake/Wrap second Rest-source timer support drift')
    elif opp['status']:
        raise ReferenceRefusal('Wake/Wrap awake opponent support drift')
    if stage == 9:
        encore = own['volatiles']['encore']
        if (encore.get('move') != 'substitute' or type(encore.get('duration')) is not int
                or encore['duration'] not in (3, 4, 5, 6)):
            raise ReferenceRefusal('Wake/Wrap new Encore duration support drift')
    return True
