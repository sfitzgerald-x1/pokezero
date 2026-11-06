"""Opt-in finite Showdown self-play capture using the original B2 v5 actors.

No default dispatch, seed selection, checkpoint loading, trainer or allocation.
Factories retain immutable weights/JITs; every game creates fresh owning actors.
Only literal p1/p2 getPlayerStreams packets reach their respective actors.
Private birth/terminal snapshots are controller-only training-label sources.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random

from . import v5_contract as c
from .dex import normalize_id
from .env import BattleStartOverride
from .local_showdown import LocalShowdownEnv
from .randbat import canonical_gen3_randbat_species_id

SCHEMA = 'pokezero.showdown_owning_collection.v5'


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def actor_binding(factory):
    """Actual immutable factory provenance, not caller-declared actor fields."""
    from gpu_engine.v5_live_protocol import check_factory
    check_factory(factory)
    metadata=json.loads(factory.metadata)
    result=dict(archive_sha256=metadata['weights_archive_sha256'],
        checkpoint_sha256=metadata['checkpoint_sha256'],model_sha256=_sha(factory.config.metadata()))
    if any(type(v)is not str or len(v)!=64 or any(c not in '0123456789abcdef'for c in v)for v in result.values()):
        raise ValueError('actual digest-bound factory provenance required')
    return result


class OwningEnv(LocalShowdownEnv):
    """Keep literal isolated streams alongside the unchanged Python env fold.

    The ordinary default env does not retain these streams. The opt-in subclass
    adds no event reconstruction and never sends omniscient lines to an actor.
    """
    def __init__(self,config):
        self.owner_lines={'p1':[],'p2':[]}
        super().__init__(config)

    def reset(self,**kwargs):
        self.owner_lines={'p1':[],'p2':[]}
        return super().reset(**kwargs)

    def reset_with_start_override(self,**kwargs):
        self.owner_lines={'p1':[],'p2':[]}
        start=kwargs['start_override']
        if type(start)is not BattleStartOverride or start.observation_format_id!='gen3randombattle':
            raise ValueError('original Random Battle packed constructor required')
        # Same original transport seam as finite_stream_comparison's pool
        # constructor. Relabeling Custom Game would change native rules/HP.
        return self._reset(seed=kwargs['seed'],format_id='gen3randombattle',start_override=start)

    def _apply_event(self,event):
        result=super()._apply_event(event)
        if (event.get('battleId')==self._battle_token and event.get('type')=='stream'
                and event.get('stream')in self.owner_lines):
            lines=event.get('lines')
            if (type(lines)is not list or any(type(x)is not str or not x.startswith('|')
                    or '\n'in x or '\r'in x for x in lines)):
                raise ValueError('literal isolated Showdown stream lines required')
            self.owner_lines[event['stream']].extend(lines)
        return result


@dataclass(frozen=True)
class Game:
    stream:int
    battle_id:str
    seed:int
    sampling_seed:int

    def __post_init__(self):
        if (type(self.stream)is not int or not 0<self.stream<2**31
                or type(self.battle_id)is not str or not self.battle_id
                or any(type(v)is not int or not 0<=v<2**32 for v in(self.seed,self.sampling_seed))):
            raise ValueError('fixed original stream/battle/game/action seeds required')


def _teams(snapshot):
    """Small literal projection of an actual simulator snapshot, never actor input."""
    sides=snapshot.bridge_snapshot['battle']['sides']
    if len(sides)!=2:raise ValueError('two actual simulator sides required')
    result=[]
    for side in sides:
        team=[]
        for mon in side['pokemon']:
            original=mon['set']
            # Pinned State.serializePokemon omits top-level name, but retains
            # the original set AFTER Pokemon canonicalizes set.name. Its actual
            # display identity is set.name.substr(0,20), in UTF-16 code units.
            # Species is not a nickname fallback (e.g. Deoxys-Attack/Deoxys).
            if type(original)is not dict or type(original.get('name'))is not str or not original['name']:
                raise ValueError('literal serialized original Pokemon name required')
            name=original['name'].encode('utf-16-le',errors='surrogatepass')[:40].decode('utf-16-le',errors='surrogatepass')
            hp,maximum=mon['hp'],mon['maxhp']
            if (not name
                    or any(type(v)is not int for v in(hp,maximum))or not 0<=hp<=maximum or maximum<=0):
                raise ValueError('literal named physical Pokemon and exact HP required')
            for key in('species','moves','level','ability','item'):
                if key not in original:raise ValueError('original simulator set missing '+key)
            team.append(dict(name=name,set=json.loads(json.dumps(original)),hp=hp,maxhp=maximum,
                hidden_power_type=mon.get('hpType')))
        if len(team)!=6 or len({x['name']for x in team})!=6:
            raise ValueError('six unambiguous original physical names required')
        result.append(tuple(team))
    return tuple(result)


def _ordered_terminal(birth,terminal):
    result=[]
    for initial,current in zip(birth,terminal):
        by_name={x['name']:x for x in current}
        if set(by_name)!={x['name']for x in initial}:raise ValueError('terminal physical identities changed')
        ordered=tuple(by_name[x['name']]for x in initial)
        if any(x['set']!=y['set']for x,y in zip(initial,ordered)):
            raise ValueError('original set identity changed before terminal capture')
        result.append(ordered)
    return tuple(result)


def _physical(old,birth,owner,vocabulary):
    result=[]
    for offset,side in((1,owner),(7,1-owner)):
        by_species={canonical_gen3_randbat_species_id(x['set']['species']):i for i,x in enumerate(birth[side])}
        if len(by_species)!=6:raise ValueError('ambiguous original species-clause mapping')
        for token in range(offset,offset+6):
            if not old.attention_mask[token]:result.append(-1);continue
            row=old.categorical_ids[token][0]
            if not 1<=row<=len(vocabulary.tokens):raise ValueError('public species category is not enumerated')
            label=vocabulary.tokens[row-1]
            if not label.startswith('species:'):raise ValueError('public Pokemon identity category missing')
            name=canonical_gen3_randbat_species_id(label.split(':',1)[1])
            if name not in by_species:raise ValueError('public species has no unambiguous original physical slot')
            result.append(by_species[name])
    return tuple(result)


def _semantic(choice,request,vocabulary):
    """Literal selected own-request identity; never next/previous frame actions."""
    parts=choice.split()
    if len(parts)!=2 or parts[0]not in('move','switch')or not parts[1].isdigit():
        raise ValueError('literal original move/switch choice required')
    index=int(parts[1])-1
    if parts[0]=='move':
        active=request.get('active')
        if type(active)is not list or len(active)!=1 or not 0<=index<len(active[0]['moves']):
            raise ValueError('selected move missing from the same own request')
        move=active[0]['moves'][index]
        identity=normalize_id(move.get('id')or move.get('move'))
        if identity=='hiddenpower':
            shown=normalize_id(move.get('move'))
            if shown.startswith('hiddenpower')and shown!='hiddenpower':identity=shown
        kind='MOVE'
    else:
        team=request['side']['pokemon']
        if not 0<=index<len(team):raise ValueError('selected switch missing from same own request')
        identity=canonical_gen3_randbat_species_id(team[index]['details'].split(',')[0]);kind='SWITCH'
    target=c.OpponentAction(kind,identity)
    # Unsupported/pseudo identities mask only this auxiliary target. They do
    # not fabricate an identity or invalidate a legal collected decision.
    return target if kind+':'+identity in vocabulary.opponent_classes else None


def _terminal_winner(owner_lines,marker,terminal):
    """Join both literal public player headers and terminal to the env result."""
    names=[]
    for seat in('p1','p2'):
        identities={'p1':set(),'p2':set()}
        for line in owner_lines[seat]:
            if line.startswith('|player|'):
                parts=line.split('|')
                if len(parts)<4 or parts[2]not in identities or not parts[3]:
                    raise ValueError('literal public player identity is malformed')
                identities[parts[2]].add(parts[3])
        if any(len(values)!=1 for values in identities.values()):
            raise ValueError('public player identities are missing or ambiguous')
        names.append(tuple(next(iter(identities[player]))for player in('p1','p2')))
    if names[0]!=names[1]or names[0][0]==names[0][1]:
        raise ValueError('owning public player identities disagree or are ambiguous')
    if marker=='|tie'or marker.startswith('|tie|'):
        winner=2;expected=None
    elif marker.startswith('|win|')and marker[len('|win|'):]in names[0]:
        winner=names[0].index(marker[len('|win|'):]);expected=('p1','p2')[winner]
    else:raise ValueError('owning terminal winner is not a public player identity')
    if terminal.winner!=expected:
        raise ValueError('owning terminal marker contradicts the environment winner')
    return winner


@dataclass(frozen=True)
class GameCapture:
    game:Game
    status:str
    decisions:tuple
    frames:int
    reason:str|None
    source:dict
    auxiliary_refusals:dict


def collect_game(env,game,*,factories,max_requests,collect_owners=(0,1),start_override=None,
        chance_seed=None,check_progress=lambda:None,enable_v5=False):
    """One declared attempt, with unknowns retained and no retry/refill.

    This low-level bridge owns neither a study roster nor training admission.
    The caller must bind any packed constructor teams and the two immutable
    factories before calling. A and B use raw stochastic actors; D is deferred.
    One game failure returns an unknown prefix, allowing the next declared game.
    Permission errors propagate so the owner stops affected work.
    """
    if enable_v5 is not True or type(game)is not Game or not isinstance(env,OwningEnv):
        raise ValueError('explicit finite Showdown v5 collector enablement required')
    if (type(max_requests)is not int or max_requests<=0 or tuple(sorted(set(collect_owners)))!=tuple(collect_owners)
            or not collect_owners or any(type(x)is not int or x not in(0,1)for x in collect_owners)
            or set(factories)!={0,1}):
        raise ValueError('bounded game, original collecting seats and both raw factories required')
    if chance_seed is not None and(type(chance_seed)is not int or not 0<=chance_seed<2**32):
        raise ValueError('explicit uint32 Showdown chance seed required')
    from gpu_engine.v5_live_protocol import check_factory
    from gpu_engine import v5_collection_data as data
    for factory in factories.values():
        check_factory(factory)
        if factory.deterministic or factory.config.arm=='D':raise ValueError('stochastic A/B raw factories required')
    bindings={(x.config.category_vocabulary_sha256,x.config.label_vocabulary.sha256)for x in factories.values()}
    if len(bindings)!=1:raise ValueError('same original v5 category/label axes required')
    actual_bindings=tuple(actor_binding(factories[owner])for owner in(0,1))
    actors={};positions={0:0,1:0};captures=[];frame=0;status='FAILED';reason=None
    source={'actor_bindings':actual_bindings};aux=Counter()
    birth=None;terminal=None
    rngs={owner:random.Random(int(_sha(dict(schema=SCHEMA,seed=game.seed,sampling_seed=game.sampling_seed,
        stream=game.stream,owner=owner)),16))for owner in(0,1)}
    def feed():
        for owner,actor in actors.items():
            lines=tuple(env.owner_lines['p'+str(owner+1)][positions[owner]:])
            if lines:
                actor.accept_owner_stream(battle_id=game.battle_id,owner=actor.owner,
                    generation=actor.generation,prefix_sha256=actor.prefix_sha256,lines=lines)
                positions[owner]+=len(lines)
    try:
        check_progress()
        actual_seed=game.seed if chance_seed is None else chance_seed
        source['chance_seed']=actual_seed
        if start_override is None:env.reset(seed=actual_seed,format_id='gen3randombattle')
        else:
            if type(start_override)is not BattleStartOverride:raise ValueError('explicit original packed teams required')
            env.reset_with_start_override(seed=actual_seed,start_override=start_override)
        birth=_teams(env.snapshot());source['birth']=birth;source['birth_sha256']=_sha(birth)
        for owner in(0,1):
            actor=factories[owner].create_actor();actors[owner]=actor
            actor.begin_game(battle_id=game.battle_id,owner='p'+str(owner+1),seed=game.seed,stream_id=game.stream)
        while frame<max_requests and env.terminal()is None:
            check_progress();feed();requested=env.requested_players()
            if not requested:raise ValueError('nonterminal game has no actionable requests')
            decisions={};snapshots={};observations={};choices={};semantic={}
            for seat in requested:
                owner=int(seat[1])-1;actor=actors[owner];snapshot=actor.snapshot_boundary()
                snapshots[owner]=snapshot;decisions[owner]=actor.decide(snapshot,rng=rngs[owner])
                observations[owner]=actor.decision_v5_observation(snapshot,frame=frame)
                choices[seat]=actor.choice(snapshot)
                semantic[owner]=_semantic(choices[seat],json.loads(snapshot.own_request_line[len('|request|'):]),
                    factories[owner].config.label_vocabulary)
                if semantic[owner]is None:aux['opponent_action_outside_namespace']+=1
            if env._choices_for_actions({'p'+str(o+1):x.action_index for o,x in decisions.items()})!=choices:
                raise ValueError('original Python request routing differs from B2 selected choice')
            pending=[]
            for owner in collect_owners:
                if owner not in decisions:continue
                old,public=observations[owner];decision=decisions[owner];factory=factories[owner]
                physical=_physical(old,birth,owner,factory.catalog.vocabulary)
                hidden=[]
                for index in physical[6:]:
                    label=None
                    if index>=0:
                        mon=birth[1-owner][index];literal=mon['set']
                        label=data.matching_hidden_set(literal['species'],level=literal['level'],moves=literal['moves'],
                            ability=literal['ability'],item=literal['item'],hidden_power_type=mon['hidden_power_type'],
                            source=factory.catalog.source)
                        if label is None:aux['hidden_set_outside_source']+=1
                    hidden.append(label)
                metadata=dict(producer=SCHEMA,stream=game.stream,frame=frame,owner=owner,physical=physical,
                    choice=decision.action_index,behavior_probability=decision.action_probability,value_estimate=decision.value_estimate,
                    actor_binding=actual_bindings[owner],opponent_actor_binding=actual_bindings[1-owner],
                    birth_sha256=source['birth_sha256'],owner_prefix_sha256=snapshots[owner].prefix_sha256,
                    own_request_sha256=hashlib.sha256(snapshots[owner].own_request_line.encode()).hexdigest())
                digest=data.sha(dict(metadata,public_inputs=public.model_inputs()))
                pending.append(data.capture_decision(old,public,battle_id=game.battle_id,seed=game.seed,turn=frame,
                    choice=decision.action_index,probability=decision.action_probability,value=decision.value_estimate,
                    physical=physical,hidden_sets=hidden,source_frame_sha256=digest,capture_metadata=metadata,
                    opponent_action=semantic.get(1-owner)))
            # Commit rows only after the original simulator accepts both choices.
            env.step({'p'+str(o+1):x.action_index for o,x in decisions.items()});captures.extend(pending);frame+=1
        ended=env.terminal()
        if ended is None or ended.capped:
            status='CAPPED';source['cap']=dict(max_requests=max_requests,frames=frame)
        else:
            final=_ordered_terminal(birth,_teams(env.snapshot()));feed()
            markers=[]
            for owner in(0,1):
                markers.append(tuple(x for x in env.owner_lines['p'+str(owner+1)]
                    if x.startswith('|win|')or x=='|tie'or x.startswith('|tie|')))
            if len(markers[0])!=1 or markers[0]!=markers[1]:raise ValueError('literal owning terminal streams disagree or missing')
            winner=_terminal_winner(env.owner_lines,markers[0][0],ended)
            source['terminal']=dict(teams=final,winner=winner,marker=markers[0][0]);status='COMPLETE'
            terminal=data.TerminalCapture(game.stream,frame,status,_sha(source['terminal']),winner,
                tuple(tuple(x['hp']for x in side)for side in final),tuple(tuple(x['maxhp']for x in side)for side in final))
    except PermissionError:raise
    except Exception as error:
        status='REFUSED'if isinstance(error,ValueError)else'FAILED';reason=type(error).__name__+': '+str(error)
    finally:
        for actor in actors.values():
            try:
                closed=actor.close_game()
                if closed['status']=='FAILED_CLOSED' and status not in('REFUSED','FAILED'):
                    raise ValueError('owning actor failed to close cleanly')
            except PermissionError:raise
            except Exception as error:
                status='FAILED';detail='ACTOR_CLOSE: '+type(error).__name__+': '+str(error)
                reason=detail if reason is None else reason+'; '+detail
    if status!='COMPLETE':
        terminal=data.TerminalCapture(game.stream,frame,status,_sha(dict(schema=SCHEMA,status=status,frame=frame,reason=reason)))
    finished=data.finish(tuple(captures),(terminal,))
    return GameCapture(game,status,finished,frame,reason,source,dict(aux))
