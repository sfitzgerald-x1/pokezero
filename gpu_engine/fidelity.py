"""Fixed unfiltered 200-pair, first-divergence diagnostic; no state repair.

Known semantic chance inputs are coupled, not native PRNG streams. Unmapped
roles stop a trajectory as unresolved and stay in the fixed denominator.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import jax
import jax.numpy as j
import numpy as np

from . import transition

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownEnv,LocalShowdownConfig
from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import showdown_choice_for_action

STATUS=['','par','brn','psn','tox','slp','frz']
# The first 50000..50199 panel is immutable historical evidence. Following
# its authored adapter correction, use one disjoint fixed panel, not a replay.
FIRST_GAME=50200
canonical=lambda name:'hiddenpower' if name.startswith('hiddenpower') else name
toid=lambda name:''.join(x for x in name.lower() if x.isalnum())


def reconstruct(pool,metadata,showdown,node,*,first_game=FIRST_GAME,games=200):
    """Verify all stored fields against the original generator before play."""
    start=2*first_game;count=2*games
    if games!=200 or start<0 or start+count>len(pool['seeds']):raise ValueError('fixed 200-pair roster required')
    seeds=pool['seeds'][start:start+count].tolist()
    if seeds!=list(range(seeds[0],seeds[0]+count)):raise ValueError('contiguous original seeds required')
    command=[node,str(ROOT/'gpu_engine/generate_teams.cjs'),str(showdown),str(count),str(seeds[0]),metadata['source_commit_label']]
    raw=subprocess.check_output(command,text=True)
    records=[json.loads(x) for x in raw.splitlines()];header=records[0]
    for key in ('compiled_tree','rng_dependency','vocabulary','node_version'):
        if header[key]!=metadata[key]:raise ValueError(f'generator binding differs: {key}')
    if len(records)!=count+2 or records[-1]!={'kind':'terminal','team_count':count}:raise ValueError('incomplete regeneration')
    fields=('species','level','types','stats','moves','pp','item','ability','nature','ivs','evs','hidden_power_type','hidden_power_power')
    for index,row in enumerate(records[1:-1]):
        if row['kind']!='team' or row['index']!=index or row['seed']!=seeds[index]:raise ValueError('regenerated roster differs')
        for field in fields:
            if [mon[field] for mon in row['members']]!=pool[field][start+index].tolist():
                raise ValueError(f'accepted tensor differs: team {start+index}/{field}')
    packed=json.loads(subprocess.check_output([node,str(ROOT/'gpu_engine/regenerate_packed.cjs'),str(showdown)],input=json.dumps(seeds),text=True))
    if [r['seed'] for r in packed]!=seeds or any(not r['packed'] for r in packed):raise ValueError('packed roster differs')
    return packed,dict(first_game=first_game,games=games,team_indices=[start,start+count],original_seed_range=[seeds[0],seeds[-1]],
        regeneration_sha256=hashlib.sha256(raw.encode()).hexdigest(),all_stored_fields_reconciled=list(fields),
        packed_sets_sha256=hashlib.sha256(json.dumps(packed,sort_keys=True).encode()).hexdigest())


def chance_tape(state,game,index):
    split=jax.vmap(lambda key:jax.random.split(key,2))(state.keys)[:,1]
    draws=jax.vmap(lambda key:jax.random.randint(key,(2,12),0,j.array([2,100,100,100,16,16,100,4,8,2,2,2],j.int32)))(split)
    secondary=jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,1618033),(2,),0,100))(split)
    sleep=jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,1414213),(2,),0,60))(split)
    if bool(np.asarray(state.pending_residual)[0]):
        draws=state.residual_draws;secondary=state.secondary_draws;sleep=state.sleep_talk_draws
    return dict(schema='pokezero.gpu_chance_tape.v1',nonce=f'{game}:{index}',draws=np.asarray(draws)[0].tolist(),
        secondary=np.asarray(secondary)[0].tolist(),sleep_talk=np.asarray(sleep)[0].tolist())


def native_view(snapshot,original,vocab):
    battle=snapshot.bridge_snapshot['battle'];parties=[];active=[]
    for side in range(2):
        rows=battle['sides'][side]['pokemon'];ordered={}
        for mon in rows:
            identity=toid(mon['set']['species'])
            if identity not in original[side] or identity in ordered:raise ValueError('party identity not stable/unique')
            ordered[identity]=mon
        if len(ordered)!=6:raise ValueError('six original party identities required')
        active.append(original[side].index(toid(rows[0]['set']['species'])))
        parties.append([ordered[name] for name in original[side]])
    return battle,parties,active


def compare(state,snapshot,original,vocab,*,initial=False):
    battle,parties,active=native_view(snapshot,original,vocab)
    differences=[]
    def check(field,native,gpu):
        if native!=gpu:differences.append(dict(field=field,showdown=native,gpu=gpu))
    check('active_party_indices',active,np.asarray(state.active)[0].tolist())
    if initial:
        weather={'':'_none','raindance':'rain','sunnyday':'sun','sandstorm':'sand','hail':'hail'}
        expected=['_none','rain','sun','sand','hail'][int(state.weather[0])]
        check('initial_weather',weather.get(battle['field']['weather'],battle['field']['weather']),expected)
    for side in range(2):
        for slot,mon in enumerate(parties[side]):
            prefix=f'{side}:{slot}:'
            check(prefix+'hp',mon['hp'],int(state.hp[0,side,slot]))
            # Showdown serializes a KO as status="fnt"; the tensor keeps
            # major status separate from HP/faint. Compare major status only
            # while either representation is alive, retaining independent KO
            # checks. This never resumes a previously stopped trajectory.
            if mon['hp']>0 or int(state.hp[0,side,slot])>0:
                check(prefix+'status',mon['status'],STATUS[int(state.status[0,side,slot])])
            check(prefix+'fainted',bool(mon['fainted']),int(state.hp[0,side,slot])<=0)
            if initial:
                check(prefix+'max_hp',mon['maxhp'],int(state.max_hp[0,side,slot]))
                check(prefix+'stats',[mon['maxhp']]+[mon['storedStats'][s] for s in ('atk','def','spa','spd','spe')],state.stats[0,side,slot].tolist())
                check(prefix+'ability',mon['ability'],vocab['abilities'][int(state.ability[0,side,slot])])
                check(prefix+'boosts',[mon['boosts'][s] for s in ('atk','def','spa','spd','spe','accuracy','evasion')],state.boosts[0,side,slot].tolist())
                check(prefix+'item',mon['item'],vocab['items'][int(state.item[0,side,slot])] if int(state.item[0,side,slot]) else '')
                expected=[canonical(vocab['moves'][int(x)]) for x in state.moves[0,side,slot] if int(x)]
                check(prefix+'moves',[x['id'] for x in mon['moveSlots']],expected)
                check(prefix+'pp',[x['pp'] for x in mon['moveSlots']],state.pp[0,side,slot,:len(expected)].tolist())
    return differences


def requested_seats(state):
    """The kernel's own request law, independent of the reference requests."""
    active_hp=np.take_along_axis(np.asarray(state.hp),np.asarray(state.active)[...,None],axis=2)[...,0]
    forced=np.asarray(state.forced_switch) | (active_hp<=0)
    return ((~forced.any(axis=1)[:,None] | forced) & ~np.asarray(state.done)[:,None])[0]


def completion(state,terminal):
    """Never pass a one-sided terminal or an unfinished twentieth turn."""
    native_done=terminal is not None;gpu_done=bool(state.done[0])
    if native_done!=gpu_done:
        return 'TERMINAL_MISMATCH',[dict(field='terminal',showdown=native_done,gpu=gpu_done)]
    if native_done:
        native_winner={'p1':0,'p2':1,None:2}[terminal.winner]
        if terminal.capped or native_winner!=int(state.winner[0]):
            return 'TERMINAL_MISMATCH',[dict(field='winner_or_cap',showdown=[native_winner,terminal.capped],gpu=[int(state.winner[0]),False])]
        return 'MATCHED_TERMINAL',None
    hp=np.take_along_axis(np.asarray(state.hp),np.asarray(state.active)[...,None],axis=2)[...,0]
    boundary=not bool(state.pending_residual[0]) and not np.any(np.asarray(state.forced_switch)[0] | (hp[0]<=0))
    if int(state.turn[0])>=20 and boundary:return 'MATCHED_THROUGH_TURN20',None
    return None,None


def boundary_projection(state,snapshot,original,vocab):
    """Bounded context retained at the stopped boundary, never used as repair."""
    try:
        battle,parties,active=native_view(snapshot,original,vocab)
        fields=('hp','maxhp','status','statusState','boosts','ability','item','moveSlots','volatiles','fainted')
        return dict(gpu={name:np.asarray(getattr(state,name))[0].tolist() for name in state._fields},
            showdown=dict(turn=battle.get('turn'),field=battle.get('field'),active=active,
                parties=[[{k:mon.get(k) for k in fields} for mon in party] for party in parties],
                requests=snapshot.bridge_snapshot.get('boundaryRequests'),log_tail=battle.get('log',[])[-80:]))
    except Exception as error:return dict(capture_error=str(error))


def reference_actions(env,state,snapshot,original,vocab,rng,*,choose=True):
    """Map full reference legal sets to original-party/canonical-move identities."""
    battle,parties,active=native_view(snapshot,original,vocab)
    mask=np.asarray(transition.action_mask(state,reference_actions.table))[0]
    gpu={};native={};maps={};differences=[]
    requested=requested_seats(state)
    native_requested=set(env.requested_players())
    expected_requested={p for s,p in enumerate(('p1','p2')) if requested[s]}
    if native_requested!=expected_requested:
        differences.append(dict(field='requested_players',showdown=sorted(native_requested),gpu=sorted(expected_requested)))
    for side,player in enumerate(('p1','p2')):
        mapping={}
        if player in env.requested_players():
            local=env._state_for_player(player)
            request=snapshot.bridge_snapshot['boundaryRequests'][player]
            for a,legal in enumerate(env.legal_actions(player)):
                if not legal:continue
                choice=showdown_choice_for_action(local,a);kind,number=choice.split();number=int(number)-1
                if kind=='switch':
                    identity=toid(battle['sides'][side]['pokemon'][number]['set']['species'])
                    target=original[side].index(identity)
                    translated=4+[s for s in range(6) if s!=active[side]].index(target)
                else:
                    name=canonical(request['active'][0]['moves'][number]['id'])
                    own=[canonical(vocab['moves'][int(x)]) for x in state.moves[0,side,active[side]]]
                    translated=own.index(name) if name in own else 0 if name=='struggle' else -1
                    if translated<0:
                        differences.append(dict(field=player+':action_identity',showdown=name,gpu=own))
                        continue  # e.g. unsupported Transform: first mismatch, never a harness retry
                mapping[a]=translated
        expected=set(np.flatnonzero(mask[side]).tolist()) if requested[side] else set();actual=set(mapping.values())
        if expected!=actual:differences.append(dict(field=player+':legal_actions',showdown=sorted(actual),gpu=sorted(expected)))
        if mapping and choose:
            chosen=rng.choice(sorted(mapping));native[player]=chosen;gpu[side]=mapping[chosen]
        maps[player]=mapping
    return native,j.array([[gpu.get(0,0),gpu.get(1,0)]],j.int32),differences,maps


def protocol_completed(rows):
    return len(rows)==200 and all(row['status']!='HARNESS_ERROR' for row in rows)


def run(pool_path,expected_sha,showdown,node,out):
    blob=Path(pool_path).read_bytes();pool_sha=hashlib.sha256(blob).hexdigest()
    if pool_sha!=expected_sha:raise ValueError('accepted pool digest differs')
    with np.load(io.BytesIO(blob),allow_pickle=False) as archive:
        metadata=json.loads(archive['metadata_utf8'].tobytes());pool={k:archive[k] for k in archive.files if k!='metadata_utf8'}
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    bound=('gpu_engine','tests','src','scripts')
    if git('status','--porcelain','--',*bound):raise RuntimeError('dirty diagnostic source')
    source=git('rev-parse','HEAD');paths=sorted((ROOT/'gpu_engine').glob('*'))+sorted((ROOT/'scripts').glob('battle_bridge*.mjs'))+sorted((ROOT/'src').rglob('*.py'))
    hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    before=hashes();packed,roster=reconstruct(pool,metadata,showdown,node)
    table=transition.tables_from_metadata(metadata);reference_actions.table=table
    base=transition.state_from_pool(pool,np.arange(2*FIRST_GAME,2*(FIRST_GAME+200)).reshape(200,2),seed=719)
    base=base._replace(keys=jax.vmap(lambda n:jax.random.fold_in(jax.random.PRNGKey(719),n))(j.arange(FIRST_GAME,FIRST_GAME+200)))
    base=jax.block_until_ready(transition.initialize(base,table));advance=jax.jit(lambda s,a:transition.step(s,a,table))
    header=dict(schema='pokezero.gpu_fidelity_trace.v1',source_commit=source,source_sha256=before,pool_sha256=pool_sha,
        roster=roster,format='gen3customgame; no Sleep/Freeze Clause',ordinary_turn_cap=20,request_cap=200,
        action_law='Uniform over complete reference legal actions, per-game string-seeded RNG; no mask intersection or state repair',
        chance='Known semantic raw inputs coupled; unknown/repeated/raw native roles stop unresolved; initialization RNG separately audited and unresolved except documented inert gender/QuickClaw setup; no native PRNG-stream parity',
        scope='Unfiltered original real teams; HP/status/faints/active identities at each normalized request; full original tensor regeneration and initial represented-state contract',
        jax=jax.__version__,backend=jax.default_backend())
    rows=[];journal=out.with_suffix('.jsonl')
    with journal.open('x') as stream:
        stream.write(json.dumps(dict(kind='ENTRY',**header))+'\n');stream.flush()
        for offset in range(200):
            game=FIRST_GAME+offset;state=jax.tree.map(lambda x:x[offset:offset+1],base)
            original=[[metadata['vocabulary']['species'][int(x)] for x in state.species[0,s]] for s in range(2)]
            row=dict(game_id=game,team_ids=[2*game,2*game+1],steps=[],status='REQUEST_CAP',first_divergence=None)
            rng=random.Random(f'gpu-fidelity-actions-v1:{game}')
            with tempfile.TemporaryDirectory(prefix='pokezero-fidelity-') as folder:
                folder=Path(folder);bridge=folder/'bridge.mjs'
                # LocalShowdownConfig resolves symlinks, so use a tiny owned
                # launcher whose argv path still identifies this temp directory.
                bridge.write_text('await import('+json.dumps((ROOT/'gpu_engine/fidelity_bridge.mjs').as_uri())+');\n')
                control=folder/'chance.json';audit=folder/'chance-audit.jsonl';initializing=folder/'initializing.json'
                # No gender-dependent selected move/ability may use an inert
                # gender draw. This check uses the original unfiltered teams.
                gender_sensitive=any(metadata['vocabulary']['moves'][int(x)]=='attract' for x in np.asarray(state.moves).flat)
                gender_sensitive |= any(metadata['vocabulary']['abilities'][int(x)]=='cutecharm' for x in np.asarray(state.ability).flat)
                initializing.write_text(json.dumps(dict(allow_inert_gender=not gender_sensitive)))
                env=LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown,node_binary=node,bridge_path=bridge,
                    feature_masks=ObservationFeatureMasks(transition_token_budget=0)))
                try:
                    env.reset_with_start_override(seed=670000000+offset,start_override=BattleStartOverride(
                        player_teams={'p1':packed[2*offset]['packed'],'p2':packed[2*offset+1]['packed']}))
                    initializing.unlink()
                    initial_events=[json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
                    row['initial_chance_audit']=initial_events
                    if audit.exists():audit.unlink()
                    snapshot=env.snapshot();diff=compare(state,snapshot,original,metadata['vocabulary'],initial=True)
                    if any(e['kind']=='INITIAL_RANDOM' for e in initial_events):
                        row.update(status='UNRESOLVED_INITIAL_CHANCE',first_divergence=diff or None)
                    elif diff:row.update(status='INITIAL_CONTRACT_MISMATCH',first_divergence=diff)
                    else:
                        for index in range(200):
                            native,actions,diff,maps=reference_actions(env,state,snapshot,original,metadata['vocabulary'],rng)
                            if diff:row.update(status='LEGAL_ACTION_MISMATCH',first_divergence=diff);break
                            if not native:row['status']='NO_REFERENCE_REQUEST';break
                            row.pop('post_boundary',None);row.pop('native_log_tail',None)
                            row['pre_boundary']=boundary_projection(state,snapshot,original,metadata['vocabulary'])
                            tape=chance_tape(state,game,index);control.write_text(json.dumps(tape))
                            step=dict(index=index,showdown_actions=native,gpu_actions=actions.tolist(),identity_maps=maps,chance=tape)
                            failure=None
                            try:env.step(native)
                            except Exception as error:failure=str(error)
                            finally:control.unlink(missing_ok=True)
                            events=[json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
                            step['chance_audit']=events;step['reference_error']=failure
                            # Audit is retained per request, never reparsed into a repaired state.
                            if audit.exists():audit.unlink()
                            row['steps'].append(step)
                            if any(e['kind']=='UNRESOLVED' for e in events):row['status']='UNRESOLVED_CHANCE_ROLE';break
                            if failure:row['status']='HARNESS_ERROR';break
                            result=jax.block_until_ready(advance(state,actions));state=result.state
                            if bool(np.asarray(result.invalid_actions)[0]):row['status']='GPU_INVALID_ACTION';break
                            snapshot=env.snapshot();diff=compare(state,snapshot,original,metadata['vocabulary'])
                            row['post_boundary']=boundary_projection(state,snapshot,original,metadata['vocabulary'])
                            row['native_log_tail']=snapshot.bridge_snapshot['battle'].get('log',[])[-80:]
                            if diff:row.update(status='STATE_MISMATCH',first_divergence=diff);break
                            status,diff=completion(state,env.terminal())
                            if status=='MATCHED_THROUGH_TURN20':
                                _,_,boundary_diff,_=reference_actions(env,state,snapshot,original,metadata['vocabulary'],rng,choose=False)
                                if boundary_diff:status,diff='LEGAL_ACTION_MISMATCH',boundary_diff
                            if status:row.update(status=status,first_divergence=diff);break
                except Exception as error:row.update(status='HARNESS_ERROR',error=str(error))
                finally:
                    try:env.close()
                    except Exception as error:row.update(status='HARNESS_ERROR',close_error=str(error))
            row['gpu_ordinary_turns']=int(state.turn[0]);rows.append(row)
            stream.write(json.dumps(dict(kind='GAME',**row))+'\n');stream.flush()
            print(json.dumps(dict(game_id=game,status=row['status'],ordinary_turns=row['gpu_ordinary_turns'])),flush=True)
            if row['status']=='HARNESS_ERROR':break
    if git('rev-parse','HEAD')!=source or git('status','--porcelain','--',*bound) or hashes()!=before:
        raise RuntimeError('bound source changed during trace')
    counts=Counter(r['status'] for r in rows)
    matched=sum(counts[s] for s in ('MATCHED_TERMINAL','MATCHED_THROUGH_TURN20'))
    result=dict(**header,protocol_completed=protocol_completed(rows),status_counts=dict(counts),rows=rows,
        matched=matched,remaining_or_nonmatched=200-matched,
        upper_nonmatch_fraction=(200-matched)/200,gate='NOT_QUALIFIED_PENDING_MECHANIC_CLASSIFICATION')
    with out.open('x') as target:json.dump(result,target,sort_keys=True,indent=2);target.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--team-pool',required=True)
    p.add_argument('--expected-pool-sha',required=True);p.add_argument('--showdown-root',required=True)
    p.add_argument('--node',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists() or out.with_suffix('.jsonl').exists():raise FileExistsError('exclusive result/journal required')
    result=run(a.team_pool,a.expected_pool_sha,Path(a.showdown_root),a.node,out)
    raise SystemExit(0 if result['protocol_completed'] else 1)
