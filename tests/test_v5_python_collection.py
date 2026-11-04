"""Inert owning transport, asymmetric choices and literal label/cache controls.

No simulator, checkpoint, forward or real game runs in these tests.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero import v5_python_collection as p,v5_contract as c
from pokezero.category_vocab import CategoryVocabulary
from pokezero.observation import PokeZeroObservationV0,ObservationPerspective
from pokezero.policy import PolicyDecision
from pokezero.randbat import Gen3RandbatVariant
from pokezero.env import TerminalState

NAMES=('blissey','swampert','absol','ditto','groudon','pikachu')
VOCAB=c.LabelVocabulary('a'*64,('surf','recover'),tuple((n,2)for n in NAMES))
CATEGORIES=CategoryVocabulary(tuple('species:'+n for n in NAMES))


def observation(owner,frame,stream):
    cats=[[0]*41 for _ in range(23)]
    for i in range(6):cats[1+i][0]=1+i
    cats[7][0]=1
    old=PokeZeroObservationV0(tuple(map(tuple,cats)),((0.,)*132,)*23,
        (0,)+(1,)*6+(2,)*6+(3,)*9+(5,),
        (True,)*8+(False,)*5+(True,)*10,(True,False,False,False,True,False,False,False,False),
        perspective=ObservationPerspective.from_showdown_slot('p'+str(owner+1),'p'+str(owner+1)),
        schema_version=c.OBSERVATION_SCHEMA_VERSION_V4)
    posteriors=(c.PublicPosterior(VOCAB,'blissey',3,(.5,.5)),)+(c.PublicPosterior(VOCAB,None,0,(),reason='UNKNOWN_SPECIES'),)*5
    return old,c.append_belief_tokens(old,posteriors,address=c.DecisionAddress(stream,frame,owner),category_vocabulary_sha256='b'*64)


def teams(final=False):
    result=[]
    for side in range(2):
        rows=[]
        for i,name in enumerate(NAMES):
            rows.append(dict(name=name,set=dict(species=name,moves=['recover','surf']if side else['surf','recover'],
                level=80,ability='pressure',item='leftovers'),hp=(0 if side or i==0 else 50)if final else 100,
                maxhp=100,hpType='Ice'))
        result.append(dict(pokemon=rows[::-1]if final else rows))
    return result


class Actor:
    def begin_game(self,**identity):
        self.__dict__.update(identity);self.lines=[];self.generation=0;self.prefix_sha256='c'*64;self.status='ACTIVE'
    def accept_owner_stream(self,**packet):
        if packet['owner']!=self.owner:raise AssertionError('peer packet')
        self.lines.extend(packet['lines']);self.generation+=1
    def snapshot_boundary(self):
        return SimpleNamespace(own_request_line=next(x for x in reversed(self.lines)if x.startswith('|request|')),
            prefix_sha256=self.prefix_sha256)
    def decide(self,snapshot,*,rng):
        self.current=snapshot;self.index=4 if any(x=='|turn|2'for x in self.lines)else 0
        return PolicyDecision(self.index,'inert',.5,value_estimate=.2)
    def decision_v5_observation(self,snapshot,*,frame):return observation(int(self.owner[1])-1,frame,self.stream_id)
    def choice(self,snapshot):return 'switch 2'if self.index==4 else'move 1'
    def close_game(self):self.status='CLOSED';return dict(status='TERMINAL_UNQUALIFIED')


class Env(p.OwningEnv):
    def __init__(self,fail=False,cap=False,missing=False):
        self.owner_lines={'p1':[],'p2':[]};self.frame=0;self.fail=fail;self.cap=cap;self.missing=missing
    def reset(self,**kwargs):
        self.frame=0;self.owner_lines={seat:['|player|p1|One|','|player|p2|Two|']for seat in('p1','p2')};self._publish()
    def _publish(self):
        for seat in('p1','p2'):
            request={'side':{'id':seat,'pokemon':[dict(details=n+', L80')for n in NAMES]},
                'active':[{'moves':[{'id':'recover'if seat=='p2'else'surf'}]}]}
            if self.frame and seat=='p2':request['wait']=True;request.pop('active')
            self.owner_lines[seat].extend(('|turn|'+str(self.frame+1),'|request|'+json.dumps(request)))
    def snapshot(self):
        result=teams(final=self.terminal()is not None)
        if self.missing:result[0]['pokemon'][0].pop('maxhp')
        return SimpleNamespace(bridge_snapshot={'battle':{'sides':result}})
    def terminal(self):return TerminalState('p1',2)if self.frame==2 and not self.cap else None
    def requested_players(self):return('p1','p2')if self.frame==0 else('p1',)
    def _choices_for_actions(self,actions):return{seat:'switch 2'if i==4 else'move 1'for seat,i in actions.items()}
    def step(self,actions):
        if self.fail and self.frame:raise RuntimeError('inert advancement failure')
        self.frame+=1
        if self.terminal():
            for lines in self.owner_lines.values():lines.append('|win|One')
        else:self._publish()


def factories():
    variants=((),)+tuple(tuple(Gen3RandbatVariant(str(i),'original',n,'role',80,
        ('Surf','Recover'),'Pressure','Leftovers')for i in range(2))for n in NAMES)
    source=SimpleNamespace(species_id=lambda n:NAMES.index(n)+1,species_names=('',)+NAMES,variants=variants)
    return{owner:SimpleNamespace(deterministic=False,config=SimpleNamespace(arm='B',category_vocabulary_sha256='b'*64,
        label_vocabulary=VOCAB),catalog=SimpleNamespace(vocabulary=CATEGORIES,source=source),create_actor=Actor)for owner in(0,1)}


class PythonCollectionTests(unittest.TestCase):
    def collect(self,env,**kwargs):
        with patch('gpu_engine.v5_live_protocol.check_factory'):
            return p.collect_game(env,p.Game(1,'fixed',42,7),factories=factories(),max_requests=2,enable_v5=True,**kwargs)

    def test_same_frame_peer_labels_and_terminal_physical_order(self):
        result=self.collect(Env());self.assertEqual(result.status,'COMPLETE');self.assertEqual(len(result.decisions),3)
        first,peer,last=(x.row for x in result.decisions)
        self.assertEqual(first.labels.opponent_next_action.identity,'recover')
        self.assertEqual(peer.labels.opponent_next_action.identity,'surf')
        self.assertIsNone(last.labels.opponent_next_action) # never borrow prior-frame peer choice
        self.assertEqual(first.labels.hidden_sets[0].variant_indices,(0,1))
        self.assertEqual(first.labels.hidden_sets[1:],(None,)*5)
        self.assertEqual(first.labels.outcome,1);self.assertEqual(peer.labels.outcome,-1)
        self.assertEqual([x.final_hp_fraction for x in first.labels.pokemon_end[:6]],[0,.5,.5,.5,.5,.5])
        self.assertEqual(first.labels.pokemon_end[7:],(None,)*5)
        self.assertTrue(all(x.source_frame_sha256 for x in(first,peer,last)))

    def test_cap_and_advancement_failure_preserve_unknowns_and_only_committed_prefix(self):
        for env,status,rows in((Env(cap=True),'CAPPED',3),(Env(fail=True),'FAILED',2),(Env(missing=True),'FAILED',0)):
            result=self.collect(env);self.assertEqual(result.status,status);self.assertEqual(len(result.decisions),rows)
            for captured in result.decisions:
                self.assertIsNone(captured.row.labels.outcome)
                self.assertEqual(captured.row.labels.pokemon_end,(None,)*12)
        # A new independent declared game remains possible on the warm env.
        self.assertEqual(self.collect(Env()).status,'COMPLETE')

    def test_selected_training_seat_does_not_collect_historical_peer_rows(self):
        result=self.collect(Env(),collect_owners=(0,));self.assertEqual(len(result.decisions),2)
        self.assertTrue(all(x.row.observation.address.owner==0 for x in result.decisions))

    def test_semantic_switch_uses_exact_request_target_and_unknown_move_masks_only_label(self):
        request={'side':{'pokemon':[dict(details=n+', L80')for n in NAMES]}}
        self.assertEqual(p._semantic('switch 2',request,VOCAB),c.OpponentAction('SWITCH','swampert'))
        request['active']=[{'moves':[{'id':'recharge'}]}]
        self.assertIsNone(p._semantic('move 1',request,VOCAB))
        with self.assertRaises(ValueError):p._semantic('switch 8',request,VOCAB)

    def test_only_original_isolated_role_packets_are_retained(self):
        env=object.__new__(p.OwningEnv);env._battle_token='owned';env.owner_lines={'p1':[],'p2':[]}
        with patch.object(p.LocalShowdownEnv,'_apply_event',return_value=False):
            for role in('omniscient','p1','p2'):
                env._apply_event(dict(type='stream',battleId='owned',stream=role,lines=['|turn|1']))
            env._apply_event(dict(type='stream',battleId='foreign',stream='p1',lines=['|turn|99']))
            self.assertEqual(env.owner_lines,{'p1':['|turn|1'],'p2':['|turn|1']})
            with self.assertRaises(ValueError):env._apply_event(dict(type='stream',battleId='owned',stream='p1',lines=[2]))

    def test_permission_error_propagates_and_no_game_retry_occurs(self):
        env=Env()
        with patch.object(env,'reset',side_effect=PermissionError('inert denied'))as reset:
            with self.assertRaises(PermissionError):self.collect(env)
            self.assertEqual(reset.call_count,1)

    def test_original_cache_consumes_bridge_rows_and_unknowns_never_become_ties(self):
        from gpu_engine import v5_collection_data as data
        from pokezero import v5_data
        from pokezero.v5_loss import V5LossConfig
        complete=self.collect(Env())
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'cache'
            data.write_cache(path,complete.decisions,loss_config=V5LossConfig(arm='B'),batch_size=2)
            batches=tuple(v5_data.iter_cache_batches(path,batch_size=2))
            self.assertEqual(sum(b.batch_size for b in batches),3)
            self.assertEqual([x for b in batches for x in b.v5_columns['outcome_targets']],[1,-1,1])
            self.assertTrue(all(x for b in batches for x in b.v5_columns['outcome_target_mask']))
        capped=self.collect(Env(cap=True))
        batch=next(data.cache_batches(capped.decisions,loss_config=V5LossConfig(arm='B'),batch_size=3))
        self.assertEqual(batch.v5_columns['outcome_target_mask'],(False,)*3)
        self.assertEqual(batch.ppo_advantage_mask,(True,False,False))
        self.assertEqual(batch.ppo_value_target_mask,(True,False,False))

    def test_private_birth_change_changes_labels_but_never_actor_inputs_or_choices(self):
        original=self.collect(Env())
        class Changed(Env):
            def snapshot(self):
                result=super().snapshot()
                for mon in result.bridge_snapshot['battle']['sides'][1]['pokemon']:
                    if mon['name']=='blissey':mon['set']['item']='lumberry'
                return result
        changed=self.collect(Changed());self.assertEqual(changed.status,'COMPLETE')
        for left,right in zip(original.decisions,changed.decisions):
            self.assertEqual(left.row.observation.model_inputs(),right.row.observation.model_inputs())
            self.assertEqual(left.row.step.action_index,right.row.step.action_index)
            self.assertEqual(left.row.step.action_probability,right.row.step.action_probability)
        self.assertIsNone(changed.decisions[0].row.labels.hidden_sets[0])
        self.assertEqual(changed.auxiliary_refusals['hidden_set_outside_source'],2)

    def test_terminal_player_headers_and_valid_tie_join_without_guessing(self):
        for names,marker,terminal,expected in(
                (('One','Two'),'|win|Two',TerminalState('p2',2),1),
                (('One','Two'),'|tie',TerminalState(None,2),2),
                (('One','Two'),'|tie|',TerminalState(None,2),2)):
            lines={seat:['|player|p1|'+names[0]+'|','|player|p2|'+names[1]+'|']for seat in('p1','p2')}
            self.assertEqual(p._terminal_winner(lines,marker,terminal),expected)
        for headers,marker,terminal in(
                (['|player|p1|One|'],'|win|One',TerminalState('p1',2)),
                (['|player|p1|One|','|player|p2|One|'],'|win|One',TerminalState('p1',2)),
                (['|player|p1|One|','|player|p2|Two|'],'|win|Other',TerminalState('p1',2)),
                (['|player|p1|One|','|player|p2|Two|'],'|win|Two',TerminalState('p1',2)),
                (['|player|p1|One|','|player|p2|Two|'],'|tie',TerminalState('p1',2))):
            with self.assertRaises(ValueError):p._terminal_winner({seat:headers for seat in('p1','p2')},marker,terminal)

    def test_valid_literal_tie_preserves_zero_outcomes(self):
        class Draw(Env):
            def terminal(self):return TerminalState(None,2)if self.frame==2 else None
            def step(self,actions):
                super().step(actions)
                if self.terminal():
                    for lines in self.owner_lines.values():lines[-1]='|tie'
        result=self.collect(Draw());self.assertEqual(result.status,'COMPLETE')
        self.assertTrue(all(x.row.labels.outcome==0 for x in result.decisions))


if __name__=='__main__':unittest.main()
