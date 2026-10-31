"""UNIT archives and mocked original callsites only; no data or runtime calls."""
import ast
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import v5_contract as c, v5_data, v5_run as run
from pokezero import v5_evaluation as evaluation, v5_evaluation_capture as capture
from gpu_engine import v5_collection_data as original
from gpu_engine import v5_evaluation_capture as producer
import test_v5_contract as fixture


def artifact(path,value):
    path=path.resolve();path.write_text(run._json(value))
    return dict(path=str(path),sha256=run._file_sha(path))


def roster(path,arm,replicate,stream,*,role=capture.TRAINING,pool='a'*64,teams=(0,1)):
    game=run.Game(stream,'game-'+str(stream),stream+1,teams)
    batch=run.Batch((game,),stream+10,stream+11)
    return artifact(path,dict(schema=capture.ROSTER,role=role,arm=arm,replicate=replicate,pool_sha256=pool,batches=[asdict(batch)]))


def bank(root):
    training=[roster(root/(arm+str(rep)+'.json'),arm,rep,10+i,teams=(2*i,2*i+1))
        for i,(arm,rep) in enumerate((('A',0),('B',0),('D',0),('B',1)))]
    heldout=roster(root/'heldout.json','EVALUATION',0,100,role=capture.NONTRAINING,pool='e'*64)
    value=dict(schema=capture.BANK,role=capture.NONTRAINING,heldout_roster=heldout,training_rosters=training,
        selectors=[dict(stream=100,frame=9,owner=0),dict(stream=100,frame=12,owner=0)],world_recipe=dict(seed=93),
        opponent_replies=[[0,1],[0,1]],collector_archive_sha256='b'*64,opponent_archive_sha256='c'*64,
        collector_seats=[0],max_requests=30)
    return artifact(root/'bank.json',value),value


def observation(address):
    old=fixture.v4(owner=address.owner)
    p=c.PublicPosterior(fixture.vocabulary(),'blissey',0b101,(.2,.3,.5))
    return old,c.append_belief_tokens(old,(p,)*6,address=address,category_vocabulary_sha256='b'*64)


def decision(address):
    old,public=observation(address)
    body=dict(stream=address.stream,frame=address.frame,owner=address.owner,turn=3,pool_sha256='e'*64,
        team_indices=(0,1),physical=tuple(range(6))*2,public_inputs=public.model_inputs(),request_binding=[1]*8,
        engine_action=0,choice=0,behavior_probability=.7,value_estimate=.9,actor_archive_sha256='b'*64,
        opponent_archive_sha256='c'*64,training_only_peer_own_capture=None)
    return original.capture_decision(old,public,battle_id='game-100',seed=101,turn=3,choice=0,probability=.7,value=.9,
        physical=body['physical'],hidden_sets=(None,)*6,source_frame_sha256=original.sha(body),
        capture_metadata={k:v for k,v in body.items() if k!='public_inputs'})


def raw_state(*,done=False,winner=-1):
    # Explicit UNIT body, not an actual kernel state. Tests never attest it as one.
    return dict(done=[done],winner=[winner],hp=[[[5]*6,[6]*6]],max_hp=[[[10]*6,[10]*6]])


def view(observation):
    return dict(schema=c.TRAINING_CONTRACT_SCHEMA,observation_schema=c.OBSERVATION_SCHEMA_VERSION_V5,
        category_vocabulary_sha256=observation.category_vocabulary_sha256,label_vocabulary_sha256=observation.label_vocabulary.sha256,
        view='PUBLIC',history_mask=[[True]],**{name:[[getattr(observation,name)]] for name in
            ('categorical_ids','numeric_features','token_type_ids','attention_mask')})


def archives(root,bank_ref,bank_doc,*,terminal=True):
    address=c.DecisionAddress(**bank_doc['selectors'][0]);actual=decision(address)
    body=dict(stream=100,frame=14,status='COMPLETE' if terminal else 'CAPPED',reason='unit-terminal',pool_sha256='e'*64)
    if terminal:body.update(winner=1,hp=[[0]*6,[7]*6],max_hp=[[10]*6,[10]*6])
    closed=original.TerminalCapture(100,14,body['status'],original.sha(body),body.get('winner'),
        original._tuple(body.get('hp',())),original._tuple(body.get('max_hp',())))
    record=dict(schema=capture.POSITION,selector=asdict(address),game=asdict(run.Game(100,'game-100',101,(0,1))),
        status='CAPTURED',decision=v5_data.row_to_dict(actual.row),
        execution=dict(stream=100,before_frame=9,after_frame=10,executed=True,invalid=False,
            source_frame_sha256=actual.row.source_frame_sha256,after_state=raw_state(),raw_frame={'unit':True}),
        terminal=dict(body=body,capture=asdict(closed)),reason=None)
    unknown=dict(record,selector=bank_doc['selectors'][1],status='UNKNOWN',decision=None,execution=None,reason='not-observed')
    groups=[]
    from pokezero.v5_gumbel import Recipe
    recipe=Recipe(**bank_doc['world_recipe'])
    for i,selector in enumerate(bank_doc['selectors']):
        address=c.DecisionAddress(**selector)
        for world in (0,1):
            reply=bank_doc['opponent_replies'][i][world]
            if i:
                groups.append(producer._unknown(address,world,reply,'not-observed'));continue
            seed=int.from_bytes(recipe._digest(address,'hypothetical-kernel',world)[:4],'big')
            lines=['unit-source-lines-not-a-game'];lines_sha=hashlib.sha256(''.join(line+'\n' for line in lines).encode()).hexdigest()
            origin=dict(scope='HYPOTHETICAL_PUBLIC_WORLD_NOT_ORDINARY_SESSION',root=lines_sha,engine=[3]*8,sampled_teams={'p1':'x','p2':'y'},seed=seed)
            history=dict(stream=100,owner=0,own_constructor=[],frames=[dict(number=9)])
            group=dict(schema=capture.SIBLING,selector=selector,world=world,reply=reply,status='CAPTURED',reason=None,
                root_source_sha256=actual.row.source_frame_sha256,
                origin=dict(body=origin,sha256=run._sha(origin),history_sha256=run._sha(history),history=history,
                    owning_lines=lines,before_state=raw_state(),root_view=view(actual.row.observation),sampled_start={'player_teams':origin['sampled_teams']},
                    posterior_seed=recipe.world_seed(address,world),kernel_seed=seed),
                chance=dict(fields={'unit':[1]},sha256=run._sha({'unit':[1]})),
                root_legal=list(actual.row.observation.legal_action_mask),branches=[])
            for action in (0,1):
                _,child=observation(replace(address,frame=address.frame+1))
                choices=[reply,reply];choices[address.owner]=action
                group['branches'].append(dict(action=action,choices=choices,status='CAPTURED',refusals=[],state=raw_state(),
                    raw_frame={'unit':True},diagnostics=[],observation=capture.observation_to_dict(child),terminal=None))
                branch=group['branches'][-1]
                branch['receipt_sha256']=run._sha(dict(selector=selector,world=world,reply=reply,origin=group['origin']['sha256'],
                    chance=group['chance']['sha256'],returned=branch))
            groups.append(group)
    header=dict(split='PHASE_C_FIXED_HELDOUT',category_vocabulary_sha256='b'*64,label_vocabulary_sha256=fixture.vocabulary().sha256,
        capture_bank=bank_ref,capture_source=capture.source_binding())
    protocol=dict(label_vocabulary_sha256=fixture.vocabulary().sha256,bins=5,batch_size=4)
    for key,schema_name,rows in (('positions',evaluation.POSITIONS,[record,unknown]),('siblings',evaluation.SIBLINGS,groups)):
        path=root/(key+'.jsonl');path.write_text('\n'.join(run._json(x) for x in (dict(header,schema=schema_name,rows=len(rows)),*rows))+'\n')
        protocol[key]=dict(path=str(path.resolve()),sha256=run._file_sha(path))
    return protocol


def mutate(root,protocol,key,change):
    rows=[json.loads(line) for line in Path(protocol[key]['path']).read_text().splitlines()]
    change(rows);path=root/(key+'-mutant.jsonl');path.write_text('\n'.join(run._json(x) for x in rows)+'\n')
    return dict(protocol,**{key:dict(path=str(path),sha256=run._file_sha(path))})


class OriginalEvaluationCaptureTests(unittest.TestCase):
    def test_nontraining_membership_all_four_training_rosters_and_actual_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,value=bank(root);checked=capture.load_bank(ref)
            training=capture._artifact(value['training_rosters'][1]).document()
            directory=root/'run';directory.mkdir()
            declaration=dict(model=dict(arm='B',replicate=0),pool_sha256=training['pool_sha256'],iterations=[training['batches']])
            (directory/'declaration.json').write_text(run._json(declaration));request=NS(run_directory=directory,declaration_sha256=run._sha(declaration))
            capture.check_training_declaration(checked,request)
            declaration['iterations'][0][0]['games'][0]['seed']+=1
            (directory/'declaration.json').write_text(run._json(declaration))
            with self.assertRaisesRegex(ValueError,'training roster'):
                capture.check_training_declaration(checked,NS(run_directory=directory,declaration_sha256=run._sha(declaration)))
            for kind in ('role','missing-arm','stream','battle','team','constructor-root','selector-seat'):
                changed=json.loads(run._json(value))
                if kind=='role':changed['role']=capture.TRAINING
                elif kind=='missing-arm':changed['training_rosters'][2]=changed['training_rosters'][0]
                elif kind=='selector-seat':changed['selectors'][0]['owner']=1
                else:
                    document=capture._artifact(changed['training_rosters'][0]).document()
                    if kind=='stream':document['batches'][0]['games'][0]['stream_id']=100
                    if kind=='battle':document['batches'][0]['games'][0]['battle_id']='game-100'
                    if kind=='team':document['pool_sha256']='e'*64
                    if kind=='constructor-root':document['batches'][0]['constructor_seed']=110
                    changed['training_rosters'][0]=artifact(root/(kind+'-roster.json'),document)
                with self.subTest(kind=kind),self.assertRaises(ValueError):capture.load_bank(artifact(root/(kind+'-bank.json'),changed))

    def test_original_capture_and_terminal_finish_not_policy_target_or_zero_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,document=bank(root);protocol=archives(root,ref,document)
            resolved=capture.require_original_capture(protocol)
            self.assertEqual([p[2] for p in resolved.positions],[-1,None])
            self.assertEqual(len(resolved.groups),4)
            # Actual unfinalized behavior value is +.9; literal winner is peer.
            self.assertNotEqual(resolved.positions[0][2],.9)
            self.assertEqual(resolved.positions[1][1:],(None,None))
            capped=archives(root,ref,document,terminal=False)
            self.assertEqual([p[2] for p in capture.resolve_original_capture(capped,None).positions],[None,None])
            no_terminal=mutate(root,capped,'positions',lambda rows:[row.update(terminal=None) for row in rows[1:]])
            self.assertTrue(all(p[2] is None for p in capture.resolve_original_capture(no_terminal,None).positions))
            self.assertNotIn('step',resolved.groups[0]['branches'][0]['observation'])

    def test_record_and_world_mutations_refuse_without_any_runtime_or_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,document=bank(root);protocol=archives(root,ref,document)
            changes=(('positions',lambda r:r.pop()),
                ('positions',lambda r:r[1]['execution'].update(after_frame=11)),
                ('positions',lambda r:r[1]['decision']['step']['metadata']['v5_capture'].update(choice=1)),
                ('positions',lambda r:r[1]['terminal']['body'].update(winner=0)),
                ('siblings',lambda r:r[1]['branches'].pop()),
                ('siblings',lambda r:r[1].update(reply=2)),
                ('siblings',lambda r:r[1]['branches'][0].update(terminal={'winner':0})),
                ('siblings',lambda r:r[1]['branches'][0]['observation']['address'].update(frame=99)),
                ('siblings',lambda r:r[1]['chance']['fields'].update(unit=[2])),
                ('siblings',lambda r:r[0]['capture_source'].pop('gpu_engine/transition.py')))
            for key,change in changes:
                with self.subTest(key=key,change=change),self.assertRaises((ValueError,run.MissingDependency)):
                    capture.resolve_original_capture(mutate(root,protocol,key,change),None)
            old=evaluation._capture_resolver()
            self.assertIs(old,capture)
            with patch.object(capture,'SOURCE_FILES',('gpu_engine/transition.py',)),self.assertRaisesRegex(ValueError,'inventory'):
                capture.source_binding()
            rows=[json.loads(x) for x in Path(protocol['positions']['path']).read_text().splitlines()]
            for key in ('capture_bank','capture_source'):rows[0].pop(key)
            path=root/'old-unit-fixture.jsonl';path.write_text('\n'.join(run._json(x) for x in rows)+'\n')
            with self.assertRaises(ValueError):capture.require_original_capture(dict(protocol,positions=dict(path=str(path),sha256=run._file_sha(path))))

    def test_arithmetic_only_public_arrays_full_unknown_denominators_and_no_ranking_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,document=bank(root);protocol=archives(root,ref,document);resolved=capture.resolve_original_capture(protocol,None)
            inputs=[]
            def forward(value):
                inputs.append(value)
                return NS(value=np.linspace(.1,.3,len(value['history_mask'])),valid=np.ones(len(value['history_mask']),bool))
            factory=NS(config=NS(category_vocabulary_sha256='b'*64,label_vocabulary=fixture.vocabulary()),forward=forward)
            fake_model=NS(prepare_inputs=lambda x,config:x)
            with patch.dict(sys.modules,{'gpu_engine.v5_model':fake_model,'gpu_engine.v5_live_protocol':NS(check_factory=lambda f:None)}):
                result=capture.evaluate_original_capture(protocol,factory,resolved)
            self.assertEqual((result['known_outcomes'],result['unknown_outcomes'],result['sibling_groups'],result['unknown_sibling_groups']),(1,1,4,2))
            self.assertEqual(result['actual_ranking_accuracy'],'UNMEASURED_NO_MATCHED_TERMINAL_CONTINUATIONS')
            self.assertIn('collector-policy',result['interpretation']);self.assertFalse(result['scientific_qualification'])
            for value in inputs:
                self.assertFalse(set(value)&{'labels','outcomes','privileged','returns','advantages','state'})
                self.assertEqual(np.asarray(value['numeric_features']).shape[2:],(29,132))
            resolved.groups[0]['reply']=2
            with self.assertRaisesRegex(ValueError,'immutable resolved'):
                capture.evaluate_original_capture(protocol,factory,resolved)

    def test_mock_original_sibling_calls_share_world_chance_cover_legal_actions_and_keep_refusal(self):
        """Fake source call graph, never a claim of original kernel coverage."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,document=bank(root);checked=capture.load_bank(ref);recorder=producer.Recorder(checked)
            selected=decision(checked.selectors[0]);address=selected.row.observation.address
            history=NS(stream_id=100,owner=0,own_constructor=(),frames=(HistoryFrame(9),));start=Start({'p1':'x','p2':'y'})
            root_source=NS(address=address,owner_lines_sha256='d'*64,
                context=NS(public_materialization_state=NS(belief_engine=NS(set_source=object()))))
            calls=[];chances=[]
            def materialize(actual,initial,**kwargs):
                body=dict(scope='HYPOTHETICAL_PUBLIC_WORLD_NOT_ORDINARY_SESSION',root='d'*64,engine=[3]*8,sampled_teams=start.player_teams,seed=kwargs['seed'])
                from collections import namedtuple
                State=namedtuple('State','done seed')
                return NS(branch=NS(origin_sha256=run._sha(body),state=State(np.array([False]),np.array([kwargs['seed']]))),refusals=())
            def step(branch,choices,*,chance_inputs,hypothetical_profiles):
                calls.append((branch,choices,chance_inputs,hypothetical_profiles))
                return NS(refusals=('unit-explicit-refusal',),state=None,raw_frame=None,diagnostics=())
            def chance(state):
                value={'original_chance':int(state.seed[0])};chances.append(value);return value
            producer._RECORDERS[recorder]=dict(bank=checked,bank_sha256=run._sha(checked.document),source=capture.source_binding(),
                learner=NS(adapter=object()),prepared=object(),runtime={'request':{'rules':NS(device=NS(engine_binding=np.array([3]*8))) }},
                contract=object(),public_config=object())
            plans=NS(starts=(start,start),refusals=(None,None))
            public_root=NS(root=root_source,decoded=NS(refusal_counts={},lines=('unit-source-line',)))
            fake_modules={'gpu_engine.v5_owning_history':NS(public_root=lambda *a,**k:public_root),
                'gpu_engine.v5_public_world':NS(public_carry=lambda *a:object(),materialize=materialize),
                'gpu_engine.v5_public_branch':NS(policy_observation=lambda *a,**k:NS(legal_action_mask=np.array([[True,True]+[False]*7]),
                    inputs=view(selected.row.observation)),step=step),
                'gpu_engine.transition':NS(sample_chance_inputs=chance),'gpu_engine.v5_observation':NS(),
                'gpu_engine.v5_depth1_provider':NS(_same_root=lambda *a:None)}
            with patch.dict(sys.modules,fake_modules),patch('pokezero.v5_teacher_public_root.plan_starts',return_value=plans) as sampler:
                groups=producer._siblings(recorder,selected,history,object())
            self.assertEqual(sampler.call_count,1);self.assertEqual(len(chances),2);self.assertEqual(len(calls),4)
            self.assertIs(calls[0][2],calls[1][2]);self.assertIs(calls[2][2],calls[3][2]);self.assertIsNot(calls[0][2],calls[2][2])
            self.assertEqual([x[1] for x in calls],[(0,0),(1,0),(0,1),(1,1)])
            self.assertTrue(all(x[3] is True for x in calls))
            self.assertEqual([len(g['branches']) for g in groups],[2,2])
            self.assertTrue(all(b['status']=='UNKNOWN' and b['observation'] is None and b['terminal'] is None for g in groups for b in g['branches']))

    def test_collector_default_erasure_and_exact_once_original_finish(self):
        path=Path(__file__).resolve().parents[1]/'gpu_engine/v5_selfplay.py'
        tree=ast.parse(path.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect_paired')
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
            and n.func.value.id=='data' and n.func.attr=='finish' for n in ast.walk(function)),1)
        # Every observation-only effect is inside its explicit default-None branch.
        for n in ast.walk(function):
            if isinstance(n,ast.If) and ast.unparse(n.test)=='evaluation_recorder is not None':
                self.assertTrue(any(isinstance(x,ast.Call) for x in ast.walk(n)))
        calls=[n.func.attr for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=='evaluation_capture']
        self.assertCountEqual(calls,['_start','_terminal','_before','_after','_finish'])
        class Erase(ast.NodeTransformer):
            def visit_If(self,n):
                if ast.unparse(n.test)=='evaluation_recorder is not None':return None
                return self.generic_visit(n)
            def visit_FunctionDef(self,n):
                if n.name=='collect_paired':
                    n.args.kwonlyargs=n.args.kwonlyargs[:-1];n.args.kw_defaults=n.args.kw_defaults[:-1]
                return self.generic_visit(n)
            def visit_Assign(self,n):
                if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='finished':return None
                return self.generic_visit(n)
            def visit_Name(self,n):
                if n.id=='finished':return ast.parse('data.finish(decisions,closed)',mode='eval').body
                return n
        expected=ast.parse(subprocess.check_output(['git','show','d1076adb69e78364b205f68456c1ddaad329b38b:gpu_engine/v5_selfplay.py'],text=True))
        self.assertEqual(ast.dump(Erase().visit(tree),include_attributes=False),ast.dump(expected,include_attributes=False))

    def test_durable_attempt_before_constructor_interruption_skips_never_replays_and_joins_batch_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,document=bank(root);checked=capture.load_bank(ref)
            def create():
                value=producer.Recorder(checked)
                producer._RECORDERS[value]=dict(bank=checked,bank_sha256=run._sha(checked.document),source=capture.source_binding(),
                    current=None,positions={},groups={},terminals={})
                return value
            recorder=create();directory=root/'output'
            with producer._ownership(directory),self.assertRaisesRegex(ValueError,'already has an owner'):
                with producer._ownership(directory):pass
            path,declaration,attempted=producer._journal(directory,recorder,resume=False)
            self.assertEqual(attempted,0)
            producer._claim(path,dict(kind='BATCH_STARTED',batch=0,binding=run._sha(declaration),recipe=asdict(checked.batches[0])))
            # Crash here, before any source session: resume skips the entire attempted batch.
            resumed=create();_,_,attempted=producer._journal(directory,resumed,resume=True)
            self.assertEqual(attempted,1);self.assertEqual(producer._record(resumed)['positions'],{})
            with self.assertRaises(ValueError):producer._journal(directory,create(),resume=False)
            end=producer._save_batch(directory,recorder,0,run._sha(declaration));producer._claim(path,end)
            self.assertEqual(producer._journal(directory,create(),resume=True)[2],1)
            producer._claim(path,dict(kind='BATCH_STARTED',batch=0,binding=run._sha(declaration),recipe=asdict(checked.batches[0])))
            with self.assertRaisesRegex(ValueError,'order/replay'):producer._journal(directory,create(),resume=True)

    def test_mock_collector_same_call_ticket_and_actual_terminal_no_supplied_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,document=bank(root);checked=capture.load_bank(ref);recorder=producer.Recorder(checked)
            actual=decision(checked.selectors[0]);learner=NS(config=NS(category_vocabulary_sha256='b'*64,label_vocabulary=fixture.vocabulary()))
            from collections import namedtuple
            State=namedtuple('State','hp max_hp done winner')
            Report=namedtuple('Report','raw_count')
            engine=State(np.array([[[5]*6,[6]*6]]),np.full((1,2,6),10),np.array([False]),np.array([-1]))
            core=NS(cursor=NS(stream_id=np.array([100])),frame_number=np.array([9]),engine=engine,frame_input_report=Report(np.array([3])))
            state=NS(core=core);session=NS(state=state,result=NS(state=state))
            diagnostics=NS(executed_rows=np.array([True]),invalid_actions=np.array([False]))
            current=dict(session=session,pending=None)
            producer._RECORDERS[recorder]=dict(bank=checked,bank_sha256=run._sha(checked.document),source=capture.source_binding(),learner=learner,
                current=current,positions={},groups={},terminals={})
            fake=NS(_checked=lambda *a:None,driver=NS(latest_diagnostics=lambda s:diagnostics))
            groups=[producer._unknown(actual.row.observation.address,w,w,'unit-no-child') for w in (0,1)]
            with patch.dict(sys.modules,{'gpu_engine.v5_collection':fake}),patch.object(producer,'_siblings',return_value=groups):
                producer._before(recorder,session,pending=((0,actual),),histories={},own_statics={})
                with self.assertRaisesRegex(ValueError,'unconsumed'):producer._before(recorder,session,pending=(),histories={},own_statics={})
                session.result=object();core.frame_number=np.array([10])
                producer._after(recorder,session,diagnostics=diagnostics)
                self.assertEqual(producer._record(recorder)['positions'][actual.row.observation.address]['execution']['after_frame'],10)
                core.frame_number=np.array([14]);core.engine=engine._replace(done=np.array([True]),winner=np.array([1]))
                body=dict(stream=100,frame=14,status='COMPLETE',reason='unit-actual-terminal',pool_sha256='e'*64,
                    winner=1,hp=core.engine.hp[0].tolist(),max_hp=core.engine.max_hp[0].tolist())
                terminal=original.TerminalCapture(100,14,'COMPLETE',original.sha(body),1,
                    original._tuple(body['hp']),original._tuple(body['max_hp']))
                producer._terminal(recorder,session,terminal=terminal,body=body)
                with self.assertRaisesRegex(ValueError,'replay'):producer._terminal(recorder,session,terminal=terminal,body=body)
                finished=original.finish((actual,),(terminal,))
                self.assertEqual(finished[0].row.labels.outcome,-1)
                with patch.object(original,'finish',side_effect=AssertionError('must not run finish twice')):
                    producer._finish(recorder,session,decisions=(actual,),terminals=(terminal,),finished=finished)
            references=producer._publish(recorder,root/'out')
            resolved=capture.resolve_original_capture(dict(references,label_vocabulary_sha256=fixture.vocabulary().sha256),None)
            self.assertEqual([row[2] for row in resolved.positions],[-1,None])
            with self.assertRaisesRegex(ValueError,'original source-bound'):producer._record(producer.Recorder(checked))

    def test_mock_declared_bank_constructs_original_seeds_only_after_durable_claim_and_never_writes_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,document=bank(root);directory=root/'capture';prepared=object();pool=object();vocabulary=object()
            config=NS(category_vocabulary_sha256='b'*64,label_vocabulary=fixture.vocabulary())
            learner=NS(config=config);opponent=NS(config=config);constructed=object();calls=[]
            class Public:pass
            class Contract:pass
            contract=Contract();contract.category_vocab_sha256='d'*64;contract.feature_masks=();contract.belief_set_source_hash='e'*64
            public=Public();public.set_belief_source=True;public.category_vocab=vocabulary;public.feature_masks=()
            runtime=dict(catalog=NS(vocabulary=vocabulary,source=NS(source_hash='e'*64)),frame_witness_compiled=object())
            def construct(actual,indices,**kwargs):
                self.assertIs(actual,pool)
                event=json.loads((directory/'capture-journal.jsonl').read_text().splitlines()[-1])
                self.assertEqual(event['kind'],'BATCH_STARTED')
                self.assertEqual(kwargs['seed'],110);self.assertEqual(kwargs['stream_ids'].tolist(),[100])
                self.assertEqual(indices.tolist(),[[0,1]]);calls.append(kwargs)
                return NS(state=NS(core=NS(frame_number=np.array([0]))),result=object())
            driver=NS(_checked=lambda p:runtime,_checked_witness=lambda r:None,construct=construct,_session=lambda s:{'constructed':constructed})
            fake_original=NS(driver=driver,_checked=lambda *a:None,Cell=run.Game,declare_cells=lambda cells:object())
            def collect(session,**kwargs):
                self.assertEqual(kwargs['sampling_seed'],111)
                recorder=kwargs['evaluation_recorder']
                producer._start(recorder,session,learner=kwargs['learner'],actors=kwargs['actors'],ledger=kwargs['ledger'],
                    team_indices=kwargs['team_indices'],sampling_seed=kwargs['sampling_seed'])
                producer._finish(recorder,session,decisions=(),terminals=(),finished=())
            def actor(actual,original):
                self.assertIs(original,prepared)
                return None,'b'*64 if actual is learner else 'c'*64
            fakes={'gpu_engine.v5_collection':fake_original,'gpu_engine.v5_selfplay':NS(collect_paired=collect),
                'gpu_engine.source_provenance':NS(_pool=lambda *a,**k:{'pool_sha256':'e'*64}),
                'gpu_engine.v5_depth1_provider':NS(_actor=actor),
                'pokezero.live_protocol_actor':NS(ActorContract=Contract,_vocab_sha=lambda v:'d'*64,_mask_values=lambda x:()),
                'pokezero.local_showdown':NS(LocalShowdownConfig=Public)}
            with patch.dict(sys.modules,fakes):
                references=producer.collect_declared_bank(ref,prepared=prepared,pool=pool,learner=learner,opponent=opponent,
                    public_config=public,contract=contract,directory=directory,max_requests=30,on_end=lambda *a,**k:None,check_progress=lambda:None)
                self.assertEqual(len(calls),1)
                with self.assertRaisesRegex(ValueError,'explicit resume'):
                    producer.collect_declared_bank(ref,prepared=prepared,pool=pool,learner=learner,opponent=opponent,
                        public_config=public,contract=contract,directory=directory,max_requests=30,on_end=lambda *a,**k:None,check_progress=lambda:None)
                self.assertEqual(len(calls),1)
            records=[json.loads(line) for line in Path(references['positions']['path']).read_text().splitlines()[1:]]
            self.assertEqual([row['status'] for row in records],['UNKNOWN','UNKNOWN'])
            self.assertEqual(sorted(path.name for path in directory.iterdir()),
                ['batch-0.json','capture-declaration.json','capture-journal.jsonl','capture.lock','positions.jsonl','siblings.jsonl'])


@dataclass(frozen=True)
class Start:
    player_teams:dict


@dataclass(frozen=True)
class HistoryFrame:
    number:int


if __name__=='__main__':unittest.main()
