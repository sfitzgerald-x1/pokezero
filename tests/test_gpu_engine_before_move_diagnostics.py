"""SAME-call literal observation only; no engine turns/models/game/RNG."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import jax.numpy as j
import numpy as np
from gpu_engine import engine_outcome_diagnostics as d,mechanics as m
from gpu_engine import benchmark_capture_gap_census as census
import test_gpu_engine_outcome_diagnostics as fixture

BASE='b34783d025a54d626404659a716d85fd174898e1'


def infatuation_refusal_proof(tree, contact_tree):
    """The omitted diagnostic fields are sound only behind these refusals."""
    dump=lambda n:ast.dump(n,include_attributes=False)
    expression=lambda s:dump(ast.parse(s,mode='eval').body)
    function=lambda t,name:next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
    def branch(body,condition):
        return next(n for n in body if isinstance(n,ast.If)and dump(n.test)==expression(condition))
    def statement(body,code):
        expected=dump(ast.parse(code).body[0])
        matches=[n for n in body if dump(n)==expected]
        assert len(matches)==1,code
        return matches[0]
    lum=function(tree,'_lum_update')
    carried=branch(lum.body,'_status_trace is not None')
    statement(carried.body,'_status_trace = status_callbacks.unknown(_status_trace, jnp.asarray(eligible) & jnp.any((state.volatile & jnp.uint32(ATTRACT)) != 0, axis=(1,2)))')
    assert lum.body.index(carried)<lum.body.index(branch(lum.body,'table.infatuation is not None and _infatuation_update'))
    assert lum.body.index(carried)<lum.body.index(branch(lum.body,'table.items is None'))
    initial=branch(function(tree,'initialize').body,'_weather_trace is not None')
    refusal=statement(initial.body,'_weather_trace = status_callbacks.unknown(_weather_trace, pending_battle & jnp.any((state.volatile & jnp.uint32(ATTRACT)) != 0, axis=(1,2)))')
    assert initial.body.index(refusal)<next(i for i,n in enumerate(initial.body)if isinstance(n,ast.Return))
    step=function(tree,'step_with_chance')
    first_update=next(n for n in step.body if isinstance(n,ast.If)and dump(n.test)==expression('_lifecycle_trace')
        and any(isinstance(x,ast.Assign)and isinstance(x.value,ast.Call)and isinstance(x.value.func,ast.Name)
                and x.value.func.id=='_lum_update' for x in n.body))
    statement(first_update.body,'work, _public_trace = _lum_update(work, table, live, _public_trace, _device_public=_device_public)')
    init_calls=[n for n in ast.walk(step)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='initialize']
    action=next(n for n in step.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='order')
    assert max(n.lineno for n in init_calls)<first_update.lineno<action.lineno
    before=next(n for n in ast.walk(step)if isinstance(n,ast.Call)and dump(n.func)==expression('m.before_move'))
    kwargs={k.arg:dump(k.value)for k in before.keywords}
    assert kwargs['infatuated']==expression('((_active(work.volatile,work,side)&jnp.uint32(ATTRACT))!=0)&(_active(work.hp,work,foe)>0)')
    assert kwargs['attract_roll']==expression('effect_draws[rows,side,4]')
    attract=next(n for n in ast.walk(step)if isinstance(n,ast.If)and dump(n.test)==expression('table.infatuation is not None')
        and any(isinstance(x,ast.Call)and dump(x.func)==expression('infatuation.apply')for x in ast.walk(n)))
    statement(branch(attract.body,'_lifecycle_trace').body,'_public_trace = status_callbacks.unknown(_public_trace, acts & table.infatuation.attract[move])')
    contact=branch(function(contact_tree,'apply_contact').body,'infatuation_table is not None')
    statement(branch(contact.body,'_status_trace is not None').body,'_status_trace = callbacks.unknown(_status_trace, attracted)')


class BeforeMoveDiagnosticsTests(unittest.TestCase):
    def setUp(self):self.t,_=fixture.OutcomeDiagnosticTests().tape()

    def inputs(self):
        z=j.arange(8,dtype=j.int32);yes=j.ones(8,bool)
        before=m.BeforeMove(j.array([1,6,0,5,1,0,1,6],j.int32),z+2,z+3,
            j.array([False,False,False,False,True,True,False,False]),j.array([False,False,True,False,False,False,False,False]),j.zeros(8,bool))
        args=(before,z%2,z%2+3,z+91,z+90,before.status,z+7,z+8)
        kwargs=dict(hp=z+100,maximum=z+200,ability=z+1,thaw_roll=z+20,paralysis_roll=z+21,confusion_roll=z+22,
            flinched=z%2==0,loafing=z%3==0,sleep_usable=z%2==1,early_bird=z%3==1,
            eligible=j.array([True,True,True,True,True,False,True,False]),active=yes)
        return args,kwargs

    def test_exact_inputs_outputs_sticky_identity_mixed_rows_and_bounded_overflow(self):
        args,kwargs=self.inputs();active=j.array([True,False]*4);kwargs['active']=active
        t=self.t;a=d.record_before_move_on_tape(t,*args,**kwargs)
        for name in t._fields:
            if name!='execution':self.assertIs(getattr(t,name),getattr(a,name))
        self.assertIs(a.execution.ledger,t.execution.ledger);self.assertIs(a.execution.risks,t.execution.risks)
        cap=a.execution.diagnostics
        self.assertEqual(cap.before_move_count.tolist(),[1,0]*4)
        row=d.refused_rows(cap,np.array([True]+[False]*7))['rows'][0]['before_move']['records'][0]
        self.assertEqual((row['actor_side'],row['actor_physical'],row['move'],row['caller_move']),(0,3,91,90))
        self.assertEqual((row['status'],row['counter'],row['confusion_counter'],row['hp'],row['maximum'],row['ability']),(1,7,8,100,200,1))
        self.assertEqual((row['thaw_roll'],row['paralysis_roll'],row['confusion_roll']),(20,21,22))
        self.assertEqual((row['result_status'],row['result_counter'],row['result_confusion_counter'],row['can_act'],row['confusion_self_hit']),(1,2,3,0,0))
        self.assertEqual((row['stream_id'],row['frame'],row['raw_count'],row['turn'],row['generation0'],row['generation1']),(12001,5,7,1,2,3))
        self.assertEqual((row['prior_tape_refusal'],row['prior_execution_refusal'],row['prior_kernel_risk']),(30,10,20))
        b=d.record_before_move_on_tape(a,*args,**kwargs);c=d.record_before_move_on_tape(b,*args,**kwargs)
        self.assertEqual(c.execution.diagnostics.before_move_count.tolist(),[2,0]*4)
        self.assertEqual(c.execution.diagnostics.before_move_overflow.tolist(),active.tolist())
        np.testing.assert_array_equal(c.execution.diagnostics.before_move_words,b.execution.diagnostics.before_move_words)
        self.assertIs(c.execution.ledger,t.execution.ledger);self.assertIs(c.execution.risks,t.execution.risks)
        np.testing.assert_array_equal(c.refusal,t.refusal);np.testing.assert_array_equal(c.words,t.words)
        # Ineligible calls remain literal records; active controls storage only.
        kwargs['active']=j.ones(8,bool);all_rows=d.record_before_move_on_tape(t,*args,**kwargs)
        self.assertEqual(all_rows.execution.diagnostics.before_move_count.tolist(),[1]*8)
        self.assertEqual(int(all_rows.execution.diagnostics.before_move_words[5,0,d.BEFORE_MOVE_FIELDS.index('eligible')]),0)

    def test_missing_malformed_store_and_legacy_serialization_are_inert(self):
        args,kwargs=self.inputs();t=self.t
        for cap in (None,object(),d.empty(1),d.Capture(*t.execution.diagnostics[:8]),d.empty(8)._replace(before_move_words=j.zeros((8,2,len(d.BEFORE_MOVE_FIELDS)),j.float32))):
            bad=t._replace(execution=t.execution._replace(diagnostics=cap))
            self.assertIs(d.record_before_move_on_tape(bad,*args,**kwargs),bad)
        bad=t._replace(execution=t.execution._replace(risks=None));self.assertIs(d.record_before_move_on_tape(bad,*args,**kwargs),bad)
        self.assertIs(d.record_before_move_on_tape(t,object(),*args[1:],**kwargs),t)
        old={};exec(compile(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_outcome_diagnostics.py'],text=True),'original-diagnostic-source','exec'),old)
        legacy=d.Capture(*t.execution.diagnostics[:8]);mask=np.array([True,False]*4)
        original=old['Capture'](*legacy[:8]);self.assertEqual(d.refused_rows(legacy,mask),old['refused_rows'](original,mask))
        class CountCopies:
            def __init__(self,value):self.value=np.asarray(value);self.shape=self.value.shape;self.dtype=self.value.dtype;self.copies=0
            def __array__(self,dtype=None,copy=None):self.copies+=1;return np.asarray(self.value,dtype=dtype)
            def __getitem__(self,index):raise AssertionError('device-side row gather')
        capture=d.record_before_move_on_tape(t,*args,**kwargs).execution.diagnostics
        wrapped=d.Capture(*(CountCopies(x)for x in capture));rows=d.refused_rows(wrapped,mask)['rows']
        self.assertEqual([x['row']for x in rows],[0,2,4,6]);self.assertTrue(all(x.copies==1 for x in wrapped))
        self.assertEqual(d.refused_rows(wrapped,np.zeros(8,bool))['rows'],[])
        self.assertTrue(all(x.copies==1 for x in wrapped))

    def test_original_call_and_whole_program_are_unchanged_except_data_observer(self):
        # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
        new=ast.parse(Path('gpu_engine/transition.py').read_text())
        calls=[n for n in ast.walk(new)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
        original=next(n for n in calls if isinstance(n.func.value,ast.Name)and n.func.value.id=='m'and n.func.attr=='before_move')
        recorded=next(n for n in calls if n.func.attr=='record_before_move_on_tape')
        dump=lambda n:ast.dump(n,include_attributes=False)
        self.assertEqual([dump(x)for x in original.args],[dump(x)for x in recorded.args[6:9]])
        kw={k.arg:k.value for k in recorded.keywords}
        # WH + C-lite: defrost_move (Sacred Fire) reaches before_move only. It changes BeforeMove only for a frozen user, and the
        # lifecycle path refuses a frozen defrost user in the statement right after the call, so the recorder never has to
        # rebuild that case. Infatuation fields have the source-bound refusal proof below.
        # If that refusal is ever removed, defrost_move must be passed to record_before_move_on_tape instead (option B).
        EXEMPT={'defrost_move','infatuated','attract_roll'}
        infatuation_refusal_proof(new,ast.parse(Path('gpu_engine/secondary.py').read_text()))
        for k in original.keywords:
            if k.arg in EXEMPT:continue
            self.assertIn(k.arg,kw,f'before_move keyword {k.arg!r} does not reach record_before_move_on_tape')
            self.assertEqual(dump(k.value),dump(kw[k.arg]),k.arg)
        self.assertEqual([dump(k.value)for k in original.keywords if k.arg=='defrost_move'],[dump(ast.parse('_defrost',mode='eval').body)])
        owner=next(b for n in ast.walk(new) for b in [getattr(n,f,None) for f in ('body','orelse')] if isinstance(b,list)
            and any(isinstance(x,ast.Assign) and x.value is original for x in b))
        at=next(i for i,x in enumerate(owner) if isinstance(x,ast.Assign) and x.value is original)
        self.assertEqual(ast.unparse(owner[at+1]),
            'if _lifecycle_trace and table.effects is not None:\n'
            '    _public_trace = status_callbacks.unknown(_public_trace, eligible & _defrost & (_active(work.status, work, side) == m.FRZ))',
            'the lifecycle refusal of a frozen defrost user (the same _defrost mask) must be the statement right after the original '
            'BeforeMove call; if it is removed, pass defrost_move to record_before_move_on_tape instead (option B)')
        self.assertEqual(recorded.args[1].id,'before');self.assertGreater(recorded.lineno,original.lineno)
        # mechanics, engine_execution_completeness, engine_kernel_frame_risk and engine_retained_encoder_driver: byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.
        for name in ('benchmark_retained_games.py',):
            self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+name]))

    def test_infatuation_omission_proof_detects_removed_or_weakened_refusals(self):
        import copy
        tree=ast.parse(Path('gpu_engine/transition.py').read_text())
        contact=ast.parse(Path('gpu_engine/secondary.py').read_text())
        infatuation_refusal_proof(tree,contact)
        for name in ('_lum_update','initialize','step_with_chance','apply_contact'):
            changed=copy.deepcopy(contact if name=='apply_contact' else tree)
            fn=next(n for n in changed.body if isinstance(n,ast.FunctionDef)and n.name==name)
            target=next(n for n in ast.walk(fn)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)
                and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='unknown'
                and ('ATTRACT' in ast.unparse(n) or 'attracted' in ast.unparse(n) or 'table.infatuation.attract' in ast.unparse(n)))
            target.value.args[-1]=ast.Constant(False)
            with self.subTest(refusal=name),self.assertRaises(AssertionError):
                infatuation_refusal_proof(tree if name=='apply_contact' else changed,changed if name=='apply_contact' else contact)

    def test_original_context_replacement_and_frame_row_commit_retain_tail(self):
        from gpu_engine import engine_execution_completeness as ex,engine_retained_encoder_driver as driver
        args,kwargs=self.inputs();after=d.record_before_move_on_tape(self.t,*args,**kwargs).execution.diagnostics
        context=ex.Context(self.t.execution.ledger,None,self.t.execution.risks,after)
        committed=ex._put_context(self.t,self.t.execution.ledger,context)
        self.assertIs(committed.execution.diagnostics,after)
        mask=j.array([True,False]*4)
        selected=driver._rows(mask,self.t.execution.diagnostics,after)
        self.assertEqual(selected.before_move_count.tolist(),[1,0]*4)
        np.testing.assert_array_equal(selected.before_move_words[::2],after.before_move_words[::2])
        np.testing.assert_array_equal(selected.before_move_words[1::2],self.t.execution.diagnostics.before_move_words[1::2])

    def test_literal_predicates_source_split_unions_and_ledger_noninterference(self):
        plan=census.Plan(max_examples_per_kind=1);streams=np.arange(12001,12001+2048,dtype=np.int32)
        baseline=census.Ledger(plan,streams);observed=census.Ledger(plan,streams)
        rec=dict(eligible=1,status=1,can_act=0,flinched=1,loafing=0,confusion_counter=0,confusion_self_hit=0)
        packet=dict(unhandled_sites=[5],raw_unsupported_kinds=[],engine_failures=[],context_unknown=False,
            literal_diagnostics=dict(before_move=dict(available=True,records=[rec,deepcopy(rec)])))
        active=np.zeros(2048,bool);active[[0,512]]=True;zero=np.zeros(2048,bool);done=zero.copy();turn=np.ones(2048,np.int32)
        with patch.object(census.time,'time',return_value=123.):
            for ledger in (baseline,observed):ledger.mark_capture(active,step=0,packets=lambda row:packet)
            for step in (0,1):
                packet['context_unknown']=bool(step)
                with patch.object(census,'_before_move_predicates',return_value=[]):
                    baseline.observe(step=step,active=active,executed=active,engine_failed=zero,done=done,turns=turn,packet=lambda row:packet,terminal=lambda row:None)
                observed.observe(step=step,active=active,executed=active,engine_failed=zero,done=done,turns=turn,packet=lambda row:packet,terminal=lambda row:None)
        old=baseline.summary();new=observed.summary()
        for field in ('unsupported','unsupported_by_source'):new[field]=[x for x in new[field]if not x['kind'].startswith('before-move-predicate:')]
        self.assertEqual(old,new)
        rows=observed.summary()['unsupported_by_source']
        target=next(x for x in rows if x['role']=='census'and x['kind']=='before-move-predicate:eligible_PAR_and_cannot_act')
        self.assertEqual((target['source'],target['occurrences'],target['games_affected']),('capture-gap',4,1))
        target=next(x for x in rows if x['role']=='timing-only'and x['kind']=='before-move-predicate:eligible_PAR_and_cannot_act')
        self.assertEqual((target['occurrences'],target['timing_rows_affected']),(4,1))
        self.assertEqual(census._before_move_predicates({}),[])


if __name__=='__main__':unittest.main()
