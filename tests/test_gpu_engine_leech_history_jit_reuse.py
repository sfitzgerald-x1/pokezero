"""Fresh pure-helper JAX controls, with no engine construction or stepping."""
import ast
from collections import namedtuple
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine import engine_leech_seed_callbacks as leech
from gpu_engine import public_active, sustain


BASE='6a55ed80787129fdfab289f20f97193e263f60c2'
SOURCE=Path('gpu_engine/engine_retained_encoder_driver.py')
Engine=namedtuple('Engine','active')
Cursor=namedtuple('Cursor','stream_id turn generation physical_active public_slot encounter')
PublicCursor=namedtuple('PublicCursor','stream_id turn source_binding')
Mons=namedtuple('Mons','species_binding active encounter')
Context=namedtuple('Context','active')
Public=namedtuple('Public','cursor mons context')
Core=namedtuple('Core','engine cursor')


def inert_inputs():
    """Literal B8 public bags; no private seed bit can supply their presence."""
    b=8;u=j.arange(8,dtype=j.uint32)+11;z=j.zeros((b,2),j.int32)
    table=public_active.ActiveTables(j.arange(len(public_active.NAMES),dtype=j.int32),
        *(j.zeros(len(public_active.NAMES),j.bool_) for _ in range(4)),
        j.array([public_active.NAMES.index(n) for n in public_active.CLOCK_NAMES],j.int32),u,u)
    installed=sustain.SustainTable(j.array([False,True]),j.array([False,False]),j.array(3,j.int32),j.array([False,False]))
    data=leech.Data(u,u+1,u+2,u+3,u+4,j.array(1,j.int32),installed,j.array(0,j.int32),j.array(2,j.int32))
    streams=j.arange(b,dtype=j.int32)+901;physical=z+j.array([3,4]);slots=z+j.array([1,2]);encounter=z+j.array([17,23])
    cursor=Cursor(streams,j.full(b,4,j.int32),z+7,physical,slots,encounter)
    engine=Engine(physical)
    mons_encounter=j.zeros((b,2,6),j.int32).at[j.arange(b)[:,None],j.arange(2)[None,:],slots].set(encounter)
    mons=Mons(j.broadcast_to(u,(b,8)),slots,mons_encounter)
    active=public_active.initialize(b,table,species_binding=u)._replace(encounter=encounter,turn=cursor.turn)
    active=active._replace(volatiles=active.volatiles.at[:,:,public_active.NAMES.index('leechseed')].set(True))
    pub=Public(PublicCursor(streams,cursor.turn,j.broadcast_to(data.source_binding,(b,8))),mons,Context(active))
    return engine,cursor,(pub,pub),j.ones(b,j.bool_),data,table


def equal_tree(test,actual,expected):
    test.assertEqual(jax.tree.structure(actual),jax.tree.structure(expected))
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):np.testing.assert_array_equal(a,b)


class LeechHistoryJitReuseTests(unittest.TestCase):
    def test_real_helper_fresh_inputs_and_tables_match_original(self):
        args=inert_inputs();context=object();data,table=args[-2:]
        proof=driver._prepare_leech_history_proof(leech.history_from_public,context,data,table)
        def compare(values):
            actual=proof(*values)
            engine,cursor,publics,healthy,current_data,current_table=values
            expected=leech.history_from_public(engine,cursor,publics,healthy=healthy,data=current_data,active_tables=current_table)
            equal_tree(self,actual,expected);return actual
        self.assertTrue(bool(compare(args).seeded.all()))
        engine,cursor,pubs,healthy,data,table=args
        changed=[
            (engine._replace(active=engine.active.at[0,0].set(2)),cursor,pubs,healthy,data,table),
            (engine,cursor._replace(stream_id=cursor.stream_id.at[1].add(1)),pubs,healthy,data,table),
            (engine,cursor._replace(generation=cursor.generation+1),pubs,healthy,data,table),
            (engine,cursor,pubs,healthy.at[2].set(False),data,table),
            (engine,cursor,pubs,healthy,data._replace(source_binding=data.source_binding+1),table),
            (engine,cursor,pubs,healthy,data,table._replace(binding=table.binding+1)),
        ]
        sid=public_active.NAMES.index('leechseed')
        for owner in (0,1):
            public=pubs[owner]
            altered=public._replace(context=Context(public.context.active._replace(
                volatiles=public.context.active.volatiles.at[3,1,sid].set(False))))
            changed.append((engine,cursor,tuple(altered if i==owner else p for i,p in enumerate(pubs)),healthy,data,table))
        for index,values in enumerate(changed):
            with self.subTest(index=index):
                result=compare(values)
                if index!=2:self.assertFalse(bool(result.seeded.all()))
        # Current, coherently changed roots remain usable after refused inputs.
        newer=cursor._replace(stream_id=cursor.stream_id+50,turn=cursor.turn+1)
        fresh=tuple(p._replace(cursor=p.cursor._replace(stream_id=newer.stream_id,turn=newer.turn),
            context=Context(p.context.active._replace(turn=newer.turn))) for p in pubs)
        self.assertTrue(bool(compare((engine,newer,fresh,healthy,data,table)).seeded.all()))
        self.assertIs(driver._checked_leech_history_proof(proof,leech.history_from_public,context,data,table),proof)

    def test_current_gates_and_untrusted_cache_use_current_eager_fallback(self):
        args=inert_inputs();engine,cursor,pubs,healthy,data,table=args
        context=SimpleNamespace(data=SimpleNamespace(leech_seed=data));catalog=SimpleNamespace(active=SimpleNamespace(device=table))
        proof=driver._prepare_leech_history_proof(leech.history_from_public,context,data,table)
        runtime=dict(direct_hit_context=context,direct_hit_mode='kernel-event',catalog=catalog,leech_history_proof=proof)
        core=Core(engine,cursor);equal_tree(self,driver._leech_history(runtime,core,pubs,healthy),proof(*args))
        for current in (dict(runtime,direct_hit_mode='diagnostic'),dict(runtime,direct_hit_context=None),
                dict(runtime,direct_hit_context=SimpleNamespace(data=SimpleNamespace(leech_seed=None)))):
            current['leech_history_proof']=lambda *a:(_ for _ in ()).throw(AssertionError('disabled callable invoked'))
            self.assertIsNone(driver._leech_history(current,core,pubs,healthy))
        foreign_context=SimpleNamespace(data=SimpleNamespace(leech_seed=data))
        foreign_data=data._replace(binding=data.binding+1);foreign_table=table._replace(binding=table.binding+1)
        def never(*a,**k):raise AssertionError('untrusted cached callable invoked')
        foreign_proof=driver._prepare_leech_history_proof(never,foreign_context,data,table)
        class UnhashableCallable:
            __hash__=None
            def __call__(self,*a):return never(*a)
        # A known callable with stale origin must not run either; its current
        # holder equality controls code selection, not original admission.
        driver._LEECH_HISTORY_ORIGINS[never]=(leech.history_from_public,context,data,table)
        for change in (dict(leech_history_proof=None),dict(leech_history_proof=lambda *a:never(*a)),
                dict(leech_history_proof=UnhashableCallable()),dict(leech_history_proof=[]),
                dict(leech_history_proof=foreign_proof),dict(direct_hit_context=foreign_context),
                dict(catalog=SimpleNamespace(active=SimpleNamespace(device=foreign_table)))):
            current=dict(runtime,**change)
            if 'leech_history_proof' not in change:current['leech_history_proof']=never
            with self.subTest(change=tuple(change)):
                expected=leech.history_from_public(engine,cursor,pubs,healthy=healthy,
                    data=current['direct_hit_context'].data.leech_seed,active_tables=current['catalog'].active.device)
                equal_tree(self,driver._leech_history(current,core,pubs,healthy),expected)
        missing=dict(runtime);missing.pop('leech_history_proof')
        equal_tree(self,driver._leech_history(missing,core,pubs,healthy),proof(*args))
        with patch.object(context.data,'leech_seed',foreign_data):
            expected=leech.history_from_public(engine,cursor,pubs,healthy=healthy,data=foreign_data,active_tables=table)
            equal_tree(self,driver._leech_history(dict(runtime,leech_history_proof=never),core,pubs,healthy),expected)
        sentinel=object()
        with patch.object(leech,'history_from_public',return_value=sentinel) as current:
            self.assertIs(driver._leech_history(runtime,core,pubs,healthy),sentinel)
            current.assert_called_once_with(engine,cursor,pubs,healthy=healthy,data=data,active_tables=table)
        with patch.object(leech,'history_from_public',side_effect=ValueError('original current helper error')):
            with self.assertRaisesRegex(ValueError,'original current helper error'):driver._leech_history(runtime,core,pubs,healthy)
        equal_tree(self,driver._leech_history(runtime,core,pubs,healthy),proof(*args))

    def test_once_prepared_dynamic_call_scope_and_all_original_boundaries(self):
        old=ast.parse(subprocess.check_output(['git','show',BASE+':'+str(SOURCE)],text=True));current=ast.parse(subprocess.check_output(['git','show','7e958020:'+str(SOURCE)],text=True))
        actual=ast.parse(SOURCE.read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:'+str(SOURCE)],text=True));self.assertEqual(ast.dump(actual),ast.dump(integrated))
        def functions(t):return {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
        before,after=functions(old),functions(current)
        allowed={'prepare','_leech_history','_prepare_leech_history_proof','_checked_leech_history_proof'}
        for name,node in before.items():
            if name=='_frame':
                reference=functions(ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/engine_retained_encoder_driver.py'],text=True)))
                self.assertEqual(ast.dump(reference[name]),ast.dump(functions(actual)[name]),name)
            elif name not in allowed:self.assertEqual(ast.dump(node),ast.dump(after[name]),name)
        preparation=deepcopy(after['prepare'])
        def added(node):
            return (isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='leech_history_proof' for n in node.targets)
                or isinstance(node,ast.If) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_prepare_leech_history_proof' for n in ast.walk(node)))
        preparation.body=[n for n in preparation.body if not added(n)]
        for node in ast.walk(preparation):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':node.keywords=[k for k in node.keywords if k.arg!='leech_history_proof']
        self.assertEqual(ast.dump(preparation),ast.dump(before['prepare']))
        self.assertEqual(ast.dump(after['_leech_history'].body[1]),ast.dump(before['_leech_history'].body[1]))
        call=after['_leech_history'].body[-1].value
        self.assertEqual([ast.unparse(n) for n in call.args],['core.engine','core.cursor','publics','healthy','data','active_tables'])
        with patch.object(driver.jax,'jit',side_effect=lambda fn:fn) as jit:
            args=inert_inputs();context=object();data,table=args[-2:]
            proof=driver._prepare_leech_history_proof(leech.history_from_public,context,data,table)
            self.assertEqual(jit.call_count,1)
            for _ in range(3):equal_tree(self,proof(*args),leech.history_from_public(*args[:3],healthy=args[3],data=data,active_tables=table))
            self.assertEqual(jit.call_count,1)
        # No helper, physics, catalog or admission code is changed. transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components;
        # state.py: byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.
        for name in ('engine_leech_seed_callbacks.py','v4_encoder.py'):
            self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+name]))


if __name__=='__main__':unittest.main()
