"""Literal status residual receipts and effective Gen3 source controls only."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_status_residual_callbacks as c, mechanics as m
from gpu_engine import engine_lifecycle_callbacks as life, engine_public_events as public
import test_gpu_engine_engine_own_facts as fixtures
import test_gpu_engine_engine_item_callbacks as tape_fixtures


SOURCE_JS=r'''
'use strict';
const root=process.argv[1],species=process.argv[2];
const {BattleStream}=require(root+'/dist/sim/battle-stream');
const {Battle}=require(root+'/dist/sim/battle');
const {Dex}=require(root+'/dist/sim/dex');
(async()=>{
 const stream=new BattleStream({noCatch:true});
 await stream.write('>start '+JSON.stringify({formatid:'gen3customgame',seed:[1,2,3,4]}));
 await stream.write('>player p1 '+JSON.stringify({name:'Literal status component',team:[
  {species,moves:['splash'],level:80}]}));
 const battle=stream.battle,pokemon=battle.p1.pokemon[0],dex=battle.dex;
 if(battle.started || battle.turn!==0 || battle.p2)throw Error('no game/choices permitted');
 battle.reportPercentages=true;battle.reportExactHP=false;
 pokemon.isActive=true;pokemon.maxhp=101;pokemon.baseMaxhp=101;
 // Numerical write is a literal method stub; source lexical damage dispatcher
 // and getHealth are ORIGINAL. This is not a simulator outcome or hook proof.
 pokemon.damage=function(amount){const d=Math.min(this.hp,Math.trunc(amount));this.hp-=d;return d;};
 const rows=[];
 for(const reason of ['brn','psn','tox'])for(const status of ['', 'par','brn','psn','tox','slp','frz']){
  pokemon.hp=95;pokemon.status=status;
  const lines=[];const effect=dex.conditions.get(reason);
  const stub={gen:3,dex,event:{target:pokemon},effect,effectState:{stage:0},
   trunc:battle.trunc.bind(battle),clampIntRange:battle.clampIntRange.bind(battle),
   damage:Battle.prototype.damage,spreadDamage:Battle.prototype.spreadDamage,
   runEvent(name,...args){if(name!=='Damage')throw Error('unexpected hook '+name);return args[3];},
   add(...args){const health=args[2]();lines.push({kind:args[0],tags:args.slice(3),health});},
   debug(){throw Error('unexpected failed damage');}};
  effect.onResidual.call(stub,pokemon);
  rows.push({reason,status,order:effect.onResidualOrder,suborder:effect.onResidualSubOrder,
   stage:stub.effectState.stage,hp:pokemon.hp,lines});
 }
 const provenance={revision:require('child_process').execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),
  burnMatchesGen6:dex.conditions.get('brn').onResidual===Dex.mod('gen6').conditions.get('brn').onResidual,
  poisonMatchesBase:dex.conditions.get('psn').onResidual===Dex.conditions.get('psn').onResidual,
  toxicMatchesBase:dex.conditions.get('tox').onResidual===Dex.conditions.get('tox').onResidual,
  functions:Object.fromEntries(['brn','psn','tox'].map(x=>[x,dex.conditions.get(x).onResidual.toString()])),
  started:battle.started,turn:battle.turn};
 console.log(JSON.stringify({provenance,rows}));stream.destroy();
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
'''


class StatusResidualCallbacksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceOwnFactsTests.setUpClass();cls.f=fixtures.DeviceOwnFactsTests
        cls.metadata=deepcopy(cls.f.metadata)
        cls.compiled=c.compile_status_residual(cls.metadata,engine_catalog=cls.f.lifecycle)
        cls.data=c.check_compiled(cls.compiled)

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def tape(self,b=1,capacity=16,prior=0,legacy=False):
        state=tape_fixtures.state(b)._replace(active=j.broadcast_to(j.array([2,4],j.int32),(b,2)))
        old=tape_fixtures.tape(state,capacity,legacy=legacy)
        if legacy:return old
        cursor=old.cursor._replace(engine_binding=j.broadcast_to(self.data.engine_binding,(b,8)))
        return life._tape(cursor,self.f.lifecycle,capacity)._replace(refusal=j.full(b,prior,j.int32))

    def record(self,tape,*,side=0,physical=2,old=95,new=83,maximum=101,status=m.BRN,before=0,after=0,eligible=True,data=None):
        return c.record_status_residual_device(tape,side,physical,old,new,maximum,status,before,after,
            self.data if data is None else data,eligible=eligible)

    def test_actual_numerical_writes_and_literal_counter_cap(self):
        statuses=j.array([m.BRN,m.PSN,m.TOX,m.TOX],j.int32)
        old=j.full(4,101,j.int32);maximum=j.full(4,101,j.int32);counter=j.array([0,0,14,15],j.int32)
        actual=m.major_status_residual(old,maximum,statuses,counter,active=j.ones(4,bool))
        result=self.record(self.tape(4,prior=128),old=old,new=actual.hp,maximum=maximum,status=statuses,
            before=counter,after=actual.counter)
        self.assertEqual(result.count.tolist(),[1]*4)
        self.assertEqual(result.words[:,0,0].tolist(),[20]*4)
        self.assertEqual(result.words[:,0,1:3].tolist(),[[0,0],[0,0],[14,15],[15,15]])
        self.assertEqual(result.words[:,0,5].tolist(),old.tolist())
        self.assertEqual(result.words[:,0,6].tolist(),actual.hp.tolist())
        self.assertEqual(result.words[:,0,9].tolist(),[c.BURN,c.POISON,c.TOXIC,c.TOXIC])
        self.assertEqual(result.refusal.tolist(),[128|256]*4)
        self.assertFalse(bool((result.words[:,:,0]==public.DIRECT_FAINT).any()))

    def test_literal_int32_above_float_precision_and_not_a_damage_or_counter_oracle(self):
        result=self.record(self.tape(),old=16777219,new=16777218,maximum=16777221,status=m.TOX,before=3,after=8)
        self.assertEqual(result.words[0,0].tolist(),[20,3,8,0,2,16777219,16777218,16777221,m.TOX,c.TOXIC])
        self.assertEqual(result.refusal.tolist(),[256])
        # Diagnostic lethal HP loss keeps UNKNOWN and never implies a faint.
        lethal=self.record(self.tape(),old=1,new=0,maximum=101,status=m.PSN)
        self.assertEqual(lethal.count.tolist(),[1]);self.assertEqual(int(lethal.words[0,0,6]),0)
        self.assertEqual(lethal.refusal.tolist(),[256])
        self.assertFalse(bool((lethal.words[:,:,0]==public.DIRECT_FAINT).any()))

    def test_noop_dead_unsupported_status_invalid_type_and_counter_are_unknown_only(self):
        for args in (dict(old=0,new=0),dict(old=95,new=95),dict(new=96),dict(new=-1),dict(maximum=90),
                     dict(status=m.NONE),dict(status=m.PAR),dict(status=m.SLP),dict(status=m.FRZ),dict(status=7),
                     dict(before=-1),dict(after=16),dict(before=4,after=3,status=m.TOX),dict(before=0,after=1),
                     dict(old=True),dict(new=j.array([83.5],j.float32)),dict(old=np.array([2**32+1],np.uint64)),
                     dict(before=j.ones(2,j.int32))):
            with self.subTest(args=str(args)):
                result=self.record(self.tape(prior=128),**args)
                self.assertEqual(result.count.tolist(),[0]);self.assertEqual(result.refusal.tolist(),[128|256])
                self.assertFalse(bool(result.words.any()))
        old=self.tape(prior=128)
        self.same(self.record(old,old=None,new=None,before=None,after=None,eligible=False),old)

    def test_current_identity_stamps_b8_inactive_and_hot_jit(self):
        tape=self.tape(8,prior=128)
        tape=tape._replace(stream_id=tape.stream_id.at[2].set(999),cursor=tape.cursor._replace(
            engine_binding=tape.cursor.engine_binding.at[3,1].add(1),initialized=tape.cursor.initialized.at[4].set(False),
            phase=tape.cursor.phase.at[5].set(1)))
        side=j.arange(8,dtype=j.int32)%2;physical=j.where(side==0,2,4).at[1].set(0)
        eligible=j.array([True]*6+[False,True])
        call=lambda value:self.record(value,side=side,physical=physical,eligible=eligible)
        expected=call(tape);fast=jax.jit(call)
        for _ in range(2):self.same(fast(tape),expected)
        self.assertEqual(expected.count.tolist(),[1,0,0,0,0,0,0,1])
        self.assertEqual(expected.refusal.tolist(),[384]*6+[128,384])
        self.assertEqual(expected.event_turn[:,0].tolist(),[int(tape.cursor.turn[0])]+[0]*6+[int(tape.cursor.turn[7])])
        self.same(expected.event_generation[0,0],tape.cursor.generation[0])
        for name in tape._fields:
            if name in ('engine_binding','source_binding','move_binding','producer_binding'):
                self.same(getattr(expected,name),getattr(tape,name))
            elif name=='cursor':self.same(jax.tree.map(lambda x:x[6],expected.cursor),jax.tree.map(lambda x:x[6],tape.cursor))
            elif name=='execution':
                self.assertIsNone(tape.execution);self.assertIsNone(expected.execution)
            else:self.same(getattr(expected,name)[6],getattr(tape,name)[6])
        text=str(jax.make_jaxpr(call)(tape))
        for callback in ('io_callback','pure_callback','debug_callback'):self.assertNotIn(callback,text)

    def test_global8word_namespace_and_canonical_reason_not_a_B8_axis(self):
        tape=self.tape(8);mask=j.array([True,False,True,False,True,False,True,False])
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            bad=tape._replace(**{name:getattr(tape,name).at[7].add(j.uint32(1))})
            result=self.record(bad,eligible=mask)
            self.assertEqual(result.count.tolist(),[0]*8)
            self.assertEqual(result.refusal.tolist(),[256 if x else 0 for x in mask.tolist()])
        for data in (self.data._replace(reason_by_status=self.data.reason_by_status.at[m.TOX].set(c.POISON)),
                     self.data._replace(damage_tag_by_reason=self.data.damage_tag_by_reason.at[c.TOXIC].set(1))):
            result=self.record(tape,data=data)
            self.assertEqual(result.count.tolist(),[0]*8);self.assertEqual(result.refusal.tolist(),[256]*8)
        with self.assertRaises(ValueError):self.record(tape,data=self.data._replace(binding=self.data.binding.astype(j.int32)))

    def test_overflow_foreign_bits_legacy_and_no_faint_promotion(self):
        tape=self.tape(capacity=1,prior=1024);first=self.record(tape);result=self.record(first)
        self.assertEqual(result.count.tolist(),[1]);self.assertTrue(int(result.refusal[0])&public.OVERFLOW)
        self.assertTrue(int(result.refusal[0])&1024);self.assertTrue(int(result.refusal[0])&256)
        self.same(result.words,first.words)
        legacy=self.tape(legacy=True);self.same(self.record(legacy),legacy)
        self.assertFalse(hasattr(c,'register_public'));self.assertFalse(hasattr(c.Data,'full_encoder_admitted'))

    def test_original_metadata_copy_and_every_checked_host_tracing_shape(self):
        with self.assertRaises(ValueError):c.check_compiled(replace(self.compiled))
        with self.assertRaises(ValueError):c.check_compiled(999)
        original=self.metadata['vocabulary']['items'][1]
        self.metadata['vocabulary']['items'][1]='changed'
        try:
            with self.assertRaises(ValueError):c.check_compiled(self.compiled)
        finally:self.metadata['vocabulary']['items'][1]=original
        tape=self.tape()
        host=lambda value:c.record_status_residual(value,0,2,95,83,101,m.BRN,0,0,compiled=self.compiled,eligible=True)
        for fn,args in ((jax.jit(host),(tape,)),(jax.jit(lambda _:host(tape)),(j.array(0,j.int32),)),
                        (jax.jit(lambda:c.check_compiled(self.compiled)),()),
                        (jax.jit(lambda:c.compile_status_residual(self.metadata,engine_catalog=self.f.lifecycle)),())):
            for _ in range(2):
                with self.assertRaisesRegex(ValueError,'host boundary cannot be traced'):fn(*args)
        self.same(host(tape),self.record(tape))

    def test_effective_original_gen3_damage_tags_and_gethealth_all_seven_statuses(self):
        root=Path(self.f.f.root)
        result=subprocess.run([str(self.f.f.node),'-e',SOURCE_JS,str(root),self.f.catalog.species.names[1]],
            text=True,capture_output=True,check=False)
        self.assertEqual(result.returncode,0,msg=result.stderr)
        report=json.loads(result.stdout);self.source_report=report
        p=report['provenance']
        self.assertEqual(p['revision'],public.SHOWDOWN)
        self.assertTrue(p['burnMatchesGen6']);self.assertTrue(p['poisonMatchesBase']);self.assertTrue(p['toxicMatchesBase'])
        self.assertFalse(p['started']);self.assertEqual(p['turn'],0)
        self.assertIn('baseMaxhp / 8',p['functions']['brn'])
        self.assertEqual(len(report['rows']),21)
        for row in report['rows']:
            self.assertEqual((row['order'],row['suborder']),(10,6))
            self.assertEqual(len(row['lines']),1);line=row['lines'][0]
            self.assertEqual(line['kind'],'-damage')
            self.assertEqual(line['tags'],['[from] brn' if row['reason']=='brn' else '[from] psn'])
            suffix=(' '+row['status']) if row['status'] else ''
            self.assertEqual(line['health']['secret'],f"{row['hp']}/101"+suffix)
            self.assertEqual(line['health']['shared'],f"{min(99,(100*row['hp']+100)//101)}/100"+suffix)
            if row['reason']=='tox':self.assertEqual(row['stage'],1)


if __name__=='__main__':unittest.main()
