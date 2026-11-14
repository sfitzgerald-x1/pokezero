"""Outgoing PAR source port; literal original callbacks, no engine turn/game."""
import ast, hashlib, json, subprocess, unittest
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import jax
import jax.numpy as j
import numpy as np
from test_gpu_engine_baton_toxic_status import (
    BatonToxicStatusTests as OriginalFixture, cb, direct, ex, risk, life, public,
    request, ordering, full, v4, public_wire, state, source_provenance, baton,
    entry, private_requests, showdown_root)
BASE = '3a1655ad73da6f61fa5cbe5961c40600587162dc'

class BatonParalysisGuards(unittest.TestCase):
    def test_outgoing_only_status_counter_and_exact_preservation(self):
        b=8;r=j.arange(b);side=r%2;old=j.zeros(b,j.int32)
        status=j.array([0,1,1,4,2,3,5,6],j.int32)
        counter=j.array([0,0,1,2,0,0,0,0],j.int32)
        s=state.empty_state(b)._replace(hp=j.full((b,2,6),100,j.int32),
            ability=j.ones((b,2,6),j.int32),base_ability=j.ones((b,2,6),j.int32))
        s=s._replace(status=s.status.at[r,side,old].set(status),
            status_counter=s.status_counter.at[r,side,old].set(counter))
        d=SimpleNamespace(quiet_abilities=j.array([False,True]),quiet_items=j.array([True]))
        self.assertEqual(cb._quiet(s,side,old,d).tolist(),[True,False,False,False,False,False,False,False])
        self.assertEqual(cb._quiet(s,side,old,d,outgoing_toxic=True).tolist(),[True,False,False,True,False,False,False,False])
        self.assertEqual(cb._quiet(s,side,old,d,outgoing_toxic=True,outgoing_paralysis=True).tolist(),[True,True,False,True,False,False,False,False])
        before=cb.snapshot(s)
        after=cb.departed_private(cb.copied_private(before,side,old,old+3),side,old)
        np.testing.assert_array_equal(after.status,before.status)
        self.assertEqual(after.status_counter[r,side,old].tolist(),[0,0,1,0,0,0,0,0])
        self.assertEqual(after.status[r,side,old+3].tolist(),[0]*b)
        mutated=s._replace(status=s.status.at[1,1,0].set(0))
        self.assertEqual(cb.equal(before,cb.snapshot(mutated)).tolist(),[True,False,True,True,True,True,True,True])
        # The new permission occurs ONLY at the two original outgoing calls.
        tree=ast.parse(Path(cb.__file__).read_text())
        found=[]
        for fn in (n for n in tree.body if isinstance(n,ast.FunctionDef)):
            for call in (n for n in ast.walk(fn)if isinstance(n,ast.Call)):
                if any(k.arg=='outgoing_paralysis'for k in call.keywords):
                    found.append((fn.name,ast.unparse(call.args[2])))
        self.assertEqual(found,[('begin_start','physical'),('before_copy','old')])

    def test_original_callbacks_preserve_par_and_numeric_files_exact(self):
        script=r"""
const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,H=x=>require('crypto').createHash('sha256').update(''+x).digest('hex');
const clear=P.prototype.clearVolatile,copy=P.prototype.copyVolatileFrom,par=d.conditions.get('par');
const make=(status,hp)=>({status,statusState:{id:status,turn:7},hp,maxhp:100,side:{},volatiles:{},boosts:{atk:0,def:0,spa:2,spd:0,spe:0,accuracy:0,evasion:0},baseMoveSlots:[],moveSlots:[],baseAbility:'innerfocus',ability:'innerfocus',baseHpType:'Dark',baseHpPower:70,abilityState:{},itemState:{},baseSpecies:{name:'Original'},species:{name:'Original'},battle:{gen:3},setSpecies(x){this.species=x},hasAbility:()=>false,clearVolatile:clear});
const s=make('par',65),t=make('',100),lines=[];t.battle={gen:3,singleEvent(){},add(){throw Error('unexpected public callback')}};
// The actual PAR beforeMove handler is tested with an inert literal decision,
// not a random generator. Only its passing branch reaches Baton.
par.onBeforeMove.call({randomChance:()=>false,add:()=>{throw Error('cant on passing branch')}},s);
d.moves.get('batonpass').onHit.call({canSwitch:()=>true},s);
d.moves.get('batonpass').self.onHit.call({},s);
copy.call(t,s,'copyvolatile');clear.call(s);
console.log(JSON.stringify({clear:H(clear),copy:H(copy),par:H(par.onBeforeMove),hooks:['onBeforeSwitchOut','onSwitchOut','onSwitchIn'].map(n=>typeof par[n]),outgoing:[s.status,s.statusState,s.hp],incoming:[t.status,t.statusState,t.hp],boost:t.boosts.spa}));
"""
        v=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
        self.assertEqual(v['clear'],cb.CLEAR_VOLATILE_SHA256)
        self.assertEqual(v['copy'],cb.COPY_VOLATILE_SHA256)
        self.assertEqual(v['par'],cb.PAR_BEFORE_MOVE_SHA256)
        self.assertEqual(v['hooks'],['undefined']*3)
        self.assertEqual(v['outgoing'],['par',dict(id='par',turn=7),65])
        self.assertEqual(v['incoming'],['',dict(id='',turn=7),100])
        self.assertEqual(v['boost'],2)
        # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
        # entry.py, baton.py and state.py are byte-certified against the reviewed SOURCE_PINS by transition_observer_wiring.
        for n in ('mechanics.py','engine_retained_encoder_driver.py','engine_kernel_frame_risk.py','engine_full_v4_event_device.py','engine_lifecycle_event_device.py'):
            self.assertEqual(Path('gpu_engine',n).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+n]),n)
        from gpu_engine import engine_source_frame_witness as witness
        witness._source()
        self.assertEqual(len(witness.SOURCE_PINS),89)   # 88, and residual_order.py since §6
        # Rollback/claimed snapshot consumers remain exactly their original AST. suspension_shape gained fidelity R's
        # saved effect block (reviewed; the file is byte-certified against SOURCE_PINS above).
        old=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/engine_baton_pass_callbacks.py'],text=True));new=ast.parse(Path(cb.__file__).read_text())
        changed={'compile_data','_quiet','begin_start','before_copy','suspension_shape'}
        dump=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name not in changed}
        self.assertEqual(dump(old),dump(new))

class BatonParalysisComponent(unittest.TestCase):
    setUpClass=classmethod(OriginalFixture.setUpClass.__func__)
    new_tape=OriginalFixture.new_tape
    produced=OriginalFixture.produced
    prepare_tape=OriginalFixture.prepare_tape
    start_component=OriginalFixture.start_component

    def initial(self):
        b = 2
        z = j.zeros(b, j.int32)
        yes = j.ones(b, bool)
        streams = j.array([199001, 199002], j.int32)
        indices = np.tile([[0, 1]], (b, 1))
        species = np.ones((2, 6), np.int32)
        species[:, 3] = 2
        moves = np.broadcast_to(np.array([self.metadata['vocabulary']['moves'].index(n) for n in self.names], np.int32), (2, 6, 4)).copy()
        pp = np.broadcast_to(np.array([64, 24, 32, 48], np.int32), (2, 6, 4)).copy()
        types = np.asarray(self.rules.device.species_types)[species]
        arrays = dict(species=species, level=np.full((2, 6), 80, np.int32), types=types, stats=np.full((2, 6, 6), 100, np.int32), moves=moves, pp=pp, item=np.zeros((2, 6), np.int32), ability=np.full((2, 6), self.ability, np.int32), hidden_power_type=np.full((2, 6), 7, np.int32), hidden_power_power=np.full((2, 6), 70, np.int32), gender=np.ones((2, 6), np.int32), nature=np.zeros((2, 6), np.int32), evs=np.zeros((2, 6, 6), np.int32), metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(), np.uint8))
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'unit.npz'
            np.savez_compressed(p, **arrays)
            pool = source_provenance.load_constructor_pool(p)
        s = state.empty_state(b)._replace(species=j.asarray(species[indices]), level=j.asarray(arrays['level'][indices]), types=j.asarray(types[indices]), base_types=j.asarray(types[indices]), stats=j.asarray(arrays['stats'][indices]), max_hp=j.full((b, 2, 6), 100, j.int32), hp=j.full((b, 2, 6), 100, j.int32), moves=j.asarray(moves[indices]), pp=j.asarray(pp[indices]), ability=j.full((b, 2, 6), self.ability, j.int32), base_ability=j.full((b, 2, 6), self.ability, j.int32), gender=j.ones((b, 2, 6), j.int32), hidden_power_type=j.full((b, 2, 6), 7, j.int32), hidden_power_power=j.full((b, 2, 6), 70, j.int32), entry_pending=j.zeros((b, 2), bool), done=~yes)
        r = j.arange(b)
        outgoing = r % 2
        s = s._replace(status=s.status.at[r, outgoing, 0].set(cb.PARALYSIS_STATUS),
                       status_counter=s.status_counter.at[r, outgoing, 0].set(0),
                       hp=s.hp.at[r, outgoing, 0].set(65))
        regs = tuple((request._registration(s, pool, indices, streams, z + side) for side in (0, 1)))
        orders = tuple((ordering.initialize_order(streams, z + side, z + 6, engine_binding=j.broadcast_to(self.rules.device.engine_binding, (b, 8))) for side in (0, 1)))
        cursor = life.register(streams, **self.kw)
        initial = life._tape(cursor, self.producer, 64)
        for side in (0, 1):
            initial = life._reveal(initial, s, z + side, z - 1, z, eligible=yes, replacement=~yes, baton=~yes, initial=True)
        initial = initial._replace(cursor=initial.cursor._replace(initialized=yes))
        initial = life._turn(initial, yes)
        frame = life.LifecycleResult(s, initial.cursor, initial, life.snapshot(cursor), life.snapshot(initial.cursor), ~yes, z)
        owners = []
        for owner in (0, 1):
            stream = full.register(streams, z + owner, compiled=self.runtime['public'])
            pub = v4.initialize(self.catalog, stream_ids=streams, owner_sides=z + owner)
            out = self.fold(stream, frame, pub)
            self.assertEqual(out.projection.error.tolist(), [0, 0])
            owners.append((out.projection.state, out.public))
        return SimpleNamespace(s=s, cursor=initial.cursor, owners=owners, regs=regs, orders=orders)

    def drain(self,x,t,s,tag):
        z=j.zeros(2,j.int32);yes=j.ones(2,bool)
        frame=life.LifecycleResult(s,t.cursor,t,life.snapshot(x.cursor),life.snapshot(t.cursor),~yes,z)
        new=[];ledgers=[];actual=[]
        for owner,(stream,pub) in enumerate(x.owners):
            ledger,out=self.consume(t.execution.ledger,frame,stream,pub)
            actual.append(dict(owner=owner,error=out.projection.error.tolist(),words=out.projection.frames.words.tolist(),valid=out.projection.frames.valid.tolist(),stage=ledger.stage.tolist(),refusal=ledger.refusal.tolist()))
            Path('/tmp/pokezero-baton-paralysis-'+tag+'-actual.json').write_text(json.dumps(actual)+'\n')
            self.assertEqual(out.projection.error.tolist(),[0,0],actual[-1])
            new.append((out.projection.state,out.public));ledgers.append(ledger)
        return frame,new,ledgers

    def test_par_two_boundaries_both_original_full_and_same_private_facts(self):
        c=self.start_component();x=c.x;b=2;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool)
        self.assertEqual(c.pending.before.status[r,c.side,0].tolist(),[1,1])
        self.assertEqual(c.pending.before.hp[r,c.side,0].tolist(),[65,65])
        # Current canAct/caller and complete current private facts stay required.
        for label,kw in [('cannot-act',dict(acts=~yes)),('called',dict(caller=z+self.move+1))]:
            args=dict(acts=yes,eligible=yes);args.update({k:v for k,v in kw.items()if k!='caller'})
            _,p=cb.begin_start(c.before,x.s,c.side,z+self.move,kw.get('caller',z+self.move),self.runtime['direct_hit_context'],**args)
            self.assertEqual(p.valid.tolist(),[False,False],label)
        corrupted=x.s._replace(status_counter=x.s.status_counter.at[0,0,0].set(1))
        _,p=cb.begin_start(c.before,corrupted,c.side,z+self.move,z+self.move,self.runtime['direct_hit_context'],acts=yes,eligible=yes)
        self.assertEqual(p.valid.tolist(),[False,True])
        frame,owners,ledgers=self.drain(x,c.t,c.s,'start')
        requests=self.produced(c.s,x.orders,x.regs,c.t.cursor)
        history=cb.mature(c.t.execution.baton_transaction.history,frame,tuple(ledgers),tuple(p for _,p in owners),requests)
        self.assertEqual(history.ready.tolist(),[True,True])
        x.s=c.s;x.cursor=c.t.cursor;x.owners=owners
        ledger=ex.join_consumed_owners(ledgers[0],ledgers[1],tables=self.runtime['execution_compiled'].device)[0]
        actions=j.zeros((b,2),j.int32).at[r,c.side].set(7)
        t=self.prepare_tape(x,self.new_tape(x,ledger),history,requests,actions)
        self.assertEqual(t.execution.baton_transaction.request_ok.tolist(),[True,True])
        t=life.begin(t,x.s,yes,~yes);target=z+3
        # Incoming PAR, changed outgoing counter/status/HP, stale history and
        # absent current source registration do not inherit the new permission.
        for name,field,index,value in [('incomingPAR','status',(0,0,3),1),('counter','status_counter',(0,0,0),1),('status','status',(0,0,0),0),('hp','hp',(0,0,0),64)]:
            bad=x.s._replace(**{field:getattr(x.s,field).at[index].set(value)})
            _,p=cb.before_copy(t,bad,c.side,target,self.runtime['direct_hit_context'],eligible=yes)
            self.assertEqual(p.valid.tolist(),[False,True],name)
        stale=history._replace(frame=history.frame-1)
        self.assertFalse(bool(self.prepare_tape(x,self.new_tape(x,ledger),stale,requests,actions).execution.baton_transaction.request_ok.any()))
        t,p=cb.before_copy(t,x.s,c.side,target,self.runtime['direct_hit_context'],eligible=yes)
        self.assertEqual(p.valid.tolist(),[True,True])
        after=baton.copy_to_recipient(x.s,c.side,target,eligible=yes)
        t=cb.after_copy(t,p,after)
        after=entry.depart(after,c.side,self.table.entry,eligible=yes)
        after=after._replace(active=after.active.at[r,c.side].set(target),baton_pass=after.baton_pass.at[r,c.side].set(False),forced_switch=after.forced_switch.at[r,c.side].set(False))
        self.assertEqual(after.status[r,c.side,0].tolist(),[1,1])
        self.assertEqual(after.status_counter[r,c.side,0].tolist(),[0,0])
        self.assertEqual(after.hp[r,c.side,0].tolist(),[65,65])
        self.assertEqual(after.status[r,c.side,target].tolist(),[0,0])
        self.assertEqual(cb.reveal_matches(t,after,c.side,z,target).tolist(),[True,True])
        forged=after._replace(status=after.status.at[0,0,0].set(0))
        self.assertEqual(cb.reveal_matches(t,forged,c.side,z,target).tolist(),[False,True])
        t=life.installed(t,after,c.side,z,target,eligible=yes,replacement=yes,baton=yes)
        t=cb.finish_install(t,after,c.side,eligible=yes)
        self.assertEqual(cb.owned(t,t.count-1,cb.INSTALL).tolist(),[True,True])
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            bad=t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))})
            self.assertEqual(cb.owned(bad,t.count-1,cb.INSTALL).tolist(),[False,False],name)
            with self.assertRaises(ValueError):
                cb.owned(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),t.count-1,cb.INSTALL)
        rec=t.execution.baton_transaction.record
        badrec=rec._replace(after=rec.after._replace(status=rec.after.status.at[0,0,0].set(0)))
        bad=t._replace(execution=t.execution._replace(baton_transaction=t.execution.baton_transaction._replace(record=badrec)))
        self.assertEqual(cb.owned(bad,t.count-1,cb.INSTALL).tolist(),[False,True])
        install_frame,new,closed=self.drain(x,t,after,'install')
        # Compare exact original lexical event words including component sequence.
        captures=[]
        for owner in (0,1):
            for row,side in enumerate(c.side.tolist()):
                initial=self.catalog.species.names[1];incoming=self.catalog.species.names[2]
                ids={s:f'p{s+1}a: '+initial for s in (0,1)}
                prefix=[f'|switch|{ids[s]}|{initial}, L80, M|'+('65/100 par'if s==side else'100/100')for s in (0,1)]
                prefix+=['|turn|1','|','|t:|1',f'|move|{ids[side]}|Tail Glow|{ids[side]}',f'|-boost|{ids[side]}|spa|2']
                wire,_=public_wire.project_lines(public_wire.WireState(),prefix,**v4.wire_kwargs(self.catalog))
                wire,start=public_wire.project_lines(wire,['|','|t:|1',f'|move|{ids[side]}|Baton Pass|{ids[side]}'],**v4.wire_kwargs(self.catalog))
                _,install=public_wire.project_lines(wire,['|','|t:|1',f'|switch|p{side+1}a: {incoming}|{incoming}, L80, M|100/100'],**v4.wire_kwargs(self.catalog))
                for tag,packets in [('start',start),('install',install)]:
                    want=np.array([e.words()for packet in packets for e in packet.events],np.int32)
                    doc=json.loads(Path('/tmp/pokezero-baton-paralysis-'+tag+'-actual.json').read_text())[owner]
                    words=np.array(doc['words'],np.int32);valid=np.array(doc['valid']);actual=words[:,row][valid[:,row]]
                    captures.append(dict(owner=owner,row=row,tag=tag,actual=actual.tolist(),expected=want.tolist(),stage=doc['stage'][row],refusal=doc['refusal'][row]))
                    Path('/tmp/pokezero-baton-paralysis-original-both-full-compact.json').write_text(json.dumps(captures)+'\n')
                    np.testing.assert_array_equal(actual,want)
                    self.assertEqual(doc['refusal'][row],0)
        orders=tuple(life.apply_installations(o,install_frame)for o in x.orders)
        current=self.produced(after,orders,x.regs,t.cursor)
        for q in current:self.assertEqual(q.actions.admitted.tolist(),[True,True])

if __name__=='__main__':unittest.main()
