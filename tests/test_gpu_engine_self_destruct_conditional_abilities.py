"""Prospective exact conditional ability inputs; no engine turn or game."""
import ast, hashlib, json, re, subprocess, unittest
from pathlib import Path
from _showdown_root import showdown_root
import ast, json, subprocess, unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex, load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant, RandbatSourceMetadata
from gpu_engine import engine_self_destruct_callbacks as cb, engine_direct_hit_controls as direct, engine_execution_completeness as ex, engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life, engine_public_events as public, engine_retained_encoder_driver as driver, engine_request_source as request, engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full, engine_leftovers_callbacks as leftovers, engine_source_frame_witness as witness
from gpu_engine import transition as tr, public_belief, v4_encoder as v4, public_wire, state
from gpu_engine_default_observer_ast import dump_default_program, assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_self_boost_input as fixtures
import test_gpu_engine_transition_observer_wiring as static
import test_gpu_engine_self_destruct_static_noncontact as old_fixture

class ConditionalSourceTests(unittest.TestCase):
    def test_original_conditional_callbacks_and_strict_source_descriptors(self):
        # Actual effective handlers on inert objects: no battle/world exists.
        script=r'''const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),out=[];
const own=d.abilities.get('owntempo'),point=d.abilities.get('poisonpoint');
own.onUpdate.call({add(){throw Error('no confusion must be silent')}},{volatiles:{},removeVolatile(){throw Error('unexpected cure')}});
own.onUpdate.call({add:(...v)=>out.push(['activate',v[0],v[2]])},{volatiles:{confusion:{}},removeVolatile:v=>out.push(['remove',v])});
for(const id of ['explosion','selfdestruct'])point.onDamagingHit.call({randomChance(){throw Error('noncontact must not draw')}},100,{}, {trySetStatus(){throw Error('unexpected status')}},d.moves.get(id));
point.onDamagingHit.call({randomChance:(n,m)=>{out.push(['chance',n,m]);return false}},100,{}, {},d.moves.get('bodyslam'));
console.log(JSON.stringify(out));'''
        self.assertEqual(json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True)),[['activate','-activate','ability: Own Tempo'],['remove','confusion'],['chance',1,3]])
        tree=ast.parse(Path('gpu_engine/engine_self_destruct_callbacks.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='compile_data')
        check=next(n for n in fn.body if isinstance(n,ast.If)and any(isinstance(c,ast.Constant)and c.value=='original conditional OwnTempo/PoisonPoint callbacks required'for c in ast.walk(n)))
        expected=cb.CONDITIONAL_ABILITIES
        exec(compile(ast.Module([check],[]),'actual-condition-source-guard','exec'),dict(v={'conditional_abilities':expected},CONDITIONAL_ABILITIES=expected))
        for name in expected:
            for key in expected[name]['hashes']:
                changed=deepcopy(expected);changed[name]['hashes'][key]='0'*64
                with self.assertRaises(ValueError):exec(compile(ast.Module([check],[]),'actual-condition-source-guard','exec'),dict(v={'conditional_abilities':changed},CONDITIONAL_ABILITIES=expected))
        # The namespace mask uses the same original Dex identity as the
        # capability predicate, even for normalized labels accepted by rules.
        source=ast.literal_eval(next(n.value for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='script'for t in n.targets)))
        expression=re.search(r"own_tempo:(meta\.vocabulary\.abilities\.map\(n=>d\.abilities\.get\(n\)\.id==='owntempo'\))",source).group(1)
        script="const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),meta={vocabulary:{abilities:['owntempo','Own Tempo','poisonpoint','static']}};console.log(JSON.stringify("+expression+"));"
        self.assertEqual(json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True)),[True,True,False,False])

class ConditionalAbilityInputTests(unittest.TestCase):
    extra_abilities=('owntempo','poisonpoint')
    fixture_ability='owntempo'
    fixture_target='poisonpoint'
    @classmethod
    def setUpClass(cls):
        old_fixture.SelfDestructStaticNoncontactTests.setUpClass.__func__(cls)
    component=old_fixture.SelfDestructStaticNoncontactTests.component
    def test_current_conditional_facts_and_original_prefix_remain_required(self):
        x=self.component();b=4;r=j.arange(b);yes=j.ones(b,bool)
        self.assertEqual(cb.owned(x.tape,j.array([2]*b)).tolist(),[True]*b)
        for name,value in (('confusion_counter',1),('volatile',j.uint32(1))):
            for seat in (x.side,1-x.side):
                # Only OwnTempo needs the new inactivity facts. PoisonPoint
                # remains proven by exact original/installed noncontact.
                actor=bool(j.array_equal(seat,x.side))
                where=(0,int(seat[0]),int(x.s.active[0,seat[0]]))
                state=x.s._replace(**{name:getattr(x.s,name).at[where].set(value)})
                _,pending=cb.begin(x.before,state,x.side,x.move,x.move,self.runtime['direct_hit_context'],requested=yes)
                self.assertEqual(pending.self_ok.tolist(),[not actor]+[True]*3)
        record=x.self_after.execution.self_destruct
        missing=record._replace(before=record.before._replace(confusion=None,volatile=None))
        self.assertEqual(cb.self_matches(x.self_after,missing).tolist(),[False]*b)
        for name in ('confusion','volatile'):
            missing=record._replace(before=record.before._replace(**{name:None}))
            self.assertEqual(cb.self_matches(x.self_after,missing).tolist(),[False]*b)
        # An unrelated state write cannot hide behind a before snapshot.
        at=(0,int(x.side[0]),int(x.s.active[0,x.side[0]]))
        state=x.s._replace(hp=record.self_after,confusion_counter=x.s.confusion_counter.at[at].set(1))
        out,p=cb.self_write(x.before,x.pending,state)
        self.assertEqual(p.self_ok.tolist(),[False]+[True]*3)
        # Exact final snapshot must still conserve those facts after the
        # target write and both literal faint events, before stage2 credit.
        end=cb.actor_faint(x.target_tape,x.after)
        end=public.faint(end,1-x.side,x.s.active[r,1-x.side],eligible=x.write.new_hp==0)
        changed=x.after._replace(confusion_counter=x.after.confusion_counter.at[at].set(1))
        refused=cb.finish(end,changed)
        self.assertNotEqual(int(refused.execution.ledger.refusal[0]),0)
        self.assertEqual(refused.execution.ledger.refusal[1:].tolist(),[0]*3)
        for field in ('engine_binding','source_binding','move_binding','producer_binding'):
            bad=x.tape._replace(**{field:getattr(x.tape,field).at[0].add(1)})
            self.assertFalse(bool(cb.owned(bad,2).any()))
        wrong=x.move.at[0].add(1)
        _,pending=cb.begin(x.before,x.s,x.side,x.move,wrong,self.runtime['direct_hit_context'],requested=yes)
        self.assertEqual(pending.self_ok.tolist(),[False]+[True]*3)
        # Preserve the original feature's unchanged physical sources, then
        # bind the complete integrated sources separately.
        for n in ('transition.py','state.py','mechanics.py','secondary.py','self_destruct.py','engine_lifecycle_event_device.py','engine_full_v4_event_device.py'):
            historical=subprocess.check_output(['git','show','6764d02b:gpu_engine/'+n])
            self.assertEqual(historical,subprocess.check_output(['git','show','f2f3b9ad725c7f3064f0420e07425ec68e95ec37:gpu_engine/'+n]))
            # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
            # state.py and secondary.py are byte-certified against the reviewed SOURCE_PINS by transition_observer_wiring.
            if n not in ('transition.py','state.py','secondary.py'):
                self.assertEqual(Path('gpu_engine',n).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+n]))

    def test_both_original_full_owners(self):
        x = self.component()
        b = 4
        z = j.zeros(b, j.int32)
        yes = j.ones(b, bool)
        frame = life.LifecycleResult(x.s, x.initial.cursor, x.initial, life.snapshot(x.cursor), life.snapshot(x.initial.cursor), ~yes, z)
        owners = []
        for owner in (0, 1):
            stream = full.register(x.cursor.stream_id, z + owner, compiled=self.runtime['public'])
            pub = v4.initialize(self.catalog, stream_ids=x.cursor.stream_id, owner_sides=z + owner)
            r = self.fold(stream, frame, pub)
            self.assertEqual(r.projection.error.tolist(), [0] * b)
            owners.append((r.projection.state, r.public))
        raw = witness._raw_complete_words(x.tape, self.runtime['frame_witness_compiled'].device)
        self.assertTrue(bool(j.all(j.where(j.arange(64)[None, :] < x.tape.count[:, None], raw, True))), raw.tolist())
        frame = life.LifecycleResult(x.after, x.tape.cursor, x.tape, life.snapshot(x.initial.cursor), life.snapshot(x.tape.cursor), ~yes, z)
        compact = []
        for (owner, (stream, pub)) in enumerate(owners):
            (closed, r) = self.consume(x.tape.execution.ledger, frame, stream, pub)
            Path('/tmp/pokezero-self-destruct-conditional-abilities-full-' + str(owner) + '.json').write_text(json.dumps(dict(stage=closed.stage.tolist(), refusal=closed.refusal.tolist(), error=r.projection.error.tolist(), words=r.projection.frames.words.tolist(), valid=r.projection.frames.valid.tolist())) + '\n')
            self.assertEqual(r.projection.error.tolist(), [0] * b)
            self.assertEqual(closed.stage[:, :2].tolist(), [[3, 3]] * b)
            for (row, side) in enumerate(x.side.tolist()):
                names = {s: self.catalog.species.names[s + 1] for s in (0, 1)}
                ids = {s: f'p{s + 1}a: ' + names[s] for s in (0, 1)}
                (wire, _) = public_wire.project_lines(public_wire.WireState(), [f'|switch|{ids[s]}|{names[s]}, L80, M|100/100' for s in (0, 1)] + ['|turn|1'], **v4.wire_kwargs(self.catalog))
                lines = ['|', '|t:|1', f"|move|{ids[side]}|{self.dex.move_info(self.metadata['vocabulary']['moves'][int(x.move[row])]).name}|{ids[1 - side]}"]
                if bool(x.write.critical[row]):
                    lines.append(f'|-crit|{ids[1 - side]}')
                if int(x.write.numerator[row]) != int(x.write.denominator[row]):
                    lines.append(f'|-supereffective|{ids[1 - side]}')
                hp = int(x.write.new_hp[row])
                lines += [f'|-damage|{ids[1 - side]}|' + (f'{hp}/100' if hp else '0 fnt'), f'|faint|{ids[side]}']
                if not hp:
                    lines.append(f'|faint|{ids[1 - side]}')
                (_, packets) = public_wire.project_lines(wire, lines, **v4.wire_kwargs(self.catalog))
                want = np.asarray([e.words() for p in packets for e in p.events], np.int32)
                actual = np.asarray(r.projection.frames.words[:, row])[np.asarray(r.projection.frames.valid[:, row])]
                compact.append(dict(owner=owner, row=row, lines=lines, actual=actual.tolist(), expected=want.tolist()))
                Path('/tmp/pokezero-self-destruct-conditional-abilities-original-both-full-compact.json').write_text(json.dumps(compact) + '\n')
                np.testing.assert_array_equal(actual, want)
