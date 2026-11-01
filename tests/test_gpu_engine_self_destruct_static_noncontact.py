"""Original conditional Static callback; noncontact input capability only."""
import ast,hashlib,json,re,subprocess,unittest
from pathlib import Path
from _showdown_root import showdown_root
BASE='6103ebdf195da8f61b4bcb0329386417b78a6207'

class StaticNoncontactSourceTests(unittest.TestCase):
 def test_original_actual_compiler_static_mapping_and_no_random_callback(self):
  source=Path('gpu_engine/engine_self_destruct_callbacks.py').read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='compile_data');script=ast.literal_eval(next(n.value for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='script'for t in n.targets)))
  predicates='\n'.join(l for l in script.splitlines()if l.startswith(('const quiet=','const compatibleAbility=')))
  actual=re.search(r'abilities:meta\.vocabulary\.abilities\.map\(n=>(\w+)\(d\.abilities\.get\(n\)\)\)',script).group(1)
  node="const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),crypto=require('crypto');"+predicates+"""
const a=d.abilities.get('static'),logs=[],source={trySetStatus:(...v)=>logs.push(['status',...v])},target={};
for(const id of['explosion','selfdestruct'])a.onDamagingHit.call({randomChance:()=>{throw Error('noncontact must not draw')}},100,target,source,d.moves.get(id));
a.onDamagingHit.call({randomChance:(n,z)=>{logs.push(['chance',n,z]);return true;}},100,target,source,d.moves.get('bodyslam'));
const names=['static','runaway','rockhead','roughskin','poisonpoint'];
console.log(JSON.stringify({allowed:Object.fromEntries(names.map(n=>[n,PREDICATE(d.abilities.get(n))])),focusband:quiet(d.items.get('focusband')),callback:crypto.createHash('sha256').update(a.onDamagingHit.toString()).digest('hex'),contact:['explosion','selfdestruct'].map(n=>!!d.moves.get(n).flags.contact),logs,unexpectedHook:PREDICATE({...a,onDamage(){}})}));""".replace('PREDICATE',actual)
  value=json.loads(subprocess.check_output(['node','-e',node,str(showdown_root())],text=True))
  self.assertEqual(value['callback'],'26759547257248edbbf4ba3a036f3aecff3dacc794bf5193286fcf29870d1b07');self.assertEqual(value['contact'],[False,False]);self.assertEqual(value['logs'][0],['chance',1,3]);self.assertEqual(value['logs'][1][0:2],['status','par']);self.assertEqual(len(value['logs']),2)
  self.assertTrue(value['allowed']['static']);self.assertTrue(value['allowed']['runaway'])
  for n in ('rockhead','roughskin','poisonpoint'):self.assertFalse(value['allowed'][n])
  self.assertFalse(value['focusband']);self.assertFalse(value['unexpectedHook'])



 def test_actual_source_guards_and_default_numeric_identity(self):
  from types import SimpleNamespace
  tree=ast.parse(Path('gpu_engine/engine_self_destruct_callbacks.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='compile_data')
  condition=lambda text:next(n for n in ast.walk(fn)if isinstance(n,ast.If)and any(isinstance(k,ast.Constant)and k.value==text for k in ast.walk(n)))
  guard=condition('original conditional Static callback required')
  for h,expected in [('26759547257248edbbf4ba3a036f3aecff3dacc794bf5193286fcf29870d1b07',False),('0'*64,True)]:
   env=dict(v={'static_callback':h},STATIC_CALLBACK_SHA256='26759547257248edbbf4ba3a036f3aecff3dacc794bf5193286fcf29870d1b07')
   if expected:
    with self.assertRaises(ValueError):exec(compile(ast.Module([guard],[]),'actual-Static-source-guard','exec'),env)
   else:exec(compile(ast.Module([guard],[]),'actual-Static-source-guard','exec'),env)
  contact=condition('noncontact installed self-destruct row required')
  for value in (False,True):
   env=dict(t=SimpleNamespace(secondary=SimpleNamespace(contact=[value])),move=0)
   if value:
    with self.assertRaises(ValueError):exec(compile(ast.Module([contact],[]),'actual-installed-contact-guard','exec'),env)
   else:exec(compile(ast.Module([contact],[]),'actual-installed-contact-guard','exec'),env)
  exec(compile(ast.Module([contact],[]),'actual-disabled-contact-branch','exec'),dict(t=SimpleNamespace(secondary=None),move=0))
  transition=ast.parse(Path('gpu_engine/transition.py').read_text())
  caller=next(n for n in ast.walk(transition)if isinstance(n,ast.If)and ast.unparse(n.test)=='table.secondary is not None'and any(isinstance(c,ast.Call)and ast.unparse(c.func)=='secondaries.apply_contact'for c in ast.walk(n)))
  self.assertEqual(ast.unparse(caller.test),'table.secondary is not None')
  for n in ('transition.py','secondary.py','self_destruct.py','entry.py','mechanics.py','state.py'):
   self.assertEqual(Path('gpu_engine',n).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+n]))

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

class SelfDestructStaticNoncontactTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fixtures.SelfBoostInputTests.setUpClass()
        f = fixtures.SelfBoostInputTests
        cls.metadata = deepcopy(f.metadata)
        dex = load_showdown_dex_cached(showdown_root())
        moves = dict(f.dex.moves)
        cls.names = ('explosion', 'selfdestruct', 'surf', 'agility')
        flags = json.loads(subprocess.check_output(['node', '-e', "const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>d.moves.get(n).flags)))", str(showdown_root()), json.dumps(cls.names)], text=True))
        for (name, flag) in zip(cls.names, flags):
            info = dex.move_info(name)
            moves[name] = info
            if name not in cls.metadata['vocabulary']['moves']:
                cls.metadata['vocabulary']['moves'].append(name)
                cls.metadata['moves'].append(None)
            i = cls.metadata['vocabulary']['moves'].index(name)
            cls.metadata['moves'][i] = dict(id=name, base_power=info.base_power, type=cls.metadata['vocabulary']['types'].index(info.type.lower()), source_type=info.type, category=info.category, pp=info.pp, accuracy=True if name == 'agility' else int(info.accuracy), priority=info.priority, flags=flag)
        for name in ('runaway', 'damp', 'static', 'stickyhold'):
            if name not in cls.metadata['vocabulary']['abilities']:
                cls.metadata['vocabulary']['abilities'].append(name)
        cls.ability = cls.metadata['vocabulary']['abilities'].index('static')
        cls.dex = ShowdownDex(moves, f.dex.species, f.dex.type_chart)
        variants = {name: Gen3RandbatSpeciesUniverse(name, 80, (Gen3RandbatVariant('explode-' + name, 'unit', name, 'unit', 80, cls.names, 'Static', 'Leftovers'),)) for name in cls.metadata['vocabulary']['species'][1:]}
        source = Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle', 3, None, None, None, 'literal-selfdestruct-component'), variants)
        labels = set(f.catalog.vocabulary.tokens) | {'ability:static', 'belief:possible_ability:static', 'ability:stickyhold', 'belief:possible_ability:stickyhold'}
        for n in cls.names:
            labels |= {'move:' + n, 'belief:possible_move:' + n}
        cls.catalog = v4.compile_encoder(public_belief.compile_source(source), cls.dex, CategoryVocabulary(tuple(sorted(labels))), checkpoint_sha256='a' * 64, feature_masks=f.catalog.masks.__dict__)
        cls.table = tr.tables_from_metadata(cls.metadata)
        cls.producer = public.compile_catalog(cls.metadata, cls.table, public_moves=cls.catalog.moves, dex=cls.dex, exposure=public.HealthExposure(public.SHOWDOWN, 'gen3randombattle', True, False))
        cls.rules = request.load_source_rules(showdown_root(), cls.metadata)
        req = request.compile_request_source(cls.metadata, vocabulary=cls.catalog.vocabulary, dex=cls.dex, checkpoint_sha256='a' * 64, rules=cls.rules, lifecycle_catalog=cls.producer)
        facts = own.compile_own_facts(req, cls.catalog)
        cls.helper = direct.compile_direct_hit_controls(engine_catalog=cls.producer, metadata=cls.metadata, source_rules=cls.rules)
        cls.prepared = driver.prepare(req, facts, public_mode='retained-full', account_source_sites=True, track_kernel_risks=True, frame_input_mode='source-witness', leftovers_compiled=leftovers.compile_leftovers(cls.metadata, engine_catalog=cls.producer), leftovers_mode='retained', direct_hit_compiled=cls.helper, direct_hit_mode='kernel-event', fake_forward=lambda _: driver.FakeScores(j.zeros(9), j.array(0, j.int32)))
        cls.runtime = driver._checked(cls.prepared)
        cls.kw = dict(catalog=cls.producer, expected_binding=cls.producer.binding, expected_engine_binding=np.asarray(cls.producer.engine_binding, np.uint32))
        cls.fold = staticmethod(jax.jit(lambda s, f, p: full.project_and_fold(s, f, cls.runtime['public'].device, p, cls.catalog)))
        cls.consume = staticmethod(jax.jit(lambda l, f, s, p: ex.consume_frame_device(l, f, s, p, active=j.ones(4, bool), tables=cls.runtime['execution_compiled'].device, adapter_tables=cls.runtime['public'].device, catalog=cls.catalog)))

    def component(self):
        b = 4
        r = j.arange(b)
        z = j.zeros(b, j.int32)
        yes = j.ones(b, bool)
        side = r % 2
        active = j.stack((z + 3, z + 4), axis=1)
        actor = active[r, side]
        target = active[r, 1 - side]
        s = state.empty_state(b)._replace(active=active, gender=j.ones((b, 2, 6), j.int32), hp=j.full((b, 2, 6), 100, j.int32), max_hp=j.full((b, 2, 6), 100, j.int32), ability=j.full((b, 2, 6), self.ability, j.int32), base_ability=j.full((b, 2, 6), self.ability, j.int32), species=j.broadcast_to(j.array([1, 2])[None, :, None], (b, 2, 6)), level=j.full((b, 2, 6), 80, j.int32), done=~yes)
        sticky=self.metadata['vocabulary']['abilities'].index('stickyhold')
        s=s._replace(ability=s.ability.at[r,1-side,target].set(sticky),base_ability=s.base_ability.at[r,1-side,target].set(sticky))
        cursor = life.register(r + 211001, **self.kw)
        initial = life._tape(cursor, self.producer, 64)
        for seat in (0, 1):
            initial = life._reveal(initial, s, z + seat, z - 1, active[:, seat], eligible=yes, replacement=~yes, baton=~yes, initial=True)
        initial = initial._replace(cursor=initial.cursor._replace(initialized=yes))
        initial = life._turn(initial, yes)
        t = life._tape(initial.cursor, self.producer, 64)
        a = self.runtime['execution_compiled']
        rc = self.runtime['kernel_risk_compiled']
        t = ex.begin_on_tape(t, ex.initialize(t.stream_id, compiled=a), a.device, active=yes)
        carry = risk.begin_frame_device(risk.initialize_device(t.stream_id, rc.device), t, rc.device, active=yes)
        t = t._replace(execution=t.execution._replace(risks=risk.Context(carry, rc.device)))
        t = life.begin(t, s, yes, ~yes)
        move = j.array([self.metadata['vocabulary']['moves'].index(n) for n in ('explosion', 'selfdestruct', 'explosion', 'selfdestruct')], j.int32)
        t = public.move(t, side, actor, target, move, eligible=yes)
        before = t
        (t, p) = cb.begin(t, s, side, move, move, self.runtime['direct_hit_context'], requested=yes)
        tree = static._step(Path('gpu_engine/transition.py').read_text())
        assignment = next((n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any((isinstance(x, ast.Name) and x.id == 'work' for x in n.targets)) and ('jnp.where(explodes, 0' in ast.unparse(n.value))))
        after = eval(compile(ast.Expression(assignment.value), 'original-self-write', 'eval'), {}, dict(work=s, side=side, explodes=yes, jnp=j, _set_active=tr._set_active, _active=tr._active))
        (t, c) = cb.self_write(t, p, after)
        self.assertEqual(cb.self_matches(t, c).tolist(), [True] * b)
        self_after = t
        new = j.array([40, 0, 0, 30], j.int32)
        w = direct.HitWrite(self.table.power[move], j.array([False, True, False, True]), yes, ~yes, j.array([1, 2, 1, 1], j.int32), z + 1, ~yes, ~yes, ~yes, j.full(b, 100, j.int32), new, j.full(b, 100, j.int32))
        after = after._replace(hp=after.hp.at[r, 1 - side, target].set(new))
        t = cb.target_write(t, c, w, secondary=yes)
        t = public.condition(t, 1 - side, target, new, z + 100, z, eligible=yes)
        outcome = risk.Outcome(power=w.power, critical=w.critical, hit=w.hit, immune=w.immune, numerator=w.numerator, denominator=w.denominator, absorbed=~yes, substitute=w.substitute, protected=w.protected, enduring=w.enduring, recoil=z, explodes=yes)
        self.assertEqual(cb.matches_outcome(t, outcome).tolist(), [True] * b)
        target_tape = t
        t = cb.actor_faint(t, after)
        t = public.faint(t, 1 - side, target, eligible=new == 0)
        t = cb.finish(t, after)
        return SimpleNamespace(s=s, after=after, cursor=cursor, initial=initial, before=before, tape=t, side=side, move=move, write=w, pending=p, self_after=self_after, target_tape=target_tape, outcome=outcome)

    def test_literal_risk_partition_sticky_and_no_output_unknown(self):
        x = self.component()
        yes = j.ones(4, bool)
        r = j.arange(4)
        rc = self.runtime['kernel_risk_compiled'].device
        for (tape, code) in ((x.self_after, risk.Branch.SELF_DESTRUCT), (x.target_tape, risk.Branch.HIT_OUTCOME)):
            carry = tape.execution.risks.carry
            out = risk.observe_branch_device(carry, tape, code, yes, rc, active=yes) if code is risk.Branch.SELF_DESTRUCT else risk.observe_outcome_device(carry, tape, x.outcome, rc, eligible=yes, active=yes)
            self.assertEqual(out.risk.tolist(), [0] * 4)
        for name in x.outcome._fields:
            v = getattr(x.outcome, name)
            changed = ~v if v.dtype == j.bool_ else v + 1
            out = risk.observe_outcome_device(x.target_tape.execution.risks.carry, x.target_tape, x.outcome._replace(**{name: changed}), rc, eligible=yes, active=yes)
            self.assertTrue(bool(j.all(out.risk != 0)), name)
        sticky = x.target_tape._replace(refusal=x.target_tape.refusal | 256)
        out = risk.observe_outcome_device(sticky.execution.risks.carry, sticky, x.outcome, rc, eligible=yes, active=yes)
        self.assertTrue(bool(j.all(out.risk & risk.UPSTREAM != 0)))
        self.assertEqual([int(rc.selected_risk[int(m)]) for m in x.move], [0] * 4)
        t = cb.actor_faint(x.self_after, x.after)
        np.testing.assert_array_equal(t.count, x.self_after.count)
        missing = cb.finish(t, x.after)
        self.assertTrue(bool(j.all(missing.execution.ledger.refusal != 0)))

    def test_original_both_full_exact_words(self):
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
            Path('/tmp/pokezero-self-destruct-static-full-' + str(owner) + '.json').write_text(json.dumps(dict(stage=closed.stage.tolist(), refusal=closed.refusal.tolist(), error=r.projection.error.tolist(), words=r.projection.frames.words.tolist(), valid=r.projection.frames.valid.tolist())) + '\n')
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
                Path('/tmp/pokezero-self-destruct-static-original-both-full-compact.json').write_text(json.dumps(compact) + '\n')
                np.testing.assert_array_equal(actual, want)

if __name__=='__main__':unittest.main()
