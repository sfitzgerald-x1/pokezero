"""Two source boundaries + actual original request APIs, no engine turn/game."""
import ast, json, subprocess, unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex, load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant, RandbatSourceMetadata
from gpu_engine import engine_baton_pass_callbacks as cb, engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex, engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life, engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver, engine_request_source as request, engine_request_order as ordering, engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full, engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr, public_belief, v4_encoder as v4, public_wire, state, source_provenance, baton, entry
from gpu_engine import engine_self_boost_callbacks as boosts, private_requests
from gpu_engine_default_observer_ast import dump_default_program, assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_self_boost_input as fixtures
import test_gpu_engine_transition_observer_wiring as static
BASE = 'e7b736c6b2a13900dd17f68413e0f408b3b0f87e'

class BatonToxicStatusTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fixtures.SelfBoostInputTests.setUpClass()
        f = fixtures.SelfBoostInputTests
        cls.f = f
        cls.metadata = deepcopy(f.metadata)
        dex = load_showdown_dex_cached(showdown_root())
        moves = dict(f.dex.moves)
        name = 'batonpass'
        info = dex.move_info(name)
        moves[name] = info
        if name not in cls.metadata['vocabulary']['moves']:
            cls.metadata['vocabulary']['moves'].append(name)
            cls.metadata['moves'].append(None)
        cls.move = cls.metadata['vocabulary']['moves'].index(name)
        cls.metadata['moves'][cls.move] = dict(id=name, base_power=0, type=1, source_type='Normal', category='Status', pp=40, accuracy=True, priority=0, flags={'metronome': 1})
        if 'runaway' not in cls.metadata['vocabulary']['abilities']:
            cls.metadata['vocabulary']['abilities'].append('runaway')
        cls.ability = cls.metadata['vocabulary']['abilities'].index('runaway')
        cls.dex = ShowdownDex(moves, f.dex.species, f.dex.type_chart)
        cls.names = ('batonpass', 'surf', 'tailglow', 'agility')
        variants = {n: Gen3RandbatSpeciesUniverse(n, 80, (Gen3RandbatVariant('baton-' + n, 'unit', n, 'unit', 80, cls.names, 'Run Away', 'Leftovers'),)) for n in cls.metadata['vocabulary']['species'][1:]}
        src = Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle', 3, None, None, None, 'literal-baton-transaction-unit'), variants)
        labels = set(f.catalog.vocabulary.tokens) | {'move:batonpass', 'belief:possible_move:batonpass', 'ability:runaway', 'belief:possible_ability:runaway'}
        cls.catalog = v4.compile_encoder(public_belief.compile_source(src), cls.dex, CategoryVocabulary(tuple(sorted(labels))), checkpoint_sha256='a' * 64, feature_masks=f.catalog.masks.__dict__)
        cls.table = tr.tables_from_metadata(cls.metadata)
        cls.producer = public.compile_catalog(cls.metadata, cls.table, public_moves=cls.catalog.moves, dex=cls.dex, exposure=public.HealthExposure(public.SHOWDOWN, 'gen3randombattle', True, False))
        cls.rules = request.load_source_rules(showdown_root(), cls.metadata)
        cls.req = request.compile_request_source(cls.metadata, vocabulary=cls.catalog.vocabulary, dex=cls.dex, checkpoint_sha256='a' * 64, rules=cls.rules, lifecycle_catalog=cls.producer)
        facts = own.compile_own_facts(cls.req, cls.catalog)
        cls.helper = direct.compile_direct_hit_controls(engine_catalog=cls.producer, metadata=cls.metadata, source_rules=cls.rules)
        cls.prepared = driver.prepare(cls.req, facts, public_mode='retained-full', account_source_sites=True, track_kernel_risks=True, frame_input_mode='source-witness', leftovers_compiled=leftovers.compile_leftovers(cls.metadata, engine_catalog=cls.producer), leftovers_mode='retained', direct_hit_compiled=cls.helper, direct_hit_mode='kernel-event', fake_forward=lambda _: driver.FakeScores(j.zeros(9), j.array(0, j.int32)))
        cls.runtime = driver._checked(cls.prepared)
        cls.kw = dict(catalog=cls.producer, expected_binding=cls.producer.binding, expected_engine_binding=np.asarray(cls.producer.engine_binding, np.uint32))
        cls.fold = staticmethod(jax.jit(lambda s, f, p: full.project_and_fold(s, f, cls.runtime['public'].device, p, cls.catalog)))
        cls.consume = staticmethod(jax.jit(lambda l, f, s, p: ex.consume_frame_device(l, f, s, p, active=j.ones(len(l.stream_id), bool), tables=cls.runtime['execution_compiled'].device, adapter_tables=cls.runtime['public'].device, catalog=cls.catalog)))

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

    def new_tape(self, x, ledger=None):
        b = len(x.s.hp)
        yes = j.ones(b, bool)
        t = life._tape(x.cursor, self.producer, 64)
        a = self.runtime['execution_compiled']
        r = self.runtime['kernel_risk_compiled']
        l = ex.initialize(x.cursor.stream_id, compiled=a) if ledger is None else ledger
        t = ex.begin_on_tape(t, l, a.device, active=yes)
        carry = risk.begin_frame_device(risk.initialize_device(t.stream_id, r.device)._replace(frame=l.frame), t, r.device, active=yes)
        return t._replace(execution=t.execution._replace(risks=risk.Context(carry, r.device)))

    def produced(self, s, orders, regs, cursor):
        return tuple((request.produce_request(self.req, s, o, reg, life.snapshot(cursor), lifecycle_admitted=j.ones(len(s.hp), bool)) for (o, reg) in zip(orders, regs)))

    def prepare_tape(self, x, t, history, requests, actions):
        proof = cb.public_proof(x.s, x.cursor, tuple((p for (_, p) in x.owners)), j.ones(2, bool), self.helper.device.baton_pass)
        return cb.prepare(t, x.s, history, proof, requests, x.orders, actions, self.helper.device.baton_pass)

    def drain(self, x, t, s, tag):
        z = j.zeros(2, j.int32)
        yes = j.ones(2, bool)
        frame = life.LifecycleResult(s, t.cursor, t, life.snapshot(x.cursor), life.snapshot(t.cursor), ~yes, z)
        new = []
        ledgers = []
        actual = []
        for (owner, (stream, pub)) in enumerate(x.owners):
            (ledger, r) = self.consume(t.execution.ledger, frame, stream, pub)
            actual.append(dict(owner=owner, error=r.projection.error.tolist(), words=r.projection.frames.words.tolist(), valid=r.projection.frames.valid.tolist(), stage=ledger.stage.tolist(), refusal=ledger.refusal.tolist()))
            Path('/tmp/pokezero-baton-toxic-' + tag + '-actual.json').write_text(json.dumps(actual) + '\n')
            self.assertEqual(r.projection.error.tolist(), [0, 0], actual[-1])
            new.append((r.projection.state, r.public))
            ledgers.append(ledger)
        return (frame, new, ledgers)

    def start_component(self):
        x = self.initial()
        b = 2
        r = j.arange(b)
        z = j.zeros(b, j.int32)
        yes = j.ones(b, bool)
        side = r % 2
        requests = self.produced(x.s, x.orders, x.regs, x.cursor)
        history = cb.empty_history(x.s, x.cursor, j.stack(tuple((o.request_to_physical for o in x.orders)), axis=1))
        t = self.prepare_tape(x, self.new_tape(x), history, requests, j.zeros((b, 2), j.int32))
        t = life.begin(t, x.s, yes, ~yes)
        move = z + self.metadata['vocabulary']['moves'].index('tailglow')
        t = public.move(t, side, z, z, move, eligible=yes)
        old = x.s.boosts[r, side, 0]
        (t, p) = boosts.begin(t, x.s, side, move, move, old, self.runtime['direct_hit_context'], self.table.effects, eligible=yes)
        updated = j.clip(old + self.table.effects.self_boosts[move], -6, 6)
        (t, c) = boosts.finish(t, p, updated, yes)
        s = x.s._replace(boosts=x.s.boosts.at[r, side, 0].set(updated))
        (_, owners, ledgers) = self.drain(x, t, s, 'prior-boost')
        x.s = s
        x.cursor = t.cursor
        x.owners = owners
        ledger = ex.join_consumed_owners(ledgers[0], ledgers[1], tables=self.runtime['execution_compiled'].device)[0]
        x.s = x.s._replace(pp=x.s.pp.at[r, side, 0, 0].add(-1))
        requests = self.produced(x.s, x.orders, x.regs, x.cursor)
        history = cb.empty_history(x.s, x.cursor, j.stack(tuple((o.request_to_physical for o in x.orders)), axis=1))
        t = self.prepare_tape(x, self.new_tape(x, ledger), history, requests, j.zeros((b, 2), j.int32))
        t = life.begin(t, x.s, yes, ~yes)
        move = z + self.move
        t = public.move(t, side, z, z, move, eligible=yes)
        before = t
        (t, p) = cb.begin_start(t, x.s, side, move, move, self.runtime['direct_hit_context'], acts=yes, eligible=yes)
        bench = j.any((x.s.hp[r, side] > 0) & (j.arange(6)[None, :] != x.s.active[r, side, None]), axis=1)
        step = static._step(Path('gpu_engine/transition.py').read_text())
        assignment = next((n for n in ast.walk(step) if isinstance(n, ast.Assign) and any((isinstance(a, ast.Name) and a.id == 'passed' for a in n.targets))))
        passed = eval(compile(ast.Expression(assignment.value), 'actual-Baton-pass', 'eval'), {}, dict(acts=yes, table=self.table, move=move, hp=x.s.hp[r, side, 0], bench_alive=bench))
        s = x.s._replace(baton_pass=x.s.baton_pass.at[r, side].set(passed))
        t = cb.finish_start(t, p, s, passed, yes)
        s = s._replace(forced_switch=s.baton_pass, pending_residual=yes, turn=z + 1)
        t = cb.finalize_request(t, s)
        self.assertEqual(cb.owned(t, t.count - 1, cb.START).tolist(), [True, True])
        return SimpleNamespace(x=x, s=s, t=t, side=side, before=before, pending=p)

    def test_two_boundaries_original_both_full_and_private_request(self):
        c = self.start_component()
        x = c.x
        b = 2
        r = j.arange(b)
        z = j.zeros(b, j.int32)
        yes = j.ones(b, bool)
        (frame, owners, ledgers) = self.drain(x, c.t, c.s, 'start')
        requests = self.produced(c.s, x.orders, x.regs, c.t.cursor)
        for (owner, q) in enumerate(requests):
            self.assertEqual(q.actions.facts.request.kind.tolist(), [int(private_requests.RequestKind.FORCE_SWITCH) if int(side) == owner else int(private_requests.RequestKind.WAIT) for side in c.side])
        history = cb.mature(c.t.execution.baton_transaction.history, frame, tuple(ledgers), tuple((p for (_, p) in owners)), requests)
        self.assertEqual(history.ready.tolist(), [True, True])
        x.s = c.s
        x.cursor = c.t.cursor
        x.owners = owners
        ledger = ex.join_consumed_owners(ledgers[0], ledgers[1], tables=self.runtime['execution_compiled'].device)[0]
        actions = j.zeros((b, 2), j.int32).at[r, c.side].set(7)
        t = self.prepare_tape(x, self.new_tape(x, ledger), history, requests, actions)
        self.assertEqual(t.execution.baton_transaction.request_ok.tolist(), [True, True])
        t = life.begin(t, x.s, yes, ~yes)
        target = z + 3
        trn = t.execution.baton_transaction
        for (name, h) in [('stale', history._replace(frame=history.frame - 1)), ('foreign', history._replace(stream_id=history.stream_id + 1)), ('unconsumed', history._replace(ready=j.zeros(b, bool)))]:
            bad = self.prepare_tape(x, self.new_tape(x, ledger), h, requests, actions)
            self.assertFalse(bool(bad.execution.baton_transaction.request_ok.any()), name)
        bad_proof = trn.public._replace(pending=j.zeros_like(trn.public.pending))
        bad = cb.prepare(self.new_tape(x, ledger), x.s, history, bad_proof, requests, x.orders, actions, self.helper.device.baton_pass)
        self.assertFalse(bool(bad.execution.baton_transaction.request_ok.any()))
        swapped = x.orders[0]._replace(request_to_physical=x.orders[0].request_to_physical.at[:, 1].set(2).at[:, 2].set(1))
        bad = cb.prepare(self.new_tape(x, ledger), x.s, history, trn.public, requests, (swapped, x.orders[1]), actions, self.helper.device.baton_pass)
        self.assertFalse(bool(bad.execution.baton_transaction.request_ok.any()))
        (_, wrong) = cb.before_copy(t, x.s, c.side, target + 1, self.runtime['direct_hit_context'], eligible=yes)
        self.assertFalse(bool(wrong.valid.any()))
        (t, p) = cb.before_copy(t, x.s, c.side, target, self.runtime['direct_hit_context'], eligible=yes)
        self.assertEqual(p.valid.tolist(), [True, True])
        after = baton.copy_to_recipient(x.s, c.side, target, eligible=yes)
        t = cb.after_copy(t, p, after)
        after = entry.depart(after, c.side, self.table.entry, eligible=yes)
        after = after._replace(active=after.active.at[r, c.side].set(target), baton_pass=after.baton_pass.at[r, c.side].set(False), forced_switch=after.forced_switch.at[r, c.side].set(False))
        plain = after
        self.assertEqual(cb.reveal_matches(t, after, c.side, z, target).tolist(), [True, True])
        t = life.installed(t, after, c.side, z, target, eligible=yes, replacement=yes, baton=yes)
        t = cb.finish_install(t, after, c.side, eligible=yes)
        Path('/tmp/pokezero-baton-toxic-install-input.json').write_text(json.dumps(dict(raw=t.words.tolist(), count=t.count.tolist(), refusal=t.refusal.tolist(), stage=t.execution.ledger.stage.tolist(), record_known=t.execution.baton_transaction.record.known.tolist(), finalized=t.execution.baton_transaction.record.finalized.tolist())) + '\n')
        self.assertEqual(cb.owned(t, t.count - 1, cb.INSTALL).tolist(), [True, True])
        for name in ('engine_binding', 'source_binding', 'move_binding', 'producer_binding'):
            bad = t._replace(**{name: getattr(t, name).at[0].add(j.uint32(1))})
            self.assertFalse(bool(cb.owned(bad, t.count - 1, cb.INSTALL).any()), name)
            with self.assertRaises(ValueError):
                cb.owned(t._replace(**{name: j.broadcast_to(getattr(t, name), (8, 8))}), t.count - 1, cb.INSTALL)
        bad = t._replace(count=t.count - 1)
        self.assertFalse(bool(cb.owned(bad, t.count - 2, cb.INSTALL).any()))
        rec = t.execution.baton_transaction.record
        bad_rec = rec._replace(boost_after=rec.boost_after.at[0, 0, 3, 2].add(1))
        bad = t._replace(execution=t.execution._replace(baton_transaction=t.execution.baton_transaction._replace(record=bad_rec)))
        self.assertEqual(cb.owned(bad, t.count - 1, cb.INSTALL).tolist(), [False, True])
        absent = x.owners[0][1]._replace(context=x.owners[0][1].context._replace(tendency=x.owners[0][1].context.tendency._replace(baton=j.zeros((b, 2), bool))))
        self.assertFalse(bool(cb.public_guard(t, absent).any()))
        (install_frame, new, closed) = self.drain(x, t, after, 'install')
        self.assertEqual(cb.public_guard(t, x.owners[0][1]).tolist(), [True, True])
        expected = []
        for (owner, ((oldstream, oldpub), (stream, pub))) in enumerate(zip(x.owners, new)):
            for (row, side) in enumerate(c.side.tolist()):
                initial = self.catalog.species.names[1]
                incoming = self.catalog.species.names[2]
                ids = {s: f'p{s + 1}a: ' + initial for s in (0, 1)}
                target_ident = f'p{side + 1}a: ' + incoming
                prefix = [f'|switch|{ids[s]}|{initial}, L80, M|100/100'+(' tox' if s==side else '') for s in (0, 1)] + ['|turn|1', '|', '|t:|1', f'|move|{ids[side]}|Tail Glow|{ids[side]}', f'|-boost|{ids[side]}|spa|2']
                (wire, _) = public_wire.project_lines(public_wire.WireState(), prefix, **v4.wire_kwargs(self.catalog))
                (wire, start) = public_wire.project_lines(wire, ['|', '|t:|1', f'|move|{ids[side]}|Baton Pass|{ids[side]}'], **v4.wire_kwargs(self.catalog))
                (_, out) = public_wire.project_lines(wire, ['|', '|t:|1', f'|switch|{target_ident}|{incoming}, L80, M|100/100'], **v4.wire_kwargs(self.catalog))
                want = np.array([e.words() for p in out for e in p.events], np.int32)
                actual_doc = json.loads(Path('/tmp/pokezero-baton-toxic-install-actual.json').read_text())[owner]
                words = np.array(actual_doc['words'], np.int32)
                valid = np.array(actual_doc['valid'])
                actual = words[:, row][valid[:, row]]
                expected.append(dict(owner=owner, row=row, actual=actual.tolist(), expected=want.tolist()))
                Path('/tmp/pokezero-baton-toxic-original-both-full-compact.json').write_text(json.dumps(expected) + '\n')
                np.testing.assert_array_equal(actual, want)
                self.assertEqual(int(pub.mons.boosts[row, side, 2]), 2)
                self.assertTrue(bool(pub.mons.arrived_by_baton[row, side]))
                self.assertFalse(bool(pub.context.tendency.baton[row, side]))
        orders = tuple((life.apply_installations(o, install_frame) for o in x.orders))
        current = self.produced(after, orders, x.regs, t.cursor)
        for (owner, q) in enumerate(current):
            self.assertEqual(q.actions.admitted.tolist(), [True, True])
            self.assertEqual(q.actions.boundary.generation.tolist(), orders[owner].generation.tolist())
            self.assertEqual(q.actions.facts.request.pp[:, 0].tolist(), [64 if int(side) == owner else 64 for side in c.side])
        for (a, z) in zip(jax.tree.leaves(after), jax.tree.leaves(plain), strict=True):
            np.testing.assert_array_equal(a, z)

class BatonToxicGuardTests(unittest.TestCase):
 def test_same_call_status_counter_and_departure_projection(self):
  from types import SimpleNamespace
  s=state.empty_state(8)._replace(hp=j.full((8,2,6),100,j.int32),status=j.zeros((8,2,6),j.int32),ability=j.ones((8,2,6),j.int32),base_ability=j.ones((8,2,6),j.int32))
  r=j.arange(8);side=r%2;old=j.zeros(8,j.int32);status=j.array([0,4,4,4,1,2,5,6],j.int32);counter=j.array([0,0,2,16,0,0,1,0],j.int32)
  s=s._replace(status=s.status.at[r,side,old].set(status),status_counter=s.status_counter.at[r,side,old].set(counter));d=SimpleNamespace(quiet_abilities=j.array([False,True]),quiet_items=j.array([True]))
  self.assertEqual(cb._quiet(s,side,old,d,outgoing_toxic=True).tolist(),[True,True,True,False,False,False,False,False])
  self.assertEqual(cb._quiet(s,side,old,d).tolist(),[True,False,False,False,False,False,False,False])
  before=cb.snapshot(s);self.assertIn('status_counter',before._fields)
  self.assertEqual(cb.departed_private(before,side,old).status_counter[r,side,old].tolist(),[0,0,0,0,0,0,1,0])
  altered=s._replace(status_counter=s.status_counter.at[2,side[2],0].add(1));self.assertEqual(cb.equal(before,cb.snapshot(altered)).tolist(),[True,True,False,True,True,True,True,True])
  self.assertEqual(cb.TOXIC_STATUS,4)
 def test_actual_same_frame_toxic_then_baton_work_operand_order(self):
  tree=ast.parse(Path('gpu_engine/transition.py').read_text());calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)]
  prior=next(n for n in calls if ast.unparse(n.func)=='toxic_callbacks.record_device');start=next(n for n in calls if ast.unparse(n.func)=='baton_callbacks.begin_start')
  self.assertLess(prior.lineno,start.lineno)
  self.assertEqual([ast.unparse(x)for x in start.args],['_public_trace','work','side','move','caller_move','_direct_hit_context if _direct_hit_registered else None'])
  self.assertEqual({k.arg:ast.unparse(k.value)for k in start.keywords},{'acts':'acts','eligible':'caller_acts'})
  # Evaluate the ORIGINAL production call, not a retyped surrogate call. These
  # inert handles distinguish updated work from pre-status state and retain the
  # source registration conditional. No engine, constructor or model is called.
  from types import SimpleNamespace
  current=object();context=object();seen=[]
  for registered in (False,True):
   env=dict(baton_callbacks=SimpleNamespace(begin_start=lambda *a,**kw:seen.append((a,kw))),_public_trace=object(),work=current,side=0,move=18,caller_move=18,_direct_hit_context=context,_direct_hit_registered=registered,acts=True,caller_acts=True)
   eval(compile(ast.Expression(start),'actual-Baton-source-call','eval'),env)
   args,kw=seen[-1];self.assertIs(args[1],current);self.assertIs(args[5],context if registered else None);self.assertEqual(kw,dict(acts=True,eligible=True))
 def test_original_toxic_counter_not_copied_and_default_physics_exact(self):
  script=r"""const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,out=[];for(const stage of[0,2]){const s={status:'tox',statusState:{stage},side:{},volatiles:{},boosts:{atk:0,def:0,spa:0,spd:0,spe:0,accuracy:0,evasion:0},clearVolatile(){this.boosts={atk:0,def:0,spa:0,spd:0,spe:0,accuracy:0,evasion:0}}},t={status:'',statusState:{},volatiles:{},clearVolatile(){this.volatiles={}},battle:{}};d.moves.get('batonpass').onHit.call({canSwitch:()=>true},s);d.moves.get('batonpass').self.onHit.call({},s);P.prototype.copyVolatileFrom.call(t,s,'copyvolatile');d.conditions.get('tox').onSwitchIn.call({effectState:s.statusState});out.push({outgoingStatus:s.status,incomingStatus:t.status,sourceStage:s.statusState.stage,targetState:t.statusState});}console.log(JSON.stringify(out));"""
  v=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(v,[dict(outgoingStatus='tox',incomingStatus='',sourceStage=0,targetState={})]*2)
  for n in('transition.py','entry.py','baton.py','state.py','mechanics.py'):
   # Integrated physical baseline; BASE below still tests the pre-fix toxic regression.
   physical_base='b8092a85'
   self.assertEqual(Path('gpu_engine',n).read_bytes(),subprocess.check_output(['git','show',physical_base+':gpu_engine/'+n]))
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_baton_pass_callbacks.py'],text=True));ns={};exec(compile(ast.Module([n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='_quiet'],[]),'baseline-quiet','exec'),dict(j=j),ns)
  s=state.empty_state(1)._replace(hp=j.ones((1,2,6),j.int32),status=j.full((1,2,6),4,j.int32),ability=j.ones((1,2,6),j.int32),base_ability=j.ones((1,2,6),j.int32));d=__import__('types').SimpleNamespace(quiet_abilities=j.array([False,True]),quiet_items=j.array([True]));self.assertEqual(ns['_quiet'](s,j.array([0]),j.array([0]),d).tolist(),[False])

if __name__=='__main__':unittest.main()
