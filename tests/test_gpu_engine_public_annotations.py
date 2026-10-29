"""Canonical main annotations retain every shared V4 event consumer."""
import importlib.util
import unittest

SETUP=('|player|p1|One|1','|player|p2|Two|2|1000','|teamsize|p1|6','|teamsize|p2|6',
       '|gametype|singles','|gen|3','|tier|[Gen 3] Random Battle','|rule|Species Clause: Limit one of each species','|start')
ANNOTATIONS=('|-crit|p2a: Absol','|-supereffective|p2a: Absol','|-resisted|p2a: Absol',
             '|-hitcount|p2a: Absol|1','|-hitcount|p2a: Absol|5','|-hint|Sleep Clause Mod activated.',
             '|-message|A public announcement.','|-fieldactivate|move: Perish Song')


class AnnotationLexicalTests(unittest.TestCase):
    def test_bounded_canonical_shapes_and_refusals(self):
        from gpu_engine.public_annotations import classify
        for line in (*SETUP,*ANNOTATIONS):self.assertIsNotNone(classify(line),line)
        for line in ('|gen|4','|gametype|doubles','|tier|gen9randombattle','|start|extra',
                '|teamsize|p1|7','|teamsize|p3|6','|player|p3|Three','|player|p1|',
                '|-crit','|-crit|p2: Absol','|-crit|p2a: Absol|extra','|-hitcount|p2a: Absol|0',
                '|-hitcount|p2a: Absol|6','|-hitcount|p2a: Absol|01','|-hint|','|-message|x|extra',
                '|-fieldactivate|move: Mud Sport','|-fieldactivate|move: Perish Song|extra',
                '|-hint|[from] ability: Pressure|[of] p2a: Absol','|request|{}','|UNSUPPORTED|x'):
            self.assertIsNone(classify(line),line)
        for line in ('|-crit|p2a: Absol\n','|-hint|two\rlines','|gen|3 '):self.assertIsNone(classify(line),line)


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class AnnotationStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_switch_context import SwitchContextTests
        SwitchContextTests.setUpClass();cls.f=SwitchContextTests.f;cls.j,cls.jax=cls.f.j,cls.f.jax

    def test_setup_and_annotations_match_all173_source_columns_and_retain_names(self):
        from gpu_engine import public_wire,public_inputs
        lines=[*SETUP,*self.f.public.prefix,'|turn|1','|move|p1a: Nick|Surf|p2a: Absol',
            *ANNOTATIONS,'|-damage|p2a: Absol|80/100']
        host,projections=public_wire.project_lines(public_wire.WireState(),lines,**self.f.e.wire_kwargs(self.f.catalog))
        self.assertEqual(host.player_names,('One','Two'))
        self.assertEqual(len(projections),len(lines))
        for projection in projections:
            self.assertFalse(projection.full_line_consumers_complete)
            self.assertEqual([event.kind for event in projection.events[:2]],
                [public_inputs.Kind.RAW_BEGIN,public_inputs.Kind.SWITCH_PRELUDE])
            self.assertEqual(sum(event.kind==public_inputs.Kind.NONPROC for event in projection.events),1)
            self.assertEqual(sum(event.kind==public_inputs.Kind.MUDSHOT for event in projection.events),1)
        self.f.compare_source(lines)

    def test_each_annotation_is_a_real_switch_boundary_and_closes_trick(self):
        from gpu_engine import public_wire,public_belief_snapshot
        from pokezero import showdown
        from pokezero.belief import PublicBattleBeliefEngine
        before=[self.f.public.prefix[1],self.f.public.prefix[0]]
        base=self.f.ingest(before)
        step=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))
        for line in (*ANNOTATIONS,*SETUP):
            engine=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=self.f.public.source)
            calls=[];original=engine._with_set_summary
            def record(pokemon):
                result=original(pokemon);calls.append(result.key);return result
            engine._with_set_summary=record
            for raw in before:engine.ingest_event(showdown._public_event_from_line(raw))
            self.assertTrue(engine._pending_switches,line)
            projections=self.f.projections([*before,line]);state=base
            for chunk in self.f.chunks(projections[-1:]):
                state,error=step(state,chunk);self.assertEqual(error.tolist(),[0],line)
            engine.ingest_event(showdown._public_event_from_line(line))
            self.assertFalse(engine._pending_switches,line);self.assertEqual(state.context.switches.side.tolist(),[-1])
            summary,ok=public_belief_snapshot.read(state.context.refresh.controller.visible,state.beliefs,self.f.catalog.source.device)
            self.assertTrue(bool(ok.all()))
            for side in (0,1):
                for slot,pokemon in enumerate(engine.snapshot().side(f'p{side+1}')):
                    self.assertEqual(int(summary.count[0,1-side,slot]),pokemon.candidate_set_count,line)
                    self.assertEqual(int(state.context.refresh.controller.visible.generation[0,1-side,slot]),calls.count(pokemon.key),line)
        trick=[*self.f.public.prefix,'|-activate|p1a: Nick|move: Trick|[of] p2a: Absol']
        for line in ANNOTATIONS:
            state=self.f.ingest([*trick,line]);self.assertEqual(state.context.items.pending.tolist(),[False],line)
            engine=PublicBattleBeliefEngine.from_events([showdown._public_event_from_line(raw) for raw in (*trick,line)],
                format_id='gen3randombattle',set_source=self.f.public.source)
            self.assertIsNone(engine._pending_trick,line)

    def test_unsupported_or_wrong_target_does_not_advance_host(self):
        from gpu_engine import public_wire
        host,_=public_wire.project_lines(public_wire.WireState(),self.f.public.prefix,**self.f.e.wire_kwargs(self.f.catalog))
        for line in ('|-crit|p2a: Unknown','|-hitcount|p2: Absol|2','|-fieldactivate|move: Mud Sport',
                '|gen|4','|gametype|doubles','|request|{}'):
            with self.assertRaises(ValueError):public_wire.project_line(host,line,**self.f.e.wire_kwargs(self.f.catalog))
        self.assertEqual(host.player_names,(None,None))
