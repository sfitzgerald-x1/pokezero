import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class NonprocStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_nonproc_refresh import NonprocRefreshTests
        NonprocRefreshTests.setUpClass();cls.f=NonprocRefreshTests.f;cls.j,cls.jax=cls.f.j,cls.f.jax
        cls.lines=NonprocRefreshTests.lines

    def test_raw_upkeep_runs_item_exclusions_before_clocks_and_matches_all_source_columns(self):
        lines=[*self.lines,'|upkeep']
        state=self.f.ingest(lines)
        self.assertFalse(bool(state.context.nonproc.maximum.any()))
        self.assertEqual(state.context.nonproc.excluded[0,0,0].tolist(),[True]*5)
        self.assertEqual(state.context.nonproc.excluded[0,1,0].tolist(),[True,False,True,True,True])
        self.assertEqual(state.context.refresh.nonproc_seen.tolist(),[0])
        self.f.compare_source(lines)

    def test_plain_sleep_upkeep_replaces_only_nonzero_skipped_records_including_a_new_pin(self):
        import numpy as np
        from gpu_engine import public_belief
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        f=self.f
        for talked,pin in ((False,False),(True,False),(True,True)):
            lines=[*self.lines[:2],'|-status|p1a: Snorlax|slp','|cant|p1a: Snorlax|slp',
                *(['|move|p1a: Snorlax|Sleep Talk|p2a: Absol'] if talked else []),
                '|upkeep','|turn|1','|cant|p1a: Snorlax|slp','|upkeep']
            projections=f.projections(lines);state=f.e.initialize(f.catalog,stream_ids=self.j.array([1],self.j.int32))
            step=self.jax.jit(lambda s,c:f.e.fold_chunk(s,c,f.catalog))
            for chunk in f.chunks(projections[:-1]):
                state,error=step(state,chunk);self.assertEqual(error.tolist(),[0])
            engine=PublicBattleBeliefEngine.from_events([_public_event_from_line(line) for line in lines[:-1]],
                format_id='gen3randombattle',set_source=f.public.source)
            if pin:
                survivor=next(v for v in f.source.variants[f.source.species_id('Snorlax')] if v.item=='lumberry').to_summary()
                self.assertTrue(engine.narrow_candidate_variants('p1:snorlax',[survivor]))
                beliefs,_=public_belief.narrow_pin(state.beliefs,f.source.pin_mask('Snorlax',[survivor]),
                    valid=self.j.zeros_like(state.beliefs.species,bool).at[0,1,0].set(True))
                state=state._replace(beliefs=beliefs)
            before=state.context.refresh.controller.visible.generation
            source_calls=[];original=engine._replace_belief
            def record(pokemon,**changes):
                result=original(pokemon,**changes)
                if 'sleep_skipped_turns' in changes:source_calls.append(result)
                return result
            engine._replace_belief=record
            for chunk in f.chunks(projections[-1:]):state,error=step(state,chunk);self.assertEqual(error.tolist(),[0])
            engine.ingest_event(_public_event_from_line('|upkeep'))
            self.assertEqual(len(source_calls),int(talked));cache=state.context.refresh.controller.visible
            self.assertEqual(int(cache.generation[0,1,0]-before[0,1,0]),int(talked))
            self.assertEqual(int(state.context.sleep.skipped[0,0,0]),0)
            self.assertFalse(bool(state.context.sleep.pending.any()))
            self.assertEqual(int(state.beliefs.conflicts[0,1,0]),engine.variant_pin_conflicts.get('p1:snorlax',0))
            pokemon=engine.snapshot().side('p1')[0]
            np.testing.assert_array_equal(cache.summary.effective[0,1,0],f.source.pin_mask('Snorlax',pokemon.candidate_variants))
