"""Actual public lifecycle events and source-summary callbacks, both seats."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicLifecycleStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataclasses
        import test_gpu_engine_public_refresh_context as fixture
        from gpu_engine import public_team,public_sleep,public_tendency,public_belief_refresh,public_belief_lifecycle
        from pokezero.dex import ShowdownDex
        fixture.PublicRefreshContextTests.setUpClass.__func__(cls)
        universe=cls.source.universes['snorlax']
        variants=tuple(dataclasses.replace(v,ability=('Immunity','Early Bird','Natural Cure')[i%3])
            for i,v in enumerate(universe.variants))
        cls.source=dataclasses.replace(cls.source,universes=dict(cls.source.universes,
            snorlax=dataclasses.replace(universe,variants=variants)))
        cls.host=cls.b.compile_source(cls.source)
        dex_moves=dict(cls.dex.moves)
        for name in ('healbell','aromatherapy'):dex_moves[name]=dex_moves['rest']
        cls.dex=ShowdownDex(dex_moves,cls.dex.species,cls.dex.type_chart)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,
            ability_names=('Pressure','Trace','Immunity','Natural Cure','Early Bird','Shed Skin'))
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities,extra_species=('Unown-Z',))
        cls.sleep_tables=public_sleep.compile_sleep(cls.species_catalog,cls.vocab)
        cls.tendency_catalog=public_tendency.compile_tendency(cls.species_catalog,cls.catalog,cls.vocab)
        cls.tables=cls.tendency_catalog.device
        cls.refresh_tables=public_belief_refresh.compile_refresh(cls.host,cls.catalog)
        cls.life=public_belief_lifecycle
        cls.lifecycle_tables=public_belief_lifecycle.compile_lifecycle(cls.species_catalog,cls.abilities,cls.catalog,cls.sleep_tables)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            sleep_tables=cls.sleep_tables,tendency_tables=cls.tables,refresh_tables=cls.refresh_tables,
            lifecycle_tables=cls.lifecycle_tables) for n in (False,True)}
        cls.kwargs=dict(species_catalog=cls.species_catalog,source=cls.host,moves=cls.catalog,abilities=cls.abilities,
            items=cls.item_catalog,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            sleep_tables=cls.sleep_tables,tendency_catalog=cls.tendency_catalog)
        cls.compiled={}
        for mode,tables in cls.contexts.items():
            def make(tables):
                return cls.jax.jit(lambda current,chunk:cls.i.fold_chunk(current[0],current[1],cls.host.device,chunk,
                    mon_state=current[2],context=current[3],context_tables=tables))
            cls.compiled[mode]=make(tables)
        from pokezero import showdown
        cls.reference_module=showdown

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    initial=__import__('test_gpu_engine_public_refresh_context').PublicRefreshContextTests.initial
    chunks=__import__('test_gpu_engine_public_refresh_context').PublicRefreshContextTests.chunks
    reference_engine=__import__('test_gpu_engine_public_refresh_context').PublicRefreshContextTests.reference_engine
    same=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.same

    def fold(self,current,chunk,narrow=False):return self.compiled[narrow](current,chunk)

    def compare(self,result,engine,calls):
        cache=result.context.refresh.controller.visible
        summary,ok=self.snapshot.read(cache,result.beliefs,self.host.device)
        self.assertTrue(bool(ok.all()))
        for side in (0,1):
            for slot,pokemon in enumerate(engine.snapshot().side(f'p{side+1}')):
                address=(0,1-side,slot);known=pokemon.candidate_set_count is not None
                # Source availability is format-global; an unknown species
                # has its own explicit reason and a zero device count.
                self.assertTrue(bool(cache.source_available[address]))
                self.assertEqual(int(summary.reason[address])==self.b.UNKNOWN_SPECIES,not known)
                self.assertEqual(int(summary.count[address]),pokemon.candidate_set_count if known else 0)
                self.assertAlmostEqual(float(summary.uncertainty[address]),pokemon.uncertainty,places=6)
                self.assertEqual(int(cache.generation[address]),calls.count(pokemon.key))
                self.assertEqual(int(result.beliefs.conflicts[address]),engine._variant_pin_conflicts.get(pokemon.key,0))
        self.assertEqual(result.context.refresh.family.tolist(),[0])
        self.assertEqual(result.context.refresh.main_seen.tolist(),[0])
        self.assertEqual(cache.source_binding.shape,(8,))

    def checked(self,lines,narrow=False):
        engine,calls=self.reference_engine([],narrow)
        current=self.initial(narrow);host=self.wire.WireState()
        for raw in lines:
            host,ps=self.project([raw],narrow,state=host)
            for chunk in self.chunks(ps):
                result=self.fold(current,chunk,narrow);self.assertEqual(result.error.tolist(),[0],raw);current=self.carry(result)
            engine.ingest_event(self.reference_module._public_event_from_line(raw))
            with self.subTest(raw=raw,narrow=narrow):self.compare(result,engine,calls)
            self.assertEqual(result.context.lifecycle.cure_count[0].tolist(),[engine._cure_all_count[x] for x in ('p1','p2')])
        return result,engine,calls,host

    def test_status_sleep_hp_cures_and_faint_callback_prefixes_both_modes(self):
        lines=[*self.prefix,'|turn|1','|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|cant|p1a: Snorlax|slp','|move|p1a: Snorlax|Sleep Talk|p2a: Absol',
            '|move|p1a: Snorlax|Surf|p2a: Absol|[from] move: Sleep Talk',
            '|-curestatus|p1a: Snorlax|slp','|-damage|p1a: Snorlax|70/100',
            '|-heal|p1a: Snorlax|80/100','|-sethp|p1a: Snorlax|75/100',
            '|cant|p2a: Absol|par','|-status|p2a: Absol|par',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100',
            '|-damage|p2a: Absol|0 fnt','|faint|p2a: Absol']
        for mode in (False,True):self.checked(lines,mode)

    def test_cure_epochs_called_move_and_bench_team_cures(self):
        common=[*self.prefix,'|-status|p1a: Snorlax|par','|switch|p1a: Gardevoir|Gardevoir, L80|100/100']
        for mode in (False,True):
            for middle in ([],['|move|p1a: Gardevoir|Heal Bell|p1a: Gardevoir|[from] move: Metronome'],
                    ['|-curestatus|p1: Snorlax|par|[silent]'],['|-cureteam|p1a: Gardevoir']):
                lines=[*common,*middle,'|switch|p1a: Snorlax|Snorlax, L80|100/100']
                self.checked(lines,mode)
                self.checked([raw.replace('p1','SIDE').replace('p2','p1').replace('SIDE','p2') for raw in lines],mode)

    def test_cosmetic_bench_redirect_skips_upsert_and_transformed_cure_move_counts(self):
        lines=['|switch|p1a: Unown|Unown-Z, L80|100/100 par',self.prefix[1],
            '|switch|p1a: Snorlax|Snorlax, L80|100/100','|-curestatus|p1: Unown|par|[silent]',
            '|switch|p1a: Unown|Unown-Z, L80|100/100']
        for mode in (False,True):self.checked(lines,mode)
        lines=[*self.prefix,'|-transform|p1a: Snorlax|p2a: Absol',
            '|move|p1a: Snorlax|Aromatherapy|p1a: Snorlax']
        for mode in (False,True):self.checked(lines,mode)

    def test_common_status_proof_survives_chunk_boundary(self):
        for count in range(64):
            lines=[*self.prefix,*(['|']*count),'|-status|p1a: Snorlax|slp|[from] move: Rest']
            _,ps=self.project(lines);events=[e for p in ps for e in p.events]
            index=next(i for i,e in enumerate(events) if e.kind==self.i.Kind.STATUS)
            if index%64==63:break
        else:self.fail('fixture did not place common status at chunk end')
        current=self.initial();pending=False
        for chunk in self.chunks(ps):
            result=self.fold(current,chunk);self.assertEqual(result.error.tolist(),[0]);current=self.carry(result)
            if int(result.context.refresh.common_seen[0]):
                pending=True;self.assertTrue(bool(result.cursor.needs_more[0]))
        self.assertTrue(pending);engine,calls=self.reference_engine(lines);self.compare(result,engine,calls)

    def test_inactive_ability_and_trace_lookup_refreshes(self):
        cases=[[*self.prefix,'|-ability|p1a: Snorlax|Pressure','|faint|p1a: Snorlax',
                '|-activate|p1a: Snorlax|ability: Pressure','|-ability|p1a: Snorlax|Shed Skin',
                '|-activate|p1a: Snorlax|ability: Shed Skin'],
            [*self.prefix,'|faint|p2a: Absol',
                '|-ability|p1a: Snorlax|Pressure|[from] ability: Trace|[of] p2a: Absol'],
            [*self.prefix,'|faint|p1a: Snorlax','|faint|p2a: Absol',
                '|-ability|p1a: Snorlax|Pressure|[from] ability: Trace|[of] p2a: Absol',
                '|-ability|p2a: Absol|Pressure|[from] ability: Trace|[of] p1a: Snorlax'],
            [*self.prefix,'|faint|p1a: Snorlax',
                '|-ability|p1a: Snorlax|Pressure|[from] ability: Trace|[of] p1a: Snorlax']]
        for mode in (False,True):
            for lines in cases:self.checked(lines,mode)
        host,_=self.project(['|switch|p1a: Nick|Snorlax, L80|100/100','|faint|p1a: Nick'])
        with self.assertRaisesRegex(ValueError,'exact encountered species'):
            self.project(['|-ability|p1a: Nick|Pressure'],state=host)

    def test_shed_skin_borrowing_and_new_ability_pin_refresh_bases(self):
        rest=[*self.prefix,'|-status|p1a: Snorlax|slp|[from] move: Rest','|cant|p1a: Snorlax|slp']
        self.checked([*rest,'|-ability|p1a: Snorlax|Pressure|[from] ability: Trace|[of] p2a: Absol',
            '|-activate|p1a: Snorlax|ability: Shed Skin','|-curestatus|p1a: Snorlax|slp'])
        for prefix,raw,expected in ((rest,'|-curestatus|p1a: Snorlax|slp',2),
                ([*self.prefix,'|-status|p1a: Snorlax|par','|switch|p1a: Gardevoir|Gardevoir, L80|100/100'],
                 '|switch|p1a: Snorlax|Snorlax, L80|100/100',1)):
            result,engine,calls,host=self.checked(prefix)
            survivor=next(v.to_summary() for v in self.host.variants[self.host.species_id('Snorlax')] if v.ability=='Immunity')
            self.assertTrue(engine.narrow_candidate_variants('p1:snorlax',[survivor]))
            mask=self.j.zeros_like(result.beliefs.species,bool).at[:,1,0].set(True)
            beliefs,_=self.b.narrow_pin(result.beliefs,self.host.pin_mask('Snorlax',[survivor]),valid=mask)
            current=(result.cursor,beliefs,result.mon_state,result.context)
            host,ps=self.project([raw],state=host)
            for chunk in self.chunks(ps):result=self.fold(current,chunk);self.assertEqual(result.error.tolist(),[0]);current=self.carry(result)
            engine.ingest_event(self.reference_module._public_event_from_line(raw));self.compare(result,engine,calls)
            self.assertEqual(int(result.beliefs.conflicts[0,1,0]),expected)

    def test_missing_common_and_shed_proofs_refuse_whole_row(self):
        from dataclasses import replace
        lines=[*self.prefix,'|-status|p1a: Snorlax|slp|[from] move: Rest','|cant|p1a: Snorlax|slp',
            '|-activate|p1a: Snorlax|ability: Shed Skin','|-curestatus|p1a: Snorlax|slp']
        _,ps=self.project(lines);events=[e for p in ps for e in p.events]
        for wanted in (self.i.Kind.STATUS,self.i.Kind.ABILITY):
            index=next(i for i,e in enumerate(events) if e.kind==wanted)
            changed=[replace(e,sequence=n+1) for n,e in enumerate(events[:index]+events[index+1:])]
            result=self.fold(self.initial(),self.chunks(ps,changed)[0])
            self.assertNotEqual(result.error.tolist(),[0]);self.same(self.initial(),self.carry(result))
        # A late valid metadata bit cannot hide a different common status.
        chunk=self.chunks(ps)[0];index=next(i for i,e in enumerate(events) if e.kind==self.i.Kind.STATUS)
        bad=chunk._replace(words=chunk.words.at[0,index,9].set(self.m.PARALYSIS))
        result=self.fold(self.initial(),bad);self.assertNotEqual(result.error.tolist(),[0]);self.same(self.initial(),self.carry(result))


if __name__=='__main__':unittest.main()
