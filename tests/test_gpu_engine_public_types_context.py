"""Live type overlays compose with canonical encounters and atomic input folds."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicTypeContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_field_context as fixtures
        fixtures.PublicFieldContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_types as t,public_team
        from pokezero.dex import ShowdownDex,SpeciesInfo
        from pokezero.category_vocab import CategoryVocabulary
        species=dict(cls.dex.species)
        species.update({key:SpeciesInfo(key,name,types,{}) for key,name,types in (
            ('castform','Castform',('Normal',)),('castformsunny','Castform-Sunny',('Fire',)),
            ('kecleon','Kecleon',('Normal',)))})
        cls.dex=ShowdownDex(cls.dex.moves,species,cls.dex.type_chart)
        cls.vocab=CategoryVocabulary(cls.vocab.tokens+('type:normal','type:fire','type:water','type:grass','type:ice'))
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,ability_names=('Pressure','Trace','Immunity','Levitate'))
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities)
        cls.field_catalog=cls.f.compile_field(cls.vocab)
        cls.t=t;cls.type_catalog=t.compile_types(cls.species_catalog,cls.dex,cls.vocab)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,type_catalog=cls.type_catalog) for n in (False,True)}
        cls.kwargs=dict(source=cls.host,species_catalog=cls.species_catalog,moves=cls.catalog,abilities=cls.abilities,
            items=cls.item_catalog,field_catalog=cls.field_catalog,type_catalog=cls.type_catalog)
        cls.prefix=('|switch|p1a: Cloud|Castform, L80|100/100','|switch|p2a: Lizard|Kecleon, L80|100/100')

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    initial=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.initial
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines
    same=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.same

    def chunks(self,projections):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[[e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding','type_binding')}))

    def compare(self,result,lines):
        from test_gpu_engine_public_types import PublicTypeTests
        check=PublicTypeTests();check.reference,check.t,check.tables,check.vocab,check.dex=self.s,self.t,self.type_catalog.device,self.vocab,self.dex
        check.compare(result.context.types,result.mon_state,lines)

    def test_typechange_forecast_and_lifecycle_match_actual_parser_across_chunks(self):
        lines=[*self.prefix,'|turn|1',*(['|']*29),'|-formechange|p1a: Cloud|Castform-Rainy',
            '|-start|p2a: Lizard|typechange|Grass / Water','|upkeep','|turn|2',
            '|faint|p1a: Cloud','|','|switch|p1a: Spare|Absol, L80|100/100',
            '|-end|p2a: Lizard|typechange']
        result,p=self.run_lines(lines);self.compare(result,lines);self.assertGreater(len(self.chunks(p)),1)
        self.assertFalse(bool(result.context.types.overridden.any()))

    def test_source_side_attribution_ignores_positionless_or_stale_nickname(self):
        lines=[*self.prefix,'|-start|p2: Other Name|typechange|Ice',
            '|-formechange|p1a: Old Name|Castform-Sunny']
        result,_=self.run_lines(lines);self.compare(result,lines)
        self.assertEqual(result.context.types.overridden.tolist(),[[True,True]])

    def test_wrong_header_payload_or_party_slot_rolls_back_every_ledger(self):
        lines=[*self.prefix,'|switch|p1a: Spare|Absol, L80|100/100',
            '|switch|p1a: Cloud|Castform, L80|100/100','|turn|1',
            '|-start|p2a: Lizard|typechange|Water','|-formechange|p1a: Cloud|Castform-Sunny']
        _,p=self.project(lines);chunk=self.chunks(p)[0];initial=self.initial()
        for bad in (chunk._replace(type_binding=None),chunk._replace(type_binding=chunk.type_binding.at[0].add(1))):
            result=self.fold(initial,bad);self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))
        index=next(n for n in range(64) if int(chunk.words[0,n,0])==self.i.Kind.TYPES)
        for changes in ({7:1},{9:self.t.ARRIVE},{10:2**31-1},{12:1},{8:99}):
            words=chunk.words
            for col,value in changes.items():words=words.at[0,index,col].set(value)
            result=self.jax.jit(lambda cur,b,m,c,ch:self.fold((cur,b,m,c),ch))(*initial,chunk._replace(words=words))
            self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))

    def test_producer_and_consumer_cannot_drop_types_on_continuation(self):
        wire,_=self.project(self.prefix)
        legacy=self.c.compile_context(self.host,self.abilities,self.catalog,items=self.item_catalog,field_catalog=self.field_catalog)
        for changes in (dict(type_catalog=None),dict(type_catalog=None,context_tables=legacy),dict(context_tables=legacy)):
            kwargs=dict(self.kwargs,context_tables=self.contexts[False]);kwargs.update(changes)
            with self.assertRaises(ValueError):self.wire.project_line(wire,'|',**kwargs)
        with self.assertRaises(ValueError):self.i.PublicEvent(self.i.Kind.TYPES,1,0,self.i.Phase.ACTION,
            target_side=0,target_slot=0,payload=(1,self.t.ARRIVE,0,0,0,0,0,0)).words()
        with self.assertRaises(ValueError):self.c.compile_context(self.host,self.abilities,self.catalog,
            items=self.item_catalog,type_catalog=self.type_catalog)


if __name__=='__main__':unittest.main()
