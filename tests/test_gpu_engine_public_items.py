"""Original/current public item state compared with the actual belief consumer."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicItemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_context as fixtures
        fixtures.PublicContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_items as items
        from pokezero import showdown as reference
        from pokezero.belief import PublicBattleBeliefEngine
        cls.items,cls.reference,cls.Reference=items,reference,PublicBattleBeliefEngine
        cls.item_catalog=items.compile_items(cls.host,cls.vocab,
            item_names=('Choice Band','Sitrus Berry','White Herb','Lum Berry','Mystery Orb'))

    def initial_items(self,batch=1):
        mons=self.m.initialize(batch)
        for side,name in ((0,'Snorlax'),(1,'Absol')):
            mons,ok=self.m.arrive(mons,side=side,slot=0,encounter=1,species=side+1,
                source_species=self.host.species_id(name),level=80,gender=0,hp_numerator=100,hp_denominator=100,status=0)
            self.assertTrue(bool(ok.all()))
        return mons,self.items.initialize(batch,self.item_catalog.device)

    def process(self,state,mons,line):
        projected=self.items.project_line(line,items=self.item_catalog)
        self.assertFalse(projected.full_line_consumers_complete)
        self.assertEqual(projected.raw_line,line)
        kwargs={}
        if projected.trick_idents:
            actor,target=projected.trick_idents
            kwargs=dict(actor_side=int(actor[1])-1,actor_slot=0,actor_encounter=1,
                        target_side=int(target[1])-1,target_slot=0,target_encounter=1)
        state,ok=self.items.begin_event(state,mons,self.item_catalog.device,
            item_event=projected.item_event,trick=bool(projected.trick_idents),**kwargs)
        self.assertTrue(bool(ok.all()))
        for ident,operation,item in projected.facts:
            state,ok=self.items.record(state,mons,self.item_catalog.device,side=int(ident[1])-1,slot=0,
                encounter=1,item_id=item,operation=operation,item_binding=self.j.array(projected.item_binding,self.j.uint32))
            self.assertTrue(bool(ok.all()))
        return state

    def compare(self,lines,*,narrow=False):
        mons,state=self.initial_items()
        prefix=['|switch|p1a: Snorlax|Snorlax, L80|100/100',
                '|switch|p2a: Absol|Absol, L80|100/100']
        ref=self.Reference.from_events([self.reference._public_event_from_line(x) for x in prefix],
            format_id='gen3randombattle',set_source=self.source,item_belief_narrowing=narrow)
        for line in lines:
            ref.ingest_event(self.reference._public_event_from_line(line))
            state=self.process(state,mons,line)
            for side in (0,1):
                belief=ref.snapshot().side(f'p{side+1}')[0]
                for field,expected in [('revealed',belief.revealed_item),('original',belief.original_public_item),
                                       ('current',belief.current_public_item)]:
                    ident=int(getattr(state,field)[0,side,0])
                    self.assertEqual(self.item_catalog.names[ident],self.reference._normalize_identifier(expected or ''),
                                     f'{line} side{side} {field}')
                self.assertEqual(bool(state.mutated[0,side,0]),belief.item_mutated)
                self.assertEqual(bool(state.removed[0,side,0]),belief.item_removed)
                key=int(self.items.matching_item(state,item_belief_narrowing=narrow)[0,side,0])
                self.assertEqual(self.item_catalog.names[key],self.reference._normalize_identifier(ref._variant_matching_item(belief) or ''))
        return mons,state

    def test_consumption_knockoff_inline_and_unaudited_receipts_match_source(self):
        cases=[['|-heal|p1a: Snorlax|100/100|[from] item: Leftovers|[of] p2a: Absol'],
            ['|-enditem|p1a: Snorlax|Sitrus Berry|[eat]','|-item|p1a: Snorlax|Leftovers'],
            ['|-enditem|p1a: Snorlax|Choice Band|[from] move: Knock Off',
             '|-item|p1a: Snorlax|Leftovers|[from] move: Thief'],
            ['|-item|p1a: Snorlax|White Herb','|-enditem|p1a: Snorlax|White Herb'],
            ['|-enditem|p1a: Snorlax|Lum Berry|[from] move: Covet'],
            ['|-enditem|p1a: Snorlax||[from] item: Leftovers'],
            ['|-item|p1a: Snorlax|Mystery Orb|[from] move: Recycle',
             '|-enditem|p1a: Snorlax|Mystery Orb|[from] move: Trick']]
        for narrow in (False,True):
            for lines in cases:
                with self.subTest(lines=lines,narrow=narrow):self.compare(lines,narrow=narrow)

    def test_trick_pairing_credits_partner_original_and_survives_two_contiguous_item_lines(self):
        activation='|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol'
        pair=['|-item|p1a: Snorlax|Choice Band|[from] move: Trick',
              '|-item|p2a: Absol|Leftovers|[from] move: Trick']
        for narrow in (False,True):
            self.compare([activation,*pair],narrow=narrow)
            self.compare([activation,*pair,activation,*reversed(pair)],narrow=narrow)
            self.compare([activation,pair[0],'|-enditem|p2a: Absol|Choice Band|[silent]|[from] move: Trick'],narrow=narrow)

    def test_missing_or_interrupted_pairing_does_not_invent_original_assignment(self):
        activation='|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol'
        item='|-item|p1a: Snorlax|Choice Band|[from] move: Trick'
        for lines in ([item],[activation,'|move|p2a: Absol|Protect|p2a: Absol',item],
                      [activation,'|',item],[activation,'|turn|1',item]):
            _,state=self.compare(lines,narrow=True)
            self.assertFalse(bool(state.original.any()))

    def test_raw_item_tag_before_direct_declaration_and_exact_source_case_rules(self):
        line='|-item|p1a: Snorlax|Choice Band|[from] item: Leftovers'
        projection=self.items.project_line(line,items=self.item_catalog)
        self.assertEqual([x[1] for x in projection.facts],[self.items.INLINE,self.items.DIRECT])
        self.compare([line])
        for tag in ('[from] Move: Trick','[from] move: trick','[from] move: Trick'):
            self.compare([f'|-item|p1a: Snorlax|Choice Band|{tag}'])
        with self.assertRaises(ValueError):self.items.project_line('|request|{"x":"[from] item: Leftovers"}',items=self.item_catalog)

    def test_jit_binding_identity_and_unsupported_ids_refuse_without_mutation(self):
        mons,state=self.initial_items(2)
        fn=self.jax.jit(lambda x,ids:self.items.record(x,mons,self.item_catalog.device,
            side=self.j.array([0,1],self.j.int32),slot=0,encounter=1,item_id=ids,
            item_binding=self.item_catalog.device.binding,operation=self.items.REMOVE))
        output,ok=fn(state,self.j.array([self.item_catalog.item_id('Leftovers'),0],self.j.int32))
        self.assertEqual(ok.tolist(),[True,False]);self.assertTrue(bool(output.removed[0,0,0]));self.assertFalse(bool(output.removed[1].any()))
        changed=self.items.compile_items(self.host,self.vocab,item_names=('Aaa','Leftovers'))
        output,ok=self.items.record(state,mons,changed.device,side=0,slot=0,encounter=1,
            item_id=changed.item_id('Leftovers'),item_binding=changed.device.binding)
        self.assertFalse(bool(ok.any()))
        for a,b in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(output)):
            self.assertTrue(bool((a==b).all()))
        for bad in (True,.5,self.j.array(True),self.j.array(.5)):
            with self.assertRaises(ValueError):self.items.record(state,mons,self.item_catalog.device,
                side=0,slot=0,encounter=1,item_id=bad,item_binding=self.item_catalog.device.binding)
        projection=self.items.project_line('|-item|p1a: Snorlax|Leftovers',items=changed)
        output,ok=self.items.record(state,mons,self.item_catalog.device,side=0,slot=0,encounter=1,
            item_id=projection.facts[0][2],item_binding=self.j.array(projection.item_binding,self.j.uint32))
        self.assertFalse(bool(ok.any()))
        self.assertFalse(bool(output.revealed.any()))


if __name__=='__main__':unittest.main()
