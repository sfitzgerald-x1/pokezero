"""Pending Intimidate resolution against actual Python source predicates."""
import importlib.util
import unittest
from typing import NamedTuple


class Mons(NamedTuple):
    encounter: object
    source_species: object
    species: object
    active: object
    species_binding: object


class Abilities(NamedTuple):
    original: object
    binding: object
    species_binding: object


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class SwitchEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        import jax
        import jax.numpy as j
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import public_switch_evidence as x,public_belief,public_abilities,public_team
        from pokezero.category_vocab import CategoryVocabulary
        V4EncoderTests.setUpClass();f=V4EncoderTests();raw=f.public.source;universes=dict(raw.universes)
        for species,names in (('snorlax',('Intimidate','Pressure')),('absol',('Pressure','Clear Body','Hyper Cutter','White Smoke'))):
            original=universes[species]
            universes[species]=replace(original,variants=tuple(replace(original.variants[0],variant_id=species+str(i),ability=n)
                for i,n in enumerate(names)))
        cls.raw=replace(raw,universes=universes);cls.source=public_belief.compile_source(cls.raw)
        labels=set(f.vocab.tokens)
        from pokezero.dex import normalize_id
        for n in ('Intimidate','Pressure','Clear Body','Hyper Cutter','White Smoke'):
            labels|={'ability:'+normalize_id(n),'belief:possible_ability:'+normalize_id(n)}
        vocab=CategoryVocabulary(tuple(sorted(labels)))
        cls.abilities=public_abilities.compile_abilities(cls.source,vocab)
        dex=replace(f.catalog.dex,species=dict(f.catalog.dex.species,snorlaxblue=replace(f.catalog.dex.species['snorlax'],name='Snorlax-Blue')))
        cls.species=public_team.compile_species(cls.source,dex,vocab,cls.abilities)
        cls.tables=x.compile_switch_evidence(cls.source,cls.species,cls.abilities)
        cls.j,cls.jax,cls.x=j,jax,x
        cls.queue=jax.jit(lambda s,m,a,e,l:x.queue_arrival(s,cls.tables,m,a,cls.source.device,e,l,
            side=0,slot=0,encounter=1,producer_binding=cls.tables.binding))
        cls.resolve=jax.jit(lambda s,m,a,e,l,arrival,boundary,claim,cs,ct,ce,literal,identity:
            x.resolve(s,cls.tables,m,a,cls.source.device,e,l,arrival=arrival,boundary=boundary,
                confirmed_ability=claim,confirmed_side=cs,confirmed_slot=ct,confirmed_encounter=ce,
                exact_ident=literal,species_ident=identity,producer_binding=cls.tables.binding))

    def snapshot(self,engine):
        import numpy as np
        enc=np.zeros((1,2,6),np.int32);species=np.zeros_like(enc);original=np.zeros_like(enc)
        public_species=np.zeros_like(enc)
        active=np.full((1,2),-1,np.int32);live=np.zeros((1,2),bool);effective=np.zeros((1,2,6,2),np.uint32)
        for side in (0,1):
            for slot,pokemon in enumerate(engine.snapshot().side(f'p{side+1}')):
                enc[0,side,slot]=slot+1;species[0,side,slot]=self.source.species_id(pokemon.species)
                public_species[0,side,slot]=self.species.species_id(pokemon.species)
                if pokemon.revealed_ability:original[0,side,slot]=self.abilities.ability_id(pokemon.revealed_ability)
                if pokemon.active:active[0,side]=slot;live[0,side]=True
                effective[0,1-side,slot]=self.source.pin_mask(pokemon.species,pokemon.candidate_variants)
        binding=self.j.broadcast_to(self.species.device.binding,(1,8))
        return Mons(*(self.j.asarray(v) for v in (enc,species,public_species,active)),binding),Abilities(self.j.asarray(original),
            self.j.broadcast_to(self.abilities.device.binding,(1,8)),binding),self.j.asarray(effective),self.j.asarray(live)

    def engine(self,lines):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        engine=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=self.raw)
        for line in lines:engine.ingest_event(_public_event_from_line(line))
        return engine

    def test_queue_negative_blockers_and_repeated_source_checkpoint(self):
        prefix=['|switch|p1a: Nick|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100']
        for known in ('','Pressure','Clear Body','Hyper Cutter','White Smoke'):
            lines=prefix+(['|-ability|p2a: Absol|'+known] if known else [])
            engine=self.engine(lines);m,a,e,l=self.snapshot(engine)
            pending,ok=type(self).queue(self.x.initialize(1,self.tables),m,a,e,l)
            self.assertTrue(bool(ok.all()));self.assertEqual(pending.side.tolist(),[0])
            record=engine.snapshot().side('p1')[0]
            self.assertEqual(engine._can_queue_intimidate_non_trigger(record),True)
            expected=engine._can_rule_out_intimidate(record)
            result=type(self).resolve(pending,m,a,e,l,False,True,0,-1,-1,0,False,0)
            self.assertTrue(bool(result.admitted.all()));self.assertEqual(result.exclude.tolist(),[expected])
            self.assertEqual(result.checkpoint.tolist(),[expected]);self.assertEqual(result.state.side.tolist(),[-1])
            self.assertEqual(bool(result.state.ruled_out[0,0,0]),expected)
            # Repeated source fallback exposes both abilities again. The
            # ledger bit stays set but the source checkpoint still occurs.
            prior=pending._replace(ruled_out=pending.ruled_out.at[0,0,0].set(True))
            again=type(self).resolve(prior,m,a,e,l,False,True,0,-1,-1,0,False,0)
            self.assertEqual(again.checkpoint.tolist(),[expected])
            self.assertTrue(bool(again.state.ruled_out[0,0,0]))

    def test_claim_matching_boundary_and_arrival_order(self):
        from pokezero.belief import _PendingSwitch,_ident_matches_pending,_is_pending_switch_boundary
        from pokezero.showdown import _public_event_from_line
        engine=self.engine(['|switch|p1a: Nick|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100',
            '|-ability|p2a: Absol|Pressure'])
        m,a,e,l=self.snapshot(engine);pending,_=type(self).queue(self.x.initialize(1,self.tables),m,a,e,l)
        iid=self.abilities.ability_id('Intimidate');pid=self.abilities.ability_id('Pressure');sid=self.species.species_id('Snorlax')
        source=_PendingSwitch('p1','p1a: Nick','Snorlax')
        for raw,side,slot,enc,exact,identity in (('p1a: Nick',0,0,1,True,0),('p1: Nick',0,0,1,False,0),
                ('p1: Snorlax',0,0,1,False,sid),('p2a: Absol',1,0,1,True,0)):
            result=type(self).resolve(pending,m,a,e,l,False,True,iid,side,slot,enc,exact,identity)
            expected=_ident_matches_pending(raw,source)
            self.assertEqual(result.reveal.tolist(),[expected]);self.assertEqual(result.exclude.tolist(),[False])
            self.assertEqual(result.state.side.tolist(),[-1 if expected else 0])
        missing=pending._replace(has_ident=self.j.array([False]))
        fallback=_PendingSwitch('p1',None,'Snorlax')
        for raw,side in (('p1: Snorlax',0),('p2: Snorlax',1)):
            result=type(self).resolve(missing,m,a,e,l,False,True,iid,side,0,1,False,sid)
            self.assertEqual(result.reveal.tolist(),[_ident_matches_pending(raw,fallback)])
        # Any confirmed ability suppresses boundary flushing when no match.
        result=type(self).resolve(pending,m,a,e,l,False,True,pid,1,0,1,True,0)
        self.assertEqual(result.state.side.tolist(),[0]);self.assertFalse(bool(result.checkpoint[0]))
        # Another arrival flushes before the new occupant's effects even if
        # the incoming line also has a confirmed-ability annotation.
        result=type(self).resolve(pending,m,a,e,l,True,True,pid,1,0,1,True,0)
        self.assertEqual(result.exclude.tolist(),[True]);self.assertEqual(result.state.side.tolist(),[-1])
        for raw in ('|-boost|p1a: Nick|atk|1','|-unboost|p2a: Absol|atk|1','|-damage|p2a: Absol|50/100',
                '|-heal|p2a: Absol|100/100','|-status|p2a: Absol|slp','|-curestatus|p2a: Absol|slp',
                '|-item|p2a: Absol|Leftovers','|turn|1','|','|faint|p2a: Absol','|upkeep'):
            arrival,boundary=self.x.lexical_control(raw)
            self.assertEqual(boundary,_is_pending_switch_boundary(_public_event_from_line(raw)))
            result=type(self).resolve(pending,m,a,e,l,arrival,boundary,0,-1,-1,0,False,0)
            self.assertEqual(result.exclude.tolist(),[boundary])

    def test_missing_ident_uses_full_public_forme_not_source_pool(self):
        from pokezero.belief import _PendingSwitch,_ident_matches_pending
        engine=self.engine(['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100'])
        m,a,e,l=self.snapshot(engine)
        forme=self.species.species_id('Snorlax-Blue');base=self.species.species_id('Snorlax')
        m=m._replace(species=m.species.at[0,0,0].set(forme))
        pending,_=type(self).queue(self.x.initialize(1,self.tables),m,a,e,l)
        pending=pending._replace(has_ident=self.j.array([False]))
        iid=self.abilities.ability_id('Intimidate');source=_PendingSwitch('p1',None,'Snorlax-Blue')
        for raw,identity in (('p1: Snorlax',base),('p1: Snorlax-Blue',forme)):
            result=type(self).resolve(pending,m,a,e,l,False,True,iid,0,0,1,False,identity)
            self.assertTrue(bool(result.admitted.all()))
            self.assertEqual(result.reveal.tolist(),[_ident_matches_pending(raw,source)])

    def test_faint_missing_active_and_invalid_namespace_refuse(self):
        engine=self.engine(['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100',
            '|-ability|p2a: Absol|Pressure','|faint|p2a: Absol'])
        m,a,e,l=self.snapshot(engine);pending,_=type(self).queue(self.x.initialize(1,self.tables),m,a,e,l)
        result=type(self).resolve(pending,m,a,e,l,False,True,0,-1,-1,0,False,0)
        self.assertEqual(result.exclude.tolist(),[False]);self.assertEqual(result.state.side.tolist(),[-1])
        bad=pending._replace(binding=pending.binding.at[0,0].set(pending.binding[0,0]^self.j.uint32(1)))
        result=type(self).resolve(bad,m,a,e,l,False,True,0,-1,-1,0,False,0)
        self.assertEqual(result.admitted.tolist(),[False]);self.assertEqual(result.state.side.tolist(),bad.side.tolist())
        with self.assertRaisesRegex(ValueError,'shape/dtype'):
            type(self).resolve(pending,m,a,e.astype(self.j.int32),l,False,True,0,-1,-1,0,False,0)
        result=type(self).resolve(pending,m,a,e,l,False,True,len(self.abilities.names),-1,-1,0,False,0)
        self.assertEqual(result.admitted.tolist(),[False])


if __name__=='__main__':unittest.main()
