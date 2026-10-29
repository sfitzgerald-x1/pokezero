"""Source-confirmed identity facts preserve full identifiers and species."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class SwitchProtocolTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        import numpy as np
        from gpu_engine import public_switch_protocol as p,public_wire
        from gpu_engine.public_context_protocol import PublicAddress
        self.p=p;self.w=public_wire
        binding=np.arange(8,dtype=np.uint32)
        class Abilities:
            device=SimpleNamespace(binding=binding)
            def ability_id(self,name):
                names={'intimidate':1,'pressure':2}
                from pokezero.showdown import _normalize_identifier
                key=_normalize_identifier(name)
                if key not in names:raise ValueError('ability outside frozen public namespace')
                return names[key]
        self.abilities=Abilities()
        self.species=SimpleNamespace(names=('', 'Snorlax','Snorlax-Blue','Absol'),device=SimpleNamespace(ability_binding=binding))
        self.registry=public_wire.WireState(encounters=((public_wire.Encounter('p1a: Nick','Snorlax-Blue',PublicAddress(0,0,1)),),
            (public_wire.Encounter('p2a: Absol','Absol',PublicAddress(1,0,1)),)),active=(0,0),live=(True,True))

    def facts(self,line):return self.p.lexical_facts(line,self.registry,species_catalog=self.species,abilities=self.abilities)

    def test_full_ident_and_full_species_are_separate_lexical_facts(self):
        for raw,exact,species in (('p1a: Nick',True,0),('p1a: NICK',True,0),('p1: Nick',False,0),
                ('p1: Snorlax',False,1),('p1: Snorlax-Blue',False,2),('p2: Snorlax-Blue',False,2)):
            f=self.facts('|-ability|'+raw+'|Intimidate')
            self.assertEqual((f.exact_ident,f.species_ident,f.confirmed_ability),(exact,species,1))
            if exact:self.assertEqual((f.confirmed_side,f.confirmed_slot,f.confirmed_encounter),(0,0,1))
        # Trace names the donor's confirmed ability; the tracer is not the
        # pending ident simply because the line's target is that tracer.
        f=self.facts('|-ability|p1a: Nick|Pressure|[from] ability: Trace|[of] p2a: Absol')
        self.assertEqual((f.confirmed_side,f.confirmed_slot,f.confirmed_encounter,f.exact_ident),(1,0,1,True))
        f=self.facts('|-ability|p1a: Nick|Pressure|[from] ability: Trace')
        self.assertEqual(f.confirmed_ability,0);self.assertFalse(f.exact_ident)

    def test_arrival_and_boundary_are_source_spelling_not_registry_inference(self):
        from pokezero.belief import _is_pending_switch_boundary
        from pokezero.showdown import _public_event_from_line
        lines=('|switch|p1a: Incoming|Absol, L80|100/100','|-boost|p1a: Nick|atk|1',
            '|-unboost|p1a: Nick|atk|1','|-damage|p1a: Nick|50/100','|-heal|p1a: Nick|100/100',
            '|-status|p1a: Nick|slp','|-curestatus|p1a: Nick|slp','|-item|p1a: Nick|Leftovers',
            '|turn|1','|upkeep','|','|-sethp|p1a: Nick|50/100')
        for line in lines:
            f=self.facts(line)
            self.assertEqual(f.boundary,_is_pending_switch_boundary(_public_event_from_line(line)))
            self.assertEqual(f.arrival,line.startswith('|switch|'))
        self.assertEqual(self.registry.encounters[0][0].ident,'p1a: Nick')
        with self.assertRaisesRegex(ValueError,'private request'):self.facts('|request|{}')
        with self.assertRaisesRegex(ValueError,'outside frozen'):self.facts('|-ability|p1a: Nick|Unknown Ability')


if __name__=='__main__':unittest.main()
