"""Authored static/B2 fixtures only; no saved data, model or engine games."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import jax
import jax.numpy as j

from gpu_engine import phase_d_live_worlds as live, phase_d_search as search, phase_d_chance as chance
from pokezero import v5_contract
from pokezero.determinization import _fixture_from_variant_payload
from pokezero.gen3_damage import gen3_stat,gen3_hp_stat
import test_gpu_engine_device_public_teacher as fixture


class WorldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu_engine import public_belief,v4_encoder
        from pokezero.dex import MoveInfo,SpeciesInfo,ShowdownDex
        from pokezero.category_vocab import CategoryVocabulary
        _,_,_,cls.tables,_=fixture.facts();cls.source=cls.tables.source
        cls.metadata=dict(vocabulary={k:list(v) for k,v in cls.source.vocabulary.items()},moves=[None]+[dict(pp=10)]*4)
        moves={n:MoveInfo(n,n,'Normal','Physical','Physical',50,100,0,False,False,False,None,{},'normal',False,pp=10) for n in ('tackle','surf','rest','protect','struggle')}
        dex=ShowdownDex(moves,{u.species.lower():SpeciesInfo(u.species.lower(),u.species,('Normal',),dict(hp=100,atk=100,**{'def':100},spa=100,spd=100,spe=100))for u in cls.source.original.universes.values()},{})
        labels={'field','stats','pokemon:self','pokemon:opponent','status:none','type:Normal'}
        labels|={'species:'+u.species.lower() for u in cls.source.original.universes.values()}
        labels|={'belief:possible_'+kind+':'+name for kind,names in (('ability',('insomnia','pressure')),('item',('oranberry','leftovers'))) for name in names}
        labels|={'move:'+n for n in moves}
        masks=dict(exact_state=True,feature_pack_last_move=True,investment_belief_narrowing=False,item_belief_narrowing=False,stats_block=True,tier2_investment=True,tier2_residuals=True,transition_token_budget=0)
        cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(cls.source.original),dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=masks)

    def request(self,owner=0):
        party=[]
        for i,u in enumerate(list(self.source.original.universes.values())[:6]):
            variant=u.variants[0]
            m=_fixture_from_variant_payload(variant.to_summary(),fallback_species=u.species,set_source=self.source.original,gender='N')
            base=self.source.original.species_metadata[u.species.lower()]['baseStats']
            mx=gen3_hp_stat(base['hp'],m.ivs['hp'],m.evs['hp'],m.level)
            stats={k:gen3_stat(base[k],m.ivs[k],m.evs[k],m.level) for k in ('atk','def','spa','spd','spe')}
            party.append(dict(ident=f'p{owner+1}: '+u.species,details=u.species+', L80',condition=f'{mx}/{mx}',stats=stats,moves=list(m.moves),baseAbility=m.ability,item=m.item,active=i==0))
        return dict(rqid=1,side=dict(id=f'p{owner+1}',pokemon=party),active=[dict(moves=[dict(id=m,pp=16,disabled=False)for m in party[0]['moves']])])

    def adapter(self,owner=0):
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        a=JaxRecordedAdapter(self.catalog,owner_side=owner);a.reset(stream_id=1,battle_id='authored')
        own=f'p{owner+1}';peer=f'p{2-owner}'
        a.feed((f'|switch|{own}a: Alpha|Alpha, L80|291/291',f'|switch|{peer}a: Beta|Beta, L80|100/100','|turn|1'))
        return a

    def test_real_b2_fold_to_16_and32_source_worlds_both_owners(self):
        for owner in (0,1):
            request=self.request(owner);a=self.adapter(owner)
            root=live.from_b2(a,request,self.source,self.metadata,address=v5_contract.DecisionAddress(1,0,owner))
            self.assertEqual(int(root.view['species'][0,0]),2)
            self.assertEqual(root.view['belief_mask'][0,0].tolist(),[3,0])
            for count in (16,32):
                result=live.sample_worlds(root,self.source,self.tables,count=count,seed=9)
                self.assertTrue(result.fully_constructed);self.assertEqual(len(set(result.seeds)),count)
                self.assertEqual([w.index for w in result.worlds],list(range(count)))
                for w in result.worlds:
                    for name in ('stats','hp','max_hp','moves','pp','item','ability'):
                        np.testing.assert_array_equal(getattr(w.state,name)[0,owner],root.own[name])
                again=live.sample_worlds(root,self.source,self.tables,count=count,seed=9)
                self.assertEqual(result.seeds,again.seeds)

    def test_missing_or_changed_own_facts_refuse(self):
        a=self.adapter()
        for mutate in ('stats','moves','baseAbility','item'):
            req=self.request();req['side']['pokemon'][0].pop(mutate)
            with self.assertRaises(ValueError):live.from_b2(a,req,self.source,self.metadata,address=v5_contract.DecisionAddress(1,0,0))
        req=self.request();req['side']['pokemon'][0]['stats']['atk']+=1
        with self.assertRaisesRegex(ValueError,'IV_FACTS'):live.from_b2(a,req,self.source,self.metadata,address=v5_contract.DecisionAddress(1,0,0))

    def test_refused_worlds_keep_all_slots_and_seeds(self):
        _,root,_,tables,_=fixture.facts()
        view=dict(root.view);view['belief_mask']=np.zeros((1,6,2),np.uint32)
        bad=replace(root,view=live.original._freeze(view))
        out=live.sample_worlds(bad,tables.source,tables,count=32,seed=99)
        self.assertFalse(out.fully_constructed);self.assertEqual(len(out.worlds),32)
        self.assertTrue(all(w.state is None and w.refusals for w in out.worlds))
        self.assertEqual(len(set(out.seeds)),32)

    def test_exact_chance_program_unchanged_and_current_pins_match(self):
        import ast
        from gpu_engine import chance_consumers
        def body(module):
            tree=ast.parse(Path(module.__file__).read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plan_for_state'),include_attributes=False)
        self.assertEqual(body(chance),body(chance_consumers))
        base=Path(chance.__file__).parent
        self.assertTrue(all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in chance.SOURCE_PINS.items()))

    def test_q_exact_mass_all_worlds_reply_denominator_and_tie_rule(self):
        legal=(True,True,*([False]*7))
        reports=[dict(replies=(0,1),probabilities=(.25,.75),pairs={(a,r):[(Fraction(1,4),-.5+a),(Fraction(3,4),.5-a)] for a in (0,1) for r in (0,1)}) for _ in range(16)]
        q,action=search.completed_q(legal,reports,requested_worlds=16)
        self.assertEqual(q[:2],(.25,-.25));self.assertEqual(action,0)
        reports[0]['pairs'][(0,0)].pop()
        with self.assertRaisesRegex(ValueError,'mass'):search.completed_q(legal,reports,requested_worlds=16)
        with self.assertRaisesRegex(ValueError,'all requested'):search.completed_q(legal,reports[:-1],requested_worlds=16)

    def test_cpu_model_search_is_rejected_before_load(self):
        with patch.object(search.jax,'devices',return_value=[SimpleNamespace(platform='cpu')]),patch.object(search.champion,'load',side_effect=AssertionError('no model')):
            with self.assertRaisesRegex(ValueError,'ONE visible GPU'):search.Search(None,expected_archive_sha256=None,catalog=None,source=None,metadata=None,engine_tables=None,observation_tables=None)


if __name__=='__main__':unittest.main()
