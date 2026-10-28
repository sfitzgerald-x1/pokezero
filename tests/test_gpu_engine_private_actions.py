"""Every action-token cell versus the actual champion writer, including HP edges."""
import copy
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PrivateActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import private_actions as p
        from pokezero import showdown as s
        from pokezero.dex import MoveInfo,SpeciesInfo,ShowdownDex
        from pokezero.category_vocab import CategoryVocabulary
        cls.jax,cls.j,cls.p,cls.s=jax,j,p,s
        def move(name,*,kind='Normal',category='Physical',power=0,effect='',chance=0,cost=0):
            return MoveInfo(name,name,kind,category,category,power,100,0,False,False,False,None,{},'normal',False,
                effect_chance=chance,effect_label=effect,self_hp_cost=cost,pp=10)
        moves={n:move(n) for n in ('reversal','flail','eruption','waterspout','return','frustration','curse','hiddenpower')}
        moves.update(surf=move('surf',kind='Water',category='Special',power=95),
            toxic=move('toxic',category='Status',effect='tox',chance=85),
            struggle=move('struggle',power=50),hiddenpowerfighting=move('hiddenpowerfighting',kind='Fighting',power=70),
            hiddenpowerice=move('hiddenpowerice',kind='Ice',category='Special',power=70))
        species={n:SpeciesInfo(n,n,types,dict(hp=80,atk=90,**{'def':100},spa=110,spd=120,spe=130))
            for n,types in [('pikachu',('Electric',)),('gengar',('Ghost','Poison')),('snorlax',('Normal',))]}
        cls.dex=ShowdownDex(moves,species,{})
        labels={'action','action:move','action:switch','move:unknown','belief:possible_item:leftovers',
            'belief:possible_ability:static','belief:possible_ability:levitate','belief:possible_ability:immunity',
            'pokemon:self','status:none','status:par','status:fnt'}
        for name in moves:labels.add('move:'+name)
        for name in species:labels.add('species:'+name)
        for name in ('Electric','Ghost','Poison','Normal','Water','Fighting','Ice','???'):labels.add('type:'+name)
        for name in ('Physical','Special','Status'):labels.add('move_category:'+name)
        for name in ('curse','curse_setup','tox'):labels.add('move_effect:'+name)
        labels.add('move_priority:0')
        for i in range(1,6):labels.update(('species:slot:'+str(i),'switch_slot:'+str(i)))
        for i in range(1,5):labels.update(('move:slot:'+str(i),'move_slot:'+str(i)))
        cls.vocab=CategoryVocabulary(tuple(sorted(labels)))
        cls.catalog=p.compile_actions(cls.dex,cls.vocab)

    def request(self,size=6,active=0,*,moves=None,species='pikachu',hp='100/200',owner=0):
        if moves is None:moves=[{'id':'surf','pp':0,'maxpp':24},{'id':'toxic','disabled':True}]
        return {'rqid':17,'side':{'id':f'p{owner+1}','pokemon':[
            {'ident':f'p{owner+1}: same','details':f'{species}, L80','condition':hp if i==active else '120/240',
             'active':i==active,'moves':['surf','hiddenpowerice'],'ability':'Static','item':'Leftovers',
             'stats':{'atk':150,'def':160,'spa':170,'spd':180,'spe':190}}
            for i in range(size)]},'active':[{'moves':moves}]}

    def facts(self,raw,owner=0,catalog=None):
        return self.p.parse_action_facts(raw,owner_side=owner,vocabulary=self.vocab,catalog=catalog or self.catalog)

    def expected(self,raw,owner=0,dex=None):
        from types import SimpleNamespace
        dex=dex or self.catalog.dex
        team=self.s._self_team_from_request(raw,f'p{owner+1}')
        state=SimpleNamespace(request=raw,self_team=team,self_active=next((p for p in team if p.active),None),
            legal_action_mask=self.s._legal_action_mask(raw))
        nums=[[0.]*self.s.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT for _ in range(23)];cats=[['']*60 for _ in range(23)]
        self.s._encode_action_tokens(cats,nums,state,dex=dex)
        numeric=self.s._project_v4_numeric_rows(nums)[13:22]
        categories=[[self.vocab.encode(label) for label in row[:41]] for row in cats[13:22]]
        return self.j.array(numeric,self.j.float32),self.j.array(categories,self.j.int32),state.legal_action_mask

    def compare(self,raw,*,owner=0):
        facts=self.facts(raw,owner);out=self.p.project_actions(facts,self.catalog.device,observer_side=owner)
        self.assertTrue(bool(out.admitted));num,cat,legal=self.expected(raw,owner)
        self.assertTrue(bool(self.j.allclose(out.numeric,num,atol=1e-6,rtol=0)),f'numeric {raw}')
        self.assertTrue(bool(self.j.array_equal(out.categorical,cat)),f'categories {raw}')
        self.assertEqual(out.legal.tolist(),list(legal))
        return out

    def test_all_action_columns_batched_against_source_request_corpus(self):
        profiles=[[],[{'pp':0,'maxpp':24}],[{'disabled':True}],
            [{'id':'surf','pp':4,'maxpp':8},None,{'pp':3,'maxpp':0},{'disabled':True}],
            [{'id':'surf','pp':-2,'maxpp':4},{'pp':99,'maxpp':8},{},{'pp':1,'maxpp':-2}]]
        flags=[{}, {'wait':True},{'forceSwitch':[True]},{'forceSwitch':[False]},
            {'trapped':True},{'maybeTrapped':True},{'teamPreview':True}]
        facts=[];numeric=[];categories=[];legal=[]
        for size in range(1,7):
            for active in range(size):
                for moves in profiles:
                    for mode in flags:
                        raw=self.request(size,active,moves=moves)
                        if 'trapped' in mode or 'maybeTrapped' in mode:raw['active'][0].update(mode)
                        else:raw.update(mode)
                        before=copy.deepcopy(raw);facts.append(self.facts(raw));self.assertEqual(before,raw)
                        n,c,l=self.expected(raw);numeric.append(n);categories.append(c);legal.append(l)
        packet=self.jax.tree.map(lambda *xs:self.j.stack(xs),*facts)
        out=self.jax.jit(lambda x:self.p.project_actions(x,self.catalog.device,observer_side=0))(packet)
        self.assertEqual(len(facts),735);self.assertTrue(bool(out.admitted.all()))
        self.assertTrue(bool(self.j.allclose(out.numeric,self.j.stack(numeric),atol=1e-6,rtol=0)))
        self.assertTrue(bool(self.j.array_equal(out.categorical,self.j.stack(categories))))
        self.assertEqual(out.legal.tolist(),[list(x) for x in legal])

    def test_hp_lookup_exhausts_normal_gen3_integer_pairs_and_projection_edges(self):
        from pokezero.dex import resolve_move_base_power
        import numpy as np
        hp=np.asarray(self.catalog.device.hp_power)
        for denominator in range(1,hp.shape[1]):
            for numerator in range(denominator+1):
                fraction=numerator/denominator
                self.assertEqual(int(hp[0,denominator,numerator]),resolve_move_base_power(self.dex.moves['reversal'],fraction))
                self.assertEqual(int(hp[1,denominator,numerator]),resolve_move_base_power(self.dex.moves['eruption'],fraction))
        for condition in ['0 fnt','1/48','2/48','4/48','5/48','9/48','10/48','16/48','17/48','32/48','33/48','1/714','713/714','250/200',None]:
            with self.subTest(condition=condition):self.compare(self.request(hp=condition,moves=[{'id':n} for n in ('reversal','flail','eruption','waterspout')]))

    def test_hidden_power_identity_resolution_ghost_curse_and_typeless_struggle(self):
        variants=[[{'id':'hiddenpower','move':'Hidden Power Fighting 70'},{'id':'struggle'},{'id':'return'},{'id':'frustration'}],
            [{'id':'hiddenpower'},{'id':'curse'},{'id':'surf','disabled':True}],
            [{'move':'Hidden Power Fighting 70'},None,{}]]
        for species in ('pikachu','gengar'):
            for moves in variants:self.compare(self.request(moves=moves,species=species))
        raw=self.request(moves=variants[0]);out=self.compare(raw)
        self.assertEqual(int(out.categorical[0,self.s.CATEGORY_PRIMARY]),self.vocab.encode('move:hiddenpower'))
        self.assertEqual(int(out.categorical[0,self.s.CATEGORY_TYPE_1]),self.vocab.encode('type:Fighting'))
        self.assertEqual(int(out.categorical[1,self.s.CATEGORY_TYPE_1]),self.vocab.encode('type:???'))

    def test_absent_waiting_forced_switch_unknown_and_duplicate_slot_defaults(self):
        for raw in (None,{}, {'wait':True},{'forceSwitch':[True]},self.request(1),self.request(6,4),
                    self.request(moves=[{},None,{'id':'unknown'}])):
            self.compare(raw)
        raw=self.request(6,4);raw.update(forceSwitch=[True]);raw['active'][0]['trapped']=True
        self.compare(raw)
        raw=self.request(hp='0.5/1',moves=[{'id':'surf'}]);self.compare(raw)
        raw['active'][0]['moves']=[{'id':'reversal'}]
        out=self.p.project_actions(self.facts(raw),self.catalog.device,observer_side=0)
        self.assertFalse(bool(out.admitted));self.assertFalse(bool(out.numeric.any()));self.assertFalse(bool(out.categorical.any()))

    def test_namespace_owner_shape_dtype_and_nonfinite_refusals(self):
        facts=self.facts(self.request());out=self.p.project_actions(facts,self.catalog.device,observer_side=1)
        self.assertFalse(bool(out.admitted));self.assertFalse(bool(out.numeric.any()));self.assertFalse(bool(out.categorical.any()))
        changed=self.p.compile_actions(self.dex,self.vocab,hp_limit=713)
        out=self.p.project_actions(facts,changed.device,observer_side=0);self.assertFalse(bool(out.admitted))
        for mutate in (facts._replace(mechanics=facts.mechanics.at[0].set(2**31-1)),
            facts._replace(move_primary=facts.move_primary.at[0].set(self.vocab.size)),
            facts._replace(team=facts.team._replace(hp_numerator=facts.team.hp_numerator.at[0].set(self.j.nan)))):
            out=self.p.project_actions(mutate,self.catalog.device,observer_side=0)
            self.assertFalse(bool(out.admitted));self.assertFalse(bool(out.numeric.any()))
        for mutate in (facts._replace(mechanics=facts.mechanics[:,None]),
            facts._replace(actor_ghost=self.j.array(1,self.j.int32)),
            facts._replace(request=facts.request._replace(owner_side=self.j.array(True))),
            facts._replace(team=facts.team._replace(categories=facts.team.categories[None]))):
            with self.assertRaises(ValueError):self.p.project_actions(mutate,self.catalog.device,observer_side=0)
        for observer in (True,.5,self.j.zeros((1,1),self.j.int32)):
            with self.assertRaises(ValueError):self.p.project_actions(facts,self.catalog.device,observer_side=observer)

    def test_compiled_dex_snapshot_is_stable_and_both_seats_jit_match(self):
        from dataclasses import replace
        from pokezero.dex import ShowdownDex
        info=replace(self.dex.species['pikachu'],base_stats=dict(self.dex.species['pikachu'].base_stats))
        original=ShowdownDex(dict(self.dex.moves),dict(self.dex.species,pikachu=info),{})
        catalog=self.p.compile_actions(original,self.vocab)
        info.base_stats['hp']=1;original.species.clear();original.moves.clear()
        raw=self.request();facts=self.facts(raw,catalog=catalog)
        self.assertEqual(float(facts.team.base_stats[0,0]),80.)
        with self.assertRaises(TypeError):catalog.dex.species['pikachu'].base_stats['hp']=1
        packets=[self.facts(self.request(owner=side),owner=side) for side in (0,1)]
        batch=self.jax.tree.map(lambda *xs:self.j.stack(xs),*packets)
        fn=self.jax.jit(lambda x,s:self.p.project_actions(x,self.catalog.device,observer_side=s))
        out=fn(batch,self.j.array([0,1],self.j.int32));self.assertEqual(out.admitted.tolist(),[True,True])
        for i,side in enumerate((0,1)):
            num,cat,_=self.expected(self.request(owner=side),side)
            self.assertTrue(bool(self.j.allclose(out.numeric[i],num,atol=1e-6,rtol=0)))
            self.assertTrue(bool(self.j.array_equal(out.categorical[i],cat)))


if __name__=='__main__':unittest.main()
