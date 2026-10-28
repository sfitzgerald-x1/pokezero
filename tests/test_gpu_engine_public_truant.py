"""Public Truant phases against the actual parser, including unknown Trace."""
import importlib.util
import unittest


def configure_truant(cls):
    from gpu_engine import public_active,public_team
    from pokezero.dex import ShowdownDex,SpeciesInfo
    from pokezero.category_vocab import CategoryVocabulary
    cls.active=public_active
    species=dict(cls.dex.species)
    for name,stats in [('Slaking',(150,160,100,95,65,100)),('Slakoth',(60,60,60,35,35,30))]:
        species[name.lower()]=SpeciesInfo(name.lower(),name,('Normal',),dict(zip(('hp','atk','def','spa','spd','spe'),stats)))
    cls.dex=ShowdownDex(cls.dex.moves,species,cls.dex.type_chart)
    cls.vocab=CategoryVocabulary(tuple(dict.fromkeys((*cls.vocab.tokens,'species:slaking','species:slakoth','ability:truant'))),
        oov_buckets=cls.vocab.oov_buckets,aliases=cls.vocab.aliases)
    cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,ability_names=('Truant','Immunity','Trace','Levitate','Pressure'))
    cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities)
    cls.active_catalog=public_active.compile_active(cls.vocab,species_catalog=cls.species_catalog)


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicTruantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_team as fixtures
        fixtures.PublicTeamTests.setUpClass.__func__(cls)
        configure_truant(cls)
        cls.tables=cls.active_catalog.device;cls.s=cls.reference

    def initial(self,batch=1):
        binding=self.species_catalog.device.binding
        return self.active.initialize(batch,self.tables,species_binding=binding),self.m.initialize(batch,species_binding=binding)

    def fact(self,state,mons,op,side=-1,value=0):
        enc=0 if side<0 else int(mons.encounter[0,side,int(mons.active[0,side])])
        out,ok=self.active.apply(state,self.tables,mons,operation=op,side=side,encounter=enc,value=value,
            producer_binding=self.tables.binding)
        self.assertTrue(bool(ok.all()));return out

    def consume(self,state,mons,line,registry):
        p=line.split('|');kind=p[1];slot=self.s._slot_from_ident(p[2]) if len(p)>2 else None
        side=int(slot[1])-1 if slot in ('p1','p2') else -1
        if kind in ('switch','drag','replace'):
            entries=registry[side];index=entries.setdefault(p[2],len(entries))
            name=p[3].split(',')[0].strip();pub=self.species_catalog.species_id(name)
            mons,ok=self.m.arrive(mons,side=side,slot=index,encounter=index+1,species=pub,
                source_species=self.host.species_id(name),level=80,gender=0,hp_numerator=100,hp_denominator=100,status=0)
            self.assertTrue(bool(ok.all()));state=self.fact(state,mons,self.active.ARRIVE,side)
        elif kind=='faint':state=self.fact(state,mons,self.active.FAINT,side)
        elif line=='|upkeep':state=self.fact(state,mons,self.active.UPKEEP)
        for op,seat,value in self.active.project_line(line,catalog=self.active_catalog).facts:
            state=self.fact(state,mons,op,seat,value)
        return state,mons

    def compare(self,state,mons,lines):
        replay=self.s.parse_showdown_replay(lines,battle_id='truant-public-fixture')
        self.assertEqual(bool(state.post_upkeep[0]),replay.post_upkeep_window)
        for side in (0,1):
            key=f'p{side+1}';phase=replay.truant_phase[key]
            self.assertEqual(bool(state.truant_known[0,side]),phase is not None)
            self.assertEqual(bool(state.truant_loaf[0,side]),phase is True)
            self.assertEqual(bool(state.truant_skip_flip[0,side]),key in replay.truant_skip_next_flip)
            self.assertEqual(bool(state.traced_truant[0,side]),replay.traced_ability.get(key)=='truant')
        patch=self.active.project(state,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertEqual(patch.admitted.tolist(),[True]);self.assertEqual(int(patch.numeric_columns.sum()),8)
        for side in (0,1):
            nums=[0.]*self.s.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT;cats=[None]*51
            self.s._encode_active_feature_pack(cats,nums,truant_loaf=replay.truant_phase[f'p{side+1}'] is True,
                last_used_move=None,arrived_by_baton_pass=False,choice_locked=False,item_swapped=False,
                traced_ability=None,last_damage_dealt=0,last_damage_taken=0)
            self.assertEqual(float(patch.numeric[0,side,self.active.TRUANT_COLUMN]),nums[self.s.NUMERIC_TRUANT_LOAF])

    def prefixes(self,events):
        state,mons=self.initial();lines=[];registry=({}, {})
        for line in ['|switch|p1a: Slaking|Slaking, L80|100/100','|switch|p2a: Gardevoir|Gardevoir, L80|100/100']+events:
            state,mons=self.consume(state,mons,line,registry);lines.append(line)
            if len(lines)>=2:self.compare(state,mons,lines)
        return state,mons,lines

    def test_native_lead_turn_flip_and_primary_called_anchor_match_source(self):
        self.prefixes(['|turn|1','|turn|2','|move|p1a: Slaking|Body Slam|p2a: Gardevoir|[from] move: Sleep Talk',
            '|move|p1a: Slaking|Body Slam|p2a: Gardevoir','|turn|3','|cant|p1a: Slaking|ability: Truant',
            '|turn|4','|cant|p1a: Slaking|slp','|cant|p1a: Slaking|ability: truant','|turn|5'])

    def test_action_switch_and_post_upkeep_replacement_have_different_first_phases(self):
        for residual in (False,True):
            events=['|turn|1','|switch|p1a: Snorlax|Snorlax, L80|100/100']
            if residual:events+=['|upkeep','|faint|p1a: Snorlax']
            events+=['|switch|p1a: Slakoth|Slakoth, L80|100/100','|turn|2']
            state,_,_=self.prefixes(events)
            self.assertEqual(bool(state.truant_loaf[0,0]),residual)

    def test_traced_truant_is_unknown_until_public_anchor_and_reacquisition_resets(self):
        self.prefixes(['|turn|1','|-ability|p2a: Gardevoir|Truant|[from] ability: Trace|[of] p1a: Slaking',
            '|upkeep','|turn|2','|move|p2a: Gardevoir|Body Slam|p1a: Slaking|[from] move: Sleep Talk',
            '|cant|p2a: Gardevoir|slp','|cant|p2a: Gardevoir|ability: Truant','|turn|3',
            '|-ability|p2a: Gardevoir|Truant|[from] ability: Trace|[of] p1a: Slaking','|turn|4',
            '|move|p2a: Gardevoir|Body Slam|p1a: Slaking','|-ability|p2a: Gardevoir|Pressure|[from] ability: Trace',
            '|turn|5','|switch|p2a: Absol|Absol, L80|100/100'])

    def test_faint_preserves_phase_and_unknown_native_trace_keeps_skip_until_turn(self):
        self.prefixes(['|turn|1','|upkeep','|switch|p1a: Slakoth|Slakoth, L80|100/100',
            '|-ability|p1a: Slakoth|Pressure|[from] ability: Trace','|faint|p1a: Slakoth',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100','|turn|2',
            '|switch|p1a: Slaking|Slaking, L80|100/100','|cant|p1a: Slaking|ability: Truant','|faint|p1a: Slaking','|turn|3'])

    def test_jit_known_unknown_namespace_and_disabled_mode_admission(self):
        state,mons,_=self.prefixes(['|turn|1'])
        state=self.jax.tree.map(lambda x:self.j.repeat(x,2,axis=0),state);mons=self.jax.tree.map(lambda x:self.j.repeat(x,2,axis=0),mons)
        out,ok=self.jax.jit(lambda x:self.active.apply(x,self.tables,mons,operation=self.active.TRUANT_TRACE,
            side=self.j.array([0,1],self.j.int32),encounter=self.j.array([99,1],self.j.int32),value=1,
            producer_binding=self.tables.binding))(state)
        self.assertEqual(ok.tolist(),[False,True]);self.assertFalse(bool(out.truant_known[1,1]))
        for x,y in zip(out,state):self.assertTrue(bool(self.j.array_equal(x[0],y[0])))
        for bad in (state._replace(truant_loaf=state.truant_loaf.at[:,1].set(True)),
                    state._replace(truant_known=state.truant_known.at[:,1].set(True)),
                    state._replace(truant_skip_flip=state.truant_skip_flip.at[:,0].set(True)),
                    state._replace(species_binding=state.species_binding.at[:,0].add(1))):
            patch=self.active.project(bad,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
            self.assertFalse(bool(patch.admitted.any()));self.assertFalse(bool(patch.numeric.any()))
        with self.assertRaises(ValueError):self.active.initialize(1,self.tables,species_binding=self.tables.public_species_binding.at[0].add(1))
        with self.assertRaises(ValueError):self.active.project(state._replace(post_upkeep=state.post_upkeep[:,None]),self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        legacy=self.active.compile_active(self.vocab)
        old=self.active.initialize(1,legacy.device,species_binding=self.tables.public_species_binding)
        blank=self.m.initialize(1,species_binding=self.tables.public_species_binding)
        out,ok=self.active.apply(old,legacy.device,blank,operation=self.active.UPKEEP,producer_binding=legacy.device.binding)
        self.assertFalse(bool(ok.any()));self.assertFalse(bool(out.post_upkeep.any()))
        patch=self.active.project(old,legacy.device,blank,expected_vocabulary=legacy.device.vocabulary_binding)
        self.assertFalse(bool(patch.numeric_columns[self.active.TRUANT_COLUMN]))


if __name__=='__main__':unittest.main()
