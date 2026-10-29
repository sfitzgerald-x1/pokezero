"""Device tendencies against actual source folds and checkpoint writers."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicTendencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_wire as fixtures
        fixtures.PublicWireTests.setUpClass.__func__(cls)
        from gpu_engine import public_tendency as t,public_team
        from pokezero import transitions
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex
        cls.t,cls.ref=t,transitions
        cls.vocab=CategoryVocabulary((*cls.vocab.tokens,'stats'))
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,ability_names=('Pressure','Trace','Immunity','Levitate'))
        moves=dict(cls.dex.moves)
        for name in ('pursuit','batonpass'):moves[name]=moves['bodyslam']
        cls.dex=ShowdownDex(moves,cls.dex.species,{})
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities)
        cls.moves=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.catalog=cls.t.compile_tendency(cls.species_catalog,cls.moves,cls.vocab);cls.tables=cls.catalog.device
        cls.step=staticmethod(cls.jax.jit(lambda s,f:cls.t.apply(s,cls.tables,f,producer_binding=cls.tables.binding)))
        cls.prefix=['|switch|p1a: Snorlax|Snorlax, L80|100/100',
                    '|switch|p2a: Absol|Absol, L80|100/100','|turn|1']

    def initial(self,batch=1,complete=True):
        return self.t.initialize(batch,self.tables,species_binding=self.tables.species_binding,complete_prefix=complete)

    def compare(self,state,lines,private_active='Snorlax'):
        from types import SimpleNamespace
        replay=self.reference.parse_showdown_replay(lines)
        for observer in (0,1):
            stats=self.ref.extract_tendency_stats(replay,perspective_slot=f'p{observer+1}')
            species=state.species[0,1-observer].tolist()
            got=self.t.project(state,self.tables,observer_side=observer,
                opponent_species=self.j.array([species],self.j.int32),
                own_active_species=self.species_catalog.species_id(private_active),expected_vocabulary=self.tables.vocabulary_binding)
            self.assertEqual(got.admitted.tolist(),[True])
            cats=[['']*60 for _ in range(23)];nums=[[0.0]*200 for _ in range(23)]
            self.reference._encode_stats_token(cats,nums,SimpleNamespace(tendency_stats=stats))
            for old,col in zip((self.reference.NUMERIC_STAT_OPP_SWITCH_COUNT,self.reference.NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES,
                self.reference.NUMERIC_STAT_BLOCKED_ON_OUR_ATTACK,self.reference.NUMERIC_STAT_PURSUIT_INTERCEPT_PREDICT,
                self.reference.NUMERIC_STAT_MY_SWITCH_TURNS,*range(self.reference.NUMERIC_STAT_WEATHER_REVEAL_OFFSET,
                    self.reference.NUMERIC_STAT_WEATHER_REVEAL_OFFSET+6)),self.t.STAT_COLUMNS):
                self.assertAlmostEqual(float(got.stats_numeric[0,col]),nums[22][old],places=7,msg=f'{observer} stat{col} {lines[-1:]}')
            self.assertEqual(int(got.stats_role[0]),self.vocab.encode(cats[22][self.reference.CATEGORY_ROLE]))
            byname={self.reference._normalize_identifier(x.species):x for x in stats.opponent_mon_tendencies}
            match=self.reference._matchup_switch_evidence(stats,[SimpleNamespace(active=True,species=private_active)])
            for slot,sid in enumerate(species):
                expected=[0.0]*200;name=self.reference._normalize_identifier(self.species_catalog.names[sid]) if sid else ''
                if name in byname:self.reference._encode_mon_tendency(expected,byname[name])
                pair=match.get(name,(0,0))
                for old,col in zip((self.reference.NUMERIC_MON_SWITCHED_BEFORE_ATTACK,self.reference.NUMERIC_MON_STAYED_AND_ATTACKED,
                    self.reference.NUMERIC_MON_TURNS_ACTIVE_TOTAL),self.t.MON_COLUMNS[:3]):
                    self.assertAlmostEqual(float(got.numeric[0,slot,col]),expected[old],places=7,msg=f'{observer} mon{col} {lines[-1:]}')
                for count,col in zip(pair,self.t.MON_COLUMNS[3:]):self.assertAlmostEqual(float(got.numeric[0,slot,col]),min(1,count/8),places=7)

    def run_prefixes(self,lines):
        state=self.initial()
        for i,line in enumerate(lines):
            state,ok=self.step(state,self.t.decode_line(line,catalog=self.catalog));self.assertTrue(bool(ok.all()),line)
            self.compare(state,lines[:i+1])
        return state

    def test_per_stay_counters_faint_occupants_and_multiple_same_turn_switches(self):
        self.run_prefixes([*self.prefix,'|move|p2a: Absol|Body Slam|p1a: Snorlax','|upkeep','|turn|2',
            '|move|p2a: Absol|Body Slam|p1a: Snorlax','|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100','|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|faint|p2a: Gardevoir','|turn|3','|switch|p2a: Absol|Absol, L80|100/100',
            '|drag|p1a: Gardevoir|Gardevoir, L80|100/100','|replace|p1a: Snorlax|Snorlax, L80|100/100'])

    def test_called_locked_charge_and_stale_baton_provenance(self):
        self.run_prefixes([*self.prefix,'|cant|p2a: Absol|slp','|move|p2a: Absol|Sleep Talk|p1a: Snorlax',
            '|move|p2a: Absol|Body Slam|p1a: Snorlax|[from]move: Sleep Talk','|upkeep','|turn|2',
            '|move|p2a: Absol|Solar Beam|p1a: Snorlax','|-prepare|p2a: Absol|Solar Beam','|upkeep','|turn|3',
            '|move|p2a: Absol|Solar Beam|p1a: Snorlax','|cant|p1a: Snorlax|recharge',
            '|move|p2a: Absol|Body Slam|p1a: Snorlax|[from]lockedmove','|upkeep','|turn|4',
            '|move|p2a: Absol|Baton Pass|p2a: Absol','|cant|p2a: Absol|Truant','|upkeep','|turn|5',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100'])

    def test_virtual_blocked_outcome_revises_before_commit_and_pursuit_scan(self):
        self.run_prefixes([*self.prefix,'|move|p1a: Snorlax|Surf|p2a: Absol',
            '|-activate|p2a: Absol|move: Protect','|-immune|p2a: Absol',
            '|-heal|p2a: Absol|100/100|[from] ability: Water Absorb','|',
            '|move|p1a: Snorlax|Surf|p2a: Absol','|-activate|p2a: Absol|Detect','|',
            '|-activate|p1a: Snorlax|move: Pursuit','|','|faint|p1a: Snorlax','|win|p2',
            '|move|p2a: Absol|Pursuit|p1a: Snorlax','|upkeep','|turn|2',
            '|move|p2a: Absol|Pursuit|p1a: Snorlax'])

    def test_weather_history_both_seats_and_private_active_matchup(self):
        lines=[*self.prefix,'|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Absol',
            '|-weather|RainDance|[upkeep]','|cant|p2a: Absol|par','|-weather|SunnyDay',
            '|','|-weather|Sandstorm','|-weather|Sandstorm|[of] p1a: Snorlax',
            '|-weather|Hail|[of] p2a: Absol','|-weather|none','|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|move|p2a: Gardevoir|Surf|p1a: Snorlax','|drag|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100']
        state=self.run_prefixes(lines);self.compare(state,lines,private_active='Gardevoir')
        got=self.t.project(state,self.tables,observer_side=0,opponent_species=state.species[:,1],
            own_active_species=0,expected_vocabulary=self.tables.vocabulary_binding,enabled=False)
        self.assertFalse(bool(got.numeric.any()));self.assertFalse(bool(got.stats_numeric.any()));self.assertFalse(bool(got.stats_attention.any()))

    def test_jit_mixed_batch_rollback_chronology_namespace_and_prefix_provenance(self):
        state=self.initial(2);fact=self.t.decode_line('|turn|1',catalog=self.catalog)
        fact=fact._replace(turn=self.j.array([1,0],self.j.int32))
        got,ok=self.step(state,fact);self.assertEqual(ok.tolist(),[True,False])
        for a,b in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(got)):self.assertTrue(bool(self.j.array_equal(a[1],b[1])))
        bad,ok=self.t.apply(state,self.tables,fact,producer_binding=self.tables.binding.at[0].add(1));self.assertFalse(bool(ok.any()))
        for a,b in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(bad)):self.assertTrue(bool(self.j.array_equal(a,b)))
        single=self.initial(complete=False)
        projection=self.t.project(single,self.tables,observer_side=0,opponent_species=single.species[:,1],
            own_active_species=0,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertFalse(bool(projection.admitted.any()))
        self.assertFalse(bool(projection.stats_attention.any()));self.assertFalse(bool(projection.stats_numeric.any()))
        self.assertFalse(bool(projection.stats_role.any()));self.assertFalse(bool(projection.numeric.any()))
        for raw in ('|turn|01','|turn','|turn|2147483648'):
            with self.assertRaises(ValueError):self.t.decode_line(raw,catalog=self.catalog)
        with self.assertRaises(ValueError):
            self.t.decode_line('|switch|p2a: absol|absol, L80|100/100',catalog=self.catalog)
        started,ok=self.step(self.initial(),self.t.decode_line('|turn|1',catalog=self.catalog))
        for raw in ('|turn|1','|turn|3'):
            refused,ok=self.step(started,self.t.decode_line(raw,catalog=self.catalog));self.assertEqual(ok.tolist(),[False])
            for a,b in zip(self.jax.tree.leaves(started),self.jax.tree.leaves(refused)):self.assertTrue(bool(self.j.array_equal(a,b)))


if __name__=='__main__':unittest.main()
