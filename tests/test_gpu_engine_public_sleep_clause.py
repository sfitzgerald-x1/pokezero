"""Separate Sleep Clause field surfaces against actual parser/belief writers."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class SleepClauseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import public_sleep_clause as clause,public_wire
        V4EncoderTests.setUpClass()
        cls.f=V4EncoderTests();cls.c=clause;cls.w=public_wire
        cls.j,cls.jax=cls.f.j,cls.f.jax
        from dataclasses import replace
        from gpu_engine import public_belief
        from pokezero.category_vocab import CategoryVocabulary
        universes=dict(cls.f.public.source.universes);template=universes['snorlax']
        variant=replace(template.variants[0],species='Parasect',ability='Effect Spore',variant_id='effect-spore-fixture')
        universes['parasect']=replace(template,species='Parasect',variants=(variant,))
        cls.source=replace(cls.f.public.source,universes=universes)
        vocab=CategoryVocabulary(cls.f.vocab.tokens+('species:parasect','ability:effectspore','belief:possible_ability:effectspore'))
        cls.catalog=cls.f.e.compile_encoder(public_belief.compile_source(cls.source),cls.f.catalog.dex,vocab,
            checkpoint_sha256='a'*64,feature_masks=cls.f.masks)
        cls.f.catalog,cls.f.vocab=cls.catalog,vocab
        cls.tables=clause.compile_clause(cls.catalog.sleep)

    def source_compare(self,ledger,lines):
        from pokezero import showdown as source
        from dataclasses import replace
        import numpy as np
        replay=source.parse_showdown_replay(lines,battle_id='clause-source-fixture')
        for owner in (0,1):
            normalized=source.normalize_for_player(replay,player_id=f'p{owner+1}',configured_showdown_slot=f'p{owner+1}',
                format_id='gen3randombattle',set_source=self.source)
            for exact in (False,True):
                masks=replace(source.DEFAULT_OBSERVATION_FEATURE_MASKS,exact_state=exact,transition_token_budget=0)
                reference=source.observation_from_player_state(normalized,category_vocab=self.f.vocab,
                    spec=source.V4_REPLAY_OBSERVATION_SPEC,dex=self.catalog.dex,feature_masks=masks)
                got=self.jax.jit(lambda s:self.c.project(s,self.tables,observer_side=owner,
                    expected_vocabulary=self.catalog.species.device.vocabulary_binding,exact_state=exact))(ledger)
                self.assertEqual(got.admitted.tolist(),[True])
                actual=np.asarray(got.numeric[0])[list(self.c.COLUMNS)]
                want=np.asarray(reference.numeric_features[0])[list(self.c.COLUMNS)]
                np.testing.assert_array_equal(actual,want,err_msg=str((owner,exact,lines[-1])))

    def run_prefixes(self,lines):
        from gpu_engine import public_inputs,public_sleep
        from gpu_engine.public_mon_protocol import STATUS
        state=self.f.e.initialize(self.catalog,stream_ids=self.j.array([1],self.j.int32))
        ledger=self.c.initialize(1,self.tables,complete_prefix=True);host=self.w.WireState();prefix=[]
        fold=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.catalog))
        apply=self.jax.jit(lambda s,m,p,o,sd,sl,en,st,re,ab,pm:self.c.apply(s,self.tables,m,p,operation=o,
            side=sd,slot=sl,encounter=en,status=st,rest=re,ability_source=ab,parser_ident_matches=pm,producer_binding=self.tables.binding))
        for raw in lines:
            before=state.context.sleep
            host,projection=self.w.project_line(host,raw,**self.f.e.wire_kwargs(self.catalog))
            for chunk in self.f.chunks([projection]):
                state,error=fold(state,chunk);self.assertEqual(error.tolist(),[0])
            p=raw.split('|');kind=p[1] if len(p)>1 else ''
            op=event=None;status=0;rest=ability=False
            if kind in ('switch','drag','replace'):
                op=self.c.ARRIVE;event=next(e for e in projection.events if e.kind in
                    (public_inputs.Kind.SWITCH,public_inputs.Kind.DRAG,public_inputs.Kind.REPLACE,public_inputs.Kind.BATON_PASS))
            elif kind in ('-status','-curestatus'):
                op=self.c.STATUS if kind=='-status' else self.c.CURE
                code=public_sleep.STATUS if kind=='-status' else public_sleep.CURE
                event=next(e for e in projection.events if e.kind==public_inputs.Kind.SLEEP and e.payload[1]==code)
                status=STATUS[p[3]];rest=kind=='-status' and 'move: Rest' in raw
                ability=kind=='-status' and '[from] ability:' in raw
            elif kind in ('faint','-cureteam'):
                op=self.c.FAINT if kind=='faint' else self.c.TEAM
                code=public_inputs.Kind.FAINT if kind=='faint' else public_inputs.Kind.CURE_TEAM
                event=next(e for e in projection.events if e.kind==code)
            if event is not None:
                team=op==self.c.TEAM
                from pokezero import showdown as source
                parser_match=False
                if op in (self.c.CURE,self.c.FAINT):
                    registered=host.encounters[event.target_side][event.target_slot].ident
                    parser_match=(source._normalize_identifier(source._species_from_ident(p[2]))==
                        source._normalize_identifier(source._species_from_ident(registered)))
                ledger,ok=apply(ledger,state.mons,before,self.j.array(op,self.j.int32),self.j.array(event.target_side,self.j.int32),
                    self.j.array(-1 if team else event.target_slot,self.j.int32),self.j.array(0 if team else event.payload[0],self.j.int32),
                    self.j.array(status,self.j.int32),self.j.array(rest),self.j.array(ability),self.j.array(parser_match))
                self.assertEqual(ok.tolist(),[True])
            prefix.append(raw);self.source_compare(ledger,prefix)
        return ledger,state

    def test_actual_prefixes_keep_ability_sleep_and_two_clocks_distinct(self):
        lines=[*self.f.public.prefix,'|turn|1','|-status|p2a: Absol|slp|[from] ability: Effect Spore',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100','|-curestatus|p2: Absol|par|[silent]',
            '|-cureteam|p2a: Gardevoir','|-status|p2a: Gardevoir|slp|[from] move: Rest',
            '|-curestatus|p2a: Gardevoir|slp','|-status|p1a: Snorlax|slp',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100','|-curestatus|p1: Snorlax|slp|[silent]']
        self.run_prefixes(lines)

    def test_literal_faint_team_gate_and_natural_cure_reentry_preserve_source(self):
        self.run_prefixes([*self.f.public.prefix,'|-status|p1a: Snorlax|slp',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100','|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|-damage|p1a: Snorlax|0 fnt','|-cureteam|p1a: Snorlax','|faint|p1a: Snorlax'])


    def test_benched_species_alias_clears_belief_but_preserves_raw_nickname_victim(self):
        self.run_prefixes(['|switch|p1a: Sleepy|Snorlax, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100','|-status|p1a: Sleepy|slp',
            '|switch|p1a: Fairy|Gardevoir, L80|100/100','|-curestatus|p1: Snorlax|slp|[silent]',
            '|-cureteam|p1a: Fairy'])

    def test_namespace_identity_partial_prefix_and_mixed_row_refusal(self):
        import numpy as np
        ledger,state=self.run_prefixes(self.f.public.prefix)
        double=lambda value:self.j.concatenate((value,value),axis=0)
        ledger=self.jax.tree.map(double,ledger);mons=self.jax.tree.map(double,state.mons)
        prior=self.jax.tree.map(double,state.context.sleep)
        ledger=ledger._replace(binding=ledger.binding.at[0,0].set(ledger.binding[0,0]^self.j.uint32(1)))
        step=self.jax.jit(lambda s,m,p,o,st:self.c.apply(s,self.tables,m,p,operation=o,
            side=1,slot=0,encounter=1,status=st,parser_ident_matches=(o==self.c.CURE),producer_binding=self.tables.binding))
        result,ok=step(ledger,mons,prior,self.j.array([self.c.STATUS,self.c.STATUS],self.j.int32),self.j.array([1,1],self.j.int32))
        self.assertEqual(ok.tolist(),[False,True])
        for a,b in zip(self.jax.tree.leaves(ledger),self.jax.tree.leaves(result)):
            np.testing.assert_array_equal(a[0],b[0])
        self.assertEqual(result.holder[1].tolist(),[0,-1]);self.assertTrue(bool(result.victims[1,1,0]))
        out=self.c.project(result,self.tables,observer_side=self.j.array([0,0],self.j.int32),
            expected_vocabulary=self.catalog.species.device.vocabulary_binding)
        self.assertEqual(out.admitted.tolist(),[False,True]);self.assertFalse(bool(out.numeric[0].any()))
        good=ledger._replace(binding=self.j.broadcast_to(self.tables.binding,(2,8)))
        for operation,status in (([self.c.CURE,self.c.STATUS],[0,1]),([99,self.c.STATUS],[1,1])):
            _,ok=step(good,mons,prior,self.j.array(operation,self.j.int32),self.j.array(status,self.j.int32))
            self.assertEqual(ok.tolist(),[False,True])
        stale=good._replace(encounter=good.encounter.at[0,0,0].set(9))
        _,ok=step(stale,mons,prior,self.j.array([self.c.STATUS]*2,self.j.int32),self.j.array([1]*2,self.j.int32))
        self.assertEqual(ok.tolist(),[False,True])
        with self.assertRaisesRegex(ValueError,'shape/dtype'):
            step(good._replace(holder=good.holder.astype(self.j.float32)),mons,prior,
                self.j.array([self.c.STATUS]*2,self.j.int32),self.j.array([1]*2,self.j.int32))
        partial=self.c.initialize(2,self.tables,complete_prefix=False)
        patch=self.c.project(partial,self.tables,observer_side=0,expected_vocabulary=self.catalog.species.device.vocabulary_binding)
        self.assertEqual(patch.admitted.tolist(),[False,False]);self.assertFalse(bool(patch.numeric.any()))


if __name__=='__main__':unittest.main()
