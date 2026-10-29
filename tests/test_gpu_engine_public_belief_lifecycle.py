"""Exit/cure epoch and Rest-wake inference against the actual source engine."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicBeliefLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_item_context as fixtures
        from gpu_engine import public_belief_lifecycle,public_team,public_sleep,public_field
        from pokezero.dex import ShowdownDex
        fixtures.PublicItemContextTests.setUpClass.__func__(cls)
        cls.life=public_belief_lifecycle
        dex_moves=dict(cls.dex.moves)
        for name in ('healbell','aromatherapy'):dex_moves[name]=dex_moves['rest']
        cls.dex=ShowdownDex(dex_moves,cls.dex.species,cls.dex.type_chart)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,
            ability_names=('Pressure','Trace','Immunity','Natural Cure','Early Bird','Shed Skin'))
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities)
        cls.sleep_tables=public_sleep.compile_sleep(cls.species_catalog,cls.vocab)
        cls.field_catalog=public_field.compile_field(cls.vocab)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,sleep_tables=cls.sleep_tables) for n in (False,True)}
        cls.kwargs=dict(species_catalog=cls.species_catalog,source=cls.host,moves=cls.catalog,abilities=cls.abilities,
            items=cls.item_catalog,field_catalog=cls.field_catalog,sleep_tables=cls.sleep_tables)
        cls.life_tables=cls.life.compile_lifecycle(cls.species_catalog,cls.abilities,cls.catalog,cls.sleep_tables)

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    initial=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.initial
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    def chunks(self,projections):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[[e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding','sleep_binding')}))

    def run_source(self,lines):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        source=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=self.source)
        current=self.initial();host=self.wire.WireState();life=self.life.initialize(1,self.life_tables);claims=[]
        for raw in lines:
            old=current[3];host,ps=self.project([raw],state=host)
            for chunk in self.chunks(ps):
                result=self.fold(current,chunk);self.assertEqual(result.error.tolist(),[0]);current=self.carry(result)
            parts=raw.split('|');kind=parts[1];side=0 if len(parts)>2 and parts[2].startswith('p1') else 1
            slot=int(result.mon_state.active[0,side]);enc=int(result.mon_state.encounter[0,side,slot])
            kwargs=dict(side=side,slot=slot,encounter=enc)
            operation=None
            if kind in ('switch','drag','replace'):operation=self.life.ARRIVE
            elif kind=='move':operation=self.life.MOVE;kwargs['move_id']=self.catalog.move_id(parts[3])
            elif kind=='-activate' and 'ability: Shed Skin' in raw:
                operation=self.life.SHED_SKIN_ACTIVATE;kwargs['ability_id']=self.abilities.ability_id('Shed Skin')
            elif kind=='-curestatus':
                operation=self.life.CURE
                if not parts[2].startswith(f'p{side+1}a:'):
                    event=next(e for p in ps for e in p.events if e.kind==self.i.Kind.SLEEP)
                    kwargs.update(slot=event.target_slot,encounter=event.payload[0],benched=True)
            elif kind=='-cureteam':operation=self.life.CURE_TEAM;kwargs.update(slot=-1,encounter=0)
            elif kind=='faint':operation=self.life.FAINT
            elif kind=='upkeep':operation=self.life.UPKEEP;kwargs=dict(side=-1,slot=-1,encounter=0)
            previous={p.key:p.revealed_ability for seat in ('p1','p2') for p in source.snapshot().side(seat)}
            source.ingest_event(_public_event_from_line(raw))
            if operation is not None:
                out=self.life.apply(life,old.sleep,old.abilities,result.mon_state,self.life_tables,operation=operation,
                    producer_binding=self.life_tables.binding,**kwargs)
                self.assertEqual(out.admitted.tolist(),[True]);life=out.state
                claim=int(out.ability_claim[0]);claims.append((raw,claim))
                if operation in (self.life.ARRIVE,self.life.CURE):
                    member=source.snapshot().side(f'p{side+1}')[kwargs['slot']]
                    expected=member.revealed_ability if previous.get(member.key) is None else None
                    self.assertEqual(claim,self.abilities.ability_id(expected) if expected else 0)
                else:self.assertEqual(claim,0)
                if claim:
                    update=self.a.reveal(result.context.abilities,result.mon_state,self.abilities.device,
                        side=side,slot=kwargs['slot'],encounter=kwargs['encounter'],ability_id=claim,authoritative=True)
                    self.assertTrue(bool(update.admitted.all()))
                    context=result.context._replace(abilities=update.state)
                    beliefs,ok=self.c.rebuild(context,self.c.basis(context,result.beliefs),self.host.device,self.contexts[False])
                    self.assertTrue(bool(ok.all()));result=result._replace(context=context,beliefs=beliefs);current=self.carry(result)
            self.assertEqual(life.cure_count[0].tolist(),[source._cure_all_count[x] for x in ('p1','p2')])
            for seat in (0,1):
                for index,p in enumerate(source.snapshot().side(f'p{seat+1}')):
                    self.assertEqual(bool(life.exit_status[0,seat,index]),bool(p.status_on_exit))
                    self.assertEqual(int(life.exit_cure_count[0,seat,index]),p.cure_all_count_on_exit)
                    self.assertEqual(bool(life.shed_skin[0,seat,index]),p.key in source._shed_skin_activated_this_turn)
        return life,current,claims

    def test_natural_cure_exit_epochs_called_team_cure_and_bench_cleanup(self):
        common=[*self.prefix,'|-status|p1a: Snorlax|par',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100']
        _,_,claims=self.run_source([*common,'|switch|p1a: Snorlax|Snorlax, L80|100/100'])
        self.assertEqual(claims[-1][1],self.abilities.ability_id('Natural Cure'))
        for cure in ('|move|p1a: Gardevoir|Heal Bell|p1a: Gardevoir|[from] move: Metronome',
                '|-curestatus|p1: Snorlax|par|[silent]','|-cureteam|p1a: Gardevoir'):
            _,_,claims=self.run_source([*common,cure,'|switch|p1a: Snorlax|Snorlax, L80|100/100'])
            self.assertEqual(claims[-1][1],0)

    def test_early_bird_and_borrowed_shed_skin_activation_suppression(self):
        rest=[*self.prefix,'|-status|p1a: Snorlax|slp|[from] move: Rest','|cant|p1a: Snorlax|slp']
        _,_,claims=self.run_source([*rest,'|-curestatus|p1a: Snorlax|slp'])
        self.assertEqual(claims[-1][1],self.abilities.ability_id('Early Bird'))
        _,_,claims=self.run_source([*rest,
            '|-ability|p1a: Snorlax|Pressure|[from] ability: Trace|[of] p2a: Absol',
            '|-activate|p1a: Snorlax|ability: Shed Skin','|-curestatus|p1a: Snorlax|slp'])
        self.assertEqual(claims[-1][1],0)

    def test_counter_overflow_and_namespace_controls_are_per_row_atomic(self):
        life,current,_=self.run_source(self.prefix)
        life=life._replace(cure_count=life.cure_count.at[0,0].set(2**31-1))
        kwargs=dict(operation=self.life.MOVE,side=0,slot=0,encounter=1,
            move_id=self.catalog.move_id('Heal Bell'),producer_binding=self.life_tables.binding)
        out=self.jax.jit(lambda s:self.life.apply(s,current[3].sleep,current[3].abilities,current[2],self.life_tables,**kwargs))(life)
        self.assertEqual(out.admitted.tolist(),[False]);self.assertEqual(out.state.cure_count.tolist(),life.cure_count.tolist())
        with self.assertRaises(ValueError):self.life.apply(life,current[3].sleep,current[3].abilities,current[2],
            self.life_tables,**dict(kwargs,producer_binding=self.life_tables.binding.astype(self.j.float32)))

    def test_encounter_rebinding_and_invalid_condition_refuse_only_bad_row(self):
        life,current,_=self.run_source([*self.prefix,'|-status|p1a: Snorlax|par',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100'])
        twice=lambda x:self.j.concatenate([x,x])
        life=self.jax.tree.map(twice,life);sleep=self.jax.tree.map(twice,current[3].sleep)
        abilities=self.jax.tree.map(twice,current[3].abilities);mons=self.jax.tree.map(twice,current[2])
        arrived=mons._replace(active=mons.active.at[:,0].set(0),status=mons.status.at[:,0,0].set(self.m.NONE),
            encounter=mons.encounter.at[0,0,0].set(99))
        out=self.jax.jit(lambda s:self.life.apply(s,sleep,abilities,arrived,self.life_tables,
            operation=self.life.ARRIVE,side=0,slot=0,encounter=self.j.array([99,1],self.j.int32),
            producer_binding=self.life_tables.binding))(life)
        self.assertEqual(out.admitted.tolist(),[False,True]);self.assertEqual(out.ability_claim.tolist(),[0,self.abilities.ability_id('Natural Cure')])
        for a,b in zip(self.jax.tree.leaves(life),self.jax.tree.leaves(out.state)):
            self.assertTrue(bool(self.j.array_equal(a[0],b[0])))
        bad=sleep._replace(condition_status=sleep.condition_status.at[0,0,0].set(99))
        out=self.jax.jit(lambda s:self.life.apply(s,bad,abilities,mons,self.life_tables,
            operation=self.life.CURE_TEAM,side=0,slot=-1,encounter=0,producer_binding=self.life_tables.binding))(life)
        self.assertEqual(out.admitted.tolist(),[False,True])
        for a,b in zip(self.jax.tree.leaves(life),self.jax.tree.leaves(out.state)):
            self.assertTrue(bool(self.j.array_equal(a[0],b[0])))


if __name__=='__main__':unittest.main()
