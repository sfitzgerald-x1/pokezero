"""Opt-in CPU encoder transport parity, not a search cost or strength result."""
import json
import unittest
from dataclasses import replace

from tests import test_leaf_encoder as fixture


@unittest.skipUnless(fixture._wheel_has('LeafEncoder'),'requires actual native encoder')
class BulkLeafPackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_world import EngineWorldUnsupported,battle_spec_from_payload
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.poke_engine_adapter import build_poke_engine_state
        if not DEFAULT_SHOWDOWN_ROOT.exists():raise unittest.SkipTest('requires pinned Dex')
        cls.native=fixture.pokezero_search
        if not hasattr(cls.native.LeafEncoder,'encode_leaves_packed'):
            raise AssertionError('installed encoder lacks the bulk API under test')
        tables=fixture._tables_json()
        if tables is None:raise unittest.SkipTest('requires encoder tables')
        dex=load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
        corpus,folds=fixture._sample_rows_with_folds()
        games={g.record.battle_id:g for g in corpus.games};cls.roots=[]
        for i,row in enumerate(corpus.decision_rows):
            game=games[row.battle_id]
            packed={s:(game.record.true_teams.get(s) or {}).get('packed') for s in ('p1','p2')}
            if not all(packed.values()):continue
            try:
                world=battle_spec_from_payload(row.public_materialization,
                    BattleStartOverride(player_teams=packed),dex=dex,
                    approximate_sleep_turns=True,approximate_substitute_health=True)
            except EngineWorldUnsupported:continue
            state=build_poke_engine_state(world.spec).to_string()
            context=dict(p1=list(world.party_species['p1']),p2=list(world.party_species['p2']),
                         turn=int(row.public_materialization.get('turn') or 0))
            encoder=cls.native.LeafEncoder(tables,fixture._row_inputs(row),json.dumps(context),state)
            party=list(world.spec.side_one.pokemon);active=world.spec.side_one.active_index
            party[active]=replace(party[active],hp=max(1,party[active].hp-1))
            changed=build_poke_engine_state(replace(world.spec,side_one=replace(
                world.spec.side_one,pokemon=tuple(party)))).to_string()
            cls.roots.append((row.player_id,encoder,state,changed,folds[i],context['turn']))
        if {r[0] for r in cls.roots}!={'p1','p2'}:
            raise AssertionError('committed sample must exercise both encoder perspectives')

    def rows(self,root):
        _,encoder,state,changed,payload,turn=root
        self.assertNotEqual(changed,state)
        folds=[self.native.FoldState.from_payload(payload) for _ in range(5)]
        folds[3].advance_in_place([f'|turn|{turn+1}'])
        folds[4]=folds[0]  # Multiple rows may borrow the SAME immutable fold.
        states=[state,changed,state,state,state]
        turns=[turn,turn,turn+1,turn+1,turn]
        lines=[None,[],[f'|turn|{turn+1}'],[f'|turn|{turn+1}'],None]
        return encoder,states,folds,turns,lines

    def serial(self,args,authoritative=False):
        encoder,states,folds,turns,lines=args
        return [encoder.encode_leaf(s,f,t,line,authoritative)
                for s,f,t,line in zip(states,folds,turns,lines)]

    def bulk(self,args,workers=2,authoritative=False):
        encoder,*rows=args
        return encoder.encode_leaves_packed(*rows,workers=workers,engine_authoritative=authoritative)

    def assert_rows(self,packed,serial):
        self.assertEqual(packed['batch_rows'],len(serial))
        self.assertEqual(set(packed),set(serial[0])|{'batch_rows'})
        for name in serial[0]:
            self.assertIs(type(packed[name]),bytes)
            self.assertEqual(packed[name],b''.join(r[name] for r in serial),name)

    def test_all_five_arrays_both_perspectives_workers_and_authoritative_modes(self):
        for root in self.roots:
            args=self.rows(root)
            before=[f.to_payload() for f in args[2]]
            for authoritative in (False,True):
                serial=self.serial(args,authoritative)
                self.assertNotEqual(serial[0],serial[1],'HP mutation must reach the arrays')
                self.assertNotEqual(serial[0],serial[2],'turn mutation must reach the arrays')
                self.assertNotEqual(serial[0],serial[3],'advanced history must reach the arrays')
                self.assertIs(args[2][0],args[2][4])
                for workers in (1,2,8,32):
                    self.assert_rows(self.bulk(args,workers,authoritative),serial)
            self.assertEqual(before,[f.to_payload() for f in args[2]],'bulk mutated folds')

    def test_input_permutation_duplicates_and_shards_keep_order(self):
        for root in self.roots:
            args=self.rows(root);serial=self.serial(args)
            order=[2,0,4,1,3]
            permuted=[args[0],*[[column[i] for i in order] for column in args[1:]]]
            self.assert_rows(self.bulk(permuted),[serial[i] for i in order])
            full=self.bulk(args)
            pieces=[self.bulk([args[0],*[column[a:b] for column in args[1:]]])
                    for a,b in ((0,2),(2,5))]
            for key in serial[0]:self.assertEqual(full[key],b''.join(p[key] for p in pieces))

    def test_alignment_size_and_worker_bounds_refuse(self):
        args=self.rows(self.roots[0])
        for workers in (0,33):
            with self.assertRaises(ValueError):self.bulk(args,workers)
        with self.assertRaises(ValueError):self.bulk([args[0],[],[],[],[]])
        for column in range(1,5):
            bad=[args[0],*[list(c) for c in args[1:]]];bad[column].pop()
            with self.assertRaises(ValueError):self.bulk(bad)
        oversized=[args[0],*[column[:1]*4097 for column in args[1:]]]
        with self.assertRaises(ValueError):self.bulk(oversized)

    def test_invalid_middle_row_refuses_entire_batch_and_next_call_remains_valid(self):
        args=self.rows(self.roots[0]);serial=self.serial(args)
        invalid=[args[0],*[list(c) for c in args[1:]]];invalid[1][2]='not a native state'
        with self.assertRaises(ValueError):self.bulk(invalid)
        self.assert_rows(self.bulk(args),serial)


if __name__=='__main__':unittest.main()
