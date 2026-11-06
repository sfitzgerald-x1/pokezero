"""Inert ladder arithmetic, constructor provenance and actual actor seams."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

from gpu_engine import phase_d_ladder_search as search,phase_d_protocol_world as protocol,v5_public_world
from pokezero import phase_d_ladder_results as results,phase_d_ladder_actor as actor
from pokezero import phase_d_ladder_panel as panel
from pokezero.v5_contract import DecisionAddress
from pokezero.v5_teacher_public_root import PublicRoot
from pokezero.env import BattleStartOverride
from pokezero.policy import PolicyDecision
from pokezero.live_protocol_actor import ActorContract,JaxBackend,OwnedProvider,RawPolicy
from pokezero.live_protocol_bridge import OwningActorFactory
from pokezero.phase_d_ladder_transport import Client
import test_phase_d_ladder_panel as raw_fixture
import test_phase_d_prerequisites as actor_fixture


class LadderMath(unittest.TestCase):
    def test_declared_budgets_and_common_draw_domains(self):
        self.assertEqual([(x['worlds'],x['chance_samples'],x['depth'],x['rollout_leaves'])for x in search.RUNGS.values()],
            [(2,1,1,0),(32,1,1,0),(32,8,1,0),(32,1,2,0),(8,1,1,8)])
        prefixes=[[search.domain_seed('a'*64,w,0,c,domain='sampled-chance')for w in range(worlds)for c in range(8)]for worlds in (2,8,32)]
        self.assertEqual(prefixes[0],prefixes[-1][:16]);self.assertEqual(prefixes[1],prefixes[-1][:64])
        self.assertNotEqual(search.domain_seed('a'*64,0,0,0,domain='sampled-chance'),search.domain_seed('a'*64,0,1,0,domain='sampled-chance'))

    def test_full_world_q_mean_and_unknown_refusal(self):
        legal=(True,True)+tuple(False for _ in range(7))
        reports=[(.5,-.5)+tuple(None for _ in range(7))for _ in range(32)]
        q,action=search.reduce_q(legal,reports,requested_worlds=32);self.assertEqual((q[:2],action),((.5,-.5),0))
        bad=list(reports);bad[7]=(None,-.5)+tuple(None for _ in range(7))
        with self.assertRaisesRegex(ValueError,'unknown'):search.reduce_q(legal,bad,requested_worlds=32)
        with self.assertRaisesRegex(ValueError,'slots'):search.reduce_q(legal,reports[:-1],requested_worlds=32)

    def subject(self,rung,*,nonterminal=False):
        s=object.__new__(search.Search);s.recipe=dict(search.RUNGS[rung]);s.cap=100000;s.lanes=8;s.observed=[]
        def surfaces(nodes):return [dict(requested=(True,True),legal=((True,True)+tuple(False for _ in range(7)),)*2,
            maps=(tuple(range(9)),)*2,predictions=(((math.log(.25),math.log(.75))+(0.,)*7,0.),)*2)for n in nodes]
        s._surfaces=surfaces
        def steps(branches):
            s.observed.append(len(branches));output=[]
            for branch in branches:
                node=branch['node'];stage=getattr(node.state,'stage',0)
                done=not nonterminal or stage>=1
                winner=node.owner if branch['action']==1 or branch['reply']==0 else 1-node.owner
                state=NS(done=np.array([done]),winner=np.array([winner]),stage=stage+1)
                output.append(replace(node,state=state))
            return output
        s._steps=steps
        return s

    def nodes(self):return [search.Node(NS(done=np.array([False]),stage=0),None,0,'b'*64,w,0)for w in range(2)]

    def test_all_replies_all_eight_chance_slots_and_terminal_outcomes(self):
        s=self.subject('R4');best,q=s._price(self.nodes(),1,0)
        self.assertEqual(s.observed,[2*2*2*8]);np.testing.assert_allclose(best,[1.,1.],atol=1e-14);np.testing.assert_allclose([[r[0],r[1]]for r in q],[[-.5,1.]]*2,atol=1e-14)

    def test_depth_two_reprices_complete_child_surfaces(self):
        s=self.subject('R7',nonterminal=True);best,q=s._price(self.nodes(),2,0)
        self.assertEqual(s.observed,[8,32]);np.testing.assert_allclose(best,[1.,1.],atol=1e-14);np.testing.assert_allclose([[r[0],r[1]]for r in q],[[1.,1.]]*2,atol=1e-14)

    def test_rollout_leaf_uses_declared_eight_and_no_value_substitute(self):
        s=self.subject('R5',nonterminal=True);seen=[]
        def values(nodes,ply):seen.append((len(nodes),ply,s.recipe['rollout_leaves']));return [.25]*len(nodes)
        s._rollout_values=values
        best,q=s._price(self.nodes(),1,0)
        self.assertEqual(seen,[(8,1,8)]);np.testing.assert_allclose(best,[.25,.25],atol=1e-14)


class ProtocolRoute(unittest.TestCase):
    def test_chronology_from_actual_public_switch_turns(self):
        self.assertEqual(protocol._chronology(('|switch|p1a: A|A|100/100','|switch|p2a: B|B|100/100','|turn|1','|turn|2')),(1,1))
        with self.assertRaisesRegex(ValueError,'arrival'):protocol._chronology(('|turn|1',))

    def root(self):
        address=DecisionAddress(1,0,0);lines=('|start','|turn|1');digest=hashlib.sha256(''.join(x+'\n'for x in lines).encode()).hexdigest()
        source=NS(metadata=NS(source_hash='bound'))
        context=NS(battle_id='same-live-cell',public_materialization_state=NS(belief_engine=NS(set_source=source),self_request={}))
        payload=dict(selfActiveRequestState={'trapped':True},sides={},pendingBatonPassSides=[])
        root=PublicRoot(address,digest,context,payload,())
        detached=NS(address=address,source_binding_sha256='fixed')
        return root,protocol.ProtocolInputs(detached,lines,NS(source=NS(source_hash='bound')),{},NS(binding_sha256='fixed')),source

    def test_explicit_protocol_route_rejects_driver_evidence_and_native_probe(self):
        root,inputs,source=self.root();start=BattleStartOverride(player_teams={'p1':'own','p2':'sample'})
        with patch.object(protocol,'battle_spec_from_payload',side_effect=AssertionError('must refuse before typed bridge')), \
                patch.object(protocol,'_removed_items',return_value={}):
            result=v5_public_world.materialize(root,start,prepared=inputs,history=None,own_static=None,seed=7)
        self.assertIsNone(result.state);self.assertEqual(result.refusals,('native-probed-hard-trapped-own-request',))
        with self.assertRaisesRegex(ValueError,'retained-driver'):
            v5_public_world.materialize(root,start,prepared=inputs,history=object(),own_static=None,seed=7)

    def test_constructor_budget_prefix_no_redraw_and_refused_draw_slots_retained(self):
        root,inputs,source=self.root();detached=inputs.root
        with patch.object(protocol,'_gen3_randbat_belief_start_override_result',return_value=(None,'named-original-refusal'))as sampler, \
                patch.object(protocol,'_carried_root',return_value=detached):
            a=protocol.sample(root,detached,lines=inputs.lines,catalog=inputs.catalog,source=NS(original=source,binding_sha256='fixed'),metadata={},config=None,battle_id='same-live-cell',seed=9,count=16)
            b=protocol.sample(root,detached,lines=inputs.lines,catalog=inputs.catalog,source=NS(original=source,binding_sha256='fixed'),metadata={},config=None,battle_id='same-live-cell',seed=9,count=32)
        self.assertEqual(sampler.call_count,48);self.assertEqual(a.seeds,b.seeds[:16]);self.assertFalse(b.fully_constructed)
        self.assertEqual(len(b.worlds),32);self.assertTrue(all(w.state is None and w.refusals==('named-original-refusal',)for w in b.worlds))


class ResultAccounting(unittest.TestCase):
    def rows(self):
        d=raw_fixture.declaration();raw=[];searched=[]
        for arm,rows in (('RAW',raw),('R1',searched)):
            for cell in panel.cells(d,arm,limit=500):
                score=float((cell['offset']+(arm=='R1'))%3==0)
                rows.append(dict(cell=cell,status='COMPLETE',score=score,declaration_sha256=panel.sha(d)))
        return d,raw,searched

    def test_seed_cluster_and_full_missing_bounds_no_survivor_screen(self):
        d,a,b=self.rows();r=results.paired(d,a,b,rung='R1',limit=500,expected_declaration_sha256=panel.sha(d))
        self.assertEqual(r['known_pairs'],500);self.assertEqual(r['complete_seed_clusters'],250);self.assertIsNotNone(r['interval90'])
        b[0].update(status='UNKNOWN',score=None)
        r=results.paired(d,a,b,rung='R1',limit=500,expected_declaration_sha256=panel.sha(d))
        self.assertIsNone(r['point_difference']);self.assertEqual(r['unknown_pairs'],1);self.assertEqual(r['complete_seed_clusters'],249)
        self.assertEqual(r['decision'],'NO_PROMOTION')
        self.assertAlmostEqual(r['full_roster_identification_bounds'][1]-r['full_roster_identification_bounds'][0],1/500)

    def test_telemetry_counts_all_refusals_and_fallbacks(self):
        rows=[dict(battle_id='g',owner='p1',decision_index=i,prefix_sha256='c'*64,fell_back=i==1,
            acting_wall_seconds=.2,search=dict(status='REFUSED'if i==1 else'COMPLETE',refusal='clock'if i==1 else None,client_wall_seconds=.1))for i in range(2)]
        out=results.telemetry(rows,declared_decisions=3)
        self.assertEqual(out['counts'],dict(decisions=2,searched=1,refused=1,raw_fallbacks=1));self.assertEqual(out['missing_telemetry'],1)
        self.assertEqual(out['refused_share'],.5)
        with self.assertRaisesRegex(ValueError,'duplicate'):results.telemetry(rows+rows)


class ActualFactorySeam(unittest.TestCase):
    def test_original_raw_fallback_uses_same_rng_and_records_refusal(self):
        class Backend(JaxBackend):
            def __init__(self,c,**kw):OwnedProvider.__init__(self,c)
            ingest=actor_fixture.Backend.ingest
            encode=actor_fixture.Backend.encode
            close=OwnedProvider.close
        observed=[];contract=ActorContract.champion()
        raw=OwningActorFactory('jax-owning',contract,lambda c,**kw:Backend(c),
            lambda c,**kw:RawPolicy(c,lambda observation,*,rng:(observed.append(rng),PolicyDecision(0,'unchanged-raw'))[1],lambda:None))
        client=object.__new__(Client);client.rung='R1';client.request=lambda *a,**kw:dict(status='REFUSED',refusal='clock',client_wall_seconds=.1)
        with TemporaryDirectory()as directory,patch.object(actor,'require_actor_factory',side_effect=lambda c,f:f):
            path=Path(directory)/'decisions.jsonl';factory=actor.factory(NS(),raw_factory=raw,client=client,telemetry_path=path)
            a=factory.actor((None,None,None));a.begin_game(battle_id='literal',owner='p1',seed=9,stream_id=1)
            actor_fixture.feed(a,('|turn|1',actor_fixture.request()));snapshot=a.snapshot_boundary();rng=random.Random(4);state=rng.getstate()
            decision=a.decide(snapshot,rng=rng);self.assertEqual(observed,[rng]);self.assertEqual(rng.getstate(),state);self.assertEqual(decision.action_index,0)
            a.close_game();row=json.loads(path.read_bytes());self.assertTrue(row['fell_back']);self.assertEqual(row['fallback_action_index'],0)



class AdmissionDenominators(unittest.TestCase):
    def roster(self,n=200):
        rows=[]
        for band,turn in enumerate((1,4,10,20)):
            for i in range(n//4):rows.append(dict(cell={'ordinal':6*i},owner='p1',decision_index=50*band+i,prefix_sha256=f'{50*band+i:064x}',turn=turn))
        return rows

    def test_latency_roster_spans_turns_before_constructor_results(self):
        from pokezero import phase_d_ladder_admission as a
        roster=self.roster();selected=a.timing_roster(roster)
        self.assertEqual(len(selected),20);self.assertEqual([sum(r['turn']==t for r in selected)for t in (1,4,10,20)],[5]*4)
        records=[dict(address=a.address(r),requested_worlds=32,constructed_worlds=32,decision_seconds=.25,refusals=[])for r in selected]
        records[-1].update(constructed_worlds=0,decision_seconds=None,refusals=['sleep-clock:foe'])
        records[-2].update(constructed_worlds=0,decision_seconds=None,refusals=['sleep-clock:own'])
        result=a.summarize(selected,records,worlds=32,constructor_only=False)
        self.assertFalse(result['admission']);self.assertIsNone(result['decision_p95_seconds'])
        self.assertEqual(result['refusals_by_root_kind'],{'sleep-clock':2})

    def test_freeze_opens_only_exact64closed_raw_files_and_keeps_unknown_prefix(self):
        from pokezero import phase_d_ladder_admission as a
        with TemporaryDirectory()as directory:
            root=Path(directory);(root/'owning-wire').mkdir()
            for pair in range(64):
                panel.write_new(root/'results'/f'{6*pair}.json',dict(cell=dict(ordinal=6*pair,pair_ordinal=pair,arm='RAW'),
                    declaration_sha256='d'*64,status='UNKNOWN',score='NEVER_A_SELECTION_INPUT'))
            (root/'owning-wire'/'0.jsonl').write_bytes(b'inert')
            (root/'owning-wire'/'384.jsonl').write_bytes(b'forbidden-other-running-cell')
            def capture(path):
                self.assertEqual(Path(path).name,'0.jsonl')
                cell=dict(ordinal=0,pair_ordinal=0,arm='RAW')
                for row in self.roster():yield dict(row,cell=cell,declaration_sha256='d'*64)
            with patch('pokezero.phase_d_owning_capture.read_capture',side_effect=capture):result=a.freeze_prefix(root,declaration_sha256='d'*64)
            self.assertEqual(len(result['selected']),200);self.assertEqual(result['capture_prefix']['games_without_capture'],63)
            self.assertTrue(all(r['cell']['arm']=='RAW'for r in result['selected']))
            (root/'results'/'378.json').unlink()
            with self.assertRaisesRegex(ValueError,'not closed'):a.freeze_prefix(root,declaration_sha256='d'*64)

    def test_freeze_retired_claim_keeps_partial_roots_without_inventing_result(self):
        from pokezero import phase_d_ladder_admission as a
        with TemporaryDirectory()as directory:
            root=Path(directory).resolve();(root/'owning-wire').mkdir()
            d=raw_fixture.declaration();declaration=root/'declaration.json';panel.write_new(declaration,d)
            digest=panel.sha(d);cells=panel.cells(d,'RAW')
            for cell in cells[1:64]:panel.write_new(root/'results'/f"{cell['ordinal']}.json",
                dict(cell=cell,declaration_sha256=digest,status='UNKNOWN'))
            cell=cells[0]
            request=dict(schema='pokezero.champion-search-ladder.raw-request.v1',run_directory=str(root),
                declaration=str(declaration),declaration_sha256=digest,cell=cell)
            for name in ('requests','claims'):panel.write_new(root/name/'0.json',request)
            start=dict(request_sha256=panel.sha(request),cell=cell);panel.write_new(root/'starts'/'0.json',start)
            terminal=root/'terminal.txt';terminal.write_bytes(b'original inert terminal receipt')
            closure=dict(schema='pokezero.champion-search-ladder.attempt-closure.v1',attempt_id='original-attempt',
                declaration_sha256=digest,state='CLOSED',dispatch_stopped=True,all_workers_terminated=True,
                retired_ordinals=[0],terminal_evidence=[dict(path=str(terminal),sha256=hashlib.sha256(terminal.read_bytes()).hexdigest())])
            closed=root/'closed.json';panel.write_new(closed,closure)
            manifest=dict(schema='pokezero.champion-search-ladder.retired-claims.v1',declaration_sha256=digest,
                closure=dict(path=str(closed),sha256=panel.sha(closure)),cells=[dict(cell=cell,
                    request_sha256=panel.sha(request),claim_sha256=panel.sha(request),start_sha256=panel.sha(start))])
            manifest_path=root/'retired.json';panel.write_new(manifest_path,manifest)
            (root/'owning-wire'/'0.jsonl').write_bytes(b'durable partial capture')
            def capture(path):
                self.assertEqual(Path(path).name,'0.jsonl')
                for row in self.roster():yield dict(row,cell=cell,declaration_sha256=digest)
            def freeze():return a.freeze_prefix(root,declaration_sha256=digest,
                retired_claims=manifest_path,retired_claims_sha256=panel.sha(manifest))
            with patch('pokezero.phase_d_owning_capture.read_capture',side_effect=capture):result=freeze()
            self.assertEqual(len(result['selected']),200)
            self.assertEqual(result['capture_prefix']['closed_cells'][0]['disposition'],'RETIRED_CLAIM_NO_RESULT')
            self.assertEqual(result['capture_prefix']['closed_cells'][0]['status'],'UNKNOWN')
            self.assertFalse((root/'results'/'0.json').exists())
            # A lost ongoing claim is not evidence of terminal closure.
            with self.assertRaisesRegex(ValueError,'not closed'):a.freeze_prefix(root,declaration_sha256=digest)
            closure['all_workers_terminated']=False;closed.write_bytes(panel.encoded(closure))
            manifest['closure']['sha256']=panel.sha(closure);manifest_path.write_bytes(panel.encoded(manifest))
            with self.assertRaisesRegex(ValueError,'terminated workers'):freeze()
            closure['all_workers_terminated']=True;closed.write_bytes(panel.encoded(closure))
            manifest['closure']['sha256']=panel.sha(closure);manifest_path.write_bytes(panel.encoded(manifest))
            (root/'claims'/'0.json').write_bytes(panel.encoded(dict(request,cell=cells[1])))
            with self.assertRaisesRegex(ValueError,'immutable ladder input differs'):freeze()

if __name__=='__main__':unittest.main()
